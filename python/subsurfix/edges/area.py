# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import json
import sqlite3
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from subsurfix.edges._tables import init_source, soft_warning
from subsurfix.edges.crs import crs_equals, parse_wkt, reproject_xy
from subsurfix.edges.schemas import SELECTION_SCHEMA, array_table
from subsurfix.errors import ConfigError, FormatError, MissingDependencyError, NotFoundError

_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
_POLYGON_TYPES = {5, 15, 25}
_POLYLINE_TYPES = {3, 13, 23}


@dataclass(frozen=True, kw_only=True)
class AreaSource:
    """One study-area file (shapefile, GeoJSON, or GeoPackage)."""

    path: str
    crs: str | None = None
    layer: str | None = None
    need_tops: bool = True

    def __post_init__(self) -> None:
        init_source(self, ())


def load_area(
    source: AreaSource,
    project_crs: str,
    *,
    allow_network: bool = False,
    allow_ballpark: bool = False,
) -> list[np.ndarray]:
    """Load closed rings in the project frame, metres.

    Each ring is an ``(m, 2)`` float64 array with the first row equal to the
    last. NaN rule: non-finite coordinates follow ``reproject_xy`` (non-finite
    input becomes NaN; a finite input that fails is ``FormatError``).
    """
    path = Path(source.path)
    name = path.name
    if not path.is_file():
        raise NotFoundError(f"{name}: not found")
    suffix = path.suffix.lower()
    messages: list[str] = []
    if suffix == ".shp":
        raw, src = _read_shapefile(path, source, name)
    elif suffix in {".geojson", ".json"}:
        raw, src = _read_geojson(path, source, name, messages)
    elif suffix == ".gpkg":
        raw, src = _read_gpkg(path, source, name, messages)
    else:
        raise FormatError(f"{name}: unsupported area type {suffix or '(none)'}")
    rings = _finish_rings(
        raw,
        src,
        project_crs,
        name,
        allow_network=allow_network,
        allow_ballpark=allow_ballpark,
        messages=messages,
    )
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    return rings


def points_in_area(
    x: np.ndarray,
    y: np.ndarray,
    rings: Sequence[np.ndarray],
) -> np.ndarray:
    """Even-odd point-in-polygon over every ring together.

    NaN rule: non-finite coordinates are outside. Points on a ring follow the
    crossing formula; there is no separate boundary case.
    """
    try:
        x_arr = np.array(x, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError("x and y must be float64 arrays of equal shape") from exc
    if x_arr.shape != y_arr.shape:
        raise ConfigError(
            f"x and y must be float64 arrays of equal shape, got {x_arr.shape} and {y_arr.shape}"
        )
    flat_x = x_arr.ravel()
    flat_y = y_arr.ravel()
    counts = np.zeros(flat_x.shape, dtype=np.int64)
    for ring in rings:
        coords = np.asarray(ring, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] < 2:
            continue
        x1 = coords[:-1, 0]
        y1 = coords[:-1, 1]
        x2 = coords[1:, 0]
        y2 = coords[1:, 1]
        for index in range(x1.shape[0]):
            # crosses = (y1 > py) != (y2 > py); count when px < x_at.
            crosses = (y1[index] > flat_y) != (y2[index] > flat_y)
            if not np.any(crosses):
                continue
            x_at = x1[index] + (flat_y - y1[index]) * (x2[index] - x1[index]) / (
                y2[index] - y1[index]
            )
            counts += crosses & (flat_x < x_at)
    inside = ((counts & 1) == 1) & np.isfinite(flat_x) & np.isfinite(flat_y)
    return inside.reshape(x_arr.shape)


def select_wells(
    headers: pa.Table,
    rings: Sequence[np.ndarray] | None,
    tops: pa.Table | None,
    need_tops: bool = True,
) -> pa.Table:
    """Flag header wells that are inside the area and, if required, have tops.

    With no tops at all (no ``[[edges.tops]]`` or an empty table) the tops rule is
    off: every well in the area is selected and ingest uses one flat datum.

    ``rings is None`` marks every well in the area. ``tops is None`` marks
    every well as having no tops. NaN rule: non-finite ``x_m`` / ``y_m`` are
    outside when an area is set. Output is sorted by ``well_id``.
    """
    well_ids = headers.column("well_id").to_pylist()
    well_keys = headers.column("well_key").to_pylist()
    count = len(well_ids)
    if rings is None:
        in_area = np.ones(count, dtype=bool)
    else:
        in_area = points_in_area(
            headers.column("x_m").to_numpy(),
            headers.column("y_m").to_numpy(),
            rings,
        )
    if tops is None or tops.num_rows == 0:
        has_tops = np.zeros(count, dtype=bool)
    else:
        have = set(tops.column("well_id").to_pylist())
        has_tops = np.array([well_id in have for well_id in well_ids], dtype=bool)
    no_tops_at_all = tops is None or tops.num_rows == 0
    selected = in_area & (has_tops | (not need_tops) | no_tops_at_all)
    order = sorted(range(count), key=lambda index: well_ids[index])
    return array_table(
        SELECTION_SCHEMA,
        {
            "well_id": [well_ids[index] for index in order],
            "well_key": [well_keys[index] for index in order],
            "in_area": [bool(in_area[index]) for index in order],
            "has_tops": [bool(has_tops[index]) for index in order],
            "selected": [bool(selected[index]) for index in order],
        },
    )


def _finish_rings(
    raw_rings: Sequence[object],
    src: str,
    project_crs: str,
    name: str,
    *,
    allow_network: bool,
    allow_ballpark: bool,
    messages: list[str],
) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    short: list[object] = []
    for raw in raw_rings:
        ring = _normalize_ring(raw)
        if ring is None:
            arr = np.asarray(raw, dtype=np.float64) if _array_like(raw) else np.zeros((0, 2))
            short.append(int(arr.shape[0]) if arr.ndim == 2 else 0)
            continue
        rings.append(ring)
    warning = soft_warning(name, "rings with fewer than 3 points", short)
    if warning is not None:
        messages.append(warning)
    if not rings:
        raise FormatError(f"{name}: no polygon or line rings found")
    if crs_equals(src, project_crs, source=name):
        return rings
    moved: list[np.ndarray] = []
    for ring in rings:
        try:
            x_m, y_m = reproject_xy(
                ring[:, 0],
                ring[:, 1],
                src,
                project_crs,
                allow_network=allow_network,
                allow_ballpark=allow_ballpark,
            )
        except (ConfigError, FormatError) as exc:
            raise type(exc)(f"{name}: {exc}") from exc
        moved.append(np.column_stack((x_m, y_m)))
    return moved


def _array_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, list, tuple))


def _normalize_ring(raw: object) -> np.ndarray | None:
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    arr = np.ascontiguousarray(arr[:, :2], dtype=np.float64)
    if arr.shape[0] < 3:
        return None
    if not (arr[0, 0] == arr[-1, 0] and arr[0, 1] == arr[-1, 1]):
        arr = np.vstack([arr, arr[0]])
    return arr


def _read_shapefile(
    path: Path,
    source: AreaSource,
    name: str,
) -> tuple[list[list[tuple[float, float]]], str]:
    try:
        import shapefile
    except ImportError as exc:
        raise MissingDependencyError(
            "shapefile is needed to read shapefiles; install subsurfix[gis]"
        ) from exc
    try:
        reader = shapefile.Reader(str(path))
    except Exception as exc:
        raise FormatError(f"{name}: {exc}") from exc
    raw: list[list[tuple[float, float]]] = []
    try:
        for shape in reader.shapes():
            if shape.shapeType == 0:
                continue
            if shape.shapeType not in _POLYGON_TYPES and shape.shapeType not in _POLYLINE_TYPES:
                raise FormatError(f"{name}: unsupported shape type {shape.shapeType}")
            points = shape.points
            starts = list(shape.parts)
            if not starts:
                if not points:
                    continue
                starts = [0]
            ends = starts[1:] + [len(points)]
            for start, end in zip(starts, ends):
                try:
                    raw.append([(point[0], point[1]) for point in points[start:end]])
                except (TypeError, IndexError) as exc:
                    raise FormatError(f"{name}: shape point is not x, y") from exc
    finally:
        reader.close()
    if source.crs is not None:
        src = source.crs
    else:
        prj = _find_prj(path)
        if prj is None:
            raise ConfigError(f"{name}: no .prj file and no crs declared")
        try:
            wkt = prj.read_text(encoding="utf-8")
        except OSError as exc:
            raise NotFoundError(f"{name}: {exc}") from exc
        src = parse_wkt(wkt.strip(), source=name)
    return raw, src


def _find_prj(path: Path) -> Path | None:
    matches = [
        item
        for item in path.parent.iterdir()
        if item.is_file() and item.stem == path.stem and item.suffix.lower() == ".prj"
    ]
    if not matches:
        return None
    for item in matches:
        if item.suffix == ".prj":
            return item
    return matches[0]


def _read_geojson(
    path: Path,
    source: AreaSource,
    name: str,
    messages: list[str],
) -> tuple[list[np.ndarray], str]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise FormatError(f"{name}: {exc}") from exc
    except OSError as exc:
        raise NotFoundError(f"{name}: {exc}") from exc
    if not isinstance(data, dict):
        raise FormatError(f"{name}: GeoJSON root must be an object")
    src = _geojson_crs(data, source, name)
    skipped: list[object] = []
    raw = _walk_geojson(data, name, skipped)
    warning = soft_warning(name, "unsupported geometries", skipped)
    if warning is not None:
        messages.append(warning)
    return raw, src


def _geojson_crs(data: dict, source: AreaSource, name: str) -> str:
    if source.crs is not None:
        return source.crs
    member = data.get("crs")
    if member is None:
        return "EPSG:4326"
    properties = member.get("properties") if isinstance(member, dict) else None
    crs_name = properties.get("name") if isinstance(properties, dict) else None
    if isinstance(member, dict) and member.get("type") == "name" and isinstance(crs_name, str):
        return crs_name
    raise ConfigError(f"{name}: unrecognized GeoJSON crs member")


def _walk_geojson(obj: object, name: str, skipped: list[object]) -> list[np.ndarray]:
    if not isinstance(obj, dict):
        raise FormatError(f"{name}: GeoJSON value is not an object")
    kind = obj.get("type")
    if kind == "FeatureCollection":
        rings: list[np.ndarray] = []
        for feature in obj.get("features") or []:
            rings.extend(_walk_geojson(feature, name, skipped))
        return rings
    if kind == "Feature":
        geometry = obj.get("geometry")
        if geometry is None:
            skipped.append("null")
            return []
        return _walk_geojson(geometry, name, skipped)
    if kind == "GeometryCollection":
        rings = []
        for geometry in obj.get("geometries") or []:
            rings.extend(_walk_geojson(geometry, name, skipped))
        return rings
    if kind == "Polygon":
        return [_positions(ring, name) for ring in obj.get("coordinates") or []]
    if kind == "MultiPolygon":
        rings = []
        for polygon in obj.get("coordinates") or []:
            if not isinstance(polygon, list):
                raise FormatError(f"{name}: GeoJSON polygon is not a list of rings")
            for ring in polygon:
                rings.append(_positions(ring, name))
        return rings
    if kind == "LineString":
        return [_positions(obj.get("coordinates") or [], name)]
    if kind == "MultiLineString":
        return [_positions(line, name) for line in obj.get("coordinates") or []]
    if kind is None:
        raise FormatError(f"{name}: GeoJSON object has no type")
    skipped.append(str(kind))
    return []


def _positions(coords: object, name: str) -> np.ndarray:
    if not isinstance(coords, list):
        raise FormatError(f"{name}: GeoJSON coordinates are not a list")
    if not coords:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        arr = np.asarray(coords, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FormatError(f"{name}: GeoJSON coordinates are not numeric") from exc
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise FormatError(f"{name}: GeoJSON coordinates must be x, y positions")
    return arr


def _read_gpkg(
    path: Path,
    source: AreaSource,
    name: str,
    messages: list[str],
) -> tuple[list[np.ndarray], str]:
    try:
        import shapely
    except ImportError as exc:
        raise MissingDependencyError(
            "shapely is needed to read GeoPackage geometries; install subsurfix[gis]"
        ) from exc
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise FormatError(f"{name}: {exc}") from exc
    try:
        try:
            layer_rows = connection.execute(
                "SELECT table_name FROM gpkg_contents "
                "WHERE data_type = 'features' ORDER BY table_name"
            ).fetchall()
            layers = [row[0] for row in layer_rows]
            if not layers:
                raise FormatError(f"{name}: no feature layers")
            if source.layer is not None and source.layer not in layers:
                raise NotFoundError(
                    f"{name}: layer {source.layer!r} not found; layers are {layers}"
                )
            if source.layer is None and len(layers) > 1:
                raise ConfigError(f"{name}: multiple layers {layers}; set layer")
            layer = source.layer if source.layer is not None else layers[0]
            geom_row = connection.execute(
                "SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?",
                (layer,),
            ).fetchone()
            if geom_row is None:
                raise FormatError(f"{name}: no geometry column for layer {layer!r}")
            column_name, srs_id = geom_row
            src = _gpkg_crs(connection, srs_id, source, name)
            blobs = connection.execute(
                f"SELECT {_quote_ident(str(column_name))} FROM {_quote_ident(str(layer))}"
            ).fetchall()
        except sqlite3.Error as exc:
            raise FormatError(f"{name}: {exc}") from exc
    finally:
        connection.close()
    skipped: list[object] = []
    raw: list[np.ndarray] = []
    for (blob,) in blobs:
        if blob is None:
            continue
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise FormatError(f"{name}: geometry value is not a blob")
        wkb = _gpkg_wkb(bytes(blob), name)
        if wkb is None:
            continue
        try:
            geometry = shapely.from_wkb(wkb)
        except Exception as exc:
            raise FormatError(f"{name}: {exc}") from exc
        raw.extend(_shapely_rings(geometry, skipped))
    warning = soft_warning(name, "unsupported geometries", skipped)
    if warning is not None:
        messages.append(warning)
    return raw, src


def _gpkg_crs(connection: sqlite3.Connection, srs_id: object, source: AreaSource, name: str) -> str:
    if source.crs is not None:
        return source.crs
    row = connection.execute(
        "SELECT organization, organization_coordsys_id, definition "
        "FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (srs_id,),
    ).fetchone()
    if row is None:
        raise ConfigError(f"{name}: no CRS for srs_id {srs_id}")
    organization, org_id, definition = row
    if (
        isinstance(organization, str)
        and organization.upper() == "EPSG"
        and isinstance(org_id, int)
        and not isinstance(org_id, bool)
        and org_id > 0
    ):
        return f"EPSG:{org_id}"
    if isinstance(definition, str):
        text = definition.strip()
        if text and text.lower() != "undefined":
            return text
    raise ConfigError(f"{name}: no CRS for srs_id {srs_id}")


def _gpkg_wkb(blob: bytes, name: str) -> bytes | None:
    if len(blob) < 8 or blob[:2] != b"GP":
        raise FormatError(f"{name}: geometry blob is not GeoPackage WKB")
    flags = blob[3]
    if flags & 0x20:
        raise FormatError(f"{name}: extended GeoPackage geometry is not supported")
    if flags & 0x10:
        return None
    code = (flags >> 1) & 7
    if code not in _ENVELOPE_BYTES:
        raise FormatError(f"{name}: GeoPackage envelope code {code} is not supported")
    start = 8 + _ENVELOPE_BYTES[code]
    if start > len(blob):
        raise FormatError(f"{name}: GeoPackage geometry blob is truncated")
    return blob[start:]


def _shapely_rings(geometry: object, skipped: list[object]) -> list[np.ndarray]:
    kind = geometry.geom_type  # type: ignore[attr-defined]
    if kind == "Polygon":
        rings = [np.asarray(geometry.exterior.coords, dtype=np.float64)]  # type: ignore[attr-defined]
        rings.extend(
            np.asarray(ring.coords, dtype=np.float64)
            for ring in geometry.interiors  # type: ignore[attr-defined]
        )
        return rings
    if kind == "MultiPolygon":
        output: list[np.ndarray] = []
        for part in geometry.geoms:  # type: ignore[attr-defined]
            output.extend(_shapely_rings(part, skipped))
        return output
    if kind in {"LineString", "LinearRing"}:
        return [np.asarray(geometry.coords, dtype=np.float64)]  # type: ignore[attr-defined]
    if kind == "MultiLineString":
        return [
            np.asarray(line.coords, dtype=np.float64)
            for line in geometry.geoms  # type: ignore[attr-defined]
        ]
    if kind == "GeometryCollection":
        output = []
        for part in geometry.geoms:  # type: ignore[attr-defined]
            output.extend(_shapely_rings(part, skipped))
        return output
    skipped.append(kind)
    return []


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
