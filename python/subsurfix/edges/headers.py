# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from subsurfix.edges._tables import (
    _parse_number,
    cell_to_key,
    cell_to_text,
    find_column,
    init_source,
    read_rows,
    soft_warning,
    source_name,
)
from subsurfix.edges.crs import reproject_xy
from subsurfix.edges.schemas import HEADERS_SCHEMA, array_table, without_well_id
from subsurfix.errors import ConfigError, FormatError

_BODY = without_well_id(HEADERS_SCHEMA)
_TUPLE_FIELDS = (
    "key_col",
    "x_col",
    "y_col",
    "label_col",
    "kb_col",
    "datum_col",
    "lat_col",
    "lon_col",
    "location_col",
    "td_col",
)


@dataclass(frozen=True, kw_only=True)
class HeaderSource:
    """One well-header table. Exactly one of ``crs`` and ``xy_unit`` is set."""

    path: str
    kb_unit: str
    sheet: str | None = None
    delimiter: str = ","
    crs: str | None = None
    xy_unit: str | None = None
    key_col: tuple[str, ...] = ("UWI",)
    x_col: tuple[str, ...] = ("Surface X", "X")
    y_col: tuple[str, ...] = ("Surface Y", "Y")
    label_col: tuple[str, ...] = ("Name", "well_name", "Well Name")
    kb_col: tuple[str, ...] = ()
    kb_match: str | None = "datum value"
    datum_col: tuple[str, ...] = ("Well datum name", "Datum")
    # Optional match hints, used only to settle LAS header/file-name conflicts.
    lat_col: tuple[str, ...] = ("Latitude", "Lat", "LAT")
    lon_col: tuple[str, ...] = ("Longitude", "Lon", "Long", "LON")
    location_col: tuple[str, ...] = ("Location", "Legal Location", "STR")
    td_col: tuple[str, ...] = ("Depth_stop", "TD", "Total Depth", "Driller TD")

    def __post_init__(self) -> None:
        init_source(self, _TUPLE_FIELDS)


class _Row:
    __slots__ = ("key", "label", "x_m", "y_m", "kb", "datum", "source", "hints")

    def __init__(
        self,
        key: str,
        label: str,
        x_m: float,
        y_m: float,
        kb: float,
        datum: str,
        source: str,
    ) -> None:
        self.key = key
        self.label = label
        self.x_m = x_m
        self.y_m = y_m
        self.kb = kb
        self.datum = datum
        self.source = source
        self.hints: dict[str, object] = {}


class _Acc:
    __slots__ = (
        "key",
        "label",
        "x_m",
        "y_m",
        "xy_source",
        "kb",
        "datum",
        "kb_source",
        "have_xy",
        "have_kb",
        "hints",
    )

    def __init__(self, key: str, xy_source: str) -> None:
        self.key = key
        self.label = ""
        self.x_m = math.nan
        self.y_m = math.nan
        self.xy_source = xy_source
        self.kb = math.nan
        self.datum = ""
        self.kb_source = ""
        self.have_xy = False
        self.have_kb = False
        self.hints: dict[str, object] = {}


def load_headers(
    sources: Sequence[HeaderSource],
    project_crs: str,
    *,
    allow_network: bool = False,
    allow_ballpark: bool = False,
) -> pa.Table:
    """Load header tables into the project frame.

    ``x_m`` / ``y_m`` are metres. ``kb_elev_ft`` is feet. NaN rule: a missing
    coordinate or KB stays NaN and does not hide a later finite value.
    Empty ``sources`` is ``ConfigError``. The table has no ``well_id`` yet.
    """
    if not sources:
        raise ConfigError("at least one [[edges.headers]] source is required")
    rows: list[_Row] = []
    for source in sources:
        part, messages = _read_source(
            source,
            project_crs,
            allow_network=allow_network,
            allow_ballpark=allow_ballpark,
        )
        rows.extend(part)
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)
    merged = _merge(rows)
    return array_table(
        _BODY,
        {
            "well_key": [row.key for row in merged],
            "well_label": [row.label for row in merged],
            "x_m": [row.x_m for row in merged],
            "y_m": [row.y_m for row in merged],
            "kb_elev_ft": [row.kb for row in merged],
            "datum_type": [row.datum for row in merged],
            "xy_source": [row.xy_source for row in merged],
            "kb_source": [row.kb_source for row in merged],
            "lat_deg": [row.hints.get("lat_deg") for row in merged],
            "lon_deg": [row.hints.get("lon_deg") for row in merged],
            "location": [row.hints.get("location") for row in merged],
            "td_ft": [row.hints.get("td_ft") for row in merged],
        },
    )


def _validate(source: HeaderSource, name: str) -> None:
    if source.kb_unit not in {"ft", "m"}:
        raise ConfigError(f"{name}: kb_unit must be 'ft' or 'm'")
    crs_set = source.crs is not None
    unit_set = source.xy_unit is not None
    if crs_set == unit_set:
        raise ConfigError(f"{name}: set exactly one of crs and xy_unit")
    if unit_set and source.xy_unit not in {"m", "ft", "us-ft"}:
        raise ConfigError(f"{name}: xy_unit must be 'm', 'ft', or 'us-ft'")


def _read_source(
    source: HeaderSource,
    project_crs: str,
    *,
    allow_network: bool,
    allow_ballpark: bool,
) -> tuple[list[_Row], list[str]]:
    name = source_name(source.path)
    _validate(source, name)
    header, table_rows = read_rows(source.path, source.sheet, source.delimiter)
    key_i = find_column(header, source.key_col, required=True, what="well key", source=name)
    x_i = find_column(header, source.x_col, required=True, what="x", source=name)
    y_i = find_column(header, source.y_col, required=True, what="y", source=name)
    label_i = find_column(header, source.label_col, required=False, what="label", source=name)
    kb_i = find_column(
        header,
        source.kb_col,
        source.kb_match,
        required=False,
        what="kb",
        source=name,
    )
    datum_i = find_column(header, source.datum_col, required=False, what="datum", source=name)
    hint_i = {
        k: find_column(header, cols, required=False, what=k, source=name)
        for k, cols in (
            ("lat_deg", source.lat_col),
            ("lon_deg", source.lon_col),
            ("location", source.location_col),
            ("td_ft", source.td_col),
        )
    }
    hint_rows: list[dict[str, object]] = []
    assert key_i is not None and x_i is not None and y_i is not None
    messages: list[str] = []
    if kb_i is None:
        messages.append(f"{name}: no kb column; kb elevation is NaN")
    dropped: list[object] = []
    unparsable: list[object] = []
    keys: list[str] = []
    labels: list[str] = []
    datums: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []
    kb_values: list[float] = []
    for row_number, row in enumerate(table_rows, start=1):
        try:
            key = cell_to_key(row[key_i])
        except FormatError as exc:
            raise FormatError(f"{name}: row {row_number}: {exc}") from exc
        if key is None:
            dropped.append(row_number)
            continue
        x_value, x_bad = _parse_number(row[x_i])
        y_value, y_bad = _parse_number(row[y_i])
        if kb_i is None:
            kb_value, kb_bad = math.nan, False
        else:
            kb_value, kb_bad = _parse_number(row[kb_i])
        if x_bad:
            unparsable.append(f"row {row_number} x={row[x_i]!r}")
        if y_bad:
            unparsable.append(f"row {row_number} y={row[y_i]!r}")
        if kb_bad:
            unparsable.append(f"row {row_number} kb={row[kb_i]!r}")
        keys.append(key)
        labels.append("" if label_i is None else cell_to_text(row[label_i]))
        datums.append("" if datum_i is None else cell_to_text(row[datum_i]))
        hint: dict[str, object] = {}
        for k, i in hint_i.items():
            if i is None:
                continue
            if k == "location":
                hint[k] = cell_to_text(row[i])
            else:
                value, bad = _parse_number(row[i])
                if not bad and math.isfinite(value):
                    hint[k] = value / 0.3048 if (k == "td_ft" and source.kb_unit == "m") else value
        hint_rows.append(hint)
        x_values.append(x_value)
        y_values.append(y_value)
        kb_values.append(kb_value)
    for message in (
        soft_warning(name, "rows with no well key", dropped),
        soft_warning(name, "unparsable numbers", unparsable),
    ):
        if message is not None:
            messages.append(message)
    if not keys:
        return [], messages
    x_arr = np.asarray(x_values, dtype=np.float64)
    y_arr = np.asarray(y_values, dtype=np.float64)
    kb_arr = np.asarray(kb_values, dtype=np.float64)
    if source.crs is not None:
        try:
            x_m, y_m = reproject_xy(
                x_arr,
                y_arr,
                source.crs,
                project_crs,
                allow_network=allow_network,
                allow_ballpark=allow_ballpark,
            )
        except (ConfigError, FormatError) as exc:
            raise type(exc)(f"{name}: {exc}") from exc
    elif source.xy_unit == "ft":
        x_m = x_arr * 0.3048
        y_m = y_arr * 0.3048
    elif source.xy_unit == "us-ft":
        x_m = x_arr * 1200.0 / 3937.0
        y_m = y_arr * 1200.0 / 3937.0
    else:
        x_m = x_arr
        y_m = y_arr
    if source.kb_unit == "m":
        kb_ft = kb_arr / 0.3048
    else:
        kb_ft = kb_arr
    if hint_i["lat_deg"] is None or hint_i["lon_deg"] is None:
        lon, lat = _lonlat(np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64), project_crs)
        for i, hint in enumerate(hint_rows):
            if math.isfinite(lat[i]) and math.isfinite(lon[i]):
                hint.setdefault("lat_deg", float(lat[i]))
                hint.setdefault("lon_deg", float(lon[i]))
    rows = [
        _Row(keys[i], labels[i], float(x_m[i]), float(y_m[i]), float(kb_ft[i]), datums[i], name)
        for i in range(len(keys))
    ]
    for row, hint in zip(rows, hint_rows):
        row.hints = hint
    return rows, messages


def _lonlat(x_m: np.ndarray, y_m: np.ndarray, project_crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort longitude/latitude of project-frame points (NaN when pyproj
    or the transform is unavailable). Used only as a LAS match hint."""
    nan = np.full(x_m.shape, np.nan)
    try:
        lon, lat = reproject_xy(x_m, y_m, project_crs, "EPSG:4326")
    except Exception:  # noqa: BLE001 - a hint, never fatal
        return nan, nan
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _merge(rows: Sequence[_Row]) -> list[_Acc]:
    by_key: dict[str, _Acc] = {}
    for row in rows:
        acc = by_key.get(row.key)
        if acc is None:
            acc = _Acc(row.key, row.source)
            by_key[row.key] = acc
        if acc.label == "" and row.label != "":
            acc.label = row.label
        for k, v in row.hints.items():
            if k not in acc.hints and v not in ("", None) and not (isinstance(v, float) and math.isnan(v)):
                acc.hints[k] = v
        if not acc.have_xy and math.isfinite(row.x_m) and math.isfinite(row.y_m):
            acc.x_m = row.x_m
            acc.y_m = row.y_m
            acc.xy_source = row.source
            acc.have_xy = True
        if not acc.have_kb and math.isfinite(row.kb):
            acc.kb = row.kb
            acc.datum = row.datum
            acc.kb_source = row.source
            acc.have_kb = True
    return [by_key[key] for key in sorted(by_key)]
