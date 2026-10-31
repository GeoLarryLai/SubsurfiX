# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq

from subsurfix.edges.area import AreaSource, load_area, points_in_area, select_wells
from subsurfix.edges.crs import make_transformer, project_crs, reproject_xy
from subsurfix.edges.headers import HeaderSource, load_headers
from subsurfix.edges.ids import assign_well_ids, attach_well_ids, well_id_table
from subsurfix.edges.schemas import (
    HEADERS_SCHEMA,
    PTS_SCHEMA,
    SELECTION_SCHEMA,
    SURVEYS_SCHEMA,
    TOPS_SCHEMA,
    WELL_IDS_SCHEMA,
    write_parquet,
)
from subsurfix.edges.pts import PtsSource, load_pts
from subsurfix.edges.surveys import SurveySource, load_surveys
from subsurfix.edges.tops import DEPTH_REF_CODES, TopsSource, load_tops
from subsurfix.errors import ConfigError, FormatError, NotFoundError

__all__ = [
    "HEADERS_SCHEMA",
    "TOPS_SCHEMA",
    "SURVEYS_SCHEMA",
    "SELECTION_SCHEMA",
    "WELL_IDS_SCHEMA",
    "PTS_SCHEMA",
    "PtsSource",
    "load_pts",
    "DEPTH_REF_CODES",
    "HeaderSource",
    "TopsSource",
    "SurveySource",
    "AreaSource",
    "project_crs",
    "make_transformer",
    "reproject_xy",
    "load_headers",
    "load_tops",
    "load_surveys",
    "load_area",
    "points_in_area",
    "select_wells",
    "assign_well_ids",
    "attach_well_ids",
    "write_parquet",
    "prepare_edges",
]

_ID_LIMIT = 2**31 - 1
_EDGES_KEYS = frozenset({"allow_network", "allow_ballpark", "headers", "tops", "surveys", "pts", "area"})
_HEADER_KEYS = frozenset(
    {
        "path",
        "kb_unit",
        "sheet",
        "delimiter",
        "crs",
        "xy_unit",
        "key_col",
        "x_col",
        "y_col",
        "label_col",
        "kb_col",
        "kb_match",
        "datum_col",
        "lat_col",
        "lon_col",
        "location_col",
        "td_col",
    }
)
_TOPS_KEYS = frozenset(
    {
        "path",
        "depth_ref",
        "depth_unit",
        "sheet",
        "delimiter",
        "layout",
        "repeat_pick",
        "key_col",
        "top_col",
        "depth_col",
        "top_columns",
        "exclude_columns",
    }
)
_SURVEY_KEYS = frozenset(
    {
        "path",
        "md_unit",
        "sheet",
        "delimiter",
        "key_col",
        "md_col",
        "inc_col",
        "azi_col",
    }
)
_PTS_KEYS = frozenset(
    {"path", "well", "key_col", "sheets", "delimiter", "temp_unit", "pressure_unit", "depth_unit"}
)
_AREA_KEYS = frozenset({"path", "crs", "layer", "need_tops"})


class _Project(NamedTuple):
    crs: str
    allow_network: bool
    allow_ballpark: bool
    headers: list[HeaderSource]
    tops: list[TopsSource]
    surveys: list[SurveySource]
    pts: list[PtsSource]
    area: AreaSource | None


def prepare_edges(project_toml: str | os.PathLike[str]) -> dict[str, Path]:
    """Read a project file and write the five edge Parquet tables.

    Paths in the file are relative to the project file's folder. NaN rule:
    follows the header, top, survey, and area loaders. Returns the written
    paths in the order well ids, headers, tops, surveys, selection, and
    ``pts`` when ``[[edges.pts]]`` is set.
    """
    path = Path(project_toml)
    if not path.is_file():
        raise NotFoundError(f"{path.name}: not found")
    project_dir = path.parent
    parsed = _parse_project(_read_toml(path), path.name, project_dir)
    try:
        project_crs(parsed.crs)
    except ConfigError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
    existing = _read_existing(project_dir / "edges" / "well_ids.parquet")
    headers = load_headers(
        parsed.headers,
        parsed.crs,
        allow_network=parsed.allow_network,
        allow_ballpark=parsed.allow_ballpark,
    )
    ids = assign_well_ids(headers.column("well_key").to_pylist(), existing)
    headers = attach_well_ids(headers, ids)
    tops = attach_well_ids(load_tops(parsed.tops), ids)
    surveys = attach_well_ids(load_surveys(parsed.surveys), ids)
    pts = attach_well_ids(load_pts(parsed.pts), ids)
    if parsed.area is None:
        rings = None
        need_tops = True
    else:
        rings = load_area(
            parsed.area,
            parsed.crs,
            allow_network=parsed.allow_network,
            allow_ballpark=parsed.allow_ballpark,
        )
        need_tops = parsed.area.need_tops
    selection = select_wells(headers, rings, tops, need_tops)
    out_dir = project_dir / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "well_ids": out_dir / "well_ids.parquet",
        "headers": out_dir / "headers.parquet",
        "tops": out_dir / "tops.parquet",
        "surveys": out_dir / "surveys.parquet",
        "selection": out_dir / "selection.parquet",
    }
    pts_path = out_dir / "pts.parquet"
    write_parquet(well_id_table(ids), written["well_ids"], "subsurfix.edges_well_ids/1")
    write_parquet(headers, written["headers"], "subsurfix.edges_headers/1")
    write_parquet(tops, written["tops"], "subsurfix.edges_tops/1")
    write_parquet(surveys, written["surveys"], "subsurfix.edges_surveys/1")
    write_parquet(selection, written["selection"], "subsurfix.edges_selection/1")
    if parsed.pts:
        write_parquet(pts, pts_path, "subsurfix.edges_pts/1")
        written["pts"] = pts_path
    elif pts_path.is_file():
        pts_path.unlink()
    return written


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
    except OSError as exc:
        raise NotFoundError(f"{path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: project file must be a table")
    return data


def _parse_project(data: Mapping[str, object], toml_name: str, project_dir: Path) -> _Project:
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("crs"), str):
        raise ConfigError(f"{toml_name}: [project] crs must be a string")
    edges = data.get("edges", {})
    if not isinstance(edges, dict):
        raise ConfigError(f"{toml_name}: [edges] must be a table")
    _reject_unknown(edges, _EDGES_KEYS, f"{toml_name}: [edges]")
    allow_network = _optional_bool(edges, "allow_network", f"{toml_name}: [edges]", False)
    allow_ballpark = _optional_bool(edges, "allow_ballpark", f"{toml_name}: [edges]", False)
    headers = [
        _header_source(item, index, toml_name, project_dir)
        for index, item in enumerate(_table_array(edges, "headers", toml_name), start=1)
    ]
    tops = [
        _tops_source(item, index, toml_name, project_dir)
        for index, item in enumerate(_table_array(edges, "tops", toml_name), start=1)
    ]
    surveys = [
        _survey_source(item, index, toml_name, project_dir)
        for index, item in enumerate(_table_array(edges, "surveys", toml_name), start=1)
    ]
    pts = [
        _pts_source(item, index, toml_name, project_dir)
        for index, item in enumerate(_table_array(edges, "pts", toml_name), start=1)
    ]
    area = None
    if "area" in edges:
        area_table = edges["area"]
        if not isinstance(area_table, dict):
            raise ConfigError(f"{toml_name}: [edges.area] must be a table")
        area = _area_source(area_table, toml_name, project_dir)
    return _Project(
        crs=project["crs"],
        allow_network=allow_network,
        allow_ballpark=allow_ballpark,
        headers=headers,
        tops=tops,
        surveys=surveys,
        pts=pts,
        area=area,
    )


def _header_source(item: dict, index: int, toml_name: str, project_dir: Path) -> HeaderSource:
    where = f"{toml_name}: [[edges.headers]] #{index}"
    _reject_unknown(item, _HEADER_KEYS, where)
    kwargs: dict[str, object] = {
        "path": _resolve(project_dir, _required_str(item, "path", where)),
        "kb_unit": _required_str(item, "kb_unit", where),
    }
    _fill_strings(kwargs, item, where, ("sheet", "delimiter", "crs", "xy_unit", "kb_match"))
    _fill_tuples(
        kwargs,
        item,
        where,
        (
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
        ),
    )
    return HeaderSource(**kwargs)  # type: ignore[arg-type]


def _tops_source(item: dict, index: int, toml_name: str, project_dir: Path) -> TopsSource:
    where = f"{toml_name}: [[edges.tops]] #{index}"
    _reject_unknown(item, _TOPS_KEYS, where)
    kwargs: dict[str, object] = {
        "path": _resolve(project_dir, _required_str(item, "path", where)),
        "depth_ref": _required_str(item, "depth_ref", where),
        "depth_unit": _required_str(item, "depth_unit", where),
    }
    _fill_strings(kwargs, item, where, ("sheet", "delimiter", "layout", "repeat_pick"))
    _fill_tuples(
        kwargs,
        item,
        where,
        ("key_col", "top_col", "depth_col", "top_columns", "exclude_columns"),
    )
    return TopsSource(**kwargs)  # type: ignore[arg-type]


def _survey_source(item: dict, index: int, toml_name: str, project_dir: Path) -> SurveySource:
    where = f"{toml_name}: [[edges.surveys]] #{index}"
    _reject_unknown(item, _SURVEY_KEYS, where)
    kwargs: dict[str, object] = {
        "path": _resolve(project_dir, _required_str(item, "path", where)),
        "md_unit": _required_str(item, "md_unit", where),
    }
    _fill_strings(kwargs, item, where, ("sheet", "delimiter"))
    _fill_tuples(kwargs, item, where, ("key_col", "md_col", "inc_col", "azi_col"))
    return SurveySource(**kwargs)  # type: ignore[arg-type]


def _pts_source(item: dict, index: int, toml_name: str, project_dir: Path) -> PtsSource:
    where = f"{toml_name}: [[edges.pts]] #{index}"
    _reject_unknown(item, _PTS_KEYS, where)
    kwargs: dict[str, object] = {"path": _resolve(project_dir, _required_str(item, "path", where))}
    _fill_strings(
        kwargs,
        item,
        where,
        ("well", "key_col", "delimiter", "temp_unit", "pressure_unit", "depth_unit"),
    )
    _fill_tuples(kwargs, item, where, ("sheets",))
    return PtsSource(**kwargs)  # type: ignore[arg-type]


def _area_source(item: dict, toml_name: str, project_dir: Path) -> AreaSource:
    where = f"{toml_name}: [edges.area]"
    _reject_unknown(item, _AREA_KEYS, where)
    kwargs: dict[str, object] = {
        "path": _resolve(project_dir, _required_str(item, "path", where)),
    }
    _fill_strings(kwargs, item, where, ("crs", "layer"))
    if "need_tops" in item:
        kwargs["need_tops"] = _optional_bool(item, "need_tops", where, True)
    return AreaSource(**kwargs)  # type: ignore[arg-type]


def _table_array(edges: Mapping[str, object], key: str, toml_name: str) -> list[dict]:
    if key not in edges:
        return []
    value = edges[key]
    if not isinstance(value, list):
        raise ConfigError(f"{toml_name}: edges.{key} must be an array of tables")
    tables: list[dict] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"{toml_name}: [[edges.{key}]] #{index} must be a table")
        tables.append(item)
    return tables


def _reject_unknown(table: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key {unknown[0]!r}")


def _required_str(table: Mapping[str, object], key: str, where: str) -> str:
    if key not in table:
        raise ConfigError(f"{where}: missing {key}")
    value = table[key]
    if not isinstance(value, str):
        raise ConfigError(f"{where}: {key} must be a string")
    return value


def _optional_bool(table: Mapping[str, object], key: str, where: str, default: bool) -> bool:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: {key} must be a bool")
    return value


def _fill_strings(
    kwargs: dict[str, object],
    table: Mapping[str, object],
    where: str,
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, str):
            raise ConfigError(f"{where}: {key} must be a string")
        kwargs[key] = value


def _fill_tuples(
    kwargs: dict[str, object],
    table: Mapping[str, object],
    where: str,
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ConfigError(f"{where}: {key} must be an array of strings")
        kwargs[key] = tuple(value)


def _resolve(project_dir: Path, raw: str) -> str:
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    return str(project_dir / candidate)


def _read_existing(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise FormatError(f"{path.name}: {exc}") from exc
    if list(table.column_names) != list(WELL_IDS_SCHEMA.names):
        raise FormatError(f"{path.name}: schema does not match subsurfix.edges_well_ids/1")
    for field in WELL_IDS_SCHEMA:
        got = table.schema.field(field.name)
        if got.type != field.type:
            raise FormatError(
                f"{path.name}: column {field.name} has type {got.type}, expected {field.type}"
            )
    keys = table.column("well_key").to_pylist()
    values = table.column("well_id").to_pylist()
    if any(not isinstance(key, str) for key in keys):
        raise FormatError(f"{path.name}: well_key values must be text")
    if len(set(keys)) != len(keys):
        raise ConfigError(f"{path.name}: duplicate well_key")
    existing: dict[str, int] = {}
    seen: set[int] = set()
    for key, value in zip(keys, values, strict=True):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _ID_LIMIT:
            raise ConfigError(f"{path.name}: well id for {key!r} must be in 0..=2**31-1")
        if value in seen:
            raise ConfigError(f"{path.name}: duplicate well id {value}")
        seen.add(value)
        existing[key] = value
    return existing
