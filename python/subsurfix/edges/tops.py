# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

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
from subsurfix.edges.schemas import TOPS_SCHEMA, array_table, without_well_id
from subsurfix.errors import ConfigError, FormatError

_BODY = without_well_id(TOPS_SCHEMA)
_TUPLE_FIELDS = ("key_col", "top_col", "depth_col", "top_columns", "exclude_columns")

DEPTH_REF_CODES: dict[str, int] = {"md": 0, "tvd": 1, "tvdss": 2, "elev": 3}

_DEFAULT_DEPTH_COL: dict[str, tuple[str, ...]] = {
    "md": ("MD", "MD_ft", "MD_m", "MD (ft)", "MD (m)", "MDT", "Depth"),
    "tvd": ("TVD", "TVD_ft", "TVD_m", "TVD (ft)", "TVD (m)", "Depth"),
    "tvdss": ("TVDSS", "TVDSS_ft", "TVDSS_m", "TVDSS (ft)", "TVDSS (m)", "Depth"),
    "elev": ("Elevation", "Elev", "Elevation_ft", "Elevation_m"),
}


@dataclass(frozen=True, kw_only=True)
class TopsSource:
    """One tops table. ``depth_ref`` and ``depth_unit`` have no defaults."""

    path: str
    depth_ref: str
    depth_unit: str
    sheet: str | None = None
    delimiter: str = ","
    layout: str = "long"
    repeat_pick: str = "shallowest"
    key_col: tuple[str, ...] = ("UWI",)
    top_col: tuple[str, ...] = ("Top", "Surface", "top_name", "topname")
    depth_col: tuple[str, ...] | None = None
    top_columns: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        init_source(self, _TUPLE_FIELDS)


class _Pick:
    __slots__ = ("well_key", "top_name", "depth_ft", "depth_ref", "source", "order")

    def __init__(
        self,
        well_key: str,
        top_name: str,
        depth_ft: float,
        depth_ref: int,
        source: str,
        order: int,
    ) -> None:
        self.well_key = well_key
        self.top_name = top_name
        self.depth_ft = depth_ft
        self.depth_ref = depth_ref
        self.source = source
        self.order = order


def load_tops(sources: Sequence[TopsSource]) -> pa.Table:
    """Load tops as depths in feet.

    NaN rule: a blank or unparsable depth is dropped, not stored. No sources
    returns an empty table. The table has no ``well_id`` yet.
    """
    winners: dict[tuple[str, str], _Pick] = {}
    for source in sources:
        picks, messages = _read_source(source)
        ignored: list[object] = []
        for pick in picks:
            pair = (pick.well_key, pick.top_name)
            if pair in winners:
                ignored.append(pair)
            else:
                winners[pair] = pick
        ignored_msg = soft_warning(
            source_name(source.path),
            "picks already set by an earlier source",
            ignored,
        )
        if ignored_msg is not None:
            messages.append(ignored_msg)
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)
    ordered = sorted(winners.values(), key=lambda pick: (pick.well_key, pick.top_name, pick.order))
    return array_table(
        _BODY,
        {
            "well_key": [pick.well_key for pick in ordered],
            "top_name": [pick.top_name for pick in ordered],
            "depth_ft": [pick.depth_ft for pick in ordered],
            "depth_ref": [pick.depth_ref for pick in ordered],
            "source": [pick.source for pick in ordered],
        },
    )


def _validate(source: TopsSource, name: str) -> None:
    if source.depth_ref not in DEPTH_REF_CODES:
        raise ConfigError(f"{name}: depth_ref must be 'md', 'tvd', 'tvdss', or 'elev'")
    if source.depth_unit not in {"ft", "m"}:
        raise ConfigError(f"{name}: depth_unit must be 'ft' or 'm'")
    if source.layout not in {"long", "wide"}:
        raise ConfigError(f"{name}: layout must be 'long' or 'wide'")
    if source.repeat_pick not in {"shallowest", "first", "error"}:
        raise ConfigError(f"{name}: repeat_pick must be 'shallowest', 'first', or 'error'")


def _depth_suffix_ok(column: str, depth_unit: str, name: str) -> None:
    folded = column.casefold()
    if depth_unit == "m" and (folded.endswith("_ft") or folded.endswith("(ft)")):
        raise ConfigError(
            f"{name}: depth column {column!r} looks like feet but depth_unit is 'm'"
        )
    if depth_unit == "ft" and (folded.endswith("_m") or folded.endswith("(m)")):
        raise ConfigError(
            f"{name}: depth column {column!r} looks like metres but depth_unit is 'ft'"
        )


def _to_feet(depth: float, depth_unit: str) -> float:
    if depth_unit == "m":
        return depth / 0.3048
    return depth


def _read_source(source: TopsSource) -> tuple[list[_Pick], list[str]]:
    name = source_name(source.path)
    _validate(source, name)
    header, rows = read_rows(source.path, source.sheet, source.delimiter)
    if source.layout == "long":
        picks, messages = _read_long(source, header, rows, name)
    else:
        picks, messages = _read_wide(source, header, rows, name)
    collapsed, repeat_msg = _collapse(picks, source, name)
    if repeat_msg is not None:
        messages.append(repeat_msg)
    return collapsed, messages


def _read_long(
    source: TopsSource,
    header: list[str],
    rows: list[list[object]],
    name: str,
) -> tuple[list[_Pick], list[str]]:
    depth_candidates = source.depth_col if source.depth_col is not None else _DEFAULT_DEPTH_COL[source.depth_ref]
    key_i = find_column(header, source.key_col, required=True, what="well key", source=name)
    top_i = find_column(header, source.top_col, required=True, what="top", source=name)
    depth_i = find_column(header, depth_candidates, required=True, what="depth", source=name)
    assert key_i is not None and top_i is not None and depth_i is not None
    _depth_suffix_ok(header[depth_i], source.depth_unit, name)
    code = DEPTH_REF_CODES[source.depth_ref]
    dropped_keys: list[object] = []
    dropped_tops: list[object] = []
    dropped_depth: list[object] = []
    unparsable: list[object] = []
    picks: list[_Pick] = []
    order = 0
    for row_number, row in enumerate(rows, start=1):
        try:
            key = cell_to_key(row[key_i])
        except FormatError as exc:
            raise FormatError(f"{name}: row {row_number}: {exc}") from exc
        top = cell_to_text(row[top_i])
        depth, bad = _parse_number(row[depth_i])
        if bad:
            unparsable.append(f"row {row_number} depth={row[depth_i]!r}")
        if key is None:
            dropped_keys.append(row_number)
            continue
        if top == "":
            dropped_tops.append(row_number)
            continue
        if depth != depth:
            dropped_depth.append(row_number)
            continue
        picks.append(_Pick(key, top, _to_feet(depth, source.depth_unit), code, name, order))
        order += 1
    messages = [
        message
        for message in (
            soft_warning(name, "rows with no well key", dropped_keys),
            soft_warning(name, "rows with an empty top name", dropped_tops),
            soft_warning(name, "rows with NaN depth", dropped_depth),
            soft_warning(name, "unparsable numbers", unparsable),
        )
        if message is not None
    ]
    return picks, messages


def _read_wide(
    source: TopsSource,
    header: list[str],
    rows: list[list[object]],
    name: str,
) -> tuple[list[_Pick], list[str]]:
    key_i = find_column(header, source.key_col, required=True, what="well key", source=name)
    assert key_i is not None
    if source.top_columns is not None:
        present = set(header)
        for column in source.top_columns:
            if column not in present:
                raise FormatError(f"{name}: top column {column!r} not found; columns are {header}")
        wanted = set(source.top_columns)
        positions = [
            index
            for index, column in enumerate(header)
            if column in wanted and column != ""
        ]
    else:
        excluded = set(source.exclude_columns)
        positions = [
            index
            for index, column in enumerate(header)
            if index != key_i and column != "" and column not in excluded
        ]
    code = DEPTH_REF_CODES[source.depth_ref]
    dropped_keys: list[object] = []
    unparsable: list[object] = []
    picks: list[_Pick] = []
    order = 0
    for row_number, row in enumerate(rows, start=1):
        try:
            key = cell_to_key(row[key_i])
        except FormatError as exc:
            raise FormatError(f"{name}: row {row_number}: {exc}") from exc
        if key is None:
            dropped_keys.append(row_number)
            continue
        for index in positions:
            if index >= len(row):
                continue
            depth, bad = _parse_number(row[index])
            if bad:
                unparsable.append(f"row {row_number} {header[index]}={row[index]!r}")
            if depth != depth:
                continue
            picks.append(
                _Pick(key, header[index], _to_feet(depth, source.depth_unit), code, name, order)
            )
            order += 1
    messages = [
        message
        for message in (
            soft_warning(name, "rows with no well key", dropped_keys),
            soft_warning(name, "unparsable numbers", unparsable),
        )
        if message is not None
    ]
    return picks, messages


def _collapse(
    picks: Sequence[_Pick],
    source: TopsSource,
    name: str,
) -> tuple[list[_Pick], str | None]:
    groups: dict[tuple[str, str], list[_Pick]] = {}
    for pick in picks:
        groups.setdefault((pick.well_key, pick.top_name), []).append(pick)
    repeated = [pair for pair, items in groups.items() if len(items) > 1]
    if source.repeat_pick == "error" and repeated:
        raise FormatError(f"{name}: repeated tops {repeated[:10]}")
    elev = source.depth_ref == "elev"
    kept: list[_Pick] = []
    dropped: list[object] = []
    for pair, items in groups.items():
        if len(items) == 1 or source.repeat_pick == "first":
            chosen = items[0]
        else:
            chosen = items[0]
            for item in items[1:]:
                better = item.depth_ft > chosen.depth_ft if elev else item.depth_ft < chosen.depth_ft
                if better:
                    chosen = item
        kept.append(chosen)
        for item in items:
            if item is not chosen:
                dropped.append((item.well_key, item.top_name, item.depth_ft))
    return kept, soft_warning(name, "repeated picks", dropped)
