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
    find_column,
    init_source,
    read_rows,
    soft_warning,
    source_name,
)
from subsurfix.edges.schemas import SURVEYS_SCHEMA, array_table, without_well_id
from subsurfix.errors import ConfigError, FormatError

_BODY = without_well_id(SURVEYS_SCHEMA)
_TUPLE_FIELDS = ("key_col", "md_col", "inc_col", "azi_col")


@dataclass(frozen=True, kw_only=True)
class SurveySource:
    """One directional-survey table. ``md_unit`` is required."""

    path: str
    md_unit: str
    sheet: str | None = None
    delimiter: str = ","
    key_col: tuple[str, ...] = ("UWI",)
    md_col: tuple[str, ...] = ("Measured Depth_ft", "Measured Depth", "MD")
    inc_col: tuple[str, ...] = ("Deviation Angle_deg", "Inclination", "Inc")
    azi_col: tuple[str, ...] = ("Deviation Azimuth_deg", "Azimuth", "Azi")

    def __post_init__(self) -> None:
        init_source(self, _TUPLE_FIELDS)


class _Station:
    __slots__ = ("well_key", "md_ft", "inc_deg", "azi_deg", "source", "order")

    def __init__(
        self,
        well_key: str,
        md_ft: float,
        inc_deg: float,
        azi_deg: float,
        source: str,
        order: int,
    ) -> None:
        self.well_key = well_key
        self.md_ft = md_ft
        self.inc_deg = inc_deg
        self.azi_deg = azi_deg
        self.source = source
        self.order = order


def load_surveys(sources: Sequence[SurveySource]) -> pa.Table:
    """Load survey stations. Measured depth is feet; angles are degrees.

    NaN rule: a row with no key or NaN MD is dropped. NaN inclination or
    azimuth is ``FormatError``. No station is invented. The table has no
    ``well_id`` yet. No sources returns an empty table.
    """
    by_well: dict[str, list[_Station]] = {}
    for source in sources:
        stations, messages = _read_source(source)
        ignored: list[object] = []
        for well_key, group in stations.items():
            if well_key in by_well:
                ignored.extend((well_key, station.md_ft) for station in group)
            else:
                by_well[well_key] = group
        ignored_msg = soft_warning(
            source_name(source.path),
            "survey rows already set by an earlier source",
            ignored,
        )
        if ignored_msg is not None:
            messages.append(ignored_msg)
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)
    flat = [station for group in by_well.values() for station in group]
    flat.sort(key=lambda station: (station.well_key, station.md_ft, station.order))
    return array_table(
        _BODY,
        {
            "well_key": [station.well_key for station in flat],
            "md_ft": [station.md_ft for station in flat],
            "inc_deg": [station.inc_deg for station in flat],
            "azi_deg": [station.azi_deg for station in flat],
            "source": [station.source for station in flat],
        },
    )


def _read_source(source: SurveySource) -> tuple[dict[str, list[_Station]], list[str]]:
    name = source_name(source.path)
    if source.md_unit not in {"ft", "m"}:
        raise ConfigError(f"{name}: md_unit must be 'ft' or 'm'")
    header, rows = read_rows(source.path, source.sheet, source.delimiter)
    key_i = find_column(header, source.key_col, required=True, what="well key", source=name)
    md_i = find_column(header, source.md_col, required=True, what="md", source=name)
    inc_i = find_column(header, source.inc_col, required=True, what="inclination", source=name)
    azi_i = find_column(header, source.azi_col, required=True, what="azimuth", source=name)
    assert key_i is not None and md_i is not None and inc_i is not None and azi_i is not None
    dropped_keys: list[object] = []
    dropped_md: list[object] = []
    unparsable: list[object] = []
    pending: dict[str, list[_Station]] = {}
    order = 0
    for row_number, row in enumerate(rows, start=1):
        try:
            key = cell_to_key(row[key_i])
        except FormatError as exc:
            raise FormatError(f"{name}: row {row_number}: {exc}") from exc
        md, md_bad = _parse_number(row[md_i])
        inc, inc_bad = _parse_number(row[inc_i])
        azi, azi_bad = _parse_number(row[azi_i])
        if md_bad:
            unparsable.append(f"row {row_number} md={row[md_i]!r}")
        if key is None:
            dropped_keys.append(row_number)
            continue
        if md != md:
            dropped_md.append(row_number)
            continue
        if inc != inc or azi != azi or inc_bad or azi_bad:
            raise FormatError(f"{name}: well {key} at md {md}: inclination or azimuth missing")
        if inc < 0.0 or inc > 180.0:
            raise FormatError(
                f"{name}: well {key} at md {md}: inclination {inc} outside [0, 180]"
            )
        md_ft = md / 0.3048 if source.md_unit == "m" else md
        azi_deg = azi % 360.0
        pending.setdefault(key, []).append(_Station(key, md_ft, inc, azi_deg, name, order))
        order += 1
    messages = [
        message
        for message in (
            soft_warning(name, "rows with no well key", dropped_keys),
            soft_warning(name, "rows with NaN md", dropped_md),
            soft_warning(name, "unparsable numbers", unparsable),
        )
        if message is not None
    ]
    repeated: list[object] = []
    kept: dict[str, list[_Station]] = {}
    for well_key, group in pending.items():
        group.sort(key=lambda station: (station.md_ft, station.order))
        unique: list[_Station] = []
        for station in group:
            if unique and unique[-1].md_ft == station.md_ft:
                repeated.append((well_key, station.md_ft))
                continue
            unique.append(station)
        kept[well_key] = unique
    repeat_msg = soft_warning(name, "repeated md", repeated)
    if repeat_msg is not None:
        messages.append(repeat_msg)
    return kept, messages
