# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping

import pyarrow as pa

from subsurfix.edges.schemas import (
    HEADERS_SCHEMA,
    PTS_SCHEMA,
    SURVEYS_SCHEMA,
    TOPS_SCHEMA,
    WELL_IDS_SCHEMA,
    array_table,
)
from subsurfix.errors import ConfigError, FormatError

_ID_LIMIT = 2**31 - 1


def assign_well_ids(
    keys: Iterable[str],
    existing: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Assign stable ``int32`` well ids.

    New keys are numbered in Unicode code-point order after every existing id.
    Ids are never reused. NaN rule: not applicable.
    """
    current = dict(existing or {})
    seen: set[int] = set()
    for key, well_id in current.items():
        if not isinstance(key, str):
            raise ConfigError(f"existing well key {key!r} is not text")
        if isinstance(well_id, bool) or not isinstance(well_id, int) or not 0 <= well_id <= _ID_LIMIT:
            raise ConfigError(f"existing well id for {key!r} must be in 0..=2**31-1")
        if well_id in seen:
            raise ConfigError(f"existing well id {well_id} is duplicated")
        seen.add(well_id)
    fresh: list[str] = []
    seen_keys: set[str] = set()
    for key in keys:
        if not isinstance(key, str):
            raise ConfigError(f"well key {key!r} is not text")
        if key in seen_keys or key in current:
            seen_keys.add(key)
            continue
        seen_keys.add(key)
        fresh.append(key)
    new_keys = sorted(fresh)
    next_id = max(current.values()) + 1 if current else 0
    for key in new_keys:
        if next_id > _ID_LIMIT:
            raise ConfigError("well id overflow past 2**31 - 1")
        current[key] = next_id
        next_id += 1
    return current


def attach_well_ids(table: pa.Table, ids: Mapping[str, int]) -> pa.Table:
    """Insert ``well_id`` as column 0 and sort by the schema row order.

    Rows whose ``well_key`` is absent from ``ids`` are dropped (one warning).
    NaN rule: not applicable (rows are kept or dropped; floats are copied).
    """
    schema = _schema_with_id(table)
    keys = table.column("well_key").to_pylist()
    kept: list[int] = []
    dropped: list[str] = []
    for index, key in enumerate(keys):
        if key in ids:
            kept.append(index)
        else:
            dropped.append(str(key))
    if dropped:
        warnings.warn(
            f"dropped {len(dropped)} rows whose well_key is not in the registry; first {dropped[:5]}",
            UserWarning,
            stacklevel=2,
        )
    well_ids = [ids[keys[index]] for index in kept]
    order = _sorted_rows(table, kept, well_ids)
    columns: dict[str, list[object]] = {
        "well_id": [ids[keys[index]] for index in order],
    }
    for field in schema:
        if field.name == "well_id":
            continue
        values = table.column(field.name).to_pylist()
        columns[field.name] = [values[index] for index in order]
    return array_table(schema, columns)


def well_id_table(ids: Mapping[str, int]) -> pa.Table:
    """Registry rows sorted by ``well_id``.

    NaN rule: not applicable.
    """
    pairs = sorted(ids.items(), key=lambda item: (item[1], item[0]))
    return array_table(
        WELL_IDS_SCHEMA,
        {
            "well_key": [key for key, _well_id in pairs],
            "well_id": [well_id for _key, well_id in pairs],
        },
    )


def _schema_with_id(table: pa.Table) -> pa.Schema:
    names = set(table.column_names)
    if "top_name" in names:
        return TOPS_SCHEMA
    if "pass_name" in names:
        return PTS_SCHEMA
    if "md_ft" in names:
        return SURVEYS_SCHEMA
    if "x_m" in names:
        return HEADERS_SCHEMA
    raise FormatError("table does not match an edges schema")


def _sorted_rows(table: pa.Table, kept: list[int], well_ids: list[int]) -> list[int]:
    names = set(table.column_names)
    if "top_name" in names:
        secondary = table.column("top_name").to_pylist()
        decorated = [(well_ids[slot], secondary[kept[slot]], slot) for slot in range(len(kept))]
    elif "pass_name" in names:
        passes = table.column("pass_name").to_pylist()
        seqs = table.column("seq").to_pylist()
        decorated = [
            (well_ids[slot], passes[kept[slot]], seqs[kept[slot]], slot) for slot in range(len(kept))
        ]
    elif "md_ft" in names:
        secondary = table.column("md_ft").to_pylist()
        decorated = [(well_ids[slot], secondary[kept[slot]], slot) for slot in range(len(kept))]
    else:
        decorated = [(well_ids[slot], slot) for slot in range(len(kept))]
    decorated.sort()
    return [kept[item[-1]] for item in decorated]
