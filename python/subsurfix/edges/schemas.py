# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from subsurfix.errors import ConfigError, FormatError

HEADERS_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.int32(), nullable=False),
        pa.field("well_key", pa.string(), nullable=False),
        pa.field("well_label", pa.string(), nullable=False),
        pa.field("x_m", pa.float64(), nullable=False),
        pa.field("y_m", pa.float64(), nullable=False),
        pa.field("kb_elev_ft", pa.float64(), nullable=False),
        pa.field("datum_type", pa.string(), nullable=False),
        pa.field("xy_source", pa.string(), nullable=False),
        pa.field("kb_source", pa.string(), nullable=False),
        # Optional LAS-match hints (null when the header table has none).
        pa.field("lat_deg", pa.float64(), nullable=True),
        pa.field("lon_deg", pa.float64(), nullable=True),
        pa.field("location", pa.string(), nullable=True),
        pa.field("td_ft", pa.float64(), nullable=True),
    ]
)
TOPS_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.int32(), nullable=False),
        pa.field("well_key", pa.string(), nullable=False),
        pa.field("top_name", pa.string(), nullable=False),
        pa.field("depth_ft", pa.float64(), nullable=False),
        pa.field("depth_ref", pa.uint8(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)
SURVEYS_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.int32(), nullable=False),
        pa.field("well_key", pa.string(), nullable=False),
        pa.field("md_ft", pa.float64(), nullable=False),
        pa.field("inc_deg", pa.float64(), nullable=False),
        pa.field("azi_deg", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)
SELECTION_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.int32(), nullable=False),
        pa.field("well_key", pa.string(), nullable=False),
        pa.field("in_area", pa.bool_(), nullable=False),
        pa.field("has_tops", pa.bool_(), nullable=False),
        pa.field("selected", pa.bool_(), nullable=False),
    ]
)
WELL_IDS_SCHEMA = pa.schema(
    [
        pa.field("well_key", pa.string(), nullable=False),
        pa.field("well_id", pa.int32(), nullable=False),
    ]
)

PTS_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.int32(), nullable=False),
        pa.field("well_key", pa.string(), nullable=False),
        pa.field("pass_name", pa.string(), nullable=False),
        pa.field("seq", pa.int32(), nullable=False),
        pa.field("md_ft", pa.float64(), nullable=False),
        pa.field("temp_c", pa.float64(), nullable=False),
        pa.field("pressure_mpa", pa.float64(), nullable=False),
        pa.field("flow", pa.float64(), nullable=False),
        pa.field("time", pa.string(), nullable=False),
        pa.field("depth_tied", pa.bool_(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "subsurfix.edges_headers/1": HEADERS_SCHEMA,
    "subsurfix.edges_tops/1": TOPS_SCHEMA,
    "subsurfix.edges_surveys/1": SURVEYS_SCHEMA,
    "subsurfix.edges_selection/1": SELECTION_SCHEMA,
    "subsurfix.edges_well_ids/1": WELL_IDS_SCHEMA,
    "subsurfix.edges_pts/1": PTS_SCHEMA,
}


def without_well_id(schema: pa.Schema) -> pa.Schema:
    """Schema of a ``load_*`` table, before ``attach_well_ids``."""
    return pa.schema([field for field in schema if field.name != "well_id"])


def array_table(schema: pa.Schema, columns: Mapping[str, Sequence[object]]) -> pa.Table:
    """Build a table whose field names, types, and nullability match ``schema``."""
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(table: pa.Table, path: Path, schema_name: str) -> None:
    """Write ``table`` as zstd Parquet with only ``subsurfix_schema`` metadata.

    NaN rule: not applicable (the table is stored as given). Unknown schema
    names are ``ConfigError``. A schema mismatch is ``FormatError``.
    """
    path = Path(path)
    try:
        schema = SCHEMAS[schema_name]
    except KeyError:
        raise ConfigError(f"unknown schema {schema_name!r}") from None
    if not table.schema.equals(schema, check_metadata=False):
        raise FormatError(f"{path.name}: table schema does not match {schema_name}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer = pq.ParquetWriter(tmp, table.schema, compression="zstd", store_schema=False)
        try:
            writer.write_table(table)
            writer.add_key_value_metadata({"subsurfix_schema": schema_name})
        finally:
            writer.close()
        descriptor = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
