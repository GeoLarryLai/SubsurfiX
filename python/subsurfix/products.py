# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Read stage products of a project folder as ``pyarrow.Table`` or NumPy.

``root`` is the project folder (the one holding ``project.toml``). Products
live at ``root/stages/<stage>/<file>`` (``root/export/<file>`` for export) and
are read with ``pyarrow.parquet.read_table(..., memory_map=True)``.

A well is given by id (``int``, column ``well_id``) or by name (``str``,
column ``well_name``). Arrays returned may be read-only views on the
memory-mapped file; call ``.copy()`` to modify them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from subsurfix import _native, errors

Root = str | os.PathLike[str]

LOG_STAGES: dict[str, tuple[str, str]] = {
    "raw": ("ingest", "logs_raw.parquet"),
    "qc": ("qc", "logs_qc.parquet"),
    "norm": ("normalize", "logs_norm.parquet"),
    "filled": ("fill", "logs_filled.parquet"),
    "final": ("refresh", "logs_final.parquet"),
}

_PRODUCTS: dict[str, list[str]] = _native.stage_products()


def read_product(root: Root, stage: str, file: str, filters=None) -> pa.Table:
    """One Parquet product of ``stage``; ``filters`` as in ``pyarrow.parquet.read_table``.

    Raises ``InvalidInputError`` if ``(stage, file)`` is not a Parquet product
    of the pipeline, ``NotFoundError`` if the file has not been written yet.
    """
    if file not in _PRODUCTS.get(stage, []) or not file.endswith(".parquet"):
        raise errors.InvalidInputError(f"{stage}/{file} is not a Parquet stage product")
    rel = f"export/{file}" if stage == "export" else f"stages/{stage}/{file}"
    path = Path(root) / rel
    if not path.is_file():
        raise errors.NotFoundError(f"{rel} not found; run the pipeline through stage '{stage}'")
    return pq.read_table(path, memory_map=True, filters=filters)


def read_wells(root: Root) -> pa.Table:
    """``ingest/wells.parquet``: one row per well, sorted by ``well_id``."""
    return read_product(root, "ingest", "wells.parquet")


def read_logs(root: Root, well: int | str | None = None, stage: str = "final") -> pa.Table:
    """Curves on the MD grid after ``stage`` (``raw``, ``qc``, ``norm``, ``filled``, ``final``).

    ``well=None`` gives every well. Raises ``InvalidInputError`` for another
    ``stage``, ``NotFoundError`` for an unknown well.
    """
    if stage not in LOG_STAGES:
        raise errors.InvalidInputError(
            f"unknown log stage {stage!r}; valid: {', '.join(LOG_STAGES)}"
        )
    stage_dir, file = LOG_STAGES[stage]
    return read_product(root, stage_dir, file, filters=_well_filter(root, well))


def read_geometry(root: Root, well: int | str | None = None) -> pa.Table:
    """``ingest/geometry.parquet``: per-sample MD/TVD/x/y; ``well=None`` gives every well."""
    return read_product(root, "ingest", "geometry.parquet", filters=_well_filter(root, well))


def read_tops(root: Root) -> pa.Table:
    """``correlate/tops.parquet``: the consistent tops of every well."""
    return read_product(root, "correlate", "tops.parquet")


def read_suspects(root: Root) -> pa.Table:
    """Rows of ``read_tops`` with ``suspect == True``, sorted by ``rank`` ascending."""
    tops = read_tops(root)
    return tops.filter(pc.equal(tops["suspect"], True)).sort_by("rank")


def read_layer_time(root: Root, well: int | str) -> np.ndarray:
    """Layer time of one well: 1-D ``float64``, one value per grid sample in ``k`` order, NaN where undefined.

    May be a read-only view; call ``.copy()`` to modify. Raises
    ``NotFoundError`` for an unknown well, ``FormatError`` if the column holds nulls.
    """
    table = read_product(
        root, "correlate", "layer_time.parquet", filters=_well_filter(root, well)
    )
    column = table.column("layer_time")
    if column.null_count:
        raise errors.FormatError("layer_time.parquet: column layer_time holds nulls")
    if column.num_chunks == 1:
        return column.chunk(0).to_numpy(zero_copy_only=True)
    if column.num_chunks == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([c.to_numpy() for c in column.chunks])


def read_subhorizons(root: Root) -> pa.Table:
    """``subhorizons/subhorizons.parquet``."""
    return read_product(root, "subhorizons", "subhorizons.parquet")


def _well_filter(root: Root, well: int | str | None) -> list[tuple[str, str, int]] | None:
    """``[("well_id", "=", id)]`` for one well, ``None`` for every well."""
    if well is None:
        return None
    return [("well_id", "=", _well_id(root, well))]


def _well_id(root: Root, well: int | str) -> int:
    if isinstance(well, (bool, np.bool_)):
        raise errors.InvalidInputError(f"well must be an int id or a str name, not {well!r}")
    if isinstance(well, (int, np.integer)):
        ids = read_wells(root).column("well_id")
        if pc.any(pc.equal(ids, int(well))).as_py():
            return int(well)
    elif isinstance(well, str):
        wells = read_wells(root)
        hit = wells.filter(pc.equal(wells["well_name"], well))
        if hit.num_rows:
            return hit.column("well_id")[0].as_py()
    else:
        raise TypeError(f"well must be an int id or a str name, not {type(well).__name__}")
    raise errors.NotFoundError(f"well {well}")


__all__ = [
    "read_product",
    "read_wells",
    "read_logs",
    "read_geometry",
    "read_tops",
    "read_suspects",
    "read_layer_time",
    "read_subhorizons",
]
