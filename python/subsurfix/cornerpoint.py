# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Corner-point grids from GRDECL files and CO2 vertical-equilibrium helpers."""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from subsurfix import _native

MD = 9.869233e-16  # m² per millidarcy


def read_cornerpoint_grdecl(path: str | pathlib.Path) -> tuple[_native.CornerPointGrid, dict[str, np.ndarray]]:
    """Read a GRDECL file (SPECGRID/DIMENS, COORD, ZCORN, ACTNUM and per-cell
    properties such as PERMX/Y/Z, PORO, NTG, MULT*). Returns ``(grid, props)``."""
    text = pathlib.Path(path).read_text()
    grid, props = _native.CornerPointGrid.from_grdecl(text)
    return grid, {k: np.asarray(v, dtype=float) for k, v in props.items()}


def flow_model_from_grdecl(
    grid: _native.CornerPointGrid, props: dict[str, Any], perm_unit: str = "mD", fault_tol_m: float = 1e-3
) -> _native.FlowModel:
    """Build a ``FlowModel`` (TPFA, fault overlap connections) from GRDECL
    properties: PERMX (PERMY, PERMZ default to PERMX), PORO, optional NTG
    and MULT* keywords. ``perm_unit`` is ``"mD"`` (GRDECL convention) or ``"m2"``."""
    if perm_unit not in ("mD", "m2"):
        raise ValueError("perm_unit must be 'mD' or 'm2'")
    f = MD if perm_unit == "mD" else 1.0
    kx = np.asarray(props["PERMX"], dtype=float)
    ky = np.asarray(props.get("PERMY", kx), dtype=float)
    kz = np.asarray(props.get("PERMZ", kx), dtype=float)
    perm = np.column_stack([kx, ky, kz]) * f
    mult = {k: list(map(float, v)) for k, v in props.items() if k.startswith("MULT")}
    ntg = props.get("NTG")
    return _native.FlowModel.from_corner_point(
        grid,
        perm,
        np.asarray(props["PORO"], dtype=float),
        ntg=None if ntg is None else np.asarray(ntg, dtype=float),
        mult=mult or None,
        fault_tol_m=fault_tol_m,
    )


def transmissibilities(model: _native.FlowModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connections of a flow model as ``(a, b, trans)`` arrays (trans in m³)."""
    return model.connections()


VeGrid = _native.VeGrid
run_co2_ve = _native.run_co2_ve
