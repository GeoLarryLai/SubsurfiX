# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Small self-contained layered block model shared by examples 09, 10 and 12 and the README animations.

Deterministic, no external data. Each example builds its grid with :func:`layered_block` and its wells with
:func:`column_wells`; a different property source (for example a window of a real property model) only has to
return the same dictionary to reuse the rest of an example.
"""
from __future__ import annotations

import numpy as np

MD = 9.869233e-16  # m² per millidarcy

# (name, layers, porosity, horizontal permeability in mD), top to bottom
UNITS = (
    ("upper sand", 3, 0.12, 5.0),
    ("main sand", 3, 0.22, 80.0),
    ("shaly baffle", 3, 0.06, 0.5),
    ("lower sand", 3, 0.16, 20.0),
)


def layered_block(nx: int = 10, ny: int = 10, cell=(500.0, 500.0, 5.0), units=UNITS, kv_kh: float = 0.1) -> dict:
    """Regular nx × ny grid with one porosity and permeability per unit (layers counted from the top).

    Returns ``origin``, ``cell``, ``dims``, ``phi`` (n,), ``k_md`` (n,), ``perm`` (n, 3) in m² (kx, ky, kz) and
    ``units``. Arrays are in SubsurfiX natural order (i fastest, then j, then layer k from the top).
    """
    nz = sum(u[1] for u in units)
    phi_z = np.concatenate([np.full(u[1], u[2]) for u in units])
    k_z = np.concatenate([np.full(u[1], u[3]) for u in units])
    phi = np.repeat(phi_z, nx * ny)
    k_md = np.repeat(k_z, nx * ny)
    k = k_md * MD
    return {"origin": (0.0, 0.0, 0.0), "cell": tuple(float(c) for c in cell), "dims": (nx, ny, nz),
            "phi": phi, "k_md": k_md, "perm": np.column_stack([k, k, kv_kh * k]), "units": units}


def cid(model: dict, i: int, j: int, k: int) -> int:
    nx, ny, _ = model["dims"]
    return i + nx * (j + ny * k)


def column_wells(model: dict, inj=(5, 5), prod=(2, 2), rate: float = 0.01, bhp: float = 2e7, radius: float = 0.1):
    """Vertical injector (reservoir rate, m³/s) and producer (bottom-hole pressure, Pa), open in every layer."""
    nz = model["dims"][2]
    return [{"name": "INJ", "kind": "injector", "rate": rate, "cells": [cid(model, *inj, k) for k in range(nz)], "radius": radius},
            {"name": "PROD", "kind": "producer", "bhp": bhp, "cells": [cid(model, *prod, k) for k in range(nz)], "radius": radius}]


def describe(model: dict) -> str:
    """Markdown table of the units."""
    rows = ["| Unit | Layers | Porosity | k (mD) |", "|---|---|---|---|"]
    top = 0
    for name, n, phi, k in model["units"]:
        rows.append(f"| {name} | {top}–{top + n - 1} | {phi:.2f} | {k:g} |")
        top += n
    return "\n".join(rows)
