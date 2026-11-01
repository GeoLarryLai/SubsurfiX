# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Kansas (KGS) property model, whole or a window, in the same structure as :func:`block_model.layered_block`.

Reads the two small files in ``examples/data/kansas_window`` (grid from notebook 03, property model from
notebook 04), so the examples can swap the block model for a real-world property field and reuse the rest of
their code (``column_wells``, ``describe``, ``cid``). :func:`kansas_model` returns the whole 22 × 22 × 12 grid
(500 × 500 × 5 m cells, 11 × 11 km × 60 m); :func:`kansas_window` returns a sub-block of it.

Every cell of the stored model has a porosity: the notebook 04 sequential Gaussian simulation (realization 0)
fills the whole grid, so there are no inactive cells and no values are substituted. The stored ``ok`` array is
the ordinary-kriging porosity estimate from notebook 04 (a smooth trend, not a validity mask); it is not used.
The grid is flat (layer tops at 0, 5, … m); the structural top surface in ``grid_upscaled.npz`` is not applied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from block_model import MD

DATA = Path(__file__).resolve().parent / "data" / "kansas_window"


def kansas_window(i0: int = 7, j0: int = 7, nx: int = 10, ny: int = 10, nz: int | None = None,
                  data_dir: str | Path | None = None, kv_kh: float = 0.1) -> dict:
    """nx × ny columns starting at (i0, j0) of the KGS grid, top ``nz`` layers (default all).

    Porosity is clipped to 0.03–0.30 and permeability follows notebook 05: k = 10^(−1 + 14 φ) mD, doubled in
    clean cells. ``units`` lists one entry per layer (mean porosity, median permeability) so ``describe`` works.
    """
    d = Path(data_dir) if data_dir is not None else DATA
    g, pm = np.load(d / "grid_upscaled.npz"), np.load(d / "property_model.npz")
    o0 = tuple(float(v) for v in g["origin"])
    cell = tuple(float(c) for c in g["cell"])
    d0 = tuple(int(v) for v in g["dims"])
    nz = d0[2] if nz is None else int(nz)
    if not (0 <= i0 and i0 + nx <= d0[0] and 0 <= j0 and j0 + ny <= d0[1] and 0 < nz <= d0[2]):
        raise ValueError(f"window ({i0}, {j0}, {nx}, {ny}, {nz}) outside grid {d0}")

    def sub(a):
        return np.asarray(a).reshape(d0[2], d0[1], d0[0])[:nz, j0:j0 + ny, i0:i0 + nx].ravel()

    phi = np.clip(sub(pm["phi"]), 0.03, 0.30).astype(float)
    clean = sub(pm["clean"]).astype(int)
    k_md = 10 ** (-1.0 + 14.0 * phi) * np.where(clean == 1, 2.0, 1.0)
    k = k_md * MD
    lay_phi = phi.reshape(nz, -1).mean(axis=1)
    lay_k = np.median(k_md.reshape(nz, -1), axis=1)
    units = tuple((f"layer {z}", 1, float(lay_phi[z]), float(np.round(lay_k[z], 2))) for z in range(nz))
    return {"origin": (o0[0] + i0 * cell[0], o0[1] + j0 * cell[1], 0.0), "cell": cell, "dims": (nx, ny, nz),
            "phi": phi, "k_md": k_md, "perm": np.column_stack([k, k, kv_kh * k]), "units": units,
            "clean": clean, "window": (i0, j0), "source": "Kansas Geological Survey (KGS) property window"}


def kansas_model(data_dir: str | Path | None = None, kv_kh: float = 0.1) -> dict:
    """The whole KGS property model: 22 × 22 × 12 cells of 500 × 500 × 5 m (11 × 11 km × 60 m, 5808 cells).

    Same porosity and permeability rules as :func:`kansas_window`; all cells are active.
    """
    d = Path(data_dir) if data_dir is not None else DATA
    nx, ny, nz = (int(v) for v in np.load(d / "grid_upscaled.npz")["dims"])
    m = kansas_window(0, 0, nx, ny, nz, data_dir=d, kv_kh=kv_kh)
    m["source"] = "Kansas Geological Survey (KGS) property model, whole grid"
    return m
