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


#: Columns left inactive by :func:`kansas_structural`: the south-east block (i >= 14, j <= 11, 8 x 12 columns,
#: about 4 x 6 km) has no well in it, so the porosity there is extrapolated; leaving it out gives the L-shaped
#: model outline that the wells support.
NOTCH = (14, 11)


def kansas_structural(data_dir: str | Path | None = None, kv_kh: float = 0.1, notch: tuple[int, int] | None = NOTCH) -> dict:
    """The whole KGS model on its structural top, with the unsupported south-east block inactive.

    Same porosity and permeability as :func:`kansas_model`. Adds ``xs``, ``ys`` (node coordinates),
    ``top_nodes`` ((ny + 1) x (nx + 1) top depths from notebook 03, m below the surface), ``active`` (per cell,
    natural order) and ``grid``, a :class:`subsurfix.CornerPointGrid` whose 12 layers of 5 m follow the top
    surface (cell depths per corner, not flat boxes). Inactive cells keep their place in the cell numbering.
    """
    from subsurfix import _native as nat

    d = Path(data_dir) if data_dir is not None else DATA
    m = kansas_model(data_dir=d, kv_kh=kv_kh)
    g = np.load(d / "grid_upscaled.npz")
    nx, ny, nz = m["dims"]
    dz = m["cell"][2]
    xs, ys = np.asarray(g["xs"], float), np.asarray(g["ys"], float)
    top = np.asarray(g["top"], float).reshape(ny + 1, nx + 1)
    act2 = np.ones((ny, nx), dtype=np.int32)
    if notch is not None:
        act2[: notch[1] + 1, notch[0]:] = 0
    active = np.tile(act2.ravel(), nz)
    surfaces = np.array([top.ravel() + dz * k for k in range(nz + 1)])
    g0 = nat.CornerPointGrid.from_surfaces(xs, ys, surfaces)
    m["grid"] = nat.CornerPointGrid(tuple(int(v) for v in m["dims"]), g0.coord, g0.zcorn, active)
    m.update(xs=xs, ys=ys, top_nodes=top, active=active.astype(bool),
             source="Kansas Geological Survey (KGS) property model on its structural top, south-east block inactive")
    return m


def section_along(bm: dict, a: tuple[int, int], b: tuple[int, int], extend: int = 2) -> dict:
    """Columns of active cells on the straight line from column ``a`` to column ``b`` (extended by ``extend``
    columns past both ends) for the ``section`` option of ``subsurfix.viz3d.animate_layered``; ``distance`` is
    measured from column ``a``."""
    nx, ny, nz = (int(v) for v in bm["dims"])
    dx, dy, dz = (float(v) for v in bm["cell"])
    tn = np.asarray(bm["top_nodes"], float)
    tops = 0.25 * (tn[:-1, :-1] + tn[1:, :-1] + tn[:-1, 1:] + tn[1:, 1:])
    act = np.asarray(bm["active"], bool)
    va = np.subtract(b, a, dtype=float)
    length = float(np.hypot(va[0] * dx, va[1] * dy))
    n = int(np.ceil(max(abs(va[0]), abs(va[1]))))
    cols, dist, top = [], [], []
    for t in np.arange(-extend, n + extend + 1) / max(n, 1):
        i, j = (int(round(v)) for v in np.add(a, t * va))
        if not (0 <= i < nx and 0 <= j < ny) or (cols and cols[-1][0] == i + nx * j):
            continue
        cells = [i + nx * (j + ny * k) for k in range(nz)]
        if not act[cells[0]]:
            continue
        cols.append(cells)
        dist.append(t * length)
        top.append(float(tops[j, i]))
    return {"columns": cols, "distance": dist, "tops": top, "dz": dz, "length": length}


_COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def section_wells(bm: dict, inj: tuple[int, int], prod: tuple[int, int], inj_open, prod_open, extend: int = 2) -> dict:
    """``section_along`` between an injector and a producer, oriented so the more northern (then more eastern)
    end is on the right whatever the well order; adds the wells and the compass names of both ends."""
    a, b = (inj, prod) if (prod[1] - inj[1], prod[0] - inj[0]) >= (0, 0) else (prod, inj)
    sec = section_along(bm, a, b, extend)
    ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    right = _COMPASS[int(np.round(ang / 45.0)) % 8]
    left = _COMPASS[(int(np.round(ang / 45.0)) + 4) % 8]
    d_inj = 0.0 if a == inj else sec["length"]
    sec["wells"] = [(d_inj, *inj_open, "#1f4e9c", "INJ"), (sec["length"] - d_inj, *prod_open, "#b2182b", "PROD")]
    sec["ends"] = (left, right)
    sec["title"] = f"cross-section {left} to {right} through the wells"
    return sec


#: footer note for the grey base of the Kansas animations
BG_NOTE = {"permeability": "log permeability (darker = lower)",
           "porosity": "porosity (darker = tighter rock)"}


def background_args(bm, a) -> dict:
    """``animate_layered`` keywords for the grey base: permeability (log, 1-1000 mD; default) or porosity
    (0.06-0.26), from the ``--background`` / ``--no-background`` options of the Kansas scripts."""
    if getattr(a, "no_background", False):
        return {"background": None}
    if getattr(a, "background", "permeability") == "porosity":
        return {"background": np.asarray(bm["phi"], float), "background_clim": (0.06, 0.26),
                "background_label": "porosity"}
    return {"background": np.asarray(bm["k_md"], float), "background_clim": (1.0, 1000.0), "background_log": True,
            "background_label": "permeability (mD)"}
