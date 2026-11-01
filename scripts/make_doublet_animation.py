"""Render the README 3D animation of a geothermal doublet on the Kansas (KGS) property window.

By default the model is the 10 x 10 x 12 window (cells of 500 m x 500 m x 5 m) of the Kansas Geological Survey
property model read from ``examples/data/kansas_window`` by ``examples/kansas_window.py`` (the window of the
"Real-world case" sections of examples 10 and 12). ``--block`` uses the layered block model of
``examples/block_model.py`` instead. Cold water (30 degC) is injected into an 80 degC reservoir for 30 years
(thermal settings of example 06).

    python scripts/make_doublet_animation.py
    python scripts/make_doublet_animation.py --block --out block.gif
    python scripts/make_doublet_animation.py --npz frames.npz   # frames from another solver (e.g. MODFLOW 6)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from subsurfix.viz3d import (
    RegularGrid,
    Well,
    animate_grid,
    load_frames_npz,
    save_frames_npz,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from block_model import layered_block
from kansas_window import kansas_window

INJ, PROD = (5, 5), (2, 2)
YEARS = np.arange(1.0, 30.01, 1.0)


def load_block(block: bool = False):
    bm = layered_block() if block else kansas_window(i0=7, j0=7, nx=10, ny=10)
    return RegularGrid(bm["origin"], bm["cell"], bm["dims"]), bm["phi"], bm["k_md"]


def run_subsurfix(block: bool = False):
    from subsurfix import _native as nat

    grid, phi, k_md = load_block(block)
    NX, NY = grid.shape[0], grid.shape[1]
    nz = grid.shape[2]
    n = grid.n_cells
    k = k_md * 9.869e-16
    m = nat.FlowModel.from_regular(
        list(grid.origin),
        list(grid.spacing),
        list(grid.shape),
        np.column_stack([k, k, 0.1 * k]),
        phi,
    )

    def cid(i, j, kk):
        return i + NX * (j + NY * kk)

    props = {
        "rho_fluid": 1000.0,
        "cp_fluid": 4180.0,
        "rho_rock": 2650.0,
        "cp_rock": 900.0,
        "cond_fluid": 0.6,
        "cond_rock": 2.5,
    }
    wells = [
        {
            "name": "INJ",
            "kind": "injector",
            "cells": [cid(*INJ, kk) for kk in range(nz)],
            "rate": 0.1,
            "radius": 0.1,
        },
        {
            "name": "PROD",
            "kind": "producer",
            "cells": [cid(*PROD, kk) for kk in range(nz)],
            "bhp": 2e7,
            "radius": 0.1,
        },
    ]
    yr = 3.156e7
    mobility = np.full(n, 1 / 5e-4)  # 1/mu for water, mu = 5e-4 Pa s (as in example 06)
    res = m.run_heat(
        mobility,
        wells,
        [30.0, 0.0],
        np.full(n, 80.0),
        [y * yr for y in YEARS],
        props=props,
        dt_max=0.05 * yr,
    )
    T = np.asarray(res["temperature"])
    bal = np.abs(np.array(res["energy_change"]) - np.array(res["energy_in"])) / np.abs(
        res["energy_in"]
    )
    tp = T[:, [cid(*PROD, kk) for kk in range(nz)]].mean(axis=1)
    print(
        f"{n} cells, median k {np.median(k_md):.1f} mD, porosity {phi.min():.3f}-{phi.max():.3f}"
    )
    print(f"energy balance error max {bal.max():.1e}")
    print(
        "producer mean T (degC) at years 1/5/10/15/20/25/30:",
        [round(float(tp[int(y) - 1]), 2) for y in (1, 5, 10, 15, 20, 25, 30)],
    )
    return grid, T, YEARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz", help="frames file written by subsurfix.viz3d.save_frames_npz"
    )
    ap.add_argument("--block", action="store_true", help="layered block model instead of the Kansas window")
    ap.add_argument("--out", default=str(ROOT / "image" / "doublet_3d.gif"))
    ap.add_argument("--still", default=None)
    ap.add_argument("--save-npz", default=None)
    a = ap.parse_args()
    if a.npz:
        grid, frames, times = load_frames_npz(a.npz)[:3]
    else:
        grid, frames, times = run_subsurfix(a.block)
    if a.save_npz:
        save_frames_npz(a.save_npz, grid, frames, times)
    (x0, y0, _), (dx, dy, dz), nz = grid.origin, grid.spacing, grid.shape[2]
    wells = [
        Well(nm, x0 + (ij[0] + 0.5) * dx, y0 + (ij[1] + 0.5) * dy, 0.0, nz * dz, col)
        for nm, ij, col in (("INJ", INJ, "#1f4e9c"), ("PROD", PROD, "#b2182b"))
    ]
    animate_grid(
        grid,
        frames,
        a.out,
        times=times,
        wells=wells,
        label="Temperature",
        units="°C",
        clim=(30, 80),
        z_exaggeration=25.0,
        title="Doublet on a layered block model" if a.block else "Doublet on the Kansas KGS property window",
        time_label="{:.0f} years",
        still=a.still,
        fps=6,
        width=720,
        height=500,
        hold_last=5,
    )
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
