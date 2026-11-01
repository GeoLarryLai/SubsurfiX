"""Side-by-side GIF: MODFLOW 6 (GWF + GWE) and SubsurfiX FIM-OBL water + heat on the layered block model.

Cold water (30 degC) is injected at 0.02 m3/s into an 80 degC reservoir for 30 years (injector at column (5, 5),
producer at (2, 2) held at 20 MPa, both open in all 12 layers; thermal settings of example 06). MODFLOW 6 solves
steady flow with constant water properties and transient heat transport; FIM-OBL solves pressure and temperature
together with temperature- and pressure-dependent water properties, so the two are expected to differ somewhat.
Needs the ``mf6`` program (``get-modflow :flopy``).

    python scripts/make_mf6_vs_fim_heat_gif.py --out image/mf6_vs_fim_heat.gif
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from subsurfix import _native as nat
from subsurfix.modflow_heat import HeatDoublet, find_mf6, run_gwf_gwe
from subsurfix.viz3d import RegularGrid, Well, animate_grid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from block_model import column_wells, layered_block  # noqa: E402

INJ, PROD = (5, 5), (2, 2)
RATE = 0.02
YR = 3.156e7
YEARS = np.arange(1.0, 30.01, 1.0)
ROCK = {"rho": 2650.0, "cp": 900.0, "conductivity": 2.5}


def run_both(linear_solver="iterative"):
    bm = layered_block()
    nx, ny, nz = bm["dims"]
    n = nx * ny * nz
    wells = column_wells(bm, INJ, PROD, rate=RATE, bhp=2e7)
    times = [float(y * YR) for y in YEARS]
    case = HeatDoublet(bm["origin"], bm["cell"], bm["dims"], bm["perm"], bm["phi"], np.full(n, 80.0), wells,
                       [30.0, 0.0], times, viscosity=5e-4, dt_max=0.05 * YR)
    with tempfile.TemporaryDirectory() as ws:
        t = time.time()
        mf = run_gwf_gwe(case, ws)
        t_mf = time.time() - t
    model = nat.FlowModel.from_regular(list(bm["origin"]), list(bm["cell"]), [nx, ny, nz], bm["perm"], bm["phi"])
    t = time.time()
    fim = nat.run_fim_thermal_water(model, wells, np.full(n, 2e7), np.full(n, 353.15), times,
                                    injection_temperature=[303.15, 353.15], rock=ROCK, p_range=(5e6, 5e7),
                                    t_range=(290.0, 360.0), table_points=(41, 71), dt_init=86400.0,
                                    dt_max=0.05 * YR, linear_solver=linear_solver)
    t_fim = time.time() - t
    T_mf = np.asarray(mf["temperature"])
    T_fim = np.asarray(fim["temperature"]) - 273.15
    if T_fim.shape[0] == len(times) + 1:
        T_fim = T_fim[1:]
    return bm, wells, T_mf, T_fim, fim, t_mf, t_fim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "image" / "mf6_vs_fim_heat.gif"))
    ap.add_argument("--linear-solver", default="iterative")
    a = ap.parse_args()
    assert find_mf6(), "install mf6 with: get-modflow :flopy"
    bm, wells, T_mf, T_fim, fim, t_mf, t_fim = run_both(a.linear_solver)
    nx, ny, nz = bm["dims"]
    pc = wells[1]["cells"]
    tp_mf, tp_fim = T_mf[:, pc].mean(axis=1), T_fim[:, pc].mean(axis=1)
    print(f"run time: MODFLOW 6 {t_mf:.1f} s, FIM-OBL {t_fim:.1f} s "
          f"({fim['n_steps']} steps, {fim['n_newton']} Newton, {fim['n_linear_iterations']} GMRES iterations, "
          f"{fim['n_linear_fallbacks']} fallbacks)")
    print(f"max |dT| any cell, any year: {np.abs(T_mf - T_fim).max():.2f} degC; "
          f"mean |dT|: {np.abs(T_mf - T_fim).mean():.3f} degC")
    print("producer layer-mean T (degC), years 1/5/10/15/20/25/30:")
    for lab, tp in (("MODFLOW 6", tp_mf), ("FIM-OBL", tp_fim)):
        print(f"  {lab:10s}", [round(float(tp[int(y) - 1]), 2) for y in (1, 5, 10, 15, 20, 25, 30)])
    grid = RegularGrid(bm["origin"], bm["cell"], (nx, ny, nz))
    (x0, y0, _), (dx, dy, dz) = grid.origin, grid.spacing
    wl = [Well(nm, x0 + (ij[0] + 0.5) * dx, y0 + (ij[1] + 0.5) * dy, 0.0, nz * dz, col)
          for nm, ij, col in (("INJ", INJ, "#1f4e9c"), ("PROD", PROD, "#b2182b"))]
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for T, title in ((T_mf, "MODFLOW 6 (GWF + GWE)"), (T_fim, "SubsurfiX FIM-OBL")):
            p = Path(tmp) / f"{len(parts)}.gif"
            animate_grid(grid, T, p, times=YEARS, wells=wl, label="Temperature", units="°C", clim=(30, 80),
                         z_exaggeration=25.0, title=f"{title}, {RATE} m³/s", time_label="{:.0f} years",
                         fps=6, width=540, height=420, hold_last=5)
            parts.append([f.convert("RGB") for f in ImageSequence.Iterator(Image.open(p))])
        dur = Image.open(Path(tmp) / "0.gif").info.get("duration", 160)
    frames = []
    for fa, fb in zip(*parts):
        c = Image.new("RGB", (fa.width + fb.width, max(fa.height, fb.height)), "white")
        c.paste(fa, (0, 0))
        c.paste(fb, (fa.width, 0))
        frames.append(c.quantize(colors=200))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
