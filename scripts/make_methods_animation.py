"""Render the README methods-comparison GIF: MODFLOW 6 against FIM-OBL on a water doublet.

Both codes run the same case on the Kansas Geological Survey property model read by
``examples/kansas_window.py`` (22 x 22 x 12 cells of 500 m x 500 m x 5 m whose layers follow the structural top;
the south-east block without well control is inactive): 30 degC water injected into an 80 degC
reservoir through injector INJ (6, 8), in the south, 19 m down-dip of the producer, open only in its three
most porous layers (1-3, porosity 0.28, about 24 D·m); producer PROD (8, 13), near the centre and 2.7 km away, open
in its three most porous layers (2-4, porosity 0.21-0.28, about 12.6 D·m) and held at 20 MPa. Wells, open layers, rate, report times,
MODFLOW 6 tolerance and the OBL pressure range are command-line options.

* MODFLOW 6 (through FloPy and ``subsurfix.modflow_heat``): steady groundwater flow (GWF) with multi-aquifer
  wells, then groundwater energy transport (GWE); constant water density and viscosity.
* FIM-OBL (``run_fim_thermal_water``): fully implicit pressure and temperature with water properties from
  IAPWS-IF97, so viscosity and density change with temperature.

The top row shows pressure (MPa), the bottom row temperature (degC). The printed summary gives the differences.

    python scripts/make_methods_animation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from subsurfix import _native as nat
from subsurfix.modflow_heat import G, HeatDoublet, run_gwf_gwe
from subsurfix.viz3d import LayeredGrid, Well, animate_layered

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from kansas_window import BG_NOTE, background_args, kansas_structural, section_wells

INJ, PROD = (6, 8), (8, 13)
INJ_K, PROD_K = list(range(0, 3)), list(range(1, 4))  # open layers, 0 = top
YR = 3.15576e7
YEARS = np.arange(5.0, 200.01, 5.0)  # default report and frame times (years); see --times
DT_MAX = 0.25  # maximum time step (years), the same for both codes
RATE, BHP, T_INJ, T_RES = 0.05, 2.0e7, 30.0, 80.0
MU = 3.55e-4  # MODFLOW 6 viscosity: water at 80 degC (the undisturbed reservoir)
ROCK = {"rho": 2650.0, "cp": 900.0, "conductivity": 2.5}
PROPS = {"rho_fluid": 1000.0, "cp_fluid": 4180.0, "rho_rock": 2650.0, "cp_rock": 900.0, "cond_fluid": 0.6,
         "cond_rock": 2.5}


def run(ws: Path, a):
    global RATE, YEARS
    RATE, YEARS = a.rate, a.years
    bm = kansas_structural()
    nx, ny, nz = (int(v) for v in bm["dims"])
    n = nx * ny * nz
    perm = np.asarray(bm["perm"], float)
    phi = np.asarray(bm["phi"], float)

    def col(i, j, ks=None):
        return [i + nx * (j + ny * k) for k in (range(nz) if ks is None else ks)]

    wells = [{"name": "INJ", "kind": "injector", "cells": col(*INJ, INJ_K), "rate": RATE, "radius": 0.1},
             {"name": "PROD", "kind": "producer", "cells": col(*PROD, PROD_K), "bhp": BHP, "radius": 0.1}]
    times = [float(y * YR) for y in YEARS]

    t = time.time()
    tn = np.asarray(bm["top_nodes"], float)
    top = 0.25 * (tn[:-1, :-1] + tn[1:, :-1] + tn[:-1, 1:] + tn[1:, 1:])  # column tops, (ny, nx)
    act = np.asarray(bm["active"], bool)
    case = HeatDoublet(bm["origin"], bm["cell"], (nx, ny, nz), perm, phi, np.full(n, T_RES), wells,
                       [T_INJ, 0.0], times, props=PROPS, viscosity=MU, dt_max=DT_MAX * YR, top=top, active=act)
    mf = run_gwf_gwe(case, ws, tolerance=a.mf6_tolerance)
    import flopy

    from subsurfix.modflow_heat import from_layered

    head = from_layered(flopy.utils.HeadFile(ws / "maw_split" / "gwf.hds").get_data(), (nx, ny, nz))
    p_mf = PROPS["rho_fluid"] * G * head / 1e6
    t_mf = time.time() - t
    T_mf = np.asarray(mf["temperature"])

    t = time.time()
    model = nat.FlowModel.from_corner_point(bm["grid"], perm, phi)
    r = nat.run_fim_thermal_water(model, wells, np.full(n, BHP), np.full(n, T_RES + 273.15), times,
                                  injection_temperature=[T_INJ + 273.15, T_RES + 273.15], rock=ROCK,
                                  p_range=(5e6, a.p_max * 1e6), t_range=(290.0, 360.0), table_points=(a.p_points, 71),
                                  dt_init=86400.0, dt_max=DT_MAX * YR)
    t_fim = time.time() - t
    P = np.asarray(r["p"]) / 1e6
    T = np.asarray(r["temperature"]) - 273.15
    if P.shape[0] == len(times) + 1:
        P, T = P[1:], T[1:]
    # Change in place versus net inflow between the first and the last report time, per component.
    mip, net = np.asarray(r["mass_in_place"]), np.asarray(r["mass_net_in"])
    bal = np.abs((mip[-1] - mip[0]) - (net[-1] - net[0])) / np.abs(net[-1] - net[0])

    pc = col(*PROD, PROD_K)
    p_mf = np.where(act, p_mf, np.nan)
    T_mf = np.where(act, T_mf, np.nan)
    P, T = np.where(act, P, np.nan), np.where(act, T, np.nan)
    dT = np.abs(T - T_mf)[:, act]
    tp_fim, tp_mf = T[:, pc].mean(1), T_mf[:, pc].mean(1)
    dp_end = (P[-1] - p_mf)[act]
    over = np.nanmax(p_mf) - BHP / 1e6
    s = {
        "run time FIM-OBL / MODFLOW 6 (s)": f"{t_fim:.1f} / {t_mf:.1f}",
        "FIM-OBL steps / cuts / Newton / GMRES iterations / fallbacks":
            f"{r['n_steps']} / {r['n_cuts']} / {r['n_newton']} / {r['n_linear_iterations']} / {r['n_linear_fallbacks']}",
        "FIM-OBL mass balance error (relative)": f"{float(np.max(bal)):.1e}",
        f"producer T at {YEARS[-1]:g} yr FIM-OBL / MODFLOW 6 (degC)": f"{tp_fim[-1]:.2f} / {tp_mf[-1]:.2f}",
        "thermal breakthrough, producer 1 degC below initial (yr) FIM-OBL / MODFLOW 6":
            " / ".join(f"{YEARS[np.argmax(x < T_RES - 1.0)]:g}" if np.any(x < T_RES - 1.0) else "none"
                       for x in (tp_fim, tp_mf)),
        "producer T every 5th report FIM-OBL (degC)": " ".join(f"{v:.1f}" for v in tp_fim[::5]),
        "producer T every 5th report MODFLOW 6 (degC)": " ".join(f"{v:.1f}" for v in tp_mf[::5]),
        "max |dT| any cell, any year (degC)": f"{dT.max():.2f}",
        "mean |dT| over cells and years (degC)": f"{dT.mean():.3f}",
        f"injector-column pressure at {YEARS[-1]:g} yr FIM-OBL / MODFLOW 6 (MPa)":
            f"{np.nanmean(P[-1][col(*INJ, INJ_K)]):.3f} / {np.nanmean(p_mf[col(*INJ, INJ_K)]):.3f}",
        f"max |dp| at {YEARS[-1]:g} yr (MPa, % of MODFLOW 6 max overpressure)":
            f"{np.abs(dp_end).max():.3f} ({100 * np.abs(dp_end).max() / over:.1f} %)",
        "max cell pressure FIM-OBL / MODFLOW 6 (MPa)": f"{np.nanmax(P):.2f} / {np.nanmax(p_mf):.2f}",
        "FIM-OBL injector BHP max (MPa)": f"{np.max(np.asarray(r['bhp'])[:, 0]) / 1e6:.2f}" if "bhp" in r else "n/a",
        f"cells colder than 79 degC at {YEARS[-1]:g} yr FIM-OBL / MODFLOW 6":
            f"{int((T[-1] < 79).sum())} / {int((T_mf[-1] < 79).sum())}",
        "active cells": f"{int(act.sum())}",
    }
    grid = LayeredGrid(bm["xs"], bm["ys"], bm["top_nodes"], nz, bm["cell"][2], act)
    return bm, grid, P, np.repeat(p_mf[None], len(times), 0), T, T_mf, s


def parse_times(spec: str) -> np.ndarray:
    """``'0.5:2:0.5,3:20:1'`` -> 0.5, 1, 1.5, 2, 3, 4, ..., 20 (years)."""
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b, h = (float(v) for v in part.split(":"))
            out.extend(np.arange(a, b + 1e-9, h))
        else:
            out.append(float(part))
    return np.unique(np.round(out, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "image" / "methods_compare.gif"))
    ap.add_argument("--still", default=None)
    ap.add_argument("--rate", type=float, default=RATE, help="injection rate (m3/s)")
    ap.add_argument("--p-max", type=float, default=50.0, help="OBL table pressure ceiling (MPa)")
    ap.add_argument("--p-points", type=int, default=41, help="OBL table pressure nodes")
    ap.add_argument("--mf6-tolerance", default="normal", choices=["tight", "normal"])
    ap.add_argument("--times", default=None,
                    help="report/frame times in years, e.g. '0.5:2:0.5,3:20:1' (start:stop:step ranges, or values)")
    ap.add_argument("--workspace", default=None, help="keep the MODFLOW 6 files here")
    ap.add_argument("--cmap-pressure", default="viridis_bright", help="pressure colormap (any matplotlib name)")
    ap.add_argument("--cmap-temperature", default="cold",
                    help="temperature colormap (any matplotlib name; default 'cold': dark blue = coldest, pale = 80 degC)")
    ap.add_argument("--p-threshold", type=float, default=3.0,
                    help="colour cells whose pressure changed by more than this (MPa) from the initial 20 MPa")
    ap.add_argument("--p-above-background", type=float, default=3.0,
                    help="default pressure overlay: colour cells whose rise exceeds the median rise over the model "
                         "in that frame by this margin (MPa), so only near-well build-up shows")
    ap.add_argument("--p-threshold-abs", action="store_true",
                    help="use the fixed --p-threshold (MPa above the initial 20 MPa) instead of --p-above-background")
    ap.add_argument("--p-threshold-frac", type=float, default=None,
                    help="instead of --p-threshold: a fraction of the largest pressure change in each panel")
    ap.add_argument("--t-threshold", type=float, default=2.0, help="colour cells cooled by more than this (degC)")
    ap.add_argument("--no-background", action="store_true", help="plain grey shell instead of the grey base")
    ap.add_argument("--background", default="permeability", choices=["permeability", "porosity"],
                    help="property of the grey base: permeability (log scale, default) or porosity")
    ap.add_argument("--save-npz", default=None, help="save the results here (for --from-npz)")
    ap.add_argument("--from-npz", default=None, help="redraw from results saved with --save-npz (no new run)")
    ap.add_argument("--background-cmap", default="porosity_grey",
                    help="colormap of the faint base (default: light greys, darker = lower permeability or porosity)")
    ap.add_argument("--inj", default="6,8", help="injector column 'i,j'")
    ap.add_argument("--prod", default="8,13", help="producer column 'i,j'")
    ap.add_argument("--inj-layers", default="0-2", help="injector open layers 'k0-k1' (0 = top) or 'all'")
    ap.add_argument("--prod-layers", default="1-3", help="producer open layers 'k0-k1' (0 = top)")
    a = ap.parse_args()
    global INJ, PROD, INJ_K, PROD_K
    INJ, PROD = (tuple(int(v) for v in a.inj.split(",")), tuple(int(v) for v in a.prod.split(",")))
    def rng(t):
        return list(range(12)) if t in ("", "all") else list(range(int(t.split("-")[0]), int(t.split("-")[-1]) + 1))

    INJ_K, PROD_K = rng(a.inj_layers), rng(a.prod_layers)
    a.years = YEARS if a.times is None else parse_times(a.times)
    if a.from_npz:
        z = np.load(a.from_npz, allow_pickle=False)
        P, Pm, T, Tm, a.years = z["P"], z["Pm"], z["T"], z["Tm"], z["years"]
        s = json.loads(str(z["stats"]))
        a.rate = float(z["rate"])
        bm = kansas_structural()
        grid = LayeredGrid(bm["xs"], bm["ys"], bm["top_nodes"], int(bm["dims"][2]), bm["cell"][2],
                           np.asarray(bm["active"], bool))
    elif a.workspace:
        bm, grid, P, Pm, T, Tm, s = run(Path(a.workspace), a)
    else:
        with TemporaryDirectory() as tmp:
            bm, grid, P, Pm, T, Tm, s = run(Path(tmp), a)
    if a.save_npz:
        np.savez_compressed(a.save_npz, P=P, Pm=Pm, T=T, Tm=Tm, years=np.asarray(a.years), rate=a.rate,
                            stats=json.dumps(s))
    for k, v in s.items():
        print(f"{k}: {v}")
    xs, ys, nz, dz = bm["xs"], bm["ys"], grid.shape[2], bm["cell"][2]
    wl = []
    for nm, (i, j), c, ks in (("INJ", INJ, "#1f4e9c", INJ_K), ("PROD", PROD, "#b2182b", PROD_K)):
        x, y = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
        z = grid.surface_depth(x, y)
        wl.append(Well(f"{nm} L{min(ks) + 1}-{max(ks) + 1}", x, y, z, z + nz * dz, c,
                       {"open": (z + min(ks) * dz, z + (max(ks) + 1) * dz)}))
    sec = section_wells(bm, INJ, PROD, wl[0].extra["open"], wl[1].extra["open"])
    lo, hi = BHP / 1e6, float(np.nanpercentile(np.concatenate([P.ravel(), Pm.ravel()]), 99.9))
    po = {"label": "Pressure", "units": "MPa", "clim": (lo, hi), "cmap": a.cmap_pressure, "bar_fmt": "%.1f",
          "plume": a.p_threshold, "plume_frac": a.p_threshold_frac,
          "plume_above_median": None if a.p_threshold_frac or a.p_threshold_abs else a.p_above_background, "ref": BHP / 1e6}  # change from the initial 20 MPa
    to = {"label": "Temperature", "units": "°C", "clim": (T_INJ, T_RES), "cmap": a.cmap_temperature, "bar_fmt": "%.0f",
          "plume": a.t_threshold, "ref": T_RES}
    panels = [("MODFLOW 6: pressure", Pm, {**po, "bar": False}), ("MODFLOW 6: temperature", Tm, {**to, "bar": False}),
              ("FIM-OBL: pressure", P, {**po, "bar": True}), ("FIM-OBL: temperature", T, {**to, "bar": True})]
    n_act = int(np.asarray(bm["active"]).sum())
    animate_layered(grid, panels, a.out, times=a.years, wells=wl, shape=(2, 2), panel_size=(480, 380), section=sec,
                    context_opacity=0.22, **background_args(bm, a),
                    background_cmap=a.background_cmap,
                    time_label="{:.1f} years" if np.any(np.mod(a.years, 1)) else "{:.0f} years", fps=4,
                    still=a.still or str(Path(a.out).with_suffix(".png")),
                    header=f"Mississippian interval, Kansas: cold-water doublet, MODFLOW 6 vs FIM-OBL, {a.rate:g} m³/s",
                    footer=(f"Vertical exaggeration 40x   model 11 x 11 km, 60 m thick, 12 layers on the structural top, "
                            f"{n_act} active cells\n30 °C water into 80 °C rock; blue: injector, red: producer at 20 MPa, "
                            f"{sec['length'] / 1000:.1f} km apart, open layers in labels\n"
                            f"grey = " + BG_NOTE[getattr(a, "background", "permeability")] + "; colour = change: pressure by > "
                            + (f"{a.p_threshold_frac:.0%} of its peak, " if a.p_threshold_frac
                               else f"{a.p_threshold:g} MPa, " if a.p_threshold_abs
                               else f"{a.p_above_background:g} MPa above the median rise, ")
                            + f"temperature by > {a.t_threshold:g} °C;"
                            f"\nmore opaque = larger change"))
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
