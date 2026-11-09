"""Render the FEM-vs-FVM GIF: the finite-element solver against FVM on a water doublet.

The FVM results use the SubsurfiX FIM-OBL solver, inspired by open-DARTS; a MODFLOW-style control-volume finite-difference solver is also available.
FEM here is the OpenGeoSys-style continuous Galerkin finite-element solver.

Both solvers run the same case on the same ``FlowModel`` with the same IAPWS-IF97 water, rock (``ROCK``),
Peaceman wells and time-step limit. The default case is the Kansas structural doublet of
``scripts/make_methods_animation.py`` (22 x 22 x 12 cells of 500 m x 500 m x 5 m whose layers follow the structural
top, the south-east block inactive): 30 degC water injected into an 80 degC reservoir through injector INJ (6, 8),
open in layers 1-3; producer PROD (8, 13), open in layers 2-4 and held at 20 MPa. ``--block`` runs the 10 x 10 x 12
layered block of ``examples/block_model.py`` instead: injector (5, 5) at 0.02 m3/s, producer (2, 2) at 20 MPa, both
open in every layer, 30 years. Rate, report times, the OBL pressure range and the FEM stabilization are
command-line options.

* FVM (``run_fim_thermal_water``): cell-centred finite volumes, fully implicit pressure and temperature,
  water properties through operator-based linearization (OBL) tables.
* FEM (``run_fem_thermal_water``): OpenGeoSys-style FEM, trilinear hexahedra with pressure and temperature
  at the cell corners, streamline-upwind stabilization, Picard iteration.

The top row shows pressure (MPa), the bottom row temperature (degC). The printed summary gives the differences;
``--check`` exits with status 1 when they exceed the ``--check-dt`` / ``--check-balance`` limits.

    python scripts/make_fem_vs_fim_animation.py
    python scripts/make_fem_vs_fim_animation.py --block
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from subsurfix import _native as nat
from subsurfix.viz3d import LayeredGrid, Well, animate_layered

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from block_model import column_wells, layered_block
from kansas_window import BG_NOTE, background_args, kansas_structural, section_wells

INJ, PROD = (6, 8), (8, 13)
INJ_K, PROD_K = list(range(0, 3)), list(range(1, 4))  # open layers, 0 = top
YR = 3.15576e7
YEARS = np.arange(5.0, 200.01, 5.0)  # default report and frame times (years); see --times
DT_MAX = 0.25  # maximum time step (years), the same for both solvers
RATE, BHP, T_INJ, T_RES = 0.05, 2.0e7, 30.0, 80.0
ROCK = {"rho": 2650.0, "cp": 900.0, "conductivity": 2.5}
BLOCK_INJ, BLOCK_PROD, BLOCK_RATE = (5, 5), (2, 2), 0.02  # --block: column wells open in every layer
BLOCK_YEARS = np.arange(1.0, 30.01, 1.0)
BLOCK_DT_MAX = 0.05  # --block: maximum time step (years), as in make_mf6_vs_fim_heat_gif.py and notebook 15


def block_grid(bm: dict) -> LayeredGrid:
    """:class:`LayeredGrid` of the flat layered block: top at the origin depth, every cell active."""
    (x0, y0, z0), (dx, dy, dz), (nx, ny, nz) = bm["origin"], bm["cell"], bm["dims"]
    return LayeredGrid(x0 + dx * np.arange(nx + 1), y0 + dy * np.arange(ny + 1), np.full((ny + 1, nx + 1), z0),
                       int(nz), float(dz), np.ones(nx * ny * nz, bool))


def case(a):
    """Property dict, drawing grid and the injector and producer (column (i, j), open layers) of the chosen case."""
    if a.block:
        bm = layered_block()
        ks = list(range(int(bm["dims"][2])))
        return bm, block_grid(bm), (BLOCK_INJ, ks), (BLOCK_PROD, ks)
    bm = kansas_structural()
    grid = LayeredGrid(bm["xs"], bm["ys"], bm["top_nodes"], int(bm["dims"][2]), bm["cell"][2],
                       np.asarray(bm["active"], bool))
    return bm, grid, (INJ, INJ_K), (PROD, PROD_K)


def run(a, bm, grid, inj, prod):
    (ij_inj, k_inj), (ij_prod, k_prod) = inj, prod
    nx, ny, nz = grid.shape
    n = nx * ny * nz
    perm = np.asarray(bm["perm"], float)
    phi = np.asarray(bm["phi"], float)
    act = np.asarray(grid.active, bool)

    def col(i, j, ks):
        return [i + nx * (j + ny * k) for k in ks]

    if a.block:
        model = nat.FlowModel.from_regular(bm["origin"], bm["cell"], bm["dims"], perm, phi)
        wells = column_wells(bm, ij_inj, ij_prod, rate=a.rate, bhp=BHP)
    else:
        model = nat.FlowModel.from_corner_point(bm["grid"], perm, phi)
        wells = [{"name": "INJ", "kind": "injector", "cells": col(*ij_inj, k_inj), "rate": a.rate, "radius": 0.1},
                 {"name": "PROD", "kind": "producer", "cells": col(*ij_prod, k_prod), "bhp": BHP, "radius": 0.1}]
    times = [float(y * YR) for y in a.years]
    p0, t0 = np.full(n, BHP), np.full(n, T_RES + 273.15)
    dt_max = (BLOCK_DT_MAX if a.block else DT_MAX) * YR

    reused = None
    if a.fvm_from_npz:
        if a.block:
            raise ValueError("--fvm-from-npz is for the original Kansas case, not --block")
        with np.load(a.fvm_from_npz, allow_pickle=False) as cached:
            if float(cached["rate"]) != a.rate or not np.array_equal(cached["years"], a.years):
                raise ValueError("reused FVM rate/report times do not match the requested Kansas case")
            reused = {"P": cached["P"].copy(), "T": cached["T"].copy(),
                      "stats": json.loads(str(cached["stats"]))}
        for key in ("P", "T"):
            if reused[key].shape != (len(times), n) or not np.array_equal(np.isfinite(reused[key]), np.broadcast_to(act, (len(times), n))):
                raise ValueError("reused FVM field shape/active mask does not match the Kansas model")
        # Validate the recorded well summaries against the intended completions.
        stats = reused["stats"]
        last = a.years[-1]
        expected_t = float(stats[f"producer T at {last:g} yr FIM-OBL / MODFLOW 6 (degC)"].split(" / ")[0])
        expected_p = float(stats[f"injector-column pressure at {last:g} yr FIM-OBL / MODFLOW 6 (MPa)"].split(" / ")[0])
        if abs(reused["T"][-1, col(*ij_prod,k_prod)].mean() - expected_t) > 0.006 or abs(reused["P"][-1, col(*ij_inj,k_inj)].mean() - expected_p) > 0.0006:
            raise ValueError("reused FVM well-completion summaries do not match this Kansas case")
        t_fim = float(stats["run time FIM-OBL / MODFLOW 6 (s)"].split(" / ")[0])
        print("Reusing original Kansas FVM arrays; no FVM solve. Original recorded runtime:", t_fim, "s", flush=True)
    else:
        t = time.time()
        r_fim = nat.run_fim_thermal_water(model, wells, p0, t0, times,
                                          injection_temperature=[T_INJ + 273.15, T_RES + 273.15], rock=ROCK,
                                          p_range=(5e6, a.p_max * 1e6), t_range=(290.0, 360.0),
                                          table_points=(a.p_points, 71), dt_init=86400.0, dt_max=dt_max)
        t_fim = time.time() - t
    print("Starting Kansas FEM coupled thermal-water solve", flush=True)
    t = time.time()
    r_fem = nat.run_fem_thermal_water(model, wells, p0, t0, times,
                                      injection_temperature=[T_INJ + 273.15, T_RES + 273.15], rock=ROCK,
                                      dt_init=86400.0, dt_max=dt_max, stabilization=a.stabilization)
    t_fem = time.time() - t
    print(f"FEM solve finished in {t_fem:.3f} s", flush=True)

    def frames(r):
        P = np.asarray(r["p"]) / 1e6
        T = np.asarray(r["temperature"]) - 273.15
        if P.shape[0] == len(times) + 1:
            P, T = P[1:], T[1:]
        return np.where(act, P, np.nan), np.where(act, T, np.nan)

    def bal(r):
        # Change in place versus net inflow between the first and the last report time, per component.
        mip, net = np.asarray(r["mass_in_place"]), np.asarray(r["mass_net_in"])
        return np.abs((mip[-1] - mip[0]) - (net[-1] - net[0])) / np.abs(net[-1] - net[0])

    def counts(r, it):
        return (f"{r['n_steps']} / {r['n_cuts']} / {r[it]} / {r['n_linear_iterations']} / "
                f"{r['n_linear_fallbacks']}")

    P_fim, T_fim = (reused["P"], reused["T"]) if reused is not None else frames(r_fim)
    P_fem, T_fem = frames(r_fem)
    if a.save_fem_npz:
        np.savez_compressed(a.save_fem_npz, P_fem=P_fem, T_fem=T_fem, times=np.asarray(a.years),
                            rate=a.rate, block=a.block, runtime=t_fem,
                            mass_in_place=np.asarray(r_fem["mass_in_place"]),
                            mass_net_in=np.asarray(r_fem["mass_net_in"]),
                            counts=json.dumps({k:int(r_fem[k]) for k in ("n_steps","n_cuts","n_iterations","n_linear_iterations","n_linear_fallbacks")}))
    b_fim, b_fem = (None if reused is not None else bal(r_fim)), bal(r_fem)
    pc, ic = col(*ij_prod, k_prod), col(*ij_inj, k_inj)
    dT = np.abs(T_fim - T_fem)[:, act]
    tp_fim, tp_fem = T_fim[:, pc].mean(1), T_fem[:, pc].mean(1)
    dtp = float(np.max(np.abs(tp_fim - tp_fem)))
    dp_end = (P_fim[-1] - P_fem[-1])[act]
    last = a.years[-1]
    s = {
        "run time FVM / FEM (s)": f"{t_fim:.1f} / {t_fem:.1f}",
        "FVM steps / cuts / Newton / GMRES iterations / fallbacks": (reused["stats"]["FIM-OBL steps / cuts / Newton / GMRES iterations / fallbacks"] if reused is not None else counts(r_fim, "n_newton")),
        "FEM steps / cuts / Picard / GMRES iterations / fallbacks": counts(r_fem, "n_iterations"),
        "FVM water / energy balance error (relative)": ("not stored separately in reused cache; original maximum " + reused["stats"]["FIM-OBL mass balance error (relative)"] if reused is not None else f"{b_fim[0]:.1e} / {b_fim[1]:.1e}"),
        "FEM water / energy balance error (relative)": f"{b_fem[0]:.1e} / {b_fem[1]:.1e}",
        f"producer T at {last:g} yr FVM / FEM (degC)": f"{tp_fim[-1]:.2f} / {tp_fem[-1]:.2f}",
        "thermal breakthrough, producer 1 degC below initial (yr) FVM / FEM":
            " / ".join(f"{a.years[np.argmax(x < T_RES - 1.0)]:g}" if np.any(x < T_RES - 1.0) else "none"
                       for x in (tp_fim, tp_fem)),
        "producer T every 5th report FVM (degC)": " ".join(f"{v:.1f}" for v in tp_fim[::5]),
        "producer T every 5th report FEM (degC)": " ".join(f"{v:.1f}" for v in tp_fem[::5]),
        "max |producer dT| over report times (degC)": f"{dtp:.2f}",
        "max |dT| any cell, any year (degC)": f"{dT.max():.2f}",
        "mean |dT| over cells and years (degC)": f"{dT.mean():.3f}",
        f"injector-column pressure at {last:g} yr FVM / FEM (MPa)":
            f"{np.nanmean(P_fim[-1][ic]):.3f} / {np.nanmean(P_fem[-1][ic]):.3f}",
        f"max |dp| at {last:g} yr (MPa)": f"{np.abs(dp_end).max():.3f}",
        "active cells": f"{int(act.sum())}",
    }
    if reused is not None:
        s["FVM provenance"] = "original saved Kansas arrays reused unchanged; runtime is the original recorded runtime"
    chk = {"producer_dT": dtp, "balance_fim": (float(reused["stats"]["FIM-OBL mass balance error (relative)"]) if reused is not None else float(np.max(b_fim))), "balance_fem": float(np.max(b_fem))}
    return P_fim, T_fim, P_fem, T_fem, s, chk


def check_failures(chk: dict, a) -> list[str]:
    """Reasons the ``--check`` limits are exceeded (NaN counts as exceeded); empty if all pass."""
    why = []
    if not chk["producer_dT"] <= a.check_dt:
        why.append(f"max |producer T difference| {chk['producer_dT']:.2f} degC > --check-dt {a.check_dt:g}")
    for name, key in (("FVM", "balance_fim"), ("FEM", "balance_fem")):
        if not chk[key] <= a.check_balance:
            why.append(f"{name} balance error {chk[key]:.1e} > --check-balance {a.check_balance:g}")
    return why


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
    ap.add_argument("--block", action="store_true",
                    help="run the 10 x 10 layered block of examples/block_model.py instead of the Kansas model")
    ap.add_argument("--out", default=None,
                    help="GIF path (default image/fem_vs_fim_kansas.gif, or image/fem_vs_fim_block.gif with --block)")
    ap.add_argument("--still", default=None)
    ap.add_argument("--rate", type=float, default=None,
                    help=f"injection rate (m3/s); default {RATE:g} (Kansas) or {BLOCK_RATE:g} (--block)")
    ap.add_argument("--p-max", type=float, default=50.0, help="Operator-table pressure ceiling (MPa)")
    ap.add_argument("--p-points", type=int, default=41, help="Operator-table pressure nodes")
    ap.add_argument("--stabilization", type=float, default=1.0,
                    help="FEM streamline-upwind factor (1 = first-order upwind, 0 = plain Galerkin)")
    ap.add_argument("--times", default=None,
                    help="report/frame times in years, e.g. '0.5:2:0.5,3:20:1' (start:stop:step ranges, or values); "
                         "default 5:200:5 (Kansas) or 1:30:1 (--block)")
    ap.add_argument("--check", action="store_true",
                    help="after the summary, print PASS, or print FAIL and exit with status 1 if a limit is exceeded")
    ap.add_argument("--check-dt", type=float, default=2.0,
                    help="--check limit on the largest producer temperature difference over the report times (degC)")
    ap.add_argument("--check-balance", type=float, default=1e-4,
                    help="--check limit on the water and energy balance error of either solver (relative)")
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
                    help="property of the grey base (Kansas only): permeability (log scale, default) or porosity")
    ap.add_argument("--fvm-from-npz", default=None, help="reuse the original Kansas heat P/T cache; compute FEM only")
    ap.add_argument("--save-fem-npz", default=None, help="cache FEM arrays and balances immediately after its solve")
    ap.add_argument("--save-npz", default=None, help="save the results here (for --from-npz)")
    ap.add_argument("--from-npz", default=None, help="redraw from results saved with --save-npz (no new run)")
    ap.add_argument("--background-cmap", default="porosity_grey",
                    help="colormap of the faint base (default: light greys, darker = lower permeability or porosity)")
    a = ap.parse_args()
    z = np.load(a.from_npz, allow_pickle=False) if a.from_npz else None
    if z is not None:
        a.rate, a.block, a.years = float(z["rate"]), bool(z["block"]), z["times"]
    else:
        a.rate = (BLOCK_RATE if a.block else RATE) if a.rate is None else a.rate
        a.years = (BLOCK_YEARS if a.block else YEARS) if a.times is None else parse_times(a.times)
    a.out = a.out or str(ROOT / "image" / ("fem_vs_fim_block.gif" if a.block else "fem_vs_fim_kansas.gif"))
    bm, grid, inj, prod = case(a)
    if z is not None:
        P_fim, T_fim, P_fem, T_fem = (z[k] for k in ("P_fim", "T_fim", "P_fem", "T_fem"))
        s, chk = json.loads(str(z["summary"])), json.loads(str(z["check"]))
    else:
        P_fim, T_fim, P_fem, T_fem, s, chk = run(a, bm, grid, inj, prod)
    if a.save_npz:
        np.savez_compressed(a.save_npz, P_fim=P_fim, T_fim=T_fim, P_fem=P_fem, T_fem=T_fem,
                            times=np.asarray(a.years), rate=a.rate, block=a.block, summary=json.dumps(s),
                            check=json.dumps(chk))
    for k, v in s.items():
        print(f"{k}: {v}")
    if a.check:
        why = check_failures(chk, a)
        if why:
            print("FAIL: " + "; ".join(why))
            sys.exit(1)
        print("PASS")
    xs, ys, (nx, ny, nz), (dx, dy, dz) = grid.xs, grid.ys, grid.shape, grid.spacing
    wl = []
    for nm, ((i, j), ks), c in (("INJ", inj, "#1f4e9c"), ("PROD", prod, "#b2182b")):
        x, y = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
        zt = grid.surface_depth(x, y)
        wl.append(Well(f"{nm} L{min(ks) + 1}-{max(ks) + 1}", x, y, zt, zt + nz * dz, c,
                       {"open": (zt + min(ks) * dz, zt + (max(ks) + 1) * dz)}))
    if a.block:  # the section and grey-base helpers need the Kansas structural top and property ranges
        sec, bg = None, {"background": None}
    else:
        sec, bg = section_wells(bm, inj[0], prod[0], wl[0].extra["open"], wl[1].extra["open"]), background_args(bm, a)
    span = float(np.hypot((prod[0][0] - inj[0][0]) * dx, (prod[0][1] - inj[0][1]) * dy))
    lo, hi = BHP / 1e6, float(np.nanpercentile(np.concatenate([P_fim.ravel(), P_fem.ravel()]), 99.9))
    po = {"label": "Pressure", "units": "MPa", "clim": (lo, hi), "cmap": a.cmap_pressure, "bar_fmt": "%.1f",
          "plume": a.p_threshold, "plume_frac": a.p_threshold_frac,
          "plume_above_median": None if a.p_threshold_frac or a.p_threshold_abs else a.p_above_background, "ref": BHP / 1e6}  # change from the initial 20 MPa
    to = {"label": "Temperature", "units": "°C", "clim": (T_INJ, T_RES), "cmap": a.cmap_temperature, "bar_fmt": "%.0f",
          "plume": a.t_threshold, "ref": T_RES}
    panels = [("FVM: pressure", P_fim, {**po, "bar": False}), ("FEM: pressure", P_fem, {**po, "bar": True}),
              ("FVM: temperature", T_fim, {**to, "bar": False}), ("FEM: temperature", T_fem, {**to, "bar": True})]
    n_act = int(np.asarray(grid.active).sum())
    where = "(flat layered block)" if a.block else "on the structural top"
    grey = "plain grey shell" if bg["background"] is None else "grey = " + BG_NOTE[a.background]
    head = "Layered block" if a.block else "Mississippian interval, Kansas"
    animate_layered(grid, panels, a.out, times=a.years, wells=wl, shape=(2, 2), panel_size=(480, 380), section=sec,
                    context_opacity=0.22, **bg,
                    background_cmap=a.background_cmap,
                    time_label="{:.1f} years" if np.any(np.mod(a.years, 1)) else "{:.0f} years", fps=4,
                    still=a.still or str(Path(a.out).with_suffix(".png")),
                    header=f"{head}: cold-water injection, FVM vs FEM, {a.rate:g} m³/s",
                    footer=(f"Vertical exaggeration 40x   model {nx * dx / 1000:g} x {ny * dy / 1000:g} km, "
                            f"{nz * dz:g} m thick, {nz} layers {where}, {n_act} active cells\n"
                            f"30 °C water into 80 °C rock; blue: injector, red: producer at 20 MPa, "
                            f"{span / 1000:.1f} km apart, open layers in labels\n"
                            f"{grey}; colour = change: pressure by > "
                            + (f"{a.p_threshold_frac:.0%} of its peak, " if a.p_threshold_frac
                               else f"{a.p_threshold:g} MPa, " if a.p_threshold_abs
                               else f"{a.p_above_background:g} MPa above the median rise, ")
                            + f"temperature by > {a.t_threshold:g} °C;"
                            f"\nmore opaque = larger change"))
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
