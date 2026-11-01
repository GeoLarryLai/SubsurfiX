"""Fluid injection on the whole Kansas (KGS) property model with FIM-OBL (local study script).

Runs the example-12 cases (water-oil, oil-gas, CO2-brine) on the Kansas model on its structural top
(``kansas_structural``: 22 x 22 x 12 cells of 500 x 500 x 5 m, layers following the top surface, the
south-east block without well control inactive). Default wells: injector INJ at column (5, 10), structurally low
and south-west of centre, open in the bottom four layers; producer PROD at (9, 16), 25 m up-dip to the north-east, open in its three
most porous layers (2-4, porosity 0.18-0.28, about 17.8 D·m); 3.6 km apart. The bottom layers are tight, so rates are small. The oil-gas producer is held at
19.5 MPa (above the bubble point) so gas does not come out of solution everywhere.
Prints run statistics and writes ``fluids_compare_kansas.gif`` (plus a still of the last frame) and an npz.

    python scripts/kansas_fluids.py --out-dir out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import fluid_cases as fc  # noqa: E402
from kansas_window import BG_NOTE, background_args, kansas_structural, section_wells  # noqa: E402
from subsurfix import _native as nat  # noqa: E402

INJECTED = {"water-oil": 0, "oil-gas": 0, "CO2-brine": 1, "CO2-brine cold": 1}
TITLES = {"water-oil": "Water into oil", "oil-gas": "Gas into black oil", "CO2-brine": "CO2 into brine",
          "CO2-brine cold": "Cold CO2 into brine"}


def parse_cols(spec: str) -> list[tuple[int, int]]:
    """``'10,20;9,19'`` -> [(10, 20), (9, 19)] (column i, j)."""
    return [tuple(int(v) for v in s.split(",")) for s in spec.split(";") if s.strip()]


#: open (perforated) layers of the injectors and producers, k = 0 at the top; None: all layers
LAYERS = {"injector": None, "producer": None}


def parse_layers(spec: str):
    """``'8-11'`` -> [8, 9, 10, 11]; ``''`` -> None (all layers)."""
    if not spec:
        return None
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def well_list(bm, inj, prod, rate, bhp):
    nx, ny, nz = bm["dims"]
    act = bm["active"]

    def col(i, j, kind):
        cells = [i + nx * (j + ny * k) for k in (LAYERS[kind] or range(nz))]
        if not all(act[c] for c in cells):
            raise ValueError(f"well column ({i}, {j}) is not active")
        return cells

    w = [{"name": f"INJ{n + 1}" if len(inj) > 1 else "INJ", "kind": "injector", "rate": rate / len(inj),
          "cells": col(*ij, "injector"), "radius": 0.1} for n, ij in enumerate(inj)]
    w += [{"name": f"PROD{n + 1}" if len(prod) > 1 else "PROD", "kind": "producer", "bhp": bhp, "cells": col(*ij, "producer"),
           "radius": 0.1} for n, ij in enumerate(prod)]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.0025, help="total injection rate (m3/s, split over injectors)")
    ap.add_argument("--years", type=int, default=100, help="run length (years, reports every 2 years)")
    ap.add_argument("--inj", default="5,10", help="injector columns 'i,j;i,j'")
    ap.add_argument("--prod", default="9,16", help="producer columns 'i,j;i,j'")
    ap.add_argument("--inj-layers", default="8-11", help="injector open layers 'k0-k1' (0 = top); empty = all")
    ap.add_argument("--prod-layers", default="1-3", help="producer open layers 'k0-k1' (0 = top); empty = all")
    ap.add_argument("--cases", default="water-oil,oil-gas,CO2-brine")
    ap.add_argument("--case-rates", default="oil-gas=0.006,CO2-brine=0.02", help="per-case rates overriding --rate, e.g. 'oil-gas=0.06,CO2-brine=0.1'")
    ap.add_argument("--case-bhp", default="oil-gas=19.5", help="per-case producer BHP in MPa, e.g. 'oil-gas=19.5'")
    ap.add_argument("--linear-solver", default="iterative")
    ap.add_argument("--out-dir", default="kansas_out")
    ap.add_argument("--gif-name", default="fluids_compare_kansas.gif")
    ap.add_argument("--cmap", default="viridis_bright", help="saturation colormap (any matplotlib name)")
    ap.add_argument("--background-cmap", default="porosity_grey",
                    help="colormap of the faint base (default: light greys, darker = lower permeability or porosity)")
    ap.add_argument("--no-background", action="store_true", help="plain grey shell instead of the grey base")
    ap.add_argument("--background", default="permeability", choices=["permeability", "porosity"],
                    help="property of the grey base: permeability (log scale, default) or porosity")
    ap.add_argument("--render-only", action="store_true", help="redraw from the npz of an earlier run")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inj, prod = parse_cols(a.inj), parse_cols(a.prod)
    LAYERS.update(injector=parse_layers(a.inj_layers), producer=parse_layers(a.prod_layers))
    bm = kansas_structural()
    nx, ny, nz = bm["dims"]

    def setup(years=40):
        model = nat.FlowModel.from_corner_point(bm["grid"], bm["perm"], bm["phi"])
        times = [0.0] + [y * fc.YR for y in range(2, years + 1, 2)]
        return bm, model, bm["phi"].size, times

    if a.render_only:
        d = np.load(out / "kansas_fluids.npz")
        names = sorted({k.split("|")[0] for k in d.files}, key=list(TITLES).index)
        render(bm, {n: {"saturation": d[f"{n}|saturation"]} for n in names}, inj, prod, a, out)
        return
    fc._setup = setup
    case_rate = {k: float(v) for k, v in (kv.split("=") for kv in a.case_rates.split(",") if kv)}
    case_bhp = {k: float(v) * 1e6 for k, v in (kv.split("=") for kv in a.case_bhp.split(",") if kv)}
    runs, stats = {}, {}
    act = bm["active"]
    for name in a.cases.split(","):
        rate = case_rate.get(name, a.rate)
        pb = case_bhp.get(name)
        fc._wells = lambda bm_, q, bhp, rate=rate, pb=pb: well_list(bm_, inj, prod, rate, bhp if pb is None else pb)
        t = time.time()
        try:
            r = fc.CASES[name](a.years, a.linear_solver)
        except Exception as e:  # report and continue with the next case
            stats[name] = {"error": str(e), "seconds": time.time() - t}
            print(name, "FAILED", e, flush=True)
            continue
        dt = time.time() - t
        bhp = np.asarray(r["bhp"])
        p = np.asarray(r["p"])[:, act]
        s = np.asarray(r["saturation"])[:, act, INJECTED[name]]
        stats[name] = {"seconds": round(dt, 1), "rate": rate,
                       "inj_bhp_max_MPa": round(float(bhp[:, :len(inj)].max() / 1e6), 2),
                       "prod_bhp_MPa": round(float(bhp[-1, len(inj)] / 1e6), 2),
                       "max_cell_p_MPa": round(float(p.max() / 1e6), 2),
                       "cells_sat_rise_gt_0.05_end": int((s[-1] - s[0] > 0.05).sum()),
                       "max_injected_sat_at_producer_end": round(float(np.asarray(r["saturation"])[-1][well_list(
                           bm, inj, prod, a.rate, 0)[-1]["cells"], INJECTED[name]].max()), 3),
                       **{k: int(r[k]) for k in ("n_steps", "n_cuts", "n_newton", "n_linear_iterations",
                                                  "n_linear_fallbacks")}}
        mip, net = np.asarray(r["mass_in_place"]), np.asarray(r["mass_net_in"])
        stats[name]["mass_balance_rel"] = float(np.abs((mip[-1] - mip[0]) - (net[-1] - net[0])).max()
                                                / np.abs(mip[0]).sum())
        runs[name] = r
        print(name, json.dumps(stats[name]), flush=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    a.rates = {k: v["rate"] for k, v in stats.items() if "rate" in v}
    np.savez_compressed(out / "kansas_fluids.npz", **{f"{k}|{q}": np.asarray(r[q]) for k, r in runs.items()
                                                       for q in ("p", "saturation", "bhp", "times")})
    if runs:
        render(bm, runs, inj, prod, a, out)


def wells_for_plot(bm, grid, inj, prod):
    from subsurfix.viz3d import Well

    xs, ys, nz, dz = bm["xs"], bm["ys"], bm["dims"][2], bm["cell"][2]
    wl = []
    for names, cols, color, kind in ((("INJ",), inj, "#1f4e9c", "injector"), (("PROD",), prod, "#b2182b", "producer")):
        ks = LAYERS[kind] or list(range(nz))
        for n, (i, j) in enumerate(cols):
            x, y = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
            top = grid.surface_depth(x, y)
            nm = names[0] + (str(n + 1) if len(cols) > 1 else "")
            if LAYERS[kind]:
                nm += f" L{min(ks) + 1}-{max(ks) + 1}"
            wl.append(Well(nm, x, y, top, top + nz * dz, color,
                           {"open": (top + min(ks) * dz, top + (max(ks) + 1) * dz)}))
    return wl


def render(bm, runs, inj, prod, a, out):
    from subsurfix.viz3d import LayeredGrid, animate_layered

    nx, ny, nz = bm["dims"]
    grid = LayeredGrid(bm["xs"], bm["ys"], bm["top_nodes"], nz, bm["cell"][2], bm["active"])
    wl = wells_for_plot(bm, grid, inj, prod)
    rates = getattr(a, "rates", None) or json.loads((out / "stats.json").read_text())
    rates = {k: (v["rate"] if isinstance(v, dict) else v) for k, v in rates.items()}
    panels = [(f"{TITLES[k]}, {rates.get(k, a.rate):g} m³/s", np.asarray(r["saturation"])[1:, :, INJECTED[k]])
              for k, r in runs.items()]
    years = np.arange(2.0, a.years + 0.01, 2.0)
    n_act = int(bm["active"].sum())
    dist = 0.5 * np.hypot(inj[0][0] - prod[0][0], inj[0][1] - prod[0][1])
    gif = out / a.gif_name
    sec = section_wells(bm, inj[0], prod[0], wl[0].extra["open"], wl[len(inj)].extra["open"])
    animate_layered(grid, panels, gif, times=years, wells=wl, label="Injected-phase saturation", clim=(0.0, 0.8),
                    cmap=getattr(a, "cmap", "viridis_bright"), background_cmap=getattr(a, "background_cmap", "porosity_grey"),
                    bar_fmt="%.1f", plume=0.05, context_opacity=0.22, panel_size=(480, 400),
                    **background_args(bm, a),
                    still=gif.with_suffix(".png"), section=sec,
                    header=f"Mississippian interval, Kansas: fluid injection with FIM-OBL over {a.years} years",
                    footer=(f"Vertical exaggeration 40x   model 11 x 11 km, 60 m thick, 12 layers on the structural top, "
                            f"{n_act} active cells (south-east block without well control inactive)\n"
                            f"blue: injector, red: producer (fixed BHP), {dist:.1f} km apart, "
                            f"open layers in the labels (L1 = top)   "
                            f"\ngrey = " + BG_NOTE[getattr(a, "background", "permeability")] + "; colour = saturation up by > 0.05, more opaque = larger rise"))
    print(gif, f"{gif.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
