# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Generate examples/10_modflow_heat.ipynb (run: python examples/_build_nb10.py, then nbconvert --execute)."""
from pathlib import Path

import nbformat as nbf
from _build_notebooks import SETUP, nb  # importing does not rewrite 01-08

HERE = Path(__file__).resolve().parent
cells = [
"""md:# 10 · Heat doublet in MODFLOW 6 (GWE), checked against SubsurfiX

The doublet of the README animation (30 °C water injected into an 80 °C reservoir for 30 years, on a small
layered block model) run in
SubsurfiX (`run_heat`) and again in MODFLOW 6 with its groundwater-energy model, GWE
([Langevin et al. 2024](https://doi.org/10.5066/P9FL1JCC)), through FloPy
([Bakker et al. 2016](https://doi.org/10.1111/gwat.12413)). `subsurfix.modflow_heat` (experimental) converts the
grid, permeability, wells and thermal properties. Both wells are open in every layer: MODFLOW 6 first solves the
flow with multi-aquifer wells (MAW) using the same Peaceman indices as SubsurfiX, and its per-layer well rates then
drive the GWE heat run. Needs the `mf6` program
(6.5 or newer): `get-modflow :`. Thermal properties are illustrative (notebook 06 values).""",
"md:## Setup", SETUP + "from IPython.display import Markdown, display\nfrom block_model import column_wells, describe, layered_block",
"""md:## The block model
A self-contained layered block (`block_model.py` next to this notebook, no external data): 10 × 10 × 12 cells of
500 × 500 × 5 m in four units of three layers, kv/kh = 0.1. Injector INJ at column (5, 5) (0.1 m³/s) and producer
PROD at (2, 2) (20 MPa), both open in all layers, so MODFLOW 6 needs its multi-layer well path.""",
"""from subsurfix import _native as nat
from subsurfix.modflow_heat import HeatDoublet, run_gwf_gwe, find_mf6
assert find_mf6(), "install mf6 with: get-modflow :flopy"
bm = layered_block()
origin, cell, dims, phi, perm, kmd = bm["origin"], bm["cell"], bm["dims"], bm["phi"], bm["perm"], bm["k_md"]
nx, ny, nz = dims
N = nx * ny * nz
INJ, PROD = (5, 5), (2, 2)
wells = column_wells(bm, INJ, PROD, rate=0.1, bhp=2e7)
yr = 3.156e7
years = np.arange(1, 31)
times = [float(y * yr) for y in years]
mu = 5e-4
case = HeatDoublet(origin, cell, dims, perm, phi, np.full(N, 80.0), wells, [30.0, 0.0], times, viscosity=mu, dt_max=0.05 * yr)
display(Markdown(describe(bm)))
print(N, "cells | median k", round(float(np.median(kmd)), 1), "mD")""",
"""md:## Run both
MODFLOW 6 runs steady flow and transient GWE; SubsurfiX runs `run_heat` on the same case.""",
"""import time
t = time.time(); mf = run_gwf_gwe(case, OUT / "mf6_heat"); t_mf = time.time() - t
dbl = nat.FlowModel.from_regular(list(origin), list(cell), list(dims), perm, phi)
t = time.time()
sx = dbl.run_heat(np.full(N, 1 / mu), wells, [30.0, 0.0], np.full(N, 80.0), times, props=case.props, dt_max=0.05 * yr)
t_sx = time.time() - t
T_sx, T_mf = np.asarray(sx["temperature"]), mf["temperature"]
assert np.allclose(mf["times"], times)
pc = wells[1]["cells"]
tp_sx, tp_mf = T_sx[:, pc].mean(axis=1), T_mf[:, pc].mean(axis=1)
print(f"run time: MODFLOW 6 {t_mf:.1f} s, SubsurfiX {t_sx:.1f} s")
print(f"max |dT| anywhere, any year: {np.abs(T_sx - T_mf).max():.2e} degC")
print(f"mean |dT| over cells and years: {np.abs(T_sx - T_mf).mean():.2e} degC")
print(f"max |dT| producer (layer mean): {np.abs(tp_sx - tp_mf).max():.2e} degC")
pd.DataFrame({"year": years, "SubsurfiX": tp_sx, "MODFLOW 6": tp_mf}).set_index("year").iloc[[0, 4, 9, 14, 19, 24, 29]].round(3)""",
"""fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
axs[0].plot(years, tp_sx, "k-", lw=2, label="SubsurfiX run_heat"); axs[0].plot(years, tp_mf, "r--", label="MODFLOW 6 GWE")
axs[0].set_xlabel("years"); axs[0].set_ylabel("producer temperature, layer mean (degC)"); axs[0].legend(fontsize=8)
ext = [origin[0], origin[0] + nx * cell[0], origin[1], origin[1] + ny * cell[1]]
lay = int(np.argmax(kmd.reshape(nz, -1).mean(axis=1)))
for ax, T, lab in zip(axs[1:], (T_sx, T_mf), ("SubsurfiX", "MODFLOW 6")):
    im = ax.imshow(T[14].reshape(nz, ny, nx)[lay], origin="lower", extent=ext, vmin=30, vmax=80, cmap="coolwarm")
    ax.contour(T_sx[14].reshape(nz, ny, nx)[lay], [55], colors="k", linewidths=0.8, extent=ext)
    for (i, j), mk in ((INJ, "kv"), (PROD, "k^")):
        ax.plot(origin[0] + (i + 0.5) * cell[0], origin[1] + (j + 0.5) * cell[1], mk)
    ax.set_title(f"{lab}, 15 yr, layer {lay} (black: SubsurfiX 55 degC)", fontsize=9)
fig.colorbar(im, ax=axs[1:], shrink=0.85, label="degC"); save(fig, "10_modflow_heat.png")""",
"""md:## Agreement
Both codes use a cell-centred finite-volume grid with two-point flow, upwind advection and implicit time steps,
and here the same Peaceman well indices, so they solve nearly the same discrete equations; the printed differences
are the check. Remaining differences come from the well models (SubsurfiX's BHP well versus MODFLOW's multi-aquifer
well), solver tolerances and time-step control. GWE dispersion is off to match `run_heat`.""",
]

SNAP = """def summary(label):
    bal = np.abs(np.asarray(sx["energy_change"]) - np.asarray(sx["energy_in"])) / np.abs(np.asarray(sx["energy_in"]))
    return {"Model": label, "Cells": f"{nx} × {ny} × {nz} = {N}", "Median k (mD)": f"{np.median(kmd):.1f}",
            "k range (mD)": f"{kmd.min():.2f}–{kmd.max():.1f}",
            "Producer T at 30 yr, SubsurfiX (°C)": f"{tp_sx[-1]:.3f}", "Producer T at 30 yr, MODFLOW 6 (°C)": f"{tp_mf[-1]:.3f}",
            "Max ΔT anywhere (°C)": f"{np.abs(T_sx - T_mf).max():.2e}", "SubsurfiX energy balance error": f"{bal.max():.1e}",
            "Run time SubsurfiX / MODFLOW 6 (s)": f"{t_sx:.2f} / {t_mf:.2f}"}
RESULTS = {"block": summary("Layered block")}
print("block-model results kept for the comparison at the end of the notebook")"""


def kansas(cell: str) -> str:
    """Same code as the block section, on the whole Kansas model (separate output names)."""
    out = (cell.replace("bm = layered_block()", "from kansas_window import kansas_model\nbm = kansas_model()")
               .replace("INJ, PROD = (5, 5), (2, 2)", "INJ, PROD = (13, 11), (9, 11)")
               .replace('OUT / "mf6_heat"', 'OUT / "mf6_heat_kansas"')
               .replace('"10_modflow_heat.png"', '"10_modflow_heat_kansas.png"'))
    return out


KANSAS = [
SNAP,
"""md:# Real-world case: the whole Kansas (KGS) property model

The same doublet on real geology: the whole 22 × 22 × 12 Kansas Geological Survey (KGS) property model built in
notebooks 01–04 (5808 cells of 500 × 500 × 5 m, 11 × 11 km × 60 m); the same model and wells as the README
animation. The two small files it reads are committed in `examples/data/kansas_window/` (see the README there).

- Porosity from the notebook 04 property model, clipped to 0.03–0.30.
- Permeability from porosity as in notebook 05: k = 10^(−1 + 14 φ) mD, doubled in clean-facies cells, kv/kh = 0.1.
- Wells in the middle of the model, 2 km apart (a typical doublet spacing): INJ at column (13, 11), 0.1 m³/s of
  30 °C water; PROD at (9, 11), 20 MPa. Same 80 °C start, thermal properties and 30 years. The cells below are the
  block-model cells with only the property source and well positions swapped.""",
"md:## Whole Kansas model",
kansas(cells[4]),
"md:## Run both",
kansas(cells[6]),
kansas(cells[7]),
"""md:## Kansas results and block-versus-Kansas comparison
Built from the runs in this notebook (nothing typed in by hand). The energy balance error is SubsurfiX's
|energy change − net energy in| / |net energy in|, worst over all report years.""",
"""RESULTS["kansas"] = summary("Kansas KGS whole model")
keys = list(RESULTS["block"])
rows = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
rows += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in RESULTS.values()]
display(Markdown("\\n".join(rows)))""",
"""md:What differs: in the block model the cold water follows the 80 mD main sand and the producer cools mainly
through that unit. In the Kansas model permeability varies from cell to cell, so the cold front is irregular and
the producer temperature reflects the connected high-permeability paths between the wells.""",
]

nbf.write(nb(cells + KANSAS), HERE / "10_modflow_heat.ipynb")
