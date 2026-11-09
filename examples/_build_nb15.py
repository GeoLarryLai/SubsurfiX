# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Generate examples/15_fem_vs_fim_heat.ipynb (run: python examples/_build_nb15.py, then nbconvert --execute)."""
from pathlib import Path

import nbformat as nbf
from _build_notebooks import SETUP, nb  # importing does not rewrite 01-08

HERE = Path(__file__).resolve().parent

cells = [
"""md:# 15 · FEM vs FVM: heat doublet

> The in-package alternatives are an open-DARTS-inspired FIM-OBL solver and a MODFLOW-inspired control-volume finite-difference solver translated into Rust. Neither is the official USGS MODFLOW 6 program. FEM supports experimental coupled water pressure and temperature; full multiphase runs remain FVM-only. Explicit `mf6`/FloPy runs are separate official-program external checks.


> FVM and FEM are the spatial methods compared here. The FVM results use the SubsurfiX FIM-OBL solver (open-DARTS inspired); the FEM results use one continuous Galerkin finite-element solver (OpenGeoSys-style). MODFLOW 6 is a separate USGS FVM-family solver using control-volume finite differences, not FEM or FIM-OBL; it is not run in this notebook.

SubsurfiX's default FVM solver (the FIM-OBL solver: fully implicit method with operator-based linearization) (`run_fim_thermal_water`), solves water flow and heat on cell-centred
finite volumes: one pressure and one temperature per cell. This notebook runs the new **experimental**
continuous Galerkin finite-element solver (FEM), `run_fem_thermal_water`, written in Rust after the HT process (water flow + heat) of
OpenGeoSys ([Kolditz et al. 2012](https://doi.org/10.1007/s12665-012-1546-x)). Both solvers get the same grid,
water (IAPWS-IF97), rock and wells, and the notebook compares their results.

How the finite-element solver differs:

- **Unknowns at the cell corners.** Each cell is a trilinear hexahedron (an 8-corner brick); pressure and
  temperature live on its corners, where FVM keeps them at the cell centres. The cell values compared here are
  the mean of each cell's 8 corners.
- **Wells.** A completed cell passes its well flow to its 8 corners (same Peaceman well index as FVM), so
  near-well pressure and the bottom-hole pressure (BHP) differ a little.
- **Heat front.** Plain finite elements wiggle at a sharp cold front, so the solver adds streamline-upwind
  conduction along the flow; `stabilization=1` (the default) behaves like the first-order upwinding of FVM.
- **Nonlinear solve.** Picard iteration (flow coefficients refreshed after each linear solve); FVM uses Newton.

The solver is experimental and does not support faulted grids (it rejects them). Method notes: `docs/fem_ht.md`.
3D GIF of the Kansas case: `scripts/make_fem_vs_fim_animation.py`.""",
"md:## Setup",
SETUP + "import time\nfrom IPython.display import Markdown, display\nfrom block_model import cid, column_wells, describe, layered_block\nfrom subsurfix import _native as nat",
"""md:## The block model
The layered block of notebooks 09, 10 and 12 (`block_model.py` next to this notebook, no external data):
10 × 10 × 12 cells of 500 × 500 × 5 m in four units of three layers, kv/kh = 0.1. Injector INJ at column (5, 5)
puts 0.02 m³/s of 30 °C water into the 80 °C reservoir; producer PROD at (2, 2) is held at 20 MPa; both are open
in all layers. 30 years, reported every year: the case of `scripts/make_mf6_vs_fim_heat_gif.py`.""",
"""bm = layered_block()
nx, ny, nz = bm["dims"]
N = nx * ny * nz
perm, phi, kmd = bm["perm"], bm["phi"], bm["k_md"]
act = np.ones(N, dtype=bool)
model = nat.FlowModel.from_regular(bm["origin"], bm["cell"], bm["dims"], perm, phi)
INJ, PROD = (5, 5), (2, 2)
wells = column_wells(bm, INJ, PROD, rate=0.02, bhp=2e7)
YR = 3.156e7
years = np.arange(1.0, 30.01, 1.0)
times = [float(y * YR) for y in years]
DT_MAX = 0.05 * YR
display(Markdown(describe(bm)))
print(N, "cells,", int(act.sum()), "active | median k", round(float(np.median(kmd[act])), 1), "mD")""",
"""md:## Run both solvers
Same model, wells, start (20 MPa, 80 °C), rock (2650 kg/m³, 900 J/kg/K, 2.5 W/m/K) and time steps (first step
1 day, longest `DT_MAX`). FVM also needs the pressure and temperature range of its property table.""",
"""ROCK = {"rho": 2650.0, "cp": 900.0, "conductivity": 2.5}
p0, t0 = np.full(N, 2e7), np.full(N, 353.15)
shared = dict(injection_temperature=[303.15, 353.15], rock=ROCK, dt_init=86400.0, dt_max=DT_MAX)
t = time.perf_counter()
fim = nat.run_fim_thermal_water(model, wells, p0, t0, times, p_range=(5e6, 5e7), t_range=(290.0, 360.0),
                                table_points=(41, 71), **shared)
t_fim = time.perf_counter() - t
t = time.perf_counter()
fem = nat.run_fem_thermal_water(model, wells, p0, t0, times, **shared)
t_fem = time.perf_counter() - t
print(f"FVM {t_fim:.1f} s: {fim['n_steps']} steps, {fim['n_cuts']} cuts, {fim['n_newton']} Newton, "
      f"{fim['n_linear_iterations']} GMRES iterations, {fim['n_linear_fallbacks']} fallbacks")
print(f"FEM     {t_fem:.1f} s: {fem['n_steps']} steps, {fem['n_cuts']} cuts, {fem['n_iterations']} Picard, "
      f"{fem['n_linear_iterations']} GMRES iterations, {fem['n_linear_fallbacks']} fallbacks")""",
"""md:## Compare
Temperature in °C, pressure in MPa, over active cells. The balance error is
|change in place − net inflow| / |net inflow| between the first and the last report, for water (kg) and energy (J).""",
"""def frames(r):
    P, T = np.asarray(r["p"]) / 1e6, np.asarray(r["temperature"]) - 273.15
    if P.shape[0] == len(times) + 1:
        P, T = P[1:], T[1:]
    return np.where(act, P, np.nan), np.where(act, T, np.nan)


def balance(r):
    mip, net = np.asarray(r["mass_in_place"]), np.asarray(r["mass_net_in"])
    return np.abs((mip[-1] - mip[0]) - (net[-1] - net[0])) / np.abs(net[-1] - net[0])


(P_fim, T_fim), (P_fem, T_fem) = frames(fim), frames(fem)
bal_fim, bal_fem = balance(fim), balance(fem)
bhp_fim, bhp_fem = np.asarray(fim["bhp"])[-1] / 1e6, np.asarray(fem["bhp"])[-1] / 1e6
pc = wells[1]["cells"]
tp_fim, tp_fem = T_fim[:, pc].mean(axis=1), T_fem[:, pc].mean(axis=1)
dT = np.abs(T_fem - T_fim)[:, act]
dp = np.abs(P_fem[-1] - P_fim[-1])[act]
print(f"max |dT| any active cell, any report: {dT.max():.2f} degC; mean |dT|: {dT.mean():.3f} degC")
print(f"max |dp| at {years[-1]:g} yr: {dp.max():.3f} MPa")
for lab, b in (("FVM", bal_fim), ("FEM", bal_fem)):
    print(f"{lab:8s} balance error: water {b[0]:.1e}, energy {b[1]:.1e}")
for lab, b in (("FVM", bhp_fim), ("FEM", bhp_fem)):
    print(f"{lab:8s} BHP at {years[-1]:g} yr: producer {b[1]:.3f} MPa, injector {b[0]:.3f} MPa")
pd.DataFrame({"year": years, "FVM": tp_fim, "FEM": tp_fem}).set_index("year").iloc[4::5].round(2)""",
"""fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
axs[0].plot(years, tp_fim, "k-", lw=2, label="FVM"); axs[0].plot(years, tp_fem, "r--", label="FEM")
axs[0].set_xlabel("years"); axs[0].set_ylabel("producer temperature, layer mean (degC)"); axs[0].legend(fontsize=8)
origin, cell = bm["origin"], bm["cell"]
ext = [origin[0], origin[0] + nx * cell[0], origin[1], origin[1] + ny * cell[1]]
lay = int(np.argmax(kmd.reshape(nz, -1).mean(axis=1)))
for ax, T, lab in zip(axs[1:], (T_fim, T_fem), ("FVM", "FEM")):
    im = ax.imshow(T[-1].reshape(nz, ny, nx)[lay], origin="lower", extent=ext, vmin=30, vmax=80, cmap="coolwarm")
    for (i, j), mk in ((INJ, "kv"), (PROD, "k^")):
        ax.plot(origin[0] + (i + 0.5) * cell[0], origin[1] + (j + 0.5) * cell[1], mk)
    ax.set_title(f"{lab}, {years[-1]:g} yr, layer {lay} (most permeable)", fontsize=9)
fig.colorbar(im, ax=axs[1:], shrink=0.85, label="degC"); save(fig, "15_block_fem_vs_fim.png")""",
"""md:## Export to ParaView / OpenGeoSys
The FEM temperature at the last report as cell data on an OpenGeoSys `.vtu` mesh. Open it in ParaView or use it
as the mesh of an OpenGeoSys project.""",
"""info = model.export_ogs(OUT / "15_block_fem.vtu", cell_data={"temperature_fem_C": T_fem[-1]})
print(OUT / "15_block_fem.vtu", info)""",
]

SNAP = """def summary(label):
    return {"Model": label, "Active cells": f"{int(act.sum())} of {N}", "Simulated (yr)": f"{years[-1]:g}",
            "Run time FVM / FEM (s)": f"{t_fim:.1f} / {t_fem:.1f}", "Max ΔT (°C)": f"{dT.max():.2f}",
            "Producer T at end, FVM / FEM (°C)": f"{tp_fim[-1]:.2f} / {tp_fem[-1]:.2f}",
            "Water balance error FVM / FEM": f"{bal_fim[0]:.1e} / {bal_fem[0]:.1e}",
            "Energy balance error FVM / FEM": f"{bal_fim[1]:.1e} / {bal_fem[1]:.1e}"}
RESULTS = {"block": summary("Layered block")}
print("block-model results kept for the comparison at the end of the notebook")"""


def kansas(cell: str) -> str:
    """Same code as the block section, on the Kansas structural model (separate output names)."""
    return cell.replace('"15_block_fem_vs_fim.png"', '"15_kansas_fem_vs_fim.png"')


KANSAS = [
SNAP,
"""md:# Real-world case: the Kansas (KGS) model on its structural top

The doublet of the README FVM-versus-FEM water pressure/temperature comparison (`scripts/make_fem_vs_fim_animation.py`) on real geology: the whole
22 × 22 × 12 Kansas Geological Survey (KGS) property model of notebooks 01–04 (500 × 500 × 5 m cells, 11 × 11 km
× 60 m) with its layers following the structural top; the south-east block without well control is inactive. The
grid has no faults, so the finite-element solver accepts it.

- Injector INJ at column (6, 8), open in layers 0–2 (0 = top), puts 0.05 m³/s of 30 °C water into the 80 °C
  reservoir; producer PROD at (8, 13), open in layers 1–3, is held at 20 MPa, 2.7 km away.
- 200 years, reported every 5 years, longest time step 0.25 years; same rock as the block.

This section takes minutes. The GIF version of the same comparison is `scripts/make_fem_vs_fim_animation.py`.""",
"md:## Kansas structural model",
"""from kansas_window import kansas_structural
bm = kansas_structural()
nx, ny, nz = bm["dims"]
N = nx * ny * nz
perm, phi, kmd = bm["perm"], bm["phi"], bm["k_md"]
act = np.asarray(bm["active"], bool)
model = nat.FlowModel.from_corner_point(bm["grid"], perm, phi)
INJ, PROD = (6, 8), (8, 13)
wells = [{"name": "INJ", "kind": "injector", "rate": 0.05, "cells": [cid(bm, *INJ, k) for k in range(0, 3)], "radius": 0.1},
         {"name": "PROD", "kind": "producer", "bhp": 2e7, "cells": [cid(bm, *PROD, k) for k in range(1, 4)], "radius": 0.1}]
YR = 3.15576e7
years = np.arange(5.0, 200.01, 5.0)
times = [float(y * YR) for y in years]
DT_MAX = 0.25 * YR
display(Markdown(describe(bm)))
print(N, "cells,", int(act.sum()), "active | median k", round(float(np.median(kmd[act])), 1), "mD")""",
"md:## Run both solvers",
kansas(cells[6]),
"md:## Compare",
kansas(cells[8]),
kansas(cells[9]),
"""md:## Block versus Kansas
Built from the runs in this notebook (nothing typed in by hand). Max ΔT is the largest |T_FEM − T_FIM| over active
cells and report times; the balance errors are water and energy between the first and the last report.""",
"""RESULTS["kansas"] = summary("Kansas KGS structural model")
keys = list(RESULTS["block"])
rows = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
rows += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in RESULTS.values()]
display(Markdown("\\n".join(rows)))""",
]

nbf.write(nb(cells + KANSAS), HERE / "15_fem_vs_fim_heat.ipynb")
