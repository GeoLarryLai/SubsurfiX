"""Generate examples/12_multiphase_fluids.ipynb (run: python examples/_build_nb12.py)."""

from pathlib import Path
import hashlib

import nbformat as nbf

HERE = Path(__file__).resolve().parent

cells = [
"""md:# 12 · Multiphase fluids with FVM

> The fluid comparison uses one FVM solver, FIM-OBL (open-DARTS inspired), with different fluids. MODFLOW 6 is another FVM-family solver using control-volume finite differences; it is neither FIM-OBL nor FEM. FEM here is the continuous Galerkin finite-element solver (OpenGeoSys-style), with the physics limitations documented below.

> Terminology note: the finite-volume method (FVM) is the method label; the FIM-OBL solver uses the fully implicit method with operator-based linearization.

Four fully implicit FVM runs on the same small layered block model used by the 3D doublet animation in the
README (`block_model.py` next to this notebook; no external data):

1. dead-oil water injection (water–oil),
2. gas injection into a black oil while the producer depletes the field (oil–gas, three components),
3. CO2 injection into brine (CO2–brine, isothermal, 50 °C, 1 mol/kg NaCl),
4. cold CO2 injection, 25 °C CO2 into a 60 °C aquifer (CO2–brine with the energy equation).

The FVM solver is the FIM-OBL solver: fully implicit method with operator-based linearization
(Voskov 2017, *J. Comput. Phys.* 337; Khait & Voskov 2017,
*J. Pet. Sci. Eng.* 157). It uses the two-point flux approximation, Newton on all unknowns, and Peaceman wells
(Peaceman 1983, *SPE J.* 23(3)). CO2 solubility is from Spycher & Pruess 2005 (*Geochim. Cosmochim. Acta* 69(13)).
Method notes: `docs/multiphase_fim.md`. The run cells compute results with the compiled solver; the optional whole-model comparison is off by default.

**Setup, common to all four cases**

- Grid: a 10 × 10 × 12 block (1200 cells of 500 × 500 × 5 m) in four units of three layers, each unit with one
  porosity and permeability (kv/kh = 0.1). The units, pore volume and median permeability are printed in the
  setup tables below.
- Gravity on (9.80665 m/s²).
- Wells: injector INJ at column (i, j) = (5, 5) and producer PROD at (2, 2), both open in all 12 layers
  (Peaceman well index, radius 0.1 m).
- Injection: 0.008 m³/s at reservoir conditions for 40 years; results reported every 2 years.
- FVM solver: fully implicit Newton with operator-based linearization. **Experimental**: not validated against
  open-DARTS on this grid.""",
"""md:## Setup""",
"""import os, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from subsurfix import _native as nat
from subsurfix.viz3d import RegularGrid, Well, animate_grid
OUT = Path(os.environ.get("SX_EXAMPLE_OUT", "outputs")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 80, "savefig.dpi": 110})
def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")""",
"""md:## Block model
Built in code by `layered_block()`; injector INJ at column (5, 5) and producer PROD at (2, 2), both open in all
12 layers. Everything after this cell only uses `model`, `phi`, `k_md`, `cell` and the well helper, so another
property model can be swapped in here.""",
"""from IPython.display import Markdown, display
from block_model import column_wells, describe, layered_block
bm = layered_block()
NX, NY, NZ = bm["dims"]; cell = bm["cell"]; origin = bm["origin"]
phi, k_md = bm["phi"], bm["k_md"]
INJ, PROD = (5, 5), (2, 2)
N = phi.size
model = nat.FlowModel.from_regular(list(origin), list(cell), [NX, NY, NZ], bm["perm"], phi)
display(Markdown(describe(bm)))
PV = model.pore_volume()
def wells(q_inj, bhp_prod):
    return column_wells(bm, INJ, PROD, rate=q_inj, bhp=bhp_prod)
YR = 3.15576e7
TIMES = demo_report_times(DEMO_YEARS, DEMO_REPORT_INTERVAL_YEARS)
DEMO_REPORT_YEARS = np.array(TIMES) / YR
G = 9.80665
print(f"{N} cells, cell {cell[0]:.0f} x {cell[1]:.0f} x {cell[2]:.1f} m, pore volume {PV.sum():.3e} m3")
print(f"median k {np.median(k_md):.1f} mD (range {k_md.min():.2f}-{k_md.max():.0f}), porosity {phi.min():.3f}-{phi.max():.3f}")
display(Markdown(
    "| Setup | Value (from this run) |\\n|---|---|\\n"
    f"| Cells | {NX} × {NY} × {NZ} = {N} |\\n"
    f"| Cell size | {cell[0]:.0f} × {cell[1]:.0f} × {cell[2]:.1f} m |\\n"
    f"| Pore volume | {PV.sum():.3e} m³ |\\n"
    f"| Median permeability | {np.median(k_md):.1f} mD |\\n"
    f"| Injection | {DEMO_RATE_M3_S:g} m³/s for {DEMO_REPORT_YEARS[-1]:.0f} years = {DEMO_RATE_M3_S * TIMES[-1] / PV.sum() * 100:.1f} % of the pore volume |\\n"
    f"| Report times | every {DEMO_REPORT_YEARS[1]:.0f} years |"))""",
"""md:Injection rates are reservoir volume rates. `DEMO_RATE_M3_S` defaults to 0.008 m³/s
(about 690 m³/day), with `DEMO_YEARS = 40` and reports every two years for both small models.
The setup table computes the injected pore-volume fraction from these editable settings.
Isothermal CO₂/brine uses `DEMO_P_RANGE_PA` / `DEMO_TABLE_POINTS` (defaults 5–200 MPa,
261 × 201 points); the other fluid systems retain their separate tables. A table bound is
not a property-validity guarantee: inspect the reported pressure and fallback count.""",
"""md:## Case 1 · Dead-oil water injection
Water (1000 kg/m³, 1 mPa·s) displaces dead oil (850 kg/m³, 2 mPa·s), Corey relative permeability, gravity on.
Initial water saturation 0.2, pressure 20 MPa. Producer at 12 MPa bottom-hole pressure.

What to look for: a water bank spreading from INJ, faster in the high-permeability layers, with the injector
pressure rising until the injected volume is balanced by production.""",
"""runs, RUNTIME = {}, {}
t = time.time()
runs["water-oil"] = nat.run_fim_water_oil(
    model, wells(DEMO_RATE_M3_S, 1.2e7), np.full(N, 0.2), np.full(N, 2e7), TIMES,
    corey={"mu_o": 2e-3}, physics={"gravity": G, "rho_o": 850.0, "c_w": 4e-10, "c_o": 1e-9},
    p_range=(5e6, 5e7), table_points=(31, 201), dt_init=86400.0, dt_max=0.25 * YR)
RUNTIME["water-oil"] = time.time() - t
print(f"water-oil: {RUNTIME['water-oil']:.0f} s")""",
"""md:## Case 2 · Gas injection into a black oil
Three components (gas, oil, water) with dissolved gas (Rs), from a PVT table. Initially the oil is saturated
at 20 MPa (no free gas), water saturation 0.2. INJ injects gas, PROD produces at 10 MPa, below the bubble point,
so gas also comes out of solution near PROD.

What to look for: free gas near INJ (injected) and near PROD (exsolved as the pressure there drops below the
bubble point). The plume cell count later in the notebook includes both.""",
"""p_tab = 1e5 + 3e6 * np.arange(11); f = (p_tab - 1e5) / 3e7
pvt = np.column_stack([p_tab, 1 + 150 * f, 1.05 + 0.35 * f, 2e-3 - 1.2e-3 * f, 1 / (1 + 300 * f), 1.2e-5 + 1.3e-5 * f])
# columns: p (Pa), Rs (sm3/sm3), Bo, mu_o (Pa s), Bg, mu_g (Pa s)
p0 = 2e7
rs = np.interp(p0, pvt[:, 0], pvt[:, 1]); bo = np.interp(p0, pvt[:, 0], pvt[:, 2])
rho_sc = (0.9, 850.0, 1020.0)  # default surface densities (gas, oil, water)
rs_mass = rs * rho_sc[0] / rho_sc[1]
oil_mass = 0.8 * (rho_sc[1] + rs * rho_sc[0]) / bo             # live oil mass per pore volume
water_mass = 0.2 * rho_sc[2] / 1.01 * (1 + 4e-10 * (p0 - 1e5))
zo0 = oil_mass / (1 + rs_mass) / (oil_mass + water_mass)
zg0 = 0.999 * zo0 * rs_mass                                    # just below saturation: no free gas at t = 0
t = time.time()
runs["oil-gas"] = nat.run_fim_black_oil(
    model, wells(DEMO_RATE_M3_S, 1.0e7), np.full(N, p0), np.full(N, zg0), np.full(N, zo0), TIMES,
    pvt=pvt, bw=1.01, gravity=G, injection=[(1.0, 0.0), (0.0, 0.0)],
    p_range=(5e6, 3e7), table_points=(51, 101, 101), dt_init=86400.0, dt_max=0.25 * YR)
RUNTIME["oil-gas"] = time.time() - t
print(f"oil-gas: {RUNTIME['oil-gas']:.0f} s")""",
"""md:## Case 3 · CO2 injection into brine (isothermal)
CO2–brine physics from the SubsurfiX CO2 module: Span & Wagner (1996) CO2 density, Fenghour et al. (1998) CO2
viscosity, brine density and viscosity, and Spycher & Pruess (2005) mutual solubility, at 50 °C with 1 mol/kg NaCl.
Initial pressure 20 MPa, no CO2. Producer at 12 MPa.

What to look for: a free CO2 plume rising under gravity towards the top layers, CO2 dissolving into the brine
at its edge, and the time when CO2 first reaches PROD.""",
"""t = time.time()
runs["CO2-brine"] = nat.run_fim_co2_brine(
    model, wells(DEMO_RATE_M3_S, 1.2e7), np.full(N, 2e7), np.zeros(N), TIMES,
    temperature=323.15, m_nacl=1.0, gravity=G, p_range=DEMO_P_RANGE_PA, table_points=DEMO_TABLE_POINTS,
    dt_init=86400.0, dt_max=0.25 * YR)
RUNTIME["CO2-brine"] = time.time() - t
print(f"CO2-brine: {RUNTIME['CO2-brine']:.0f} s, property fallbacks: {runs['CO2-brine']['n_fallbacks']}")""",
"""md:## Case 4 · Cold CO2 injection (energy equation on)
Same as case 3 with temperature as an unknown: aquifer at 60 °C, CO2 injected at 25 °C. Rock: 2650 kg/m³,
900 J/(kg·K), 2.5 W/(m·K). The energy equation includes advection, conduction and rock heat storage.

What to look for: a cooled zone around INJ that lags behind the CO2 front (rock heat storage), and the energy
balance row in the statistics below.""",
"""t = time.time()
runs["CO2-brine cold"] = nat.run_fim_co2_brine(
    model, wells(DEMO_RATE_M3_S, 1.2e7), np.full(N, 2e7), np.zeros(N), TIMES,
    m_nacl=1.0, thermal=True, t0=np.full(N, 333.15), injection_temperature=[298.15, 333.15],
    gravity=G, p_range=(5e6, 5e7), t_range=(290.0, 340.0), table_points=[31, 101, 26],
    dt_init=86400.0, dt_max=0.25 * YR)
RUNTIME["CO2-brine cold"] = time.time() - t
print(f"CO2-brine cold: {RUNTIME['CO2-brine cold']:.0f} s, property fallbacks: {runs['CO2-brine cold']['n_fallbacks']}")""",
"""md:## Solver statistics and mass balance
Mass balance error: (mass in place − initial mass − net mass in through wells) per component, divided by
the largest mass in place of that component (CO2 starts at zero), worst over all report times. For the thermal case the energy row is the same check on energy.""",
"""INJECTED = {"water-oil": (0, "Sw"), "oil-gas": (0, "Sg"), "CO2-brine": (1, "Sg"), "CO2-brine cold": (1, "Sg")}
print(f"{'case':16s} {'steps':>5s} {'cuts':>4s} {'newton':>6s} {'table pts':>9s}  component balance errors")
bal = {}
def comp_names(r, n):
    c = list(r["component_names"])
    return (c + ["energy"])[:n] if len(c) < n else c[:n]
for name, r in runs.items():
    m = np.asarray(r["mass_in_place"]); net = np.asarray(r["mass_net_in"])
    err = (m - m[0] - (net - net[0])) / np.abs(m).max(0)
    bal[name] = err
    comps = comp_names(r, err.shape[1])
    s = ", ".join(f"{c} {np.abs(err[:, i]).max():.1e}" for i, c in enumerate(comps))
    print(f"{name:16s} {r['n_steps']:5d} {r['n_cuts']:4d} {r['n_newton']:6d} {r['n_table_points']:9d}  {s}")""",
"""fig, ax = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)
for a, (name, err) in zip(ax, bal.items()):
    comps = comp_names(runs[name], err.shape[1])
    for i, c in enumerate(comps):
        a.semilogy(DEMO_REPORT_YEARS[1:], np.abs(err[1:, i]) + 1e-18, "o-", ms=3, label=c)
    a.set_title(name); a.set_xlabel("years"); a.legend(fontsize=8)
ax[0].set_ylabel("relative balance error")
save(fig, "fim12_mass_balance.png")""",
"""md:## Saturation maps
Injected-phase saturation after 40 years: top layer (map view) and the vertical section through INJ (row j = 5).""",
"""fig, ax = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw={"hspace": 0.4})
ext_xy = [0, NX * cell[0] / 1e3, 0, NY * cell[1] / 1e3]
ext_xz = [0, NX * cell[0] / 1e3, NZ * cell[2], 0]
for c, (name, r) in enumerate(runs.items()):
    ph, lab = INJECTED[name]
    s = np.asarray(r["saturation"])[-1, :, ph].reshape(NZ, NY, NX)
    im = ax[0, c].imshow(s[0], origin="lower", extent=ext_xy, vmin=0, vmax=0.8, cmap="viridis")
    ax[0, c].plot([(INJ[0] + 0.5) * cell[0] / 1e3, (PROD[0] + 0.5) * cell[0] / 1e3],
                  [(INJ[1] + 0.5) * cell[1] / 1e3, (PROD[1] + 0.5) * cell[1] / 1e3], "w^", ms=7)
    ax[0, c].set_title(f"{name}: {lab}, top layer"); ax[0, c].set_xlabel("x (km)")
    ax[1, c].imshow(s[:, INJ[1], :], extent=ext_xz, vmin=0, vmax=0.8, cmap="viridis", aspect="auto")
    ax[1, c].set_title(f"{lab}, section j = {INJ[1]}"); ax[1, c].set_xlabel("x (km)")
ax[0, 0].set_ylabel("y (km)"); ax[1, 0].set_ylabel("depth (m)")
fig.colorbar(im, ax=ax, shrink=0.6, label="saturation")
save(fig, "fim12_saturation_maps.png")""",
"""fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
Tc = np.asarray(runs["CO2-brine cold"]["temperature"])[-1].reshape(NZ, NY, NX) - 273.15
im = ax[0].imshow(Tc[0], origin="lower", extent=ext_xy, cmap="coolwarm"); fig.colorbar(im, ax=ax[0], label="°C")
ax[0].set_title(f"cold CO2: temperature, top layer, {DEMO_YEARS:g} yr"); ax[0].set_xlabel("x (km)"); ax[0].set_ylabel("y (km)")
im = ax[1].imshow(Tc[:, INJ[1], :], extent=ext_xz, cmap="coolwarm", aspect="auto"); fig.colorbar(im, ax=ax[1], label="°C")
ax[1].set_title(f"temperature, section j = {INJ[1]}"); ax[1].set_xlabel("x (km)"); ax[1].set_ylabel("depth (m)")
print(f"cold CO2 case: minimum temperature {Tc.min():.2f} °C after {DEMO_YEARS:g} years")
save(fig, "fim12_temperature.png")""",
"""md:## Time series
Injected-phase pore volume in place (saturation × pore volume, summed), and well rates (kg/s per component;
positive is into the reservoir).""",
"""fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
for name, r in runs.items():
    ph, lab = INJECTED[name]
    s = np.asarray(r["saturation"])[:, :, ph]
    base = s[0]
    ax[0].plot(DEMO_REPORT_YEARS, ((s - base) * PV).sum(1) / PV.sum(), "o-", ms=3, label=f"{name} ({lab})")
    bhp = np.asarray(r["bhp"])
    ax[1].plot(DEMO_REPORT_YEARS, bhp[:, 0] / 1e6, "o-", ms=3, label=name)
ax[0].set_ylabel("injected-phase volume / PV"); ax[1].set_ylabel("INJ bottom-hole pressure (MPa)")
for a in ax[:2]:
    a.set_xlabel("years"); a.legend(fontsize=8)
r = runs["CO2-brine"]; q = np.asarray(r["well_mass_rate"])
for i, c in enumerate(r["component_names"]):
    ax[2].plot(DEMO_REPORT_YEARS[1:], -q[1:, 1, i], "o-", ms=3, label=f"PROD {c}")
ax[2].set_title("CO2-brine: producer mass rates"); ax[2].set_ylabel("kg/s out"); ax[2].set_xlabel("years"); ax[2].legend(fontsize=8)
save(fig, "fim12_time_series.png")""",
"""fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
for a, (name, r) in zip(ax, runs.items()):
    q = np.asarray(r["well_mass_rate"])
    for w, wn in enumerate(("INJ", "PROD")):
        for i, c in enumerate(r["component_names"]):
            if np.abs(q[1:, w, i]).max() > 1e-6:
                a.plot(DEMO_REPORT_YEARS[1:], q[1:, w, i], "o-" if w == 0 else "s--", ms=3, label=f"{wn} {c}")
    a.axhline(0, color="k", lw=0.5); a.set_title(name); a.set_xlabel("years"); a.legend(fontsize=7)
ax[0].set_ylabel("well mass rate (kg/s, + in)")
save(fig, "fim12_well_rates.png")""",
"""md:## Side-by-side comparison
Same grid, wells and reservoir injection rate. The plume extent is the number of cells with injected-phase
saturation above 0.05 (above the initial value for water).""",
"""fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for name, r in runs.items():
    ph, lab = INJECTED[name]
    s = np.asarray(r["saturation"])[:, :, ph]
    ax[0].plot(DEMO_REPORT_YEARS, ((s - s[0]) > 0.05).sum(1), "o-", ms=3, label=name)
    p = np.asarray(r["p"])
    ax[1].plot(DEMO_REPORT_YEARS, (p * PV).sum(1) / PV.sum() / 1e6, "o-", ms=3, label=name)
ax[0].set_ylabel("plume cells (ΔS > 0.05)"); ax[1].set_ylabel("mean pressure (MPa)")
for a in ax:
    a.set_xlabel("years"); a.legend(fontsize=8)
save(fig, "fim12_comparison.png")
for name, r in runs.items():
    ph, lab = INJECTED[name]
    s = np.asarray(r["saturation"])[:, :, ph]
    print(f"{name:16s} max {lab} {s[-1].max():.3f}, plume cells {int(((s[-1] - s[0]) > 0.05).sum())}, "
          f"INJ BHP {np.asarray(r['bhp'])[-1, 0] / 1e6:.2f} MPa")""",
"""md:## 3D animations
One GIF per fluid system with the same grid, camera, colour scale and wells (vertical exaggeration 25).
Written to `SX_EXAMPLE_OUT`. The README copies in `image/` come from the Kansas section at the end of this notebook.""",
"""grid = RegularGrid(origin, cell, (NX, NY, NZ))
wl = [Well(nm, origin[0] + (ij[0] + 0.5) * cell[0], origin[1] + (ij[1] + 0.5) * cell[1], 0.0, NZ * cell[2], col)
      for nm, ij, col in (("INJ", INJ, "#1f4e9c"), ("PROD", PROD, "#b2182b"))]
for name, fn, title in (("water-oil", "fim_water_oil.gif", "Water injection into dead oil"),
                        ("oil-gas", "fim_oil_gas.gif", "Gas injection into black oil"),
                        ("CO2-brine", "fim_co2_brine.gif", "CO2 injection into brine")):
    ph, lab = INJECTED[name]
    frames = np.asarray(runs[name]["saturation"])[:, :, ph]
    animate_grid(grid, frames, OUT / fn, times=DEMO_REPORT_YEARS, wells=wl, label=lab, units="", clim=(0.0, 0.8),
                 cmap="viridis", z_exaggeration=25.0, title=title, time_label="{:.0f} years",
                 fps=5, width=720, height=500, hold_last=5, bar_fmt="%.1f",
                 rotate=DEMO_ROTATE, rotate_deg=DEMO_ROTATE_DEG)
    print(fn, f"{(OUT / fn).stat().st_size / 1e6:.2f} MB")""",
"""md:## Results summary
The table below is built by the next cell from the run results above (nothing is typed in by hand). Balance
error is the worst relative mass (and, for the cold case, energy) balance error over all report times.
CO2 breakthrough is the first report time at which PROD produces CO2 at more than 1 % of the injected CO2
mass rate.""",
"""def breakthrough(r, frac=0.01):
    q = np.asarray(r["well_mass_rate"]); ic = list(r["component_names"]).index("CO2")
    hit = np.nonzero(-q[1:, 1, ic] > frac * q[1:, 0, ic])[0]
    return f"{DEMO_REPORT_YEARS[1:][hit[0]]:.0f} yr" if hit.size else f"not by {DEMO_YEARS:g} yr"
rows = [f"| Case | Run time | Steps / cuts / Newton | Worst balance error | Max injected saturation | INJ pressure at {DEMO_YEARS:g} yr | CO2 breakthrough | Min temperature |",
        "|---|---|---|---|---|---|---|---|"]
for name, r in runs.items():
    ph, lab = INJECTED[name]
    s = np.asarray(r["saturation"])[:, :, ph]
    bt = breakthrough(r) if "CO2" in name else "–"
    tmin = f"{np.asarray(r['temperature']).min() - 273.15:.2f} °C" if name == "CO2-brine cold" else "–"
    rows.append(f"| {name} | {RUNTIME[name]:.0f} s | {r['n_steps']} / {r['n_cuts']} / {r['n_newton']} | "
                f"{np.abs(bal[name]).max():.1e} | {lab} {s[-1].max():.2f} | {np.asarray(r['bhp'])[-1, 0] / 1e6:.2f} MPa | "
                f"{bt} | {tmin} |")
display(Markdown("\\n".join(rows)))""",
"""md:## Iterative versus direct linear solver
Each Newton step of the fully implicit solver solves one large sparse linear system. By default this is done
iteratively: GMRES with a two-stage CPR preconditioner (an algebraic multigrid solve on the cell pressures, then
an ILU(0) sweep on all unknowns). The alternative, `linear_solver="direct"`, factorizes the whole matrix with a
sparse LU. If GMRES ever fails to converge, that Newton step falls back to the direct LU automatically and the
run reports it in `n_linear_fallbacks`.

Both linear solvers solve the same equations to a tight tolerance (relative residual 1e-8), so the answers should agree
to round-off and the time-step history should be identical. The cell below re-runs the four cases with the direct solver, in parallel
processes (`subsurfix.parallel.run_cases`, using the same case setup stored in `fluid_cases.py`), and compares
them with the iterative runs above.""",
"""from fluid_cases import CASES
from subsurfix.parallel import run_cases

direct = run_cases({k: (f, dict(linear_solver="direct", years=DEMO_YEARS,
                               rate=DEMO_RATE_M3_S, report_times_s=TIMES,
                               **(dict(p_range=DEMO_P_RANGE_PA, table_points=DEMO_TABLE_POINTS)
                                  if k=="CO2-brine" else {}))) for k, f in CASES.items()}, parallel=True)
rows = ["| Case | Iterative | Direct | GMRES iterations | Fallbacks | Same steps / Newton | Max rel. Δp | Max Δ saturation |",
        "|---|---|---|---|---|---|---|---|"]
for k in CASES:
    a, b = runs[k], direct.results[k]
    pa, pb = np.asarray(a["p"]), np.asarray(b["p"])
    dp = np.abs(pa - pb).max() / np.abs(pb).max()
    ds = np.abs(np.asarray(a["saturation"]) - np.asarray(b["saturation"])).max()
    same = (a["n_steps"], a["n_newton"]) == (b["n_steps"], b["n_newton"])
    rows.append(f"| {k} | {RUNTIME[k]:.1f} s | {direct.seconds[k]:.1f} s | {a['n_linear_iterations']} | "
                f"{a['n_linear_fallbacks']} | {'yes' if same else 'no'} | {dp:.1e} | {ds:.1e} |")
display(Markdown("\\n".join(rows)))
print(f"direct runs: {direct.processes} processes, wall {direct.wall_seconds:.1f} s "
      f"(sum of case times {sum(direct.seconds.values()):.1f} s)")""",
"""md:How to read the table: the differences are at round-off level (around 1e-11 or smaller), so the
two linear solvers give the same answer. The iterative solver is the faster of the two here, and the gap grows with the
size of the problem and the number of unknowns per cell. The last line shows the gain from running the four
cases at once in separate processes instead of one after another.""",
"""md:## Limitations
- The small-demo defaults use 0.008 m³/s for 40 years. Rates, duration and the isothermal CO₂ table are
  editable in the demo settings cell. The 5–200 MPa numerical table does not extend the 60 MPa
  Spycher & Pruess (2005) flash validity limit; fallback results are not a validated solution.
- The black-oil PVT table is synthetic (same shape as the unit tests), not a measured fluid.
- CO2–brine uses a salt-free mass basis (NaCl is fixed in the brine), with no salt precipitation and constant
  Corey relative permeability without hysteresis.
- The oil–gas plume cell count includes gas that comes out of solution near PROD, not only injected gas.
- The cold-CO2 case is shown as temperature maps only; there is no GIF for it.
- Experimental solver: not validated as equivalent to open-DARTS.""",
]

SNAP = """RES = {"Layered block": {"runs": runs, "RUNTIME": RUNTIME, "bal": bal, "PV": PV.sum(), "k_md": k_md, "phi": phi}}
print("block-model results kept for the comparison at the end of the notebook")"""


def kansas(cell: str) -> str:
    """Same code as the block section, on the Kansas window (separate output names)."""
    out = (cell.replace("from block_model import column_wells, describe, layered_block",
                        "from block_model import column_wells, describe\nfrom kansas_window import kansas_window")
               .replace("bm = layered_block()", "bm = kansas_window(i0=7, j0=7, nx=10, ny=10)")
               .replace('"fim12_mass_balance.png"', '"fim12_mass_balance_kansas.png"')
               .replace('"fim12_saturation_maps.png"', '"fim12_saturation_maps_kansas.png"')
               .replace('OUT / fn', 'OUT / ("kansas_" + fn)')
               .replace('title=title', 'title=title + " (Kansas KGS window)"'))
    return out


KANSAS = [
SNAP,
"""md:# Real-world case: Kansas (KGS) property window

The same four runs on real geology: a 10 × 10 × 12 window (columns i = 7–16, j = 7–16, all 12 layers, 1200
cells of 500 × 500 × 5 m) of the Kansas Geological Survey (KGS) property model built in notebooks 01–04. The
two small files it reads are committed in `examples/data/kansas_window/` (see the README there), so no download
is needed.

- Porosity from the notebook 04 property model, clipped to 0.03–0.30.
- Permeability from porosity as in notebook 05: k = 10^(−1 + 14 φ) mD, doubled in clean-facies cells, kv/kh = 0.1.
  Unlike the block model, porosity and permeability vary from cell to cell (printed below).
- Same wells, fluids, initial states, injection rate (0.008 m³/s), report times and solver settings as the block
  section. The code cells below are the block-model cells with only the property source (and output file names)
  swapped.
- The three GIFs written here (`kansas_fim_*.gif`) are the animations shown in the repository README.""",
"md:## Kansas window",
kansas(cells[4]),
"md:## Case 1 · Dead-oil water injection (Kansas)",
kansas(cells[7]),
"md:## Case 2 · Gas injection into a black oil (Kansas)",
kansas(cells[9]),
"md:## Case 3 · CO2 injection into brine, isothermal (Kansas)",
kansas(cells[11]),
"md:## Case 4 · Cold CO2 injection, energy equation on (Kansas)",
kansas(cells[13]),
"md:## Solver statistics and mass balance (Kansas)",
kansas(cells[15]),
kansas(cells[16]),
"md:## Saturation maps (Kansas)",
kansas(cells[18]),
"md:## 3D animations (Kansas)\nSame camera, colour scale and wells as the block GIFs.",
kansas(cells[26]),
"""md:## Whole-model Kansas animations: what you can change
The README GIFs of the whole Kansas model (`scripts/kansas_fluids.py` for the three fluid cases and
`scripts/make_fem_vs_fim_animation.py` for FVM versus FEM pressure and temperature) run on the structural grid from
`examples/kansas_window.py` (`kansas_structural`: 12 layers of 5 m on the mapped top, the south-east block
without wells inactive). Settings you can adjust on the command line:

- Well locations: `--inj "i,j;i,j"` and `--prod "i,j"` (several wells separated by `;`; the rate is split over
  the injectors). `--inj`/`--prod` columns must be active.
- Perforated (open) layers: `--inj-layers 8-11` and `--prod-layers 0-3` (0 = top layer); the MODFLOW 6 well
  connections use the same layers.
- Rates: `--rate` (m³/s) and `--case-rates "oil-gas=...,CO2-brine=..."`; producer pressure per case with
  `--case-bhp "oil-gas=19.5"` (MPa).
- Run length and frames: `--years` (one frame every 2 years); `--times "2:60:2"` for the methods comparison.
- Separate official MODFLOW 6 external check only (`make_methods_animation.py --external-check`), well-flow closure: `--mf6-tolerance normal` (or `tight`).
- Operator-table pressure range: `--p-max` (MPa) for the heat comparison. Isothermal CO₂/brine uses
  `p_range=(5e6, 2e8)` Pa and `table_points=[261, 201]`: 5–200 MPa with 0.75 MPa pressure spacing.
  Water/oil and thermal CO₂ retain their separate tables; oil/gas uses 5–30 MPa. Table coverage does
  **not** certify property-model validity or acceptable injection pressure.
- Colormaps (any matplotlib colormap name): `--cmap` (saturation, default `viridis_bright`: viridis without its dark end) in `kansas_fluids.py`;
  `--cmap-pressure` (default `viridis_bright`) and `--cmap-temperature` (default `cold`) in
  `make_methods_animation.py`; `--background-cmap` (faint porosity base, default `porosity_grey`: darker = tighter rock) in both,
  `--no-background` to turn the base off;
  `scripts/make_porosity_image.py --cmap` (porosity picture, default `copper`). In Python, `animate_layered`
  takes `cmap` (or a per-panel `cmap` option) and `background_cmap`, as names or `Colormap` objects.
- Look of the 3D panels: a faint greyscale porosity shell (`background`, `context_opacity`), with only cells whose
  value changed by more than the threshold (`plume`, or `plume_frac` as a fraction of the largest change) drawn in
  colour, more opaque for larger changes. Thresholds: `--p-above-background` (default 3 MPa above the median rise in each frame; `--p-threshold-abs` with `--p-threshold` for a fixed rise, `--p-threshold-frac` for a fraction of the peak), `--t-threshold`
  (default 2 °C).

### Kansas branching-fault CO₂ comparison controls
The existing Kansas structural model, not the geometry-preview box, is used by
`scripts/make_faulted_co2_kansas.py`. It keeps 4656 active cells (22×22×12 with the SE notch),
INJ (5,10) open layers 8–11 and PROD (9,16) layers 1–3 (zero-based), 0.008 m³/s,
20 MPa producer BHP (matching the 20 MPa initial reservoir pressure), and 100 years / reports every 2 years. FVM results use the FIM-OBL solver.

- `--cache-dir`: separate caches for this matched pair; `--save-npz` / `--from-npz` save or render
  combined results. A table/setup fingerprint rejects stale caches. Do not mix the old 5–50 MPa cache
  with a 5–200 MPa run.
- `--multiplier 1e-4`: the approved no-throw staircase barriers; `--years 100 --frames 50` fixes the
  approved schedule. Thread cap is eight. These are explicit controls, not permission to change the
  approved geometry, wells or rate for the featured comparison.
- Main dip 3.75° SW and branch dip 4.67° NE come from `scripts/kansas_branching_geometry.py`;
  the branch soles into the main. Plot overlays are smooth geometric planes, while equations use
  staircase face transmissibilities. No red north–south 56° plane and no FEM multiphase panels.
- Both cases use the **same** widened pressure table above, also used by the isothermal CO₂ call
  in `fluid_cases.py`. Widening is a numerical table change, not a new physical model.

**Validation history:** at 0.02 m³/s the widened-table faulted integration returned with 95,402
property fallbacks and was rejected. A fresh matched pair at **0.008 m³/s** passed with zero
fallbacks and component balance errors below 1e-5. The reason for lowering the shared rate was
the 60 MPa property-model limit, **not** absent producer breakthrough. Both cases have CO₂
at the producer by year 100; a missing faulted breakthrough would also be an acceptable result.
The executable settings below default to the chosen rate, with the heavy solve disabled.
""",
"""md:## Results summary (Kansas)
Built by the next cell from the Kansas runs above (nothing typed in by hand); same definitions as the block summary.""",
kansas(cells[28]),
"""md:## Block versus Kansas
One row per case and model, built from both sections' runs. Plume cells: cells with injected-phase saturation
above 0.05 (above the initial value for water) after 40 years.""",
"""RES["Kansas KGS window"] = {"runs": runs, "RUNTIME": RUNTIME, "bal": bal, "PV": PV.sum(), "k_md": k_md, "phi": phi}
rows = ["| Model | Pore volume (m³) | Median k (mD) |", "|---|---|---|"]
for mname, R in RES.items():
    rows.append(f"| {mname} | {R['PV']:.3e} | {np.median(R['k_md']):.1f} |")
display(Markdown("\\n".join(rows)))
rows = [f"| Case | Model | Run time | Steps / cuts | Worst balance error | Max injected saturation | Plume cells | INJ pressure at {DEMO_YEARS:g} yr | CO2 breakthrough |",
        "|---|---|---|---|---|---|---|---|---|"]
for case in RES["Layered block"]["runs"]:
    for mname, R in RES.items():
        r = R["runs"][case]; ph, lab = INJECTED[case]
        s = np.asarray(r["saturation"])[:, :, ph]
        bt = breakthrough(r) if "CO2" in case else "–"
        rows.append(f"| {case} | {mname} | {R['RUNTIME'][case]:.0f} s | {r['n_steps']} / {r['n_cuts']} | "
                    f"{np.abs(R['bal'][case]).max():.1e} | {lab} {s[-1].max():.2f} | {int(((s[-1] - s[0]) > 0.05).sum())} | "
                    f"{np.asarray(r['bhp'])[-1, 0] / 1e6:.2f} MPa | {bt} |")
display(Markdown("\\n".join(rows)))""",
"""md:What differs: in the block model each unit is uniform, so the injected phase spreads as a layered front
mostly through the 80 mD main sand. In the Kansas window permeability varies from cell to cell and is highest
near the top, so the plumes are irregular and follow the connected high-permeability cells. The same limitations
listed above apply to both sections.""",
]
cells = cells + KANSAS

DEMO_SETTINGS_MD = """md:## Editable settings · small notebook demos
Run this cell before the block and Kansas-window cases. `DEMO_RATE_M3_S` is reservoir-volume
injection rate; `DEMO_YEARS` and the report interval control their shared schedule.
`DEMO_P_RANGE_PA` bounds the isothermal CO₂/brine interpolation table in pascals;
`DEMO_TABLE_POINTS` gives pressure/composition axis counts. The defaults preserve 0.75 MPa
pressure spacing. These imported defaults are shared with `fluid_cases.py`; the linear-solver
comparison below passes your edits explicitly, rather than silently using module defaults.
The water/oil, black-oil and thermal cases retain their own constitutive tables.
`DEMO_ROTATE` switches the small-demo GIF camera sweep on/off; `DEMO_ROTATE_DEG` is the
total sweep in degrees (40, the previous animation default). These visual controls are unrelated to
`DEMO_RATE_M3_S`.

**Physical limit:** the Spycher & Pruess (2005) CO₂/brine flash is valid only to **60 MPa
(600 bar)**. A 200 MPa table is numerical coverage, not physical certification. The original
5–50 MPa table was too tight for the faulted whole-model injector; widening it produced
95,402 property fallbacks and that run was rejected. Zero producer CO₂ is not a rejection
criterion. The optional whole-model comparison has separate settings and is off by default."""
DEMO_SETTINGS_CODE = """from fluid_cases import (DEMO_RATE_M3_S as _DEFAULT_DEMO_RATE,
    DEMO_YEARS as _DEFAULT_DEMO_YEARS, DEMO_REPORT_INTERVAL_YEARS as _DEFAULT_DEMO_INTERVAL,
    DEMO_P_RANGE_PA as _DEFAULT_DEMO_P_RANGE, DEMO_TABLE_POINTS as _DEFAULT_DEMO_TABLE,
    demo_report_times)
DEMO_RATE_M3_S = _DEFAULT_DEMO_RATE          # 0.008 m³/s: both small models, all four fluids
DEMO_YEARS = _DEFAULT_DEMO_YEARS            # 40 years
DEMO_REPORT_INTERVAL_YEARS = _DEFAULT_DEMO_INTERVAL  # 2 years
DEMO_P_RANGE_PA = _DEFAULT_DEMO_P_RANGE     # (5e6, 2e8): isothermal CO₂/brine only
DEMO_TABLE_POINTS = list(_DEFAULT_DEMO_TABLE)       # [261, 201]
DEMO_ROTATE = True                          # camera sweep on/off for small-demo GIFs
DEMO_ROTATE_DEG = 40.0                      # total degrees over each demo clip (previous default)
"""
FAULT_SETTINGS_MD = """md:## Editable settings · whole Kansas branching-fault comparison
This is the existing 22×22×12 Kansas structural model with its inactive SE notch, not the
10×10 window above. The following named variables are passed to the **same runner API**
as `scripts/make_faulted_co2_kansas.py`; geometry is imported, not duplicated.
Both cases share the rate, pressure table and report schedule. `FAULT_MULTIPLIER` scales
cross-plane face transmissibilities without throw; smooth plot planes remain distinct from
those staircase faces. Well columns and open layers are zero-based; `N_THREADS` is capped at eight.
`ROTATE` turns the 3D camera sweep on/off, and `ROTATE_DEG` sets its total degrees over the clip.
`VIEW_DIR` is the starting direction from the model toward the camera (x east, y north, z up),
chosen a bit more from the SSE than the previous ESE view; `ROTATE_DEG = 35` is sized so the clip
still ends near the previous almost-east angle. These are visual controls, not injection-rate
settings: `RATE_M3_S` remains the shared reservoir-volume rate.

The earlier 0.02 m³/s faulted run was rejected after 95,402 property fallbacks, not because a
plume missed the producer. The chosen lower shared rate is **0.008 m³/s**, and the producer is
held at **20 MPa**, the same as the initial reservoir pressure. The featured pair passed the
fallback and mass-balance checks, with peak reported injector BHP 27.60 MPa / 52.31 MPa
(no faults / two faults), below the 60 MPa flash limit. Table coverage remains 5–200 MPa with 261×201 points.
The 52.31 MPa faulted BHP still exceeds an illustrative lithostatic estimate (~25.46 MPa, about 2.05×);
no fracture opening or geomechanical safety is modeled. **Do not enable the solve casually:** `RUN_FAULTED_COMPARISON = False`
means this section performs no model setup, native solve, timing or rendering. If explicitly
enabled later, the runner rejects fallback values, poor mass balance and injector BHP ≥60 MPa.
It does not reject absent producer breakthrough. No FEM multiphase panels are available."""
FAULT_SETTINGS_CODE = """RUN_FAULTED_COMPARISON = False
RATE_M3_S = 0.008
P_RANGE_PA = (5e6, 2e8)
TABLE_POINTS = [261, 201]
YEARS = 100
REPORT_INTERVAL_YEARS = 2
FAULT_MULTIPLIER = 1e-4
INJ = (5, 10)
INJ_LAYERS = tuple(range(8, 12))
PROD = (9, 16)
PROD_LAYERS = tuple(range(1, 4))
PROD_BHP_PA = 20e6
N_THREADS = 8
ROTATE = True        # camera sweep on/off
ROTATE_DEG = 35      # total degrees over the clip; ends near the previous almost-east view
VIEW_DIR = (0.88, -0.60, 0.75)  # starting model-to-camera direction (x east, y north, z up)
COMPARISON_CACHE = OUT / "kansas_branching_faults"
"""
FAULT_RUN_CODE = """if RUN_FAULTED_COMPARISON:
    import sys
    _examples = Path.cwd() if Path("fluid_cases.py").is_file() else Path.cwd() / "examples"
    _root = _examples.resolve().parent
    sys.path.insert(0, str(_root / "scripts"))
    from make_faulted_co2_kansas import run_comparison
    _frames = int(YEARS / REPORT_INTERVAL_YEARS)
    if _frames < 1 or _frames * REPORT_INTERVAL_YEARS != YEARS:
        raise ValueError("Use a run length divisible by the report interval")
    comparison_result = run_comparison(cache_dir=COMPARISON_CACHE,
        rate=RATE_M3_S, p_range=P_RANGE_PA, table_points=TABLE_POINTS,
        years=YEARS, frames=_frames, fault_multiplier=FAULT_MULTIPLIER,
        inj=INJ, prod=PROD, inj_layers=INJ_LAYERS, prod_layers=PROD_LAYERS,
        prod_bhp=PROD_BHP_PA, n_threads=N_THREADS, render_output=True,
        rotate=ROTATE, rotate_deg=ROTATE_DEG, view_dir=VIEW_DIR,
        gif=OUT / "faulted_co2_kansas.gif", still=OUT / "faulted_co2_kansas.png")
else:
    print("Kansas matched-pair solve is OFF. Edit settings and explicitly opt in only after reviewing validity.")
"""
cells[3:3] = [DEMO_SETTINGS_MD, DEMO_SETTINGS_CODE]
cells += [FAULT_SETTINGS_MD, FAULT_SETTINGS_CODE, FAULT_RUN_CODE]


n = nbf.v4.new_notebook()
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip()) for c in cells]
for i, c in enumerate(n.cells):
    c.id = f"nb12-{i:03d}-" + hashlib.sha256(c.source.encode()).hexdigest()[:8]
nbf.write(n, HERE / "12_multiphase_fluids.ipynb")
