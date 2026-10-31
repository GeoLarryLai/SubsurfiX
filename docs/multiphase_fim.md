# Experimental fully implicit multiphase flow (OBL)

Status: **experimental** in 0.1.1. The existing IMPES path (`run_two_phase`,
`run_impes`) is unchanged and remains the supported solver.

Code: `crates/sx-core/src/flow/implicit.rs` (solver, OBL table, wells),
`crates/sx-core/src/flow/fim_physics.rs` (dead-oil and black-oil physics),
`crates/sx-py/src/fim.rs` (Python: `run_fim_water_oil`, `run_fim_black_oil`,
`fim_water_mass_fraction`).

## Method

* Cell-centred finite volumes, two-point flux approximation (TPFA), reusing
  the transmissibilities of `FlowModel`.
* Fully implicit Newton on the natural OBL state ω = [p, z_1 … z_{nc-1}]
  (overall **mass** fractions). Residual per cell and component:
  `PV0 (α_c(ω) − α_c(ωⁿ)) + Δt Σ_faces Σ_p T β_{c,p}(ω_up) ΔΦ_p + Δt q_c = 0`.
* Operator-based linearization (Voskov 2017; Khait & Voskov 2017):
  accumulation α_c = φ(p)/φ0 · Σ_p x_{c,p} ρ_p S_p and flux operators
  β_{c,p} = x_{c,p} ρ_p k_rp/μ_p are tabulated on a uniform multilinear grid in
  ω. Nodes are evaluated lazily (only hypercubes the run visits). Values and
  Jacobian entries come from the interpolant and its piecewise-constant
  derivatives.
* Phase-potential upwinding: ΔΦ_p = (p_a − Pc_a) − (p_b − Pc_b) − ρ̄_p g (z_a − z_b),
  ρ̄ the arithmetic mean of the two cells' phase densities (depth positive down).
* Peaceman wells (Peaceman 1978/1983) with one BHP unknown per well, rate
  control on total reservoir volume rate or BHP control, optional BHP limit
  that switches control and repeats the step. Wellbore head uses a mixture
  density frozen at the start of the step. Injectors use operators evaluated at
  the cell pressure and the injection composition.
* Time stepping: Newton with update chopping (max Δz) and damping; Δt grows by
  1.5 after convergence and is halved on failure.
* Sparse direct LU (faer).

## Physics in this release

* `DeadOil`: water–oil, immiscible, Corey kr and optional capillary pressure
  from the existing two-phase physics settings; slightly compressible phases
  and rock.
* `BlackOil`: gas/oil/water components, dissolved gas via Rs(p), Bo, Bg, μo, μg
  tables (mass-basis flash with standard-condition densities). Undersaturated
  oil uses Bo(p_b)(1 − c_o (p − p_b)). Relative permeability is Corey per
  phase, **not** Stone. No capillary pressure.
* Trait `OblPhysics` for new physics: implement `evaluate` (fills the operator
  vector described by `OpLayout`) and `saturations`. Physics with
  `thermal() == true` add temperature as the last unknown and energy as the
  last equation.
* `ThermalWater`: single-phase liquid water with IAPWS-IF97 properties and the
  energy equation.
* `Co2Brine`: CO2/H2O components, aqueous and gas phases, mass basis
  (salt-free; NaCl molality fixed). Each state is converted to a CO2 mole
  fraction and flashed with Spycher & Pruess (2005) (default) or Ziabakhsh-Ganji & Kooi (2012,
  *Int. J. Greenh. Gas Control* 11) (`solubility_model`). Corey kr per phase. Isothermal or thermal.
  A failed flash is retried with p, T clamped to the model's validity range,
  then constant properties are used; each fallback is counted
  (`n_fallbacks`) and Python raises a `RuntimeWarning` when it is non-zero.

## Energy equation

Accumulation: fluid internal energy per pore volume (α_E) plus rock energy
ρ_r c_r (T − T_ref) per rock volume (bulk − pore volume). Advection uses the
phase-upwinded β_E = h ρ kr/μ with the same phase potentials as the mass
fluxes. Conduction between cells uses (φ κ_f + (1 − φ) κ_r) times the
geometric factor of each half-cell, combined in series. Injector streams give
the injected composition and temperature. The energy residual is divided by
the cell heat capacity, so its Newton tolerance is in kelvin; temperature
updates are limited to `max_dtemp` (20 K) per iteration. The formulation
follows the OBL thermal form of Khait & Voskov (2018).

## Verification done (all from the compiled solver)

* Rust `crates/sx-core/tests/flow_fim.rs` and Python `tests/python/test_fim.py`.
* Buckley–Leverett, 200 cells, μo/μw = 2: L1 error vs the analytic profile
  0.011 (t_D = 0.2) and 0.013 (t_D = 0.4); front at x_D 0.475 vs 0.456 and
  0.935 vs 0.911; water mass gained equals mass injected to 1e-6 relative.
* Gravity column at rest: the pressure difference equals ρ̄ g Δz with ρ̄ the
  phase density at the local pressures to ~1.5e-6 relative. (An earlier ~2 %
  mismatch came from comparing against reference densities; at 20 MPa the
  compressible phases are ~2 % denser. It was not a solver error.)
* Thermal water, 100 cells: thermal front at 19.87 m and 49.93 m vs 20.00 m
  and 50.00 m from advective heat velocity; energy and mass balances to 1e-6.
* CO2–brine, 50 cells, 1 mol/kg NaCl, 50 °C: aqueous CO2 mass fraction in
  two-phase cells equals the CO2 module's flash (0.04072 SP2005, 0.03964 vs
  0.03963 ZGK2012); CO2 and H2O balances to 1e-6; no fallbacks.
* Cold CO2 injection (305 K into 340 K): CO2, H2O and energy balances hold.
* `examples/12_multiphase_fluids.ipynb`: water–oil, black-oil gas injection,
  CO2–brine and cold CO2–brine on a 1200-cell layered block model; relative
  balance errors ≤ 6e-8 for every component and energy.
* Black-oil depletion below the bubble point liberates gas (max Sg ≈ 0.23);
  per-component mass change equals well production to 1e-7 of total mass.

No comparison against open-DARTS or OGS output has been run yet.

## Compared with open-DARTS

Same: FV/TPFA, fully implicit, OBL operator form with multilinear
interpolation and table-derived Jacobian, phase-potential upwinding,
Peaceman wells with a BHP unknown.

Different (this prototype):

* Mass fractions, not mole fractions, as the state.
* Uniform table resolution chosen by the user; no adaptive/hierarchical
  storage. States outside the table bounds are **clamped silently** — choose
  `p_range` wide enough.
* Corey per phase for three-phase kr instead of Stone/Baker.
* No well crossflow, no multisegment wells, wellbore density frozen per step.
* No geomechanics, no diffusion/dispersion; black oil and water–oil are isothermal.
* Single-threaded assembly and direct LU (no CPR/AMG); slower on large grids.

## Compared with OGS `TwoPhaseFlowWithPP`

OGS solves the two-phase pressure–pressure (pc–pg) formulation with Galerkin
FEM and nodal unknowns; this solver uses cell-centred FV with overall-
composition unknowns, so results should agree in the converged limit but not
cell by cell. OGS has no OBL table and no Peaceman wells in that process.


## References

* Voskov, D. V. (2017). Operator-based linearization approach for modeling of
  multiphase multi-component flow in porous media. *J. Comput. Phys.* 337, 275–288.
* Khait, M., Voskov, D. V. (2017). Operator-based linearization for general
  purpose reservoir simulation. *J. Pet. Sci. Eng.* 157, 990–998.
* Peaceman, D. W. (1983). Interpretation of well-block pressures in numerical
  reservoir simulation with nonsquare grid blocks and anisotropic permeability.
  *SPE J.* 23(3), 531–543.
* Buckley, S. E., Leverett, M. C. (1942). Mechanism of fluid displacement in
  sands. *Trans. AIME* 146, 107–116.
* Khait, M., Voskov, D. (2018). Operator-based linearization for efficient
  modeling of geothermal processes. *Geothermics* 74, 7–18.
* Spycher, N., Pruess, K. (2005). CO2–H2O mixtures in the geological
  sequestration of CO2. II. *Geochim. Cosmochim. Acta* 69(13), 3309–3320.
* Aziz, K., Settari, A. (1979). *Petroleum Reservoir Simulation*. (black-oil model)
