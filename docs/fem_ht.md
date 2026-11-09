# Experimental finite-element water flow and heat (OpenGeoSys HT style)

The two in-package FVM solvers are the open-DARTS-inspired FIM-OBL solver and the
MODFLOW-inspired control-volume finite-difference solver (Rust translation). The latter is not the
official USGS MODFLOW 6 program. Where an example explicitly invokes `mf6` through FloPy, that is
an outside check with official USGS MODFLOW 6. FEM uses the OpenGeoSys-style continuous Galerkin
finite-element solver, including experimental coupled water pressure and temperature; complete
multiphase simulation remains FVM-only.


FVM and FEM name spatial methods. The FVM solver compared here is the SubsurfiX FIM-OBL solver
(open-DARTS inspired); FEM here uses one continuous Galerkin finite-element solver (OpenGeoSys-style).
MODFLOW 6 is a separate FVM-family solver using a control-volume finite-difference scheme, not FEM
and not FIM-OBL. The optional official-program external check compares two FVM-family solvers.
The featured README Kansas GIF now compares FVM and FEM pressure and temperature in a 2×2 layout.
The FVM results use the FIM-OBL solver; FEM uses the OpenGeoSys-style continuous Galerkin finite-element solver. This comparison shows water-injection pressure and temperature, not saturation.

Status: **experimental**, unreleased (see `CHANGELOG.md`). Not yet benchmarked against an OpenGeoSys run. The
fully implicit FVM solver (`run_fim_thermal_water`, FVM) is unchanged and remains the default.

Code: `crates/sx-core/src/flow/fem/ht.rs` (solver, wells), `crates/sx-core/src/flow/fem/hex8.rs` (shape
functions, Gauss points), `crates/sx-core/src/flow/fem/mesh.rs` (node mesh from the cell corners),
`crates/sx-py/src/fem.rs` (Python: `run_fem_thermal_water`). Rust entry point: `flow::fem::run_fem_ht`.

The continuous Galerkin finite-element solver (FEM) solves single-phase water flow with heat transport, after the OpenGeoSys HT (hydro-thermal) process
(Kolditz et al. 2012). It is written from the published equations; no OpenGeoSys code was copied (OpenGeoSys latest
release when accessed on 2026-11-01: 6.5.9). Water, rock and wells are the same as FVM, so the two solvers can
be run on the same `FlowModel` and compared.

## Equations

Unknowns are pressure p (Pa) and temperature T (K). Depth z is positive down.

```text
∂(φρ)/∂t + ∇·(ρ q) = Q_w
∂(φρU + (1 − φ) ρ_r c_r (T − T_ref))/∂t + ∇·(ρ h q) − ∇·(κ ∇T) = Q_E
q = −(k/μ)(∇p − ρ g ∇z),   κ = φ κ_f + (1 − φ) κ_r
```

* q is the Darcy flux (m/s), ρ, μ, h and U = h − p/ρ the water density, viscosity, enthalpy and internal energy,
  κ_f and κ_r the water and rock thermal conductivities, ρ_r c_r the rock heat capacity per volume.
* Porosity φ(p) = φ₀ (1 + c_rock (p − p_ref)).
* Q_w (kg/m³/s) and Q_E (W/m³) are the well terms.
* Water properties are IAPWS-IF97 from the same `ThermalWater` model as FVM.
* The outer boundary is closed: no flow and insulated.

## Discretization

* Trilinear 8-node hexahedra on the grid cell corners. Corners that coincide share a node (the merge rule of the
  OpenGeoSys `.vtu` export). Inactive cells are left out of the mesh.
* Continuous Galerkin weak form with 2 × 2 × 2 Gauss points. Water properties are evaluated at the Gauss points;
  permeability, porosity and conductivity are taken per element (cell).
* Storage is lumped to the nodes (the OpenGeoSys `mass_lumping` option). Each element spreads its pore volume and
  rock volume (bulk − pore) to its nodes in proportion to the integral of each shape function, ∫ N_a dV, so the
  total pore volume equals the `FlowModel`'s.

## Heat advection stabilization

Plain Galerkin oscillates near a sharp thermal front when the cell Péclet number (advection over conduction across
one cell) is large. Heat advection is therefore stabilized by streamline-upwind artificial conduction along the
Darcy flux (Brooks & Hughes 1982):

```text
K_su = s (ℓ/2) ρ c_p q qᵀ / |q|
```

with ℓ the element length along the flow and s the `stabilization` setting. s = 1 (default) is first-order
upwinding in 1D, as in the finite-volume solvers; s = 0 is plain Galerkin.

The extra conduction acts only along the flow. OpenGeoSys's isotropic diffusion was not used: with cells 500 m wide
and 5 m thick (the block and Kansas models) it would also spread heat across the flow and mix the layers vertically.

## Wells

* Same Peaceman (1978) well index and controls as FVM: rate (reservoir volume) or BHP control, rate wells with
  `bhp_limit` switch to BHP at the limit, no crossflow, wellbore density frozen over each step. One BHP unknown per
  well.
* A completed cell exchanges `WI β (p_wb − p_c)` with its 8 corner nodes, where p_c is the value at the cell centre
  (mean of the 8 nodes). The rate is split 1/8 per node.
* Because the well sees the cell-centre value of a node field, near-well pressure and BHP differ a little from
  FVM.

## Time stepping and nonlinear solve

* Backward Euler with the step control of FVM: Δt grows by 1.5 after a converged step and is halved after a
  failed one.
* Each step is solved by Picard iteration (`max_iter`, default 30). The residual is exact. The Jacobian keeps the
  flux coefficients (ρk/μ, h, κ, K_su) at the last iterate and adds the storage derivatives (forward differences)
  and the advective c_p ∂T term.
* Temperature changes are limited to 20 K per iteration (`max_dtemp`).
* Convergence (`tol`, default 1e-6) is on the normalized residual: water residual over nodal water mass, energy
  residual over nodal heat capacity (so in K).

## Linear solver

Same as FVM (`linear_solver`): by default CPR-preconditioned GMRES with a fall back to the direct LU, or
`linear_solver="direct"` for the LU at every step. Unknowns are `[p, T]` per node, then one BHP per well (the FVM
layout). Counters: `n_linear_solves`, `n_linear_iterations`, `n_linear_fallbacks`. See `docs/multiphase_fim.md`.

## Outputs

`run_fem_thermal_water` returns a dict with the FVM keys that apply:

* `times` (s); `p` (Pa) and `temperature` (K), shape (n_t, n_cells): the value at each cell centre (mean of its 8
  nodes). Inactive cells keep `p0` and `t0`.
* `bhp` (Pa); `well_mass_rate` (n_t, n_wells, 2): water (kg/s) and energy (W), positive into the reservoir.
* `mass_in_place` and `mass_net_in` (n_t, 2): water (kg) and energy (J, fluid + rock) in place, and cumulative
  injected minus produced, for balance checks.
* `phase_names`, `component_names`, `n_steps`, `n_cuts`, `n_fallbacks` and the linear counters.

Added by the finite-element solver: `p_nodes` and `temperature_nodes` (n_t, n_nodes), `nodes` (n_nodes, 3; x, y,
depth in m) and `n_iterations` (Picard iterations in total).

A non-zero `n_fallbacks` (water states outside the property model's range) raises a `RuntimeWarning`.

## Python usage

Same model, wells and settings for both solvers; compare temperature at the last output time.

```python
import numpy as np
from subsurfix import _native as nat

YR = 3.156e7
nx, ny, nz = 10, 10, 4
n = nx * ny * nz
perm = np.tile([1e-13, 1e-13, 1e-14], (n, 1))                    # m² (kx, ky, kz)
model = nat.FlowModel.from_regular([0.0, 0.0, 0.0], [100.0, 100.0, 5.0], [nx, ny, nz], perm, np.full(n, 0.2))


def column(i, j):
    return [i + nx * (j + ny * k) for k in range(nz)]


wells = [{"name": "INJ", "kind": "injector", "rate": 0.005, "cells": column(7, 7), "radius": 0.1},
         {"name": "PROD", "kind": "producer", "bhp": 2e7, "cells": column(2, 2), "radius": 0.1}]
p0, t0 = np.full(n, 2e7), np.full(n, 353.15)                     # Pa, K
times = [y * YR for y in (1.0, 5.0, 10.0)]
common = dict(injection_temperature=[303.15, 353.15], rock={"rho": 2650.0, "cp": 900.0, "conductivity": 2.5},
              dt_init=86400.0, dt_max=0.05 * YR)

fim = nat.run_fim_thermal_water(model, wells, p0, t0, times, p_range=(5e6, 5e7), t_range=(290.0, 360.0), **common)
fem = nat.run_fem_thermal_water(model, wells, p0, t0, times, stabilization=1.0, **common)

dT = np.asarray(fem["temperature"])[-1] - np.asarray(fim["temperature"])[-1]   # K per cell
print(f"largest |T_FEM - T_FVM| {np.abs(dT).max():.2f} K, mean {np.abs(dT).mean():.2f} K")
```

## Verification

Tests and comparison tools in this version (no results are recorded here yet):

* Rust unit tests in `crates/sx-core/src/flow/fem/`: shape functions, the gradient of a linear field, element volume
  (also against `CornerPointGrid::volume`), a patch test (a linear pressure field gives no flux at the interior node
  of a distorted 2 × 2 × 2 patch) and mesh node counts.
* Integration tests in `crates/sx-core/tests/fem_ht.rs`: steady 1D flow between a rate injector and a BHP producer
  (linear pressure with the Darcy slope), 1D conduction against the erfc solution, 1D
  advection–conduction against Ogata & Banks (1961) with the numerical dispersion of the upwind term added,
  mass and energy balance (relative error below 1e-6), and no temperature undershoot below the injection
  temperature with `stabilization=1`.
  Run with `cargo test -p sx-core`.
* FEM against FVM on a 10 × 10 × 4 block doublet (producer temperature curve, thermal breakthrough year, FEM
  balance): marked `#[ignore]` (slow). Run with `cargo test -p sx-core --test fem_ht -- --ignored --nocapture`.
  Its limits (5 K, 3 years) are provisional until the first run.
* `scripts/make_fem_vs_fim_animation.py`: FEM and FVM side by side on the Kansas doublet (default) or the block
  model (`--block`); `--check` compares the two runs; writes `image/fem_vs_fim_kansas.gif`.
* `examples/15_fem_vs_fim_heat.ipynb`: the same comparison in a notebook.

## Limits

* Closed outer boundary only (no flow, insulated); water and heat enter and leave through wells.
* Single-phase liquid water only.
* Faulted grids are rejected: nodes do not match across fault throws. Use `run_fim_thermal_water` for those.
* Near-well pressure and BHP differ a little from FVM (cell-centre well coupling, see Wells).
* Experimental; not yet benchmarked against an OpenGeoSys run.

## References

* Kolditz, O., et al. (2012). OpenGeoSys: an open-source initiative for numerical simulation of
  thermo-hydro-mechanical/chemical (THM/C) processes in porous media. *Environ. Earth Sci.* 67, 589–599.
  doi:10.1007/s12665-012-1546-x
* Brooks, A. N., Hughes, T. J. R. (1982). Streamline upwind/Petrov–Galerkin formulations for convection dominated
  flows with particular emphasis on the incompressible Navier–Stokes equations. *Comput. Methods Appl. Mech. Eng.*
  32, 199–259.
* Peaceman, D. W. (1978). Interpretation of well-block pressures in numerical reservoir simulation. *SPE J.* 18(3),
  183–194.
* Ogata, A., Banks, R. B. (1961). A solution of the differential equation of longitudinal dispersion in porous
  media. USGS Professional Paper 411-A.

## Bulk mesh input and thread controls

`FlowModel.from_bulk_mesh(nodes, cells, perm, poro, active=None)` accepts an explicit conforming
hexahedral bulk mesh for FVM and FEM. Coordinates are x/y/depth in metres, cells are int64 arrays
in corner order `di + 2*dj + 4*dk`, permeability is diagonal in world coordinates (m²).
Shared faces must share vertex IDs. The constructor rejects warped or non-permeability-orthogonal
faces rather than claiming that a two-point flux approximation works on arbitrary skew meshes.
The flat `dims=(n_cells,1,1)` is storage only: use explicit cell IDs for wells. Mesh conformance,
including absence of hanging faces or overlapping cells, remains the caller's responsibility.
The Peaceman approximation still uses cell extents; general rotated well geometry is not validated.

FEM rejects fault/pinch connections and any FVM link without a shared four-node face. These need
explicit interface treatment or a conforming remesh; they are not solved by relabelling the mesh.
FEM remains single-phase water with heat: no FEM oil/gas/CO2 multiphase solver is implemented.

FEM uses the existing Rust `set_num_threads` / `with_threads` solver pool. In Python,
`subsurfix._native.set_flow_threads(8)` sets the solver default. The automatic default also respects
`RAYON_NUM_THREADS`, including scoped pools. Independent element
contributions are assembled in parallel above the existing size threshold and scattered in fixed
element order, with no atomic numerical accumulation. Small meshes stay sequential; one-thread
runs remain available. Parallel scatter can differ from the earlier assembly by floating-point
round-off, not by governing equations. No speedup is claimed without a benchmark.

## Multiphase and interface assembly groundwork (not a runnable solver)

The experimental Rust modules `flow::fem::multiphase` and `flow::fem::interface` provide:

- Cached hexahedral quadrature for the isothermal continuous Galerkin component-balance residual,
  reusing the actual water–oil, black-oil and CO₂–brine constitutive operators, not FVM fluxes.
  Accumulation is consistent; capillary phase-pressure gradients and depth-positive-down gravity
  enter the weak flux term. Large batches use scoped parallel assembly in deterministic input order.
- Conservative, upstream-mobility transfer across an already paired planar, zero-thickness resistive
  face of arbitrary orientation. The interface hydraulic conductance is normal permeability divided
  by thickness; both face sides use distinct nodal unknowns. No fault storage or tangential flow is included.

These are **assembly kernels only**, not a multiphase FEM time integrator. The thermal-water FEM
routine is unchanged and must not be given multiphase fluid labels. Missing pieces are global
coupled nonlinear assembly/Jacobians, bounded transport stabilization, well/boundary coupling,
adaptive stepping and end-to-end component-conservation validation.

The face kernel does **not** cut the read-in model or pair its fault faces. Arbitrary plane intersections
still need conforming cut-element meshing, duplicated interface degrees of freedom and their global
coupling, including intersecting-fault topology. Existing fault/pinch/nonmatching-face rejections remain
in place. Bulk-mesh FVM still rejects non-orthogonal geometry instead of using an inconsistent two-point
flux approximation. Therefore neither the three-fluid FVM/FEM animation nor the two-fault 2×2 comparison
is enabled by these kernels. Unit element/interface balances are not substitutes for those demos.

## Kansas water-injection pressure and temperature comparison

`image/fem_vs_fim_kansas.gif` shows FVM pressure | FEM pressure above FVM temperature | FEM temperature.
The original Kansas FVM arrays were reused unchanged; only FEM was newly computed, with eight solver
threads. At 200 years the producer temperatures are 69.08 °C (FVM) and 65.36 °C (FEM); the maximum
producer difference is 3.72 °C, exceeding the optional default 2 °C comparison limit. FEM water and energy
balance errors are 1.0 × 10⁻⁷ and 8.3 × 10⁻⁹. This is a thermal-water comparison, not a multiphase
saturation result or an OpenGeoSys benchmark. The new FEM runtime is 322.7 s; the original FVM cache
records 107.3 s, which is not a same-machine speed comparison. Reuse both columns with the renderer's
`--from-npz`, or reuse only the original FVM with `--fvm-from-npz` when computing FEM.
