# Faulted corner-point grids and CO₂ vertical equilibrium

Status: corner-point TPFA is part of the flow model. The CO₂ vertical-equilibrium (VE) model is **experimental**.
Example: `examples/13_cornerpoint_and_co2_ve.ipynb`.

## 1. Corner-point grids with faults

**Input.** `sx.read_cornerpoint_grdecl(path)` reads `SPECGRID`/`DIMENS`, `COORD`, `ZCORN`, `ACTNUM` and per-cell
arrays such as `PERMX/Y/Z`, `PORO`, `NTG` and `MULTX/Y/Z`. `sx.flow_model_from_grdecl(grid, props)` builds a
`FlowModel`, with permeability in mD by default. `sx.transmissibilities(model)` returns the connection list
`(a, b, T)`, and the model runs with any FIM driver (`run_fim_water_oil`, `run_fim_black_oil`, ...).

**Transmissibility.** TPFA half-transmissibilities are combined harmonically,
T_i = A (K_i n·d_i)/|d_i|², T = mult / (1/T_a + 1/T_b),
where A is the face area vector and d_i runs from the cell centre to the face centre. Permeability is diagonal in
the cell axes, and NTG scales the horizontal components. When the corners on the two sides of a face do not match
(a fault), the shared face is the overlap of the two quadrilaterals, computed in the plane of the pillars, so the
cells across the fault are connected by non-neighbour connections. Collapsed (pinched-out) cells are inactive.

**Validation:**

| Check | Result |
|---|---|
| Unfaulted corner-point vs Cartesian TPFA (20 connections) | max relative difference 3.3e-16 |
| 3 m throw, hand-computed overlap transmissibility | equal to 1e-12 |
| Faulted water–oil FIM, mass balance | 46 steps, 0 cuts, worst 5.3e-10; water crosses the fault (Sw 0.746) |
| GRDECL write/read round trip | identical connections |

**Not implemented:** `MINPV`/`PINCH` processing, `MAPAXES`, fault-rock (`FAULTS`/`MULTFLT`) keywords, and the exact
face-splitting of MRST `processGRDECL`. A fault is represented only by its geometric overlap.

## 2. CO₂ vertical-equilibrium sharp-interface model

`sx.VeGrid.from_corner_point(grid, perm, poro)` (or `VeGrid.from_cartesian`) collapses every column to a top
depth z_t, a thickness H, a pore-volume-weighted porosity φ and a thickness-weighted permeability K. CO₂ is a
layer of thickness h below the caprock, over brine. The maximum past thickness h_max tracks the residual CO₂
left behind when the interface retreats.

**Equations** (Nilsen et al. 2011, 2016; Nordbotten & Celia 2012). Column relative permeabilities are taken from
the MRST "simple" sharp interface:

- k_rg = (1 − s_wr) h / H
- k_rw = (H − h_max + (1 − s_gr)(h_max − h)) / H

Phase potentials at the caprock:

- Φ_g = p − ρ_c g z_t
- Φ_w = p − ρ_w g z_t − Δρ g h

The pressure equation ∇·(K H (λ_g∇Φ_g + λ_w∇Φ_w)) = q is solved implicitly with a Jacobi-preconditioned CG,
using the two-point column transmissibility of the top surface. The CO₂ volume per column,

V = A φ [(1 − s_wr) h + s_gr (h_max − h)],

is then advanced explicitly with phase-upwind fluxes under a CFL limit (IMPES). The conserved variable is V, and
h and h_max are recovered from it, so mass is conserved to round-off. Open edges are held hydrostatic and
CO₂ that leaves through them is counted as *exited*. Fluid properties are constant during a run. By default
they come from the CO₂–brine module at (p_fluid, t_fluid) through `ve_fluid_properties`.

**Trapping.** Spill points are found with a priority flood over the top surface (Barnes et al. 2014), not with
MRST's cell- or node-based `trapAnalysis`. A column is in a trap if its caprock lies above that trap's spill
depth. The inventory (kg) splits into residual (s_gr over h_max − h), structural (mobile CO₂ above the spill
depth in trap columns), free plume (the rest of the mobile CO₂) and exited.

**Validation:**

| Check | Result |
|---|---|
| Mass conservation (2201 steps, with residual and outflow) | relative error < 1e-10 |
| Nordbotten & Celia (2006) radial similarity solution, h = H(√(λV/(πφ_eff H r²)) − 1)/(λ − 1) | L1 relative error 0.105; plume radius 336 m vs 324 m analytic |
| Upslope speed on a tilted caprock, c = K k_rg Δρ g sinθ / (μ_c φ (1 − s_wr)) | 311.5 vs 315.0 m/yr (1.1 %) |
| Fill and spill under a dome | plume base 976.43 m vs spill 976.36 m; trapped 31 942 m³ vs capacity 31 719 m³ (+0.7 %); the rest exits |

In notebook 13, the structural trap fills to its computed capacity (0.082 Mt) and the mass-balance error
stays below 2e-13.

**Limitations.**

- Constant densities and viscosities.
- No capillary fringe, CO₂ dissolution or convective mixing, caprock rugosity, or relative-permeability
  hysteresis beyond a single s_gr.
- Explicit transport, so the time step scales with cell size.
- Top-surface connections use logical (i, j) neighbours, so fault juxtaposition and sealing faults are not
  represented.
- Not compared against MRST-co2lab output, because no Octave/MATLAB run was available. Checked against
  analytic solutions only.

## FIM-OBL and VE on the same case

Example 13 also runs the full 3D solver (FIM-OBL, `run_fim_co2_brine`) on the same faulted grid and injection, for
the 10 injection years, and compares plume footprint, farthest plume edge, CO₂ mass, mass balance and run time with
VE. VE runs in seconds on map columns; FIM-OBL solves every 3D cell (minutes) and includes CO₂ dissolution and
pressure. The FIM run has no shut-in schedule, so the comparison stops at the end of injection.

## References

- Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: an optimal depression-filling and watershed-labeling
  algorithm. *Computers & Geosciences* 62, 117–127.
- Lie, K.-A. (2019). *An Introduction to Reservoir Simulation Using MATLAB/GNU Octave* (MRST). Cambridge Univ. Press.
- Nilsen, H. M., Herrera, P. A., Ashraf, M., Ligaarden, I., Iding, M., Hermanrud, C., Lie, K.-A., Nordbotten, J. M.,
  Dahle, H. K., Keilegavlen, E. (2011). Field-case simulation of CO₂-plume migration using vertical-equilibrium
  models. *Energy Procedia* 4, 3801–3808.
- Nilsen, H. M., Lie, K.-A., Andersen, O. (2016). Fully-implicit simulation of vertical-equilibrium models with
  hysteresis and capillary fringe. *Computational Geosciences* 20, 49–67.
- Nordbotten, J. M., Celia, M. A. (2006). Similarity solutions for fluid injection into confined aquifers.
  *J. Fluid Mech.* 561, 307–327.
- Nordbotten, J. M., Celia, M. A. (2012). *Geological Storage of CO₂: Modeling Approaches for Large-Scale
  Simulation*. Wiley.

## MRST attribution

The sharp-interface relative permeabilities in `crates/sx-core/src/flow/ve.rs` are derived from MRST
(co2lab, `addVERelpermSharpInterface.m`), Copyright 2009-2026 SINTEF Digital, Mathematics & Cybernetics. That
code is GPL-3.0-or-later and is distributed here under AGPL-3.0-or-later. The rest of the VE model, the trap
analysis and the corner-point code are independent implementations. See `NOTICE`.
