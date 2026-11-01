# Flow diagnostics

Flow diagnostics describe where injected fluid goes, and how fast, using a single converged flux field. There is
no time stepping. The methods follow MRST's `diagnostics` module (Shahvali et al. 2012; Møyner, Krogstad and Lie
2015). The Rust code in `crates/sx-core/src/flow/diagnostics.rs` is derived from MRST (GPL-3.0-or-later,
SINTEF Digital) and distributed under AGPL-3.0-or-later; see `NOTICE`.

## API

```python
import subsurfix as sx
d = sx.diagnostics.flow_diagnostics(model, mobility, wells)   # one pressure solve + diagnostics
d["tof_forward"], d["tof_backward"]          # s, per cell (NaN for inactive cells)
d["injector_tracer"], d["producer_tracer"]   # (n_wells, n_cells) partitions, each column sums to 1
d["injector_partition"], d["producer_partition"]  # index of the dominant well (-1 if none)
d["well_pairs"]                              # pore_volume, injector_allocation, producer_allocation per pair
d["F"], d["Phi"], d["lorenz"], d["Ev"], d["tD"]  # F–Φ curve, Lorenz coefficient, sweep vs PVI
```

- `FlowModel.flow_diagnostics(mobility, wells, fixed=None, ..., max_tof=None)` is the native call. It solves the
  pressure with `solve_pressure_with` and returns the fields above together with `p`, `flux` and the solved wells.
  `sx.diagnostics.flow_diagnostics` also adds `F`, `Phi`, `lorenz`, `Ev` and `tD`. The free functions
  `sx.diagnostics.f_phi`, `sx.diagnostics.lorenz_coefficient` and `sx.diagnostics.sweep_efficiency` take arrays.
- The mobility can be single phase (1/μ), or the frozen total mobility of a multiphase state:
  `sx.diagnostics.total_mobility(sw, corey)` uses the same Corey curves and defaults as the solvers. This gives
  diagnostics on fully implicit (FIM) total fluxes, recomputed by a pressure solve at fixed λt.
- The flux field can come from Cartesian or corner-point models. Fault-overlap connections are handled like any
  other connection.

## Methods

- **Time of flight.** Upwind finite-volume operator: in each cell, outflow minus upwind inflow equals pore volume.
  This gives forward TOF from the sources, and backward TOF from the sinks after reversing the fluxes. TOF is
  cell-centred, so in 1D, Tf = (i + 1)·PV/q and Tb = (N − i)·PV/q. Cells that would exceed the cap
  (`max_tof`, default 50 × total PV / total injection) get the cap and pass nothing downstream, as in MRST.
- **Tracers.** Each well's tracer is solved with the same operator (one sparse LU factorization per direction).
  The partition is the well with the largest share.
- **Well pairs.** A pair's pore volume is Σ c_inj · c_prod · PV. Allocation splits each well's perforation rates
  by the tracer of the opposite wells.
- **F–Φ, Lorenz and sweep.** Cells are sorted by total travel time τ = Tf + Tb; Φ = cumulative PV and
  F = cumulative PV/τ, both normalized. The Lorenz coefficient is 2 × (area under F–Φ − ½). Sweep:
  tD = dΦ/dF, Ev = Φ + (1 − F)·tD.

## Default and option for producer response

`sx.diagnostics.producer_tracer_response(model, mobility, wells, times, method="transport")` keeps the existing
time-stepped `run_tracer` as the **default**. `method="tof"` is the new option: the produced concentration at
time t is the fraction of producer flux whose travel time is ≤ t. Notebook 14 runs both on the same flux
field. On coarse grids, the explicit tracer arrives earlier because of numerical dispersion, and TOF arrives
about half a cell late at each end. The two converge as the grid is refined.

## Validation

The diagnostics were checked against analytic answers:

| Case | Result |
|---|---|
| 1D line, TOF = pore volume / flux | max relative error 7e-14 |
| Homogeneous 1D piston | Lorenz 1e-15, sweep 1 at 1 PVI |
| Layered (k = 8, 4, 2, 1), fixed-pressure ends | Lorenz 0.3833333333 = analytic 0.3833333333 |
| Quarter five-spot, 21 × 21 | diagonal symmetry 4e-13, forward/backward symmetry 4e-12 |
| Tracer partitions sum to 1 | worst 9e-16 (also on a faulted corner-point grid) |
| F–Φ endpoints | (0, 0) and (1, 1) |

## Limitations

- Steady state only. Each result is a snapshot of one incompressible flux field, with no gravity or
  compressibility in the diagnostics pressure solve. For multiphase states, the total mobility is frozen.
- Fixed-pressure inflow boundaries act as TOF sources, but partitions and pairs are computed only for wells.
- Stagnant cells at the TOF cap pass nothing downstream, so their neighbours can have tracer sums below 1.
