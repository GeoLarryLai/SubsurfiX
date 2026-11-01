<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Changelog

## [0.1.3] - 2026-11-01

- Fully implicit (FIM-OBL) runs use GMRES with a CPR preconditioner (aggregation AMG on cell pressures + ILU(0))
  by default, with an automatic fall back to the direct LU if GMRES does not converge. `linear_solver="direct"`
  on the four `run_fim_*` functions selects the direct LU. Results agree with the direct LU to round-off (largest
  relative pressure difference about 1e-10) with the same time steps, and runs are 1.1–8× faster on the
  example cases. New result counters: `n_linear_solves`, `n_linear_iterations`, `n_linear_fallbacks`.
- The direct LU reuses its symbolic factorization while the Jacobian pattern is unchanged. AMD and
  nested-dissection orderings were tried and were slower than faer's COLAMD, which is kept.
- `subsurfix.parallel.run_cases` runs several independent cases in separate processes (or one after another with
  `parallel=False`) and reports per-case and wall times; `examples/fluid_cases.py` holds the four example-12 cases.
- Examples 12 and 13: sections comparing the iterative and direct linear solvers.
- `scripts/make_mf6_vs_fim_heat_gif.py`: MODFLOW 6 and FIM-OBL cold-water injection side by side on the block
  model at 0.02 m³/s.
- Faster FIM-OBL runs with identical results: the Jacobian sparsity structure is reused between Newton
  iterations while it is unchanged, OBL table lookups no longer allocate, and the CPR preconditioner multiplies
  only the pressure columns. About 12–15 % less run time on the example cases, results bitwise the same.
- `subsurfix.viz3d.animate_grids`: several 3D grid panels in one animated GIF.
- Whole Kansas model: `scripts/kansas_fluids.py` (water, gas and CO₂ injection; `image/fluids_compare_kansas.gif`)
  and `scripts/make_methods_animation.py` (MODFLOW 6 against FIM-OBL, water and heat; `--rate`, `--p-max`,
  `--mf6-tolerance`, `--times`; `image/methods_compare_kansas.gif`).
- `subsurfix.modflow_heat`: `tolerance="tight"` (default, unchanged) or `"normal"` for the steady well-flow solve
  (`MAW_IMS`), for large grids where the tight limits are not reached.

- Kansas animations (`scripts/kansas_fluids.py`, `scripts/make_methods_animation.py`): layers on the structural top,
  wells 2.7 km (heat) and 3.6 km (fluids) apart, a vertical section through both wells under each panel (south or
  west on the left), one colour bar per panel and per section, and a grey base showing permeability on a log scale
  (`--background permeability`, the default, or `porosity`) with its own colour bar; `--from-npz` / `--render-only`
  redraw from saved results.
- `subsurfix.viz3d`: `LayeredGrid` and `animate_layered`; fixed `animate_grid` and `animate_grids` showing the last
  frame in every frame.
- README: the block-model animations (`image/mf6_vs_fim_heat.gif`, `image/fluids_compare.gif`) are removed; the
  whole-Kansas animations are the comparisons, and the block versions stay in notebooks 10 and 12.
- README model picture: porosity and permeability side by side. Notebook 04 maps are grouped by method (one box per method) and use one copper scale with a colour
  bar under each map, the well data and a short method table with references; notebook 02 shows the named formation
  tops, the dynamic time warping (DTW) correlation as colour bands and a Wheeler-style relative geological time view.

## [0.1.2] - 2026-10-31

- Examples 09, 10 and 12 add a "Real-world case: Kansas (KGS) property window" section after the block model,
  reading two small committed files (`examples/data/kansas_window/`, about 90 KB) through `examples/kansas_window.py`,
  with results and block-versus-Kansas tables built from the runs. Example 09 now has a builder script (`_build_nb09.py`).
- README animations (doublet and the three fully implicit GIFs) are rendered from the Kansas window runs.
- README rewritten in plain language: fundamentals, a method overview table (FIM-OBL, MODFLOW 6, CO₂ VE,
  corner-point grids) with credits and pros/cons, and references grouped by subject. The three fully implicit GIFs
  are shown as one side-by-side GIF (`image/fluids_compare.gif`).
- Examples 09, 10 and 14 run their real-world sections on the whole Kansas model (22 × 22 × 12 cells,
  `kansas_model()`); example 13 compares the full 3D solver (FIM-OBL) with the CO₂ VE model on the same faulted case.
- Corner-point grids: GRDECL reader helpers and fault-overlap transmissibilities in the Python API, with validation
  tests (Cartesian equivalence, fault overlap, a fully implicit run across a fault with mass balance, GRDECL round trip).
- Experimental CO₂ vertical-equilibrium (sharp-interface) migration model in Rust (`sx.run_co2_ve`, `VeGrid`), with
  spill-point and residual trapping inventories; relative permeabilities derived from MRST co2lab (see `NOTICE`).
- Example 13 (faulted corner-point block and CO₂ vertical equilibrium) and `docs/cornerpoint_and_ve.md`.
- Flow diagnostics derived from MRST `diagnostics` (see `NOTICE`): forward/backward time of flight, injector and
  producer tracer partitions, well-pair volumes and allocation, F–Φ, Lorenz coefficient and sweep efficiency
  (`FlowModel.flow_diagnostics`, `sx.diagnostics`), plus an optional TOF-based producer response next to the default
  `run_tracer`; example 14 and `docs/flow_diagnostics.md`.

## [0.1.1] - 2026-10-31

- Requires Python 3.12 or newer (0.1.0 allowed 3.10); GIS, spreadsheet, MODFLOW, OMF and 3D-plot packages are regular dependencies (no extras).
- MODFLOW 6 bridges (experimental): `subsurfix.modflow` (GWF flow) and `subsurfix.modflow_heat` (GWF + GWE heat
  doublet, including wells open in several layers); examples 09 and 10 on a layered block model.
- OMF import and export for Leapfrog (experimental), example 11.
- Experimental fully implicit multiphase engine (operator-based linearization): water–oil, black oil, CO₂–brine
  and thermal water, with CO₂–brine fluid properties; example 12 with 3D GIFs on the same block model.
- `subsurfix.viz3d` animation helper and a 3D doublet GIF on the block model.
- Flow solver speed: cached symbolic factorizations, fixed-pattern assembly, optional threads (`n_threads`),
  report decimation (`report_every`) and streamed snapshots (`snapshot_dir`).
- Packaging: no optional extras. pyproj, shapely, pyshp, openpyxl, python-calamine, flopy, pyvista, imageio and
  omf are now regular dependencies; development tools are in `requirements-dev.txt`. maturin is pinned below 1.15.

## [0.1.0] - 2026-10-31

First public release (source only).
