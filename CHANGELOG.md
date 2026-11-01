<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Changelog

## [0.1.2] - unreleased

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
