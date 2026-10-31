<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Changelog

## [0.1.1] - unreleased

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
