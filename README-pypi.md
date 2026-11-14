# SubsurfiX

**SubsurfiX** is developed to automate workflows for subsurface analysis, including machine-learning (ML) assisted stratigraphic model building, 3D property modeling, and flow and heat simulations for injection–production purposes.

## Capabilities

- **Grids and properties:** regular and faulted corner-point grids (GRDECL), filled by kriging, sequential Gaussian or indicator simulation, or machine learning, with P10/P50/P90 cases.
- **Flow and heat, finite-volume method (FVM):** two in-package FVM solvers, an open-DARTS-inspired fully implicit solver with operator-based linearization (FIM-OBL) for water, water–oil, oil–gas and CO₂–brine with heat, and a MODFLOW-inspired control-volume finite-difference Rust translation for single-fluid pressure and heat. An optional external check runs the USGS MODFLOW 6 program.
- **Flow and heat, finite-element method (FEM), experimental:** an OpenGeoSys-style continuous Galerkin solver for coupled water flow and heat on unfaulted grids. Multiphase flow and faulted models remain FVM-only.
- **CO₂ vertical-equilibrium model** (after MRST) and data exchange with Leapfrog (OMF) and OpenGeoSys (`.vtu`).

## Status and documentation

SubsurfiX is under active development, and many functions are still experimental and not fully validated. Please use with caution.

Requires Python 3.12 or newer. Installation details, executed example notebooks, figures and release history are on [GitHub](https://github.com/GeoLarryLai/SubsurfiX).

## Citation

Lai, L.S.-H. (2026) SubsurfiX. *Zenodo*, [https://doi.org/10.5281/zenodo.23832007](https://doi.org/10.5281/zenodo.23832007).

License: **AGPL-3.0-or-later**. You are free to use, modify, and share it; anyone distributing modified code or running a modified version as a network service must share its source under the same license.

Method and data credits are in NOTICE.
