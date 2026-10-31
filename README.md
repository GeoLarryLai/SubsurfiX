<p align="center"><img src="image/logo.png" alt="SubsurfiX" width="220"></p>

# SubsurfiX

[![PyPI version](https://img.shields.io/pypi/v/subsurfix.svg)](https://pypi.org/project/subsurfix/) [![pepy downloads](https://static.pepy.tech/badge/subsurfix)](https://pepy.tech/project/subsurfix) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23832007.svg)](https://doi.org/10.5281/zenodo.23832007) [![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](./LICENSE)

v0.1.1

> [!WARNING]
> **SubsurfiX** is under active development, and many functions are still experimental and not fully validated. Please use with caution.

**SubsurfiX** takes you from a folder of well logs to a checked, correlated log set and a 3D property model you can run flow and heat simulations on. I built it because these steps usually live in separate tools, and every hand-off (cleaning logs, filling gaps, correlating wells, gridding, simulating) is where things get lost or quietly changed. Here it's one project file and one run: every filled value carries a provenance code, so you always know what was measured and what was filled. The engine is in Rust, and you drive it from Python or the command line.

## Installation

```bash
pip install subsurfix
```

Prebuilt wheels are for macOS arm64; elsewhere pip builds from source, which needs Rust (stable), Python 3.12+, a C/C++ compiler and libclang (for the LightGBM build). To build from a clone:

```bash
git clone https://github.com/GeoLarryLai/SubsurfiX subsurfix && cd subsurfix
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt                   # maturin, pytest and notebook tools
maturin develop -r                                    # builds subsurfix._native and installs the dependencies
```

Tests: `cargo test --workspace`, then `maturin develop -r --features testing` and `python -m pytest tests/python -q`.

The MODFLOW 6 bridges (examples 09 and 10) need the `mf6` program. FloPy installs it with:

```bash
get-modflow :flopy
```

## Citation

Lai, L.S.-H. (2026) SubsurfiX. *Zenodo*, [https://doi.org/10.5281/zenodo.23832007](https://doi.org/10.5281/zenodo.23832007)

![3D porosity model built from 60 Kansas wells](image/hero_3d_porosity_model.png)
*A 3D porosity model built from 60 Kansas wells, the end product of the whole workflow.*

## What it can do

- Read LAS files and surveys, mask bad intervals, and fill gaps (rules + LightGBM), tagging every value with a provenance code
- Correlate wells with DTW between neighbouring wells, solved as one network for all wells ([Wheeler and Hale, 2014](https://doi.org/10.1190/segam2014-0227.1); [Sylvester, 2023](https://doi.org/10.1111/bre.12787); [Samant et al., 2025](https://doi.org/10.1029/2024PA005082))
- Build regular or corner-point (GRDECL) grids, upscale, and estimate properties by kriging, SGS, SIS or ML (P10/P50/P90); `method="ember"` adds embedded kriging, well conditioning and spatially correlated realizations, with blocked spatial cross-validation (`ember_cv`)
- Score how many lithology classes the logs separate and interpret facies (sandstone, shale, limestone, dolomite, cherty carbonate) from cutoffs or a mineral-mixing solve (`subsurfix.facies`)
- Run pressure, tracer, two-phase water/oil and heat transport, and export to OpenGeoSys (`.vtu`)
- Cross-check flow and heat runs against MODFLOW 6 (GWF and GWE) through FloPy
- Exchange grids and properties with Leapfrog through OMF files (experimental)
- Experimental fully implicit multiphase engine (operator-based linearization): water–oil, oil–gas (black oil), CO₂–brine and thermal runs
- Run from Python (`sx.Project(...).run()`) or the command line (`subsurfix run --config project.toml`), with a built-in synthetic demo project that needs no data

## Examples

![Raw, QC'd and gap-filled logs](image/ex_qc_gap_fill.png)
*One Kansas well, showing the raw log (grey), what QC kept (black) and the filled samples (red), so you can see exactly what was touched.*

![Well correlation section](image/ex_well_correlation.png)
*A west–east section with GR plotted on elevation (left) and on correlated layer time (right). On the right, the beds line up.*

![Porosity maps of one layer](image/ex_property_maps.png)
*Porosity in one layer mapped four ways (kriging, SGS, SIS and ML propagation), side by side for comparison.*

![Geothermal doublet](image/ex_heat_doublet.png)
*A heat transport check against the Ogata–Banks solution, plus the cold front of an injector–producer doublet moving through time.*

![3D geothermal doublet on a layered block model](image/doublet_3d.gif)
*A geothermal doublet solved with SubsurfiX's heat model on a small layered block model (10 × 10 × 12 cells of 500 × 500 × 5 m, four units with porosity 0.06–0.22 and permeability 0.5–80 mD): 30 °C water injected into an 80 °C reservoir over 30 years, with a corner cut away to show the cold front through the layers.*

![Water injected into oil on the block model](image/fim_water_oil.gif)
*Experimental fully implicit run on the same block model: water injected into dead oil for 40 years, showing water saturation.*

![Gas injected into black oil on the block model](image/fim_oil_gas.gif)
*Experimental fully implicit run on the same block model: gas injected into a black oil (synthetic PVT table) for 40 years, showing gas saturation (including gas coming out of solution near the producer).*

![CO₂ injected into brine on the block model](image/fim_co2_brine.gif)
*Experimental fully implicit run on the same block model: CO₂ injected into brine at 50 °C for 40 years, showing CO₂ saturation.*

One executed notebook per step on open data (Kansas Geological Survey, FORCE 2020): see
[`examples/README.md`](examples/README.md). `scripts/` fetches the data.

## Quick start

```python
import subsurfix as sx

toml = sx.write_demo_project("sx_demo")   # small synthetic project, no data needed
sx.prepare_edges(toml)                    # headers, tops, surveys -> tables
p = sx.Project(toml)
p.run()                                   # ingest -> QC -> align -> normalize -> fill -> correlate -> export
print(p.status().to_pandas())
logs, tops = p.logs(well=1), p.tops()
```

Your own data: `sx.init_project("my_project", crs="EPSG:32614", las_dirs=["las"])` writes a commented
`project.toml`; `sx.template_toml()` lists every setting. Command line: `python -m subsurfix.edges project.toml`,
then `subsurfix run --config project.toml`.

## Acknowledgements

Log workflow: [Gonzalez et al. (2023)](https://www.tgs.com/technical-library/machinelearningworkflowforlogdata).
`ml_propagation`: [Daly (2021)](https://doi.org/10.3389/frai.2021.624697),
[Daly (2022)](https://doi.org/10.1007/s11004-021-09972-8), [Meinshausen (2006)](https://jmlr.org/papers/v7/meinshausen06a.html).
Flow and heat: [open-DARTS](https://gitlab.com/open-darts/open-darts), [OpenGeoSys](https://www.opengeosys.org),
[OPM](https://opm-project.org). The experimental fully implicit multiphase engine follows methods described by
[open-DARTS](https://gitlab.com/open-darts/open-darts) (operator-based linearization) and
[OpenGeoSys](https://www.opengeosys.org); it has not been validated as equivalent to either code.
MODFLOW 6 and FloPy: [USGS](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model). Data credits are in `NOTICE`.

Third-party notices: [`NOTICE`](NOTICE).
