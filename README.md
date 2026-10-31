<p align="center"><img src="image/logo.png" alt="SubsurfiX" width="200"></p>

# SubsurfiX

[![PyPI version](https://img.shields.io/pypi/v/subsurfix.svg)](https://pypi.org/project/subsurfix/) [![pepy downloads](https://static.pepy.tech/badge/subsurfix)](https://pepy.tech/project/subsurfix) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23832007.svg)](https://doi.org/10.5281/zenodo.23832007) [![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](./LICENSE)

v0.1.0

From a folder of well logs to a checked, correlated log set and a 3D property model you can run flow and heat
simulations on. Rust engine, driven from Python or the command line.

Status: early-stage research software. macOS arm64 wheels are on PyPI; other platforms build from the sdist
(needs Rust, cmake and a C++ compiler). APIs may change; check results before use.

## Installation

```bash
pip install subsurfix
```

Prebuilt wheels are for macOS arm64; elsewhere pip builds from source, which needs Rust (stable), Python 3.10+, a C/C++ compiler and libclang (for the LightGBM build). To build from a clone:

```bash
git clone https://github.com/GeoLarryLai/SubsurfiX subsurfix && cd subsurfix
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy pyarrow pandas pyproj
maturin develop -r                                    # builds subsurfix._native
pip install matplotlib lasio jupyter python-calamine  # optional: notebooks, .xls PTS files
```

Tests: `cargo test --workspace`, then `maturin develop -r --features testing` and `python -m pytest tests/python -q`.

## Citation

Lai, L.S.-H. (2026) SubsurfiX. *Zenodo*, [https://doi.org/10.5281/zenodo.23832007](https://doi.org/10.5281/zenodo.23832007)

![3D porosity model built from 60 Kansas wells](image/hero_3d_porosity_model.png)

## Key Features

- **QC and gap fill:** reads LAS files and surveys, masks bad intervals, and fills gaps (rules + LightGBM) with a
  provenance code on every value.
- **Well correlation:** DTW between neighbouring wells, solved as one network for all wells
  ([Wheeler and Hale, 2014](https://doi.org/10.1190/segam2014-0227.1);
  [Sylvester, 2023](https://doi.org/10.1111/bre.12787); [Samant et al., 2025](https://doi.org/10.1029/2024PA005082)).
- **Grids and properties:** regular or corner-point (GRDECL) grids, upscaling, kriging, SGS, SIS and ML P10/P50/P90.
- **Flow and heat:** pressure, tracer, two-phase water/oil and heat transport; export to OpenGeoSys (`.vtu`).

## Examples

![Raw, QC'd and gap-filled logs](image/ex_qc_gap_fill.png)
*One Kansas well: raw log (grey), what QC kept (black) and filled samples (red).*

![Well correlation section](image/ex_well_correlation.png)
*West–east section: GR on elevation (left) and on correlated layer time (right), where beds line up.*

![Porosity maps of one layer](image/ex_property_maps.png)
*Porosity in one layer by kriging, SGS, SIS and ML propagation.*

![Geothermal doublet](image/ex_heat_doublet.png)
*Heat transport check against Ogata–Banks, and the cold front of an injector–producer doublet through time.*

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
`ml_propagation`: [Meinshausen (2006)](https://jmlr.org/papers/v7/meinshausen06a.html).
Flow and heat: [open-DARTS](https://gitlab.com/open-darts/open-darts), [OpenGeoSys](https://www.opengeosys.org),
[OPM](https://opm-project.org). Data credits are in `NOTICE`.

Third-party notices: [`NOTICE`](NOTICE).
