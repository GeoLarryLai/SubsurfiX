# Examples

One notebook per workflow step. Run them in order; 03–05 and 08 reuse files written by earlier notebooks. 09–12 are self-contained: 09, 10 and 12 first run a layered block model built in code (`block_model.py`), then the same workflow in a "Real-world case" section on the Kansas (KGS) property model committed in `data/kansas_window/` (read by `kansas_window.py`): 09 and 10 use the whole model (22 × 22 × 12 cells, 11 × 11 km × 60 m), 12 a 10 × 10 × 12 window of it. 13 builds its faulted corner-point grid in code and compares the full 3D solver (FIM-OBL) with CO₂ VE on it; 14 reuses the block model, the whole Kansas model and a smaller version of the faulted block.

| Notebook | What it shows |
|---|---|
| 01_log_qc_and_gap_fill | Pipeline run on 60 KGS wells: QC and machine-learning gap fill |
| 02_well_correlation | DTW well correlation onto a relative-geologic-time axis |
| 03_grid_and_upscaling | Corner-point grid following a mapped top; log upscaling to cells |
| 04_property_modelling | Variogram, kriging, SGS, SIS and ML propagation |
| 05_water_injection | Two-phase water injection with the built-in flow solver |
| 06_heat_doublet_and_ogs_export | Heat transport check, geothermal doublet, OpenGeoSys VTU export |
| 07_wister_geothermal_well | Log washout flags and PTS survey for a geothermal well |
| 08_grdecl_export | GRDECL grid and property export with a read-back check |
| 09_modflow_flow | MODFLOW 6 groundwater-flow cross-check on a layered block model and a Kansas KGS window (needs `mf6`) |
| 10_modflow_heat | MODFLOW 6 GWE heat doublet with multi-layer wells on a layered block model and a Kansas KGS window (needs `mf6`) |
| 11_leapfrog_omf | OMF export and read-back for Leapfrog (experimental) |
| 12_multiphase_fluids | Experimental fully implicit water–oil, oil–gas and CO₂–brine runs on a layered block model and a Kansas KGS window, with 3D GIFs |
| 13_cornerpoint_and_co2_ve | Faulted corner-point block: fault-overlap transmissibilities, a FIM run across the fault, and an experimental CO₂ vertical-equilibrium plume with spill-point and residual trapping |
| 14_flow_diagnostics | Flow diagnostics (time of flight, tracer partitions, well pairs, F–Φ, Lorenz, sweep) on the block model, the Kansas KGS window and a faulted corner-point block, with a run_tracer vs TOF comparison |

Data: `scripts/fetch_open_data.sh` (KGS wells, Wister) and `scripts/fetch_force2020.sh` (FORCE 2020).
Environment variables: `SX_EXAMPLE_PROJECT` (KGS project folder, default `kgs_project/`), `SX_EXAMPLE_OUT`
(figures and intermediate files, default `outputs/`), `SX_WISTER` (Wister data folder).
To regenerate the notebooks: `python _build_notebooks.py` (01–08), `_build_nb09.py`, `_build_nb10.py`, `_build_nb12.py`, `_build_nb13.py` and `_build_nb14.py`, then execute them with `jupyter nbconvert --execute --inplace`.

## Setup

Build from a clone (needs Rust, Python 3.12+, a C/C++ compiler and libclang for the LightGBM build):

```bash
git clone https://github.com/GeoLarryLai/SubsurfiX subsurfix && cd subsurfix
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt                   # maturin, ruff and notebook tools
maturin develop -r                                    # builds subsurfix._native and installs the dependencies
```

Examples 09 and 10 need the MODFLOW 6 program `mf6`. FloPy installs it with `get-modflow :flopy`.

## Quick start details

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
