# Examples

One notebook per workflow step. Run them in order; 03–05 and 08 reuse files written by earlier notebooks. 09–12 are self-contained (09, 10 and 12 build the block model in `block_model.py`).

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
| 09_modflow_flow | MODFLOW 6 groundwater-flow cross-check on a layered block model (needs `mf6`) |
| 10_modflow_heat | MODFLOW 6 GWE heat doublet with multi-layer wells on a layered block model (needs `mf6`) |
| 11_leapfrog_omf | OMF export and read-back for Leapfrog (experimental) |
| 12_multiphase_fluids | Experimental fully implicit water–oil, oil–gas and CO₂–brine runs on one layered block model, with 3D GIFs |

Data: `scripts/fetch_open_data.sh` (KGS wells, Wister) and `scripts/fetch_force2020.sh` (FORCE 2020).
Environment variables: `SX_EXAMPLE_PROJECT` (KGS project folder, default `kgs_project/`), `SX_EXAMPLE_OUT`
(figures and intermediate files, default `outputs/`), `SX_WISTER` (Wister data folder).
To regenerate the notebooks: `python _build_notebooks.py`, then execute them with `jupyter nbconvert --execute --inplace`.
