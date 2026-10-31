# Examples

One notebook per workflow step. Run them in order; 03–05 and 08 reuse files written by earlier notebooks.

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

Data: `scripts/fetch_open_data.sh` (KGS wells, Wister) and `scripts/fetch_force2020.sh` (FORCE 2020).
Environment variables: `SX_EXAMPLE_PROJECT` (KGS project folder, default `kgs_project/`), `SX_EXAMPLE_OUT`
(figures and intermediate files, default `outputs/`), `SX_WISTER` (Wister data folder).
To regenerate the notebooks: `python _build_notebooks.py`, then execute them with `jupyter nbconvert --execute --inplace`.
