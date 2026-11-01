# Kansas (KGS) property model

Two small files used by the "Real-world case" sections of examples 09, 10, 12 and 14, by
`scripts/make_doublet_animation.py` and by the README animations. They are read by `examples/kansas_window.py`.

| File | Contents |
|---|---|
| `grid_upscaled.npz` | Regular grid written by notebook 03: `origin`, `cell` (500 × 500 × 5 m), `dims` (22 × 22 × 12), plus the upscaled well-cell porosity and clean-facies flags |
| `property_model.npz` | Property model written by notebook 04 on that grid: porosity `phi`, clean-facies flag `clean`, and `ok`, the ordinary-kriging porosity estimate (one value per cell, i fastest, then j, then layer from the top). `ok` is not a mask and the examples do not use it |

Source: public well logs and formation tops from the Kansas Geological Survey (KGS),
<https://www.kgs.ku.edu/>, as fetched by `scripts/fetch_open_data.sh` and processed by notebooks 01–04
(log QC and gap fill, correlation, gridding and upscaling, property modelling). The files only hold numeric
arrays derived from that data; rerun notebooks 01–04 on the KGS download to regenerate them.

Permeability is not stored: the examples derive it from porosity as in notebook 05,
k = 10^(−1 + 14 φ) mD (φ clipped to 0.03–0.30), doubled in clean cells, kv/kh = 0.1.

The examples use the whole grid (`kansas_model()`, 22 × 22 × 12 = 5808 cells, 11 × 11 km × 60 m).
Every cell has a porosity value (the first sequential Gaussian simulation realization), so there are no
inactive cells and nothing is filled in or substituted. The grid is flat: the structural top is not applied.
The folder name is kept from earlier versions, which used a 10 × 10 window (`kansas_window()` still does that).
