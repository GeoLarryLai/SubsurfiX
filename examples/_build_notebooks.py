# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Write example notebooks 01-08 (run with nbconvert --execute afterwards). 09 is edited by hand;
10 and 12 come from _build_nb10.py and _build_nb12.py."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SETUP = '''import os
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib.pyplot as plt
%matplotlib inline
# SX_EXAMPLE_PROJECT: KGS project folder (las/, edges/, project.toml), see scripts/fetch_open_data.sh.
# SX_EXAMPLE_OUT: folder for figures and the small model files passed between notebooks.
PROJ = Path(os.environ.get("SX_EXAMPLE_PROJECT", "kgs_project")).resolve()
OUT = Path(os.environ.get("SX_EXAMPLE_OUT", "outputs")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 80, "savefig.dpi": 110})
def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip()) for c in cells]
    return n


SETUP_MD = "md:## Setup\nSet `SX_EXAMPLE_PROJECT` to your KGS project folder (default `kgs_project/` next to this notebook)."

books = {}
books["01_log_qc_and_gap_fill.ipynb"] = [
"""md:# 01 · Log QC and machine-learning gap fill

Runs the SubsurfiX pipeline on 60 public Kansas Geological Survey (KGS) wells: ingest, quality control (QC),
and gap fill of six logs (GR, RHOB, NPHI, DTC, RESD, PEF). Missing intervals are predicted with gradient-boosted
trees, LightGBM (Ke et al. 2017,
*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS).

Data: KGS digital LAS archive. Download with `scripts/fetch_open_data.sh`.""",
SETUP_MD, SETUP,
"""md:## Run the pipeline
`Project.run()` reruns only the stages whose inputs changed.""",
"""import subsurfix as sx
p = sx.Project(PROJ / "project.toml")
p.run()
p.status().to_pandas()[["stage", "state"]]""",
"""md:## QC report
Samples removed by each QC rule, summed over wells.""",
"""rep = pd.read_csv(PROJ / "stages/qc/qc_report.csv")
cols = [c for c in rep.columns if c.startswith("n_")]
rep.groupby("log")[cols].sum()""",
"""md:## Fill provenance
Share of each final log that is measured, predicted by the model, copied from a neighbor, or left unfilled.""",
"""fin = pq.read_table(PROJ / "export/logs_final.parquet").to_pandas()
logs = ["gr", "rhob", "nphi", "dtc", "resd", "pef"]
lab = lambda c: "measured" if c == 0 else "model" if c == 1 else "unfilled" if c == 9 else "copy/other"
prov = pd.DataFrame({l: fin[f"{l}_prov"].map(lab).value_counts(normalize=True) for l in logs}).fillna(0)
fig, ax = plt.subplots(figsize=(7, 3.2))
prov.T.reindex(columns=["measured", "model", "copy/other", "unfilled"], fill_value=0).plot.bar(stacked=True, ax=ax, width=0.7)
ax.set_ylabel("share of samples"); ax.legend(fontsize=8, loc="lower right")
save(fig, "01_fill_provenance.png")
prov.round(3)""",
"""md:## One well: raw, after QC, filled
Grey is the raw log, black what QC kept, red the filled samples.""",
"""raw = pq.read_table(PROJ / "stages/ingest/logs_raw.parquet").to_pandas()
qc = pq.read_table(PROJ / "stages/qc/logs_qc.parquet").to_pandas()
share = fin.groupby("well_id").rhob_prov.apply(lambda s: (s == 1).mean()).sort_values()
wid = share.index[len(share) // 2]
cols = [("gr_api", "GR (API)"), ("rhob_gcc", "RHOB (g/cc)"), ("nphi_frac", "NPHI"), ("dtc_usft", "DTC (us/ft)")]
r, q, f = (d[d.well_id == wid] for d in (raw, qc, fin))
fig, axs = plt.subplots(1, 4, figsize=(10, 6), sharey=True)
for ax, (c, lab) in zip(axs, cols):
    ax.plot(r[c], r.md_ft, color="0.75", lw=0.6, label="raw")
    ax.plot(q[c], q.md_ft, color="k", lw=0.6, label="after QC")
    ax.plot(f[c].where(f[f"{c.split('_')[0]}_prov"] != 0), f.md_ft, color="tab:red", lw=0.6, label="filled")
    ax.set_xlabel(lab)
axs[0].invert_yaxis(); axs[0].set_ylabel("MD (ft)"); axs[0].legend(fontsize=7)
fig.suptitle(f"well_id {wid}"); save(fig, "01_well_tracks.png")""",
]

books["02_well_correlation.ipynb"] = [
"""md:# 02 · Well correlation

The correlate stage aligns all wells to a shared relative-geologic-time axis (`layer_time`) by dynamic time
warping ([Sakoe & Chiba 1978](https://doi.org/10.1109/TASSP.1978.1163055)) between neighboring wells, solved
jointly in the spirit of [Wheeler & Hale (2014)](https://doi.org/10.1190/segam2014-0227.1), with formation
tops as anchors. Run notebook 01 first.""",
SETUP_MD, SETUP,
"""md:## Load the stage outputs""",
"""wells = pq.read_table(PROJ / "export/wells.parquet").to_pandas()
tops = pq.read_table(PROJ / "export/tops.parquet").to_pandas()
lt = pq.read_table(PROJ / "export/layer_time.parquet").to_pandas()
geo = pq.read_table(PROJ / "export/geometry.parquet").to_pandas()
fin = pq.read_table(PROJ / "export/logs_final.parquet", columns=["well_id", "k", "gr_api"]).to_pandas()
print(len(wells), "wells;", int(tops.suspect.sum()), "of", len(tops), "tops flagged suspect")
tops.groupby("top_name").agg(n=("md_ft", "size"), suspect=("suspect", "sum"), mean_elev_ft=("elev_ft", "mean")).sort_values("mean_elev_ft", ascending=False)""",
"""md:## West–east section
Eight wells from west to east, GR curves (black) between 3,000 and 3,700 ft below sea level, with the KGS formation
tops named in the legend. Right: the same curves on `layer_time`, the relative geological time from the
correlation: each top is an anchor (`layer_time` = top number + 1) and the DTW alignment fills in between, so this
is a Wheeler-style (chronostratigraphic) view in which correlated beds line up. Left: the same correlation on depth;
each colour band joins one `layer_time` interval (0.25 wide) between neighbouring wells, coloured by the mean GR of
that interval over the eight wells.""",
"""sel = wells.sort_values("x_m").iloc[:: max(1, len(wells) // 8)].head(8)
d = fin.merge(lt, on=["well_id", "k"]).merge(geo[["well_id", "k", "elev_ft"]], on=["well_id", "k"])
d = d[(d.elev_ft < -3000) & (d.elev_ft > -3700)]
names = tops.groupby("top_name").agg(e=("elev_ft", "mean"), t=("top_index", "first")).sort_values("e", ascending=False)
names = names[(names.e < -2990) & (names.e > -3700)]                     # tops inside the window
tcol = dict(zip(names.index, plt.cm.tab20(np.arange(len(names)) % 20)))
levels = np.arange(np.floor(d.layer_time.min()), np.ceil(d.layer_time.max()) + 1e-9, 0.25)
gr_band = d.groupby(np.digitize(d.layer_time, levels)).gr_api.mean()         # mean GR of each correlated interval
norm = plt.Normalize(30, 120); band_cmap = plt.cm.YlOrBr
def elev_at(s, lv):                                                     # depth of a layer_time level in one well
    s = s.dropna(subset=["layer_time"]).sort_values("layer_time")
    return np.interp(lv, s.layer_time, s.elev_ft, left=np.nan, right=np.nan)
fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), layout="constrained")
per = [d[d.well_id == w] for w in sel.well_id]
for i in range(len(per) - 1):                                            # correlated intervals between neighbours
    ea, eb = elev_at(per[i], levels), elev_at(per[i + 1], levels)
    for m in range(len(levels) - 1):
        q = [ea[m], eb[m], eb[m + 1], ea[m + 1]]
        if np.all(np.isfinite(q)) and (m + 1) in gr_band.index:
            axs[0].fill([i + 0.45, i + 1.45, i + 1.45, i + 0.45], q, color=band_cmap(norm(gr_band[m + 1])), alpha=0.45, lw=0)
for i, s in enumerate(per):
    g = (s.gr_api.clip(0, 150) / 150) * 0.9 + i
    axs[0].plot(g, s.elev_ft, lw=0.5, color="k"); axs[1].plot(g, s.layer_time, ".", ms=0.6, color="k")
    for _, row in tops[(tops.well_id == s.well_id.iloc[0]) & tops.top_name.isin(list(tcol))].iterrows():
        axs[0].plot([i, i + 0.9], [row.elev_ft] * 2, color=tcol[row.top_name], lw=1.8)
for n, r in names.iterrows():                                            # tops are the anchors: layer_time = index + 1
    axs[1].axhline(r.t + 1, color=tcol[n], lw=1.2, alpha=0.8)
for m in range(len(levels) - 1):
    if (m + 1) in gr_band.index:
        axs[1].axhspan(levels[m], levels[m + 1], color=band_cmap(norm(gr_band[m + 1])), alpha=0.45, lw=0, zorder=0)
axs[0].set_ylabel("elevation (ft, relative to sea level)"); axs[1].set_ylabel("relative geological time (layer_time)")
axs[0].set_ylim(-3720, -2980); axs[1].set_ylim(d.layer_time.max() + 0.2, d.layer_time.min() - 0.2)
for ax, t in zip(axs, ["on depth: colour bands join the same layer_time between wells", "on relative geological time (Wheeler-style)"]):
    ax.set_title(t, fontsize=10); ax.set_xticks(np.arange(len(sel)) + 0.45); ax.set_xticklabels(sel.well_id, fontsize=7)
    ax.set_xlabel("well_id, west to east")
fig.legend([plt.Line2D([], [], color=tcol[n], lw=2) for n in names.index], list(names.index), loc="outside lower center",
           ncol=min(len(names), 6), fontsize=8, title="formation tops (KGS picks)", title_fontsize=8)
fig.colorbar(plt.cm.ScalarMappable(norm, band_cmap), ax=axs, shrink=0.6, label="mean GR of the correlated interval (API)")
save(fig, "02_correlation_section.png")""",
"""md:## Top consistency
Spread of each top among neighboring wells after correlation; large spreads point to tops worth checking.""",
"""order = tops.groupby("top_name").elev_ft.mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(7, 3))
ax.boxplot([tops[tops.top_name == n].spread_ft.dropna() for n in order], tick_labels=list(order), showfliers=False)
ax.set_ylabel("spread (ft)"); ax.tick_params(axis="x", rotation=60, labelsize=7)
save(fig, "02_top_spread.png")""",
"""md:The same workflow runs on the FORCE 2020 North Sea wells ([Bormann et al. 2020](https://doi.org/10.5281/zenodo.4351155)):
fetch them with `scripts/fetch_force2020.sh`, convert with `force2020_to_las.py`, and use `force2020_project.toml`.""",
]

LOAD_MISS = '''from subsurfix import _native as nat
fin = pq.read_table(PROJ / "export/logs_final.parquet").to_pandas()
geo = pq.read_table(PROJ / "export/geometry.parquet").to_pandas()
tops = pq.read_table(PROJ / "export/tops.parquet").to_pandas()
mtop = tops[tops.top_name == "Mississippian"].set_index("well_id").elev_ft
d = fin.merge(geo[["well_id", "k", "elev_ft", "x_m", "y_m"]], on=["well_id", "k"])
d = d[d.well_id.isin(mtop.index)]
d["zrel_m"] = (mtop.reindex(d.well_id).values - d.elev_ft) * 0.3048          # depth below the top, m
d = d[(d.zrel_m >= 0) & (d.zrel_m < 60)].dropna(subset=["rhob_gcc", "gr_api"])
d["phi"] = ((2.71 - d.rhob_gcc.astype(float)) / 1.71).clip(0, 0.35)          # limestone density porosity
d["clean"] = (d.gr_api < 60).astype(np.int32)
print(len(d), "samples in", d.well_id.nunique(), "wells")'''

books["03_grid_and_upscaling.ipynb"] = [
"""md:# 03 · Grid building and upscaling

Builds a 22 × 22 × 12 grid over the 60 m below the Mississippian top: a corner-point grid that follows the
mapped top, and its flattened (stratigraphic) twin used for property modelling. Well-log porosity and a
clean-rock indicator are upscaled to the cells the wells cross. Run notebook 01 first.

Example assumptions: density porosity with a 2.71 g/cc limestone matrix, clean rock where GR < 60 API.""",
SETUP_MD, SETUP,
"""md:## Well samples in the interval""",
LOAD_MISS,
"""md:## Corner-point grid following the Mississippian top
The top is mapped by ordinary kriging of the well picks on the grid nodes; 12 layers of 5 m hang below it.""",
"""origin, cell, dims = (-4500.0, -5250.0, 0.0), (500.0, 500.0, 5.0), (22, 22, 12)
nx, ny, nz = dims
xs = origin[0] + cell[0] * np.arange(nx + 1)
ys = origin[1] + cell[1] * np.arange(ny + 1)
XN, YN = np.meshgrid(xs, ys)
wxy = d.groupby("well_id")[["x_m", "y_m"]].first()
pick = np.column_stack([wxy.x_m, wxy.y_m, np.zeros(len(wxy))])
top_m = -mtop.reindex(wxy.index).to_numpy() * 0.3048                      # depth below datum, m
nodes = np.column_stack([XN.ravel(), YN.ravel(), np.zeros(XN.size)])
top, _ = nat.ordinary_kriging(pick, top_m, nodes, "spherical", 0.0, float(np.var(top_m)), 6000.0)
cpg = nat.CornerPointGrid.from_surfaces(xs, ys, np.array([top + cell[2] * k for k in range(nz + 1)]))
print("corner-point grid", cpg.dims, "| top depth", top.min().round(1), "to", top.max().round(1), "m")
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.contourf(XN, YN, top.reshape(XN.shape), 15, cmap="viridis_r"); ax.plot(wxy.x_m, wxy.y_m, "k.", ms=3)
fig.colorbar(im, label="Mississippian top depth (m)"); ax.set_aspect("equal"); save(fig, "03_top_surface.png")""",
"""md:## Upscale to cells
Samples are averaged per cell (arithmetic for porosity, most frequent code for the indicator); cells with fewer
than 3 samples are left empty. Upscaling is done on the flattened grid, whose cells map one-to-one to the
corner-point cells.""",
"""pts = d[["x_m", "y_m", "zrel_m"]].to_numpy(np.float64)
idx, phi_up, cnt = nat.upscale_continuous(origin, cell, dims, pts, d.phi.to_numpy(np.float64), "arithmetic", 3)
idf, clean_up, _ = nat.upscale_facies(origin, cell, dims, pts, d.clean.to_numpy(), 3)
print(len(idx), "porosity cells and", len(idf), "indicator cells of", np.prod(dims))
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(d.phi, 40, density=True, alpha=0.5, label="log samples"); ax.hist(phi_up, 20, density=True, alpha=0.6, label="upscaled cells")
ax.set_xlabel("porosity"); ax.legend(fontsize=8); save(fig, "03_upscaling_hist.png")""",
"""md:## Save for the next notebooks""",
"""np.savez_compressed(OUT / "grid_upscaled.npz", origin=origin, cell=cell, dims=dims, xs=xs, ys=ys, top=top,
                    idx=idx, phi_up=phi_up, idf=idf, clean_up=clean_up)
print("saved grid_upscaled.npz")""",
]

books["04_property_modelling.ipynb"] = [
"""md:# 04 · Property modelling

Fills the grid from notebook 03 with porosity and the clean-rock indicator using ordinary kriging
([Matheron 1963](https://doi.org/10.2113/gsecongeo.58.8.1246)), sequential Gaussian simulation
([Gómez-Hernández & Journel 1993](https://doi.org/10.1007/978-94-011-1739-5_8)), sequential indicator simulation
([Journel 1983](https://doi.org/10.1007/BF01031292)), and machine-learning propagation with quantile regression
forests (Meinshausen 2006, *Quantile Regression Forests*, JMLR 7) or LightGBM. Run notebook 03 first.""",
SETUP_MD, SETUP,
"""md:## Load the upscaled cells and fit a variogram
Depth is stretched 50× so a 5 m vertical step weighs like 250 m laterally.""",
"""import subsurfix as sx
from subsurfix import _native as nat
g = np.load(OUT / "grid_upscaled.npz")
origin, cell, dims = tuple(g["origin"]), tuple(g["cell"]), tuple(int(v) for v in g["dims"])
idx, phi_up, idf, clean_up = g["idx"], g["phi_up"], g["idf"], g["clean_up"]
C = nat.grid_centers(origin, cell, dims)
S = np.array([1.0, 1.0, 50.0])
lag, gam, npair = nat.experimental_variogram(C[idx] * S, phi_up, 250.0, 16)
vg = nat.fit_variogram(lag, gam, npair, "spherical")
fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(lag, gam, "o")
h = np.linspace(0, lag.max(), 100); hr = np.minimum(h / vg["range"], 1)
ax.plot(h, vg["nugget"] + vg["psill"] * (1.5 * hr - 0.5 * hr ** 3), "r-"); ax.set_xlabel("lag (m)"); ax.set_ylabel("semivariance")
save(fig, "04_variogram.png"); vg""",
"""md:## Kriging, SGS, SIS and ML propagation""",
"""ok, okv = nat.ordinary_kriging(C[idx] * S, phi_up, C * S, vg["kind"], vg["nugget"], vg["psill"], vg["range"])
real = nat.sgs(C[idx] * S, phi_up, C * S, vg["kind"], vg["nugget"], vg["psill"], vg["range"], n_realizations=8, seed=1)
fac = nat.sis(C[idf] * S, clean_up, C * S, "spherical", 0.05, 0.2, 2000.0, n_realizations=8, seed=2)
cfg = sx.Project(PROJ / "project.toml").config["propagation"]
X = np.column_stack([C, ok])
q = nat.ml_propagation(X[idx], phi_up, X, model=cfg["method"], quantiles=cfg["quantiles"], n_trees=cfg["qrf_trees"], min_leaf=3, seed=3)
print("SGS", real.shape, "| SIS", fac.shape, "| ML quantiles", q.shape)""",
"""md:## Maps of one layer
Grid layer 5, 20-25 m below the Mississippian top (the top of the Mississippian carbonates, mapped in notebook 03;
952-1029 m below sea level in these 60 wells). The four copper panels share one porosity scale (0-0.2).""",
"""k = 4
sh = lambda v: np.asarray(v).reshape(dims[2], dims[1], dims[0])[k]
ext = [origin[0], origin[0] + dims[0] * cell[0], origin[1], origin[1] + dims[1] * cell[1]]
lay = np.asarray(idx) // (dims[0] * dims[1]) == k
cond = C[np.asarray(idx)[lay]][:, :2]                                   # cells with well data in this layer
from matplotlib.colors import ListedColormap
# panels grouped by method: one framed box per method, no line inside a method
groups = [("Kriging (one run)", [("estimate", sh(ok), "phi"), ("variance", sh(okv), "var")]),
          ("ML: quantile regression forest", [("P50 (median)", sh(q[:, len(cfg["quantiles"]) // 2]), "phi")]),
          ("SGS: sequential Gaussian simulation", [("realization 1", sh(real[0]), "phi"), ("mean of 8 realizations", sh(real.mean(0)), "phi")]),
          ("SIS: sequential indicator simulation", [("realization 1, clean-rock indicator", sh(fac[0]), "sis")])]
fig = plt.figure(figsize=(11, 9.2), layout="constrained")
rows = fig.subfigures(2, 1, hspace=0.04)
boxes = [b for r in rows for b in r.subfigures(1, 2, width_ratios=[2, 1], wspace=0.03)]
for box, (gname, panels) in zip(boxes, groups):
    box.set_edgecolor("#444444"); box.set_linewidth(1.2)        # frame: one box per method
    box.suptitle(gname, fontsize=10.5, fontweight="bold")
    for sf, (t, v, kind) in zip(box.subfigures(1, len(panels), wspace=0.02) if len(panels) > 1 else [box], panels):
        ax = sf.subplots()
        if kind == "phi":
            im = ax.imshow(v, origin="lower", extent=ext, cmap="copper", vmin=0.0, vmax=0.2)
            cb = sf.colorbar(im, ax=ax, location="bottom", shrink=0.85, ticks=[0, 0.1, 0.2])
            cb.set_label("porosity" + (" (above 0.2 shown as 0.2)" if float(np.nanmax(v)) > 0.2 else ""), fontsize=8)
        elif kind == "var":
            im = ax.imshow(v, origin="lower", extent=ext, cmap="Purples")
            cb = sf.colorbar(im, ax=ax, location="bottom", shrink=0.85)
            cb.set_label("kriging variance (porosity²)", fontsize=8)
        else:
            im = ax.imshow(v, origin="lower", extent=ext, cmap=ListedColormap(["#5a5a5a", "#e8d39a"]), vmin=-0.5, vmax=1.5)
            cb = sf.colorbar(im, ax=ax, location="bottom", shrink=0.85, ticks=[0, 1])
            cb.ax.set_xticklabels(["tight", "clean"]); cb.set_label("clean rock: GR < 60 API", fontsize=8)
        cb.ax.tick_params(labelsize=8)
        ax.plot(cond[:, 0], cond[:, 1], "^", ms=4, mfc="white", mec="black", mew=0.6)
        ax.set_title(t, fontsize=9.5); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        ax.set_xlabel("easting (m)", fontsize=8); ax.set_ylabel("northing (m)", fontsize=8)
fig.suptitle(f"Porosity of grid layer {k + 1} ({k * 5}-{k * 5 + 5} m below the Mississippian top); triangles: well data in this layer", fontsize=11)
fig.text(0.5, -0.015, "Easting and northing: project frame (UTM zone 14N, EPSG:32614, metres) with 0, 0 at the mean surface "
         "location of the 60 wells.", ha="center", va="top", fontsize=8)
save(fig, "04_property_maps.png")""",
"""md:| Map | What it is | Follows |
|---|---|---|
| Kriging | The best linear estimate from nearby wells, weighted by the fitted variogram; smooth. | [Matheron 1963](https://doi.org/10.2113/gsecongeo.58.8.1246); Journel & Huijbregts 1978, *Mining Geostatistics* |
| Kriging variance | How uncertain the kriging estimate is: low at wells, high far from them. | as kriging |
| SGS | Sequential Gaussian simulation: one equally likely map with the data's spread and variogram; the mean of many approaches kriging. | Deutsch & Journel 1998, *GSLIB* (2nd ed.); [Gómez-Hernández & Journel 1993](https://doi.org/10.1007/978-94-011-1739-5_8) |
| SIS | Sequential indicator simulation of a category, here clean (GR < 60 API) or tight rock. | [Journel 1983](https://doi.org/10.1007/BF01031292); Deutsch & Journel 1998, *GSLIB* |
| ML P50 | Median of a quantile regression forest trained on the cells the wells cross, with position and kriging as inputs. | Meinshausen 2006, *Quantile Regression Forests*, JMLR 7 |

All five are SubsurfiX's own Rust implementations based on these methods, not wrappers of another package.""",
"""md:## Save the model
SGS realization 1 and SIS realization 1 are used by notebooks 05 and 08.""",
"""np.savez_compressed(OUT / "property_model.npz", phi=real[0], clean=fac[0], ok=ok)
print(f"porosity mean {real[0].mean():.3f}, clean share {fac[0].mean():.2f}")""",
]

books["05_water_injection.ipynb"] = [
"""md:# 05 · Water injection

Two-phase water–oil flow on the property model of notebook 04, with the SubsurfiX finite-volume solver:
two-point flux approximation, implicit pressure and explicit saturation, gravity, Brooks–Corey capillary
pressure ([Brooks & Corey 1966](https://doi.org/10.1061/JRCEA4.0000425)) and Peaceman well indices
([Peaceman 1978](https://doi.org/10.2118/6893-PA)). One vertical injector, one deviated producer.

Example assumptions: k = 10^(−1 + 14·φ) mD, doubled in clean cells; kv/kh = 0.1.""",
SETUP_MD, SETUP,
"""md:## Build the flow model
An 8 × 8 × 6 window around the densest well cluster.""",
"""from subsurfix import _native as nat
g, mdl = np.load(OUT / "grid_upscaled.npz"), np.load(OUT / "property_model.npz")
o0, cell, d0 = tuple(g["origin"]), tuple(g["cell"]), tuple(int(v) for v in g["dims"])
i0, j0, nx, ny, nz = 9, 8, 8, 8, 6
sub = lambda a: np.asarray(a).reshape(d0[2], d0[1], d0[0])[:nz, j0:j0 + ny, i0:i0 + nx].ravel()
origin, dims = (o0[0] + i0 * cell[0], o0[1] + j0 * cell[1], 0.0), (nx, ny, nz)
phi = np.clip(sub(mdl["phi"]), 0.03, 0.30)
k_md = 10 ** (-1.0 + 14.0 * phi) * np.where(sub(mdl["clean"]) == 1, 2.0, 1.0)
k = k_md * 9.869e-16
m = nat.FlowModel.from_regular(origin, cell, dims, np.column_stack([k, k, 0.1 * k]), phi)
n = m.n_cells
print(n, "cells; pore volume", round(m.pore_volume().sum() / 1e6, 2), "million m3; median k", np.median(k_md).round(2), "mD")""",
"""md:## Wells and run
Edit the rates, pressures and fluid properties here.""",
"""cid = lambda i, j, kk: i + nx * (j + ny * kk)
inj = {"name": "INJ", "kind": "injector", "rate": 0.02, "cells": [cid(1, 1, kk) for kk in range(nz)], "ref_depth": 0.0}
cx = lambda i: origin[0] + (i + 0.5) * cell[0]
cy = lambda j: origin[1] + (j + 0.5) * cell[1]
path = np.array([[cx(6), cy(6), 0.0], [cx(6), cy(6), 5.0], [cx(4), cy(6), 28.0]])
comp = m.trajectory_completions(path, 0.1, 0.0)
prod = {"name": "PROD", "kind": "producer", "bhp": 1.5e7, "cells": list(comp["cells"]), "wi": list(comp["wi"]), "ref_depth": 0.0}
corey = {"swc": 0.2, "sor": 0.2, "nw": 2.0, "no": 2.0, "mu_w": 5e-4, "mu_o": 2e-3}
phys = {"gravity": 9.81, "rho_w": 1050.0, "rho_o": 800.0, "capillary": "brooks_corey", "pd": 5e3, "lambda": 2.0,
        "c_rock": 5e-10, "c_w": 4e-10, "c_o": 1e-9}
yr = 3.156e7
times = [1 * yr, 3 * yr, 6 * yr, 10 * yr]
p0 = 2e7 + 800 * 9.81 * m.centers()[:, 2]
r = m.run_two_phase([inj, prod], np.full(n, 0.2), p0, times, corey=corey, physics=phys, cfl=0.5, max_dt=2e6)
print("steps", r["n_steps"], "| producer completions", len(comp["cells"]))""",
"""md:## Results
Water saturation in the bottom layer, producer rates, and the mass balance check.""",
"""sw = np.array(r["sw"])
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
ext = [origin[0], origin[0] + nx * cell[0], origin[1], origin[1] + ny * cell[1]]
for ax, t, s in zip(axs, times, sw):
    im = ax.imshow(s.reshape(nz, ny, nx)[-1], origin="lower", extent=ext, vmin=0.2, vmax=0.8, cmap="Blues")
    ax.plot(*path[1:, :2].T, "r-", lw=2); ax.plot(cx(1), cy(1), "kv")
    ax.set_title(f"{t / yr:.0f} yr, Sw", fontsize=9)
fig.colorbar(im, ax=axs, shrink=0.8); save(fig, "05_sw_maps.png")""",
"""wr, orate = np.array(r["water_rate"]), np.array(r["oil_rate"])
for ph in ("water", "oil"):
    ip, ni = np.array(r[f"{ph}_in_place"]), np.array(r[f"{ph}_net_in"])
    print(f"{ph} balance error / net inflow: {(np.abs(np.diff(ip) - np.diff(ni)) / np.abs(np.diff(ni))).max():.1e}")
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(np.array(times) / yr, -orate[:, 1], "o-", label="oil"); ax.plot(np.array(times) / yr, -wr[:, 1], "s-", label="water")
ax.set_xlabel("years"); ax.set_ylabel("producer rate (m3/s)"); ax.legend(fontsize=8); save(fig, "05_rates.png")""",
]

books["06_heat_doublet_and_ogs_export.ipynb"] = [
"""md:# 06 · Heat doublet and OpenGeoSys export

Single-phase heat transport after the pressure solve: upwind advection and conduction, implicit in temperature.
First a check against the 1D solution of [Ogata & Banks (1961)](https://doi.org/10.3133/pp411A), then a cold-water
doublet injecting 30 °C water into an 80 °C reservoir for 30 years, exported as a VTU mesh for OpenGeoSys.
Properties are illustrative.""",
"md:## Setup", SETUP,
"""md:## 1D check against Ogata–Banks""",
"""from subsurfix import _native as nat
from math import erfc, exp, sqrt
n, dx, k, phi = 1000, 0.05, 1e-12, 0.2
m = nat.FlowModel.from_regular([0, 0, 0], [dx, 1, 1], [n, 1, 1], np.full((n, 3), k), np.full(n, phi))
props = dict(rho_fluid=1000.0, cp_fluid=4180.0, rho_rock=2650.0, cp_rock=900.0, cond_fluid=0.6, cond_rock=2.5)
cap = phi * 1000 * 4180 + (1 - phi) * 2650 * 900
lam = phi * 0.6 + (1 - phi) * 2.5
u = 1e-6
v, D = u * 1000 * 4180 / cap, lam / cap
t_end = 20.0 / v
wells = [{"name": "I", "kind": "injector", "cells": [0], "rate": u, "radius": 0.1}]
fixed = [nat.FixedPressure(n - 1, m.boundary_trans(n - 1, 0), 1e7, pressure_unit="Pa")]
inlet = [{"cell": 0, "conductance": lam * m.boundary_geom(0, 0), "temperature": 1.0}]
r = m.run_heat(np.full(n, 1e3), wells, [1.0], np.zeros(n), [t_end], fixed=fixed, fixed_temp=[0.0],
               fixed_t=inlet, props=props, dt_max=1e4)
x = (np.arange(n) + 0.5) * dx
s = 2 * sqrt(D * t_end)
ob = np.array([0.5 * (erfc((xi - v * t_end) / s) + exp(v * xi / D) * erfc((xi + v * t_end) / s)) if v * xi / D < 700 else 0.0 for xi in x])
err = np.abs(r["temperature"][0] - ob).max()
print(f"max abs error {err:.4f} (normalised temperature)")""",
"""md:## Doublet
Edit the rate, temperatures and permeability here.""",
"""nx = ny = 30; cell = (25.0, 25.0, 20.0)
N = nx * ny
rng = np.random.default_rng(7)
kmd = np.exp(rng.normal(np.log(300), 0.5, N))
dbl = nat.FlowModel.from_regular([0, 0, 2000], cell, [nx, ny, 1], np.column_stack([kmd * 9.869e-16] * 3), np.full(N, 0.2))
cid = lambda i, j: i + nx * j
wells = [{"name": "INJ", "kind": "injector", "cells": [cid(5, 5)], "rate": 0.02, "radius": 0.1},
         {"name": "PROD", "kind": "producer", "cells": [cid(24, 24)], "bhp": 2e7, "radius": 0.1}]
yr = 3.156e7
years = [1, 2, 5, 10, 15, 20, 25, 30]
res = dbl.run_heat(np.full(N, 1 / 5e-4), wells, [30.0, 0.0], np.full(N, 80.0), [y * yr for y in years], props=props, dt_max=0.05 * yr)
T = res["temperature"]
bal = np.abs(np.array(res["energy_change"]) - np.array(res["energy_in"])) / np.abs(res["energy_in"])
print("energy balance error max", f"{bal.max():.1e}")
print("producer temperature (degC):", dict(zip(years, T[:, cid(24, 24)].round(2))))""",
"""fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
axs[0].plot(x, ob, "k-", lw=2, label="Ogata-Banks"); axs[0].plot(x, r["temperature"][0], "r--", label="SubsurfiX")
axs[0].set_xlim(0, 40); axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("normalised T"); axs[0].legend(fontsize=8)
ext = [0, nx * cell[0], 0, ny * cell[1]]
for ax, it in zip(axs[1:], (3, 7)):
    im = ax.imshow(T[it].reshape(ny, nx), origin="lower", extent=ext, vmin=30, vmax=80, cmap="coolwarm")
    ax.plot(5.5 * 25, 5.5 * 25, "kv"); ax.plot(24.5 * 25, 24.5 * 25, "k^")
    ax.set_title(f"{years[it]} yr, temperature (degC)", fontsize=9)
fig.colorbar(im, ax=axs[1:], shrink=0.85); save(fig, "06_heat_doublet.png")""",
"""md:## Export to OpenGeoSys
Writes the mesh with the model properties, the input permeability and the final temperature as cell data. Open it in ParaView or
use it as the mesh of an OpenGeoSys project.""",
"""info = dbl.export_ogs(OUT / "doublet.vtu", cell_data={"permeability_mD": kmd, "temperature_30yr": np.asarray(T[-1])})
print(info)""",
]

books["07_wister_geothermal_well.ipynb"] = [
"""md:# 07 · Wister geothermal well 85-20

One well log and a pressure–temperature–spinner (PTS) survey from the Wister prospect, Imperial Valley,
California (DOE Geothermal Data Repository submission 904, CC BY 4.0). Download with `scripts/fetch_open_data.sh`
and set `SX_WISTER` to the `wister` folder.""",
"md:## Setup", SETUP,
"""md:## Read the log and flag washouts
Caliper more than 2 in over the bit size (or the median caliper) marks bad hole.""",
"""import lasio, glob
ROOT = Path(os.environ.get("SX_WISTER", "data_public/california/wister")).resolve()
f = sorted(glob.glob(str(ROOT / "Wister_85-20_Run1_AIT-SP-GR-CALI/*.las")))[0]
df = lasio.read(f).df().reset_index()
dep = df.columns[0]
pick = lambda *names: next((c for c in df.columns if any(c.upper().startswith(n) for n in names)), None)
gr, cal, res, bs = pick("GR"), pick("HCAL", "CALI", "C1"), pick("AT90", "AHT90", "RT", "AF90"), pick("BS", "BIT")
nom = df[bs].ffill().bfill() if bs and df[bs].notna().any() else np.nanmedian(df[cal])
bad = (df[cal] - nom) > 2.0
print("curves:", gr, cal, res, "| bad-hole share", f"{bad.mean():.1%}")""",
"""tracks = [c for c in (gr, cal, res) if c]
fig, axs = plt.subplots(1, len(tracks), figsize=(3 * len(tracks), 6), sharey=True)
for ax, c in zip(np.atleast_1d(axs), tracks):
    ax.plot(df[c], df[dep], lw=0.5, color="k"); ax.plot(df[c].where(bad), df[dep], lw=0.8, color="tab:red")
    if c == res: ax.set_xscale("log")
    ax.set_xlabel(c)
axs[0].invert_yaxis(); axs[0].set_ylabel(f"{dep} (ft)"); fig.suptitle("Wister 85-20 (red: bad hole)")
save(fig, "07_wister_tracks.png")""",
"""md:## PTS survey
`load_pts` reads every pass, converts to degC and MPa, and warns about passes without depth.""",
"""import warnings
from subsurfix.edges import PtsSource, load_pts
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pts = load_pts([PtsSource(path=str(ROOT / "Wister%2085-20%20PTS%20survey%208-30-11.xls"), well="85-20")]).to_pandas()
for w in caught: print("warning:", w.message)
fig, ax = plt.subplots(figsize=(4, 5))
for name, g in pts[pts.depth_tied].groupby("pass_name"):
    ax.plot(g.temp_c, g.md_ft, label=name)
ax.invert_yaxis(); ax.set_xlabel("temperature (degC)"); ax.set_ylabel("MD (ft)"); ax.legend(fontsize=8)
save(fig, "07_wister_pts.png")
pts.groupby("pass_name").agg(rows=("seq", "size"), max_temp_c=("temp_c", "max"))""",
]

books["08_grdecl_export.ipynb"] = [
"""md:# 08 · GRDECL export

Writes the corner-point grid of notebook 03 with the properties of notebook 04 as an Eclipse-style GRDECL file
(SPECGRID, COORD, ZCORN, ACTNUM and property keywords), then reads it back. This is the grid and properties
only, not a full simulation deck: add RUNSPEC, PROPS, SOLUTION and SCHEDULE sections yourself to run it in a
simulator such as OPM Flow.""",
"md:## Setup", SETUP,
"""md:## Build the grid and properties""",
"""from subsurfix import _native as nat
g, mdl = np.load(OUT / "grid_upscaled.npz"), np.load(OUT / "property_model.npz")
nz = int(g["dims"][2]); dz = float(g["cell"][2])
cpg = nat.CornerPointGrid.from_surfaces(g["xs"], g["ys"], np.array([g["top"] + dz * k for k in range(nz + 1)]))
phi = np.clip(mdl["phi"], 0.01, 0.35)
perm = 10 ** (-1.0 + 14.0 * phi) * np.where(mdl["clean"] == 1, 2.0, 1.0)      # mD, illustrative
props = {"PORO": phi.tolist(), "PERMX": perm.tolist(), "PERMY": perm.tolist(), "PERMZ": (0.1 * perm).tolist()}
print(cpg.dims, "cells; porosity mean", phi.mean().round(3))""",
"""md:## Write and read back""",
"""text = cpg.to_grdecl(props)
(OUT / "model.grdecl").write_text(text)
back, bprops = nat.CornerPointGrid.from_grdecl(text)
print("wrote", round(len(text) / 1e6, 2), "MB; keywords read back:", sorted(bprops))
print("geometry identical:", np.allclose(back.zcorn, cpg.zcorn) and np.allclose(back.coord, cpg.coord),
      "| PORO identical:", np.allclose(bprops["PORO"], phi, atol=1e-6))
print(text[:300])""",
]

if __name__ == "__main__":
    for name, cells in books.items():
        nbf.write(nb(cells), HERE / name)
    print("wrote", len(books))
