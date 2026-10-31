// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Hand-written `project.toml` template. Values are the section defaults.

/// Commented template with every section at its default (4.8).
///
/// `crs` is `EPSG:32614`. The text loads with no data files. Units in the
/// comments are the units of each key (ft, m, fractions, counts).
pub fn template_toml() -> String {
    TEMPLATE.to_string()
}

const TEMPLATE: &str = r#"# SubsurfiX project file. Every key shown has its default value.
# Delete any key or section you do not need; a file with only [project] crs works.
seed = 0  # global seed mixed into every random stream

[project]
crs = "EPSG:32614"  # CRS of the project frame, any PROJ/EPSG text, e.g. "EPSG:32614"; read by Python, only checked non-empty here

# Runtime thread pool.
[runtime]
threads = 0  # size of the global rayon pool; 0 = all available cores; never hashed (threads)

# Edge sources.
[edges]
allow_network = false  # allow PROJ to download datum grids (Python edge only)
allow_ballpark = false  # allow ballpark datum transforms (Python edge only)
# [[edges.headers]]                 # one or more
# path = "headers.xlsx"             # required
# kb_unit = "ft"                    # required
# sheet = "Sheet1"                  # optional
# delimiter = ","
# crs = "EPSG:32613"                # or xy_unit = "m" | "ft" | "us-ft" (exactly one)
# key_col = ["UWI"]                 # optional overrides of the default candidate lists
# x_col = ["Surface X", "X"]
# y_col = ["Surface Y", "Y"]
# label_col = ["Name", "well_name", "Well Name"]
# kb_col = []
# kb_match = "datum value"
# datum_col = ["Well datum name", "Datum"]
# lat_col = ["Latitude", "Lat", "LAT"]   # optional hints that settle LAS header/file-name conflicts
# lon_col = ["Longitude", "Lon", "Long", "LON"]
# location_col = ["Location", "Legal Location", "STR"]
# td_col = ["Depth_stop", "TD", "Total Depth", "Driller TD"]
# [[edges.tops]]                    # zero or more (none: one flat datum)
# path = "tops.csv"
# depth_ref = "md"                  # required
# depth_unit = "ft"                 # required
# layout = "long"
# repeat_pick = "shallowest"
# # sheet, delimiter, key_col, top_col, depth_col, top_columns, exclude_columns: optional
# [[edges.surveys]]                 # zero or more
# path = "surveys.xlsx"
# md_unit = "ft"                    # required
# # sheet, delimiter, key_col, md_col, inc_col, azi_col: optional
# [[edges.pts]]                     # zero or more: pressure-temperature-spinner surveys
# path = "pts.xlsx"                 # CSV, XLSX, or XLS (needs python-calamine)
# well = "15007012340000"           # well key (or key_col = "UWI")
# # sheets, delimiter, temp_unit (degC/degF/K), pressure_unit (psia/psig/kPa/MPa/bar), depth_unit (ft/m): optional
# [edges.area]                      # optional
# path = "area.shp"
# need_tops = true
# # crs, layer: optional

# Ingest.
[ingest]
las_dirs = []  # LAS folders in priority order; relative paths are relative to the project folder (paths)
wells = []  # only these wells; empty = every selected well (well names)
exclude_wells = []  # wells to skip (well names)
# alias_dictionary = "aliases.csv"
max_gap_ft = 2.0  # widest source gap bridged by interpolation (ft)
top_names = []  # top order to use; empty = order from the data (top names)
las_match = "header_first"  # how a LAS file finds its well: "header_first" (UWI/API/WELL in the LAS header, then the file name) or "file_name"

# Extra ingest mnemonics.
[ingest.alias_extra]
# GR = ["GAMMA", "GRC"]

# Manual LAS-to-well assignments: file name (no folder) = well key. Always wins
# over header-ID and file-name matching; use it for files the match report excludes.
[ingest.las_overrides]
# "15007235520000.las" = "15007235520000"

# QC.
[qc]
constant_min_count = 5  # finite samples needed before the constant-channel test runs (samples)
constant_rel_tol = 1e-9  # channel is constant when max−min is at most this tolerance (× (|median|+1))
tail_pct = [0.25, 99.75]  # low and high percentiles that set the tail limits (percent)
tail_pad_frac = 0.1  # pad beyond the percentiles before a tail sample is masked; 0 = plain clip (× (p_hi−p_lo))
tail_min_count = 5  # finite samples needed before tails are masked (samples)
rescue_min_count = 100  # finite samples a well needs to take part in unit rescue (samples)
rescue_candidate_band = [0.2, 5.0]  # [lo, hi] level-ratio band: below lo (factor > 1) or above hi (factor < 1) makes a well a rescue candidate (× fleet median)
rescue_accept_band = [0.5, 2.0]  # [lo, hi] band of normal wells and of the accepted converted level (ratio)
rescue_min_cv = 0.25  # minimum std/|mean| for a channel to count as alive (ratio)
rescue_ref_wells = 6  # nearest normal wells used as the local reference (wells)
rescue_min_ref_wells = 4  # normal wells needed to use a local reference (else fleet median) (wells)
repeat_min_len = 20  # shortest exact-repeat run that is masked (samples)
repeat_rel_tol = 0.001  # max−min allowed inside a repeat run (× (|median|+1))
repeat_min_count = 5  # finite samples needed before repeats are tested (samples)
plateau_window = 20  # rolling-std window (samples)
plateau_window_min = 10  # finite values a window needs to give a std (samples)
plateau_min_len = 20  # shortest quiet run that is masked (samples)
plateau_min_count = 60  # finite samples needed before plateaus are tested (also at least 3 × window) (samples)
plateau_std_share = 0.02  # rolling std below this share of the channel std is quiet (× channel std)
run_min_len = 20  # finite runs shorter than this are masked (the longest run never is) (samples)
stats_min_count = 30  # finite samples needed for a channel stats row (samples)
stats_pct = [10.0, 90.0]  # percentiles giving p_low and p_high (also the normalization anchors) (percent)
fleet_min_wells = 3  # wells needed for any fleet summary or fleet baseline (wells)
mad_floor = 1e-10  # floor for MAD_n (native units)
lowvar_ratio = 0.1  # std below this ratio of the fleet median std = low variance (channel blanked) (× fleet median std)
post_flat_ratio = 0.1  # post-normalization flat check ratio (× median std)
post_flat_min_count = 30  # finite values a well needs in the post-normalization flat check (samples)
dev_limit = 3.0  # ring-test deviation limit (MAD units)
near_count = 16  # nearest wells in the local baseline (wells)
near_min = 5  # minimum near wells for a local baseline (else fleet baseline) (wells)
ring_first = 17  # first rank of the outer ring in the two-scale check (rank, 1-based)
ring_last = 64  # last rank (inclusive) of the outer ring (rank, 1-based)
ring_min = 5  # minimum wells in the outer ring (wells)
unit_band = [0.2, 5.0]  # [lo, hi] band for the median ratio in the unit-mismatch flag (ratio)
jsd_k = 5  # k of the k-NN divergence estimate (neighbors)
jsd_neighbors = 10  # nearest scoreable wells per divergence score (wells)
jsd_min_neighbors = 5  # neighbors needed for a valid score (wells)
jsd_min_wells = 5  # valid wells needed before outliers are flagged (wells)
jsd_max_count = 50000  # per-well subsample cap (samples)
jsd_mad_mult = 3.5  # outlier threshold multiplier (MAD units)
jsd_floor = 1e-12  # floor for k-NN distances and for the MAD (z units)
gr_span_frac = 0.2  # a flagged GR channel whose span is below this share of the reference span is blanked (fraction)
resd_decade_log10 = 0.7  # RESD median offset from the reference that blanks the channel (log10 ohm·m)
pileup_min_len = 20  # shortest saturation pileup that is masked (samples)
pileup_min_share = 0.01  # minimum share of the channel at the pileup value (fraction)
pileup_rel_tol = 1e-8  # equality tolerance of a pileup (× max(max|v|, 1))
badhole_logs = ["RHOB", "PEF", "NPHI"]  # logs masked where the borehole is bad (caliper / density correction); empty switches the bad-hole rule off
cal_open_min_in = 5.0  # caliper below this reads as closed or cased hole (in)
cal_undergauge_in = 1.0  # caliper this far below the well's nominal hole size is undergauge (in)
cal_washout_in = 2.0  # caliper this far above the well's nominal hole size is washout (in)
drho_limit = 0.15  # |density correction| above this is a bad pad contact (g/cm³)
low_density_rhob = 1.65  # RHOB below this is dropped only in bad hole, kept in gauge (coal), counted as unverified without caliper / DRHO (g/cm³; 0 = off)

# QC unit-rescue factors.
[qc.rescue_factors]
GR = 14.0
NPHI = 100.0

# QC plausible sample range per log (linear unit); samples outside are masked.
[qc.plausible_range]
PEF = [1.0, 7.0]
RHOB = [1.0, 3.1]

# QC plausible channel median per log (linear unit); a channel outside is blanked.
[qc.plausible_median]
PEF = [1.8, 6.0]
RHOB = [2.0, 2.9]

# QC closed-caliper RHOB rescue. Arm closed = pad not on the wall = standoff-biased RHOB.
[qc.closed_caliper]
mode = "rescue"  # "rescue" (shift and keep inside the neighbor envelope) or "reject" (drop every closed-caliper sample)
k_neighbors = 8  # nearest wells used for the envelope (wells)
min_neighbors = 3  # wells with valid RHOB needed in a depth bin before the envelope exists there (wells)
neighbor_min_bins = 100  # bins a neighbor must share with the closed interval to be used (bins)
min_overlap = 200  # despiked closed-caliper samples with an envelope needed to estimate the well's bias (samples)
bin_ft = 2.0  # elevation bin of the envelope (ft)
envelope_pct = [10.0, 90.0]  # envelope percentiles (percent)
drho_limit = 0.3  # |density correction| above this is dropped before the rescue (g/cm³)
min_rhob = 1.8  # raw RHOB below this is dropped before the rescue (g/cm³)
despike_window = 11  # rolling-median window of the despike (samples)
mad_window = 21  # rolling window of the median absolute deviation (samples)
mad_k = 3.0  # spike threshold (MAD units, MAD scaled to sigma)
spike_floor = 0.05  # smallest spike threshold (g/cm³)

# QC jury.
[qc.jury]
min_count = 20  # finite samples needed to sit or be judged (samples)
near_jurors = 8  # tier-1 size (wells)
max_jurors = 20  # candidate cap (tier 2) (wells)
max_dist_m = 30000.0  # candidate distance limit (m)
quorum = 3  # contributions needed per row (jurors)
dist_power = 2.0  # inverse-distance power
dist_floor_m = 1.0  # distance floor (m)
path_min_quality = 0.35  # below this a path is not used
unaligned_weight = 0.35  # factor for layer-map jurors (×)
window_ft = [32.0, 64.0]  # window scales (ft (MD))
window_min = 16  # minimum window, samples (samples)
hop_min = 8  # minimum hop, samples (samples)
joint_share = 0.6  # joint rows needed per window, share of W (fraction of window)
joint_floor = 8  # joint rows floor (samples)
anticorr_limit = -0.6  # anti-correlation flag (correlation)
flat_std_share = 0.1  # flat-vs-wiggle flag (fraction)
min_streak = 4  # consecutive flagged windows to convict (windows)
bad_share = 0.55  # convicted share for a whole-channel verdict (fraction)
corr_min_count = 20  # joint rows for the jury correlation (samples)

# Align.
[align]
study_top = ""  # first top of the study range; "" = first canonical top
study_base = ""  # last top of the study range; "" = last canonical top
pad_ft = 200.0  # distance of the upper and lower pads from the first and last top (MD) (ft (MD))
k_pairs = 8  # donor wells matched to each target well (wells)

# Align tops.
[align.tops]
k = 12  # nearest wells that vote for a missing top (wells)
max_dist_m = 30000.0  # voter distance limit (m)
quorum = 3  # voters needed to estimate a top (wells)
power = 2.0  # distance power of the vote weight
min_dist_m = 1.0  # distance floor of the vote weight (m)
min_own = 2  # own picks needed before a well accepts estimates (picks)

# Align pairs.
[align.pairs]
min_span_ratio = 0.25  # smallest accepted thickness ratio donor/target
max_span_ratio = 4.0  # largest accepted thickness ratio
depth_nodes = 512  # nodes of the elevation-matched DTW (nodes)
depth_band_frac = 0.25  # band of the elevation-matched DTW (fraction of nodes)
depth_min_shared = 64  # shared nodes needed (else the pair fails) (nodes)
depth_quality = 0.2  # quality assigned to an elevation-matched pair
fail_quality = 0.0  # quality stored for a failed pair
fail_quality_no_coord = 0.15  # quality stored for a failed pair without coordinates

# Align confidence.
[align.confidence]
observed = 1.0  # confidence of an observed interval
pinched_user = 1.0  # confidence of a pinch between user picks
pinched_other = 0.6  # confidence of any other pinch
estimated_one = 0.75  # confidence with one estimated boundary
estimated_none = 0.5  # confidence with no usable boundary pair
pad_floor = 0.5  # floor of a pad confidence

# DTW interval.
[dtw.interval]
grid_budget = 512  # total node budget shared by the intervals of a pair (nodes per pair)
min_nodes = 24  # fewest nodes per interval (nodes)
max_nodes = 128  # most nodes per interval (nodes)
min_shared_floor = 8  # floor of the shared-node requirement (nodes)
min_shared_cap = 64  # cap of the shared-node requirement (nodes)
min_shared_div = 4  # nodes divided by this give the shared-node requirement
band_frac = 0.12  # DTW band as a fraction of the node count (fraction of `N`)
gap_cost = 1.5  # cost of a cell with no log finite at both nodes (cost units)
min_channel_samples = 8  # whole-well finite samples a log needs (samples)
min_channel_nodes = 8  # finite nodes a log needs in an interval (nodes)
max_gap_steps = 4.0  # widest NaN gap bridged when resampling (node steps)
pct_low = 1.0  # low percentile of the linear rescale (percent)
pct_high = 99.0  # high percentile of the linear rescale (percent)
min_span = 1e-9  # smallest percentile span kept (working units)
dtw_factor = 1.0  # quality factor of a DTW match
linear_factor = 0.45  # quality factor of a linear match
resd_floor_ohmm = 1e-6  # floor applied before log10 of RESD (ohm·m)

# DTW full span.
[dtw.full]
min_nodes = 24  # fewest resampled nodes (nodes)
max_nodes = 2000  # most resampled nodes (nodes)
band_frac = 0.12  # DTW band as a fraction of the node count (fraction of `N`)
gr_weight = 0.5  # cost weight of GR when valid (other logs share the rest)
z_cap = 3.0  # cap of the robust z in the local cost (robust z)
mad_factor = 1.482602218505602  # MAD to sigma factor (see 4.9)
scale_floor = 1e-6  # floor of the pooled robust scale (working units)
max_gap_steps = 4.0  # widest NaN gap bridged when resampling (node steps)
min_gap_ft = 1.0  # smallest gap width bridged (ft)
resd_floor_ohmm = 1e-6  # floor applied before log10 of RESD (ohm·m)

# DTW window span.
[dtw.window]
min_nodes = 24  # fewest resampled nodes (nodes)
max_nodes = 2000  # most resampled nodes (nodes)
band_frac = 0.25  # DTW band as a fraction of the node count (fraction of `N`)
gr_weight = 0.5  # cost weight of GR when valid (other logs share the rest)
z_cap = 3.0  # cap of the robust z in the local cost (robust z)
mad_factor = 1.482602218505602  # MAD to sigma factor (see 4.9)
scale_floor = 1e-6  # floor of the pooled robust scale (working units)
max_gap_steps = 4.0  # widest NaN gap bridged when resampling (node steps)
min_gap_ft = 1.0  # smallest gap width bridged (ft)
resd_floor_ohmm = 1e-6  # floor applied before log10 of RESD (ohm·m)

# Normalize.
[normalize]
logs = ["GR", "NPHI", "PEF"]  # normalized logs; RHOB, DTC, RESD rejected
scope = "layer"  # `layer` or `all`
ref_k = 5  # inverse-distance voters (wells)
ref_power = 2.0  # distance power
ref_dist_floor_m = 1e-6  # distance floor (m)
ref_min_voters = 2  # below → fleet pair (wells)
ref_span_ratio_max = 5.0  # sanity revert (×)
pick_min_count = 600  # second pick minimum samples (samples)
gain_cap = 2.0  # gain cap; second-pick stretch limit (×)
gain_cap_extended = 4.0  # extended cap (×)
stretch_min_corr = 0.3  # jury correlation for the extended cap (correlation)
shift_only_frac = 0.15  # shift-only threshold (fraction)
degenerate_span = 1e-10  # anchor span treated as zero (native units)
depth_max_cover = 0.9  # cover at or above → whole-interval (fraction)
depth_bins = 20  # layer bins per interval (bins per interval)
depth_candidates = 61  # voters walked per target (wells)
nphi_offset = true  # shift wells whose NPHI departs from the field NPHI - DPHI offset in clean tight rock
nphi_offset_gr_max = 30.0  # GR at or below this marks clean rock for the NPHI offset (API)
nphi_offset_rhob_min = 2.6  # RHOB at or above this marks tight rock for the NPHI offset (g/cm³)
nphi_offset_rho_matrix = 2.71  # matrix density of the density porosity (g/cm³)
nphi_offset_rho_fluid = 1.0  # fluid density of the density porosity (g/cm³)
nphi_offset_min_count = 30  # clean tight samples a well needs for an offset (samples)
nphi_offset_limit = 0.04  # departure from the field offset that triggers the shift (v/v)
nphi_shift = 0.0  # manual field-wide shift subtracted from NPHI (v/v); nonzero overrides an evaluated offset

# Field-wide NPHI offset (`subsurfix nphi-offset`); applied in normalize once evaluated.
[nphi_offset]
apply = true  # subtract the evaluated offset from NPHI in normalize when an evaluation is stored
gr_max = 30.0  # clean rock: GR at or below (API)
phi_max = 0.05  # tight rock: neutron-density porosity (NPHI + DPHI_ls) / 2 at or below (v/v)
pef_min = 4.0  # limestone PEF window, lower bound, where PEF is measured (b/e)
pef_max = 6.0  # limestone PEF window, upper bound, where PEF is measured (b/e)
require_pef = false  # reject samples without PEF
rho_matrix = 2.71  # matrix density of the density porosity (g/cm³)
rho_fluid = 1.0  # fluid density of the density porosity (g/cm³)
min_well_samples = 30  # qualifying samples a well needs to count (samples)
min_wells = 5  # qualifying wells needed for a field estimate (wells)
n_boot = 1000  # bootstrap resamples of the wells
ci_level = 0.9  # two-sided confidence level of the interval
outlier_limit = 0.04  # well offsets further than this from the field offset are outliers (v/v)
fit_phi_max = 0.25  # trend fit: clean limestone up to this neutron-density porosity (v/v)
fit_bin = 0.01  # trend fit: porosity bin width (v/v)
slope_tol = 0.1  # largest |slope| of (NPHI - DPHI_ls) vs neutron-density porosity accepted as a constant offset
n_boot_slope = 200  # bootstrap resamples for the slope interval
max_plot_samples = 20000  # qualifying samples written for plots (samples)

# Neighbors.
[neighbors]
knn_list_len = 30  # nearest wells examined, target included (wells (target included))
search_cap = 20  # wells in the capped donor list (wells)
k_primary = 5  # donors in the tier-1 stack (wells)
batch_size = 1  # donors added per step in the later tiers (wells per step)
quorum = 3  # contributors needed to accept a row in tiers 1 and 2 (contributors)
max_dist_m = 30000.0  # donors farther than this are removed (first pass) (m)
min_dist_m = 1.0  # distance floor in the donor weight (m)
idw_power = 2.0  # distance power in the donor weight
min_finite_donor = 5  # finite measured samples a donor needs for a log (samples)
donor_max_bracket_ft = 8.0  # widest MD spacing between the two measured samples that bracket a donor sample (ft (MD))
interval_eps = 1e-9  # epsilon of the interval-of-coordinate test (layer-coordinate units)
iq_no_coord = 0.2  # interval quality of an elevation-matched row
aq_unaligned = 0.35  # alignment quality when no pair path exists
conf_observed = 1.0  # confidence of an observed interval
conf_estimated_one_user = 0.75  # confidence of an interval with one estimated boundary
conf_estimated_no_user = 0.5  # confidence of an interval with no user boundary
conf_pad_floor = 0.5  # confidence floor of a pad interval
validate_split_ft = 8.0  # MD gap that starts a new piece in the stored-curve check (ft (MD))
validate_min_run = 20  # finite values a piece needs to be checked (samples)
flat_rel_tol = 1e-10  # flat-piece tolerance (× max(max|v|, 1))
saturation_frac = 0.5  # share of a piece at a physical bound that blanks the piece (fraction)
saturation_rel_tol = 1e-6  # tolerance of "at a bound" (× max(|hi − lo|, 1))
range_map_logs = ["GR", "NPHI", "PEF"]  # logs whose neighbor curve is range-mapped to the reference
range_map_min_n = 20  # finite values needed to range-map a curve (samples)
range_p_low = 10.0  # low percentile of the range map (percent)
range_p_high = 90.0  # high percentile of the range map (percent)
max_gain = 2.0  # gain clamp of the range map (1/max_gain .. max_gain)
min_span_frac = 0.15  # shift-only threshold of the range map
trusted_min_n = 3  # contributors needed for a trusted value (contributors)
trusted_max_dist_m = 30000.0  # farthest donor distance for a trusted value (m)
trusted_min_quality = 0.25  # minimum quality of a trusted value

# Fill.
[fill]
order = ["GR", "DTC", "RHOB", "NPHI", "RESD", "PEF"]  # order in which target logs are modeled (each log once)
min_train_rows = 100  # training rows needed to build a model for a log (rows)
mostly_predicted_share = 0.5  # a well's log is flagged mostly_predicted in fill_summary.csv above this filled share (fraction)
dropout_frac = 0.25  # share of training rows copied with feature dropout (fraction of rows)
dropout_layer_prob = 0.5  # probability that a dropout copy also loses the layer coordinate (probability)
local_windows_ft = [2.5, 10.0]  # the two half-windows of the depth-local deviation features (ft (half-window, MD))
local_min_count = 3  # finite values needed in a local window (values)
range_map_logs = ["GR", "RHOB", "NPHI", "DTC", "RESD", "PEF"]  # logs whose model output is range-mapped to the neighbor curve
range_min_n = 30  # finite values needed to range-map a well (values)
range_p_low = 10.0  # low percentile of the range map (percent)
range_p_high = 90.0  # high percentile of the range map (percent)
max_gain = 2.0  # gain clamp of the range map
min_span_frac = 0.15  # shift-only threshold of the range map
degenerate_span = 1e-10  # prediction span treated as zero (log unit)
flat_min_run = 20  # shortest run tested by the pre-write rejection (rows)
near_const_rel = 0.0001  # near-constant stretch relative tolerance (× |v_i|)
near_const_abs = 1e-9  # near-constant stretch absolute floor (log unit)
plateau_window = 20  # rolling-std window of the soft-plateau test (rows)
plateau_min_count = 10  # finite values a plateau window needs (values)
plateau_std_frac = 0.05  # rolling std below this share of the scale is a plateau (× scale)
quiet_std_frac = 0.08  # neighbor std below this share of the scale is quiet (× scale)
quiet_min_count = 5  # finite neighbor values needed to call a stretch quiet (values)
exact_flat_tol = 1e-10  # max−min at or below this is an exact flat (log unit)
saturation_frac = 0.5  # share of a run at a bound that rejects the run (fraction)
saturation_rel_tol = 1e-6  # tolerance of "at a bound" (× max(|hi − lo|, 1))
line_min_rows = 5  # shortest exact-line stretch that is rejected (rows)
line_rel_tol = 1e-5  # second-difference tolerance of an exact line (× max(|v|, 1))
scale_floor = 1e-6  # floor of the rejection scale (log unit)
holdout_n_wells = 5  # held-out wells picked by the fill scorer (wells)
holdout_min_samples = 80  # measured samples a well needs to be held out (samples)
holdout_mode = "interval"  # scorer masking mode: "interval" or "whole_log" (`interval` or `whole_log`)
holdout_frac = 0.3  # share of measured rows masked in interval mode (fraction)
holdout_blocks = 5  # target block count in interval mode (blocks)
holdout_min_block = 30  # smallest masked block (rows)
holdout_max_attempts = 200  # attempts to place masked blocks (attempts)

# Fill LightGBM.
[fill.lightgbm]
num_iterations = 400  # boosting rounds (trees)
learning_rate = 0.05  # shrinkage per round
max_depth = 6  # tree depth limit (levels)
num_leaves = 63  # leaves per tree (leaves)
feature_fraction = 0.8  # share of features tried per tree (fraction)
bagging_fraction = 1.0  # share of rows per bagging round (fraction)
bagging_freq = 0  # bagging every k rounds (0 = off) (iterations)
lambda_l1 = 0.0  # L1 regularization
lambda_l2 = 1.5  # L2 regularization
min_gain_to_split = 0.0  # smallest split gain
min_data_in_leaf = 200  # rows per leaf (rows)
min_sum_hessian_in_leaf = 0.001  # hessian sum per leaf
max_bin = 255  # histogram bins of a normal feature (bins)
max_bin_wide = 1023  # histogram bins of the wide-bin features (bins)
wide_bin_features = ["elev_ft", "layer_coord"]  # feature names that get max_bin_wide bins
bin_construct_sample_cnt = 2000000  # rows sampled to build bins (rows)
num_threads = 8  # LightGBM threads (not the rayon pool) (threads)
objective = "regression"  # "regression" (L2), "huber" or "quantile"
alpha = 0.5  # Huber delta or quantile level (used by "huber" and "quantile")

# Fill ladder.
[fill.ladder]
real_donor_max_n = 20  # donor pool size of the RealDonor copy (donors)
donor_max_dist_m = 30000.0  # donor distance limit of the donor steps (m)
donor_max_n = 12  # donor pool size of the DonorShape fill (donors)
anchor_rows = 5  # rows averaged into an anchor (rows)
max_hole_rows = 4  # longest interior NaN stretch that is hole-filled (rows)
tilt_max_scale = 3.0  # largest allowed anchor offset (× scale)
anchor_tol_scale = 1.5  # anchor distance within which the guess is used untilted (× scale)
physical_max_out_frac = 0.05  # share outside bounds above which a candidate is rejected (fraction)
exact_flat_min_rows = 20  # shortest exact-flat candidate rejected by the guard (rows)
guard_quiet_frac = 0.05  # candidate std below this share of the scale is rejected (× scale)
crosslog_min_rows = 200  # fit rows needed by the CrossLogFit step (rows)
crosslog_min_r2 = 0.5  # smallest in-sample R² accepted
crosslog_window_ft = 300.0  # fit window around the gap (ft (MD))
crosslog_min_std = 1e-9  # smallest predictor std (predictor unit)
texture_window = 20  # centered window of the own-texture step (rows)
scale_min_measured = 10  # measured values a well needs for its own scale (values)
scale_fallback = 1.0  # scale when no well qualifies (log unit)

# Correlate graph.
[correlate.graph]
k_graph = 6  # nearest wells each well links to (wells)
max_edge_m = 8000.0  # edges longer than this are dropped (except spanning-tree edges) (m)

# Correlate solve.
[correlate.solve]
knot_step_ft = 10.0  # target spacing of solve knots (ft (MD))
match_step = 20  # spacing of path samples used as targets (grid samples)
lambda_smooth = 1.0  # smoothness weight
lambda_ridge = 0.01  # ridge weight
robust_passes = 3  # reweighting passes of the robust solve (passes)
tukey_c = 4.685  # Tukey biweight cutoff (× scale)

# Correlate tops.
[correlate.tops]
k_check = 8  # voters per top in the tops check (wells)
min_votes = 3  # votes needed for a verdict (votes)
tol_ft = 10.0  # smallest offset that makes a pick suspect (ft)
spread_mult = 2.5  # offset must also exceed this multiple of the vote spread (× spread)
side_share = 0.6  # share of votes on the same side of the pick (fraction)

# Subhorizons.
[subhorizons]
n_grid = 200  # samples per interval on the layer-time grid (samples per interval)
min_wells = 3  # wells needed to build a regional curve (wells)
min_segment_samples = 40  # shortest segment kept (grid samples)
scales = [1, 2, 4, 8, 16]  # wavelet scales, ascending (grid samples)
kernel_half_width_mult = 5  # kernel half-width = mult × scale (× scale)
link_tol_mult = 0.5  # crossing link tolerance (× next scale)
min_persistence = 3  # scales a crossing must persist through (1 ..= len(scales)) (scales)
min_strength_api = 10.0  # smallest crossing strength (API)
edge_gap_samples = 10  # margin kept clear at both interval ends (grid samples)
min_spacing_samples = 8  # smallest spacing between kept proposals (grid samples)

# Refresh.
[refresh]
refresh_tol_ft = 5.0  # a well is refreshed when its percentile movement is strictly above this (ft (MD))
percentile = 95.0  # percentile of per-sample movement used as the well score (percent)

# Export.
[export]
las = true  # write one LAS file per well
parquet = true  # copy the final Parquet products into export/

[propagation]
method = "qrf"  # model: "qrf" (quantile regression forest, default) or "lightgbm" (quantile LightGBM)
quantiles = [0.1, 0.5, 0.9]  # predicted quantile levels, each in (0, 1)
qrf_trees = 200  # QRF trees
qrf_min_leaf = 10  # QRF minimum rows per leaf
lgbm_iterations = 300  # LightGBM boosting rounds
lgbm_conformal = true  # widen LightGBM bands by cross-conformal CQR
qrf_conformal = false  # widen QRF bands by cross-conformal CQR
conformal_folds = 5  # folds for the conformal scores (well folds when well ids exist)
"#;
