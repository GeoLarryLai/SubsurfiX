// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[qc]` channel tests and the jury subtable.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Closed-caliper RHOB rescue (`[qc.closed_caliper]`).
///
/// With the caliper arm closed the density pad is not pressed to the wall, so
/// RHOB reads with mud standoff (biased low, spiky). In `"rescue"` mode such
/// samples are despiked, shifted by a per-well bias and kept only where they fall
/// with an envelope of nearby wells' valid RHOB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QcClosedCalConfig {
    /// "rescue" (shift and keep inside the neighbor envelope) or "reject" (drop every closed-caliper sample)
    pub mode: String,
    /// nearest wells used for the envelope (wells)
    pub k_neighbors: usize,
    /// wells with valid RHOB needed in a depth bin before the envelope exists there (wells)
    pub min_neighbors: usize,
    /// bins a neighbor must share with the closed interval to be used (bins)
    pub neighbor_min_bins: usize,
    /// despiked closed-caliper samples with an envelope needed to estimate the well's bias (samples)
    pub min_overlap: usize,
    /// elevation bin of the envelope (ft)
    pub bin_ft: f64,
    /// envelope percentiles (percent)
    pub envelope_pct: [f64; 2],
    /// |density correction| above this is dropped before the rescue (g/cm³)
    pub drho_limit: f64,
    /// raw RHOB below this is dropped before the rescue (g/cm³)
    pub min_rhob: f64,
    /// rolling-median window of the despike (samples)
    pub despike_window: usize,
    /// rolling window of the median absolute deviation (samples)
    pub mad_window: usize,
    /// spike threshold (MAD units, MAD scaled to sigma)
    pub mad_k: f64,
    /// smallest spike threshold (g/cm³)
    pub spike_floor: f64,
}

impl Default for QcClosedCalConfig {
    fn default() -> Self {
        Self {
            mode: "rescue".to_string(),
            k_neighbors: 8,
            min_neighbors: 3,
            neighbor_min_bins: 100,
            min_overlap: 200,
            bin_ft: 2.0,
            envelope_pct: [10.0, 90.0],
            drho_limit: 0.30,
            min_rhob: 1.8,
            despike_window: 11,
            mad_window: 21,
            mad_k: 3.0,
            spike_floor: 0.05,
        }
    }
}

/// Jury that convicts a channel from neighboring wells (`[qc.jury]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QcJuryConfig {
    /// finite samples needed to sit or be judged (samples)
    pub min_count: usize,
    /// tier-1 size (wells)
    pub near_jurors: usize,
    /// candidate cap (tier 2) (wells)
    pub max_jurors: usize,
    /// candidate distance limit (m)
    pub max_dist_m: f64,
    /// contributions needed per row (jurors)
    pub quorum: usize,
    /// inverse-distance power
    pub dist_power: f64,
    /// distance floor (m)
    pub dist_floor_m: f64,
    /// below this a path is not used
    pub path_min_quality: f64,
    /// factor for layer-map jurors (×)
    pub unaligned_weight: f64,
    /// window scales (ft, MD)
    pub window_ft: [f64; 2],
    /// minimum window, samples (samples)
    pub window_min: usize,
    /// minimum hop, samples (samples)
    pub hop_min: usize,
    /// joint rows needed per window, share of W (fraction of window)
    pub joint_share: f64,
    /// joint rows floor (samples)
    pub joint_floor: usize,
    /// anti-correlation flag (correlation)
    pub anticorr_limit: f64,
    /// flat-vs-wiggle flag (fraction)
    pub flat_std_share: f64,
    /// consecutive flagged windows to convict (windows)
    pub min_streak: usize,
    /// convicted share for a whole-channel verdict (fraction)
    pub bad_share: f64,
    /// joint rows for the jury correlation (samples)
    pub corr_min_count: usize,
}

impl Default for QcJuryConfig {
    fn default() -> Self {
        Self {
            min_count: 20,
            near_jurors: 8,
            max_jurors: 20,
            max_dist_m: 30000.0,
            quorum: 3,
            dist_power: 2.0,
            dist_floor_m: 1.0,
            path_min_quality: 0.35,
            unaligned_weight: 0.35,
            window_ft: [32.0, 64.0],
            window_min: 16,
            hop_min: 8,
            joint_share: 0.6,
            joint_floor: 8,
            anticorr_limit: -0.6,
            flat_std_share: 0.1,
            min_streak: 4,
            bad_share: 0.55,
            corr_min_count: 20,
        }
    }
}

/// QC masks, fleet baselines, and the jury.
///
/// `rescue_factors` and `jury` are last so the TOML writer emits values before tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QcConfig {
    /// finite samples needed before the constant-channel test runs (samples)
    pub constant_min_count: usize,
    /// channel is constant when max−min is at most this tolerance (× (|median|+1))
    pub constant_rel_tol: f64,
    /// low and high percentiles that set the tail limits (percent)
    pub tail_pct: [f64; 2],
    /// pad beyond the percentiles before a tail sample is masked; 0 = plain clip (× (p_hi−p_lo))
    pub tail_pad_frac: f64,
    /// finite samples needed before tails are masked (samples)
    pub tail_min_count: usize,
    /// finite samples a well needs to take part in unit rescue (samples)
    pub rescue_min_count: usize,
    /// [lo, hi] level-ratio band: below lo (factor > 1) or above hi (factor < 1) makes a well a rescue candidate (× fleet median)
    pub rescue_candidate_band: [f64; 2],
    /// [lo, hi] band of normal wells and of the accepted converted level (ratio)
    pub rescue_accept_band: [f64; 2],
    /// minimum std/|mean| for a channel to count as alive (ratio)
    pub rescue_min_cv: f64,
    /// nearest normal wells used as the local reference (wells)
    pub rescue_ref_wells: usize,
    /// normal wells needed to use a local reference (else fleet median) (wells)
    pub rescue_min_ref_wells: usize,
    /// shortest exact-repeat run that is masked (samples)
    pub repeat_min_len: usize,
    /// max−min allowed inside a repeat run (× (|median|+1))
    pub repeat_rel_tol: f64,
    /// finite samples needed before repeats are tested (samples)
    pub repeat_min_count: usize,
    /// rolling-std window (samples)
    pub plateau_window: usize,
    /// finite values a window needs to give a std (samples)
    pub plateau_window_min: usize,
    /// shortest quiet run that is masked (samples)
    pub plateau_min_len: usize,
    /// finite samples needed before plateaus are tested (also at least 3 × window) (samples)
    pub plateau_min_count: usize,
    /// rolling std below this share of the channel std is quiet (× channel std)
    pub plateau_std_share: f64,
    /// finite runs shorter than this are masked (the longest run never is) (samples)
    pub run_min_len: usize,
    /// finite samples needed for a channel stats row (samples)
    pub stats_min_count: usize,
    /// percentiles giving p_low and p_high (also the normalization anchors) (percent)
    pub stats_pct: [f64; 2],
    /// wells needed for any fleet summary or fleet baseline (wells)
    pub fleet_min_wells: usize,
    /// floor for MAD_n (native units)
    pub mad_floor: f64,
    /// std below this ratio of the fleet median std = low variance (channel blanked) (× fleet median std)
    pub lowvar_ratio: f64,
    /// post-normalization flat check ratio (× median std)
    pub post_flat_ratio: f64,
    /// finite values a well needs in the post-normalization flat check (samples)
    pub post_flat_min_count: usize,
    /// ring-test deviation limit (MAD units)
    pub dev_limit: f64,
    /// nearest wells in the local baseline (wells)
    pub near_count: usize,
    /// minimum near wells for a local baseline (else fleet baseline) (wells)
    pub near_min: usize,
    /// first rank of the outer ring in the two-scale check (rank, 1-based)
    pub ring_first: usize,
    /// last rank (inclusive) of the outer ring (rank, 1-based)
    pub ring_last: usize,
    /// minimum wells in the outer ring (wells)
    pub ring_min: usize,
    /// [lo, hi] band for the median ratio in the unit-mismatch flag (ratio)
    pub unit_band: [f64; 2],
    /// k of the k-NN divergence estimate (neighbors)
    pub jsd_k: usize,
    /// nearest scoreable wells per divergence score (wells)
    pub jsd_neighbors: usize,
    /// neighbors needed for a valid score (wells)
    pub jsd_min_neighbors: usize,
    /// valid wells needed before outliers are flagged (wells)
    pub jsd_min_wells: usize,
    /// per-well subsample cap (samples)
    pub jsd_max_count: usize,
    /// outlier threshold multiplier (MAD units)
    pub jsd_mad_mult: f64,
    /// floor for k-NN distances and for the MAD (z units)
    pub jsd_floor: f64,
    /// a flagged GR channel whose span is below this share of the reference span is blanked (fraction)
    pub gr_span_frac: f64,
    /// RESD median offset from the reference that blanks the channel (log10 ohm·m)
    pub resd_decade_log10: f64,
    /// shortest saturation pileup that is masked (samples)
    pub pileup_min_len: usize,
    /// minimum share of the channel at the pileup value (fraction)
    pub pileup_min_share: f64,
    /// equality tolerance of a pileup (× max(max|v|, 1))
    pub pileup_rel_tol: f64,
    /// logs masked where the borehole is bad (caliper / density correction); empty switches the bad-hole rule off
    pub badhole_logs: Vec<String>,
    /// caliper below this reads as closed or cased hole (in)
    pub cal_open_min_in: f64,
    /// caliper this far below the well's nominal hole size is undergauge (in)
    pub cal_undergauge_in: f64,
    /// caliper this far above the well's nominal hole size is washout (in)
    pub cal_washout_in: f64,
    /// |density correction| above this is a bad pad contact (g/cm³)
    pub drho_limit: f64,
    /// RHOB below this is "low density": dropped only where caliper / DRHO show bad hole, kept in gauge (coal), flagged in the report where no caliper / DRHO exists (g/cm³; 0 = off)
    pub low_density_rhob: f64,
    /// unit-rescue factor per log name (key = log name); an empty table switches rescue off (× multiplier)
    pub rescue_factors: BTreeMap<String, f64>,
    /// plausible sample range per log name, linear unit; samples outside are masked (empty table = off)
    pub plausible_range: BTreeMap<String, [f64; 2]>,
    /// plausible channel median per log name, linear unit; a channel outside is blanked (empty table = off)
    pub plausible_median: BTreeMap<String, [f64; 2]>,
    /// neighboring-well jury
    pub jury: QcJuryConfig,
    /// closed-caliper RHOB rescue
    pub closed_caliper: QcClosedCalConfig,
}

impl Default for QcConfig {
    fn default() -> Self {
        let mut rescue_factors = BTreeMap::new();
        rescue_factors.insert("GR".to_string(), 14.0);
        rescue_factors.insert("NPHI".to_string(), 100.0);
        Self {
            constant_min_count: 5,
            constant_rel_tol: 1e-9,
            tail_pct: [0.25, 99.75],
            tail_pad_frac: 0.1,
            tail_min_count: 5,
            rescue_min_count: 100,
            rescue_candidate_band: [0.2, 5.0],
            rescue_accept_band: [0.5, 2.0],
            rescue_min_cv: 0.25,
            rescue_ref_wells: 6,
            rescue_min_ref_wells: 4,
            repeat_min_len: 20,
            repeat_rel_tol: 0.001,
            repeat_min_count: 5,
            plateau_window: 20,
            plateau_window_min: 10,
            plateau_min_len: 20,
            plateau_min_count: 60,
            plateau_std_share: 0.02,
            run_min_len: 20,
            stats_min_count: 30,
            stats_pct: [10.0, 90.0],
            fleet_min_wells: 3,
            mad_floor: 1e-10,
            lowvar_ratio: 0.1,
            post_flat_ratio: 0.1,
            post_flat_min_count: 30,
            dev_limit: 3.0,
            near_count: 16,
            near_min: 5,
            ring_first: 17,
            ring_last: 64,
            ring_min: 5,
            unit_band: [0.2, 5.0],
            jsd_k: 5,
            jsd_neighbors: 10,
            jsd_min_neighbors: 5,
            jsd_min_wells: 5,
            jsd_max_count: 50000,
            jsd_mad_mult: 3.5,
            jsd_floor: 1e-12,
            gr_span_frac: 0.2,
            resd_decade_log10: 0.7,
            pileup_min_len: 20,
            pileup_min_share: 0.01,
            pileup_rel_tol: 1e-8,
            badhole_logs: vec!["RHOB".to_string(), "PEF".to_string()],
            cal_open_min_in: 5.0,
            cal_undergauge_in: 1.0,
            cal_washout_in: 2.0,
            drho_limit: 0.15,
            low_density_rhob: 1.65,
            rescue_factors,
            plausible_range: [
                ("RHOB".to_string(), [1.0, 3.10]),
                ("PEF".to_string(), [1.0, 7.0]),
            ]
            .into_iter()
            .collect(),
            plausible_median: [
                ("RHOB".to_string(), [2.0, 2.9]),
                ("PEF".to_string(), [1.8, 6.0]),
            ]
            .into_iter()
            .collect(),
            jury: QcJuryConfig::default(),
            closed_caliper: QcClosedCalConfig::default(),
        }
    }
}
