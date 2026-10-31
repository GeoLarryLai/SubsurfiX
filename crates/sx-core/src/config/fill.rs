// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[fill]` model, LightGBM, and ladder settings.

use serde::{Deserialize, Serialize};

use crate::types::LogKind;

/// LightGBM hyperparameters for the gap-fill model (`[fill.lightgbm]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FillLightgbmConfig {
    /// boosting rounds (trees)
    pub num_iterations: usize,
    /// shrinkage per round
    pub learning_rate: f64,
    /// tree depth limit (levels)
    pub max_depth: usize,
    /// leaves per tree (leaves)
    pub num_leaves: usize,
    /// share of features tried per tree (fraction)
    pub feature_fraction: f64,
    /// share of rows per bagging round (fraction)
    pub bagging_fraction: f64,
    /// bagging every k rounds (0 = off) (iterations)
    pub bagging_freq: usize,
    /// L1 regularization
    pub lambda_l1: f64,
    /// L2 regularization
    pub lambda_l2: f64,
    /// smallest split gain
    pub min_gain_to_split: f64,
    /// rows per leaf (rows)
    pub min_data_in_leaf: usize,
    /// hessian sum per leaf
    pub min_sum_hessian_in_leaf: f64,
    /// histogram bins of a normal feature (bins)
    pub max_bin: usize,
    /// histogram bins of the wide-bin features (bins)
    pub max_bin_wide: usize,
    /// feature names that get max_bin_wide bins
    pub wide_bin_features: Vec<String>,
    /// rows sampled to build bins (rows)
    pub bin_construct_sample_cnt: usize,
    /// LightGBM threads (not the rayon pool) (threads)
    pub num_threads: usize,
    /// LightGBM objective: `regression` (L2, default), `huber`, or `quantile`.
    pub objective: String,
    /// Quantile level for `objective = "quantile"` (0.5 = median).
    pub alpha: f64,
}

impl Default for FillLightgbmConfig {
    fn default() -> Self {
        Self {
            num_iterations: 400,
            learning_rate: 0.05,
            max_depth: 6,
            num_leaves: 63,
            feature_fraction: 0.8,
            bagging_fraction: 1.0,
            bagging_freq: 0,
            lambda_l1: 0.0,
            lambda_l2: 1.5,
            min_gain_to_split: 0.0,
            min_data_in_leaf: 200,
            min_sum_hessian_in_leaf: 0.001,
            max_bin: 255,
            max_bin_wide: 1023,
            wide_bin_features: vec!["elev_ft".to_string(), "layer_coord".to_string()],
            bin_construct_sample_cnt: 2_000_000,
            num_threads: 8,
            objective: "regression".to_string(),
            alpha: 0.5,
        }
    }
}

/// Shaped-fill ladder (`[fill.ladder]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FillLadderConfig {
    /// donor pool size of the RealDonor copy (donors)
    pub real_donor_max_n: usize,
    /// donor distance limit of the donor steps (m)
    pub donor_max_dist_m: f64,
    /// donor pool size of the DonorShape fill (donors)
    pub donor_max_n: usize,
    /// rows averaged into an anchor (rows)
    pub anchor_rows: usize,
    /// longest interior NaN stretch that is hole-filled (rows)
    pub max_hole_rows: usize,
    /// largest allowed anchor offset (× scale)
    pub tilt_max_scale: f64,
    /// anchor distance within which the guess is used untilted (× scale)
    pub anchor_tol_scale: f64,
    /// share outside bounds above which a candidate is rejected (fraction)
    pub physical_max_out_frac: f64,
    /// shortest exact-flat candidate rejected by the guard (rows)
    pub exact_flat_min_rows: usize,
    /// candidate std below this share of the scale is rejected (× scale)
    pub guard_quiet_frac: f64,
    /// fit rows needed by the CrossLogFit step (rows)
    pub crosslog_min_rows: usize,
    /// smallest in-sample R² accepted
    pub crosslog_min_r2: f64,
    /// fit window around the gap (ft, MD)
    pub crosslog_window_ft: f64,
    /// smallest predictor std (predictor unit)
    pub crosslog_min_std: f64,
    /// centered window of the own-texture step (rows)
    pub texture_window: usize,
    /// measured values a well needs for its own scale (values)
    pub scale_min_measured: usize,
    /// scale when no well qualifies (log unit)
    pub scale_fallback: f64,
}

impl Default for FillLadderConfig {
    fn default() -> Self {
        Self {
            real_donor_max_n: 20,
            donor_max_dist_m: 30000.0,
            donor_max_n: 12,
            anchor_rows: 5,
            max_hole_rows: 4,
            tilt_max_scale: 3.0,
            anchor_tol_scale: 1.5,
            physical_max_out_frac: 0.05,
            exact_flat_min_rows: 20,
            guard_quiet_frac: 0.05,
            crosslog_min_rows: 200,
            crosslog_min_r2: 0.5,
            crosslog_window_ft: 300.0,
            crosslog_min_std: 1e-9,
            texture_window: 20,
            scale_min_measured: 10,
            scale_fallback: 1.0,
        }
    }
}

/// Gap-fill model order, range map, rejection, and holdout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FillConfig {
    /// order in which target logs are modeled (each log once)
    #[serde(with = "crate::config::log_names")]
    pub order: Vec<LogKind>,
    /// training rows needed to build a model for a log (rows)
    pub min_train_rows: usize,
    /// a well's log is flagged mostly_predicted in fill_summary.csv above this filled share (fraction)
    pub mostly_predicted_share: f64,
    /// share of training rows copied with feature dropout (fraction of rows)
    pub dropout_frac: f64,
    /// probability that a dropout copy also loses the layer coordinate (probability)
    pub dropout_layer_prob: f64,
    /// the two half-windows of the depth-local deviation features (ft, half-window, MD)
    pub local_windows_ft: Vec<f64>,
    /// finite values needed in a local window (values)
    pub local_min_count: usize,
    /// logs whose model output is range-mapped to the neighbor curve
    #[serde(with = "crate::config::log_names")]
    pub range_map_logs: Vec<LogKind>,
    /// finite values needed to range-map a well (values)
    pub range_min_n: usize,
    /// low percentile of the range map (percent)
    pub range_p_low: f64,
    /// high percentile of the range map (percent)
    pub range_p_high: f64,
    /// gain clamp of the range map
    pub max_gain: f64,
    /// shift-only threshold of the range map
    pub min_span_frac: f64,
    /// prediction span treated as zero (log unit)
    pub degenerate_span: f64,
    /// shortest run tested by the pre-write rejection (rows)
    pub flat_min_run: usize,
    /// near-constant stretch relative tolerance (× |v_i|)
    pub near_const_rel: f64,
    /// near-constant stretch absolute floor (log unit)
    pub near_const_abs: f64,
    /// rolling-std window of the soft-plateau test (rows)
    pub plateau_window: usize,
    /// finite values a plateau window needs (values)
    pub plateau_min_count: usize,
    /// rolling std below this share of the scale is a plateau (× scale)
    pub plateau_std_frac: f64,
    /// neighbor std below this share of the scale is quiet (× scale)
    pub quiet_std_frac: f64,
    /// finite neighbor values needed to call a stretch quiet (values)
    pub quiet_min_count: usize,
    /// max−min at or below this is an exact flat (log unit)
    pub exact_flat_tol: f64,
    /// share of a run at a bound that rejects the run (fraction)
    pub saturation_frac: f64,
    /// tolerance of "at a bound" (× max(|hi − lo|, 1))
    pub saturation_rel_tol: f64,
    /// shortest exact-line stretch that is rejected (rows)
    pub line_min_rows: usize,
    /// second-difference tolerance of an exact line (× max(|v|, 1))
    pub line_rel_tol: f64,
    /// floor of the rejection scale (log unit)
    pub scale_floor: f64,
    /// held-out wells picked by the fill scorer (wells)
    pub holdout_n_wells: usize,
    /// measured samples a well needs to be held out (samples)
    pub holdout_min_samples: usize,
    /// scorer masking mode: "interval" or "whole_log" (`interval` or `whole_log`)
    pub holdout_mode: String,
    /// share of measured rows masked in interval mode (fraction)
    pub holdout_frac: f64,
    /// target block count in interval mode (blocks)
    pub holdout_blocks: usize,
    /// smallest masked block (rows)
    pub holdout_min_block: usize,
    /// attempts to place masked blocks (attempts)
    pub holdout_max_attempts: usize,
    /// LightGBM hyperparameters
    pub lightgbm: FillLightgbmConfig,
    /// shaped-fill ladder
    pub ladder: FillLadderConfig,
}

impl Default for FillConfig {
    fn default() -> Self {
        Self {
            order: vec![
                LogKind::Gr,
                LogKind::Dtc,
                LogKind::Rhob,
                LogKind::Nphi,
                LogKind::Resd,
                LogKind::Pef,
            ],
            min_train_rows: 100,
            mostly_predicted_share: 0.5,
            dropout_frac: 0.25,
            dropout_layer_prob: 0.5,
            local_windows_ft: vec![2.5, 10.0],
            local_min_count: 3,
            range_map_logs: vec![
                LogKind::Gr,
                LogKind::Rhob,
                LogKind::Nphi,
                LogKind::Dtc,
                LogKind::Resd,
                LogKind::Pef,
            ],
            range_min_n: 30,
            range_p_low: 10.0,
            range_p_high: 90.0,
            max_gain: 2.0,
            min_span_frac: 0.15,
            degenerate_span: 1e-10,
            flat_min_run: 20,
            near_const_rel: 0.0001,
            near_const_abs: 1e-9,
            plateau_window: 20,
            plateau_min_count: 10,
            plateau_std_frac: 0.05,
            quiet_std_frac: 0.08,
            quiet_min_count: 5,
            exact_flat_tol: 1e-10,
            saturation_frac: 0.5,
            saturation_rel_tol: 1e-6,
            line_min_rows: 5,
            line_rel_tol: 1e-5,
            scale_floor: 1e-6,
            holdout_n_wells: 5,
            holdout_min_samples: 80,
            holdout_mode: "interval".to_string(),
            holdout_frac: 0.3,
            holdout_blocks: 5,
            holdout_min_block: 30,
            holdout_max_attempts: 200,
            lightgbm: FillLightgbmConfig::default(),
            ladder: FillLadderConfig::default(),
        }
    }
}
