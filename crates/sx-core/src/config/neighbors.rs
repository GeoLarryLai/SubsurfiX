// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[neighbors]` settings.

use serde::{Deserialize, Serialize};

use crate::types::LogKind;

/// Donor search, interval quality, and range-map settings for neighbor curves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeighborsConfig {
    /// nearest wells examined, target included (wells, target included)
    pub knn_list_len: usize,
    /// wells in the capped donor list (wells)
    pub search_cap: usize,
    /// donors in the tier-1 stack (wells)
    pub k_primary: usize,
    /// donors added per step in the later tiers (wells per step)
    pub batch_size: usize,
    /// contributors needed to accept a row in tiers 1 and 2 (contributors)
    pub quorum: usize,
    /// donors farther than this are removed (first pass) (m)
    pub max_dist_m: f64,
    /// distance floor in the donor weight (m)
    pub min_dist_m: f64,
    /// distance power in the donor weight
    pub idw_power: f64,
    /// finite measured samples a donor needs for a log (samples)
    pub min_finite_donor: usize,
    /// widest MD spacing between the two measured samples that bracket a donor sample (ft, MD)
    pub donor_max_bracket_ft: f64,
    /// epsilon of the interval-of-coordinate test (layer-coordinate units)
    pub interval_eps: f64,
    /// interval quality of an elevation-matched row
    pub iq_no_coord: f64,
    /// alignment quality when no pair path exists
    pub aq_unaligned: f64,
    /// confidence of an observed interval
    pub conf_observed: f64,
    /// confidence of an interval with one estimated boundary
    pub conf_estimated_one_user: f64,
    /// confidence of an interval with no user boundary
    pub conf_estimated_no_user: f64,
    /// confidence floor of a pad interval
    pub conf_pad_floor: f64,
    /// MD gap that starts a new piece in the stored-curve check (ft, MD)
    pub validate_split_ft: f64,
    /// finite values a piece needs to be checked (samples)
    pub validate_min_run: usize,
    /// flat-piece tolerance (× max(max|v|, 1))
    pub flat_rel_tol: f64,
    /// share of a piece at a physical bound that blanks the piece (fraction)
    pub saturation_frac: f64,
    /// tolerance of "at a bound" (× max(|hi − lo|, 1))
    pub saturation_rel_tol: f64,
    /// logs whose neighbor curve is range-mapped to the reference
    #[serde(with = "crate::config::log_names")]
    pub range_map_logs: Vec<LogKind>,
    /// finite values needed to range-map a curve (samples)
    pub range_map_min_n: usize,
    /// low percentile of the range map (percent)
    pub range_p_low: f64,
    /// high percentile of the range map (percent)
    pub range_p_high: f64,
    /// gain clamp of the range map (1/max_gain .. max_gain)
    pub max_gain: f64,
    /// shift-only threshold of the range map
    pub min_span_frac: f64,
    /// contributors needed for a trusted value (contributors)
    pub trusted_min_n: usize,
    /// farthest donor distance for a trusted value (m)
    pub trusted_max_dist_m: f64,
    /// minimum quality of a trusted value
    pub trusted_min_quality: f64,
}

impl Default for NeighborsConfig {
    fn default() -> Self {
        Self {
            knn_list_len: 30,
            search_cap: 20,
            k_primary: 5,
            batch_size: 1,
            quorum: 3,
            max_dist_m: 30000.0,
            min_dist_m: 1.0,
            idw_power: 2.0,
            min_finite_donor: 5,
            donor_max_bracket_ft: 8.0,
            interval_eps: 1e-9,
            iq_no_coord: 0.2,
            aq_unaligned: 0.35,
            conf_observed: 1.0,
            conf_estimated_one_user: 0.75,
            conf_estimated_no_user: 0.5,
            conf_pad_floor: 0.5,
            validate_split_ft: 8.0,
            validate_min_run: 20,
            flat_rel_tol: 1e-10,
            saturation_frac: 0.5,
            saturation_rel_tol: 1e-6,
            range_map_logs: vec![LogKind::Gr, LogKind::Nphi, LogKind::Pef],
            range_map_min_n: 20,
            range_p_low: 10.0,
            range_p_high: 90.0,
            max_gain: 2.0,
            min_span_frac: 0.15,
            trusted_min_n: 3,
            trusted_max_dist_m: 30000.0,
            trusted_min_quality: 0.25,
        }
    }
}
