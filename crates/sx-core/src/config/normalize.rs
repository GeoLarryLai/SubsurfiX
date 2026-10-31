// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[normalize]` settings.

use serde::{Deserialize, Serialize};

use crate::types::LogKind;

/// Per-log normalization against neighbor or fleet anchors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NormalizeConfig {
    /// normalized logs; RHOB, DTC, RESD rejected
    #[serde(with = "crate::config::log_names")]
    pub logs: Vec<LogKind>,
    /// `layer` or `all`
    pub scope: String,
    /// inverse-distance voters (wells)
    pub ref_k: usize,
    /// distance power
    pub ref_power: f64,
    /// distance floor (m)
    pub ref_dist_floor_m: f64,
    /// below → fleet pair (wells)
    pub ref_min_voters: usize,
    /// sanity revert (×)
    pub ref_span_ratio_max: f64,
    /// second pick minimum samples (samples)
    pub pick_min_count: usize,
    /// gain cap; second-pick stretch limit (×)
    pub gain_cap: f64,
    /// extended cap (×)
    pub gain_cap_extended: f64,
    /// jury correlation for the extended cap (correlation)
    pub stretch_min_corr: f64,
    /// shift-only threshold (fraction)
    pub shift_only_frac: f64,
    /// anchor span treated as zero (native units)
    pub degenerate_span: f64,
    /// cover at or above → whole-interval (fraction)
    pub depth_max_cover: f64,
    /// layer bins per interval (bins per interval)
    pub depth_bins: usize,
    /// voters walked per target (wells)
    pub depth_candidates: usize,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            logs: vec![LogKind::Gr, LogKind::Nphi, LogKind::Pef],
            scope: "layer".to_string(),
            ref_k: 5,
            ref_power: 2.0,
            ref_dist_floor_m: 1e-6,
            ref_min_voters: 2,
            ref_span_ratio_max: 5.0,
            pick_min_count: 600,
            gain_cap: 2.0,
            gain_cap_extended: 4.0,
            stretch_min_corr: 0.3,
            shift_only_frac: 0.15,
            degenerate_span: 1e-10,
            depth_max_cover: 0.9,
            depth_bins: 20,
            depth_candidates: 61,
        }
    }
}
