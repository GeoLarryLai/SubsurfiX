// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[align]` rough-alignment settings.

use serde::{Deserialize, Serialize};

/// Vote for a missing top from nearby wells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TopsParams {
    /// nearest wells that vote for a missing top (wells)
    pub k: usize,
    /// voter distance limit (m)
    pub max_dist_m: f64,
    /// voters needed to estimate a top (wells)
    pub quorum: usize,
    /// distance power of the vote weight
    pub power: f64,
    /// distance floor of the vote weight (m)
    pub min_dist_m: f64,
    /// own picks needed before a well accepts estimates (picks)
    pub min_own: usize,
}

impl Default for TopsParams {
    fn default() -> Self {
        Self {
            k: 12,
            max_dist_m: 30000.0,
            quorum: 3,
            power: 2.0,
            min_dist_m: 1.0,
            min_own: 2,
        }
    }
}

/// Elevation-matched pair search between a target and its donors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PairParams {
    /// smallest accepted thickness ratio donor/target
    pub min_span_ratio: f64,
    /// largest accepted thickness ratio
    pub max_span_ratio: f64,
    /// nodes of the elevation-matched DTW (nodes)
    pub depth_nodes: usize,
    /// band of the elevation-matched DTW (fraction of nodes)
    pub depth_band_frac: f64,
    /// shared nodes needed (else the pair fails) (nodes)
    pub depth_min_shared: usize,
    /// quality assigned to an elevation-matched pair
    pub depth_quality: f32,
    /// quality stored for a failed pair
    pub fail_quality: f32,
    /// quality stored for a failed pair without coordinates
    pub fail_quality_no_coord: f32,
}

impl Default for PairParams {
    fn default() -> Self {
        Self {
            min_span_ratio: 0.25,
            max_span_ratio: 4.0,
            depth_nodes: 512,
            depth_band_frac: 0.25,
            depth_min_shared: 64,
            depth_quality: 0.2,
            fail_quality: 0.0,
            fail_quality_no_coord: 0.15,
        }
    }
}

/// Confidence stored on each kind of aligned interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfidenceParams {
    /// confidence of an observed interval
    pub observed: f32,
    /// confidence of a pinch between user picks
    pub pinched_user: f32,
    /// confidence of any other pinch
    pub pinched_other: f32,
    /// confidence with one estimated boundary
    pub estimated_one: f32,
    /// confidence with no usable boundary pair
    pub estimated_none: f32,
    /// floor of a pad confidence
    pub pad_floor: f32,
}

impl Default for ConfidenceParams {
    fn default() -> Self {
        Self {
            observed: 1.0,
            pinched_user: 1.0,
            pinched_other: 0.6,
            estimated_one: 0.75,
            estimated_none: 0.5,
            pad_floor: 0.5,
        }
    }
}

/// Study range, donor count, and the tops / pairs / confidence subtables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AlignParams {
    /// first top of the study range; "" = first canonical top
    pub study_top: String,
    /// last top of the study range; "" = last canonical top
    pub study_base: String,
    /// distance of the upper and lower pads from the first and last top (MD) (ft, MD)
    pub pad_ft: f64,
    /// donor wells matched to each target well (wells)
    pub k_pairs: usize,
    /// missing-top vote
    pub tops: TopsParams,
    /// donor/target pair search
    pub pairs: PairParams,
    /// interval confidence values
    pub confidence: ConfidenceParams,
}

impl Default for AlignParams {
    fn default() -> Self {
        Self {
            study_top: String::new(),
            study_base: String::new(),
            pad_ft: 200.0,
            k_pairs: 8,
            tops: TopsParams::default(),
            pairs: PairParams::default(),
            confidence: ConfidenceParams::default(),
        }
    }
}
