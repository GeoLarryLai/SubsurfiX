// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[correlate]` graph, solve, and tops settings.

use serde::{Deserialize, Serialize};

/// Nearest-well graph used by the network solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorrelateGraphConfig {
    /// nearest wells each well links to (wells)
    pub k_graph: usize,
    /// edges longer than this are dropped (except spanning-tree edges) (m)
    pub max_edge_m: f64,
}

impl Default for CorrelateGraphConfig {
    fn default() -> Self {
        Self {
            k_graph: 6,
            max_edge_m: 8000.0,
        }
    }
}

/// Knot spacing and weights of the network solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorrelateSolveConfig {
    /// target spacing of solve knots (ft, MD)
    pub knot_step_ft: f64,
    /// spacing of path samples used as targets (grid samples)
    pub match_step: usize,
    /// smoothness weight
    pub lambda_smooth: f64,
    /// ridge weight
    pub lambda_ridge: f64,
    /// reweighting passes of the robust solve (passes)
    pub robust_passes: usize,
    /// Tukey biweight cutoff (× scale)
    pub tukey_c: f64,
}

impl Default for CorrelateSolveConfig {
    fn default() -> Self {
        Self {
            knot_step_ft: 10.0,
            match_step: 20,
            lambda_smooth: 1.0,
            lambda_ridge: 0.01,
            robust_passes: 3,
            tukey_c: 4.685,
        }
    }
}

/// Consistency check of picked tops against neighboring wells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorrelateTopsConfig {
    /// voters per top in the tops check (wells)
    pub k_check: usize,
    /// votes needed for a verdict (votes)
    pub min_votes: usize,
    /// smallest offset that makes a pick suspect (ft)
    pub tol_ft: f64,
    /// offset must also exceed this multiple of the vote spread (× spread)
    pub spread_mult: f64,
    /// share of votes on the same side of the pick (fraction)
    pub side_share: f64,
}

impl Default for CorrelateTopsConfig {
    fn default() -> Self {
        Self {
            k_check: 8,
            min_votes: 3,
            tol_ft: 10.0,
            spread_mult: 2.5,
            side_share: 0.6,
        }
    }
}

/// `[correlate]` container: graph, solve, and tops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct CorrelateConfig {
    /// nearest-well graph
    pub graph: CorrelateGraphConfig,
    /// network solve weights and knot spacing
    pub solve: CorrelateSolveConfig,
    /// tops consistency check
    pub tops: CorrelateTopsConfig,
}
