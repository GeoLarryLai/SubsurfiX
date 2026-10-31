// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[subhorizons]` settings.

use serde::{Deserialize, Serialize};

/// Regional sub-horizon proposal settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubhorizonsConfig {
    /// samples per interval on the layer-time grid (samples per interval)
    pub n_grid: usize,
    /// wells needed to build a regional curve (wells)
    pub min_wells: usize,
    /// shortest segment kept (grid samples)
    pub min_segment_samples: usize,
    /// wavelet scales, ascending (grid samples)
    pub scales: Vec<usize>,
    /// kernel half-width = mult × scale (× scale)
    pub kernel_half_width_mult: usize,
    /// crossing link tolerance (× next scale)
    pub link_tol_mult: f64,
    /// scales a crossing must persist through (1 ..= len(scales)) (scales)
    pub min_persistence: usize,
    /// smallest crossing strength (API)
    pub min_strength_api: f64,
    /// margin kept clear at both interval ends (grid samples)
    pub edge_gap_samples: usize,
    /// smallest spacing between kept proposals (grid samples)
    pub min_spacing_samples: usize,
}

impl Default for SubhorizonsConfig {
    fn default() -> Self {
        Self {
            n_grid: 200,
            min_wells: 3,
            min_segment_samples: 40,
            scales: vec![1, 2, 4, 8, 16],
            kernel_half_width_mult: 5,
            link_tol_mult: 0.5,
            min_persistence: 3,
            min_strength_api: 10.0,
            edge_gap_samples: 10,
            min_spacing_samples: 8,
        }
    }
}
