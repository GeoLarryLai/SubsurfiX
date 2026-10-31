// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[refresh]` settings.

use serde::{Deserialize, Serialize};

/// A well is refreshed when its percentile movement is strictly above this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RefreshConfig {
    /// a well is refreshed when its percentile movement is strictly above this (ft, MD)
    pub refresh_tol_ft: f64,
    /// percentile of per-sample movement used as the well score (percent)
    pub percentile: f64,
}

impl Default for RefreshConfig {
    fn default() -> Self {
        Self {
            refresh_tol_ft: 5.0,
            percentile: 95.0,
        }
    }
}
