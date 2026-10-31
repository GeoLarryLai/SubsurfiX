// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Small numeric utilities.
//!
//! Linear interpolation, percentiles, median, weighted median, MAD, regression
//! metrics, cross-validation splits, a standard scaler, isotonic regression,
//! and seeded random helpers. Each item documents its NaN rule and empty-input
//! rule.

#![warn(missing_docs)]

pub mod cv;
pub mod interp;
pub mod isotonic;
pub mod metrics;
pub mod rng;
pub mod scaler;
pub mod stats;

pub use cv::{kfold_by_well, spatial_block_folds, Fold, WellXY};
pub use interp::{interp_linear, OutOfRange};
pub use isotonic::isotonic_increasing;
pub use metrics::{regression_metrics, regression_metrics_f32, Metrics};
pub use scaler::StandardScaler;
pub use stats::{
    mad, median, median_f32, percentile, percentile_f32, percentiles, percentiles_f32,
    robust_scale, weighted_median, MAD_TO_SIGMA,
};
