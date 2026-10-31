// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Banded DTW with path output.
//!
//! One banded DTW engine ([`banded_dtw`]) and three ways to feed it:
//!
//! - interval mode ([`interval_path`]): one interval, layer
//!   coordinate axis, unit-scaled channels, linear fallback;
//! - full mode ([`full_path`], new design): one interval between the current
//!   boundaries, MD axis, GR-first capped robust cost;
//! - window mode ([`window_path`], new design): a window whose ends only are
//!   pinned.
//!
//! Path helpers turn node pairs into grid pairs ([`pairs_from_md`]) and sample
//! them ([`map_a_to_b`], [`map_b_to_a`]). Depths are feet, grid indices are
//! 0.5 ft steps, missing values are NaN (skipped in resampling and costs,
//! `None` in sampling). No function allocates a dense `N × N` matrix.

mod cost;
mod engine;
mod modes;
mod params;
mod path;
mod resample;

pub use cost::{shared_nodes, unit_cost};
pub use engine::{band_half_width, banded_dtw, banded_dtw_dense, DtwPath};
pub use modes::{
    full_path, interval_path, interval_plan, window_path, IntervalMatch, IntervalPlan,
    IntervalSide, MatchMethod, SpanSide,
};
pub use params::{IntervalParams, SpanParams};
pub use path::{map_a_to_b, map_b_to_a, md_to_k, pairs_from_md, row_mean_map};
pub use resample::{resample_curve, rescale_unit, working_values};
