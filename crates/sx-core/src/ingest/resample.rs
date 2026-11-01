// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Gap-aware interpolation, grid range and resampling onto the MD grid
//!.

use crate::types::{GridRange, GRID_STEP_FT};

/// Distance (ft) within which a target counts as sitting on a source sample.
const EPS: f64 = 1e-6;

/// Linear interpolation that refuses to bridge wide gaps.
///
/// `x` must be strictly ascending with a finite `y` for every entry (callers
/// drop NaN pairs first); `t` may be in any order. Per target: empty `x`, or
/// `t` outside `[x[0] - EPS, x[n-1] + EPS]` gives NaN; a sample within `EPS` of
/// `t` gives its value (closest, tie to the lower one); otherwise the gap
/// `x[j] - x[j-1]` around `t` must be `<= max_gap_ft`, else NaN. Units: feet
/// for `t`, `x`, `max_gap_ft`; `y` in any unit. Mismatched `x`/`y` lengths use
/// the shorter. O(m log n) for m targets.
pub(crate) fn interp_gap(t: &[f64], x: &[f64], y: &[f64], max_gap_ft: f64) -> Vec<f64> {
    let n = x.len().min(y.len());
    let x = x.get(..n).unwrap_or(&[]);
    let y = y.get(..n).unwrap_or(&[]);
    t.iter().map(|&t| interp_one(t, x, y, max_gap_ft)).collect()
}

fn interp_one(t: f64, x: &[f64], y: &[f64], max_gap_ft: f64) -> f64 {
    let (Some(&first), Some(&last)) = (x.first(), x.last()) else {
        return f64::NAN;
    };
    if t.is_nan() || t < first - EPS || t > last + EPS {
        return f64::NAN;
    }
    // First index whose x is >= t.
    let j = x.partition_point(|&v| v < t);
    let lo = j.checked_sub(1);
    let d_lo = lo.and_then(|i| x.get(i)).map_or(f64::INFINITY, |&v| t - v);
    let d_hi = x.get(j).map_or(f64::INFINITY, |&v| v - t);
    if d_lo <= EPS || d_hi <= EPS {
        let pick = if d_lo <= d_hi { lo } else { Some(j) };
        return pick.and_then(|i| y.get(i)).copied().unwrap_or(f64::NAN);
    }
    let Some(i0) = lo else {
        return f64::NAN;
    };
    let (Some(&x0), Some(&x1), Some(&y0), Some(&y1)) = (x.get(i0), x.get(j), y.get(i0), y.get(j))
    else {
        return f64::NAN;
    };
    if x1 - x0 > max_gap_ft {
        return f64::NAN;
    }
    y0 + (t - x0) * (y1 - y0) / (x1 - x0)
}

/// Grid range covering `[md_lo_ft, md_hi_ft]`.
///
/// `k_start = ceil((lo - EPS) / 0.5)`, `k_end = floor((hi + EPS) / 0.5)`.
/// `None` when no grid point lies inside, when an input is not finite, or when
/// the indices do not fit the `i32` grid index.
pub(crate) fn grid_range_for(md_lo_ft: f64, md_hi_ft: f64) -> Option<GridRange> {
    if !md_lo_ft.is_finite() || !md_hi_ft.is_finite() {
        return None;
    }
    let k_start = ((md_lo_ft - EPS) / GRID_STEP_FT).ceil();
    let k_end = ((md_hi_ft + EPS) / GRID_STEP_FT).floor();
    let limit = f64::from(i32::MAX) / 2.0;
    if k_end < k_start || k_start.abs() > limit || k_end.abs() > limit {
        return None;
    }
    let len = (k_end - k_start) as i64 + 1;
    GridRange::new(k_start as i32, usize::try_from(len).ok()?).ok()
}

/// Resample one log onto `grid`: [`interp_gap`] at every grid MD, stored as
/// `f32`. No samples gives all NaN. Samples are `(md_ft ascending strictly,
/// value)`.
pub(crate) fn resample_log(
    grid: GridRange,
    md_ft: &[f64],
    values: &[f32],
    max_gap_ft: f64,
) -> Vec<f32> {
    if md_ft.is_empty() {
        return vec![f32::NAN; grid.len];
    }
    let t: Vec<f64> = (0..grid.len).map(|i| grid.md_ft(i)).collect();
    let y: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    interp_gap(&t, md_ft, &y, max_gap_ft)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}
