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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_a() {
        let x = [1000.2, 1000.7, 1001.2, 1004.0, 1004.5];
        let y = [10.0, 20.0, 30.0, 50.0, 60.0];
        let grid = grid_range_for(1000.2, 1004.5).unwrap();
        assert_eq!(
            grid,
            GridRange {
                k_start: 2001,
                len: 9
            }
        );
        let t: Vec<f64> = (0..grid.len).map(|i| grid.md_ft(i)).collect();
        assert_eq!(t[0], 1000.5);
        assert_eq!(t[8], 1004.5);
        let got = interp_gap(&t, &x, &y, 2.0);
        let want = [
            16.0,
            26.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            50.0,
            60.0,
        ];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            if w.is_nan() {
                assert!(g.is_nan(), "{g}");
            } else {
                assert!((g - w).abs() < 1e-9, "{g} vs {w}");
            }
        }
        // A wider limit bridges the 2.8 ft gap.
        let wide = interp_gap(&[1002.0], &x, &y, 3.0);
        assert!((wide[0] - (30.0 + 0.8 * 20.0 / 2.8)).abs() < 1e-9);
    }

    #[test]
    fn interp_edges() {
        let x = [10.0, 11.0];
        let y = [1.0, 3.0];
        let g = |t: f64| interp_gap(&[t], &x, &y, 2.0)[0];
        assert!(g(9.0).is_nan());
        assert!(g(12.0).is_nan());
        assert!(g(f64::NAN).is_nan());
        assert_eq!(g(10.0), 1.0);
        assert_eq!(g(10.0 - 5e-7), 1.0);
        assert_eq!(g(11.0 + 5e-7), 3.0);
        assert_eq!(g(10.5), 2.0);
        assert!(interp_gap(&[1.0], &[], &[], 2.0)[0].is_nan());
        // A single sample answers only on itself.
        assert_eq!(interp_gap(&[5.0], &[5.0], &[7.0], 2.0)[0], 7.0);
        assert!(interp_gap(&[5.5], &[5.0], &[7.0], 2.0)[0].is_nan());
        // A gap of exactly max_gap is bridged.
        assert_eq!(interp_gap(&[11.0], &[10.0, 12.0], &[0.0, 4.0], 2.0)[0], 2.0);
        assert!(interp_gap(&[11.0], &[10.0, 12.1], &[0.0, 4.0], 2.0)[0].is_nan());
    }

    #[test]
    fn grid_range_cases() {
        assert_eq!(
            grid_range_for(1000.0, 1001.0),
            Some(GridRange {
                k_start: 2000,
                len: 3
            })
        );
        // No grid point between 1000.1 and 1000.4.
        assert_eq!(grid_range_for(1000.1, 1000.4), None);
        assert_eq!(grid_range_for(f64::NAN, 1.0), None);
        assert_eq!(grid_range_for(0.0, f64::INFINITY), None);
        assert_eq!(
            grid_range_for(0.0, 0.0),
            Some(GridRange { k_start: 0, len: 1 })
        );
        assert_eq!(grid_range_for(0.0, 1.0e15), None);
        // A sample a hair below a grid point counts as on it.
        assert_eq!(
            grid_range_for(999.999_999_9, 1000.0),
            Some(GridRange {
                k_start: 2000,
                len: 1
            })
        );
    }

    #[test]
    fn resample_empty_and_values() {
        let grid = GridRange::new(2000, 3).unwrap();
        let none = resample_log(grid, &[], &[], 2.0);
        assert!(none.iter().all(|v| v.is_nan()) && none.len() == 3);
        let got = resample_log(grid, &[1000.0, 1001.0], &[2.4, 2.5], 2.0);
        assert!((got[0] - 2.4).abs() < 1e-6);
        assert!((got[1] - 2.45).abs() < 1e-6);
        assert!((got[2] - 2.5).abs() < 1e-6);
    }
}
