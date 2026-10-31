// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Break table and layer coordinate.
//!
//! One break rule feeds both the interval table and the coordinate, so the two
//! always agree. The same functions recompute the coordinate from updated
//! boundaries in later stages.

use std::cmp::Ordering;

use crate::types::GridRange;

/// Epsilon of the interval-of-sample rule (coordinate units, fixed).
const INTERVAL_EPS: f64 = 1e-9;

/// Break table of one well.
///
/// Breaks are the distinct kept boundary MDs, shallow to deep. A break that
/// holds a pinch stack (several boundaries at one MD) has
/// `coord_above < coord_below`.
#[derive(Debug, Clone, PartialEq)]
pub struct Breaks {
    /// MD of each break, ft, strictly increasing.
    pub md_ft: Vec<f64>,
    /// Smallest kept boundary index of each break.
    pub coord_above: Vec<u16>,
    /// Largest kept boundary index of each break.
    pub coord_below: Vec<u16>,
    /// One flag per input index: `true` if that boundary is kept. A finite
    /// boundary that is not kept is an order conflict.
    pub kept: Vec<bool>,
}

impl Breaks {
    /// `true` when the well has a layer coordinate (at least 2 breaks).
    ///
    /// NaN rule: not applicable. Errors: none. Complexity: `O(1)`.
    pub fn has_coord(&self) -> bool {
        self.md_ft.len() >= 2
    }
}

/// Build the break table from boundary MDs indexed by boundary.
///
/// Items `(md, index)` for finite MDs are sorted by `(md, index)` and grouped
/// by exactly equal MD. Walking shallow to deep, a group keeps only indices
/// greater than the last kept index; a group with none left is skipped. Each
/// kept group is one break with `coord_above` = smallest and `coord_below` =
/// largest kept index. Indices above `u16::MAX` are never kept.
///
/// Units: ft. NaN rule: skip (a non-finite MD is an absent boundary).
/// Errors: none; empty input gives no breaks. Complexity: `O(n log n)`.
pub fn build_breaks(md_by_index: &[f64]) -> Breaks {
    let mut items: Vec<(f64, u16)> = md_by_index
        .iter()
        .enumerate()
        .filter(|(_, md)| md.is_finite())
        .filter_map(|(i, &md)| u16::try_from(i).ok().map(|i| (md, i)))
        .collect();
    items.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let mut out = Breaks {
        md_ft: Vec::new(),
        coord_above: Vec::new(),
        coord_below: Vec::new(),
        kept: vec![false; md_by_index.len()],
    };
    let mut last: Option<u16> = None;
    for group in items.chunk_by(|a, b| a.0 == b.0) {
        let mut range: Option<(u16, u16)> = None;
        for &(_, idx) in group {
            if last.is_none_or(|l| idx > l) {
                if let Some(k) = out.kept.get_mut(usize::from(idx)) {
                    *k = true;
                }
                range = Some(range.map_or((idx, idx), |(lo, hi)| (lo.min(idx), hi.max(idx))));
            }
        }
        if let (Some((lo, hi)), Some(&(md, _))) = (range, group.first()) {
            out.md_ft.push(md);
            out.coord_above.push(lo);
            out.coord_below.push(hi);
            last = Some(hi);
        }
    }
    out
}

/// Layer coordinate at one MD.
///
/// NaN before the first or after the last break (no extrapolation) and when
/// the well has no coordinate; exactly at a break: `coord_below`; between
/// breaks `L` and `R`:
/// `coord_below[L] + (md − md[L]) / (md[R] − md[L]) × (coord_above[R] − coord_below[L])`.
///
/// Units: `md_ft` in ft, result in coordinate units (interval + fraction).
/// NaN rule: propagate. Errors: none. Complexity: `O(log breaks)`.
pub fn coord_at_md(breaks: &Breaks, md_ft: f64) -> f64 {
    if !breaks.has_coord() || md_ft.is_nan() {
        return f64::NAN;
    }
    let right = breaks.md_ft.partition_point(|&b| b <= md_ft);
    let Some(left) = right.checked_sub(1) else {
        return f64::NAN;
    };
    let (Some(&md_l), Some(&below_l)) = (breaks.md_ft.get(left), breaks.coord_below.get(left))
    else {
        return f64::NAN;
    };
    let below_l = f64::from(below_l);
    if md_ft == md_l {
        return below_l;
    }
    let (Some(&md_r), Some(&above_r)) = (breaks.md_ft.get(right), breaks.coord_above.get(right))
    else {
        return f64::NAN;
    };
    below_l + (md_ft - md_l) / (md_r - md_l) * (f64::from(above_r) - below_l)
}

/// Layer coordinate of every sample of `grid` (`md = 0.5 k`), by
/// [`coord_at_md`]. All NaN when the well has no coordinate.
///
/// Units: coordinate units. NaN rule: NaN where undefined. Errors: none.
/// Complexity: `O(n log breaks)`.
pub fn layer_coord(grid: GridRange, breaks: &Breaks) -> Vec<f64> {
    (0..grid.len)
        .map(|i| coord_at_md(breaks, grid.md_ft(i)))
        .collect()
}

/// Interval of a sample from its coordinate: `min(⌊u + 1e-9⌋, n_tops)`, or
/// −1 for NaN. The cap puts a sample exactly at the lower pad
/// boundary into the lower pad.
///
/// NaN rule: −1. Errors: none. Complexity: `O(1)`.
pub fn interval_of_coord(u: f64, n_tops: usize) -> i32 {
    if u.is_nan() {
        return -1;
    }
    let cap = i32::try_from(n_tops).unwrap_or(i32::MAX);
    let f = (u + INTERVAL_EPS).floor();
    if f >= f64::from(cap) {
        cap
    } else {
        f as i32
    }
}
