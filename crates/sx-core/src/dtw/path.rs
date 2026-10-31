// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Node maps, grid pairs and path sampling.

use crate::dtw::engine::DtwPath;
use crate::error::{SxError, SxResult};
use crate::types::GRID_STEP_FT;

/// Node map of a DTW path.
///
/// `mapped[i]` = mean column index `j` of the path cells in row `i` (f64, in
/// node steps), then a running maximum so the map is non-decreasing, then the
/// ends are pinned: `mapped[0] = 0` and `mapped[n_rows − 1]` = the last cell's
/// column. A row with no cell takes the running maximum.
///
/// NaN rule: not applicable (integer input). Errors: none; `n_rows = 0` gives
/// an empty vector. Complexity: `O(cells + n_rows)`.
pub fn row_mean_map(path: &DtwPath, n_rows: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; n_rows];
    let mut counts = vec![0usize; n_rows];
    for &(i, j) in &path.cells {
        if let (Some(s), Some(c)) = (sums.get_mut(i), counts.get_mut(i)) {
            *s += j as f64;
            *c += 1;
        }
    }
    let mut mapped = Vec::with_capacity(n_rows);
    let mut running = 0.0f64;
    for (s, &c) in sums.iter().zip(counts.iter()) {
        let mean = if c > 0 { s / c as f64 } else { running };
        running = running.max(mean);
        mapped.push(running);
    }
    if let Some(first) = mapped.first_mut() {
        *first = 0.0;
    }
    if let (Some(last), Some(&(_, end_j))) = (mapped.last_mut(), path.cells.last()) {
        *last = end_j as f64;
    }
    mapped
}

/// Grid index of a measured depth: `k = ⌊md_ft / 0.5 + 0.5⌋` (round half up)
///
/// `md_ft` in feet. NaN rule: error. Errors: [`SxError::InvalidInput`] if
/// `md_ft` is not finite or `k` does not fit `i32`. Complexity: `O(1)`.
pub fn md_to_k(md_ft: f64) -> SxResult<i32> {
    if !md_ft.is_finite() {
        return Err(SxError::InvalidInput(format!(
            "measured depth must be finite, got {md_ft}"
        )));
    }
    let k = (md_ft / GRID_STEP_FT + 0.5).floor();
    if k < f64::from(i32::MIN) || k > f64::from(i32::MAX) {
        return Err(SxError::InvalidInput(format!(
            "measured depth {md_ft} ft is outside the i32 grid index range"
        )));
    }
    Ok(k as i32)
}

/// Convert `(md_a_ft, md_b_ft)` pairs to grid pairs `(k_a, k_b)`
///
/// Depths in feet; order is kept; a pair equal to the previous kept pair is
/// dropped. NaN rule: error. Errors: [`SxError::InvalidInput`] for a
/// non-finite or out-of-range depth. Complexity: `O(n)`.
pub fn pairs_from_md(pairs_md: &[(f64, f64)]) -> SxResult<Vec<(i32, i32)>> {
    let mut out: Vec<(i32, i32)> = Vec::with_capacity(pairs_md.len());
    for &(md_a, md_b) in pairs_md {
        let pair = (md_to_k(md_a)?, md_to_k(md_b)?);
        if out.last() != Some(&pair) {
            out.push(pair);
        }
    }
    Ok(out)
}

/// Sample a path at grid index `k_a`.
///
/// `path` holds `(k_a, k_b)` pairs, non-decreasing in both. Keys are the
/// distinct `k_a` values, each with the `k_b` of its first pair. An exact key
/// returns its `k_b`; between consecutive keys the result is linear; below the
/// first key, above the last key, or NaN input returns `None`.
///
/// Units: grid indices (0.5 ft steps). NaN rule: `None`. Errors: none.
/// Complexity: `O(n)`.
pub fn map_a_to_b(path: &[(i32, i32)], k_a: f64) -> Option<f64> {
    sample(path.iter().map(|&(a, b)| (a, b)), k_a)
}

/// Sample a path at grid index `k_b`: [`map_a_to_b`] with the roles swapped
/// (keys = distinct `k_b`, first pair with that `k_b`).
///
/// Units: grid indices. NaN rule: `None`. Errors: none. Complexity: `O(n)`.
pub fn map_b_to_a(path: &[(i32, i32)], k_b: f64) -> Option<f64> {
    sample(path.iter().map(|&(a, b)| (b, a)), k_b)
}

/// Shared sampler over `(key, value)` pairs in key order.
fn sample(pairs: impl Iterator<Item = (i32, i32)>, x: f64) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    let mut prev: Option<(i32, i32)> = None;
    for (key, val) in pairs {
        if let Some((pk, pv)) = prev {
            if key == pk {
                // Not the first pair with this key.
                continue;
            }
            let (kf, vf) = (f64::from(key), f64::from(val));
            if x == kf {
                return Some(vf);
            }
            if x < kf {
                let p = f64::from(pk);
                if x > p {
                    return Some(f64::from(pv) + (x - p) / (kf - p) * (vf - f64::from(pv)));
                }
                // Below the previous key: only reachable on unsorted input.
                return None;
            }
        } else {
            let kf = f64::from(key);
            if x < kf {
                return None;
            }
            if x == kf {
                return Some(f64::from(val));
            }
        }
        prev = Some((key, val));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_half_up() {
        assert_eq!(md_to_k(5000.0).unwrap(), 10000);
        assert_eq!(md_to_k(6000.3).unwrap(), 12001);
        assert_eq!(md_to_k(6001.0).unwrap(), 12002);
        assert_eq!(md_to_k(6001.7).unwrap(), 12003);
        assert_eq!(md_to_k(6002.4).unwrap(), 12005);
        assert_eq!(md_to_k(6003.1).unwrap(), 12006);
        assert_eq!(md_to_k(0.25).unwrap(), 1);
        assert_eq!(md_to_k(-0.25).unwrap(), 0);
        assert!(md_to_k(f64::NAN).is_err());
        assert!(md_to_k(f64::INFINITY).is_err());
        assert!(md_to_k(1e12).is_err());
    }

    #[test]
    fn sampling_is_safe_on_edges() {
        assert_eq!(map_a_to_b(&[], 1.0), None);
        let one = [(5, 7)];
        assert_eq!(map_a_to_b(&one, 5.0), Some(7.0));
        assert_eq!(map_a_to_b(&one, 5.5), None);
        assert_eq!(map_a_to_b(&one, f64::NAN), None);
    }
}
