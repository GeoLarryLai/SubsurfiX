// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Lower weighted median, spread and trust values.

use std::cmp::Ordering;

use crate::error::{SxError, SxResult};

/// Result of a weighted median.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedMedian {
    /// Lower weighted median, in the unit of the values.
    pub value: f64,
    /// Weighted median absolute deviation from `value`, same unit; NaN when `n < 2`.
    pub spread: f64,
    /// Number of contributing pairs.
    pub n: usize,
    /// Sum of the contributing weights (`f64`, input order).
    pub total_weight: f64,
}

/// Lower weighted median over pairs with finite value and weight > 0, in input order.
///
/// Contributors are sorted by value (stable: equal values keep input order); the
/// median is the first value whose cumulative weight reaches `0.5 × total_weight`.
/// `spread` is the lower weighted median of `|v − value|` with the same weights,
/// NaN when `n < 2`. `None` when no pair contributes (including empty input).
/// NaN rule: skip (a NaN value or a NaN / non-positive weight drops the pair).
/// Errors: `Shape` if lengths differ. Complexity: `O(n log n)`.
pub fn weighted_median(values: &[f64], weights: &[f64]) -> SxResult<Option<WeightedMedian>> {
    if values.len() != weights.len() {
        return Err(SxError::Shape(format!(
            "weighted median: {} values but {} weights",
            values.len(),
            weights.len()
        )));
    }
    let pairs: Vec<(f64, f64)> = values
        .iter()
        .zip(weights)
        .filter(|(v, w)| v.is_finite() && **w > 0.0)
        .map(|(&v, &w)| (v, w))
        .collect();
    if pairs.is_empty() {
        return Ok(None);
    }
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    let mut scratch = Vec::with_capacity(pairs.len());
    let (value, spread) = median_and_spread(&pairs, total, &mut scratch);
    Ok(Some(WeightedMedian {
        value,
        spread,
        n: pairs.len(),
        total_weight: total,
    }))
}

/// One donor's contribution at one row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Contrib {
    /// Donor value (working units).
    pub value: f64,
    /// Row weight `w(r)` (> 0).
    pub weight: f64,
    /// Surface distance target–donor, m.
    pub dist_m: f64,
    /// `iq × max(aq, 0)`.
    pub quality: f64,
}

/// Aggregate of one row over its contributors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RowStats {
    pub value: f64,
    pub spread: f64,
    pub n: usize,
    pub dist_m: f64,
    pub quality: f64,
}

/// Median, spread, distance trust and quality of `contribs` (stack order).
/// Every item must have a finite value and a weight > 0. Empty input gives NaN
/// statistics with `n = 0`.
pub(crate) fn aggregate(contribs: &[Contrib], scratch: &mut Vec<(f64, f64)>) -> RowStats {
    let n = contribs.len();
    let mut tot = 0.0f64;
    let mut sum_d = 0.0f64;
    let mut sum_q = 0.0f64;
    for c in contribs {
        tot += c.weight;
        sum_d += c.dist_m * c.weight;
        sum_q += c.weight * c.quality;
    }
    if n == 0 || tot.is_nan() || tot <= 0.0 {
        return RowStats {
            value: f64::NAN,
            spread: f64::NAN,
            n,
            dist_m: f64::NAN,
            quality: f64::NAN,
        };
    }
    let pairs: Vec<(f64, f64)> = contribs.iter().map(|c| (c.value, c.weight)).collect();
    let (value, spread) = median_and_spread(&pairs, tot, scratch);
    RowStats {
        value,
        spread,
        n,
        dist_m: sum_d / tot,
        quality: sum_q / tot,
    }
}

/// `(median, spread)` of `pairs` (stack order) with total weight `tot`.
fn median_and_spread(pairs: &[(f64, f64)], tot: f64, scratch: &mut Vec<(f64, f64)>) -> (f64, f64) {
    scratch.clear();
    scratch.extend_from_slice(pairs);
    let m = lower_median(scratch, tot);
    if pairs.len() < 2 || m.is_nan() {
        return (m, f64::NAN);
    }
    scratch.clear();
    scratch.extend(pairs.iter().map(|&(v, w)| ((v - m).abs(), w)));
    (m, lower_median(scratch, tot))
}

/// First value (stable ascending sort) whose cumulative weight is `>= 0.5 × tot`.
fn lower_median(items: &mut [(f64, f64)], tot: f64) -> f64 {
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let half = 0.5 * tot;
    let mut cum = 0.0f64;
    for &(v, w) in items.iter() {
        cum += w;
        if cum >= half {
            return v;
        }
    }
    items.last().map_or(f64::NAN, |x| x.0)
}
