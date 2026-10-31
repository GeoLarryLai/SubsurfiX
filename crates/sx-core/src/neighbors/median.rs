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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, abs: f64, rel: f64) -> bool {
        (a - b).abs() <= abs + rel * b.abs()
    }

    /// Weighted-median table.
    #[test]
    fn neighbors_weighted_median_four_donors() {
        let m = weighted_median(
            &[60.0, 75.0, 52.0, 68.0],
            &[0.5e-6, 0.6e-6, 0.25e-6, 0.2e-6],
        )
        .unwrap()
        .unwrap();
        assert!(close(m.value, 68.0, 1e-4, 1e-5));
        assert!(close(m.spread, 7.0, 1e-4, 1e-5));
        assert_eq!(m.n, 4);

        let contribs: Vec<Contrib> = [
            (60.0, 1000.0, 0.5, 1.0),
            (75.0, 1000.0, 1.0, 0.6),
            (52.0, 2000.0, 1.0, 1.0),
            (68.0, 2000.0, 0.8, 1.0),
        ]
        .iter()
        .map(|&(value, d, iq, aq): &(f64, f64, f64, f64)| Contrib {
            value,
            weight: 1.0 / (d * d) * iq * aq,
            dist_m: d,
            quality: iq * aq,
        })
        .collect();
        let s = aggregate(&contribs, &mut Vec::new());
        assert!(close(s.value, 68.0, 1e-4, 1e-5));
        assert!(close(s.spread, 7.0, 1e-4, 1e-5));
        assert!(close(s.dist_m, 1290.32, 1e-2, 0.0));
        assert!(close(s.quality, 0.6581, 1e-4, 0.0));
        assert_eq!(s.n, 4);
    }

    #[test]
    fn neighbors_weighted_median_tie_rule() {
        let m = weighted_median(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4])
            .unwrap()
            .unwrap();
        assert_eq!(m.value, 2.0);
    }

    /// Same row with B and C missing.
    #[test]
    fn neighbors_weighted_median_skips_nan() {
        let m = weighted_median(&[60.0, f64::NAN, f64::NAN, 68.0], &[0.5, 0.6, 0.25, 0.2])
            .unwrap()
            .unwrap();
        assert!(close(m.value, 60.0, 1e-4, 1e-5));
        assert_eq!(m.n, 2);
    }
}
