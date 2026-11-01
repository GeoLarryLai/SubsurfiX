// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Percentiles, median, MAD, and the lower weighted median.

use crate::error::{SxError, SxResult};

/// Normal-consistency factor for the median absolute deviation.
///
/// Exact value `1.482602218505602`. A spec that writes `1.4826` means this
/// constant. Unitless.
pub const MAD_TO_SIGMA: f64 = 1.482602218505602;

/// NumPy `nanpercentile` with the default `"linear"` method.
///
/// `p` is a percent in `[0, 100]`. NaN samples are skipped; ±infinity is kept.
/// No non-NaN sample (empty or all NaN) returns NaN, not an error.
///
/// Errors: [`SxError::InvalidInput`] if `p` is NaN or outside `[0, 100]`.
///
/// Complexity: `O(n log n)`.
pub fn percentile(values: &[f64], p: f64) -> SxResult<f64> {
    let mut got = percentiles(values, &[p])?;
    match got.pop() {
        Some(v) => Ok(v),
        None => Err(SxError::Numerical("percentile returned no value".into())),
    }
}

/// [`percentile`] at each percent in `ps`, sorting the samples once.
///
/// Every `p` is checked before any value is read. An empty `ps` returns an
/// empty vector. NaN rule and empty-sample rule match [`percentile`].
///
/// Complexity: `O(n log n + m)` for `m` percents.
pub fn percentiles(values: &[f64], ps: &[f64]) -> SxResult<Vec<f64>> {
    for &p in ps {
        check_percentile(p)?;
    }
    if ps.is_empty() {
        return Ok(Vec::new());
    }
    let sorted = non_nan_sorted(values);
    if sorted.is_empty() {
        return Ok(vec![f64::NAN; ps.len()]);
    }
    let mut out = Vec::with_capacity(ps.len());
    for &p in ps {
        out.push(percentile_sorted(&sorted, p));
    }
    Ok(out)
}

/// [`percentile`] after converting each `f32` with [`f64::from`].
///
/// NaN rule, empty rule, and errors match [`percentile`]. The result is `f64`.
///
/// Complexity: `O(n log n)`.
pub fn percentile_f32(values: &[f32], p: f64) -> SxResult<f64> {
    percentile(&as_f64(values), p)
}

/// [`percentiles`] after converting each `f32` with [`f64::from`].
///
/// NaN rule, empty rule, and errors match [`percentiles`]. Results are `f64`.
///
/// Complexity: `O(n log n + m)`.
pub fn percentiles_f32(values: &[f32], ps: &[f64]) -> SxResult<Vec<f64>> {
    percentiles(&as_f64(values), ps)
}

/// NumPy `nanmedian`.
///
/// NaN samples are skipped. No non-NaN sample returns NaN. An odd count returns
/// the middle sorted value; an even count returns the average of the two
/// middle values. This can differ from [`percentile`] at 50 in the last bit.
///
/// Complexity: `O(n log n)`.
#[must_use]
pub fn median(values: &[f64]) -> f64 {
    median_sorted(&non_nan_sorted(values))
}

/// [`median`] after converting each `f32` with [`f64::from`].
///
/// NaN rule and empty rule match [`median`]. The result is `f64`.
///
/// Complexity: `O(n log n)`.
#[must_use]
pub fn median_f32(values: &[f32]) -> f64 {
    median(&as_f64(values))
}

/// Median absolute deviation of the non-NaN samples.
///
/// `median(|v - median(values)|)`. No non-NaN sample returns NaN.
///
/// Complexity: `O(n log n)`.
#[must_use]
pub fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    if center.is_nan() {
        return f64::NAN;
    }
    let deviations: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .map(|v| (v - center).abs())
        .collect();
    median(&deviations)
}

/// [`MAD_TO_SIGMA`] times [`mad`].
///
/// NaN rule and empty rule match [`mad`]. No floor is applied; callers that
/// need one apply it themselves.
///
/// Complexity: `O(n log n)`.
#[must_use]
pub fn robust_scale(values: &[f64]) -> f64 {
    MAD_TO_SIGMA * mad(values)
}

/// Lower weighted median, with no interpolation between samples.
///
/// A sample contributes when its value is finite and its weight is `> 0`.
/// `tot` is the sum of weights of samples with a finite value, in input order.
/// If `tot <= 0` or fewer than `min_count` samples contribute, the result is
/// NaN (including empty input). Ties keep the lower value: the first sample
/// whose running weight reaches `0.5 * tot`.
///
/// Errors, in order: lengths differ ([`SxError::Shape`]); a weight is NaN,
/// ±infinity, or negative ([`SxError::InvalidInput`]).
///
/// Complexity: `O(n log n)`.
pub fn weighted_median(values: &[f64], weights: &[f64], min_count: usize) -> SxResult<f64> {
    if values.len() != weights.len() {
        return Err(SxError::Shape(format!(
            "values length {} does not match weights length {}",
            values.len(),
            weights.len()
        )));
    }
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err(SxError::InvalidInput(
                "weights must be finite and >= 0".into(),
            ));
        }
    }
    let mut total = 0.0;
    let mut samples = Vec::new();
    for (&value, &weight) in values.iter().zip(weights.iter()) {
        if value.is_finite() {
            total += weight;
            if weight > 0.0 {
                samples.push(WeightedSample { value, weight });
            }
        }
    }
    if total <= 0.0 || samples.len() < min_count {
        return Ok(f64::NAN);
    }
    samples.sort_by(|a, b| a.value.total_cmp(&b.value));
    let half = 0.5 * total;
    let mut running = 0.0;
    for sample in &samples {
        running += sample.weight;
        if running >= half {
            return Ok(sample.value);
        }
    }
    match samples.last() {
        Some(sample) => Ok(sample.value),
        None => Ok(f64::NAN),
    }
}

struct WeightedSample {
    value: f64,
    weight: f64,
}

fn check_percentile(p: f64) -> SxResult<()> {
    if (0.0..=100.0).contains(&p) {
        Ok(())
    } else {
        Err(SxError::InvalidInput(format!(
            "percentile {p} is outside [0, 100]"
        )))
    }
}

fn as_f64(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

fn non_nan_sorted(values: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(f64::total_cmp);
    sorted
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        match sorted.get(n / 2) {
            Some(&v) => v,
            None => f64::NAN,
        }
    } else {
        match (sorted.get(n / 2 - 1), sorted.get(n / 2)) {
            (Some(&a), Some(&b)) => (a + b) / 2.0,
            _ => f64::NAN,
        }
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (p / 100.0) * (n - 1) as f64;
    let lo = index_floor(h, n);
    let hi = (lo + 1).min(n - 1);
    let t = h - lo as f64;
    let Some(&a) = sorted.get(lo) else {
        return f64::NAN;
    };
    let Some(&b) = sorted.get(hi) else {
        return f64::NAN;
    };
    let d = b - a;
    // NumPy's linear percentile, including the split at t >= 0.5.
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

fn index_floor(h: f64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let last = n - 1;
    if !h.is_finite() {
        return 0;
    }
    let floored = h.floor();
    if floored <= 0.0 {
        0
    } else if floored >= last as f64 {
        last
    } else {
        floored as usize
    }
}
