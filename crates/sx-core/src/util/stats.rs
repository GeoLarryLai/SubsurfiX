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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SxError;

    fn assert_close(got: f64, expect: f64, tol: f64) {
        if expect.is_nan() {
            assert!(got.is_nan(), "got {got}, expected NaN");
            return;
        }
        let diff = (got - expect).abs();
        assert!(
            diff <= tol,
            "got {got}, expected {expect}, diff {diff}, tol {tol}"
        );
    }

    fn assert_close_slice(got: &[f64], expect: &[f64], tol: f64) {
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect).enumerate() {
            if e.is_nan() {
                assert!(g.is_nan(), "index {i}: got {g}, expected NaN");
            } else {
                let diff = (g - e).abs();
                assert!(diff <= tol, "index {i}: got {g}, expected {e}, diff {diff}");
            }
        }
    }

    #[test]
    fn percentile_example() {
        let values = [3.0, f64::NAN, 1.0, 2.0, 4.0];
        let ps = [0.0, 25.0, 50.0, 90.0, 100.0];
        let expect = [1.0, 1.75, 2.5, 3.7, 4.0];
        let got = percentiles(&values, &ps).unwrap();
        assert_close_slice(&got, &expect, 1e-12);
        for (&p, &batch) in ps.iter().zip(&got) {
            assert_close(percentile(&values, p).unwrap(), batch, 0.0);
        }
        let values_f32 = [3.0_f32, f32::NAN, 1.0, 2.0, 4.0];
        assert_close_slice(&percentiles_f32(&values_f32, &ps).unwrap(), &expect, 1e-12);
        assert_close(percentile_f32(&values_f32, 90.0).unwrap(), 3.7, 1e-12);
    }

    #[test]
    fn percentile_rejects_bad_p_before_empty_result() {
        assert!(matches!(
            percentile(&[1.0], f64::NAN),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            percentile(&[1.0], -0.1),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            percentile(&[1.0], 100.1),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            percentiles(&[1.0], &[50.0, f64::NAN]),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            percentiles(&[], &[101.0]),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            percentiles(&[1.0], &[10.0, -1.0]),
            Err(SxError::InvalidInput(_))
        ));
    }

    #[test]
    fn percentile_empty_and_all_nan_are_nan() {
        assert!(percentile(&[], 50.0).unwrap().is_nan());
        assert!(percentile(&[f64::NAN, f64::NAN], 0.0).unwrap().is_nan());
        let got = percentiles(&[f64::NAN], &[0.0, 50.0, 100.0]).unwrap();
        assert!(got.iter().all(|v| v.is_nan()));
        assert!(percentiles(&[1.0], &[]).unwrap().is_empty());
        assert!(percentile_f32(&[], 50.0).unwrap().is_nan());
        assert!(percentiles_f32(&[f32::NAN], &[25.0]).unwrap()[0].is_nan());
    }

    #[test]
    fn percentile_keeps_infinities() {
        // Infinity stays in the sample. Dropping it would make the 25th
        // percentile of [1, 2, 3] equal 1.5, not 1.75. An infinite gap still
        // follows the linear formula, which is NaN (NumPy does the same).
        let got = percentiles(&[1.0, 2.0, 3.0, f64::INFINITY], &[0.0, 25.0, 50.0, 100.0]).unwrap();
        assert_eq!(got[0], 1.0);
        assert_eq!(got[1], 1.75);
        assert_eq!(got[2], 2.5);
        assert!(got[3].is_nan());
        let got = percentiles(
            &[f64::NAN, f64::NEG_INFINITY, 0.0, f64::INFINITY],
            &[0.0, 100.0],
        )
        .unwrap();
        assert!(got[0].is_nan());
        assert!(got[1].is_nan());
    }

    #[test]
    fn median_examples() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[5.0, f64::NAN, 1.0, 3.0]), 3.0);
        assert!(median(&[]).is_nan());
        assert!(median(&[f64::NAN]).is_nan());
        assert_eq!(median_f32(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_f32(&[5.0, f32::NAN, 1.0, 3.0]), 3.0);
        assert!(median_f32(&[]).is_nan());
    }

    #[test]
    fn mad_and_robust_scale_example() {
        let values = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(median(&values), 3.0);
        assert_eq!(mad(&values), 1.0);
        assert_eq!(MAD_TO_SIGMA, 1.482602218505602);
        assert_eq!(robust_scale(&values), MAD_TO_SIGMA);
        // Specs that write 1.4826 mean MAD_TO_SIGMA. Check the rounded figure
        // at the shared relative tolerance.
        let rounded = 1.4826;
        let rel = (robust_scale(&values) - rounded).abs() / rounded.abs();
        assert!(rel <= 1e-4, "relative {rel}");
    }

    #[test]
    fn mad_empty_is_nan() {
        assert!(mad(&[]).is_nan());
        assert!(mad(&[f64::NAN, f64::NAN]).is_nan());
        assert!(robust_scale(&[]).is_nan());
        assert!(robust_scale(&[f64::NAN]).is_nan());
    }

    #[test]
    fn weighted_median_examples() {
        assert_eq!(
            weighted_median(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0], 1).unwrap(),
            2.0
        );
        assert_eq!(
            weighted_median(&[5.0, f64::NAN, 1.0, 3.0], &[1.0, 5.0, 1.0, 2.0], 1).unwrap(),
            3.0
        );
        assert_eq!(
            weighted_median(&[10.0, 20.0, 30.0], &[0.2, 0.2, 0.6], 1).unwrap(),
            30.0
        );
        assert_eq!(
            weighted_median(&[10.0, 20.0, 30.0], &[0.5, 0.25, 0.25], 1).unwrap(),
            10.0
        );
        assert!(
            weighted_median(&[5.0, f64::NAN, 1.0, 3.0], &[1.0, 5.0, 1.0, 2.0], 4)
                .unwrap()
                .is_nan()
        );
    }

    #[test]
    fn weighted_median_errors_and_nan_rules() {
        assert!(matches!(
            weighted_median(&[1.0, 2.0], &[-1.0, 1.0], 1),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            weighted_median(&[1.0], &[1.0, -1.0], 1),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            weighted_median(&[1.0], &[f64::NAN], 1),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            weighted_median(&[1.0], &[f64::INFINITY], 1),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            weighted_median(&[1.0], &[f64::NEG_INFINITY], 1),
            Err(SxError::InvalidInput(_))
        ));
        assert!(weighted_median(&[], &[], 1).unwrap().is_nan());
        assert!(weighted_median(&[1.0, 2.0], &[0.0, 0.0], 1)
            .unwrap()
            .is_nan());
        assert_eq!(weighted_median(&[1.0, 2.0], &[0.0, 1.0], 1).unwrap(), 2.0);
        assert_eq!(
            weighted_median(&[f64::INFINITY, 1.0], &[10.0, 1.0], 1).unwrap(),
            1.0
        );
        assert_eq!(
            weighted_median(&[10.0, 20.0], &[1.0, 1.0], 0).unwrap(),
            10.0
        );
    }

    #[test]
    fn percentile_matches_numpy_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/util_percentile_numpy.json"
        );
        let text = std::fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"));
        let root: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {path}: {err}"));
        let cases = root["cases"]
            .as_array()
            .unwrap_or_else(|| panic!("fixture has no cases array"));
        assert_eq!(cases.len(), 50);
        for (case_i, case) in cases.iter().enumerate() {
            let values = json_f64_array(&case["values"]);
            let ps = json_f64_array(&case["ps"]);
            let got = percentiles(&values, &ps).unwrap_or_else(|err| {
                panic!("case {case_i}: {err}");
            });
            assert_eq!(got.len(), ps.len(), "case {case_i}");
            if case["expected"].is_null() {
                assert!(
                    got.iter().all(|v| v.is_nan()),
                    "case {case_i}: expected all NaN, got {got:?}"
                );
                continue;
            }
            let expected = json_f64_array(&case["expected"]);
            assert_eq!(got.len(), expected.len(), "case {case_i}");
            for (j, (g, e)) in got.iter().zip(&expected).enumerate() {
                if e.is_nan() {
                    assert!(g.is_nan(), "case {case_i} index {j}: got {g}");
                } else {
                    let diff = (g - e).abs();
                    let tol = 1e-12 + 1e-12 * e.abs();
                    assert!(
                        diff <= tol,
                        "case {case_i} index {j}: got {g}, expected {e}, diff {diff}"
                    );
                }
            }
        }
    }

    fn json_f64_array(value: &serde_json::Value) -> Vec<f64> {
        let Some(arr) = value.as_array() else {
            panic!("expected a JSON array, got {value}");
        };
        arr.iter().map(json_f64).collect()
    }

    fn json_f64(value: &serde_json::Value) -> f64 {
        if value.is_null() {
            f64::NAN
        } else {
            value
                .as_f64()
                .unwrap_or_else(|| panic!("expected a JSON number, got {value}"))
        }
    }

    proptest::proptest! {
        #[test]
        fn percentile_is_nondecreasing(
            values in proptest::collection::vec(-1.0e6f64..1.0e6, 1..200),
            mut ps in proptest::collection::vec(0.0f64..=100.0, 1..16),
        ) {
            ps.sort_by(f64::total_cmp);
            let mut prev = f64::NEG_INFINITY;
            for &p in &ps {
                let v = percentile(&values, p).unwrap_or_else(|err| panic!("{err}"));
                proptest::prop_assert!(v >= prev, "p {p}: {v} < {prev}");
                prev = v;
            }
        }
    }
}
