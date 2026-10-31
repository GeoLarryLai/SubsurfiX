// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Regression error summaries.

use crate::error::{SxError, SxResult};

/// Error summary for paired truth and prediction samples.
///
/// `rmse`, `mae`, and `bias` use the units of the inputs. `r2` and `pearson_r`
/// are unitless. `bias` is mean(prediction − truth): positive means the
/// predictions are high.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Number of pairs where both values are finite.
    pub n: usize,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Mean signed error, prediction minus truth.
    pub bias: f64,
    /// Coefficient of determination, or NaN when truth has zero variance.
    pub r2: f64,
    /// Pearson correlation in `[-1, 1]`, or NaN when either side has zero variance.
    pub pearson_r: f64,
}

/// Regression metrics on finite `(truth, prediction)` pairs.
///
/// A pair is used only when both values are finite. NaN and ±infinity are
/// skipped. `n == 0`, including empty input, is [`SxError::EmptyInput`].
///
/// Errors, in order: lengths differ ([`SxError::Shape`]), no finite pair.
pub fn regression_metrics(truth: &[f64], pred: &[f64]) -> SxResult<Metrics> {
    if truth.len() != pred.len() {
        return Err(SxError::Shape(format!(
            "truth length {} does not match pred length {}",
            truth.len(),
            pred.len()
        )));
    }
    let mut n = 0usize;
    let mut sum_t = 0.0;
    let mut sum_p = 0.0;
    for (&t, &p) in truth.iter().zip(pred.iter()) {
        if t.is_finite() && p.is_finite() {
            n += 1;
            sum_t += t;
            sum_p += p;
        }
    }
    if n == 0 {
        return Err(SxError::EmptyInput(
            "regression metrics need at least one finite pair".into(),
        ));
    }
    let nf = n as f64;
    let mean_t = sum_t / nf;
    let mean_p = sum_p / nf;
    let mut sum_e2 = 0.0;
    let mut sum_abs_e = 0.0;
    let mut sum_e = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (&t, &p) in truth.iter().zip(pred.iter()) {
        if !(t.is_finite() && p.is_finite()) {
            continue;
        }
        let e = p - t;
        sum_e2 += e * e;
        sum_abs_e += e.abs();
        sum_e += e;
        let dt = t - mean_t;
        let dp = p - mean_p;
        sxx += dt * dt;
        syy += dp * dp;
        sxy += dt * dp;
    }
    let r2 = if sxx > 0.0 {
        1.0 - sum_e2 / sxx
    } else {
        f64::NAN
    };
    let pearson_r = if sxx > 0.0 && syy > 0.0 {
        (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    };
    Ok(Metrics {
        n,
        rmse: (sum_e2 / nf).sqrt(),
        mae: sum_abs_e / nf,
        bias: sum_e / nf,
        r2,
        pearson_r,
    })
}

/// [`regression_metrics`] after converting each `f32` with [`f64::from`].
///
/// NaN rule, empty rule, and errors match [`regression_metrics`].
pub fn regression_metrics_f32(truth: &[f32], pred: &[f32]) -> SxResult<Metrics> {
    if truth.len() != pred.len() {
        return Err(SxError::Shape(format!(
            "truth length {} does not match pred length {}",
            truth.len(),
            pred.len()
        )));
    }
    let truth: Vec<f64> = truth.iter().copied().map(f64::from).collect();
    let pred: Vec<f64> = pred.iter().copied().map(f64::from).collect();
    regression_metrics(&truth, &pred)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SxError;

    fn assert_close(got: f64, expect: f64) {
        let diff = (got - expect).abs();
        assert!(diff <= 1e-12, "got {got}, expected {expect}, diff {diff}");
    }

    fn assert_example(m: &Metrics) {
        assert_eq!(m.n, 3);
        assert_close(m.rmse, 0.408248290463863);
        assert_close(m.mae, 0.3333333333333333);
        assert_close(m.bias, 0.0);
        assert_close(m.r2, 0.75);
        assert_close(m.pearson_r, 1.0);
    }

    #[test]
    fn example() {
        let m = regression_metrics(&[1.0, 2.0, 3.0, 4.0], &[1.5, 2.0, 2.5, f64::NAN]).unwrap();
        assert_example(&m);
        let skipped =
            regression_metrics(&[f64::NAN, 1.0, 2.0, 3.0], &[9.0, 1.5, 2.0, 2.5]).unwrap();
        assert_example(&skipped);
    }

    #[test]
    fn example_f32() {
        let m = regression_metrics_f32(&[1.0, 2.0, 3.0, 4.0], &[1.5, 2.0, 2.5, f32::NAN]).unwrap();
        assert_example(&m);
    }

    #[test]
    fn shape_and_empty() {
        assert!(matches!(
            regression_metrics(&[], &[1.0]),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            regression_metrics(&[], &[]),
            Err(SxError::EmptyInput(_))
        ));
        assert!(matches!(
            regression_metrics(&[f64::NAN], &[1.0]),
            Err(SxError::EmptyInput(_))
        ));
        assert!(matches!(
            regression_metrics(&[1.0], &[f64::INFINITY]),
            Err(SxError::EmptyInput(_))
        ));
        assert!(matches!(
            regression_metrics_f32(&[1.0], &[]),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            regression_metrics_f32(&[], &[]),
            Err(SxError::EmptyInput(_))
        ));
    }

    #[test]
    fn zero_variance_is_nan() {
        let flat_truth = regression_metrics(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(flat_truth.n, 3);
        assert!(flat_truth.r2.is_nan());
        assert!(flat_truth.pearson_r.is_nan());

        let flat_pred = regression_metrics(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]).unwrap();
        assert!(flat_pred.pearson_r.is_nan());
        assert_close(flat_pred.r2, 0.0);
    }
}
