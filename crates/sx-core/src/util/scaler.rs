// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Column-wise standard scaler.

use crate::error::{SxError, SxResult};
use crate::types::FeatureMatrix;

/// Per-column mean and population standard deviation.
///
/// `mean` and `scale` are in the units of the fitted column. A column with no
/// finite values has mean `0` and scale `1`. A population standard deviation
/// below `1e-9` is replaced by `1`.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardScaler {
    /// Mean of the finite values in each column.
    pub mean: Vec<f64>,
    /// Population standard deviation of each column, or `1` when that
    /// deviation is below `1e-9` or the column has no finite values.
    pub scale: Vec<f64>,
}

impl StandardScaler {
    /// Fit on finite values only (NaN and ±infinity skipped), in row order.
    ///
    /// The standard deviation divides by the finite count (`ddof = 0`).
    ///
    /// NaN rule: skip. `n_rows == 0` is [`SxError::EmptyInput`].
    /// A matrix whose stored shape does not match its data is [`SxError::Shape`].
    pub fn fit(x: &FeatureMatrix) -> SxResult<Self> {
        if x.n_rows == 0 {
            return Err(SxError::EmptyInput(
                "standard scaler fit got no rows".into(),
            ));
        }
        ensure_shape(x)?;
        let mut mean = vec![0.0; x.n_cols];
        let mut scale = vec![1.0; x.n_cols];
        for (c, (mean_c, scale_c)) in mean.iter_mut().zip(scale.iter_mut()).enumerate() {
            let mut count = 0usize;
            let mut sum = 0.0;
            for r in 0..x.n_rows {
                let v = value_at(x, r, c)?;
                if v.is_finite() {
                    sum += f64::from(v);
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let center = sum / count as f64;
            let mut sq = 0.0;
            for r in 0..x.n_rows {
                let v = value_at(x, r, c)?;
                if v.is_finite() {
                    let d = f64::from(v) - center;
                    sq += d * d;
                }
            }
            let std = (sq / count as f64).sqrt();
            *mean_c = center;
            *scale_c = if std < 1e-9 { 1.0 } else { std };
        }
        Ok(Self { mean, scale })
    }

    /// Standardize each entry as `((value - mean) / scale)` cast to `f32`.
    ///
    /// NaN propagates. Column names are copied unchanged. `n_rows == 0`
    /// returns an empty matrix. A column count other than `mean.len()` is
    /// [`SxError::Shape`].
    pub fn transform(&self, x: &FeatureMatrix) -> SxResult<FeatureMatrix> {
        if x.n_cols != self.mean.len() || self.scale.len() != self.mean.len() {
            return Err(SxError::Shape(
                "standard scaler columns do not match the matrix".into(),
            ));
        }
        ensure_shape(x)?;
        let mut data = Vec::with_capacity(x.data.len());
        for r in 0..x.n_rows {
            for c in 0..x.n_cols {
                let v = value_at(x, r, c)?;
                let mean = coeff(&self.mean, c)?;
                let scale = coeff(&self.scale, c)?;
                let z = ((f64::from(v) - mean) / scale) as f32;
                data.push(z);
            }
        }
        FeatureMatrix::new(x.n_rows, x.names.clone(), data)
    }
}

fn ensure_shape(x: &FeatureMatrix) -> SxResult<()> {
    if x.names.len() != x.n_cols {
        return Err(SxError::Shape(
            "feature matrix names do not match n_cols".into(),
        ));
    }
    let Some(n) = x.n_rows.checked_mul(x.n_cols) else {
        return Err(SxError::Shape("feature matrix is too large".into()));
    };
    if x.data.len() != n {
        return Err(SxError::Shape(
            "feature matrix data length does not match the shape".into(),
        ));
    }
    Ok(())
}

fn value_at(x: &FeatureMatrix, r: usize, c: usize) -> SxResult<f32> {
    let row = x
        .row(r)
        .ok_or_else(|| SxError::Shape("feature matrix row is missing".into()))?;
    row.get(c)
        .copied()
        .ok_or_else(|| SxError::Shape("feature matrix column is missing".into()))
}

fn coeff(values: &[f64], c: usize) -> SxResult<f64> {
    values
        .get(c)
        .copied()
        .ok_or_else(|| SxError::Shape("standard scaler is missing a column coefficient".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SxError;

    fn assert_close(got: f64, expect: f64) {
        let diff = (got - expect).abs();
        assert!(diff <= 1e-12, "got {got}, expected {expect}, diff {diff}");
    }

    fn assert_f32_close(got: f32, expect: f32) {
        if expect.is_nan() {
            assert!(got.is_nan(), "got {got}");
            return;
        }
        let diff = (got - expect).abs();
        assert!(diff <= 1e-6, "got {got}, expected {expect}, diff {diff}");
    }

    #[test]
    fn example() {
        let x = FeatureMatrix::new(4, vec!["c".into()], vec![1.0, 2.0, 3.0, f32::NAN]).unwrap();
        let scaler = StandardScaler::fit(&x).unwrap();
        assert_close(scaler.mean[0], 2.0);
        assert_close(scaler.scale[0], 0.816496580927726);
        let y = scaler.transform(&x).unwrap();
        assert_eq!(y.names, vec!["c".to_string()]);
        assert_eq!(y.n_rows, 4);
        assert_eq!(y.n_cols, 1);
        assert_f32_close(y.data[0], -1.2247449);
        assert_f32_close(y.data[1], 0.0);
        assert_f32_close(y.data[2], 1.2247449);
        assert!(y.data[3].is_nan());
    }

    #[test]
    fn empty_rows() {
        let empty = FeatureMatrix::new(0, vec!["c".into()], vec![]).unwrap();
        assert!(matches!(
            StandardScaler::fit(&empty),
            Err(SxError::EmptyInput(_))
        ));
        let fitted =
            StandardScaler::fit(&FeatureMatrix::new(1, vec!["c".into()], vec![4.0]).unwrap())
                .unwrap();
        let y = fitted.transform(&empty).unwrap();
        assert_eq!(y.n_rows, 0);
        assert!(y.data.is_empty());
        assert_eq!(y.names, vec!["c".to_string()]);
    }

    #[test]
    fn shape_mismatch() {
        let x = FeatureMatrix::new(1, vec!["c".into()], vec![1.0]).unwrap();
        let scaler = StandardScaler::fit(&x).unwrap();
        let other = FeatureMatrix::new(1, vec!["a".into(), "b".into()], vec![1.0, 2.0]).unwrap();
        assert!(matches!(scaler.transform(&other), Err(SxError::Shape(_))));
        let empty_mismatch = FeatureMatrix::new(0, vec!["a".into(), "b".into()], vec![]).unwrap();
        assert!(matches!(
            scaler.transform(&empty_mismatch),
            Err(SxError::Shape(_))
        ));
    }

    #[test]
    fn non_finite_columns_and_zero_variance() {
        let missing = FeatureMatrix::new(2, vec!["c".into()], vec![f32::NAN, f32::NAN]).unwrap();
        let scaler = StandardScaler::fit(&missing).unwrap();
        assert_eq!(scaler.mean, vec![0.0]);
        assert_eq!(scaler.scale, vec![1.0]);
        let y = scaler
            .transform(&FeatureMatrix::new(2, vec!["c".into()], vec![5.0, f32::NAN]).unwrap())
            .unwrap();
        assert_eq!(y.data[0], 5.0);
        assert!(y.data[1].is_nan());

        let with_inf =
            FeatureMatrix::new(3, vec!["c".into()], vec![1.0, f32::INFINITY, 3.0]).unwrap();
        let scaler = StandardScaler::fit(&with_inf).unwrap();
        assert_close(scaler.mean[0], 2.0);
        assert_close(scaler.scale[0], 1.0);
        let y = scaler.transform(&with_inf).unwrap();
        assert_eq!(y.data[0], -1.0);
        assert!(y.data[1].is_infinite());
        assert_eq!(y.data[2], 1.0);

        let flat = FeatureMatrix::new(3, vec!["c".into()], vec![5.0, 5.0, 5.0]).unwrap();
        let scaler = StandardScaler::fit(&flat).unwrap();
        assert_close(scaler.mean[0], 5.0);
        assert_eq!(scaler.scale[0], 1.0);
        assert_eq!(scaler.transform(&flat).unwrap().data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn two_columns_stay_in_row_major_order() {
        let x = FeatureMatrix::new(
            3,
            vec!["a".into(), "b".into()],
            vec![1.0, 10.0, 2.0, 10.0, 3.0, 10.0],
        )
        .unwrap();
        let scaler = StandardScaler::fit(&x).unwrap();
        assert_close(scaler.mean[0], 2.0);
        assert_close(scaler.scale[0], 0.816496580927726);
        assert_close(scaler.mean[1], 10.0);
        assert_eq!(scaler.scale[1], 1.0);
        let y = scaler.transform(&x).unwrap();
        assert_eq!(y.names, vec!["a".to_string(), "b".to_string()]);
        assert_f32_close(y.data[0], -1.2247449);
        assert_eq!(y.data[1], 0.0);
        assert_f32_close(y.data[2], 0.0);
        assert_eq!(y.data[3], 0.0);
        assert_f32_close(y.data[4], 1.2247449);
        assert_eq!(y.data[5], 0.0);
    }
}
