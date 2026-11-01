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
