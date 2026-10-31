// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `ml_propagation`.
//!
//! Learns a property from location and stratigraphic features (for example
//! x, y, depth below a formation top) at well samples and predicts quantiles
//! (default P10, P50, P90) at target points such as grid cells. The default
//! model is the in-house quantile regression forest ([`crate::sim::qrf`],
//! Meinshausen 2006), which gave calibrated P10-P90 bands on the KGS check;
//! the option is LightGBM with the quantile objective (one booster per
//! quantile, crossing quantiles sorted per row), whose bands came out narrow.
//!
//! Incomplete: no spatial cross-validation inside, no hyperparameter
//! search, no conditioning to well data at the target (predictions do not
//! honour data exactly, unlike SGS).

use serde_json::json;

use crate::error::{SxError, SxResult};
use crate::sim::qrf::{QrfParams, QuantileForest};

/// Model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationModel {
    /// Quantile LightGBM (lightgbm3), option.
    LightGbm,
    /// In-house quantile regression forest (Meinshausen 2006), default.
    QuantileForest,
}

impl PropagationModel {
    /// Parse "lightgbm" / "qrf".
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for other names.
    pub fn parse(s: &str) -> SxResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "lightgbm" | "lgbm" | "gbm" => Ok(Self::LightGbm),
            "qrf" | "quantile_forest" | "forest" => Ok(Self::QuantileForest),
            _ => Err(SxError::InvalidInput(format!(
                "unknown ml_propagation model {s:?}"
            ))),
        }
    }
}

/// `ml_propagation` settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PropagationParams {
    /// Model family.
    pub model: PropagationModel,
    /// Quantile levels, each in (0, 1).
    pub quantiles: Vec<f64>,
    /// LightGBM boosting rounds.
    pub lgbm_iterations: usize,
    /// LightGBM learning rate.
    pub lgbm_learning_rate: f64,
    /// LightGBM leaves per tree.
    pub lgbm_num_leaves: usize,
    /// LightGBM rows per leaf.
    pub lgbm_min_data_in_leaf: usize,
    /// QRF settings (seed is overridden by `seed`).
    pub qrf: QrfParams,
    /// Seed.
    pub seed: u64,
}

impl Default for PropagationParams {
    fn default() -> Self {
        Self {
            model: PropagationModel::QuantileForest,
            quantiles: vec![0.1, 0.5, 0.9],
            lgbm_iterations: 300,
            lgbm_learning_rate: 0.05,
            lgbm_num_leaves: 31,
            lgbm_min_data_in_leaf: 20,
            qrf: QrfParams::default(),
            seed: 0,
        }
    }
}

impl PropagationParams {
    /// Settings from the `[propagation]` config section and the project seed.
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for an unknown method.
    pub fn from_config(c: &crate::config::PropagationConfig, seed: u64) -> SxResult<Self> {
        let d = Self::default();
        Ok(Self {
            model: PropagationModel::parse(&c.method)?,
            quantiles: c.quantiles.clone(),
            lgbm_iterations: c.lgbm_iterations,
            qrf: QrfParams {
                n_trees: c.qrf_trees,
                min_leaf: c.qrf_min_leaf,
                ..d.qrf
            },
            seed,
            ..d
        })
    }
}

/// Train on `(train_x, train_y)` and predict `[row][quantile]` at `pred_x`.
/// Rows with a non-finite target are dropped from training.
///
/// # Errors
/// [`SxError::Shape`] on ragged input; [`SxError::InvalidInput`] for a
/// quantile outside (0, 1); [`SxError::EmptyInput`] with no training rows;
/// [`SxError::Model`] if LightGBM fails.
pub fn ml_propagation(
    train_x: &[Vec<f64>],
    train_y: &[f64],
    pred_x: &[Vec<f64>],
    p: &PropagationParams,
) -> SxResult<Vec<Vec<f64>>> {
    if train_x.len() != train_y.len() {
        return Err(SxError::Shape(format!(
            "{} rows vs {} targets",
            train_x.len(),
            train_y.len()
        )));
    }
    if p.quantiles.is_empty() || p.quantiles.iter().any(|q| !(*q > 0.0 && *q < 1.0)) {
        return Err(SxError::InvalidInput("quantiles must lie in (0, 1)".into()));
    }
    let keep: Vec<usize> = (0..train_y.len())
        .filter(|&i| train_y[i].is_finite())
        .collect();
    if keep.is_empty() {
        return Err(SxError::EmptyInput(
            "ml_propagation: no finite training targets".into(),
        ));
    }
    let x: Vec<Vec<f64>> = keep.iter().map(|&i| train_x[i].clone()).collect();
    let y: Vec<f64> = keep.iter().map(|&i| train_y[i]).collect();
    let nf = x[0].len();
    if nf == 0 || x.iter().chain(pred_x).any(|r| r.len() != nf) {
        return Err(SxError::Shape(
            "ml_propagation rows must share a non-zero length".into(),
        ));
    }
    let mut out = match p.model {
        PropagationModel::QuantileForest => {
            let f = QuantileForest::fit(
                &x,
                &y,
                &QrfParams {
                    seed: p.seed,
                    ..p.qrf
                },
            )?;
            f.predict_quantiles(pred_x, &p.quantiles)?
        }
        PropagationModel::LightGbm => {
            let yf: Vec<f32> = y.iter().map(|v| *v as f32).collect();
            let flat: Vec<f64> = pred_x.iter().flatten().copied().collect();
            let mut cols = vec![];
            for &q in &p.quantiles {
                let params = json!({
                    "objective": "quantile", "alpha": q,
                    "num_iterations": p.lgbm_iterations, "learning_rate": p.lgbm_learning_rate,
                    "num_leaves": p.lgbm_num_leaves, "min_data_in_leaf": p.lgbm_min_data_in_leaf,
                    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
                    "deterministic": true, "force_row_wise": true, "num_threads": 4,
                    "seed": p.seed as i64 % 2_147_483_647, "verbosity": -1
                });
                let ds = lightgbm3::Dataset::from_vec_of_vec(x.clone(), yf.clone(), true)
                    .map_err(|e| SxError::Model(format!("lightgbm dataset: {e}")))?;
                let b = lightgbm3::Booster::train(ds, &params)
                    .map_err(|e| SxError::Model(format!("lightgbm train: {e}")))?;
                let pr = if pred_x.is_empty() {
                    vec![]
                } else {
                    b.predict_with_params(&flat, nf as i32, true, "num_threads=4")
                        .map_err(|e| SxError::Model(format!("lightgbm predict: {e}")))?
                };
                cols.push(pr);
            }
            (0..pred_x.len())
                .map(|r| cols.iter().map(|c| c[r]).collect())
                .collect()
        }
    };
    // Monotone rearrangement: sort each row's quantiles to match sorted levels.
    let mut order: Vec<usize> = (0..p.quantiles.len()).collect();
    order.sort_by(|a, b| p.quantiles[*a].total_cmp(&p.quantiles[*b]));
    for row in &mut out {
        let mut v: Vec<f64> = row.clone();
        v.sort_by(f64::total_cmp);
        for (rank, &qi) in order.iter().enumerate() {
            row[qi] = v[rank];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_models_bracket_a_noisy_trend() {
        let mut s = 11u64;
        let mut r = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mk = |n: usize, r: &mut dyn FnMut() -> f64| {
            let (mut x, mut y) = (vec![], vec![]);
            for _ in 0..n {
                let a = r();
                x.push(vec![a, r()]);
                y.push(3.0 * a + (r() - 0.5));
            }
            (x, y)
        };
        let (xt, yt) = mk(1500, &mut r);
        let (xv, yv) = mk(500, &mut r);
        for m in [PropagationModel::LightGbm, PropagationModel::QuantileForest] {
            let p = PropagationParams {
                model: m,
                qrf: QrfParams {
                    n_trees: 80,
                    ..QrfParams::default()
                },
                ..PropagationParams::default()
            };
            let q = ml_propagation(&xt, &yt, &xv, &p).unwrap();
            let cover = q
                .iter()
                .zip(&yv)
                .filter(|(r, y)| **y >= r[0] && **y <= r[2])
                .count() as f64
                / yv.len() as f64;
            let mae = q
                .iter()
                .zip(&yv)
                .map(|(r, y)| (r[1] - y).abs())
                .sum::<f64>()
                / yv.len() as f64;
            // Quantile boosting is known to give somewhat narrow intervals.
            assert!((0.6..0.92).contains(&cover), "{m:?} coverage {cover}");
            assert!(mae < 0.35, "{m:?} MAE {mae}");
        }
        assert!(ml_propagation(
            &xt,
            &yt,
            &xv,
            &PropagationParams {
                quantiles: vec![1.0],
                ..PropagationParams::default()
            }
        )
        .is_err());
    }
}
