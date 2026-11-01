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
//! quantile, crossing quantiles sorted per row), whose raw bands come out
//! narrow. LightGBM bands are therefore conformalized by default (CQR,
//! Romano et al. 2019) with cross-conformal scores: out-of-fold quantiles on
//! the training rows (well folds when groups are given, random folds
//! otherwise) give the scores `max(q_lo - y, y - q_hi)`, and their conformal
//! quantile widens the bands of the model fitted on all training rows. The
//! correction only widens; a negative score quantile leaves the band as is.
//!
//! This is the plain feature-to-quantile model: no spatial
//! cross-validation inside, no conditioning to well data. For embedded
//! kriging inputs, well conditioning and spatially correlated realizations
//! see [`crate::sim::ember`].

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
    /// Conformalize LightGBM bands (CQR).
    pub conformal_lgbm: bool,
    /// Conformalize QRF bands (CQR).
    pub conformal_qrf: bool,
    /// Folds for the out-of-fold conformity scores.
    pub conformal_folds: usize,
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
            conformal_lgbm: true,
            conformal_qrf: false,
            conformal_folds: 5,
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
            conformal_lgbm: c.lgbm_conformal,
            conformal_qrf: c.qrf_conformal,
            conformal_folds: c.conformal_folds,
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
    ml_propagation_grouped(train_x, train_y, None, pred_x, p)
}

/// [`ml_propagation`] with optional training groups (for example well ids),
/// used for the conformal folds.
///
/// # Errors
/// As [`ml_propagation`]; [`SxError::Shape`] when `groups` has another length.
pub fn ml_propagation_grouped(
    train_x: &[Vec<f64>],
    train_y: &[f64],
    groups: Option<&[i32]>,
    pred_x: &[Vec<f64>],
    p: &PropagationParams,
) -> SxResult<Vec<Vec<f64>>> {
    if groups.is_some_and(|g| g.len() != train_y.len()) {
        return Err(SxError::Shape("groups length differs from targets".into()));
    }
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
    let g: Option<Vec<i32>> = groups.map(|g| keep.iter().map(|&i| g[i]).collect());
    let mut out = fit_predict(&x, &y, pred_x, p)?;
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
    let conformal = match p.model {
        PropagationModel::LightGbm => p.conformal_lgbm,
        PropagationModel::QuantileForest => p.conformal_qrf,
    };
    // CQR after sorting, so the median is untouched.
    if conformal {
        for (lo, hi, e) in conformal_offsets(&x, &y, g.as_deref(), p)? {
            let e = e.max(0.0);
            for row in &mut out {
                row[lo] -= e;
                row[hi] += e;
            }
        }
    }
    Ok(out)
}

fn fit_predict(
    x: &[Vec<f64>],
    y: &[f64],
    pred_x: &[Vec<f64>],
    p: &PropagationParams,
) -> SxResult<Vec<Vec<f64>>> {
    match p.model {
        PropagationModel::QuantileForest => {
            let f = QuantileForest::fit(
                x,
                y,
                &QrfParams {
                    seed: p.seed,
                    ..p.qrf
                },
            )?;
            f.predict_quantiles(pred_x, &p.quantiles)
        }
        PropagationModel::LightGbm => lgbm_quantiles(x, y, pred_x, &p.quantiles, p),
    }
}

/// Fold index per row: whole groups per fold when at least `k` groups are
/// given, otherwise a seeded random split of rows.
fn conformal_folds(n: usize, groups: Option<&[i32]>, k: usize, seed: u64) -> Vec<usize> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 11
    };
    let mut shuffle = |v: &mut Vec<usize>| {
        for i in (1..v.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    };
    if let Some(g) = groups {
        let mut ids: Vec<i32> = g.to_vec();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() >= k {
            let mut order: Vec<usize> = (0..ids.len()).collect();
            shuffle(&mut order);
            let mut fold_of = std::collections::HashMap::new();
            for (rank, &i) in order.iter().enumerate() {
                fold_of.insert(ids[i], rank % k);
            }
            return g.iter().map(|v| fold_of[v]).collect();
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order);
    let mut f = vec![0; n];
    for (rank, &i) in order.iter().enumerate() {
        f[i] = rank % k;
    }
    f
}

/// Cross-conformal CQR offsets `(lower level index, upper level index,
/// score quantile)` for each symmetric level pair. The score quantile is
/// the `ceil((n + 1)(q_hi - q_lo))`-th smallest out-of-fold score
/// `max(q_lo - y, y - q_hi)`; negative means the raw band is wider than needed.
///
/// # Errors
/// As the underlying model fit.
pub fn conformal_offsets(
    x: &[Vec<f64>],
    y: &[f64],
    groups: Option<&[i32]>,
    p: &PropagationParams,
) -> SxResult<Vec<(usize, usize, f64)>> {
    let q = &p.quantiles;
    let pairs: Vec<(usize, usize)> = (0..q.len())
        .filter(|&i| q[i] < 0.5)
        .filter_map(|i| {
            (0..q.len())
                .find(|&j| (q[j] - (1.0 - q[i])).abs() < 1e-9)
                .map(|j| (i, j))
        })
        .collect();
    let k = p.conformal_folds.max(2);
    if pairs.is_empty() || y.len() < 2 * k {
        return Ok(vec![]);
    }
    let fold = conformal_folds(y.len(), groups, k, p.seed);
    let mut oof = vec![vec![]; y.len()];
    for f in 0..k {
        let (te, tr): (Vec<usize>, Vec<usize>) = (0..y.len()).partition(|&i| fold[i] == f);
        if te.is_empty() || tr.is_empty() {
            continue;
        }
        let trx: Vec<Vec<f64>> = tr.iter().map(|&i| x[i].clone()).collect();
        let try_: Vec<f64> = tr.iter().map(|&i| y[i]).collect();
        let tex: Vec<Vec<f64>> = te.iter().map(|&i| x[i].clone()).collect();
        let pr = fit_predict(&trx, &try_, &tex, p)?;
        for (r, &i) in te.iter().enumerate() {
            oof[i] = pr[r].clone();
        }
    }
    let mut out = vec![];
    for (lo, hi) in pairs {
        let mut sc: Vec<f64> = (0..y.len())
            .filter(|&i| !oof[i].is_empty())
            .map(|i| {
                let (a, b) = (oof[i][lo].min(oof[i][hi]), oof[i][lo].max(oof[i][hi]));
                (a - y[i]).max(y[i] - b)
            })
            .collect();
        if sc.is_empty() {
            continue;
        }
        sc.sort_by(f64::total_cmp);
        let n = sc.len();
        let rank = (((n + 1) as f64) * (q[hi] - q[lo])).ceil() as usize;
        out.push((lo, hi, sc[rank.clamp(1, n) - 1]));
    }
    Ok(out)
}

/// One quantile LightGBM booster per level; `[row][level]`, unsorted.
pub(crate) fn lgbm_quantiles(
    x: &[Vec<f64>],
    y: &[f64],
    pred_x: &[Vec<f64>],
    quantiles: &[f64],
    p: &PropagationParams,
) -> SxResult<Vec<Vec<f64>>> {
    let nf = x[0].len();
    let yf: Vec<f32> = y.iter().map(|v| *v as f32).collect();
    let flat: Vec<f64> = pred_x.iter().flatten().copied().collect();
    let mut cols = vec![];
    for &q in quantiles {
        let params = json!({
            "objective": "quantile", "alpha": q,
            "num_iterations": p.lgbm_iterations, "learning_rate": p.lgbm_learning_rate,
            "num_leaves": p.lgbm_num_leaves, "min_data_in_leaf": p.lgbm_min_data_in_leaf,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
            "deterministic": true, "force_row_wise": true, "num_threads": 4,
            "seed": p.seed as i64 % 2_147_483_647, "verbosity": -1
        });
        let ds = lightgbm3::Dataset::from_vec_of_vec(x.to_vec(), yf.clone(), true)
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
    Ok((0..pred_x.len())
        .map(|r| cols.iter().map(|c| c[r]).collect())
        .collect())
}
