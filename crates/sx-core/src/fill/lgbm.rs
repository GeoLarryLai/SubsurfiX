// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Built-in LightGBM model: parameters, seeds and the `Regressor`
//! wrapper. The only module in `sx-core` that uses `unsafe` (LightGBM C API).
//!
//! Training builds the dataset through `lightgbm3-sys` so the dataset
//! parameters (`max_bin_by_feature`, `bin_construct_sample_cnt`, …) reach
//! LightGBM; `lightgbm3::Dataset` would pass an empty parameter string.
//! The dataset and the booster both get the booster parameters plus the
//! dataset parameters (as the Python package does when it trains), then
//! `num_iterations` boosting rounds run and the model is saved as text.
//! The wrapper keeps only the model text; every `predict` call
//! loads a `lightgbm3::Booster` from it on the calling thread and drops it
//! before returning, so no booster outlives a call or crosses threads.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;

use lightgbm3_sys::{
    BoosterHandle, DatasetHandle, LGBM_BoosterCreate, LGBM_BoosterFree,
    LGBM_BoosterSaveModelToString, LGBM_BoosterUpdateOneIter, LGBM_DatasetCreateFromMat,
    LGBM_DatasetFree, LGBM_DatasetSetFeatureNames, LGBM_DatasetSetField, LGBM_GetLastError,
    C_API_DTYPE_FLOAT32, C_API_FEATURE_IMPORTANCE_SPLIT,
};
use serde_json::{json, Value};

use crate::config::FillLightgbmConfig;
use crate::error::{SxError, SxResult};
use crate::hash::stream_seed;
use crate::types::{FeatureMatrix, LogKind, Regressor};

/// Seed stream name of the LightGBM seeds.
const SEED_STREAM: &str = "fill.lgbm";

/// First buffer size for the model text (bytes); grown if LightGBM asks for more.
const MODEL_TEXT_BUFFER: usize = 1 << 20;

/// Parameters of (from `[fill.lightgbm]`).
#[derive(Debug, Clone, PartialEq)]
pub struct LgbmParams {
    /// Boosting rounds (trees).
    pub num_iterations: usize,
    /// Shrinkage per round.
    pub learning_rate: f64,
    /// Tree depth limit (levels).
    pub max_depth: usize,
    /// Leaves per tree.
    pub num_leaves: usize,
    /// Share of features tried per tree (fraction).
    pub feature_fraction: f64,
    /// Share of rows per bagging round (fraction).
    pub bagging_fraction: f64,
    /// Bagging every k rounds (0 = off).
    pub bagging_freq: usize,
    /// L1 regularization.
    pub lambda_l1: f64,
    /// L2 regularization.
    pub lambda_l2: f64,
    /// Smallest split gain.
    pub min_gain_to_split: f64,
    /// Rows per leaf.
    pub min_data_in_leaf: usize,
    /// Hessian sum per leaf.
    pub min_sum_hessian_in_leaf: f64,
    /// Histogram bins of a normal feature.
    pub max_bin: usize,
    /// Histogram bins of the wide-bin features.
    pub max_bin_wide: usize,
    /// Feature names that get `max_bin_wide` bins.
    pub wide_bin_features: Vec<String>,
    /// Rows sampled to build bins.
    pub bin_construct_sample_cnt: usize,
    /// LightGBM threads for training and prediction (not the rayon pool).
    pub num_threads: usize,
    /// Objective (`regression`, `huber`, `quantile`).
    pub objective: String,
    /// Quantile level for `quantile`.
    pub alpha: f64,
}

impl LgbmParams {
    /// Copy of `[fill.lightgbm]`. Errors: none.
    pub fn from_config(cfg: &FillLightgbmConfig) -> Self {
        Self {
            num_iterations: cfg.num_iterations,
            learning_rate: cfg.learning_rate,
            max_depth: cfg.max_depth,
            num_leaves: cfg.num_leaves,
            feature_fraction: cfg.feature_fraction,
            bagging_fraction: cfg.bagging_fraction,
            bagging_freq: cfg.bagging_freq,
            lambda_l1: cfg.lambda_l1,
            lambda_l2: cfg.lambda_l2,
            min_gain_to_split: cfg.min_gain_to_split,
            min_data_in_leaf: cfg.min_data_in_leaf,
            min_sum_hessian_in_leaf: cfg.min_sum_hessian_in_leaf,
            max_bin: cfg.max_bin,
            max_bin_wide: cfg.max_bin_wide,
            wide_bin_features: cfg.wide_bin_features.clone(),
            bin_construct_sample_cnt: cfg.bin_construct_sample_cnt,
            num_threads: cfg.num_threads,
            objective: cfg.objective.clone(),
            alpha: cfg.alpha,
        }
    }

    /// Booster parameter JSON for seed `seed` (all four seed keys set to it).
    ///
    /// Objective `regression`, the `[fill.lightgbm]` training keys, `verbosity = -1`,
    /// `deterministic = true`, `force_row_wise = true`, `num_threads`, and `seed`,
    /// `bagging_seed`, `feature_fraction_seed`, `data_random_seed`. Errors: none.
    pub fn booster_json(&self, seed: i32) -> Value {
        json!({
            "objective": self.objective,
            "alpha": self.alpha,
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "verbosity": -1,
            "deterministic": true,
            "force_row_wise": true,
            "num_threads": self.num_threads,
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
        })
    }

    /// Dataset parameter string, e.g. "max_bin=255 max_bin_by_feature=255,...,1023,1023,255
    /// bin_construct_sample_cnt=2000000 use_missing=true zero_as_missing=false data_random_seed=<seed>".
    ///
    /// `max_bin_by_feature` has one entry per name: `max_bin_wide` for names in
    /// `wide_bin_features`, else `max_bin`. Errors: none.
    pub fn dataset_params(&self, feature_names: &[String], seed: i32) -> String {
        let bins: Vec<String> = feature_names
            .iter()
            .map(|n| {
                if self.wide_bin_features.contains(n) {
                    self.max_bin_wide
                } else {
                    self.max_bin
                }
                .to_string()
            })
            .collect();
        format!(
            "max_bin={} max_bin_by_feature={} bin_construct_sample_cnt={} use_missing=true \
             zero_as_missing=false data_random_seed={seed}",
            self.max_bin,
            bins.join(","),
            self.bin_construct_sample_cnt
        )
    }
}

/// `(stream_seed(project_seed, "fill.lgbm", log as i32) & 0x7fff_ffff) as i32`.
pub fn lgbm_seed(project_seed: u64, log: LogKind) -> i32 {
    (stream_seed(project_seed, SEED_STREAM, log as i32) & 0x7fff_ffff) as i32
}

/// `key=value` pairs of a parameter object, space-separated. Lists are
/// comma-separated without brackets; strings without quotes. `skip` keys are left out.
fn param_string(params: &Value, skip: &[&str]) -> String {
    let Value::Object(map) = params else {
        return String::new();
    };
    map.iter()
        .filter(|(k, _)| !skip.contains(&k.as_str()))
        .map(|(k, v)| format!("{k}={}", param_value(v)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn param_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => items.iter().map(param_value).collect::<Vec<_>>().join(","),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Last LightGBM error text of this thread.
fn last_error() -> String {
    // SAFETY: `LGBM_GetLastError` takes no arguments and returns a pointer to a
    // NUL-terminated thread-local buffer owned by LightGBM (or null).
    let msg = unsafe { LGBM_GetLastError() };
    if msg.is_null() {
        return "unknown LightGBM error".to_string();
    }
    // SAFETY: `msg` is non-null and points to a NUL-terminated string that stays
    // valid until the next LightGBM call on this thread; it is copied at once.
    unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned()
}

/// `SxError::Model` with LightGBM's last error text if `rc != 0`.
fn check(rc: c_int, what: &str) -> SxResult<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(SxError::Model(format!("lightgbm {what}: {}", last_error())))
    }
}

fn c_string(text: String, what: &str) -> SxResult<CString> {
    CString::new(text)
        .map_err(|_| SxError::Model(format!("lightgbm {what}: text holds a NUL byte")))
}

fn c_len(n: usize, what: &str) -> SxResult<i32> {
    i32::try_from(n).map_err(|_| SxError::Shape(format!("lightgbm: {what} {n} exceeds i32")))
}

/// Owned LightGBM dataset; freed on drop if not closed.
struct Dataset(DatasetHandle);

impl Dataset {
    /// Row-major `f32` matrix with the given parameter string.
    fn from_mat(x: &FeatureMatrix, params: &CStr) -> SxResult<Self> {
        let n_rows = c_len(x.n_rows, "row count")?;
        let n_cols = c_len(x.n_cols, "column count")?;
        let mut handle: DatasetHandle = ptr::null_mut();
        // SAFETY: `x.data` holds `n_rows * n_cols` f32 values in row-major order
        // (checked by the caller) and outlives the call; `params` is a valid C
        // string; a null reference dataset is allowed; `handle` is a valid out pointer.
        let rc = unsafe {
            LGBM_DatasetCreateFromMat(
                x.data.as_ptr().cast::<c_void>(),
                C_API_DTYPE_FLOAT32 as c_int,
                n_rows,
                n_cols,
                1,
                params.as_ptr(),
                ptr::null_mut(),
                &mut handle,
            )
        };
        check(rc, "dataset create")?;
        Ok(Self(handle))
    }

    fn set_feature_names(&self, names: &[String]) -> SxResult<()> {
        let owned: Vec<CString> = names
            .iter()
            .map(|n| c_string(n.clone(), "feature name"))
            .collect::<SxResult<_>>()?;
        let mut ptrs: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();
        let n = c_len(ptrs.len(), "feature name count")?;
        // SAFETY: `self.0` is a live dataset handle; `ptrs` holds `n` pointers to
        // NUL-terminated strings owned by `owned`, which outlives the call.
        let rc = unsafe { LGBM_DatasetSetFeatureNames(self.0, ptrs.as_mut_ptr(), n) };
        check(rc, "set feature names")
    }

    /// `label` or `weight` as `f32`, one value per row.
    fn set_field(&self, field: &CStr, values: &[f32]) -> SxResult<()> {
        let n = c_len(values.len(), "field length")?;
        // SAFETY: `self.0` is a live dataset handle; `field` is a valid C string;
        // `values` holds `n` f32 values and outlives the call (LightGBM copies them).
        let rc = unsafe {
            LGBM_DatasetSetField(
                self.0,
                field.as_ptr(),
                values.as_ptr().cast::<c_void>(),
                n,
                C_API_DTYPE_FLOAT32 as c_int,
            )
        };
        check(rc, "set dataset field")
    }

    /// Free and check the return code.
    fn close(mut self) -> SxResult<()> {
        let handle = std::mem::replace(&mut self.0, ptr::null_mut());
        // SAFETY: `handle` is the live dataset handle created in `from_mat`; it is
        // taken out of `self`, so `Drop` will not free it again.
        let rc = unsafe { LGBM_DatasetFree(handle) };
        check(rc, "dataset free")
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` is a live dataset handle not freed before (`close`
            // nulls it); the return code cannot be reported from `drop`.
            let _rc = unsafe { LGBM_DatasetFree(self.0) };
        }
    }
}

/// Owned LightGBM booster used only during training; freed on drop if not closed.
struct TrainBooster(BoosterHandle);

impl TrainBooster {
    fn create(dataset: &Dataset, params: &CStr) -> SxResult<Self> {
        let mut handle: BoosterHandle = ptr::null_mut();
        // SAFETY: `dataset.0` is a live dataset handle that outlives the booster
        // (the booster is closed first); `params` is a valid C string; `handle`
        // is a valid out pointer.
        let rc = unsafe { LGBM_BoosterCreate(dataset.0, params.as_ptr(), &mut handle) };
        check(rc, "booster create")?;
        Ok(Self(handle))
    }

    fn update_one_iter(&self) -> SxResult<()> {
        let mut finished: c_int = 0;
        // SAFETY: `self.0` is a live booster handle; `finished` is a valid out pointer.
        let rc = unsafe { LGBM_BoosterUpdateOneIter(self.0, &mut finished) };
        check(rc, "boosting round")
    }

    /// Model text of every iteration.
    fn model_text(&self) -> SxResult<String> {
        let mut buf: Vec<u8> = vec![0; MODEL_TEXT_BUFFER];
        let mut len: i64 = 0;
        for _ in 0..2 {
            let cap = i64::try_from(buf.len())
                .map_err(|_| SxError::Model("lightgbm: model text too large".into()))?;
            // SAFETY: `self.0` is a live booster handle; `buf` has `cap` writable
            // bytes; LightGBM writes at most `cap` bytes and reports the needed
            // length (with the trailing NUL) in `len`.
            let rc = unsafe {
                LGBM_BoosterSaveModelToString(
                    self.0,
                    0,
                    -1,
                    C_API_FEATURE_IMPORTANCE_SPLIT as c_int,
                    cap,
                    &mut len,
                    buf.as_mut_ptr().cast::<c_char>(),
                )
            };
            check(rc, "save model")?;
            if len <= cap {
                break;
            }
            let need = usize::try_from(len)
                .map_err(|_| SxError::Model("lightgbm: bad model text length".into()))?;
            buf = vec![0; need];
        }
        let used = usize::try_from(len)
            .ok()
            .filter(|&l| l <= buf.len())
            .ok_or_else(|| SxError::Model("lightgbm: bad model text length".into()))?;
        buf.truncate(used);
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf)
            .map_err(|_| SxError::Model("lightgbm: model text is not UTF-8".into()))
    }

    /// Free and check the return code.
    fn close(mut self) -> SxResult<()> {
        let handle = std::mem::replace(&mut self.0, ptr::null_mut());
        // SAFETY: `handle` is the live booster handle created in `create`; it is
        // taken out of `self`, so `Drop` will not free it again.
        let rc = unsafe { LGBM_BoosterFree(handle) };
        check(rc, "booster free")
    }
}

impl Drop for TrainBooster {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` is a live booster handle not freed before (`close`
            // nulls it); the return code cannot be reported from `drop`.
            let _rc = unsafe { LGBM_BoosterFree(self.0) };
        }
    }
}

/// Trained model: LightGBM text and its feature count.
struct Trained {
    text: String,
    n_features: usize,
}

/// Built-in LightGBM model.
///
/// Holds the parameters, the seed and, once trained, the model text. Training
/// and prediction run LightGBM on the calling thread with `num_threads` threads.
pub struct LgbmRegressor {
    params: LgbmParams,
    seed: i32,
    model: Option<Trained>,
}

impl LgbmRegressor {
    /// Untrained model with `params` and seed `seed` (all four LightGBM seeds).
    pub fn new(params: LgbmParams, seed: i32) -> Self {
        Self {
            params,
            seed,
            model: None,
        }
    }

    /// LightGBM model text. Errors: `Model` if not trained.
    pub fn to_model_text(&self) -> SxResult<String> {
        self.model
            .as_ref()
            .map(|m| m.text.clone())
            .ok_or_else(|| SxError::Model("lightgbm: model is not trained".into()))
    }

    /// Load a trained model from text. Errors: `Model` on a bad text.
    pub fn from_model_text(text: &str, params: LgbmParams, seed: i32) -> SxResult<Self> {
        let booster = load_booster(text)?;
        let n_features = usize::try_from(booster.num_features())
            .map_err(|_| SxError::Model("lightgbm: model has a negative feature count".into()))?;
        drop(booster);
        Ok(Self {
            params,
            seed,
            model: Some(Trained {
                text: text.to_string(),
                n_features,
            }),
        })
    }
}

fn load_booster(text: &str) -> SxResult<lightgbm3::Booster> {
    lightgbm3::Booster::from_string(text)
        .map_err(|e| SxError::Model(format!("lightgbm: cannot load model text: {e}")))
}

/// Train on `x`, `y` (and `weight`) and return the model text.
fn train_model_text(
    params: &LgbmParams,
    seed: i32,
    x: &FeatureMatrix,
    y: &[f32],
    weight: Option<&[f32]>,
) -> SxResult<String> {
    if x.n_rows == 0 || x.n_cols == 0 {
        return Err(SxError::EmptyInput(format!(
            "lightgbm: training matrix is {} x {}",
            x.n_rows, x.n_cols
        )));
    }
    let want = x
        .n_rows
        .checked_mul(x.n_cols)
        .ok_or_else(|| SxError::Shape("lightgbm: training matrix too large".into()))?;
    if x.data.len() != want || x.names.len() != x.n_cols {
        return Err(SxError::Shape(format!(
            "lightgbm: training matrix {} x {} has {} values and {} names",
            x.n_rows,
            x.n_cols,
            x.data.len(),
            x.names.len()
        )));
    }
    if y.len() != x.n_rows || weight.is_some_and(|w| w.len() != x.n_rows) {
        return Err(SxError::Shape(format!(
            "lightgbm: {} rows, {} targets, {:?} weights",
            x.n_rows,
            y.len(),
            weight.map(<[f32]>::len)
        )));
    }
    let all_params = c_string(
        format!(
            "{} {}",
            param_string(&params.booster_json(seed), &["data_random_seed"]),
            params.dataset_params(&x.names, seed)
        ),
        "parameters",
    )?;

    let dataset = Dataset::from_mat(x, &all_params)?;
    dataset.set_feature_names(&x.names)?;
    dataset.set_field(c"label", y)?;
    if let Some(w) = weight {
        dataset.set_field(c"weight", w)?;
    }
    let booster = TrainBooster::create(&dataset, &all_params)?;
    for _ in 0..params.num_iterations {
        booster.update_one_iter()?;
    }
    let text = booster.model_text()?;
    booster.close()?;
    dataset.close()?;
    Ok(text)
}

/// `name()` = "lightgbm". `predict` before `fit` → `SxError::Model`.
///
/// `fit` trains `num_iterations` rounds (no early stopping) and replaces any
/// earlier model. `predict` returns LightGBM's `f64` output stored as `f32`;
/// zero rows give an empty vector. NaN rule: NaN features are LightGBM missing
/// values. Errors: `Shape` for mismatched lengths or a column count other than
/// the model's; `EmptyInput` for an empty training matrix; `Model` for any
/// LightGBM failure.
impl Regressor for LgbmRegressor {
    fn name(&self) -> &str {
        "lightgbm"
    }

    fn fit(&mut self, x: &FeatureMatrix, y: &[f32], weight: Option<&[f32]>) -> SxResult<()> {
        let text = train_model_text(&self.params, self.seed, x, y, weight)?;
        self.model = Some(Trained {
            text,
            n_features: x.n_cols,
        });
        Ok(())
    }

    fn predict(&self, x: &FeatureMatrix) -> SxResult<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| SxError::Model("lightgbm: predict before fit".into()))?;
        if x.n_cols != model.n_features || x.data.len() != x.n_rows * x.n_cols {
            return Err(SxError::Shape(format!(
                "lightgbm: predict matrix {} x {} with {} values, model has {} features",
                x.n_rows,
                x.n_cols,
                x.data.len(),
                model.n_features
            )));
        }
        if x.n_rows == 0 {
            return Ok(Vec::new());
        }
        let n_cols = c_len(x.n_cols, "column count")?;
        let booster = load_booster(&model.text)?;
        let pred_params = format!("num_threads={} verbosity=-1", self.params.num_threads);
        let out = booster
            .predict_with_params(&x.data, n_cols, true, &pred_params)
            .map_err(|e| SxError::Model(format!("lightgbm predict: {e}")))?;
        drop(booster);
        if out.len() != x.n_rows {
            return Err(SxError::Model(format!(
                "lightgbm predict: {} outputs for {} rows",
                out.len(),
                x.n_rows
            )));
        }
        Ok(out.into_iter().map(|v| v as f32).collect())
    }
}
