// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Sequential gap fill: per target in `cfg.order` build
//! features, train (first fill) or reuse (re-fill) the model, predict and
//! write; then copy steps and shaped gap fill per well and log.
//!
//! The first fill and the re-fill share the feature build, the predict-and-write
//! pass and the ladder, so the same inputs and models give the same output.
//! Feature builds, model writes and the ladder run with rayon over wells and
//! are collected in `WellId` order; model training and prediction run on the
//! calling thread, one call at a time. One predict call per
//! target over the predicted rows of all wells (per-row results do not depend
//! on the batch).

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rayon::prelude::*;
use tracing::info;

use crate::config::FillConfig;
use crate::error::{SxError, SxResult};
use crate::fill::features::{
    any_slot_value, apply_dropout, build_well_features, feature_names, WellFeatures,
};
use crate::fill::ladder::{fleet_scale, run_copy_steps, run_shaped_fill};
use crate::fill::lgbm::{lgbm_seed, LgbmParams, LgbmRegressor};
use crate::fill::rules::apply_model_write;
use crate::fill::{
    curve, is_open, sort_report, FillInputs, FillReportRow, FillStep, ReportKind, WellFillState,
};
use crate::types::{FeatureMatrix, LogKind, Provenance, Regressor, WellId, WellLogs};

/// A fill model: a `Regressor` that may export itself as text for later re-fill.
pub trait FillModel: Regressor {
    /// Model text for saving, or `None` if the model cannot be saved (e.g. a plug-in).
    fn to_text(&self) -> SxResult<Option<String>> {
        Ok(None)
    }
}

/// `Some(to_model_text)`; `Model` error if not trained.
impl FillModel for LgbmRegressor {
    fn to_text(&self) -> SxResult<Option<String>> {
        self.to_model_text().map(Some)
    }
}

/// Makes a fresh, untrained model per target log.
pub trait RegressorFactory: Send + Sync {
    /// New untrained model for target `log`.
    fn make(&self, log: LogKind) -> SxResult<Box<dyn FillModel>>;
}

/// Built-in factory: `LgbmRegressor::new(params.clone(), lgbm_seed(project_seed, log))`.
pub struct LgbmFactory {
    /// LightGBM parameters of every model.
    pub params: LgbmParams,
    /// Project seed (`config.seed`).
    pub project_seed: u64,
}

impl RegressorFactory for LgbmFactory {
    fn make(&self, log: LogKind) -> SxResult<Box<dyn FillModel>> {
        Ok(Box::new(LgbmRegressor::new(
            self.params.clone(),
            lgbm_seed(self.project_seed, log),
        )))
    }
}

/// Trained models of one fill run. A log with too few training rows has no entry.
pub struct TrainedModels {
    /// Trained model per target log.
    pub models: BTreeMap<LogKind, Box<dyn FillModel>>,
    /// Training rows per target before dropout, also for logs without a
    /// model. Empty for models rebuilt from text.
    pub n_train_rows: BTreeMap<LogKind, usize>,
}

impl TrainedModels {
    /// Rebuild built-in models from saved texts (seed per log from `lgbm_seed`).
    ///
    /// `n_train_rows` stays empty. Errors: `Model` on a bad text.
    pub fn from_lgbm_texts(
        texts: &BTreeMap<LogKind, String>,
        params: &LgbmParams,
        project_seed: u64,
    ) -> SxResult<Self> {
        let mut models: BTreeMap<LogKind, Box<dyn FillModel>> = BTreeMap::new();
        for (&log, text) in texts {
            let m =
                LgbmRegressor::from_model_text(text, params.clone(), lgbm_seed(project_seed, log))?;
            models.insert(log, Box::new(m));
        }
        Ok(Self {
            models,
            n_train_rows: BTreeMap::new(),
        })
    }

    /// Texts of every model that can export one (for `stages/fill/models/`).
    ///
    /// Models whose `to_text` gives `None` are left out. Errors: from `to_text`.
    pub fn texts(&self) -> SxResult<BTreeMap<LogKind, String>> {
        let mut out = BTreeMap::new();
        for (&log, m) in &self.models {
            if let Some(t) = m.to_text()? {
                out.insert(log, t);
            }
        }
        Ok(out)
    }
}

/// Final logs and report of a fill run.
#[derive(Debug, Clone, PartialEq)]
pub struct FillOutput {
    /// Every filled well, all six logs, values + provenance on the well's grid.
    pub logs: BTreeMap<WellId, WellLogs>,
    /// Sorted with `fill::sort_report`.
    pub report: Vec<FillReportRow>,
}

/// First fill of every well in `inputs.neighbors`.
/// `train_exclude`: wells whose rows never enter training (holdout); empty in production.
/// Errors: `NotFound` for missing per-well inputs; `Model` from training or prediction.
///
/// Per target in `cfg.order`: features of every well from the current state;
/// training rows (measured target rows in (`WellId`, `k`) order, without
/// `train_exclude` wells and channels in `inputs.nbr.exclude`); with fewer than
/// `min_train_rows` rows no model is made and every well with gap rows gets a
/// `policy` row with reason `no_model` (over its gap rows, NaN gain and offset);
/// else dropout, `factory.make`, `fit(x, y, None)`, then prediction and the
/// model write in every well. Then copy steps and shaped gap fill per well and
/// log in suite order, with the fleet scale of each log from all wells of
/// `inputs`. Values are in working units (RESD log10) as given.
/// NaN rule: skip. Errors also: `Shape` / `InvalidInput` from the fill rules.
/// Empty `inputs.neighbors`: empty output.
pub fn train_and_fill(
    inputs: &FillInputs<'_>,
    cfg: &FillConfig,
    project_seed: u64,
    factory: &dyn RegressorFactory,
    train_exclude: &BTreeSet<WellId>,
) -> SxResult<(FillOutput, TrainedModels)> {
    let start = Instant::now();
    let wells: Vec<WellId> = inputs.neighbors.keys().copied().collect();
    let mut states = reset_states(inputs, &wells)?;
    let mut report = Vec::new();
    let mut trained = TrainedModels {
        models: BTreeMap::new(),
        n_train_rows: BTreeMap::new(),
    };
    for &target in &cfg.order {
        let t0 = Instant::now();
        let feats = build_all(inputs, &states, target, cfg)?;
        let (x, y) = training_rows(inputs, &states, &feats, target, train_exclude)?;
        let n_train = y.len();
        trained.n_train_rows.insert(target, n_train);
        if n_train > 0 && n_train >= cfg.min_train_rows {
            let (xa, ya) = apply_dropout(&x, &y, target, project_seed, cfg)?;
            drop(x);
            let mut model = factory.make(target)?;
            model.fit(&xa, &ya, None)?;
            trained.models.insert(target, model);
        }
        let model = trained.models.get(&target).map(|m| m.as_ref());
        predict_and_write(inputs, cfg, &mut states, &feats, target, model, &mut report)?;
        info!(
            log = target.name(),
            train_rows = n_train,
            model = model.is_some(),
            seconds = t0.elapsed().as_secs_f64(),
            "fill target"
        );
    }
    run_ladder(inputs, cfg, &mut states, &mut report)?;
    info!(
        wells = states.len(),
        seconds = start.elapsed().as_secs_f64(),
        "fill done"
    );
    Ok((finish(states, report), trained))
}

/// Re-fill only `wells` with already trained models. Output holds only those wells.
/// Empty `wells`: empty output.
///
/// Each listed well (repeats once) restarts from `WellFillState::reset_from`;
/// per target in `cfg.order` its features are built and predicted with the model
/// of `models` (no training, no dropout; a log without a model gets the
/// `no_model` policy rows), then the copy steps and shaped gap fill run with the
/// fleet scale from all wells of `inputs`. Same inputs and models as the first
/// fill give the same output for those wells, bit for bit.
/// NaN rule: skip. Errors: `NotFound` if a listed well is missing from
/// `inputs.neighbors` or `inputs.nbr.logs` (or from the maps the features read);
/// `Model` from prediction; `Shape` / `InvalidInput` from the fill rules.
pub fn refill_wells(
    inputs: &FillInputs<'_>,
    cfg: &FillConfig,
    models: &TrainedModels,
    wells: &[WellId],
) -> SxResult<FillOutput> {
    let start = Instant::now();
    let ids: Vec<WellId> = wells
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut states = reset_states(inputs, &ids)?;
    let mut report = Vec::new();
    if states.is_empty() {
        return Ok(finish(states, report));
    }
    for &target in &cfg.order {
        let feats = build_all(inputs, &states, target, cfg)?;
        let model = models.models.get(&target).map(|m| m.as_ref());
        predict_and_write(inputs, cfg, &mut states, &feats, target, model, &mut report)?;
    }
    run_ladder(inputs, cfg, &mut states, &mut report)?;
    info!(
        wells = states.len(),
        seconds = start.elapsed().as_secs_f64(),
        "fill refill done"
    );
    Ok(finish(states, report))
}

/// For `wells` (sorted, unique), in that order.
fn reset_states(inputs: &FillInputs<'_>, wells: &[WellId]) -> SxResult<Vec<WellFillState>> {
    let built: Vec<SxResult<WellFillState>> = wells
        .par_iter()
        .map(|&w| {
            if !inputs.neighbors.contains_key(&w) {
                return Err(SxError::NotFound(format!(
                    "fill: well {} has no neighbor curves",
                    w.0
                )));
            }
            let logs = inputs
                .nbr
                .logs
                .get(&w)
                .ok_or_else(|| SxError::NotFound(format!("fill: well {} has no logs", w.0)))?;
            if logs.id != w {
                return Err(SxError::InvalidInput(format!(
                    "fill: logs key {} holds well {}",
                    w.0, logs.id.0
                )));
            }
            WellFillState::reset_from(logs, inputs.nbr.coverage.get(&w).copied())
        })
        .collect();
    built.into_iter().collect()
}

/// Features of every well for `target`, aligned with `states`.
fn build_all(
    inputs: &FillInputs<'_>,
    states: &[WellFillState],
    target: LogKind,
    cfg: &FillConfig,
) -> SxResult<Vec<WellFeatures>> {
    let built: Vec<SxResult<WellFeatures>> = states
        .par_iter()
        .map(|s| build_well_features(inputs, s, target, cfg))
        .collect();
    built.into_iter().collect()
}

/// Rows in (`WellId`, `k`) order.
fn training_rows(
    inputs: &FillInputs<'_>,
    states: &[WellFillState],
    feats: &[WellFeatures],
    target: LogKind,
    train_exclude: &BTreeSet<WellId>,
) -> SxResult<(FeatureMatrix, Vec<f32>)> {
    let mut data = Vec::new();
    let mut y = Vec::new();
    for (state, f) in states.iter().zip(feats) {
        let w = state.logs.id;
        if train_exclude.contains(&w) || inputs.nbr.exclude.contains(&(w, target)) {
            continue;
        }
        let (values, prov) = curve(&state.logs, target)?;
        for (r, &i) in f.rows.iter().enumerate() {
            let (Some(&v), Some(&p)) = (values.get(i), prov.get(i)) else {
                continue;
            };
            if p != Provenance::Measured.code() || !v.is_finite() {
                continue;
            }
            if let Some(row) = f.x.row(r) {
                data.extend_from_slice(row);
                y.push(v);
            }
        }
    }
    Ok((FeatureMatrix::new(y.len(), feature_names(target), data)?, y))
}

/// For `target` in every well; `None` model → `no_model` policy rows.
fn predict_and_write(
    inputs: &FillInputs<'_>,
    cfg: &FillConfig,
    states: &mut [WellFillState],
    feats: &[WellFeatures],
    target: LogKind,
    model: Option<&dyn FillModel>,
    report: &mut Vec<FillReportRow>,
) -> SxResult<()> {
    let mut gaps: Vec<Vec<usize>> = Vec::with_capacity(states.len());
    for s in states.iter() {
        let (values, prov) = curve(&s.logs, target)?;
        gaps.push(
            values
                .iter()
                .zip(prov)
                .enumerate()
                .filter(|(_, (&v, &p))| is_open(v, p))
                .map(|(i, _)| i)
                .collect(),
        );
    }
    let Some(model) = model else {
        for (s, g) in states.iter().zip(&gaps) {
            if g.is_empty() {
                continue;
            }
            let mut row = FillReportRow::over_rows(
                ReportKind::Policy,
                s,
                &well_name(inputs, s.logs.id),
                target,
                FillStep::Model,
                g,
            );
            row.reason = "no_model".to_string();
            report.push(row);
        }
        return Ok(());
    };

    let mut picked: Vec<Vec<usize>> = Vec::with_capacity(feats.len());
    let mut data = Vec::new();
    for (f, g) in feats.iter().zip(&gaps) {
        let mut rs = Vec::new();
        if !g.is_empty() {
            for (r, row) in f.x.data.chunks_exact(f.x.n_cols.max(1)).enumerate() {
                if any_slot_value(row) {
                    data.extend_from_slice(row);
                    rs.push(r);
                }
            }
        }
        picked.push(rs);
    }
    let total: usize = picked.iter().map(Vec::len).sum();
    let pred = if total == 0 {
        Vec::new()
    } else {
        model.predict(&FeatureMatrix::new(total, feature_names(target), data)?)?
    };
    if pred.len() != total {
        return Err(SxError::Model(format!(
            "fill {}: model {} gave {} predictions for {total} rows",
            target.name(),
            model.name(),
            pred.len()
        )));
    }

    let mut per_well: Vec<Option<Vec<f32>>> = Vec::with_capacity(states.len());
    let mut offset = 0;
    for ((s, f), (g, rs)) in states.iter().zip(feats).zip(gaps.iter().zip(&picked)) {
        if g.is_empty() {
            per_well.push(None);
            continue;
        }
        let mut p = vec![f32::NAN; s.logs.grid.len];
        for (k, &r) in rs.iter().enumerate() {
            let (Some(&i), Some(&v)) = (f.rows.get(r), pred.get(offset + k)) else {
                continue;
            };
            if let Some(slot) = p.get_mut(i) {
                *slot = v;
            }
        }
        offset += rs.len();
        per_well.push(Some(p));
    }

    let written: Vec<SxResult<Vec<FillReportRow>>> = states
        .par_iter_mut()
        .zip(per_well.par_iter())
        .map(|(s, p)| {
            let Some(p) = p else {
                return Ok(Vec::new());
            };
            let w = s.logs.id;
            let nbr = inputs.neighbors.get(&w).and_then(|c| c.curves.get(&target));
            apply_model_write(s, target, p, nbr, &well_name(inputs, w), cfg)
        })
        .collect();
    for rows in written {
        report.extend(rows?);
    }
    Ok(())
}

/// Per well (rayon), per log in suite order.
fn run_ladder(
    inputs: &FillInputs<'_>,
    cfg: &FillConfig,
    states: &mut [WellFillState],
    report: &mut Vec<FillReportRow>,
) -> SxResult<()> {
    let scales: Vec<(LogKind, f64)> = LogKind::ALL
        .iter()
        .map(|&log| (log, fleet_scale(inputs, log, cfg)))
        .collect();
    let done: Vec<SxResult<Vec<FillReportRow>>> = states
        .par_iter_mut()
        .map(|s| {
            let mut rows = Vec::new();
            for &(log, scale) in &scales {
                rows.extend(run_copy_steps(inputs, s, log, cfg)?);
                rows.extend(run_shaped_fill(inputs, s, log, scale, cfg)?);
            }
            Ok(rows)
        })
        .collect();
    for rows in done {
        report.extend(rows?);
    }
    Ok(())
}

fn finish(states: Vec<WellFillState>, mut report: Vec<FillReportRow>) -> FillOutput {
    sort_report(&mut report);
    FillOutput {
        logs: states.into_iter().map(|s| (s.logs.id, s.logs)).collect(),
        report,
    }
}

fn well_name(inputs: &FillInputs<'_>, w: WellId) -> String {
    inputs
        .nbr
        .headers
        .get(&w)
        .map(|h| h.name.clone())
        .unwrap_or_default()
}
