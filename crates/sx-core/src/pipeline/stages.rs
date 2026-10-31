// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! One runner per stage: load inputs from files, call the stage entry point,
//! write every output atomically, return the outputs with their hashes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use serde_json::{json, Map, Value};

use super::assemble::{CoordChoice, NeighborData};
use super::atomic::write_atomic;
use super::models::OverridingFactory;
use super::products::*;
use super::readers::{
    read_estimated_picks, read_flags, read_geometry, read_intervals, read_layer_coord,
    read_layer_time, read_logs, read_logs_with_prov, read_models, read_neighbor_curves, read_pairs,
    read_prior, read_tops_input, read_wells,
};
use super::stage::Stage;
use super::ModelOverrides;
use crate::align::{self, AlignInput};
use crate::config::ProjectConfig;
use crate::correlate::{self, CorrelationConfig, CorrelationInputs};
use crate::error::{SxError, SxResult};
use crate::fill::features::feature_names;
use crate::fill::lgbm::{lgbm_seed, LgbmParams};
use crate::fill::sequential::{train_and_fill, LgbmFactory, TrainedModels};
use crate::fill::{
    fill_report_csv, fill_summary_csv, logs_filled_to_batch, FillInputs, LOGS_FILLED_TAG,
};
use crate::io::atomic::io_error;
use crate::io::products::{
    geometry_batch, logs_batch, tops_input_batch, wells_batch, LOGS_RAW_TAG,
};
use crate::io::{csv, glossary, las, parquet};
use crate::neighbors::{build_neighbor_curves, neighbor_curves_to_batch};
use crate::normalize::{
    logs_norm_batch, norm_anchors_batch, norm_report_batch, nphi_offset_batch, run_normalize_stage,
    NormalizeInput, QC_FLAGS_POST_TAG,
};
use crate::qc::{self, QcInput, LOGS_QC_TAG, QC_FLAGS_TAG};
use crate::refresh::{self, RefreshInputs, LOGS_FINAL_TAG};
use crate::types::{LogKind, WellGeometry, WellHeader, WellId, WellLogs};
use crate::{ingest, ingest::ingest_report_batch};

/// What a stage runner gets.
pub(crate) struct Ctx<'a> {
    pub(crate) project: &'a Path,
    pub(crate) cfg: &'a ProjectConfig,
    pub(crate) models: &'a ModelOverrides,
    /// Models trained by fill in this run.
    pub(crate) trained: Option<Arc<TrainedModels>>,
    /// Saved model texts of the fill record, used when fill did not run now.
    pub(crate) model_files: Vec<(LogKind, PathBuf)>,
}

/// What a stage runner wrote.
pub(crate) struct StageRun {
    /// `(relative path, SHA-256)` of every output.
    pub(crate) outputs: Vec<(String, String)>,
    /// Models trained by fill.
    pub(crate) trained: Option<Arc<TrainedModels>>,
}

/// Collects outputs written through [`write_atomic`].
struct Out<'a> {
    project: &'a Path,
    files: Vec<(String, String)>,
}

impl<'a> Out<'a> {
    fn new(project: &'a Path) -> Self {
        Self {
            project,
            files: Vec::new(),
        }
    }

    fn put(&mut self, rel: &str, write: impl FnOnce(&Path) -> SxResult<()>) -> SxResult<()> {
        let hash = write_atomic(&self.project.join(rel), write)?;
        self.files.push((rel.to_string(), hash));
        Ok(())
    }

    fn parquet(&mut self, rel: &str, batch: &RecordBatch) -> SxResult<()> {
        self.put(rel, |t| parquet::write_batch(t, batch))
    }

    fn csv(&mut self, rel: &str, batch: &RecordBatch) -> SxResult<()> {
        self.put(rel, |t| csv::write_batch(t, batch))
    }

    fn text(&mut self, rel: &str, text: &str) -> SxResult<()> {
        self.put(rel, |t| std::fs::write(t, text).map_err(|e| io_error(t, e)))
    }

    fn done(self) -> StageRun {
        StageRun {
            outputs: self.files,
            trained: None,
        }
    }
}

/// Run `stage` on the files of `ctx.project`.
///
/// # Errors
/// Reader errors, the stage entry point's errors, `SxError::Io` from writes.
pub(crate) fn run_stage(stage: Stage, ctx: &Ctx<'_>) -> SxResult<StageRun> {
    match stage {
        Stage::Ingest => run_ingest(ctx),
        Stage::Qc => run_qc(ctx),
        Stage::RoughAlign => run_rough_align(ctx),
        Stage::Normalize => run_normalize(ctx),
        Stage::Neighbors => run_neighbors(ctx),
        Stage::Fill => run_fill(ctx),
        Stage::Correlate => run_correlate(ctx),
        Stage::Subhorizons => run_subhorizons(ctx),
        Stage::Refresh => run_refresh(ctx),
        Stage::Export => run_export(ctx),
    }
}

/// Headers and geometry of exactly `ids` (ascending), same order.
fn wells_with_logs(
    headers: Vec<WellHeader>,
    geometry: Vec<WellGeometry>,
    ids: &[WellId],
) -> SxResult<(Vec<WellHeader>, Vec<WellGeometry>)> {
    let mut h: BTreeMap<WellId, WellHeader> = headers.into_iter().map(|x| (x.id, x)).collect();
    let mut g: BTreeMap<WellId, WellGeometry> = geometry.into_iter().map(|x| (x.id, x)).collect();
    let mut hs = Vec::with_capacity(ids.len());
    let mut gs = Vec::with_capacity(ids.len());
    for id in ids {
        match (h.remove(id), g.remove(id)) {
            (Some(a), Some(b)) => {
                hs.push(a);
                gs.push(b);
            }
            _ => {
                return Err(SxError::Shape(format!(
                    "well {} has logs but no header or geometry",
                    id.0
                )))
            }
        }
    }
    Ok((hs, gs))
}

fn run_ingest(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let out = ingest::run_ingest(&ctx.cfg.ingest, ctx.project)?;
    let grids: BTreeMap<WellId, _> = out.geometry.iter().map(|g| (g.id, g.grid)).collect();
    if grids.len() != out.logs.len() || grids.len() != out.geometry.len() {
        return Err(SxError::Shape(
            "ingest: geometry and logs cover different wells".into(),
        ));
    }
    for l in &out.logs {
        if grids.get(&l.id) != Some(&l.grid) {
            return Err(SxError::Shape(format!(
                "ingest: well {}: geometry and logs grids differ",
                l.id.0
            )));
        }
    }
    let mut w = Out::new(ctx.project);
    w.parquet(WELLS, &wells_batch(&out.well_rows)?)?;
    w.parquet(GEOMETRY, &geometry_batch(&out.geometry)?)?;
    w.parquet(LOGS_RAW, &logs_batch(&out.logs, LOGS_RAW_TAG)?)?;
    w.parquet(
        AUX_RAW,
        &crate::io::products::aux_batch(&out.aux, crate::io::products::AUX_RAW_TAG)?,
    )?;
    w.parquet(TOPS_INPUT, &tops_input_batch(&out.tops, &out.picks)?)?;
    w.csv(INGEST_REPORT, &ingest_report_batch(&out.report)?)?;
    w.csv(LAS_MATCH, &ingest::las_match_batch(&out.las_matches)?)?;
    Ok(w.done())
}

fn run_qc(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let p = |rel: &str| ctx.project.join(rel);
    let (headers, _) = read_wells(&p(WELLS))?;
    let raw = read_logs(&p(LOGS_RAW), LOGS_RAW_TAG)?;
    let aux = super::readers::read_aux(&p(AUX_RAW))?;
    let out = qc::run_qc_stage(
        &QcInput {
            wells: &headers,
            logs: &raw,
            aux: &aux,
        },
        &ctx.cfg.qc,
    )?;
    let mut w = Out::new(ctx.project);
    w.parquet(LOGS_QC, &qc::logs_batch(&out.logs, LOGS_QC_TAG)?)?;
    w.parquet(QC_FLAGS, &qc::flags_batch(&out.flags, QC_FLAGS_TAG)?)?;
    w.csv(QC_REPORT, &qc::qc_report_batch(&out.report)?)?;
    w.csv(
        QC_LOW_DENSITY,
        &qc::low_density_batch(&out.low_density_unverified)?,
    )?;
    w.parquet(
        QC_RHOB_CLOSED_CAL,
        &qc::closed_cal_samples_batch(&out.closed_caliper_samples)?,
    )?;
    w.csv(
        QC_CLOSED_CAL_RUNS,
        &qc::closed_cal_runs_batch(&out.closed_caliper_runs)?,
    )?;
    // LAS match report: ingest's conflicts and overrides, published with the QC reports.
    let las_match = p(LAS_MATCH);
    let text = std::fs::read_to_string(&las_match).map_err(|e| io_error(&las_match, e))?;
    let n_left_out = text
        .lines()
        .skip(1)
        .filter(|l| l.contains(",excluded,"))
        .count();
    if n_left_out > 0 {
        tracing::warn!(
            n_files = n_left_out,
            report = QC_LAS_MATCH,
            "LAS files left out as ambiguous; see the report and [ingest.las_overrides]"
        );
    }
    w.text(QC_LAS_MATCH, &text)?;
    Ok(w.done())
}

fn run_rough_align(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let p = |rel: &str| ctx.project.join(rel);
    let (headers, _) = read_wells(&p(WELLS))?;
    let geometry = read_geometry(&p(GEOMETRY))?;
    let (_, picks) = read_tops_input(&p(TOPS_INPUT))?;
    let logs = read_logs(&p(LOGS_QC), LOGS_QC_TAG)?;
    let ids: Vec<WellId> = logs.iter().map(|l| l.id).collect();
    let (headers, geometry) = wells_with_logs(headers, geometry, &ids)?;
    let r = align::rough_align(
        &AlignInput {
            headers: &headers,
            geometry: &geometry,
            logs: &logs,
            picks: &picks,
            order: &ctx.cfg.ingest.top_names,
        },
        &ctx.cfg.align,
        &ctx.cfg.dtw.interval,
    )?;
    let mut w = Out::new(ctx.project);
    w.parquet(INTERVALS, &align::intervals_batch(&r)?)?;
    w.parquet(TOPS_ESTIMATED, &align::tops_estimated_batch(&r)?)?;
    w.parquet(PAIRS_ROUGH, &align::pairs_rough_batch(&r)?)?;
    w.parquet(LAYER_COORD, &align::layer_coord_batch(&r)?)?;
    Ok(w.done())
}

fn run_normalize(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let p = |rel: &str| ctx.project.join(rel);
    let (headers, _) = read_wells(&p(WELLS))?;
    let geometry = read_geometry(&p(GEOMETRY))?;
    let logs_qc = read_logs(&p(LOGS_QC), LOGS_QC_TAG)?;
    let flags = read_flags(&p(QC_FLAGS), QC_FLAGS_TAG)?;
    let prior = read_prior(&p(QC_REPORT))?;
    let (_, intervals) = read_intervals(&p(INTERVALS))?;
    let pairs = read_pairs(&p(PAIRS_ROUGH))?;
    let (layer, _) = read_layer_coord(&p(LAYER_COORD))?;
    let out = run_normalize_stage(
        &NormalizeInput {
            wells: &headers,
            logs_qc: &logs_qc,
            flags: &flags,
            prior: &prior,
            geometry: &geometry,
            intervals: &intervals,
            layer: &layer,
            pairs: &pairs,
        },
        ctx.cfg,
    )?;
    let mut w = Out::new(ctx.project);
    w.parquet(LOGS_NORM, &logs_norm_batch(&out.logs)?)?;
    w.parquet(
        QC_FLAGS_POST,
        &qc::flags_batch(&out.flags, QC_FLAGS_POST_TAG)?,
    )?;
    w.csv(NORM_ANCHORS, &norm_anchors_batch(&out.anchors)?)?;
    w.csv(NORM_REPORT, &norm_report_batch(&out.report)?)?;
    w.csv(NPHI_OFFSET_REPORT, &nphi_offset_batch(&out.nphi_offset)?)?;
    Ok(w.done())
}

fn run_neighbors(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let data = NeighborData::load(ctx.project, ctx.cfg, CoordChoice::Rough)?;
    let targets: Vec<WellId> = data.logs.keys().copied().collect();
    let curves = build_neighbor_curves(&data.inputs(), &targets, &ctx.cfg.neighbors)?;
    let mut w = Out::new(ctx.project);
    w.parquet(NEIGHBOR_CURVES, &neighbor_curves_to_batch(&curves)?)?;
    Ok(w.done())
}

/// `models.json`: per trained log its model name, text file, LightGBM seed,
/// training rows, feature names and `[fill.lightgbm]`.
fn models_json(
    trained: &TrainedModels,
    texts: &BTreeMap<LogKind, String>,
    ctx: &Ctx<'_>,
) -> SxResult<String> {
    let lightgbm =
        serde_json::to_value(&ctx.cfg.fill.lightgbm).map_err(|e| SxError::Config(e.to_string()))?;
    let mut models = Map::new();
    for &log in trained.models.keys() {
        let model = ctx
            .models
            .get(&log)
            .map_or_else(|| "lightgbm".to_string(), |(d, _)| d.clone());
        let file = texts
            .contains_key(&log)
            .then(|| format!("models/{}.txt", log.name()));
        models.insert(
            log.name().to_string(),
            json!({
                "model": model,
                "file": file,
                "seed": lgbm_seed(ctx.cfg.seed, log),
                "n_train_rows": trained.n_train_rows.get(&log).copied().unwrap_or(0),
                "features": feature_names(log),
                "lightgbm": lightgbm.clone(),
            }),
        );
    }
    let value = json!({ "schema": "subsurfix.fill_models/1", "models": Value::Object(models) });
    let mut text = serde_json::to_string_pretty(&value)
        .map_err(|e| SxError::Format(format!("models.json: {e}")))?;
    text.push('\n');
    Ok(text)
}

fn run_fill(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let cfg = ctx.cfg;
    let data = NeighborData::load(ctx.project, cfg, CoordChoice::Rough)?;
    let curves = read_neighbor_curves(&ctx.project.join(NEIGHBOR_CURVES))?;
    let inputs = FillInputs {
        nbr: data.inputs(),
        neighbors: &curves,
        resd_family: &data.resd_family,
        coord_source: &data.coord_source,
        neighbors_cfg: &cfg.neighbors,
    };
    let base = LgbmFactory {
        params: LgbmParams::from_config(&cfg.fill.lightgbm),
        project_seed: cfg.seed,
    };
    let none = BTreeSet::new();
    let (out, trained) = if ctx.models.is_empty() {
        train_and_fill(&inputs, &cfg.fill, cfg.seed, &base, &none)?
    } else {
        let factory = OverridingFactory {
            base,
            overrides: ctx.models,
        };
        train_and_fill(&inputs, &cfg.fill, cfg.seed, &factory, &none)?
    };
    let texts = trained.texts()?;
    let mut w = Out::new(ctx.project);
    w.parquet(LOGS_FILLED, &logs_filled_to_batch(&out.logs)?)?;
    w.text(FILL_REPORT, &fill_report_csv(&out.report))?;
    let names: BTreeMap<WellId, &str> = out
        .report
        .iter()
        .map(|r| (r.well_id, r.well_name.as_str()))
        .collect();
    w.text(
        FILL_SUMMARY,
        &fill_summary_csv(&out.logs, &names, cfg.fill.mostly_predicted_share),
    )?;
    for (log, text) in &texts {
        w.text(&format!("{MODELS_DIR}/{}.txt", log.name()), text)?;
    }
    w.text(MODELS_JSON, &models_json(&trained, &texts, ctx)?)?;
    let mut run = w.done();
    run.trained = Some(Arc::new(trained));
    Ok(run)
}

/// Headers, geometry and filled logs of the wells in `logs_filled`, plus the
/// study top list and interval rows.
struct Filled {
    headers: Vec<WellHeader>,
    geometry: Vec<WellGeometry>,
    logs: Vec<WellLogs>,
    tops: crate::types::TopSet,
    intervals: Vec<crate::types::IntervalRow>,
}

fn load_filled(project: &Path) -> SxResult<Filled> {
    let p = |rel: &str| project.join(rel);
    let (headers, _) = read_wells(&p(WELLS))?;
    let geometry = read_geometry(&p(GEOMETRY))?;
    let filled = read_logs_with_prov(&p(LOGS_FILLED), LOGS_FILLED_TAG)?;
    let (tops, intervals) = read_intervals(&p(INTERVALS))?;
    let ids: Vec<WellId> = filled.keys().copied().collect();
    let (headers, geometry) = wells_with_logs(headers, geometry, &ids)?;
    Ok(Filled {
        headers,
        geometry,
        logs: filled.into_values().collect(),
        tops,
        intervals,
    })
}

fn run_correlate(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let f = load_filled(ctx.project)?;
    let picks = read_estimated_picks(&ctx.project.join(TOPS_ESTIMATED), f.tops.len())?;
    let cc = &ctx.cfg.correlate;
    let out = correlate::run_correlation(
        &CorrelationInputs {
            headers: &f.headers,
            geometry: &f.geometry,
            logs: &f.logs,
            tops: &f.tops,
            intervals: &f.intervals,
            picks: &picks,
        },
        &CorrelationConfig {
            graph: &cc.graph,
            solve: &cc.solve,
            tops: &cc.tops,
            dtw: &ctx.cfg.dtw,
        },
    )?;
    let table = correlate::tops::build_tops_table(
        &f.headers,
        &f.geometry,
        &f.tops,
        &out.tops_check,
        &out.boundaries_pass1,
        &out.boundaries_pass2,
    )?;
    let tops = correlate::tops::tops_batch(&table)?;
    let mut w = Out::new(ctx.project);
    w.parquet(
        LAYER_TIME,
        &correlate::solve::layer_time_batch(&out.layer_time)?,
    )?;
    w.parquet(TOPS, &tops)?;
    w.csv(TOPS_CSV, &tops)?;
    w.csv(SUSPECTS, &correlate::tops::suspects_batch(&table)?)?;
    w.csv(
        SOLVE_REPORT,
        &correlate::solve::solve_report_batch(&out.solve_report)?,
    )?;
    Ok(w.done())
}

fn run_subhorizons(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let f = load_filled(ctx.project)?;
    let layer_time = read_layer_time(&ctx.project.join(LAYER_TIME))?;
    let sh = correlate::subhorizons::propose_subhorizons(
        &f.headers,
        &f.geometry,
        &f.logs,
        &layer_time,
        &f.intervals,
        &f.tops,
        &ctx.cfg.subhorizons,
    )?;
    let batch = correlate::subhorizons::subhorizons_batch(&sh, &f.headers)?;
    let mut w = Out::new(ctx.project);
    w.parquet(SUBHORIZONS, &batch)?;
    w.csv(SUBHORIZONS_CSV, &batch)?;
    Ok(w.done())
}

fn run_refresh(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let cfg = ctx.cfg;
    let data = NeighborData::load(ctx.project, cfg, CoordChoice::LayerTime)?;
    let filled = read_logs_with_prov(&ctx.project.join(LOGS_FILLED), LOGS_FILLED_TAG)?;
    let models = match &ctx.trained {
        Some(m) => Arc::clone(m),
        None => Arc::new(read_models(&ctx.model_files, cfg)?),
    };
    let out = refresh::run_refresh(
        RefreshInputs {
            nbr_tau: data.inputs(),
            resd_family: &data.resd_family,
            coord_source: &data.coord_source,
            n_tops: data.tops.len(),
            layer_coord: &data.rough_u,
            logs_filled: &filled,
        },
        &models,
        &cfg.refresh,
        &cfg.neighbors,
        &cfg.fill,
    )?;
    let mut w = Out::new(ctx.project);
    w.parquet(LOGS_FINAL, &refresh::logs_final_batch(&out.logs)?)?;
    w.csv(REFRESH_REPORT, &refresh::report_batch(&out.report)?)?;
    Ok(w.done())
}

/// Well name with every char outside `[A-Za-z0-9._-]` replaced by `_`, cut to 64 chars.
fn safe_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn parquet_schema(path: &Path) -> SxResult<SchemaRef> {
    let file = File::open(path).map_err(|e| io_error(path, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        SxError::Format(format!(
            "{}: not a readable Parquet file: {e}",
            path.display()
        ))
    })?;
    Ok(builder.schema().clone())
}

fn run_export(ctx: &Ctx<'_>) -> SxResult<StageRun> {
    let p = |rel: &str| ctx.project.join(rel);
    let mut w = Out::new(ctx.project);
    if ctx.cfg.export.las {
        let (headers, _) = read_wells(&p(WELLS))?;
        let geometry = read_geometry(&p(GEOMETRY))?;
        let finals = read_logs_with_prov(&p(LOGS_FINAL), LOGS_FINAL_TAG)?;
        let ids: Vec<WellId> = finals.keys().copied().collect();
        let (headers, geometry) = wells_with_logs(headers, geometry, &ids)?;
        let jobs: Vec<(String, &WellHeader, &WellGeometry, &WellLogs)> = headers
            .iter()
            .zip(&geometry)
            .zip(finals.values())
            .map(|((h, g), l)| {
                let rel = format!("{EXPORT_LAS_DIR}/{}_{}.las", h.id.0, safe_name(&h.name));
                (rel, h, g, l)
            })
            .collect();
        let project = ctx.project;
        let written: Vec<SxResult<(String, String)>> = jobs
            .par_iter()
            .map(|(rel, h, g, l)| {
                let hash = write_atomic(&project.join(rel), |t| las::write_well_las(t, h, g, l))?;
                Ok((rel.clone(), hash))
            })
            .collect();
        let mut written: Vec<(String, String)> = written.into_iter().collect::<SxResult<_>>()?;
        written.sort();
        w.files.extend(written);
    }
    if ctx.cfg.export.parquet {
        for (_, src, dst) in EXPORT_COPIES {
            let from = p(src);
            w.put(dst, |t| {
                std::fs::copy(&from, t)
                    .map(|_| ())
                    .map_err(|e| io_error(&from, e))
            })?;
        }
    }
    let schemas: Vec<(&str, SchemaRef)> = EXPORT_COPIES
        .iter()
        .map(|(name, src, _)| Ok((*name, parquet_schema(&p(src))?)))
        .collect::<SxResult<_>>()?;
    let refs: Vec<(&str, &arrow::datatypes::Schema)> =
        schemas.iter().map(|(n, s)| (*n, s.as_ref())).collect();
    w.csv(EXPORT_GLOSSARY, &glossary::column_glossary(&refs)?)?;
    Ok(w.done())
}
