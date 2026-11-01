// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Leak-free holdout scoring of the gap fill.
//!
//! [`score_holdout`] reads the finished stage products of a
//! project, picks wells, masks their measured rows in `logs_qc`
//!, reruns normalization, neighbor curves (held-out wells donate
//! nothing) and the fill (no row of a held-out well trains a model), and
//! scores the masked rows. Rough alignment and every other product
//! are reused as read. Truth of a masked row is its production `logs_norm`
//! value (measured and finite there), in the fill's working units (RESD
//! log10). Outputs go to [`HOLDOUT_DIR`] only; production products are never
//! written.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::{info, warn};

use crate::config::{FillConfig, ProjectConfig};
use crate::error::{SxError, SxResult};
use crate::fill::lgbm::LgbmParams;
use crate::fill::sequential::{train_and_fill, LgbmFactory};
use crate::fill::{curve, is_fill_code, pop_std, FillInputs, FillStep};
use crate::hash::stream_seed;
use crate::io::atomic::write_atomic;
use crate::io::csv;
use crate::neighbors::build_neighbor_curves;
use crate::normalize::{run_normalize_stage, NormalizeInput, NormalizeOutput};
use crate::pipeline::assemble::{CoordChoice, NeighborData};
use crate::pipeline::products::{
    GEOMETRY, INTERVALS, LAYER_COORD, LOGS_QC, PAIRS_ROUGH, QC_FLAGS, QC_REPORT, WELLS,
};
use crate::pipeline::readers::{
    read_flags, read_geometry, read_intervals, read_layer_coord, read_logs, read_pairs, read_prior,
    read_wells,
};
use crate::qc::{LOGS_QC_TAG, QC_FLAGS_TAG};
use crate::types::{LogKind, Provenance, WellHeader, WellId, WellLogs, CODE_NONE};
use crate::util::regression_metrics;
use crate::util::rng::{rng_for, shuffle};

/// Folder of the holdout products, relative to the project folder.
pub const HOLDOUT_DIR: &str = "stages/fill/holdout";
/// Scores per (well, log, scope).
pub const SCORES_CSV: &str = "holdout_scores.csv";
/// Scores per (log, scope), pooled over the log's held-out wells.
pub const SUMMARY_CSV: &str = "holdout_summary.csv";

const PICK_STREAM: &str = "holdout.pick";
const MASK_STREAM: &str = "holdout.mask.";
/// Centered rolling window of `texture_ratio`: grid rows `j − 10 ..= j + 9`.
const TEXTURE_BEFORE: usize = 10;
const TEXTURE_AFTER: usize = 9;
const TEXTURE_MIN: usize = 10;
const EXACT_RUN_MIN: usize = 20;

/// Fill steps whose share of the scored rows is reported, in report order.
pub const SHARE_STEPS: [FillStep; 9] = [
    FillStep::Model,
    FillStep::NeighborWeighted,
    FillStep::RealDonor,
    FillStep::NeighborWeak,
    FillStep::NeighborShape,
    FillStep::CrossLogFit,
    FillStep::DonorShape,
    FillStep::OwnTexture,
    FillStep::SmoothedModel,
];

/// One well offered to [`pick_holdout_wells`].
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutCandidate {
    /// Well id.
    pub id: WellId,
    /// Surface x, m.
    pub x_m: f64,
    /// Surface y, m.
    pub y_m: f64,
    /// Measured samples per log; a missing log has none.
    pub measured: BTreeMap<LogKind, usize>,
}

impl HoldoutCandidate {
    fn count(&self, log: LogKind) -> usize {
        self.measured.get(&log).copied().unwrap_or(0)
    }

    fn n_eligible(&self, min: usize) -> usize {
        LogKind::ALL
            .iter()
            .filter(|&&l| self.count(l) >= min)
            .count()
    }

    fn total(&self) -> usize {
        self.measured.values().sum()
    }
}

/// Wells picked by [`pick_holdout_wells`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HoldoutPick {
    /// Round-robin picks in pick order.
    pub wells: Vec<WellId>,
    /// Held-out wells per log, ascending (picks eligible for the log plus the
    /// top-up wells). Logs without any eligible well are absent.
    pub per_log: BTreeMap<LogKind, Vec<WellId>>,
}

impl HoldoutPick {
    /// Every well with a held-out log: it donates nothing and never trains.
    pub fn held_out(&self) -> BTreeSet<WellId> {
        self.per_log.values().flatten().copied().collect()
    }
}

/// Rows masked in one (well, log) channel by [`mask_logs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskedChannel {
    /// Measured rows of the channel before masking (`F`).
    pub n_measured: usize,
    /// Block length in measured-row positions (`F` in `whole_log` mode).
    pub block: usize,
    /// Blocks drawn; blocks may overlap (1 in `whole_log` mode).
    pub n_blocks: usize,
    /// Masked local grid rows, ascending.
    pub rows: Vec<usize>,
}

/// Scored rows of a holdout score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldoutScope {
    /// Rows written by the model (provenance 1).
    Model,
    /// Rows written by any fill rule.
    Final,
}

impl HoldoutScope {
    /// CSV text (`model`, `final`).
    pub fn name(self) -> &'static str {
        match self {
            HoldoutScope::Model => "model",
            HoldoutScope::Final => "final",
        }
    }
}

/// Scores of one scope. NaN where undefined (no rows, zero truth spread).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeScores {
    /// Scored rows.
    pub n: usize,
    /// `1 − SS_res / SS_tot`.
    pub r2: f64,
    /// Root mean squared error, log unit.
    pub rmse: f64,
    /// Mean of `pred − truth`, log unit.
    pub bias: f64,
    /// Population std of pred over that of truth.
    pub spread_ratio: f64,
    /// Least-squares slope of pred on truth.
    pub slope: f64,
    /// Least-squares intercept of pred on truth, log unit.
    pub intercept: f64,
    /// Median centered rolling sample std of pred over that of truth.
    pub texture_ratio: f64,
    /// Exactly constant runs of at least 20 consecutive rows in pred.
    pub exact_runs: usize,
}

/// Holdout scores of one (well, log), or of one log pooled over its wells.
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutScore {
    /// Well id; `None` on a per-log summary.
    pub well_id: Option<WellId>,
    /// Well name; empty on a per-log summary.
    pub well_name: String,
    /// Log.
    pub log: LogKind,
    /// Wells pooled (1 for one well).
    pub n_wells: usize,
    /// Masked rows (measured in `logs_qc`).
    pub n_masked: usize,
    /// Masked rows measured and finite in the production `logs_norm` (scored rows).
    pub n_truth: usize,
    /// Share of the scored rows left `Unfilled`.
    pub unfilled_share: f64,
    /// Share of the scored rows written by each step of [`SHARE_STEPS`], same order.
    pub step_share: Vec<(FillStep, f64)>,
    /// `model` scope.
    pub model_scope: ScopeScores,
    /// `final` scope.
    pub final_scope: ScopeScores,
}

/// Result of [`score_holdout`].
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutReport {
    /// Picked wells.
    pub pick: HoldoutPick,
    /// Masked channels.
    pub masked: BTreeMap<(WellId, LogKind), MaskedChannel>,
    /// One per masked channel, by (well, log).
    pub scores: Vec<HoldoutScore>,
    /// One per held-out log, suite order.
    pub summary: Vec<HoldoutScore>,
    /// Written `holdout_scores.csv`.
    pub scores_path: PathBuf,
    /// Written `holdout_summary.csv`.
    pub summary_path: PathBuf,
}

/// Wells with at least `holdout_min_samples` measured samples of
/// a log are eligible for it; candidates (eligible for any log) fall into 4
/// tiles split at the median surface `x_m` and `y_m` (`<=` median = low bin,
/// median of an even count = mean of the two middle values, non-finite
/// coordinates are left out of the median and go to the low bin); tile `t =
/// 2·x_bin + y_bin` order is shuffled with `holdout.pick`; round robin over the
/// tiles gives each tile's best remaining well (most eligible logs, then most
/// measured samples in total, then lower id) until `holdout_n_wells` are
/// picked. Per log, wells eligible for it are added nearest-first (surface
/// distance to the nearest pick, tie → lower id) up to `holdout_n_wells`.
///
/// Repeated ids are taken once. Empty input: empty pick. Errors: none.
pub fn pick_holdout_wells(
    wells: &[HoldoutCandidate],
    cfg: &FillConfig,
    project_seed: u64,
) -> HoldoutPick {
    let min = cfg.holdout_min_samples;
    let n = cfg.holdout_n_wells;
    let mut cands: Vec<&HoldoutCandidate> =
        wells.iter().filter(|c| c.n_eligible(min) > 0).collect();
    cands.sort_by_key(|c| c.id);
    cands.dedup_by_key(|c| c.id);
    let mx = median(cands.iter().map(|c| c.x_m).collect());
    let my = median(cands.iter().map(|c| c.y_m).collect());
    let mut tiles: [Vec<&HoldoutCandidate>; 4] = Default::default();
    for &c in &cands {
        let t = 2 * usize::from(c.x_m > mx) + usize::from(c.y_m > my);
        if let Some(tile) = tiles.get_mut(t) {
            tile.push(c);
        }
    }
    for tile in &mut tiles {
        tile.sort_by(|a, b| {
            b.n_eligible(min)
                .cmp(&a.n_eligible(min))
                .then(b.total().cmp(&a.total()))
                .then(a.id.cmp(&b.id))
        });
    }
    let mut order = [0usize, 1, 2, 3];
    shuffle(&mut rng_for(project_seed, PICK_STREAM, 0), &mut order);
    let mut queues: Vec<std::vec::IntoIter<&HoldoutCandidate>> = order
        .iter()
        .map(|&t| tiles.get(t).cloned().unwrap_or_default().into_iter())
        .collect();
    let mut picked: Vec<&HoldoutCandidate> = Vec::new();
    'rounds: loop {
        let mut gave = false;
        for q in &mut queues {
            if picked.len() >= n {
                break 'rounds;
            }
            if let Some(c) = q.next() {
                picked.push(c);
                gave = true;
            }
        }
        if !gave {
            break;
        }
    }

    let picked_ids: BTreeSet<WellId> = picked.iter().map(|c| c.id).collect();
    let nearest = |c: &HoldoutCandidate| {
        picked
            .iter()
            .map(|p| (c.x_m - p.x_m).hypot(c.y_m - p.y_m))
            .fold(f64::INFINITY, f64::min)
    };
    let mut per_log = BTreeMap::new();
    for log in LogKind::ALL {
        let mut list: Vec<WellId> = picked
            .iter()
            .filter(|c| c.count(log) >= min)
            .map(|c| c.id)
            .collect();
        if list.len() < n {
            let mut extra: Vec<(f64, WellId)> = cands
                .iter()
                .filter(|c| c.count(log) >= min && !picked_ids.contains(&c.id))
                .map(|c| (nearest(c), c.id))
                .collect();
            extra.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let need = n - list.len();
            list.extend(extra.into_iter().take(need).map(|(_, id)| id));
        }
        list.sort();
        if !list.is_empty() {
            per_log.insert(log, list);
        }
    }
    HoldoutPick {
        wells: picked.iter().map(|c| c.id).collect(),
        per_log,
    }
}

/// In every (log, well) of `pick.per_log`, mask rows of the
/// channel's measured rows (provenance 0, finite; positions `0..F` in MD
/// order): value NaN, provenance `CODE_NONE`, as `logs_qc` stores a missing
/// sample.
///
/// `interval`: `n_target = floor(holdout_frac · F)`, `block =
/// max(holdout_min_block, n_target / holdout_blocks)`; RNG
/// `ChaCha8Rng::seed_from_u64(stream_seed(project_seed, "holdout.mask.<LOG>",
/// well))`; up to `holdout_max_attempts` times while fewer than `n_target`
/// positions are masked: `s = random_range(0..=F − block)`, mask `s..s+block`.
/// A channel with `F < block` is not masked. `whole_log`: every measured row.
///
/// Wells of `logs` not in the pick are untouched. Errors: `Shape` if a picked
/// well lacks the log; `Config` for an unknown `holdout_mode`.
pub fn mask_logs(
    logs: &mut [WellLogs],
    pick: &HoldoutPick,
    cfg: &FillConfig,
    project_seed: u64,
) -> SxResult<BTreeMap<(WellId, LogKind), MaskedChannel>> {
    let mut out = BTreeMap::new();
    for w in logs.iter_mut() {
        let id = w.id;
        for (&log, wells) in &pick.per_log {
            if !wells.contains(&id) {
                continue;
            }
            let (Some(values), Some(prov)) = (w.curves.get_mut(&log), w.provenance.get_mut(&log))
            else {
                return Err(SxError::Shape(format!(
                    "holdout: well {} has no {} curve",
                    id.0,
                    log.name()
                )));
            };
            let measured: Vec<usize> = values
                .iter()
                .zip(prov.iter())
                .enumerate()
                .filter(|(_, (v, &c))| c == Provenance::Measured.code() && v.is_finite())
                .map(|(i, _)| i)
                .collect();
            let (positions, block, n_blocks) =
                mask_positions(measured.len(), log, id, cfg, project_seed)?;
            let rows: Vec<usize> = positions
                .iter()
                .filter_map(|&p| measured.get(p).copied())
                .collect();
            for &i in &rows {
                if let (Some(v), Some(c)) = (values.get_mut(i), prov.get_mut(i)) {
                    *v = f32::NAN;
                    *c = CODE_NONE;
                }
            }
            out.insert(
                (id, log),
                MaskedChannel {
                    n_measured: measured.len(),
                    block,
                    n_blocks,
                    rows,
                },
            );
        }
    }
    Ok(out)
}

/// Masked positions (ascending), block length and blocks drawn for a channel
/// with `f` measured rows. See [`mask_logs`].
fn mask_positions(
    f: usize,
    log: LogKind,
    well: WellId,
    cfg: &FillConfig,
    project_seed: u64,
) -> SxResult<(Vec<usize>, usize, usize)> {
    match cfg.holdout_mode.as_str() {
        "whole_log" => Ok(((0..f).collect(), f, usize::from(f > 0))),
        "interval" => {
            let n_target = (cfg.holdout_frac * f as f64).floor() as usize;
            let block = cfg
                .holdout_min_block
                .max(n_target / cfg.holdout_blocks.max(1));
            let mut hit = vec![false; f];
            let (mut count, mut n_blocks) = (0usize, 0usize);
            if block <= f {
                let stream = format!("{MASK_STREAM}{}", log.name());
                let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(project_seed, &stream, well.0));
                for _ in 0..cfg.holdout_max_attempts {
                    if count >= n_target {
                        break;
                    }
                    let s = rng.random_range(0..=f - block);
                    n_blocks += 1;
                    for slot in hit.iter_mut().skip(s).take(block) {
                        if !*slot {
                            *slot = true;
                            count += 1;
                        }
                    }
                }
            }
            let positions = hit
                .iter()
                .enumerate()
                .filter(|(_, &h)| h)
                .map(|(p, _)| p)
                .collect();
            Ok((positions, block, n_blocks))
        }
        other => Err(SxError::Config(format!(
            "fill.holdout_mode: unknown mode {other:?}"
        ))),
    }
}

/// Run the leak-free holdout on the finished project at
/// `project_dir` and write `holdout_scores.csv` and `holdout_summary.csv`
/// into [`HOLDOUT_DIR`].
///
/// Reads `wells`, `geometry`, `logs_qc`, `qc_flags`, `qc_report`,
/// `intervals`, `tops_estimated`, `pairs_rough`, `layer_coord`, `logs_raw`
/// (coverage) and the production `logs_norm` (truth). Picks and masks as
/// [`pick_holdout_wells`] and [`mask_logs`]; reruns `run_normalize_stage` on
/// the masked `logs_qc`; rebuilds every neighbor curve with the held-out wells
/// as `donor_exclude_wells` and the masked run's reference ranges and
/// exclusions; reruns `train_and_fill` (built-in LightGBM) with the held-out
/// wells as `train_exclude`. Scores per masked channel: `model` rows have
/// provenance 1, `final` rows any fill code, both with a finite prediction;
/// shares are over the scored rows. The summary pools the rows of every well
/// of a log. NaN rule: rows without a finite truth are not scored.
///
/// # Errors
/// Reader errors; `Shape` if the masked run's wells or grids differ from
/// `logs_norm`; errors of normalization, neighbor curves and fill; `Io` from
/// the writes.
pub fn score_holdout(project_dir: &Path, cfg: &ProjectConfig) -> SxResult<HoldoutReport> {
    let cfg = &crate::nphi_offset::effective_config(project_dir, cfg)?;
    let start = Instant::now();
    let p = |rel: &str| project_dir.join(rel);
    let (headers, _) = read_wells(&p(WELLS))?;
    let geometry = read_geometry(&p(GEOMETRY))?;
    let mut logs_qc = read_logs(&p(LOGS_QC), LOGS_QC_TAG)?;
    let flags = read_flags(&p(QC_FLAGS), QC_FLAGS_TAG)?;
    let prior = read_prior(&p(QC_REPORT))?;
    let (_, intervals) = read_intervals(&p(INTERVALS))?;
    let pairs = read_pairs(&p(PAIRS_ROUGH))?;
    let (layer, _) = read_layer_coord(&p(LAYER_COORD))?;

    let pick = pick_holdout_wells(&candidates(&headers, &logs_qc), &cfg.fill, cfg.seed);
    let held_out = pick.held_out();
    let masked = mask_logs(&mut logs_qc, &pick, &cfg.fill, cfg.seed)?;
    info!(
        wells = ?pick.wells.iter().map(|w| w.0).collect::<Vec<_>>(),
        held_out = held_out.len(),
        channels = masked.len(),
        "holdout masked"
    );

    let norm = run_normalize_stage(
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
        cfg,
    )?;
    drop(logs_qc);

    let mut data = NeighborData::load(project_dir, cfg, CoordChoice::Rough)?;
    let truth = std::mem::take(&mut data.logs);
    use_masked_norm(&mut data, norm, &truth)?;
    data.donor_exclude_wells = held_out.clone();
    let targets: Vec<WellId> = data.logs.keys().copied().collect();
    let curves = build_neighbor_curves(&data.inputs(), &targets, &cfg.neighbors)?;
    let inputs = FillInputs {
        nbr: data.inputs(),
        neighbors: &curves,
        resd_family: &data.resd_family,
        coord_source: &data.coord_source,
        neighbors_cfg: &cfg.neighbors,
    };
    let factory = LgbmFactory {
        params: LgbmParams::from_config(&cfg.fill.lightgbm),
        project_seed: cfg.seed,
    };
    let (filled, _) = train_and_fill(&inputs, &cfg.fill, cfg.seed, &factory, &held_out)?;

    let names: BTreeMap<WellId, &str> = headers.iter().map(|h| (h.id, h.name.as_str())).collect();
    let mut scores = Vec::with_capacity(masked.len());
    let mut pooled: BTreeMap<LogKind, Acc> = BTreeMap::new();
    for (&(w, log), ch) in &masked {
        let missing = |what: &str| {
            SxError::Shape(format!(
                "holdout: well {} is missing from the {what} logs",
                w.0
            ))
        };
        let t = truth.get(&w).ok_or_else(|| missing("production"))?;
        let f = filled.logs.get(&w).ok_or_else(|| missing("holdout"))?;
        let acc = channel_acc(ch, t, f, log)?;
        let name = names.get(&w).copied().unwrap_or_default();
        scores.push(acc.score(Some(w), name, log));
        pooled.entry(log).or_default().merge(acc);
    }
    let summary: Vec<HoldoutScore> = pooled
        .into_iter()
        .map(|(log, acc)| acc.score(None, "", log))
        .collect();

    let dir = project_dir.join(HOLDOUT_DIR);
    let scores_path = dir.join(SCORES_CSV);
    let summary_path = dir.join(SUMMARY_CSV);
    let b = scores_batch(&scores, true)?;
    write_atomic(&scores_path, |t| csv::write_batch(t, &b))?;
    let b = scores_batch(&summary, false)?;
    write_atomic(&summary_path, |t| csv::write_batch(t, &b))?;
    info!(
        channels = scores.len(),
        seconds = start.elapsed().as_secs_f64(),
        "holdout scored"
    );
    Ok(HoldoutReport {
        pick,
        masked,
        scores,
        summary,
        scores_path,
        summary_path,
    })
}

/// Candidates of: wells of `logs` with a header, measured counts per log.
fn candidates(headers: &[WellHeader], logs: &[WellLogs]) -> Vec<HoldoutCandidate> {
    let by_id: BTreeMap<WellId, &WellHeader> = headers.iter().map(|h| (h.id, h)).collect();
    logs.iter()
        .filter_map(|l| {
            let h = by_id.get(&l.id)?;
            let measured = LogKind::ALL
                .iter()
                .map(|&log| (log, n_measured(l, log)))
                .collect();
            Some(HoldoutCandidate {
                id: l.id,
                x_m: h.x_m,
                y_m: h.y_m,
                measured,
            })
        })
        .collect()
}

fn n_measured(logs: &WellLogs, log: LogKind) -> usize {
    match (logs.curves.get(&log), logs.provenance.get(&log)) {
        (Some(v), Some(p)) => v
            .iter()
            .zip(p)
            .filter(|(x, &c)| x.is_finite() && c == Provenance::Measured.code())
            .count(),
        _ => 0,
    }
}

/// Replace the production normalization in `data` by the masked run's: logs,
/// reference ranges (both ends finite) and exclusions (`jsd_outlier` channels
/// plus channels without a measured sample), as `NeighborData::load` builds
/// them from the files.
fn use_masked_norm(
    data: &mut NeighborData,
    norm: NormalizeOutput,
    truth: &BTreeMap<WellId, WellLogs>,
) -> SxResult<()> {
    let logs: BTreeMap<WellId, WellLogs> = norm.logs.into_iter().map(|l| (l.id, l)).collect();
    if !logs.keys().eq(truth.keys()) {
        return Err(SxError::Shape(
            "holdout: masked normalization covers other wells than logs_norm".into(),
        ));
    }
    for (id, l) in &logs {
        if truth.get(id).map(|t| t.grid) != Some(l.grid) {
            return Err(SxError::Shape(format!(
                "holdout: well {}: masked normalization grid differs from logs_norm",
                id.0
            )));
        }
    }
    data.ref_ranges = norm
        .anchors
        .iter()
        .filter(|r| r.ref_low_native.is_finite() && r.ref_high_native.is_finite())
        .map(|r| ((r.well_id, r.log), (r.ref_low_native, r.ref_high_native)))
        .collect();
    let mut exclude: BTreeSet<(WellId, LogKind)> = norm
        .report
        .iter()
        .filter(|r| r.jsd_outlier)
        .map(|r| (r.well_id, r.log))
        .collect();
    for l in logs.values() {
        for log in LogKind::ALL {
            if n_measured(l, log) == 0 {
                exclude.insert((l.id, log));
            }
        }
    }
    data.exclude = exclude;
    data.logs = logs;
    Ok(())
}

/// Scored rows of one scope, pooled over wells.
#[derive(Debug, Default)]
struct ScopeAcc {
    truth: Vec<f64>,
    pred: Vec<f64>,
    /// Rolling stds at rows where both are defined.
    tex_truth: Vec<f64>,
    tex_pred: Vec<f64>,
    exact_runs: usize,
}

impl ScopeAcc {
    /// `rows` ascending local grid rows; `truth`, `pred` the full curves.
    fn new(rows: &[usize], truth: &[f32], pred: &[f32]) -> Self {
        let mut in_scope = vec![false; truth.len().min(pred.len())];
        for &i in rows {
            if let Some(s) = in_scope.get_mut(i) {
                *s = true;
            }
        }
        let series = |src: &[f32]| -> Vec<f64> {
            src.iter()
                .zip(&in_scope)
                .map(|(&v, &s)| if s { f64::from(v) } else { f64::NAN })
                .collect()
        };
        let (ts, ps) = (series(truth), series(pred));
        let mut acc = Self::default();
        for &i in rows {
            let (Some(&t), Some(&p)) = (ts.get(i), ps.get(i)) else {
                continue;
            };
            acc.truth.push(t);
            acc.pred.push(p);
            let (st, sp) = (rolling_std(&ts, i), rolling_std(&ps, i));
            if st.is_finite() && sp.is_finite() {
                acc.tex_truth.push(st);
                acc.tex_pred.push(sp);
            }
        }
        acc.exact_runs = exact_runs(rows, pred);
        acc
    }

    fn merge(&mut self, other: Self) {
        self.truth.extend(other.truth);
        self.pred.extend(other.pred);
        self.tex_truth.extend(other.tex_truth);
        self.tex_pred.extend(other.tex_pred);
        self.exact_runs += other.exact_runs;
    }

    fn scores(&self) -> ScopeScores {
        let n = self.truth.len();
        let (r2, rmse, bias) = match regression_metrics(&self.truth, &self.pred) {
            Ok(m) => (m.r2, m.rmse, m.bias),
            Err(_) => (f64::NAN, f64::NAN, f64::NAN),
        };
        let st = pop_std(&self.truth);
        let spread_ratio = if n > 0 && st > 0.0 {
            pop_std(&self.pred) / st
        } else {
            f64::NAN
        };
        let (slope, intercept) = if n > 0 {
            let nf = n as f64;
            let mt = self.truth.iter().sum::<f64>() / nf;
            let mp = self.pred.iter().sum::<f64>() / nf;
            let (mut sxx, mut sxy) = (0.0, 0.0);
            for (&t, &p) in self.truth.iter().zip(&self.pred) {
                sxx += (t - mt) * (t - mt);
                sxy += (t - mt) * (p - mp);
            }
            let slope = if sxx > 0.0 { sxy / sxx } else { f64::NAN };
            (slope, mp - slope * mt)
        } else {
            (f64::NAN, f64::NAN)
        };
        let tt = median(self.tex_truth.clone());
        let texture_ratio = if tt > 0.0 {
            median(self.tex_pred.clone()) / tt
        } else {
            f64::NAN
        };
        ScopeScores {
            n,
            r2,
            rmse,
            bias,
            spread_ratio,
            slope,
            intercept,
            texture_ratio,
            exact_runs: self.exact_runs,
        }
    }
}

/// Counts and scored rows of one channel, or of one log pooled over wells.
#[derive(Debug, Default)]
struct Acc {
    n_wells: usize,
    n_masked: usize,
    n_truth: usize,
    codes: BTreeMap<u8, usize>,
    model: ScopeAcc,
    fin: ScopeAcc,
}

impl Acc {
    fn merge(&mut self, other: Self) {
        self.n_wells += other.n_wells;
        self.n_masked += other.n_masked;
        self.n_truth += other.n_truth;
        for (c, k) in other.codes {
            *self.codes.entry(c).or_default() += k;
        }
        self.model.merge(other.model);
        self.fin.merge(other.fin);
    }

    fn share(&self, code: u8) -> f64 {
        if self.n_truth == 0 {
            return f64::NAN;
        }
        self.codes.get(&code).copied().unwrap_or(0) as f64 / self.n_truth as f64
    }

    fn score(&self, well_id: Option<WellId>, well_name: &str, log: LogKind) -> HoldoutScore {
        HoldoutScore {
            well_id,
            well_name: well_name.to_string(),
            log,
            n_wells: self.n_wells,
            n_masked: self.n_masked,
            n_truth: self.n_truth,
            unfilled_share: self.share(Provenance::Unfilled.code()),
            step_share: SHARE_STEPS
                .iter()
                .map(|&s| (s, self.share(s.provenance().code())))
                .collect(),
            model_scope: self.model.scores(),
            final_scope: self.fin.scores(),
        }
    }
}

/// Scored rows of one masked channel: truth from production `logs_norm`,
/// prediction and provenance from the holdout fill.
fn channel_acc(
    ch: &MaskedChannel,
    truth: &WellLogs,
    filled: &WellLogs,
    log: LogKind,
) -> SxResult<Acc> {
    if truth.grid != filled.grid {
        return Err(SxError::Shape(format!(
            "holdout: well {}: fill grid differs from logs_norm",
            truth.id.0
        )));
    }
    let (tv, tp) = curve(truth, log)?;
    let (fv, fp) = curve(filled, log)?;
    let mut acc = Acc {
        n_wells: 1,
        n_masked: ch.rows.len(),
        ..Acc::default()
    };
    let (mut model_rows, mut final_rows) = (Vec::new(), Vec::new());
    for &i in &ch.rows {
        let (Some(&t), Some(&tc), Some(&v), Some(&c)) =
            (tv.get(i), tp.get(i), fv.get(i), fp.get(i))
        else {
            continue;
        };
        if tc != Provenance::Measured.code() || !t.is_finite() {
            continue;
        }
        acc.n_truth += 1;
        *acc.codes.entry(c).or_default() += 1;
        if !is_fill_code(c) && c != Provenance::Unfilled.code() {
            warn!(
                well = truth.id.0,
                log = log.name(),
                row = i,
                code = c,
                "holdout: masked row has a non-fill code"
            );
        }
        if !v.is_finite() {
            continue;
        }
        if c == Provenance::Model.code() {
            model_rows.push(i);
        }
        if is_fill_code(c) {
            final_rows.push(i);
        }
    }
    acc.model = ScopeAcc::new(&model_rows, tv, fv);
    acc.fin = ScopeAcc::new(&final_rows, tv, fv);
    Ok(acc)
}

/// Sample std (`N − 1`) of the finite values of `series` over rows
/// `j − 10 ..= j + 9` clipped to the series; NaN with fewer than 10 values.
fn rolling_std(series: &[f64], j: usize) -> f64 {
    let lo = j.saturating_sub(TEXTURE_BEFORE);
    let hi = (j + TEXTURE_AFTER + 1).min(series.len());
    let w: Vec<f64> = series
        .get(lo..hi)
        .unwrap_or(&[])
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if w.len() < TEXTURE_MIN {
        return f64::NAN;
    }
    let n = w.len() as f64;
    let mean = w.iter().sum::<f64>() / n;
    (w.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Runs of at least 20 consecutive grid rows of `rows` (ascending) with
/// exactly equal `pred`.
fn exact_runs(rows: &[usize], pred: &[f32]) -> usize {
    let mut count = 0;
    let mut len = 0usize;
    let mut prev: Option<(usize, f32)> = None;
    for &i in rows {
        let Some(&v) = pred.get(i) else {
            continue;
        };
        len = match prev {
            Some((pi, pv)) if i == pi + 1 && v == pv => len + 1,
            _ => {
                if len >= EXACT_RUN_MIN {
                    count += 1;
                }
                1
            }
        };
        prev = Some((i, v));
    }
    if len >= EXACT_RUN_MIN {
        count += 1;
    }
    count
}

/// Median of the finite values (mean of the two middle ones for an even
/// count); NaN if none.
fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    match (
        v.len() % 2,
        mid.checked_sub(1).and_then(|i| v.get(i)),
        v.get(mid),
    ) {
        (1, _, Some(&m)) => m,
        (0, Some(&a), Some(&b)) => (a + b) / 2.0,
        _ => f64::NAN,
    }
}

/// Long-format batch: two rows (`model`, `final`) per score. `per_well` adds
/// `well_id`, `well_name`; else `n_wells`.
fn scores_batch(rows: &[HoldoutScore], per_well: bool) -> SxResult<RecordBatch> {
    let long: Vec<(&HoldoutScore, HoldoutScope, &ScopeScores)> = rows
        .iter()
        .flat_map(|r| {
            [
                (r, HoldoutScope::Model, &r.model_scope),
                (r, HoldoutScope::Final, &r.final_scope),
            ]
        })
        .collect();
    let count = |v: usize| i64::try_from(v).unwrap_or(i64::MAX);
    let mut fields: Vec<Field> = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    let mut push = |name: &str, dt: DataType, col: ArrayRef| {
        fields.push(Field::new(name, dt, false));
        cols.push(col);
    };
    let ints = |f: &dyn Fn(&(&HoldoutScore, HoldoutScope, &ScopeScores)) -> usize| -> ArrayRef {
        Arc::new(Int64Array::from(
            long.iter().map(|r| count(f(r))).collect::<Vec<_>>(),
        ))
    };
    let floats = |f: &dyn Fn(&(&HoldoutScore, HoldoutScope, &ScopeScores)) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from(long.iter().map(f).collect::<Vec<_>>()))
    };
    let texts = |f: &dyn Fn(&(&HoldoutScore, HoldoutScope, &ScopeScores)) -> String| -> ArrayRef {
        Arc::new(StringArray::from(long.iter().map(f).collect::<Vec<_>>()))
    };
    if per_well {
        push(
            "well_id",
            DataType::Int32,
            Arc::new(Int32Array::from(
                long.iter()
                    .map(|r| r.0.well_id.map_or(-1, |w| w.0))
                    .collect::<Vec<_>>(),
            )),
        );
        push(
            "well_name",
            DataType::Utf8,
            texts(&|r| r.0.well_name.clone()),
        );
    }
    push(
        "log",
        DataType::Utf8,
        texts(&|r| r.0.log.name().to_string()),
    );
    push("scope", DataType::Utf8, texts(&|r| r.1.name().to_string()));
    if !per_well {
        push("n_wells", DataType::Int64, ints(&|r| r.0.n_wells));
    }
    push("n_masked", DataType::Int64, ints(&|r| r.0.n_masked));
    push("n_truth", DataType::Int64, ints(&|r| r.0.n_truth));
    push("n", DataType::Int64, ints(&|r| r.2.n));
    push("r2", DataType::Float64, floats(&|r| r.2.r2));
    push("rmse", DataType::Float64, floats(&|r| r.2.rmse));
    push("bias", DataType::Float64, floats(&|r| r.2.bias));
    push(
        "spread_ratio",
        DataType::Float64,
        floats(&|r| r.2.spread_ratio),
    );
    push("slope", DataType::Float64, floats(&|r| r.2.slope));
    push("intercept", DataType::Float64, floats(&|r| r.2.intercept));
    push(
        "texture_ratio",
        DataType::Float64,
        floats(&|r| r.2.texture_ratio),
    );
    push("exact_runs", DataType::Int64, ints(&|r| r.2.exact_runs));
    push(
        "unfilled_share",
        DataType::Float64,
        floats(&|r| r.0.unfilled_share),
    );
    for (k, step) in SHARE_STEPS.iter().enumerate() {
        push(
            &format!("share_{}", step.name()),
            DataType::Float64,
            floats(&|r| r.0.step_share.get(k).map_or(f64::NAN, |s| s.1)),
        );
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)
        .map_err(|e| SxError::InvalidInput(format!("cannot build holdout batch: {e}")))
}

impl HoldoutReport {
    /// Plain-text table of [`HoldoutReport::summary`]: one line per log with
    /// wells, masked and scored rows, the `model` and `final` scores and the
    /// unfilled share, then one line per log with the step shares.
    pub fn summary_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<5} {:>5} {:>7} {:>7} | {:>6} {:>7} {:>8} {:>8} {:>6} {:>6} {:>3} | {:>6} {:>7} {:>8} {:>8} {:>6} {:>6} {:>3} | {:>8}",
            "log", "wells", "masked", "scored",
            "m_n", "m_r2", "m_rmse", "m_bias", "m_sprd", "m_tex", "m_x",
            "f_n", "f_r2", "f_rmse", "f_bias", "f_sprd", "f_tex", "f_x",
            "unfilled"
        );
        let scope = |s: &ScopeScores| {
            format!(
                "{:>6} {:>7.3} {:>8.4} {:>8.4} {:>6.3} {:>6.3} {:>3}",
                s.n, s.r2, s.rmse, s.bias, s.spread_ratio, s.texture_ratio, s.exact_runs
            )
        };
        for s in &self.summary {
            let _ = writeln!(
                out,
                "{:<5} {:>5} {:>7} {:>7} | {} | {} | {:>8.4}",
                s.log.name(),
                s.n_wells,
                s.n_masked,
                s.n_truth,
                scope(&s.model_scope),
                scope(&s.final_scope),
                s.unfilled_share
            );
        }
        let _ = write!(out, "{:<5}", "log");
        for step in SHARE_STEPS {
            let _ = write!(out, " {:>8}", short_step(step));
        }
        let _ = writeln!(out, " {:>8}", "unfilled");
        for s in &self.summary {
            let _ = write!(out, "{:<5}", s.log.name());
            for (_, v) in &s.step_share {
                let _ = write!(out, " {v:>8.4}");
            }
            let _ = writeln!(out, " {:>8.4}", s.unfilled_share);
        }
        out
    }
}

fn short_step(step: FillStep) -> &'static str {
    match step {
        FillStep::Model => "model",
        FillStep::NeighborWeighted => "nbr_w",
        FillStep::RealDonor => "donor",
        FillStep::NeighborWeak => "nbr_weak",
        FillStep::NeighborShape => "nbr_shp",
        FillStep::CrossLogFit => "xlog",
        FillStep::DonorShape => "don_shp",
        FillStep::OwnTexture => "own_tex",
        FillStep::SmoothedModel => "smooth",
        FillStep::None => "none",
    }
}
