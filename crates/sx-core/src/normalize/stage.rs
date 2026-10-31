// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stage 4 runner (`run_normalize_stage`), its row types, and the batch
//! builders of `logs_norm`, `norm_anchors.csv` and `norm_report.csv`.
//!
//! Order of work: N1 native copy → N2 consistency jury →
//! N3 pileups → N4 scope and stats → N5 ring test → N6 divergence check →
//! N7 anchors → N8 voters and whole-interval references → N9 GR low span →
//! N10 depth-aware references → N11 resistivity decade check → N12 transform
//! → N13 post-normalization checks → N14 outputs. N7 to N12 each read and
//! write one log only, so they run log by log; the result equals the stage
//! order. A check writes its bit only on samples finite right before it ran.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, UInt8Array,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use tracing::{debug, info};

use super::reference::{
    depth_pair, fleet_pair, idw_mean, idw_pair, layer_bin, second_pick, stretch_need, Anchor,
    DepthReason, DepthResult, DepthTarget, DepthVoter, FleetSource, RefPair, RefSource, Scope,
    Voter, VoterExclusion,
};
use super::transform::{gain_offset, stretch_cap, transform_channel, GainOffset};
use crate::align::PairMap;
use crate::config::ProjectConfig;
use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;
use crate::qc::{
    apply_jury, apply_mask, blank_channel, bounds_mask, channel_stats, gr_low_span, jsd_scores,
    native_bounds, native_value, pileup_mask, post_flat_wells, resd_decade_off, ring_test,
    run_jury, short_run_mask, Baseline, ChannelStats, DivRow, JuryInput, JuryResult, JuryVerdict,
    QcFlags, RingRow, QC_CHANNEL, QC_JURY_SEGMENT, QC_OUT_OF_BOUNDS, QC_PILEUP, QC_SHORT_RUN,
};
use crate::types::{
    GridRange, IntervalRow, LogKind, WellGeometry, WellHeader, WellId, WellLogs, CODE_NONE,
};
use crate::util::median;

/// Schema tag of `logs_norm.parquet`.
pub const LOGS_NORM_TAG: &str = "subsurfix.logs_norm/1";
/// Schema tag of `qc_flags_post.parquet`.
pub const QC_FLAGS_POST_TAG: &str = "subsurfix.qc_flags_post/1";
/// Schema tag carried by the `norm_anchors` batch (the CSV writer ignores it).
pub const NORM_ANCHORS_TAG: &str = "subsurfix.norm_anchors/1";
/// Schema tag carried by the `norm_report` batch (the CSV writer ignores it).
pub const NORM_REPORT_TAG: &str = "subsurfix.norm_report/1";

/// RESD value column of `logs_norm` (log10 values).
const RESD_LOG10_COLUMN: &str = "resd_log10ohmm";

/// Per-channel facts carried from stage 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorChannel {
    /// Well id.
    pub well: WellId,
    /// Log.
    pub log: LogKind,
    /// Provisional ring-test bits of stage 2; non-zero keeps the channel off the jury.
    pub sigma_bits: u8,
    /// Unit rescue applied in stage 2 (not a voter).
    pub rescued: bool,
}

/// Input of stage 4. Wells are matched by `WellId`; slice order does not matter.
#[derive(Debug, Clone, Copy)]
pub struct NormalizeInput<'a> {
    /// Well headers (name, surface `x_m`, `y_m`).
    pub wells: &'a [WellHeader],
    /// Linear values from `logs_qc.parquet`.
    pub logs_qc: &'a [WellLogs],
    /// Stage-2 bits from `qc_flags.parquet`.
    pub flags: &'a [QcFlags],
    /// Stage-2 channel facts.
    pub prior: &'a [PriorChannel],
    /// Per-sample geometry (jury interval thickness).
    pub geometry: &'a [WellGeometry],
    /// Interval rows of every well (jury).
    pub intervals: &'a [IntervalRow],
    /// Layer coordinate per sample on each well's grid; a well without an
    /// entry is all NaN.
    pub layer: &'a BTreeMap<WellId, Vec<f64>>,
    /// `pairs_rough`.
    pub pairs: &'a [PairMap],
}

/// Why a whole channel was blanked in stage 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormChannelAction {
    /// Not blanked as a whole.
    None,
    /// Jury verdict `bad`.
    Jury,
    /// Flagged GR well with a span far below its reference span.
    GrLowSpan,
    /// RESD median a decade or more off its reference.
    ResdDecade,
    /// Finite samples but no anchor.
    NoAnchor,
    /// Flat after normalization.
    PostFlat,
}

impl NormChannelAction {
    /// Text used in `norm_report.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            NormChannelAction::None => "none",
            NormChannelAction::Jury => "jury",
            NormChannelAction::GrLowSpan => "gr_low_span",
            NormChannelAction::ResdDecade => "resd_decade",
            NormChannelAction::NoAnchor => "no_anchor",
            NormChannelAction::PostFlat => "post_flat",
        }
    }
}

/// One row of `norm_anchors.csv` (one well, one normalized log with an anchor
/// at N7). Field names are the CSV column names; `_native` = the log's own
/// unit. NaN is written as an empty CSV field.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorRow {
    /// Well id.
    pub well_id: WellId,
    /// Well name.
    pub well_name: String,
    /// Log.
    pub log: LogKind,
    /// Scope of the anchor samples.
    pub scope: Scope,
    /// Anchor samples.
    pub n_anchor: usize,
    /// Lower anchor.
    pub p_low_native: f64,
    /// Upper anchor.
    pub p_high_native: f64,
    /// Eligible voter after the second pick.
    pub voter: bool,
    /// First matching voter exclusion (`None` = voter).
    pub voter_exclusion: VoterExclusion,
    /// Fleet pair of build 2, lower.
    pub fleet_low_native: f64,
    /// Fleet pair of build 2, upper.
    pub fleet_high_native: f64,
    /// Fleet pair source.
    pub fleet_source: FleetSource,
    /// Whole-interval reference `W`, lower.
    pub whole_low_native: f64,
    /// Whole-interval reference `W`, upper.
    pub whole_high_native: f64,
    /// Source of `W` (`whole`, `fleet`, `none`).
    pub whole_source: RefSource,
    /// Voters in the inverse-distance sum of `W`.
    pub whole_n_voters: usize,
    /// `W` reverted to the fleet pair by the sanity check.
    pub whole_reverted: bool,
    /// Stretch against the build-1 reference.
    pub need_first: f64,
    /// Layer-bin cover `|B_t| / |B_fleet|`; NaN unless computed.
    pub cover_frac: f64,
    /// Outcome of the depth-aware pass.
    pub depth_reason: DepthReason,
    /// Depth-aware pair, lower; NaN unless computed.
    pub depth_low_native: f64,
    /// Depth-aware pair, upper; NaN unless computed.
    pub depth_high_native: f64,
    /// Qualifying depth-aware voters.
    pub depth_n_voters: usize,
    /// Final reference `R`, lower.
    pub ref_low_native: f64,
    /// Final reference `R`, upper.
    pub ref_high_native: f64,
    /// Source of `R`.
    pub ref_source: RefSource,
    /// Stretch against `R`.
    pub need_final: f64,
    /// Extended gain cap used.
    pub extended: bool,
    /// Gain cap used (NaN when not transformed).
    pub gain_cap: f64,
    /// Shift-only transform.
    pub shift_only: bool,
    /// Gain (NaN when not transformed).
    pub gain: f64,
    /// Offset (NaN when not transformed).
    pub offset_native: f64,
    /// False if the channel was blanked before N12 (`gr_low_span`).
    pub transformed: bool,
}

/// One row of `norm_report.csv` (one well, one log). Field names are the CSV
/// column names; `_native` = native space (log10 ohm·m for RESD). NaN is
/// written as an empty CSV field. Sample counts are stage-4 blanks.
#[derive(Debug, Clone, PartialEq)]
pub struct NormChannelRow {
    /// Well id.
    pub well_id: WellId,
    /// Well name.
    pub well_name: String,
    /// Log.
    pub log: LogKind,
    /// Finite samples in `logs_qc` (native copy).
    pub n_qc: usize,
    /// Jury verdict.
    pub jury_verdict: JuryVerdict,
    /// Jury correlation.
    pub jury_corr: f64,
    /// Jury convicted share.
    pub jury_convicted_frac: f64,
    /// Scoreable jury windows.
    pub jury_n_windows: usize,
    /// Flagged jury windows.
    pub jury_n_flagged: usize,
    /// Convicted jury segments.
    pub jury_n_segments: usize,
    /// Distinct jurors.
    pub jury_n_jurors: usize,
    /// Samples blanked by `QC_JURY_SEGMENT`.
    pub n_jury_segment: usize,
    /// Samples blanked by `QC_PILEUP`.
    pub n_pileup: usize,
    /// Scope of the N4 stats.
    pub scope: Scope,
    /// Rows used for the N4 stats; 0 if no stats row.
    pub stats_n: usize,
    /// N4 mean.
    pub mean_native: f64,
    /// N4 median.
    pub median_native: f64,
    /// N4 std.
    pub std_native: f64,
    /// N4 lower percentile.
    pub p_low_native: f64,
    /// N4 upper percentile.
    pub p_high_native: f64,
    /// Official ring-test bits.
    pub sigma_bits: u8,
    /// Largest own deviation.
    pub sigma_max_dev: f64,
    /// Cluster deviation.
    pub sigma_cluster_dev: f64,
    /// Ring-test baseline.
    pub sigma_baseline: Baseline,
    /// Median ratio to the fleet median.
    pub unit_ratio: f64,
    /// Divergence score.
    pub jsd_score: f64,
    /// Divergence neighbors.
    pub jsd_n_neighbors: usize,
    /// Divergence score valid.
    pub jsd_valid: bool,
    /// Divergence outlier.
    pub jsd_outlier: bool,
    /// Decade-check reference (RESD only, else NaN).
    pub decade_ref_log10ohmm: f64,
    /// Median minus the decade reference (RESD only, else NaN).
    pub decade_diff_log10ohmm: f64,
    /// Whole-channel action.
    pub channel_action: NormChannelAction,
    /// Samples blanked by `QC_CHANNEL`.
    pub n_channel: usize,
    /// Samples blanked by `QC_OUT_OF_BOUNDS`.
    pub n_out_of_bounds: usize,
    /// Samples blanked by `QC_SHORT_RUN`.
    pub n_short_run: usize,
    /// Finite samples in `logs_norm`.
    pub n_norm: usize,
}

/// Output of stage 4. Vectors are sorted by `WellId`; `anchors` and `report`
/// are then in suite order per well.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeOutput {
    /// Native values (RESD in log10), all six logs per well, grids as in
    /// `logs_qc`; provenance `0` where finite, [`CODE_NONE`] where NaN.
    pub logs: Vec<WellLogs>,
    /// Stage-2 bits OR stage-4 bits.
    pub flags: Vec<QcFlags>,
    /// `norm_anchors.csv` rows.
    pub anchors: Vec<AnchorRow>,
    /// `norm_report.csv` rows.
    pub report: Vec<NormChannelRow>,
}

/// Working state of one channel.
struct Chan {
    /// Native working copy.
    values: Vec<f32>,
    /// Stage-4 bits.
    bits: Vec<u8>,
    stage2: Vec<u8>,
    n_qc: usize,
    action: NormChannelAction,
    jury: Option<JuryResult>,
    scope: Scope,
    stats: Option<ChannelStats>,
    ring: Option<RingRow>,
    div: Option<DivRow>,
    decade_ref: f64,
    decade_diff: f64,
}

struct Work<'a> {
    id: WellId,
    name: &'a str,
    grid: GridRange,
    xy: Option<(f64, f64)>,
    layer: Option<&'a [f64]>,
    /// Layer bin per sample (`normalize.depth_bins`).
    bins: Vec<Option<i64>>,
    /// Suite order.
    chans: Vec<Chan>,
}

/// Facts of one well kept from N1 until the working channels are built.
struct Shell<'a> {
    name: &'a str,
    xy: Option<(f64, f64)>,
    layer: Option<&'a [f64]>,
    stage2: Vec<Vec<u8>>,
    n_qc: Vec<usize>,
}

fn native_f32(log: LogKind, v: f32) -> f32 {
    let x = native_value(log, v);
    if x.is_finite() {
        x as f32
    } else {
        f32::NAN
    }
}

fn to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn finite_count(values: &[f32]) -> usize {
    values.iter().filter(|v| v.is_finite()).count()
}

fn count_bit(bits: &[u8], bit: u8) -> usize {
    bits.iter().filter(|&&b| b & bit != 0).count()
}

/// Scope of one channel and its scope samples (4.0).
fn scope_samples(
    values: &[f32],
    layer: Option<&[f64]>,
    layer_scope: bool,
    min_count: usize,
) -> (Scope, Vec<f64>) {
    if let (true, Some(l)) = (layer_scope, layer) {
        let in_layer: Vec<f64> = values
            .iter()
            .zip(l)
            .filter(|(v, lv)| v.is_finite() && lv.is_finite())
            .map(|(v, _)| f64::from(*v))
            .collect();
        if !in_layer.is_empty() && in_layer.len() >= min_count {
            return (Scope::Layer, in_layer);
        }
    }
    let all = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| f64::from(v))
        .collect();
    (Scope::All, all)
}

/// Both anchor ends finite and inside `[lo, hi]`.
fn plausible(a: &Anchor, lo: f64, hi: f64) -> bool {
    [a.p_low, a.p_high]
        .iter()
        .all(|v| v.is_finite() && (lo..=hi).contains(v))
}

/// Run stage 4 (`normalize`).
///
/// Applies N1 to N14 of the module doc: the consistency jury (`off_jury` =
/// channels whose `prior.sigma_bits != 0`), pileups, stats, the official ring
/// test and divergence check (flag only), regional references, GR low span,
/// the resistivity decade check, the two-point transform of
/// `cfg.normalize.logs` (RHOB, DTC and RESD pass through), and the
/// post-normalization checks on all six logs. Per-well steps run in
/// parallel; fleet steps collect, then reduce in `WellId` order, so the
/// result does not depend on input order or thread count. Empty `logs_qc`
/// gives empty output. A `PriorChannel` for an unknown well is ignored.
/// Infinite or non-positive-RESD samples become NaN in the native copy.
///
/// Invariant per report row: `n_qc − n_norm = n_jury_segment + n_pileup +
/// n_channel + n_out_of_bounds + n_short_run`.
///
/// # Errors
/// `SxError::Config` for a config that fails `config::validate` (for example
/// `normalize.logs = ["RHOB"]`); `SxError::Shape` when a curve, bits or layer
/// vector differs from its grid, a well's flags grid differs from its logs
/// grid, or a well in `logs_qc` has no header; `SxError::InvalidInput` for a
/// duplicate well id in `logs_qc`; jury errors as `qc::run_jury`.
pub fn run_normalize_stage(
    input: &NormalizeInput,
    cfg: &ProjectConfig,
) -> SxResult<NormalizeOutput> {
    crate::config::validate(cfg)?;
    if input.logs_qc.is_empty() {
        return Ok(NormalizeOutput {
            logs: Vec::new(),
            flags: Vec::new(),
            anchors: Vec::new(),
            report: Vec::new(),
        });
    }
    let qc = &cfg.qc;
    let nc = &cfg.normalize;
    let layer_scope = nc.scope == "layer";

    let mut headers: BTreeMap<WellId, &WellHeader> = BTreeMap::new();
    for h in input.wells {
        headers.entry(h.id).or_insert(h);
    }
    let mut flags_by: BTreeMap<WellId, &QcFlags> = BTreeMap::new();
    for f in input.flags {
        flags_by.entry(f.id).or_insert(f);
    }

    // N1: native copy, stage-2 bits, zeroed stage-4 bits.
    let mut sorted: Vec<&WellLogs> = input.logs_qc.iter().collect();
    sorted.sort_by_key(|w| w.id);
    if let Some(dup) = sorted.windows(2).find_map(|p| match p {
        [a, b] if a.id == b.id => Some(a.id),
        _ => None,
    }) {
        return Err(SxError::InvalidInput(format!(
            "duplicate well id {} in logs_qc",
            dup.0
        )));
    }
    let mut native_logs: Vec<WellLogs> = Vec::with_capacity(sorted.len());
    let mut bits4: Vec<QcFlags> = Vec::with_capacity(sorted.len());
    let mut shells: Vec<Shell> = Vec::with_capacity(sorted.len());
    for w in &sorted {
        let header = headers.get(&w.id).ok_or_else(|| {
            SxError::Shape(format!("well {} is in logs_qc but has no header", w.id.0))
        })?;
        let n = w.grid.len;
        let prior_flags = flags_by.get(&w.id).copied();
        match prior_flags {
            Some(f) if f.grid != w.grid => {
                return Err(SxError::Shape(format!(
                    "well {}: qc_flags grid differs from logs_qc grid",
                    w.id.0
                )))
            }
            Some(_) => {}
            None => debug!(well = w.id.0, "normalize: no qc_flags, stage-2 bits 0"),
        }
        let layer = match input.layer.get(&w.id) {
            Some(l) if l.len() != n => {
                return Err(SxError::Shape(format!(
                    "well {}: layer coordinate has {} samples, grid has {n}",
                    w.id.0,
                    l.len()
                )))
            }
            Some(l) => Some(l.as_slice()),
            None => None,
        };
        let mut nat = WellLogs::new(w.id, w.grid);
        let mut b4 = QcFlags {
            id: w.id,
            grid: w.grid,
            bits: BTreeMap::new(),
        };
        let mut stage2 = Vec::with_capacity(LogKind::ALL.len());
        let mut n_qc = Vec::with_capacity(LogKind::ALL.len());
        for log in LogKind::ALL {
            let values: Vec<f32> = match w.curves.get(&log) {
                Some(c) if c.len() != n => {
                    return Err(SxError::Shape(format!(
                        "well {}: {} has {} samples, grid has {n}",
                        w.id.0,
                        log.name(),
                        c.len()
                    )))
                }
                Some(c) => c.iter().map(|&v| native_f32(log, v)).collect(),
                None => vec![f32::NAN; n],
            };
            let s2 = match prior_flags.and_then(|f| f.bits.get(&log)) {
                Some(b) if b.len() != n => {
                    return Err(SxError::Shape(format!(
                        "well {}: {} bits have {} samples, grid has {n}",
                        w.id.0,
                        log.name(),
                        b.len()
                    )))
                }
                Some(b) => b.clone(),
                None => vec![0u8; n],
            };
            n_qc.push(finite_count(&values));
            nat.curves.insert(log, values);
            b4.bits.insert(log, vec![0u8; n]);
            stage2.push(s2);
        }
        native_logs.push(nat);
        bits4.push(b4);
        shells.push(Shell {
            name: header.name.as_str(),
            xy: Some((header.x_m, header.y_m)).filter(|(x, y)| x.is_finite() && y.is_finite()),
            layer,
            stage2,
            n_qc,
        });
    }

    let known: BTreeSet<WellId> = sorted.iter().map(|w| w.id).collect();
    let mut off_jury: BTreeSet<(WellId, LogKind)> = BTreeSet::new();
    let mut rescued: BTreeSet<(WellId, LogKind)> = BTreeSet::new();
    for p in input.prior {
        if !known.contains(&p.well) {
            debug!(
                well = p.well.0,
                "normalize: prior channel for an unknown well, ignored"
            );
            continue;
        }
        if p.sigma_bits != 0 {
            off_jury.insert((p.well, p.log));
        }
        if p.rescued {
            rescued.insert((p.well, p.log));
        }
    }

    // N2: consistency jury on the native copy, then blank convictions.
    let results = run_jury(
        &JuryInput {
            wells: input.wells,
            logs: &native_logs,
            geometry: input.geometry,
            intervals: input.intervals,
            layer: input.layer,
            pairs: input.pairs,
            off_jury: &off_jury,
        },
        &qc.jury,
    )?;
    apply_jury(&results, &mut native_logs, &mut bits4)?;
    let mut jury_by: BTreeMap<(WellId, LogKind), JuryResult> =
        results.into_iter().map(|r| ((r.well, r.log), r)).collect();

    let mut works: Vec<Work> = Vec::with_capacity(shells.len());
    for ((nat, b4), shell) in native_logs.into_iter().zip(bits4).zip(shells) {
        let (id, grid) = (nat.id, nat.grid);
        let (mut curves, mut bits) = (nat.curves, b4.bits);
        let mut chans = Vec::with_capacity(LogKind::ALL.len());
        for ((&log, stage2), n_qc) in LogKind::ALL.iter().zip(shell.stage2).zip(shell.n_qc) {
            let jury = jury_by.remove(&(id, log));
            let action = if jury.as_ref().is_some_and(|j| j.verdict == JuryVerdict::Bad) {
                NormChannelAction::Jury
            } else {
                NormChannelAction::None
            };
            chans.push(Chan {
                values: curves
                    .remove(&log)
                    .unwrap_or_else(|| vec![f32::NAN; grid.len]),
                bits: bits.remove(&log).unwrap_or_else(|| vec![0u8; grid.len]),
                stage2,
                n_qc,
                action,
                jury,
                scope: Scope::All,
                stats: None,
                ring: None,
                div: None,
                decade_ref: f64::NAN,
                decade_diff: f64::NAN,
            });
        }
        let bins = match shell.layer {
            Some(l) => l.iter().map(|&x| layer_bin(x, nc.depth_bins)).collect(),
            None => vec![None; grid.len],
        };
        works.push(Work {
            id,
            name: shell.name,
            grid,
            xy: shell.xy,
            layer: shell.layer,
            bins,
            chans,
        });
    }

    // N3 pileups, N4 scope and stats (per well).
    works.par_iter_mut().try_for_each(|w| -> SxResult<()> {
        for (chan, &log) in w.chans.iter_mut().zip(LogKind::ALL.iter()) {
            let mask = pileup_mask(
                &to_f64(&chan.values),
                qc.pileup_min_len,
                qc.pileup_min_share,
                qc.pileup_rel_tol,
            );
            apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_PILEUP)?;
            let (scope, samples) =
                scope_samples(&chan.values, w.layer, layer_scope, qc.stats_min_count);
            chan.scope = scope;
            chan.stats = channel_stats(w.id, log, &samples, qc.stats_min_count, qc.stats_pct);
        }
        Ok(())
    })?;

    // N5 ring test, N6 divergence check (per log, flag only).
    let xy: BTreeMap<WellId, (f64, f64)> = works
        .iter()
        .filter_map(|w| w.xy.map(|p| (w.id, p)))
        .collect();
    for (li, &log) in LogKind::ALL.iter().enumerate() {
        let rows: Vec<ChannelStats> = works
            .iter()
            .filter_map(|w| w.chans.get(li).and_then(|c| c.stats))
            .collect();
        let ring: BTreeMap<WellId, RingRow> = ring_test(log, &rows, &xy, qc)
            .into_iter()
            .map(|r| (r.well, r))
            .collect();
        let channels: Vec<(WellId, Vec<f64>)> = works
            .par_iter()
            .map(|w| {
                let samples = w.chans.get(li).map_or_else(Vec::new, |c| {
                    scope_samples(&c.values, w.layer, layer_scope, qc.stats_min_count).1
                });
                (w.id, samples)
            })
            .collect();
        let div: BTreeMap<WellId, DivRow> = jsd_scores(log, &channels, &xy, qc, cfg.seed)
            .into_iter()
            .map(|r| (r.well, r))
            .collect();
        for w in works.iter_mut() {
            if let Some(c) = w.chans.get_mut(li) {
                c.ring = ring.get(&w.id).copied();
                c.div = div.get(&w.id).copied();
            }
        }
    }

    // N11 resistivity decade check; RESD is never normalized, so it is
    // independent of N7-N10 and N12 of the other logs.
    decade_check(&mut works, cfg)?;

    // N7-N10 and N12 per normalized log.
    let fleet_bins = works
        .iter()
        .flat_map(|w| w.bins.iter().flatten().copied())
        .collect::<BTreeSet<i64>>()
        .len();
    let mut anchors = Vec::new();
    for (li, &log) in LogKind::ALL.iter().enumerate() {
        if nc.logs.contains(&log) {
            anchors.extend(normalize_log(
                li, log, &mut works, fleet_bins, &rescued, cfg,
            )?);
        }
    }

    // N13 post-normalization checks: flat wells, bounds sweep, short runs.
    for li in 0..LogKind::ALL.len() {
        let values: Vec<(WellId, Vec<f64>)> = works
            .par_iter()
            .map(|w| {
                let v = w.chans.get(li).map_or_else(Vec::new, |c| to_f64(&c.values));
                (w.id, v)
            })
            .collect();
        let view: Vec<(WellId, &[f64])> =
            values.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let flat = post_flat_wells(
            &view,
            qc.post_flat_ratio,
            qc.post_flat_min_count,
            qc.fleet_min_wells,
        );
        for w in works.iter_mut().filter(|w| flat.contains(&w.id)) {
            if let Some(c) = w.chans.get_mut(li) {
                blank_channel(&mut c.values, &mut c.bits)?;
                c.action = NormChannelAction::PostFlat;
                debug!(well = w.id.0, li, "normalize: flat after normalization");
            }
        }
    }
    works.par_iter_mut().try_for_each(|w| -> SxResult<()> {
        for (chan, &log) in w.chans.iter_mut().zip(LogKind::ALL.iter()) {
            let (lo, hi) = native_bounds(log);
            let mask = bounds_mask(&to_f64(&chan.values), lo, hi);
            apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_OUT_OF_BOUNDS)?;
            let mask = short_run_mask(&to_f64(&chan.values), qc.run_min_len);
            apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_SHORT_RUN)?;
        }
        Ok(())
    })?;

    // N14 outputs.
    info!(wells = works.len(), "normalize stage: done");
    let mut out_logs = Vec::with_capacity(works.len());
    let mut out_flags = Vec::with_capacity(works.len());
    let mut report = Vec::with_capacity(works.len() * LogKind::ALL.len());
    for w in works {
        let mut logs = WellLogs::new(w.id, w.grid);
        let mut flags = QcFlags {
            id: w.id,
            grid: w.grid,
            bits: BTreeMap::new(),
        };
        for (&log, chan) in LogKind::ALL.iter().zip(w.chans) {
            report.push(report_row(w.id, w.name, log, &chan));
            let merged: Vec<u8> = chan
                .stage2
                .iter()
                .zip(&chan.bits)
                .map(|(a, b)| a | b)
                .collect();
            logs.provenance.insert(
                log,
                chan.values
                    .iter()
                    .map(|v| if v.is_finite() { 0 } else { CODE_NONE })
                    .collect(),
            );
            logs.curves.insert(log, chan.values);
            flags.bits.insert(log, merged);
        }
        out_logs.push(logs);
        out_flags.push(flags);
    }
    Ok(NormalizeOutput {
        logs: out_logs,
        flags: out_flags,
        anchors,
        report,
    })
}

/// N11: RESD median against the inverse-distance mean of the voters' medians.
fn decade_check(works: &mut [Work], cfg: &ProjectConfig) -> SxResult<()> {
    let (qc, nc) = (&cfg.qc, &cfg.normalize);
    let Some(li) = LogKind::ALL.iter().position(|&l| l == LogKind::Resd) else {
        return Ok(());
    };
    struct Row {
        wi: usize,
        id: WellId,
        xy: Option<(f64, f64)>,
        med: f64,
    }
    let rows: Vec<Row> = works
        .iter()
        .enumerate()
        .filter_map(|(wi, w)| {
            let s = w.chans.get(li)?.stats?;
            Some(Row {
                wi,
                id: w.id,
                xy: w.xy,
                med: s.median,
            })
        })
        .collect();
    let voters: Vec<(WellId, (f64, f64), f64)> = rows
        .iter()
        .filter_map(|r| {
            let c = works.get(r.wi)?.chans.get(li)?;
            let ok = c.ring.is_some_and(|g| g.bits == 0) && !c.div.is_some_and(|d| d.outlier);
            ok.then_some((r.id, r.xy?, r.med))
        })
        .collect();
    for &Row { wi, id, xy, med } in &rows {
        let others: Vec<f64> = voters.iter().filter(|v| v.0 != id).map(|v| v.2).collect();
        let fallback = if !others.is_empty() && others.len() >= qc.fleet_min_wells {
            median(&others)
        } else {
            f64::NAN
        };
        let r = idw_mean(id, xy, &voters, fallback, nc);
        let Some(c) = works.get_mut(wi).and_then(|w| w.chans.get_mut(li)) else {
            continue;
        };
        c.decade_ref = r;
        c.decade_diff = if r.is_finite() && med.is_finite() {
            med - r
        } else {
            f64::NAN
        };
        if resd_decade_off(med, r, qc.resd_decade_log10) {
            blank_channel(&mut c.values, &mut c.bits)?;
            c.action = NormChannelAction::ResdDecade;
            debug!(well = id.0, "normalize: RESD decade off, channel blanked");
        }
    }
    Ok(())
}

/// N7-N10 and N12 of one normalized log; returns its anchor rows (WellId order).
fn normalize_log(
    li: usize,
    log: LogKind,
    works: &mut [Work],
    fleet_bins: usize,
    rescued: &BTreeSet<(WellId, LogKind)>,
    cfg: &ProjectConfig,
) -> SxResult<Vec<AnchorRow>> {
    let (qc, nc) = (&cfg.qc, &cfg.normalize);
    let (lo, hi) = native_bounds(log);

    // N7 anchors.
    let mut anchored: Vec<usize> = Vec::new();
    let mut anchors: Vec<Anchor> = Vec::new();
    let mut xys: Vec<Option<(f64, f64)>> = Vec::new();
    for (wi, w) in works.iter().enumerate() {
        let Some(c) = w.chans.get(li) else { continue };
        if let Some(s) = c.stats {
            anchored.push(wi);
            anchors.push(Anchor {
                well: w.id,
                log,
                n: s.n,
                p_low: s.p_low,
                p_high: s.p_high,
                scope: c.scope,
            });
            xys.push(w.xy);
        }
    }

    // N8 voters, fleet pair, build 1, second pick, build 2.
    let mut excl: Vec<VoterExclusion> = anchored
        .iter()
        .zip(&anchors)
        .zip(&xys)
        .map(|((&wi, a), xy)| {
            let c = works.get(wi).and_then(|w| w.chans.get(li));
            if c.and_then(|c| c.ring).is_some_and(|r| r.bits != 0) {
                VoterExclusion::Sigma
            } else if c.and_then(|c| c.div).is_some_and(|d| d.outlier) {
                VoterExclusion::Jsd
            } else if rescued.contains(&(a.well, log)) {
                VoterExclusion::Rescued
            } else if !plausible(a, lo, hi) {
                VoterExclusion::Implausible
            } else if xy.is_none() {
                VoterExclusion::NoXy
            } else {
                VoterExclusion::None
            }
        })
        .collect();
    let pick = |excl: &[VoterExclusion]| -> (Vec<Anchor>, Vec<Voter>) {
        let mut va = Vec::new();
        let mut vv = Vec::new();
        for ((a, xy), e) in anchors.iter().zip(&xys).zip(excl) {
            if let (VoterExclusion::None, Some(p)) = (e, xy) {
                va.push(*a);
                vv.push(Voter {
                    well: a.well,
                    xy: *p,
                    low: a.p_low,
                    high: a.p_high,
                });
            }
        }
        (va, vv)
    };
    let build = |voters: &[Voter], fleet: (f64, f64, FleetSource)| -> Vec<RefPair> {
        anchors
            .par_iter()
            .zip(xys.par_iter())
            .map(|(a, xy)| idw_pair(a.well, *xy, voters, (fleet.0, fleet.1), nc))
            .collect()
    };
    let (va1, v1) = pick(&excl);
    let fleet1 = fleet_pair(&va1, &anchors, qc.fleet_min_wells);
    let build1 = build(&v1, fleet1);
    for ((e, a), w1) in excl.iter_mut().zip(&anchors).zip(&build1) {
        if *e == VoterExclusion::None {
            *e = second_pick(a, w1, nc);
        }
    }
    let need_first: Vec<f64> = anchors
        .iter()
        .zip(&build1)
        .map(|(a, w1)| stretch_need(w1, a.p_low, a.p_high))
        .collect();
    let (va2, v2) = pick(&excl);
    let fleet2 = fleet_pair(&va2, &anchors, qc.fleet_min_wells);
    let whole = build(&v2, fleet2);

    // N9 GR low span.
    let mut dropped = vec![false; anchors.len()];
    if log == LogKind::Gr {
        for (((&wi, a), wr), d) in anchored
            .iter()
            .zip(&anchors)
            .zip(&whole)
            .zip(dropped.iter_mut())
        {
            let Some(c) = works.get_mut(wi).and_then(|w| w.chans.get_mut(li)) else {
                continue;
            };
            let flagged = c.ring.is_some_and(|r| r.bits != 0) || c.div.is_some_and(|dv| dv.outlier);
            let span = a.p_high - a.p_low;
            let ref_span = (wr.high - wr.low).abs();
            if gr_low_span(span, ref_span, flagged, qc.gr_span_frac) {
                blank_channel(&mut c.values, &mut c.bits)?;
                c.action = NormChannelAction::GrLowSpan;
                *d = true;
                debug!(well = a.well.0, "normalize: GR low span, channel blanked");
            }
        }
    }

    // N10 depth-aware references.
    let depth: Vec<Option<DepthResult>> = {
        let ro: &[Work] = works;
        let voters: Vec<DepthVoter> = anchored
            .iter()
            .zip(&anchors)
            .zip(&excl)
            .zip(&whole)
            .filter(|(((_, _), e), _)| **e == VoterExclusion::None)
            .filter_map(|(((&wi, a), _), wr)| {
                let w = ro.get(wi)?;
                Some(DepthVoter {
                    well: a.well,
                    xy: w.xy?,
                    values: &w.chans.get(li)?.values,
                    bins: &w.bins,
                    p_low: a.p_low,
                    p_high: a.p_high,
                    whole: *wr,
                })
            })
            .collect();
        anchored
            .par_iter()
            .zip(anchors.par_iter())
            .zip(dropped.par_iter())
            .map(|((&wi, a), &d)| {
                if d {
                    return None;
                }
                let w = ro.get(wi)?;
                let c = w.chans.get(li)?;
                let bins: BTreeSet<i64> = c
                    .values
                    .iter()
                    .zip(&w.bins)
                    .filter(|(v, _)| v.is_finite())
                    .filter_map(|(_, b)| *b)
                    .collect();
                let target = DepthTarget {
                    well: a.well,
                    xy: w.xy,
                    scope: a.scope,
                    bins: &bins,
                    fleet_bins,
                };
                Some(depth_pair(&target, &voters, qc, nc))
            })
            .collect()
    };

    // N12 transform.
    let mut slot: Vec<Option<usize>> = vec![None; works.len()];
    for (ai, &wi) in anchored.iter().enumerate() {
        if let Some(s) = slot.get_mut(wi) {
            *s = Some(ai);
        }
    }
    let mut rows = Vec::with_capacity(anchors.len());
    for (w, s) in works.iter_mut().zip(slot) {
        let (id, name) = (w.id, w.name);
        let Some(c) = w.chans.get_mut(li) else {
            continue;
        };
        let Some(ai) = s else {
            if c.values.iter().any(|v| v.is_finite()) {
                blank_channel(&mut c.values, &mut c.bits)?;
                c.action = NormChannelAction::NoAnchor;
                debug!(
                    well = id.0,
                    log = log.name(),
                    "normalize: no anchor, blanked"
                );
            }
            continue;
        };
        let (Some(a), Some(wr), Some(&nf), Some(&e), Some(dep)) = (
            anchors.get(ai),
            whole.get(ai),
            need_first.get(ai),
            excl.get(ai),
            depth.get(ai),
        ) else {
            continue;
        };
        let mut row = AnchorRow {
            well_id: id,
            well_name: name.to_string(),
            log,
            scope: a.scope,
            n_anchor: a.n,
            p_low_native: a.p_low,
            p_high_native: a.p_high,
            voter: e == VoterExclusion::None,
            voter_exclusion: e,
            fleet_low_native: fleet2.0,
            fleet_high_native: fleet2.1,
            fleet_source: fleet2.2,
            whole_low_native: wr.low,
            whole_high_native: wr.high,
            whole_source: wr.source,
            whole_n_voters: wr.n_voters,
            whole_reverted: wr.reverted,
            need_first: nf,
            cover_frac: f64::NAN,
            depth_reason: DepthReason::NotRun,
            depth_low_native: f64::NAN,
            depth_high_native: f64::NAN,
            depth_n_voters: 0,
            ref_low_native: f64::NAN,
            ref_high_native: f64::NAN,
            ref_source: RefSource::None,
            need_final: f64::NAN,
            extended: false,
            gain_cap: f64::NAN,
            shift_only: false,
            gain: f64::NAN,
            offset_native: f64::NAN,
            transformed: false,
        };
        if let Some(d) = dep {
            let r = if d.reason == DepthReason::LayerWindow {
                RefPair {
                    low: d.low,
                    high: d.high,
                    source: RefSource::Depth,
                    n_voters: d.n_voters,
                    reverted: false,
                }
            } else {
                *wr
            };
            let need = if r.source == RefSource::None {
                f64::NAN
            } else {
                (r.high - r.low) / (a.p_high - a.p_low)
            };
            let (verdict, corr) = c
                .jury
                .as_ref()
                .map_or((JuryVerdict::NoVerdict, f64::NAN), |j| (j.verdict, j.corr));
            let (cap, extended) = stretch_cap(need, verdict, corr, nc);
            let go = if r.source == RefSource::None {
                GainOffset::identity(cap)
            } else {
                gain_offset(
                    a.p_low,
                    a.p_high,
                    r.low,
                    r.high,
                    cap,
                    nc.shift_only_frac,
                    nc.degenerate_span,
                )
            };
            transform_channel(&mut c.values, &mut c.bits, &go, lo, hi)?;
            row.cover_frac = d.cover;
            row.depth_reason = d.reason;
            row.depth_low_native = d.low;
            row.depth_high_native = d.high;
            row.depth_n_voters = d.n_voters;
            row.ref_low_native = r.low;
            row.ref_high_native = r.high;
            row.ref_source = r.source;
            row.need_final = need;
            row.extended = extended;
            row.gain_cap = go.cap;
            row.shift_only = go.shift_only;
            row.gain = go.gain;
            row.offset_native = go.offset;
            row.transformed = true;
        }
        rows.push(row);
    }
    Ok(rows)
}

fn verdict_str(v: JuryVerdict) -> &'static str {
    match v {
        JuryVerdict::Ok => "ok",
        JuryVerdict::Bad => "bad",
        JuryVerdict::NoVerdict => "no_verdict",
        JuryVerdict::TooFew => "too_few",
    }
}

fn report_row(id: WellId, name: &str, log: LogKind, c: &Chan) -> NormChannelRow {
    let nan = f64::NAN;
    let j = c.jury.as_ref();
    let st = c.stats;
    NormChannelRow {
        well_id: id,
        well_name: name.to_string(),
        log,
        n_qc: c.n_qc,
        jury_verdict: j.map_or(JuryVerdict::NoVerdict, |j| j.verdict),
        jury_corr: j.map_or(nan, |j| j.corr),
        jury_convicted_frac: j.map_or(nan, |j| j.convicted_frac),
        jury_n_windows: j.map_or(0, |j| j.n_windows),
        jury_n_flagged: j.map_or(0, |j| j.n_flagged),
        jury_n_segments: j.map_or(0, |j| j.segments.len()),
        jury_n_jurors: j.map_or(0, |j| j.n_jurors),
        n_jury_segment: count_bit(&c.bits, QC_JURY_SEGMENT),
        n_pileup: count_bit(&c.bits, QC_PILEUP),
        scope: c.scope,
        stats_n: st.map_or(0, |s| s.n),
        mean_native: st.map_or(nan, |s| s.mean),
        median_native: st.map_or(nan, |s| s.median),
        std_native: st.map_or(nan, |s| s.std),
        p_low_native: st.map_or(nan, |s| s.p_low),
        p_high_native: st.map_or(nan, |s| s.p_high),
        sigma_bits: c.ring.map_or(0, |r| r.bits),
        sigma_max_dev: c.ring.map_or(nan, |r| r.max_dev),
        sigma_cluster_dev: c.ring.map_or(nan, |r| r.cluster_dev),
        sigma_baseline: c.ring.map_or(Baseline::None, |r| r.baseline),
        unit_ratio: c.ring.map_or(nan, |r| r.unit_ratio),
        jsd_score: c.div.map_or(nan, |d| d.score),
        jsd_n_neighbors: c.div.map_or(0, |d| d.n_neighbors),
        jsd_valid: c.div.is_some_and(|d| d.valid),
        jsd_outlier: c.div.is_some_and(|d| d.outlier),
        decade_ref_log10ohmm: c.decade_ref,
        decade_diff_log10ohmm: c.decade_diff,
        channel_action: c.action,
        n_channel: count_bit(&c.bits, QC_CHANNEL),
        n_out_of_bounds: count_bit(&c.bits, QC_OUT_OF_BOUNDS),
        n_short_run: count_bit(&c.bits, QC_SHORT_RUN),
        n_norm: finite_count(&c.values),
    }
}

// ----------------------------------------------------------------- batches

fn i64_of(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

fn col_i32<R>(rows: &[&R], f: impl Fn(&R) -> i32) -> ArrayRef {
    Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| f(r))))
}

fn col_i64<R>(rows: &[&R], f: impl Fn(&R) -> usize) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        rows.iter().map(|r| i64_of(f(r))),
    ))
}

fn col_u8<R>(rows: &[&R], f: impl Fn(&R) -> u8) -> ArrayRef {
    Arc::new(UInt8Array::from_iter_values(rows.iter().map(|r| f(r))))
}

fn col_f64<R>(rows: &[&R], f: impl Fn(&R) -> f64) -> ArrayRef {
    Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| f(r))))
}

fn col_str<R>(rows: &[&R], f: impl Fn(&R) -> String) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(rows.iter().map(|r| f(r))))
}

fn col_bool<R>(rows: &[&R], f: impl Fn(&R) -> bool) -> ArrayRef {
    Arc::new(BooleanArray::from(
        rows.iter().map(|r| f(r)).collect::<Vec<bool>>(),
    ))
}

/// Tagged batch of non-nullable columns, in order.
fn finish(cols: Vec<(&str, ArrayRef)>, tag: &str) -> SxResult<RecordBatch> {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), false))
        .collect();
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, tag)
}

/// `logs_norm` batch: `qc::logs_batch(logs, LOGS_NORM_TAG)` with the RESD
/// column named `resd_log10ohmm` (the values are log10 ohm·m). Columns:
/// `well_id, k, md_ft, gr_api, rhob_gcc, nphi_frac, dtc_usft,
/// resd_log10ohmm, pef_bfrac`; rows sorted by id then `k`. NaN rule: stored
/// as is.
///
/// # Errors
/// As `qc::logs_batch`; `SxError::InvalidInput` if the batch cannot be rebuilt.
pub fn logs_norm_batch(logs: &[WellLogs]) -> SxResult<RecordBatch> {
    let batch = crate::qc::logs_batch(logs, LOGS_NORM_TAG)?;
    let schema = batch.schema();
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let f = f.as_ref().clone();
            if f.name() == LogKind::Resd.column() {
                f.with_name(RESD_LOG10_COLUMN)
            } else {
                f
            }
        })
        .collect();
    let renamed = Schema::new_with_metadata(fields, schema.metadata().clone());
    RecordBatch::try_new(Arc::new(renamed), batch.columns().to_vec())
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))
}

/// `norm_anchors` batch with the 33 columns of `norm_anchors.csv` in order,
/// rows sorted by `well_id` then suite order. Schema tag
/// [`NORM_ANCHORS_TAG`] (ignored by the CSV writer). Counts are int64, bools
/// boolean, enums their text; NaN stays NaN (the CSV writer prints an empty
/// field).
///
/// # Errors
/// `SxError::InvalidInput` if the batch cannot be built.
pub fn norm_anchors_batch(rows: &[AnchorRow]) -> SxResult<RecordBatch> {
    let mut s: Vec<&AnchorRow> = rows.iter().collect();
    s.sort_by_key(|r| (r.well_id, r.log));
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("well_id", col_i32(&s, |r| r.well_id.0)),
        ("well_name", col_str(&s, |r| r.well_name.clone())),
        ("log", col_str(&s, |r| r.log.name().to_string())),
        ("scope", col_str(&s, |r| r.scope.as_str().to_string())),
        ("n_anchor", col_i64(&s, |r| r.n_anchor)),
        ("p_low_native", col_f64(&s, |r| r.p_low_native)),
        ("p_high_native", col_f64(&s, |r| r.p_high_native)),
        ("voter", col_bool(&s, |r| r.voter)),
        (
            "voter_exclusion",
            col_str(&s, |r| r.voter_exclusion.as_str().to_string()),
        ),
        ("fleet_low_native", col_f64(&s, |r| r.fleet_low_native)),
        ("fleet_high_native", col_f64(&s, |r| r.fleet_high_native)),
        (
            "fleet_source",
            col_str(&s, |r| r.fleet_source.as_str().to_string()),
        ),
        ("whole_low_native", col_f64(&s, |r| r.whole_low_native)),
        ("whole_high_native", col_f64(&s, |r| r.whole_high_native)),
        (
            "whole_source",
            col_str(&s, |r| r.whole_source.as_str().to_string()),
        ),
        ("whole_n_voters", col_i64(&s, |r| r.whole_n_voters)),
        ("whole_reverted", col_bool(&s, |r| r.whole_reverted)),
        ("need_first", col_f64(&s, |r| r.need_first)),
        ("cover_frac", col_f64(&s, |r| r.cover_frac)),
        (
            "depth_reason",
            col_str(&s, |r| r.depth_reason.as_str().to_string()),
        ),
        ("depth_low_native", col_f64(&s, |r| r.depth_low_native)),
        ("depth_high_native", col_f64(&s, |r| r.depth_high_native)),
        ("depth_n_voters", col_i64(&s, |r| r.depth_n_voters)),
        ("ref_low_native", col_f64(&s, |r| r.ref_low_native)),
        ("ref_high_native", col_f64(&s, |r| r.ref_high_native)),
        (
            "ref_source",
            col_str(&s, |r| r.ref_source.as_str().to_string()),
        ),
        ("need_final", col_f64(&s, |r| r.need_final)),
        ("extended", col_bool(&s, |r| r.extended)),
        ("gain_cap", col_f64(&s, |r| r.gain_cap)),
        ("shift_only", col_bool(&s, |r| r.shift_only)),
        ("gain", col_f64(&s, |r| r.gain)),
        ("offset_native", col_f64(&s, |r| r.offset_native)),
        ("transformed", col_bool(&s, |r| r.transformed)),
    ];
    finish(cols, NORM_ANCHORS_TAG)
}

/// `norm_report` batch with the 36 columns of `norm_report.csv` in order,
/// rows sorted by `well_id` then suite order. Schema tag [`NORM_REPORT_TAG`]
/// (ignored by the CSV writer). Counts are int64, `sigma_bits` uint8, bools
/// boolean, enums their text; NaN stays NaN.
///
/// # Errors
/// `SxError::InvalidInput` if the batch cannot be built.
pub fn norm_report_batch(rows: &[NormChannelRow]) -> SxResult<RecordBatch> {
    let mut s: Vec<&NormChannelRow> = rows.iter().collect();
    s.sort_by_key(|r| (r.well_id, r.log));
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("well_id", col_i32(&s, |r| r.well_id.0)),
        ("well_name", col_str(&s, |r| r.well_name.clone())),
        ("log", col_str(&s, |r| r.log.name().to_string())),
        ("n_qc", col_i64(&s, |r| r.n_qc)),
        (
            "jury_verdict",
            col_str(&s, |r| verdict_str(r.jury_verdict).to_string()),
        ),
        ("jury_corr", col_f64(&s, |r| r.jury_corr)),
        (
            "jury_convicted_frac",
            col_f64(&s, |r| r.jury_convicted_frac),
        ),
        ("jury_n_windows", col_i64(&s, |r| r.jury_n_windows)),
        ("jury_n_flagged", col_i64(&s, |r| r.jury_n_flagged)),
        ("jury_n_segments", col_i64(&s, |r| r.jury_n_segments)),
        ("jury_n_jurors", col_i64(&s, |r| r.jury_n_jurors)),
        ("n_jury_segment", col_i64(&s, |r| r.n_jury_segment)),
        ("n_pileup", col_i64(&s, |r| r.n_pileup)),
        ("scope", col_str(&s, |r| r.scope.as_str().to_string())),
        ("stats_n", col_i64(&s, |r| r.stats_n)),
        ("mean_native", col_f64(&s, |r| r.mean_native)),
        ("median_native", col_f64(&s, |r| r.median_native)),
        ("std_native", col_f64(&s, |r| r.std_native)),
        ("p_low_native", col_f64(&s, |r| r.p_low_native)),
        ("p_high_native", col_f64(&s, |r| r.p_high_native)),
        ("sigma_bits", col_u8(&s, |r| r.sigma_bits)),
        ("sigma_max_dev", col_f64(&s, |r| r.sigma_max_dev)),
        ("sigma_cluster_dev", col_f64(&s, |r| r.sigma_cluster_dev)),
        (
            "sigma_baseline",
            col_str(&s, |r| r.sigma_baseline.as_str().to_string()),
        ),
        ("unit_ratio", col_f64(&s, |r| r.unit_ratio)),
        ("jsd_score", col_f64(&s, |r| r.jsd_score)),
        ("jsd_n_neighbors", col_i64(&s, |r| r.jsd_n_neighbors)),
        ("jsd_valid", col_bool(&s, |r| r.jsd_valid)),
        ("jsd_outlier", col_bool(&s, |r| r.jsd_outlier)),
        (
            "decade_ref_log10ohmm",
            col_f64(&s, |r| r.decade_ref_log10ohmm),
        ),
        (
            "decade_diff_log10ohmm",
            col_f64(&s, |r| r.decade_diff_log10ohmm),
        ),
        (
            "channel_action",
            col_str(&s, |r| r.channel_action.as_str().to_string()),
        ),
        ("n_channel", col_i64(&s, |r| r.n_channel)),
        ("n_out_of_bounds", col_i64(&s, |r| r.n_out_of_bounds)),
        ("n_short_run", col_i64(&s, |r| r.n_short_run)),
        ("n_norm", col_i64(&s, |r| r.n_norm)),
    ];
    finish(cols, NORM_REPORT_TAG)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ProjectConfig {
        ProjectConfig::default_with_crs("EPSG:32614")
    }

    /// One well of four on a 1 km square: smooth, non-repeating curves inside
    /// the bounds (GR, RHOB, DTC, RESD; NPHI and PEF absent).
    fn well(id: i32) -> (WellHeader, WellLogs, QcFlags) {
        let n = 200usize;
        let grid = GridRange::new(1000, n).unwrap();
        let phase = f64::from(id);
        let wave = |base: f64, amp: f64, freq: f64| -> Vec<f32> {
            (0..n)
                .map(|j| (base + amp * (freq * j as f64 + phase).sin()) as f32)
                .collect()
        };
        let mut logs = WellLogs::new(WellId(id), grid);
        logs.curves.insert(LogKind::Gr, wave(60.0, 30.0, 0.13));
        logs.curves.insert(LogKind::Rhob, wave(2.3, 0.2, 0.1));
        logs.curves.insert(LogKind::Dtc, wave(80.0, 15.0, 0.07));
        let resd = wave(1.0, 0.5, 0.05)
            .iter()
            .map(|&x| 10f32.powf(x))
            .collect();
        logs.curves.insert(LogKind::Resd, resd);
        for log in [LogKind::Gr, LogKind::Rhob, LogKind::Dtc, LogKind::Resd] {
            logs.provenance.insert(log, vec![0u8; n]);
        }
        let header = WellHeader {
            id: WellId(id),
            name: format!("W{id}"),
            x_m: 1000.0 * f64::from(id % 2),
            y_m: 1000.0 * f64::from(id / 2),
            kb_elev_ft: 0.0,
        };
        let flags = QcFlags {
            id: WellId(id),
            grid,
            bits: LogKind::ALL.iter().map(|&l| (l, vec![0u8; n])).collect(),
        };
        (header, logs, flags)
    }

    /// RHOB and DTC come out bit-identical, RESD as its log10; no stage-4
    /// bits on them; the report invariant holds on every row.
    #[test]
    fn rhob_dtc_resd_pass_through() {
        let (mut wells, mut logs, mut flags) = (Vec::new(), Vec::new(), Vec::new());
        for id in 0..4 {
            let (h, l, f) = well(id);
            wells.push(h);
            logs.push(l);
            flags.push(f);
        }
        let layer = BTreeMap::new();
        let input = NormalizeInput {
            wells: &wells,
            logs_qc: &logs,
            flags: &flags,
            prior: &[],
            geometry: &[],
            intervals: &[],
            layer: &layer,
            pairs: &[],
        };
        let out = run_normalize_stage(&input, &cfg()).unwrap();
        assert_eq!(out.logs.len(), 4);
        assert_eq!(out.report.len(), 4 * LogKind::ALL.len());
        for ((src, got), fl) in logs.iter().zip(&out.logs).zip(&out.flags) {
            assert_eq!(src.grid, got.grid);
            for log in [LogKind::Rhob, LogKind::Dtc] {
                let a: Vec<u32> = src.curves[&log].iter().map(|v| v.to_bits()).collect();
                let b: Vec<u32> = got.curves[&log].iter().map(|v| v.to_bits()).collect();
                assert_eq!(a, b, "well {} {}", src.id.0, log.name());
            }
            let want: Vec<f32> = src.curves[&LogKind::Resd]
                .iter()
                .map(|&v| native_value(LogKind::Resd, v) as f32)
                .collect();
            assert_eq!(got.curves[&LogKind::Resd], want);
            for log in [LogKind::Rhob, LogKind::Dtc, LogKind::Resd] {
                assert!(fl.bits[&log].iter().all(|&b| b == 0));
                assert!(got.provenance[&log].iter().all(|&p| p == 0));
            }
        }
        for r in &out.report {
            assert_eq!(
                r.n_qc - r.n_norm,
                r.n_jury_segment + r.n_pileup + r.n_channel + r.n_out_of_bounds + r.n_short_run,
                "{r:?}"
            );
        }
        assert!(out.anchors.iter().all(|a| a.log == LogKind::Gr));
    }

    #[test]
    fn rhob_in_normalize_logs_is_config_error() {
        let mut c = cfg();
        c.normalize.logs = vec![LogKind::Rhob];
        let layer = BTreeMap::new();
        let input = NormalizeInput {
            wells: &[],
            logs_qc: &[],
            flags: &[],
            prior: &[],
            geometry: &[],
            intervals: &[],
            layer: &layer,
            pairs: &[],
        };
        assert!(matches!(
            run_normalize_stage(&input, &c),
            Err(SxError::Config(_))
        ));
    }
}
