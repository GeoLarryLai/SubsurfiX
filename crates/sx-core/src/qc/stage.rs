// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stage 2 runner (`run_qc_stage`), the report row type and the Arrow batch
//! builders of the three products `logs_qc`, `qc_flags` and `qc_report`.
//!
//! Order of work (fixed): bounds → constant channel → tails → unit rescue
//! (+ bounds re-check) → exact repeats → soft plateaus → short runs → stats
//! and low variance → provisional ring test → unit-mismatch blank → output.
//! A rule writes its bit only on samples that were finite right before it
//! ran, so every sample carries at most one bit.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use super::closed_cal;
use super::masks::{
    apply_mask, badhole_eval, blank_channel, bounds_mask, closed_caliper_mask, constant_channel,
    plateau_mask, range_mask, repeat_mask, short_run_mask, tail_mask, BadHoleParams, PlateauParams,
};
use super::rescue::{apply_rescue, plan_rescue, rescue_factor, RescueReason, RescueRow};
use super::ring::{ring_test, Baseline, RingRow, SIGMA_UNIT};
use super::stats::{channel_stats, low_variance_wells, ChannelStats};
use super::{
    native_vec, QcFlags, QC_CHANNEL, QC_OUT_OF_BOUNDS, QC_PLATEAU, QC_REPEAT, QC_SHORT_RUN, QC_TAIL,
};
use crate::config::QcConfig;
use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;
use crate::types::{GridRange, LogKind, WellAux, WellHeader, WellId, WellLogs, CODE_NONE};
use crate::units;
use crate::util::stats::median;

/// Schema tag of `logs_qc.parquet`.
pub const LOGS_QC_TAG: &str = "subsurfix.logs_qc/1";
/// Schema tag of `qc_flags.parquet`.
pub const QC_FLAGS_TAG: &str = "subsurfix.qc_flags/1";
/// Schema tag carried by the `qc_report` batch (the CSV writer ignores it).
pub const QC_REPORT_TAG: &str = "subsurfix.qc_report/1";
/// Schema tag of `rhob_closed_caliper.parquet`.
pub const RHOB_CLOSED_CALIPER_TAG: &str = "subsurfix.rhob_closed_caliper/1";

/// Input of stage 2: `wells.parquet` headers and `logs_raw.parquet` curves.
#[derive(Debug, Clone, Copy)]
pub struct QcInput<'a> {
    /// Well headers (name, x_m, y_m).
    pub wells: &'a [WellHeader],
    /// Raw curves, one entry per well; a log missing from `curves` is all NaN.
    pub logs: &'a [WellLogs],
    /// Caliper / density-correction curves (`aux_raw`); may be empty.
    pub aux: &'a [WellAux],
}

/// Output of stage 2. All vectors are sorted by `WellId`; `report` is then in
/// suite order per well.
#[derive(Debug, Clone, PartialEq)]
pub struct QcStageOutput {
    /// QC'd curves (all six logs per well). Provenance `0` where finite,
    /// [`CODE_NONE`] where NaN.
    pub logs: Vec<WellLogs>,
    /// Per-sample QC bits.
    pub flags: Vec<QcFlags>,
    /// One row per well per log.
    pub report: Vec<QcChannelRow>,
    /// Low-density samples kept unverified (no caliper or density-correction
    /// sample there): `(well, k, rhob_gcc)`, sorted by well then `k`.
    pub low_density_unverified: Vec<(WellId, i32, f32)>,
    /// Closed-caliper RHOB samples kept with a bias shift, sorted
    /// by well then `k`. Empty with `qc.closed_caliper.mode = "reject"`.
    pub closed_caliper_samples: Vec<ClosedCalSample>,
    /// One row per well with closed-caliper RHOB samples, sorted by well.
    pub closed_caliper_runs: Vec<ClosedCalRun>,
}

/// One closed-caliper RHOB sample kept by the rescue (corrected value).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosedCalSample {
    /// Well.
    pub well_id: WellId,
    /// Grid index.
    pub k: i32,
    /// Measured depth (ft).
    pub md_ft: f64,
    /// Raw RHOB (g/cm³), as in `logs_raw`.
    pub rhob_raw_gcc: f64,
    /// Corrected RHOB written to `logs_qc` (raw − bias).
    pub rhob_qc_gcc: f64,
    /// Per-well bias (g/cm³).
    pub bias_gcc: f64,
    /// Neighbor envelope lower percentile, median, upper percentile.
    pub env_lo: f64,
    /// See `env_lo`.
    pub env_p50: f64,
    /// See `env_lo`.
    pub env_hi: f64,
}

/// Per-well closed-caliper note (`closed_caliper_runs.csv`).
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedCalRun {
    /// Well.
    pub well_id: WellId,
    /// Well name.
    pub well_name: String,
    /// Shallowest and deepest closed-caliper candidate (MD, ft).
    pub md_top_ft: f64,
    /// See `md_top_ft`.
    pub md_base_ft: f64,
    /// RHOB samples rejected by the bad-hole rule with the caliper closed.
    pub n_closed: usize,
    /// Of those, still rejected.
    pub n_rejected: usize,
    /// Of those, rescued (bias-shifted, kept).
    pub n_rescued: usize,
    /// Candidates left after the despike.
    pub n_despiked: usize,
    /// Despiked candidates with a neighbor envelope.
    pub n_with_envelope: usize,
    /// Neighbor wells in the envelope.
    pub n_neighbors: usize,
    /// Bias (g/cm³; NaN when not estimated).
    pub bias_gcc: f64,
    /// `rescued`, `no_neighbors`, `too_little_overlap` or `reject_mode`.
    pub status: &'static str,
    /// Median logging speed over the closed / open caliper samples (ft/min;
    /// NaN when the file has no speed curve).
    pub lspd_closed_ftmin: f64,
    /// See `lspd_closed_ftmin`.
    pub lspd_open_ftmin: f64,
    /// `fast pass, arm closed` when the closed run was logged at least 1.5×
    /// faster than the open run, else `arm closed`.
    pub note: String,
}

/// Why a whole channel was blanked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAction {
    /// Not blanked as a whole.
    None,
    /// Constant channel.
    Constant,
    /// Std far below the fleet median std.
    LowVariance,
    /// Ring test unit mismatch.
    UnitMismatch,
    /// Channel median outside `qc.plausible_median` (blanked).
    ImplausibleLevel,
}

impl ChannelAction {
    /// Text used in `qc_report.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelAction::None => "none",
            ChannelAction::Constant => "constant",
            ChannelAction::LowVariance => "low_variance",
            ChannelAction::UnitMismatch => "unit_mismatch",
            ChannelAction::ImplausibleLevel => "implausible_level",
        }
    }
}

/// One row of `qc_report.csv` (one well, one log). Field names are the CSV
/// column names. `_native` = native space (log10 ohm·m for RESD); `_lin` =
/// the log's linear unit. NaN is written as an empty CSV field.
#[derive(Debug, Clone, PartialEq)]
pub struct QcChannelRow {
    /// Well id.
    pub well_id: WellId,
    /// Well name.
    pub well_name: String,
    /// Log.
    pub log: LogKind,
    /// Finite samples in `logs_raw`.
    pub n_raw: usize,
    /// Samples blanked by `QC_OUT_OF_BOUNDS`.
    pub n_out_of_bounds: usize,
    /// Samples blanked by `QC_TAIL`.
    pub n_tail: usize,
    /// Samples blanked by `QC_REPEAT`.
    pub n_repeat: usize,
    /// Samples blanked by `QC_PLATEAU`.
    pub n_plateau: usize,
    /// Samples blanked by `QC_SHORT_RUN`.
    pub n_short_run: usize,
    /// Samples blanked by `QC_CHANNEL`.
    pub n_channel: usize,
    /// Of `n_out_of_bounds`: samples masked by the bad-hole rule.
    pub n_bad_hole: usize,
    /// Of `n_out_of_bounds`: samples outside `qc.plausible_range`.
    pub n_implausible: usize,
    /// RHOB samples below `qc.low_density_rhob` with no caliper / density
    /// correction to check the hole: flagged here, kept in the log.
    pub n_low_density_unverified: usize,
    /// Closed-caliper RHOB samples kept with a bias shift.
    pub n_closed_cal_rescued: usize,
    /// Closed-caliper RHOB bias of the well (g/cm³; NaN when none).
    pub closed_cal_bias_gcc: f64,
    /// Finite samples in `logs_qc`.
    pub n_qc: usize,
    /// Whole-channel action.
    pub channel_action: ChannelAction,
    /// Unit rescue decision.
    pub rescue_reason: RescueReason,
    /// Rescue factor (NaN if none).
    pub rescue_factor: f64,
    /// `median / fleet_med`.
    pub rescue_level_ratio: f64,
    /// `std / |mean|`.
    pub rescue_cv: f64,
    /// Reference median (linear).
    pub rescue_ref_median_lin: f64,
    /// Normal wells behind the reference; 0 = fleet reference.
    pub rescue_n_ref: usize,
    /// Rows used for the stats; 0 if no stats row.
    pub stats_n: usize,
    /// Mean (native).
    pub mean_native: f64,
    /// Median (native).
    pub median_native: f64,
    /// Std (native).
    pub std_native: f64,
    /// Lower percentile (native).
    pub p_low_native: f64,
    /// Upper percentile (native).
    pub p_high_native: f64,
    /// Provisional ring-test bits.
    pub sigma_bits: u8,
    /// Largest own deviation.
    pub sigma_max_dev: f64,
    /// Cluster deviation.
    pub sigma_cluster_dev: f64,
    /// Baseline used.
    pub sigma_baseline: Baseline,
    /// Median ratio to the fleet median.
    pub unit_ratio: f64,
}

/// Check the `[qc]` values that stage 2 and the stage-4 checks rely on.
///
/// # Errors
/// `SxError::Config` naming the offending key.
pub(crate) fn validate_config(cfg: &QcConfig) -> SxResult<()> {
    let bad = |msg: String| Err(SxError::Config(msg));
    for (name, [lo, hi]) in [
        ("qc.tail_pct", cfg.tail_pct),
        ("qc.stats_pct", cfg.stats_pct),
        ("qc.rescue_candidate_band", cfg.rescue_candidate_band),
        ("qc.rescue_accept_band", cfg.rescue_accept_band),
        ("qc.unit_band", cfg.unit_band),
    ] {
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return bad(format!(
                "{name} must be two finite values, strictly increasing"
            ));
        }
    }
    for (name, [lo, hi]) in [
        ("qc.tail_pct", cfg.tail_pct),
        ("qc.stats_pct", cfg.stats_pct),
    ] {
        if lo < 0.0 || hi > 100.0 {
            return bad(format!("{name} must lie in [0, 100]"));
        }
    }
    if cfg.ring_first > cfg.ring_last {
        return bad("qc.ring_first must be <= qc.ring_last".into());
    }
    for (name, v) in [
        ("qc.constant_min_count", cfg.constant_min_count),
        ("qc.tail_min_count", cfg.tail_min_count),
        ("qc.rescue_min_count", cfg.rescue_min_count),
        ("qc.rescue_ref_wells", cfg.rescue_ref_wells),
        ("qc.rescue_min_ref_wells", cfg.rescue_min_ref_wells),
        ("qc.repeat_min_len", cfg.repeat_min_len),
        ("qc.repeat_min_count", cfg.repeat_min_count),
        ("qc.plateau_window", cfg.plateau_window),
        ("qc.plateau_window_min", cfg.plateau_window_min),
        ("qc.plateau_min_len", cfg.plateau_min_len),
        ("qc.plateau_min_count", cfg.plateau_min_count),
        ("qc.run_min_len", cfg.run_min_len),
        ("qc.stats_min_count", cfg.stats_min_count),
        ("qc.fleet_min_wells", cfg.fleet_min_wells),
        ("qc.post_flat_min_count", cfg.post_flat_min_count),
        ("qc.near_count", cfg.near_count),
        ("qc.near_min", cfg.near_min),
        ("qc.ring_first", cfg.ring_first),
        ("qc.ring_last", cfg.ring_last),
        ("qc.ring_min", cfg.ring_min),
        ("qc.jsd_k", cfg.jsd_k),
        ("qc.jsd_neighbors", cfg.jsd_neighbors),
        ("qc.jsd_min_neighbors", cfg.jsd_min_neighbors),
        ("qc.jsd_min_wells", cfg.jsd_min_wells),
        ("qc.jsd_max_count", cfg.jsd_max_count),
        ("qc.pileup_min_len", cfg.pileup_min_len),
    ] {
        if v == 0 {
            return bad(format!("{name} must be >= 1"));
        }
    }
    for (key, &f) in &cfg.rescue_factors {
        if LogKind::from_name(key).is_err() {
            return bad(format!("qc.rescue_factors: unknown log name {key:?}"));
        }
        if !(f.is_finite() && f > 0.0 && f != 1.0) {
            return bad(format!(
                "qc.rescue_factors.{key} must be finite, > 0 and != 1, got {f}"
            ));
        }
    }
    Ok(())
}

fn to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn finite_count(values: &[f32]) -> usize {
    values.iter().filter(|v| v.is_finite()).count()
}

/// Working state of one channel.
struct Chan {
    values: Vec<f32>,
    bits: Vec<u8>,
    n_raw: usize,
    action: ChannelAction,
    n_bad_hole: usize,
    n_implausible: usize,
    n_low_density_unverified: usize,
    n_closed_cal_rescued: usize,
    closed_cal_bias: f64,
}

/// Fleet-level results kept for the report.
#[derive(Clone, Copy)]
struct Info {
    rescue: Option<RescueRow>,
    stats: Option<ChannelStats>,
    ring: Option<RingRow>,
}

struct Work<'a> {
    id: WellId,
    name: &'a str,
    grid: GridRange,
    chans: Vec<Chan>, // suite order
    info: Vec<Info>,
}

/// Bounds mask on the channel's linear values.
fn bounds_step(log: LogKind, chan: &mut Chan) -> SxResult<()> {
    let b = units::physical_bounds(log);
    let mask = bounds_mask(&to_f64(&chan.values), b.lo, b.hi);
    apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_OUT_OF_BOUNDS)?;
    Ok(())
}

/// Steps 2 to 4: bounds, constant channel, tails.
fn first_pass(log: LogKind, chan: &mut Chan, cfg: &QcConfig) -> SxResult<()> {
    bounds_step(log, chan)?;
    if constant_channel(
        &to_f64(&chan.values),
        cfg.constant_min_count,
        cfg.constant_rel_tol,
    ) {
        blank_channel(&mut chan.values, &mut chan.bits)?;
        chan.action = ChannelAction::Constant;
    }
    let mask = tail_mask(
        &native_vec(log, &chan.values),
        cfg.tail_pct,
        cfg.tail_pad_frac,
        cfg.tail_min_count,
    );
    apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_TAIL)?;
    Ok(())
}

/// Steps 6 to 8: exact repeats, soft plateaus, short runs.
fn sequence_pass(log: LogKind, chan: &mut Chan, cfg: &QcConfig) -> SxResult<()> {
    let mask = repeat_mask(
        &to_f64(&chan.values),
        cfg.repeat_min_len,
        cfg.repeat_rel_tol,
        cfg.repeat_min_count,
    );
    apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_REPEAT)?;
    let params = PlateauParams {
        window: cfg.plateau_window,
        window_min: cfg.plateau_window_min,
        min_len: cfg.plateau_min_len,
        min_count: cfg.plateau_min_count,
        std_share: cfg.plateau_std_share,
    };
    let mask = plateau_mask(&native_vec(log, &chan.values), &params);
    apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_PLATEAU)?;
    let mask = short_run_mask(&to_f64(&chan.values), cfg.run_min_len);
    apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_SHORT_RUN)?;
    Ok(())
}

/// Run stage 2 (`qc`).
///
/// Applies the rules in the fixed order of the module doc and returns the
/// QC'd curves, the per-sample bits and the per-channel report. Per-well
/// steps run in parallel; fleet steps collect, then reduce in `WellId` order.
/// Infinite input samples are treated as missing (NaN, no bit). Empty
/// `input.logs` gives empty output.
///
/// Invariants: a sample is NaN with bits `0` exactly when it was NaN in the
/// input; every sample has at most one bit; `n_raw - n_qc` equals the sum of
/// the `n_*` counts of each report row.
///
/// # Errors
/// `SxError::Config` for a bad `cfg`; `SxError::Shape` if a curve or
/// provenance length differs from its grid, or a well in `logs` has no header
/// in `wells`; `SxError::InvalidInput` for a duplicate well id in `logs`.
pub fn run_qc_stage(input: &QcInput, cfg: &QcConfig) -> SxResult<QcStageOutput> {
    validate_config(cfg)?;
    if input.logs.is_empty() {
        return Ok(QcStageOutput {
            logs: Vec::new(),
            flags: Vec::new(),
            report: Vec::new(),
            low_density_unverified: Vec::new(),
            closed_caliper_samples: Vec::new(),
            closed_caliper_runs: Vec::new(),
        });
    }

    // Headers, coordinates.
    let mut headers: BTreeMap<WellId, &WellHeader> = BTreeMap::new();
    for h in input.wells {
        headers.entry(h.id).or_insert(h);
    }
    let xy: BTreeMap<WellId, (f64, f64)> = headers
        .iter()
        .filter(|(_, h)| h.x_m.is_finite() && h.y_m.is_finite())
        .map(|(id, h)| (*id, (h.x_m, h.y_m)))
        .collect();

    // Step 1: copy values, bits all 0.
    let mut sorted: Vec<&WellLogs> = input.logs.iter().collect();
    sorted.sort_by_key(|w| w.id);
    if let Some(pair) = sorted.windows(2).find(|p| match p {
        [a, b] => a.id == b.id,
        _ => false,
    }) {
        if let Some(w) = pair.first() {
            return Err(SxError::InvalidInput(format!(
                "duplicate well id {} in logs",
                w.id.0
            )));
        }
    }
    let mut works: Vec<Work> = Vec::with_capacity(sorted.len());
    for w in sorted {
        w.validate()?;
        let header = headers.get(&w.id).ok_or_else(|| {
            SxError::Shape(format!("well {} is in logs but has no header", w.id.0))
        })?;
        let n = w.grid.len;
        let chans = LogKind::ALL
            .iter()
            .map(|log| {
                let mut values = w
                    .curves
                    .get(log)
                    .cloned()
                    .unwrap_or_else(|| vec![f32::NAN; n]);
                for v in values.iter_mut() {
                    if v.is_infinite() {
                        *v = f32::NAN;
                    }
                }
                Chan {
                    n_raw: finite_count(&values),
                    values,
                    bits: vec![0u8; n],
                    action: ChannelAction::None,
                    n_bad_hole: 0,
                    n_implausible: 0,
                    n_low_density_unverified: 0,
                    n_closed_cal_rescued: 0,
                    closed_cal_bias: f64::NAN,
                }
            })
            .collect();
        works.push(Work {
            id: w.id,
            name: header.name.as_str(),
            grid: w.grid,
            chans,
            info: vec![
                Info {
                    rescue: None,
                    stats: None,
                    ring: None
                };
                LogKind::ALL.len()
            ],
        });
    }

    // Step 1b: bad hole (caliper / density correction) on `qc.badhole_logs`.
    let badhole: Vec<usize> = LogKind::ALL
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            cfg.badhole_logs
                .iter()
                .any(|n| n.eq_ignore_ascii_case(l.name()))
        })
        .map(|(i, _)| i)
        .collect();
    if !badhole.is_empty() {
        let aux: BTreeMap<WellId, &WellAux> = input.aux.iter().map(|a| (a.id, a)).collect();
        let params = BadHoleParams {
            open_min_in: cfg.cal_open_min_in,
            undergauge_in: cfg.cal_undergauge_in,
            washout_in: cfg.cal_washout_in,
            drho_limit: cfg.drho_limit,
        };
        for w in works.iter_mut() {
            let Some(a) = aux.get(&w.id) else { continue };
            if a.grid != w.grid {
                return Err(SxError::Shape(format!(
                    "well {}: aux grid differs from logs grid",
                    w.id.0
                )));
            }
            let (mask, _) = badhole_eval(&a.cal_in, &a.drho, &a.bs_in, params);
            for &li in &badhole {
                if let Some(chan) = w.chans.get_mut(li) {
                    chan.n_bad_hole +=
                        apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_OUT_OF_BOUNDS)?;
                }
            }
        }
    }

    // Step 1c: low density (RHOB < `qc.low_density_rhob`). Bad-hole samples are
    // already gone (step 1b); what is left is in-gauge where a caliper or DRHO
    // sample exists (coal, kept) or unverified where none does (kept, counted).
    let mut low_density_unverified: Vec<(WellId, i32, f32)> = Vec::new();
    if cfg.low_density_rhob > 0.0 {
        let aux: BTreeMap<WellId, &WellAux> = input.aux.iter().map(|a| (a.id, a)).collect();
        let params = BadHoleParams {
            open_min_in: cfg.cal_open_min_in,
            undergauge_in: cfg.cal_undergauge_in,
            washout_in: cfg.cal_washout_in,
            drho_limit: cfg.drho_limit,
        };
        let li = LogKind::ALL
            .iter()
            .position(|l| *l == LogKind::Rhob)
            .unwrap_or(0);
        for w in works.iter_mut() {
            let verified = aux
                .get(&w.id)
                .map(|a| badhole_eval(&a.cal_in, &a.drho, &a.bs_in, params).1);
            if let Some(chan) = w.chans.get_mut(li) {
                let before = low_density_unverified.len();
                low_density_unverified.extend(
                    chan.values
                        .iter()
                        .enumerate()
                        .filter(|&(i, &v)| {
                            v.is_finite()
                                && f64::from(v) < cfg.low_density_rhob
                                && !verified
                                    .as_ref()
                                    .is_some_and(|m| m.get(i).copied().unwrap_or(false))
                        })
                        .map(|(i, &v)| {
                            (
                                w.id,
                                w.grid.k_start + i32::try_from(i).unwrap_or(i32::MAX),
                                v,
                            )
                        }),
                );
                chan.n_low_density_unverified = low_density_unverified.len() - before;
            }
        }
    }

    // Steps 2-4.
    works.par_iter_mut().try_for_each(|w| -> SxResult<()> {
        for (chan, &log) in w.chans.iter_mut().zip(LogKind::ALL.iter()) {
            first_pass(log, chan, cfg)?;
        }
        Ok(())
    })?;

    // Step 5: unit rescue per log with a factor.
    for (li, &log) in LogKind::ALL.iter().enumerate() {
        let factor = rescue_factor(cfg, log);
        let lin: Vec<Vec<f64>> = works
            .iter()
            .map(|w| w.chans.get(li).map_or_else(Vec::new, |c| to_f64(&c.values)))
            .collect();
        let view: Vec<(WellId, &[f64])> = works
            .iter()
            .zip(lin.iter())
            .map(|(w, v)| (w.id, v.as_slice()))
            .collect();
        let rows = plan_rescue(log, factor, &view, &xy, cfg);
        drop(view);
        drop(lin);
        for (w, row) in works.iter_mut().zip(rows) {
            if row.reason == RescueReason::Rescued {
                if let Some(chan) = w.chans.get_mut(li) {
                    apply_rescue(&mut chan.values, row.factor);
                    bounds_step(log, chan)?;
                }
            }
            if let Some(info) = w.info.get_mut(li) {
                info.rescue = Some(row);
            }
        }
    }

    // Step 5b: plausible sample range (`qc.plausible_range`).
    for (li, &log) in LogKind::ALL.iter().enumerate() {
        let Some(&[lo, hi]) = cfg
            .plausible_range
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(log.name()))
            .map(|(_, v)| v)
        else {
            continue;
        };
        for w in works.iter_mut() {
            if let Some(chan) = w.chans.get_mut(li) {
                let mask = range_mask(&chan.values, lo, hi);
                chan.n_implausible +=
                    apply_mask(&mut chan.values, &mut chan.bits, &mask, QC_OUT_OF_BOUNDS)?;
            }
        }
    }

    // Steps 6-8.
    works.par_iter_mut().try_for_each(|w| -> SxResult<()> {
        for (chan, &log) in w.chans.iter_mut().zip(LogKind::ALL.iter()) {
            sequence_pass(log, chan, cfg)?;
        }
        Ok(())
    })?;

    // Steps 9-11 per log: stats, low variance, ring test, unit blank.
    for (li, &log) in LogKind::ALL.iter().enumerate() {
        let stats: Vec<Option<ChannelStats>> = works
            .par_iter()
            .map(|w| {
                let chan = w.chans.get(li)?;
                channel_stats(
                    w.id,
                    log,
                    &native_vec(log, &chan.values),
                    cfg.stats_min_count,
                    cfg.stats_pct,
                )
            })
            .collect();
        let rows: Vec<ChannelStats> = stats.iter().flatten().copied().collect();
        let lowvar: BTreeSet<WellId> =
            low_variance_wells(&rows, cfg.lowvar_ratio, cfg.fleet_min_wells);
        let med_band = cfg
            .plausible_median
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(log.name()))
            .map(|(_, v)| *v);
        let mut kept: Vec<ChannelStats> = Vec::with_capacity(rows.len());
        for (w, st) in works.iter_mut().zip(stats) {
            let st = match st {
                Some(s) => s,
                None => continue,
            };
            let median_lin = if log == LogKind::Resd {
                10f64.powf(st.median)
            } else {
                st.median
            };
            let implausible = med_band.is_some_and(|[lo, hi]| !(lo..=hi).contains(&median_lin));
            if implausible {
                if let Some(chan) = w.chans.get_mut(li) {
                    blank_channel(&mut chan.values, &mut chan.bits)?;
                    chan.action = ChannelAction::ImplausibleLevel;
                }
            } else if lowvar.contains(&w.id) {
                if let Some(chan) = w.chans.get_mut(li) {
                    blank_channel(&mut chan.values, &mut chan.bits)?;
                    chan.action = ChannelAction::LowVariance;
                }
            } else {
                kept.push(st);
                if let Some(info) = w.info.get_mut(li) {
                    info.stats = Some(st);
                }
            }
        }
        let ring = ring_test(log, &kept, &xy, cfg);
        let ring: BTreeMap<WellId, RingRow> = ring.into_iter().map(|r| (r.well, r)).collect();
        for w in works.iter_mut() {
            let row = match ring.get(&w.id) {
                Some(r) => *r,
                None => continue,
            };
            if row.bits & SIGMA_UNIT != 0 {
                if let Some(chan) = w.chans.get_mut(li) {
                    blank_channel(&mut chan.values, &mut chan.bits)?;
                    chan.action = ChannelAction::UnitMismatch;
                }
            }
            if let Some(info) = w.info.get_mut(li) {
                info.ring = Some(row);
            }
        }
    }

    // Step 11b: closed-caliper RHOB rescue.
    let (closed_caliper_samples, closed_caliper_runs) =
        closed_caliper_step(input, cfg, &headers, &xy, &mut works)?;

    // Step 12: output.
    tracing::info!(wells = works.len(), "qc stage: rules applied");
    let mut out_logs = Vec::with_capacity(works.len());
    let mut out_flags = Vec::with_capacity(works.len());
    let mut report = Vec::with_capacity(works.len() * LogKind::ALL.len());
    for w in works {
        tracing::debug!(well = w.id.0, "qc stage: well done");
        let mut logs = WellLogs::new(w.id, w.grid);
        let mut flags = QcFlags {
            id: w.id,
            grid: w.grid,
            bits: BTreeMap::new(),
        };
        for ((&log, chan), info) in LogKind::ALL.iter().zip(w.chans).zip(w.info) {
            report.push(report_row(w.id, w.name, log, &chan, &info));
            logs.provenance.insert(
                log,
                chan.values
                    .iter()
                    .map(|v| if v.is_finite() { 0 } else { CODE_NONE })
                    .collect(),
            );
            logs.curves.insert(log, chan.values);
            flags.bits.insert(log, chan.bits);
        }
        out_logs.push(logs);
        out_flags.push(flags);
    }
    Ok(QcStageOutput {
        logs: out_logs,
        flags: out_flags,
        report,
        low_density_unverified,
        closed_caliper_samples,
        closed_caliper_runs,
    })
}

/// Step 11b, see [`super::closed_cal`]. Only RHOB samples blanked by the
/// bad-hole rule (bit `QC_OUT_OF_BOUNDS`, `n_bad_hole`) where the caliper reads
/// closed are candidates. A rescued sample gets raw − bias, its bit cleared and
/// is moved from `n_bad_hole` to `n_closed_cal_rescued`.
fn closed_caliper_step(
    input: &QcInput,
    cfg: &QcConfig,
    headers: &BTreeMap<WellId, &WellHeader>,
    xy: &BTreeMap<WellId, (f64, f64)>,
    works: &mut [Work],
) -> SxResult<(Vec<ClosedCalSample>, Vec<ClosedCalRun>)> {
    let c = &cfg.closed_caliper;
    let li = LogKind::Rhob as usize;
    if !cfg
        .badhole_logs
        .iter()
        .any(|n| n.eq_ignore_ascii_case(LogKind::Rhob.name()))
    {
        return Ok((Vec::new(), Vec::new()));
    }
    let aux: BTreeMap<WellId, &WellAux> = input.aux.iter().map(|a| (a.id, a)).collect();
    let raw: BTreeMap<WellId, &Vec<f32>> = input
        .logs
        .iter()
        .filter_map(|w| w.curves.get(&LogKind::Rhob).map(|v| (w.id, v)))
        .collect();
    let kb = |id: WellId| headers.get(&id).map_or(f64::NAN, |h| h.kb_elev_ft);
    // Snapshot of every well's valid RHOB, binned by elevation.
    let valid: BTreeMap<WellId, BTreeMap<i64, f64>> = works
        .iter()
        .filter_map(|w| {
            let ch = w.chans.get(li)?;
            let k = kb(w.id);
            if !k.is_finite() {
                return None;
            }
            let e: Vec<f64> = (0..w.grid.len).map(|i| k - w.grid.md_ft(i)).collect();
            let v: Vec<f64> = ch.values.iter().map(|&x| f64::from(x)).collect();
            let curve = closed_cal::bin_curve(&e, &v, c.bin_ft);
            (!curve.is_empty()).then_some((w.id, curve))
        })
        .collect();
    let mut samples = Vec::new();
    let mut runs = Vec::new();
    for w in works.iter_mut() {
        let (Some(a), Some(r)) = (aux.get(&w.id), raw.get(&w.id)) else {
            continue;
        };
        let Some(chan) = w.chans.get_mut(li) else {
            continue;
        };
        if chan.action != ChannelAction::None {
            continue;
        }
        let closed = closed_caliper_mask(&a.cal_in, cfg.cal_open_min_in);
        let cand: Vec<bool> = (0..w.grid.len)
            .map(|i| {
                closed.get(i).copied().unwrap_or(false)
                    && r.get(i).is_some_and(|v| v.is_finite())
                    && chan.bits.get(i) == Some(&QC_OUT_OF_BOUNDS)
                    && chan.values.get(i).is_some_and(|v| v.is_nan())
            })
            .collect();
        let idx: Vec<usize> = (0..cand.len()).filter(|&i| cand[i]).collect();
        let (Some(&first), Some(&last)) = (idx.first(), idx.last()) else {
            continue;
        };
        let mut neighbors: Vec<(f64, &BTreeMap<i64, f64>)> = match xy.get(&w.id) {
            Some(&(x0, y0)) => valid
                .iter()
                .filter(|(id, _)| **id != w.id)
                .filter_map(|(id, curve)| xy.get(id).map(|&(x, y)| ((x - x0).hypot(y - y0), curve)))
                .collect(),
            None => Vec::new(),
        };
        neighbors.sort_by(|p, q| p.0.total_cmp(&q.0));
        let refs: Vec<&BTreeMap<i64, f64>> = neighbors.iter().map(|p| p.1).collect();
        let target = closed_cal::ClosedCalTarget {
            kb_ft: kb(w.id),
            grid: w.grid,
            raw: r,
            candidate: &cand,
            drho: &a.drho,
        };
        let res = closed_cal::rescue_well(&target, &refs, c)?;
        for &(i, v, e) in &res.kept {
            if let (Some(val), Some(bit)) = (chan.values.get_mut(i), chan.bits.get_mut(i)) {
                *val = v as f32;
                *bit = 0;
            }
            samples.push(ClosedCalSample {
                well_id: w.id,
                k: w.grid.k(i),
                md_ft: w.grid.md_ft(i),
                rhob_raw_gcc: f64::from(r[i]),
                rhob_qc_gcc: v,
                bias_gcc: res.bias,
                env_lo: e.lo,
                env_p50: e.mid,
                env_hi: e.hi,
            });
        }
        let n_rescued = res.kept.len();
        chan.n_bad_hole -= n_rescued;
        chan.n_closed_cal_rescued = n_rescued;
        chan.closed_cal_bias = res.bias;
        let speed = |want_closed: bool| {
            let v: Vec<f64> = (0..w.grid.len)
                .filter(|&i| {
                    let cl = closed.get(i).copied().unwrap_or(false);
                    let finite_cal = a.cal_in.get(i).is_some_and(|x| x.is_finite());
                    finite_cal && cl == want_closed
                })
                .filter_map(|i| a.lspd_ftmin.get(i).map(|&x| f64::from(x)))
                .filter(|x| x.is_finite())
                .collect();
            median(&v)
        };
        let (sc, so) = (speed(true), speed(false));
        let kind = if sc.is_finite() && so.is_finite() && so > 0.0 && sc >= 1.5 * so {
            "fast pass, arm closed"
        } else {
            "arm closed"
        };
        let speed_txt = if sc.is_finite() && so.is_finite() {
            format!("logging speed {sc:.0} ft/min closed vs {so:.0} ft/min open")
        } else {
            "logging speed not recorded".to_string()
        };
        let (top, base) = (w.grid.md_ft(first), w.grid.md_ft(last));
        runs.push(ClosedCalRun {
            well_id: w.id,
            well_name: w.name.to_string(),
            md_top_ft: top,
            md_base_ft: base,
            n_closed: idx.len(),
            n_rejected: idx.len() - n_rescued,
            n_rescued,
            n_despiked: res.n_despiked,
            n_with_envelope: res.n_with_envelope,
            n_neighbors: res.n_neighbors,
            bias_gcc: res.bias,
            status: res.status.as_str(),
            lspd_closed_ftmin: sc,
            lspd_open_ftmin: so,
            note: format!(
                "{kind}: MD {top:.1}-{base:.1} ft, {speed_txt}, {} RHOB samples rejected, {n_rescued} rescued ({})",
                idx.len() - n_rescued,
                res.status.as_str()
            ),
        });
    }
    Ok((samples, runs))
}

fn count_bit(bits: &[u8], bit: u8) -> usize {
    bits.iter().filter(|&&b| b & bit != 0).count()
}

fn report_row(id: WellId, name: &str, log: LogKind, chan: &Chan, info: &Info) -> QcChannelRow {
    let (reason, factor, level_ratio, cv, ref_median, n_ref) = match &info.rescue {
        Some(r) => (
            r.reason,
            r.factor,
            r.level_ratio,
            r.cv,
            r.ref_median,
            r.n_ref,
        ),
        None => (
            RescueReason::NoFactor,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0,
        ),
    };
    let st = info.stats;
    let nan = f64::NAN;
    QcChannelRow {
        well_id: id,
        well_name: name.to_string(),
        log,
        n_raw: chan.n_raw,
        n_out_of_bounds: count_bit(&chan.bits, QC_OUT_OF_BOUNDS),
        n_tail: count_bit(&chan.bits, QC_TAIL),
        n_repeat: count_bit(&chan.bits, QC_REPEAT),
        n_plateau: count_bit(&chan.bits, QC_PLATEAU),
        n_short_run: count_bit(&chan.bits, QC_SHORT_RUN),
        n_channel: count_bit(&chan.bits, QC_CHANNEL),
        n_bad_hole: chan.n_bad_hole,
        n_implausible: chan.n_implausible,
        n_low_density_unverified: chan.n_low_density_unverified,
        n_closed_cal_rescued: chan.n_closed_cal_rescued,
        closed_cal_bias_gcc: chan.closed_cal_bias,
        n_qc: finite_count(&chan.values),
        channel_action: chan.action,
        rescue_reason: reason,
        rescue_factor: factor,
        rescue_level_ratio: level_ratio,
        rescue_cv: cv,
        rescue_ref_median_lin: ref_median,
        rescue_n_ref: n_ref,
        stats_n: st.map_or(0, |s| s.n),
        mean_native: st.map_or(nan, |s| s.mean),
        median_native: st.map_or(nan, |s| s.median),
        std_native: st.map_or(nan, |s| s.std),
        p_low_native: st.map_or(nan, |s| s.p_low),
        p_high_native: st.map_or(nan, |s| s.p_high),
        sigma_bits: info.ring.map_or(0, |r| r.bits),
        sigma_max_dev: info.ring.map_or(nan, |r| r.max_dev),
        sigma_cluster_dev: info.ring.map_or(nan, |r| r.cluster_dev),
        sigma_baseline: info.ring.map_or(Baseline::None, |r| r.baseline),
        unit_ratio: info.ring.map_or(nan, |r| r.unit_ratio),
    }
}

// ----------------------------------------------------------------- batches

/// `logs_qc` batch: the same layout as `logs_raw` (`well_id`, `k`, `md_ft`,
/// then the six `LogKind::column()` value columns as f32), tagged `schema`
/// (use [`LOGS_QC_TAG`]). Rows are sorted by id then `k`; NaN = missing or
/// blanked.
///
/// # Errors
/// As `io::products::logs_batch`: `SxError::InvalidInput` (duplicate id,
/// empty tag), `SxError::Shape` (a well fails validation).
pub fn logs_batch(logs: &[WellLogs], schema: &str) -> SxResult<RecordBatch> {
    crate::io::products::logs_batch(logs, schema)
}

/// `qc_flags` batch: `well_id` int32, `k` int32, `md_ft` float64, then
/// `gr_qc_bits`, `rhob_qc_bits`, `nphi_qc_bits`, `dtc_qc_bits`,
/// `resd_qc_bits`, `pef_qc_bits` (uint8), tagged `schema` (use
/// [`QC_FLAGS_TAG`]). Rows are sorted by id then `k`. A log without an entry
/// in `bits` is written as zeros.
///
/// # Errors
/// `SxError::InvalidInput` (duplicate id, empty tag), `SxError::Shape` (a bits
/// vector whose length differs from the grid).
pub fn flags_batch(flags: &[QcFlags], schema: &str) -> SxResult<RecordBatch> {
    let mut sorted: Vec<&QcFlags> = flags.iter().collect();
    sorted.sort_by_key(|f| f.id);
    if sorted.windows(2).any(|p| match p {
        [a, b] => a.id == b.id,
        _ => false,
    }) {
        return Err(SxError::InvalidInput("duplicate well id in flags".into()));
    }
    let total: usize = sorted.iter().map(|f| f.grid.len).sum();
    let mut ids = Vec::with_capacity(total);
    let mut ks = Vec::with_capacity(total);
    let mut mds = Vec::with_capacity(total);
    for f in &sorted {
        for i in 0..f.grid.len {
            ids.push(f.id.0);
            ks.push(f.grid.k(i));
            mds.push(f.grid.md_ft(i));
        }
    }
    let mut fields = vec![
        Field::new("well_id", DataType::Int32, false),
        Field::new("k", DataType::Int32, false),
        Field::new("md_ft", DataType::Float64, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(ids)),
        Arc::new(Int32Array::from(ks)),
        Arc::new(Float64Array::from(mds)),
    ];
    for log in LogKind::ALL {
        let mut col: Vec<u8> = Vec::with_capacity(total);
        for f in &sorted {
            match f.bits.get(&log) {
                Some(b) if b.len() == f.grid.len => col.extend_from_slice(b),
                Some(b) => {
                    return Err(SxError::Shape(format!(
                        "well {}: {} bits have {} samples, grid has {}",
                        f.id.0,
                        log.name(),
                        b.len(),
                        f.grid.len
                    )))
                }
                None => col.resize(col.len() + f.grid.len, 0),
            }
        }
        fields.push(Field::new(
            format!("{}_qc_bits", log.name().to_ascii_lowercase()),
            DataType::UInt8,
            false,
        ));
        columns.push(Arc::new(UInt8Array::from(col)));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, schema)
}

fn i64_of(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// `qc_report` batch with the 34 columns of `qc_report.csv` in order, rows
/// sorted by `well_id` then suite order of the log. Schema tag
/// [`QC_REPORT_TAG`] (ignored by the CSV writer). Integer counts are int64;
/// NaN stays NaN (the CSV writer prints an empty field).
///
/// # Errors
/// `SxError::InvalidInput` if the batch cannot be built.
/// Build the `low_density_unverified.csv` batch: `well_id`, `k`, `rhob_gcc`,
/// one row per low-density RHOB sample kept without a caliper or
/// density-correction check (sorted by well then `k`).
///
/// # Errors
/// `SxError::InvalidInput` when the batch cannot be built.
pub fn low_density_batch(rows: &[(WellId, i32, f32)]) -> SxResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("well_id", DataType::Int32, false),
        Field::new("k", DataType::Int32, false),
        Field::new("rhob_gcc", DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.0 .0))),
        Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.1))),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| f64::from(r.2)),
        )),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))
}

pub fn qc_report_batch(rows: &[QcChannelRow]) -> SxResult<RecordBatch> {
    let mut sorted: Vec<&QcChannelRow> = rows.iter().collect();
    sorted.sort_by_key(|r| (r.well_id, r.log));
    let i32s = |f: fn(&QcChannelRow) -> i32| -> ArrayRef {
        Arc::new(Int32Array::from_iter_values(sorted.iter().map(|r| f(r))))
    };
    let strs = |f: &dyn Fn(&QcChannelRow) -> String| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(sorted.iter().map(|r| f(r))))
    };
    let counts = |f: fn(&QcChannelRow) -> usize| -> ArrayRef {
        Arc::new(Int64Array::from_iter_values(
            sorted.iter().map(|r| i64_of(f(r))),
        ))
    };
    let floats = |f: fn(&QcChannelRow) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(sorted.iter().map(|r| f(r))))
    };
    let cols: Vec<(&str, DataType, ArrayRef)> = vec![
        ("well_id", DataType::Int32, i32s(|r| r.well_id.0)),
        ("well_name", DataType::Utf8, strs(&|r| r.well_name.clone())),
        ("log", DataType::Utf8, strs(&|r| r.log.name().to_string())),
        ("n_raw", DataType::Int64, counts(|r| r.n_raw)),
        (
            "n_out_of_bounds",
            DataType::Int64,
            counts(|r| r.n_out_of_bounds),
        ),
        ("n_tail", DataType::Int64, counts(|r| r.n_tail)),
        ("n_repeat", DataType::Int64, counts(|r| r.n_repeat)),
        ("n_plateau", DataType::Int64, counts(|r| r.n_plateau)),
        ("n_short_run", DataType::Int64, counts(|r| r.n_short_run)),
        ("n_channel", DataType::Int64, counts(|r| r.n_channel)),
        ("n_bad_hole", DataType::Int64, counts(|r| r.n_bad_hole)),
        (
            "n_implausible",
            DataType::Int64,
            counts(|r| r.n_implausible),
        ),
        (
            "n_low_density_unverified",
            DataType::Int64,
            counts(|r| r.n_low_density_unverified),
        ),
        (
            "n_closed_cal_rescued",
            DataType::Int64,
            counts(|r| r.n_closed_cal_rescued),
        ),
        (
            "closed_cal_bias_gcc",
            DataType::Float64,
            floats(|r| r.closed_cal_bias_gcc),
        ),
        ("n_qc", DataType::Int64, counts(|r| r.n_qc)),
        (
            "channel_action",
            DataType::Utf8,
            strs(&|r| r.channel_action.as_str().to_string()),
        ),
        (
            "rescue_reason",
            DataType::Utf8,
            strs(&|r| r.rescue_reason.as_str().to_string()),
        ),
        (
            "rescue_factor",
            DataType::Float64,
            floats(|r| r.rescue_factor),
        ),
        (
            "rescue_level_ratio",
            DataType::Float64,
            floats(|r| r.rescue_level_ratio),
        ),
        ("rescue_cv", DataType::Float64, floats(|r| r.rescue_cv)),
        (
            "rescue_ref_median_lin",
            DataType::Float64,
            floats(|r| r.rescue_ref_median_lin),
        ),
        ("rescue_n_ref", DataType::Int64, counts(|r| r.rescue_n_ref)),
        ("stats_n", DataType::Int64, counts(|r| r.stats_n)),
        ("mean_native", DataType::Float64, floats(|r| r.mean_native)),
        (
            "median_native",
            DataType::Float64,
            floats(|r| r.median_native),
        ),
        ("std_native", DataType::Float64, floats(|r| r.std_native)),
        (
            "p_low_native",
            DataType::Float64,
            floats(|r| r.p_low_native),
        ),
        (
            "p_high_native",
            DataType::Float64,
            floats(|r| r.p_high_native),
        ),
        (
            "sigma_bits",
            DataType::UInt8,
            Arc::new(UInt8Array::from_iter_values(
                sorted.iter().map(|r| r.sigma_bits),
            )),
        ),
        (
            "sigma_max_dev",
            DataType::Float64,
            floats(|r| r.sigma_max_dev),
        ),
        (
            "sigma_cluster_dev",
            DataType::Float64,
            floats(|r| r.sigma_cluster_dev),
        ),
        (
            "sigma_baseline",
            DataType::Utf8,
            strs(&|r| r.sigma_baseline.as_str().to_string()),
        ),
        ("unit_ratio", DataType::Float64, floats(|r| r.unit_ratio)),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, t, _)| Field::new(*n, t.clone(), false))
        .collect();
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, _, a)| a).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, QC_REPORT_TAG)
}

/// `rhob_closed_caliper.parquet` batch: one row per rescued sample with the raw
/// and corrected RHOB, the bias and the envelope.
///
/// # Errors
/// `SxError::InvalidInput` if Arrow rejects the batch.
pub fn closed_cal_samples_batch(rows: &[ClosedCalSample]) -> SxResult<RecordBatch> {
    let f = |g: fn(&ClosedCalSample) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(rows.iter().map(g)))
    };
    let cols: Vec<(&str, DataType, ArrayRef)> = vec![
        (
            "well_id",
            DataType::Int32,
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.well_id.0),
            )),
        ),
        (
            "k",
            DataType::Int32,
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.k))),
        ),
        ("rhob_raw_gcc", DataType::Float64, f(|r| r.rhob_raw_gcc)),
        ("rhob_qc_gcc", DataType::Float64, f(|r| r.rhob_qc_gcc)),
        ("bias_gcc", DataType::Float64, f(|r| r.bias_gcc)),
        ("env_lo_gcc", DataType::Float64, f(|r| r.env_lo)),
        ("env_p50_gcc", DataType::Float64, f(|r| r.env_p50)),
        ("env_hi_gcc", DataType::Float64, f(|r| r.env_hi)),
        // Sparse (rescued rows only), so md_ft is not the third column: the
        // per-sample row-order contract applies to dense well/k/md tables.
        ("md_ft", DataType::Float64, f(|r| r.md_ft)),
    ];
    simple_batch(cols, Some(RHOB_CLOSED_CALIPER_TAG))
}

/// `closed_caliper_runs.csv` batch: one row per well with closed-caliper RHOB.
///
/// # Errors
/// `SxError::InvalidInput` if Arrow rejects the batch.
pub fn closed_cal_runs_batch(rows: &[ClosedCalRun]) -> SxResult<RecordBatch> {
    let f = |g: fn(&ClosedCalRun) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(rows.iter().map(g)))
    };
    let n = |g: fn(&ClosedCalRun) -> usize| -> ArrayRef {
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| i64_of(g(r))),
        ))
    };
    let t = |g: &dyn Fn(&ClosedCalRun) -> String| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(rows.iter().map(g)))
    };
    let cols: Vec<(&str, DataType, ArrayRef)> = vec![
        (
            "well_id",
            DataType::Int32,
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.well_id.0),
            )),
        ),
        ("well_name", DataType::Utf8, t(&|r| r.well_name.clone())),
        ("md_top_ft", DataType::Float64, f(|r| r.md_top_ft)),
        ("md_base_ft", DataType::Float64, f(|r| r.md_base_ft)),
        ("n_closed", DataType::Int64, n(|r| r.n_closed)),
        ("n_rejected", DataType::Int64, n(|r| r.n_rejected)),
        ("n_rescued", DataType::Int64, n(|r| r.n_rescued)),
        ("n_despiked", DataType::Int64, n(|r| r.n_despiked)),
        ("n_with_envelope", DataType::Int64, n(|r| r.n_with_envelope)),
        ("n_neighbors", DataType::Int64, n(|r| r.n_neighbors)),
        ("bias_gcc", DataType::Float64, f(|r| r.bias_gcc)),
        ("status", DataType::Utf8, t(&|r| r.status.to_string())),
        (
            "lspd_closed_ftmin",
            DataType::Float64,
            f(|r| r.lspd_closed_ftmin),
        ),
        (
            "lspd_open_ftmin",
            DataType::Float64,
            f(|r| r.lspd_open_ftmin),
        ),
        ("note", DataType::Utf8, t(&|r| r.note.clone())),
    ];
    simple_batch(cols, None)
}

fn simple_batch(cols: Vec<(&str, DataType, ArrayRef)>, tag: Option<&str>) -> SxResult<RecordBatch> {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, t, _)| Field::new(*n, t.clone(), false))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        cols.into_iter().map(|c| c.2).collect(),
    )
    .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    match tag {
        Some(tag) => with_schema_tag(batch, tag),
        None => Ok(batch),
    }
}
