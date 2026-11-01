// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Model write: two-point range map, pre-write rejection, the
//! exact-line test and the gaps-only write.
//!
//! No step writes a straight line: besides the
//! straight-line test on each candidate run, every write also checks the
//! stretch of written rows it joins (earlier fills of the log plus the new
//! candidates; measured rows break a stretch). A stretch of `n >= 3` written
//! rows may not hold an exact-line stretch of `min(n, line_min_rows)` rows
//!; candidate rows inside one are rejected
//! (`exact_line`). This catches whole 3–4-row gaps and lines across steps.

use std::collections::BTreeMap;
use std::ops::Range;

use tracing::debug;

use crate::config::FillConfig;
use crate::error::{SxError, SxResult};
use crate::fill::{
    curve, curve_mut, in_coverage, is_fill_code, is_open, measured_stats, pop_std, true_runs,
    FillReportRow, FillStep, ReportKind, WellFillState,
};
use crate::neighbors::NeighborCurve;
use crate::types::{LogKind, Provenance};
use crate::units::{working_bounds, Bounds};
use crate::util::percentiles;

/// Why a two-point map was or was not applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeReason {
    /// Gain and offset from the two points.
    Mapped,
    /// Prediction span too small: gain 1, offset only.
    ShiftOnly,
    /// Prediction span at or below the degenerate threshold: identity.
    Degenerate,
    /// Too few finite values (or a non-finite percentile): identity.
    Insufficient,
    /// Log not in `range_map_logs`: identity.
    Off,
}

impl RangeReason {
    /// Report text (`mapped`, `shift_only`, `degenerate`, `insufficient`, `off`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            RangeReason::Mapped => "mapped",
            RangeReason::ShiftOnly => "shift_only",
            RangeReason::Degenerate => "degenerate",
            RangeReason::Insufficient => "insufficient",
            RangeReason::Off => "off",
        }
    }
}

/// A two-point map `y' = gain · y + offset` (offset in the log's working unit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPoint {
    /// Gain (unitless).
    pub gain: f64,
    /// Offset, working unit of the log (RESD log10).
    pub offset: f64,
    /// Why this map.
    pub reason: RangeReason,
}

impl TwoPoint {
    fn identity(reason: RangeReason) -> Self {
        Self {
            gain: 1.0,
            offset: 0.0,
            reason,
        }
    }
}

/// Two-point map of (c_low, c_high) onto (r_low, r_high).
/// Identity is `gain = 1, offset = 0`.
///
/// `|c_high − c_low| <= degenerate` → identity (`Degenerate`); a span below
/// `min_span_frac · |r_high − r_low|` → gain 1 (`ShiftOnly`); else
/// `gain = sign(g) · clamp(|g|, 1/max_gain, max_gain)` (`Mapped`).
/// NaN rule: a non-finite input gives identity (`Insufficient`). Errors: none.
pub fn two_point(
    c_low: f64,
    c_high: f64,
    r_low: f64,
    r_high: f64,
    max_gain: f64,
    min_span_frac: f64,
    degenerate: f64,
) -> TwoPoint {
    if ![c_low, c_high, r_low, r_high].iter().all(|v| v.is_finite()) {
        return TwoPoint::identity(RangeReason::Insufficient);
    }
    let den = c_high - c_low;
    if den.abs() <= degenerate {
        return TwoPoint::identity(RangeReason::Degenerate);
    }
    let span = r_high - r_low;
    let (gain, reason) = if den.abs() < min_span_frac * span.abs() {
        (1.0, RangeReason::ShiftOnly)
    } else {
        let g = span / den;
        let (lo, hi) = (
            (1.0 / max_gain).min(max_gain),
            (1.0 / max_gain).max(max_gain),
        );
        (g.signum() * g.abs().max(lo).min(hi), RangeReason::Mapped)
    };
    let offset = (r_low + r_high) / 2.0 - gain * (c_low + c_high) / 2.0;
    TwoPoint {
        gain,
        offset,
        reason,
    }
}

/// On one well. `pred` and `nbr` have the grid length; only coverage rows are
/// finite in `pred`. Returns mapped predictions (same length) and the map used.
///
/// `nbr` is used as given (callers pass it restricted to coverage rows). Needs
/// `range_min_n` finite values in each; percentiles `range_p_low` / `range_p_high`
/// (linear, position `(N−1)·p/100`). A mapped value outside the working bounds
/// keeps the raw prediction. NaN rule: skip (NaN predictions stay NaN).
/// Errors: `Shape` if `nbr` length differs; `InvalidInput` for a bad percentile.
pub fn fill_range_map(
    pred: &[f32],
    nbr: Option<&[f32]>,
    log: LogKind,
    cfg: &FillConfig,
) -> SxResult<(Vec<f32>, TwoPoint)> {
    if let Some(r) = nbr {
        if r.len() != pred.len() {
            return Err(SxError::Shape(format!(
                "range map: neighbor has {} rows, prediction {}",
                r.len(),
                pred.len()
            )));
        }
    }
    if !cfg.range_map_logs.contains(&log) {
        return Ok((pred.to_vec(), TwoPoint::identity(RangeReason::Off)));
    }
    // Paired rows only: both percentile pairs must describe the same depths, or a
    // neighbor curve that covers part of the well shifts the map.
    let (p, r): (Vec<f64>, Vec<f64>) = match nbr {
        Some(nb) => pred
            .iter()
            .zip(nb)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(&a, &b)| (f64::from(a), f64::from(b)))
            .unzip(),
        None => (Vec::new(), Vec::new()),
    };
    if p.len() < cfg.range_min_n {
        return Ok((pred.to_vec(), TwoPoint::identity(RangeReason::Insufficient)));
    }
    let ps = [cfg.range_p_low, cfg.range_p_high];
    let (pl, ph) = two_values(percentiles(&p, &ps)?)?;
    let (rl, rh) = two_values(percentiles(&r, &ps)?)?;
    let tp = two_point(
        pl,
        ph,
        rl,
        rh,
        cfg.max_gain,
        cfg.min_span_frac,
        cfg.degenerate_span,
    );
    if !matches!(tp.reason, RangeReason::Mapped | RangeReason::ShiftOnly) {
        return Ok((pred.to_vec(), tp));
    }
    let b = working_bounds(log);
    let out = pred
        .iter()
        .map(|&y| {
            if !y.is_finite() {
                return y;
            }
            let m = tp.gain * f64::from(y) + tp.offset;
            if b.contains(m) {
                m as f32
            } else {
                y
            }
        })
        .collect();
    Ok((out, tp))
}

/// Why a candidate was rejected, in report order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectReason {
    /// Outside the physical bounds.
    Physical,
    /// Flagged stretch with `max − min <= exact_flat_tol`.
    ExactFlat,
    /// Flagged stretch without a quiet neighbor reference.
    SoftPlateau,
    /// Run mostly at a bound.
    Saturation,
    /// Part of an exact straight line.
    ExactLine,
}

impl RejectReason {
    /// Report text (`physical`, `exact_flat`, `soft_plateau`, `saturation`, `exact_line`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            RejectReason::Physical => "physical",
            RejectReason::ExactFlat => "exact_flat",
            RejectReason::SoftPlateau => "soft_plateau",
            RejectReason::Saturation => "saturation",
            RejectReason::ExactLine => "exact_line",
        }
    }
}

/// `cand` NaN = no candidate. Output per row: `None` = kept or no candidate.
/// NaN rule: skip. Errors: `Shape` if `nbr_ref` length differs.
///
/// `measured_std` is `σ_meas` (0 if fewer than 2 measured values). Runs are
/// consecutive grid rows of in-bounds candidates. A row hit by
/// several tests reports the first in step order (flat scan, saturation, exact
/// line); "at least `len/2`" quiet neighbor values means `2·count >= len`.
/// Complexity: `O(n × plateau_window)`.
pub fn reject_candidates(
    cand: &[f32],
    measured_std: f64,
    nbr_ref: Option<&[f32]>,
    log: LogKind,
    cfg: &FillConfig,
) -> SxResult<Vec<Option<RejectReason>>> {
    if let Some(r) = nbr_ref {
        if r.len() != cand.len() {
            return Err(SxError::Shape(format!(
                "rejection: neighbor reference has {} rows, candidates {}",
                r.len(),
                cand.len()
            )));
        }
    }
    let b = working_bounds(log);
    let mut out = vec![None; cand.len()];
    let mut ok = vec![false; cand.len()];
    for ((o, k), &v) in out.iter_mut().zip(ok.iter_mut()).zip(cand) {
        if v.is_nan() {
            continue;
        }
        if b.contains(f64::from(v)) {
            *k = true;
        } else {
            *o = Some(RejectReason::Physical);
        }
    }
    for run in true_runs(&ok) {
        let Some(vals32) = cand.get(run.clone()) else {
            continue;
        };
        let vals: Vec<f64> = vals32.iter().map(|&v| f64::from(v)).collect();
        let nbr_run = nbr_ref.and_then(|r| r.get(run.clone()));
        let mut flags: Vec<Option<RejectReason>> = vec![None; vals.len()];
        if vals.len() >= cfg.flat_min_run {
            let nbr_finite = nbr_run.map(finite_f64).unwrap_or_default();
            let sigma_nbr = if nbr_finite.len() >= 2 {
                pop_std(&nbr_finite)
            } else {
                0.0
            };
            let scale = measured_std
                .max(sigma_nbr)
                .max(pop_std(&vals))
                .max(cfg.scale_floor);
            flat_scan(&vals, nbr_run, scale, cfg, &mut flags);
            saturation(&vals, b, cfg, &mut flags);
        }
        if vals.len() >= cfg.line_min_rows {
            for s in exact_line_stretches(vals32, cfg.line_min_rows, cfg.line_rel_tol) {
                for f in flags.get_mut(s).into_iter().flatten() {
                    f.get_or_insert(RejectReason::ExactLine);
                }
            }
        }
        for (o, f) in out.get_mut(run).into_iter().flatten().zip(flags) {
            if f.is_some() {
                *o = f;
            }
        }
    }
    Ok(out)
}

/// On one run.
fn flat_scan(
    vals: &[f64],
    nbr: Option<&[f32]>,
    scale: f64,
    cfg: &FillConfig,
    flags: &mut [Option<RejectReason>],
) {
    let n = vals.len();
    let mut flagged = vec![false; n];
    let mut i = 0;
    while let Some(&vi) = vals.get(i) {
        let tol = cfg.near_const_abs.max(vi.abs() * cfg.near_const_rel);
        let len = vals
            .iter()
            .skip(i + 1)
            .take_while(|&&vj| (vj - vi).abs() <= tol)
            .count()
            + 1;
        if len >= cfg.flat_min_run {
            for f in flagged.iter_mut().skip(i).take(len) {
                *f = true;
            }
        }
        i += len;
    }
    let w = cfg.plateau_window.max(1);
    let min_count = cfg.plateau_min_count.max(2);
    let thr = cfg.plateau_std_frac * scale;
    let marked: Vec<bool> = (0..n)
        .map(|j| {
            let lo = (j + 1).saturating_sub(w);
            vals.get(lo..=j)
                .is_some_and(|win| win.len() >= min_count && sample_std(win) < thr)
        })
        .collect();
    for s in true_runs(&marked) {
        if s.len() >= cfg.flat_min_run {
            for f in flagged.get_mut(s).into_iter().flatten() {
                *f = true;
            }
        }
    }
    for s in true_runs(&flagged) {
        let Some(seg) = vals.get(s.clone()) else {
            continue;
        };
        let mn = seg.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = seg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let reason = if mx - mn <= cfg.exact_flat_tol {
            Some(RejectReason::ExactFlat)
        } else if is_quiet(nbr.and_then(|r| r.get(s.clone())), s.len(), scale, cfg) {
            None
        } else {
            Some(RejectReason::SoftPlateau)
        };
        if let Some(r) = reason {
            for f in flags.get_mut(s).into_iter().flatten() {
                f.get_or_insert(r);
            }
        }
    }
}

/// On one run.
fn saturation(vals: &[f64], b: Bounds, cfg: &FillConfig, flags: &mut [Option<RejectReason>]) {
    if vals.is_empty() {
        return;
    }
    let tol = cfg.saturation_rel_tol * (b.hi - b.lo).abs().max(1.0);
    let at = vals
        .iter()
        .filter(|&&v| (v - b.lo).abs() <= tol || (v - b.hi).abs() <= tol)
        .count();
    if at as f64 >= cfg.saturation_frac * vals.len() as f64 {
        for f in flags.iter_mut() {
            f.get_or_insert(RejectReason::Saturation);
        }
    }
}

/// Quiet reference over a stretch of `len` rows: at least
/// `max(quiet_min_count, len/2)` finite values with std `< quiet_std_frac × scale`.
pub(crate) fn is_quiet(nbr: Option<&[f32]>, len: usize, scale: f64, cfg: &FillConfig) -> bool {
    let Some(r) = nbr else {
        return false;
    };
    let v = finite_f64(r);
    v.len() >= cfg.quiet_min_count && 2 * v.len() >= len && pop_std(&v) < cfg.quiet_std_frac * scale
}

/// Maximal stretches (index ranges into `v`) of at least `min_rows` consecutive finite values
/// lying on one straight line. `min_rows >= 3`.
///
/// Every second difference must satisfy `|v_{j−1} − 2v_j + v_{j+1}| <= rel_tol ×
/// max(|v_{j−1}|, |v_j|, |v_{j+1}|, 1)`, in `f64`. A `min_rows` below 3 is
/// treated as 3. Stretches may share an end row. NaN rule: skip (a NaN breaks a
/// stretch). Errors: none. Empty input: empty output.
pub fn exact_line_stretches(v: &[f32], min_rows: usize, rel_tol: f64) -> Vec<Range<usize>> {
    let min_rows = min_rows.max(3);
    let ok: Vec<bool> = v
        .windows(3)
        .map(|w| match w {
            [a, b, c] => {
                let (a, b, c) = (f64::from(*a), f64::from(*b), f64::from(*c));
                a.is_finite()
                    && b.is_finite()
                    && c.is_finite()
                    && (a - 2.0 * b + c).abs()
                        <= rel_tol * a.abs().max(b.abs()).max(c.abs()).max(1.0)
            }
            _ => false,
        })
        .collect();
    true_runs(&ok)
        .into_iter()
        .map(|r| r.start..r.end + 2)
        .filter(|r| r.len() >= min_rows)
        .collect()
}

/// Candidate rows (finite `cand`) that would join a straight line of written
/// rows (module doc). `values`, `prov`, `cand` have the grid length.
pub(crate) fn written_line_rows(
    values: &[f32],
    prov: &[u8],
    cand: &[f32],
    cfg: &FillConfig,
) -> Vec<bool> {
    let view: Vec<f32> = cand
        .iter()
        .zip(values.iter().zip(prov))
        .map(|(&c, (&v, &p))| {
            if c.is_finite() {
                c
            } else if is_fill_code(p) && v.is_finite() {
                v
            } else {
                f32::NAN
            }
        })
        .collect();
    let finite: Vec<bool> = view.iter().map(|v| v.is_finite()).collect();
    let mut out = vec![false; cand.len()];
    let reach = cfg.line_min_rows.max(3);
    for run in true_runs(&finite) {
        let n = run.len();
        let Some(seg) = view.get(run.clone()) else {
            continue;
        };
        if n < 3 {
            continue;
        }
        for s in exact_line_stretches(seg, n.min(reach), cfg.line_rel_tol) {
            for j in s {
                let i = run.start + j;
                if cand.get(i).is_some_and(|c| c.is_finite()) {
                    if let Some(o) = out.get_mut(i) {
                        *o = true;
                    }
                }
            }
        }
    }
    out
}

/// True if the gap candidate `c` at rows `gap` joins a straight line with the
/// written rows next to it (same test as [`written_line_rows`], local).
pub(crate) fn gap_joins_line(
    values: &[f32],
    prov: &[u8],
    gap: Range<usize>,
    c: &[f32],
    cfg: &FillConfig,
) -> bool {
    let reach = cfg.line_min_rows.max(3);
    let written = |i: usize| {
        prov.get(i).is_some_and(|&p| is_fill_code(p))
            && values.get(i).is_some_and(|v| v.is_finite())
    };
    let mut left = 0;
    while left < reach && gap.start > left && written(gap.start - left - 1) {
        left += 1;
    }
    let mut right = 0;
    while right < reach && written(gap.end + right) {
        right += 1;
    }
    let mut view: Vec<f32> = Vec::with_capacity(left + c.len() + right);
    view.extend_from_slice(values.get(gap.start - left..gap.start).unwrap_or(&[]));
    view.extend_from_slice(c);
    view.extend_from_slice(values.get(gap.end..gap.end + right).unwrap_or(&[]));
    if view.len() < 3 {
        return false;
    }
    exact_line_stretches(&view, view.len().min(reach), cfg.line_rel_tol)
        .iter()
        .any(|s| s.start < left + c.len() && s.end > left)
}

/// (neighbor reference = the target's curve) + on one well and log.
/// `pred` has the grid length. Writes `Model` at surviving gap rows and fills `model_guess`.
///
/// Predictions and the neighbor curve are used on coverage rows only. Only open
/// gap rows (`Unfilled`, NaN) are written; rejected mapped candidates at those
/// rows go to `model_guess`. A channel without open gap rows: no rows, no report.
/// Report: one `policy` row, one `write` row if anything was written, one
/// `reject` row per reason (gap rows only). NaN rule: skip (NaN = no prediction).
/// Errors: `Shape` if `pred`, the neighbor curve, the log or its model guess
/// does not have the grid length.
pub fn apply_model_write(
    state: &mut WellFillState,
    log: LogKind,
    pred: &[f32],
    nbr: Option<&NeighborCurve>,
    well_name: &str,
    cfg: &FillConfig,
) -> SxResult<Vec<FillReportRow>> {
    let grid = state.logs.grid;
    let n = grid.len;
    if pred.len() != n {
        return Err(SxError::Shape(format!(
            "model write: well {}: {} predictions, grid has {n}",
            state.logs.id.0,
            pred.len()
        )));
    }
    if let Some(c) = nbr {
        if c.value.len() != n {
            return Err(SxError::Shape(format!(
                "model write: well {}: neighbor curve has {} rows, grid has {n}",
                state.logs.id.0,
                c.value.len()
            )));
        }
    }
    let (values, prov) = curve(&state.logs, log)?;
    let open: Vec<bool> = values
        .iter()
        .zip(prov)
        .map(|(&v, &p)| is_open(v, p))
        .collect();
    if !open.contains(&true) {
        return Ok(Vec::new());
    }
    let coverage = state.coverage;
    let mask = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .enumerate()
            .map(|(i, &x)| {
                if in_coverage(grid, coverage, i) {
                    x
                } else {
                    f32::NAN
                }
            })
            .collect()
    };
    let pred_cov = mask(pred);
    let nbr_cov = nbr.map(|c| mask(&c.value));
    let (mapped, tp) = fill_range_map(&pred_cov, nbr_cov.as_deref(), log, cfg)?;
    let (n_meas, sd) = measured_stats(values, prov);
    let sigma = if n_meas >= 2 { sd } else { 0.0 };
    let mut rej = reject_candidates(&mapped, sigma, nbr_cov.as_deref(), log, cfg)?;
    let mut cand: Vec<f32> = mapped
        .iter()
        .zip(&open)
        .zip(&rej)
        .map(|((&m, &o), r)| if o && r.is_none() { m } else { f32::NAN })
        .collect();
    let extra = written_line_rows(values, prov, &cand, cfg);
    for ((r, c), &x) in rej.iter_mut().zip(cand.iter_mut()).zip(&extra) {
        if x {
            *r = Some(RejectReason::ExactLine);
            *c = f32::NAN;
        }
    }
    let predicted: Vec<usize> = pred_cov
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, _)| i)
        .collect();

    let mut written = Vec::new();
    let mut rejected: BTreeMap<RejectReason, Vec<usize>> = BTreeMap::new();
    {
        let WellFillState {
            logs, model_guess, ..
        } = &mut *state;
        let guess = model_guess.entry(log).or_insert_with(|| vec![f32::NAN; n]);
        if guess.len() != n {
            return Err(SxError::Shape(format!(
                "model write: well {}: model guess has {} rows, grid has {n}",
                logs.id.0,
                guess.len()
            )));
        }
        let (values, prov) = curve_mut(logs, log)?;
        let rows = values
            .iter_mut()
            .zip(prov.iter_mut())
            .zip(guess.iter_mut())
            .zip(open.iter().zip(mapped.iter().zip(cand.iter().zip(&rej))));
        for (i, (((v, p), g), (&o, (&m, (&c, r))))) in rows.enumerate() {
            if !o || !m.is_finite() {
                continue;
            }
            if c.is_finite() {
                *v = c;
                *p = Provenance::Model.code();
                written.push(i);
            } else {
                *g = m;
                if let Some(r) = r {
                    rejected.entry(*r).or_default().push(i);
                }
            }
        }
    }

    let mut report = Vec::new();
    let mut policy = FillReportRow::over_rows(
        ReportKind::Policy,
        state,
        well_name,
        log,
        FillStep::Model,
        &predicted,
    );
    policy.range_gain = tp.gain;
    policy.range_offset = tp.offset;
    policy.reason = tp.reason.name().to_string();
    report.push(policy);
    if !written.is_empty() {
        let mut w = FillReportRow::over_rows(
            ReportKind::Write,
            state,
            well_name,
            log,
            FillStep::Model,
            &written,
        );
        w.code = Provenance::Model.code();
        report.push(w);
    }
    let n_rejected: usize = rejected.values().map(Vec::len).sum();
    for (reason, rows) in &rejected {
        let mut r = FillReportRow::over_rows(
            ReportKind::Reject,
            state,
            well_name,
            log,
            FillStep::Model,
            rows,
        );
        r.reason = reason.name().to_string();
        report.push(r);
    }
    debug!(
        well = state.logs.id.0,
        log = log.name(),
        written = written.len(),
        rejected = n_rejected,
        range = tp.reason.name(),
        "model write"
    );
    Ok(report)
}

fn finite_f64(v: &[f32]) -> Vec<f64> {
    v.iter()
        .filter(|x| x.is_finite())
        .map(|&x| f64::from(x))
        .collect()
}

fn two_values(v: Vec<f64>) -> SxResult<(f64, f64)> {
    match v.as_slice() {
        [a, b] => Ok((*a, *b)),
        _ => Err(SxError::Numerical(
            "range map: expected two percentiles".into(),
        )),
    }
}

/// Sample std (`N − 1`); 0 for fewer than two values.
fn sample_std(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}
