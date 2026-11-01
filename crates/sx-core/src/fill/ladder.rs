// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Copy steps and shaped gap fill: scale, anchors, hole fill,
//! tilt, guards, cross-log fit, centered mean.
//!
//! The donor helpers of `neighbors` ([`eligible_donors`], [`sample_donor`],
//! [`donor_score`]) run with the project `[neighbors]` settings from
//! `FillInputs::neighbors_cfg`.
//! `CrossLogFit` solves the normal equations by Cholesky in
//! `f64`; a pivot below `1e-12 ×` the largest diagonal entry is singular.
//! Measured values of other wells (`inputs.nbr.logs`) are already in working
//! units (RESD log10, from logs_norm) and are read as given.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use tracing::debug;

use crate::config::{FillConfig, NeighborsConfig};
use crate::error::{SxError, SxResult};
use crate::fill::rules::{
    exact_line_stretches, gap_joins_line, is_quiet, reject_candidates, written_line_rows,
    RejectReason,
};
use crate::fill::{
    curve, curve_mut, in_coverage, is_open, measured_stats, pop_std, true_runs, FillInputs,
    FillReportRow, FillStep, ReportKind, WellFillState,
};
use crate::neighbors::{donor_score, eligible_donors, sample_donor, NeighborCurve};
use crate::types::{GridRange, LogKind, Provenance, WellId};
use crate::units::working_bounds;
use crate::util::median;

/// Fleet fallback scale for `log`: median per-well measured std over wells
/// with at least `scale_min_measured` measured values; `scale_fallback` if none.
///
/// Reads `inputs.nbr.logs` (`Measured`, finite, working units as given); std is
/// the population std; median of an even count = mean of the two middle values.
/// Unit: the log's working unit. NaN rule: skip. Errors: none.
/// Complexity: `O(wells × grid)`.
pub fn fleet_scale(inputs: &FillInputs<'_>, log: LogKind, cfg: &FillConfig) -> f64 {
    let mut stds = Vec::new();
    for logs in inputs.nbr.logs.values() {
        let (Some(v), Some(p)) = (logs.curves.get(&log), logs.provenance.get(&log)) else {
            continue;
        };
        let m: Vec<f64> = v
            .iter()
            .zip(p)
            .filter(|(_, &c)| c == Provenance::Measured.code())
            .map(|(&x, _)| f64::from(x))
            .filter(|x| x.is_finite())
            .collect();
        if m.len() >= cfg.ladder.scale_min_measured {
            stds.push(pop_std(&m));
        }
    }
    if stds.is_empty() {
        cfg.ladder.scale_fallback
    } else {
        median(&stds)
    }
}

/// On one well and log: NeighborWeighted, RealDonor, NeighborWeak, in order.
///
/// Each step sees what the previous one wrote and writes only open gap rows
/// (`Unfilled`, NaN). Candidates pass the write checks with no neighbor reference and the
/// well's `σ_meas`. Report: `write` rows per (step, interval), `reject` rows per
/// (step, reason). No open gap row: empty report. NaN rule: skip.
/// Errors: `Shape` if the state, neighbor curve or coordinate does not have the
/// grid length; `NotFound` / `Shape` / `InvalidInput` from the donor helpers
/// (e.g. the well missing from `inputs.nbr.headers` or `geometry`).
pub fn run_copy_steps(
    inputs: &FillInputs<'_>,
    state: &mut WellFillState,
    log: LogKind,
    cfg: &FillConfig,
) -> SxResult<Vec<FillReportRow>> {
    let ncfg = inputs.neighbors_cfg;
    let ctx = Ctx::new(inputs, state, log, ncfg)?;
    let (values, prov) = curve(&state.logs, log)?;
    if !values.iter().zip(prov).any(|(&v, &p)| is_open(v, p)) {
        return Ok(Vec::new());
    }
    let (n_meas, sd) = measured_stats(values, prov);
    let sigma = if n_meas >= 2 { sd } else { 0.0 };
    let mut report = Vec::new();

    let cand = open_candidates(state, log, |i| {
        ctx.nbr
            .is_some_and(|c| c.trusted.get(i).copied().unwrap_or(false))
            .then(|| ctx.nbr_value(i))
    })?;
    let step = FillStep::NeighborWeighted;
    let res = copy_write(state, log, &cand, sigma, step, cfg)?;
    push_writes(&mut report, state, &ctx, log, step, &res.written, None);
    push_rejects(&mut report, state, &ctx, log, step, &res.rejected);
    let n_weighted = res.written.len();

    let n_donor = real_donor(inputs, state, log, &ctx, sigma, ncfg, cfg, &mut report)?;

    let cand = open_candidates(state, log, |i| Some(ctx.nbr_value(i)))?;
    let step = FillStep::NeighborWeak;
    let res = copy_write(state, log, &cand, sigma, step, cfg)?;
    push_writes(&mut report, state, &ctx, log, step, &res.written, None);
    push_rejects(&mut report, state, &ctx, log, step, &res.rejected);

    debug!(
        well = ctx.id.0,
        log = log.name(),
        neighbor_weighted = n_weighted,
        real_donor = n_donor,
        neighbor_weak = res.written.len(),
        "copy steps"
    );
    Ok(report)
}

/// On one well and log. Gaps left at the end stay `Unfilled` and are reported.
///
/// Gaps (maximal runs of open gap rows) are processed shallow to deep; a
/// written gap is visible to later gaps. Steps per gap: NeighborShape,
/// CrossLogFit, DonorShape, OwnTexture, SmoothedModel; first success wins and
/// the whole gap is written. `fleet_scale` is used when the well has fewer
/// than `scale_min_measured` measured values. Report: one `write` or one
/// `unfilled` row per gap. NaN rule: skip. Errors as [`run_copy_steps`].
/// Complexity: about `O(gaps × (grid + donors × donor grid))`.
pub fn run_shaped_fill(
    inputs: &FillInputs<'_>,
    state: &mut WellFillState,
    log: LogKind,
    fleet_scale: f64,
    cfg: &FillConfig,
) -> SxResult<Vec<FillReportRow>> {
    let ncfg = inputs.neighbors_cfg;
    let ctx = Ctx::new(inputs, state, log, ncfg)?;
    let (values, prov) = curve(&state.logs, log)?;
    let open: Vec<bool> = values
        .iter()
        .zip(prov)
        .map(|(&v, &p)| is_open(v, p))
        .collect();
    let gaps = true_runs(&open);
    if gaps.is_empty() {
        return Ok(Vec::new());
    }
    let (n_meas, sd) = measured_stats(values, prov);
    let scale = if n_meas >= cfg.ladder.scale_min_measured {
        sd
    } else {
        fleet_scale
    };
    let scale = if scale.is_finite() {
        scale
    } else {
        cfg.ladder.scale_fallback
    }
    .max(cfg.scale_floor);

    let mut donors: Option<Vec<(WellId, f64)>> = None;
    let mut report = Vec::new();
    let mut counts: BTreeMap<FillStep, usize> = BTreeMap::new();
    for gap in gaps {
        let chosen = fill_gap(
            inputs,
            state,
            log,
            &ctx,
            gap.clone(),
            scale,
            &mut donors,
            ncfg,
            cfg,
        )?;
        let rows: Vec<usize> = gap.clone().collect();
        let interval = ctx.interval(gap.start);
        let Some(ch) = chosen else {
            let mut r = FillReportRow::over_rows(
                ReportKind::Unfilled,
                state,
                &ctx.name,
                log,
                FillStep::None,
                &rows,
            );
            r.code = Provenance::Unfilled.code();
            r.interval = interval;
            report.push(r);
            *counts.entry(FillStep::None).or_default() += rows.len();
            continue;
        };
        let code = ch.step.provenance().code();
        {
            let (values, prov) = curve_mut(&mut state.logs, log)?;
            let slots = values
                .get_mut(gap.clone())
                .into_iter()
                .flatten()
                .zip(prov.get_mut(gap.clone()).into_iter().flatten());
            for ((v, p), &x) in slots.zip(&ch.values) {
                if is_open(*v, *p) {
                    *v = x;
                    *p = code;
                }
            }
        }
        let mut r =
            FillReportRow::over_rows(ReportKind::Write, state, &ctx.name, log, ch.step, &rows);
        r.code = code;
        r.interval = interval;
        r.donor_well_id = ch.donor_well_id;
        r.score = ch.score;
        report.push(r);
        *counts.entry(ch.step).or_default() += rows.len();
    }
    debug!(
        well = ctx.id.0,
        log = log.name(),
        rows = ?counts,
        "shaped gap fill"
    );
    Ok(report)
}

///
/// Every interior run of non-finite values of at most `max_hole_rows` rows with
/// finite values on both sides is filled linearly (in `f64`, by row). Longer
/// runs and runs touching an end stay as they are. Same length as `v`.
/// NaN rule: fills short interior NaN runs only. Errors: none.
pub fn hole_fill(v: &[f32], max_hole_rows: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    let holes: Vec<bool> = v.iter().map(|x| !x.is_finite()).collect();
    for h in true_runs(&holes) {
        if h.start == 0 || h.end >= v.len() || h.len() > max_hole_rows {
            continue;
        }
        let (Some(&a), Some(&b)) = (v.get(h.start - 1), v.get(h.end)) else {
            continue;
        };
        let (a, b) = (f64::from(a), f64::from(b));
        let span = (h.len() + 1) as f64;
        for (t, slot) in out.get_mut(h).into_iter().flatten().enumerate() {
            *slot = (a + (b - a) * (t + 1) as f64 / span) as f32;
        }
    }
    out
}

/// Centered mean over rows j-10 ..= j+9 for window 20 (generally j - w/2 ..= j + w/2 - 1), finite values only.
///
/// Odd `window`: rows `j − w/2 ..= j + w/2` (same rule as
/// `j − w/2 ..= j + w − w/2 − 1`); `window = 0` acts as 1. The window is
/// clipped to the series. Mean in `f64`; NaN where no finite value.
/// NaN rule: skip. Errors: none. Complexity: `O(n × window)`.
pub fn centered_mean(v: &[f32], window: usize) -> Vec<f32> {
    centered_mean_f64(v, window)
        .into_iter()
        .map(|m| m as f32)
        .collect()
}

fn centered_mean_f64(v: &[f32], window: usize) -> Vec<f64> {
    let w = window.max(1);
    let before = w / 2;
    let after = w - before - 1;
    (0..v.len())
        .map(|j| {
            let lo = j.saturating_sub(before);
            let hi = (j + after + 1).min(v.len());
            mean_finite(v.get(lo..hi).unwrap_or(&[])).unwrap_or(f64::NAN)
        })
        .collect()
}

/// Per-call view of the target well.
struct Ctx<'a> {
    id: WellId,
    name: String,
    nbr: Option<&'a NeighborCurve>,
    /// Mapping coordinate, only if it has a finite value.
    coord: Option<&'a [f64]>,
    eps: f64,
}

impl<'a> Ctx<'a> {
    fn new(
        inputs: &FillInputs<'a>,
        state: &WellFillState,
        log: LogKind,
        ncfg: &NeighborsConfig,
    ) -> SxResult<Self> {
        let id = state.logs.id;
        let n = state.logs.grid.len;
        let neighbors = inputs.neighbors;
        let nbr = neighbors.get(&id).and_then(|w| w.curves.get(&log));
        if let Some(c) = nbr {
            if c.value.len() != n || c.trusted.len() != n {
                return Err(SxError::Shape(format!(
                    "fill: well {}: {} neighbor curve has {} rows, grid has {n}",
                    id.0,
                    log.name(),
                    c.value.len()
                )));
            }
        }
        let coords = inputs.nbr.coord;
        let coord = coords.get(&id).map(Vec::as_slice);
        if let Some(u) = coord {
            if u.len() != n {
                return Err(SxError::Shape(format!(
                    "fill: well {}: coordinate has {} rows, grid has {n}",
                    id.0,
                    u.len()
                )));
            }
        }
        Ok(Self {
            id,
            name: inputs
                .nbr
                .headers
                .get(&id)
                .map(|h| h.name.clone())
                .unwrap_or_default(),
            nbr,
            coord: coord.filter(|u| u.iter().any(|x| x.is_finite())),
            eps: ncfg.interval_eps,
        })
    }

    /// `floor(u + eps)` at row `i`; `None` without a finite coordinate.
    fn interval_key(&self, i: usize) -> Option<i64> {
        let u = *self.coord?.get(i)?;
        u.is_finite().then(|| (u + self.eps).floor() as i64)
    }

    /// Report interval id at row `i`; −1 = none.
    fn interval(&self, i: usize) -> i32 {
        self.interval_key(i)
            .and_then(|k| i32::try_from(k).ok())
            .unwrap_or(-1)
    }

    fn nbr_value(&self, i: usize) -> f32 {
        self.nbr
            .and_then(|c| c.value.get(i))
            .copied()
            .unwrap_or(f32::NAN)
    }
}

/// Candidates on the grid: `pick(i)` at open gap rows, NaN elsewhere.
fn open_candidates(
    state: &WellFillState,
    log: LogKind,
    pick: impl Fn(usize) -> Option<f32>,
) -> SxResult<Vec<f32>> {
    let (values, prov) = curve(&state.logs, log)?;
    Ok(values
        .iter()
        .zip(prov)
        .enumerate()
        .map(|(i, (&v, &p))| {
            if is_open(v, p) {
                pick(i).unwrap_or(f32::NAN)
            } else {
                f32::NAN
            }
        })
        .collect())
}

#[derive(Default)]
struct StepRows {
    written: Vec<usize>,
    rejected: BTreeMap<RejectReason, Vec<usize>>,
}

/// Write checks (no reference) and the written-line check on `cand`, then write the
/// survivors at open rows with `step`'s code.
fn copy_write(
    state: &mut WellFillState,
    log: LogKind,
    cand: &[f32],
    sigma: f64,
    step: FillStep,
    cfg: &FillConfig,
) -> SxResult<StepRows> {
    let mut out = StepRows::default();
    if !cand.iter().any(|v| !v.is_nan()) {
        return Ok(out);
    }
    let rej = reject_candidates(cand, sigma, None, log, cfg)?;
    let mut surv: Vec<f32> = cand
        .iter()
        .zip(&rej)
        .map(|(&c, r)| if r.is_none() { c } else { f32::NAN })
        .collect();
    let (values, prov) = curve(&state.logs, log)?;
    let extra = written_line_rows(values, prov, &surv, cfg);
    for (i, (s, (r, &x))) in surv.iter_mut().zip(rej.iter().zip(&extra)).enumerate() {
        let reason = r.or(x.then_some(RejectReason::ExactLine));
        if let Some(reason) = reason {
            out.rejected.entry(reason).or_default().push(i);
            *s = f32::NAN;
        }
    }
    let code = step.provenance().code();
    let (values, prov) = curve_mut(&mut state.logs, log)?;
    for (i, ((v, p), &s)) in values
        .iter_mut()
        .zip(prov.iter_mut())
        .zip(&surv)
        .enumerate()
    {
        if s.is_finite() && is_open(*v, *p) {
            *v = s;
            *p = code;
            out.written.push(i);
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn real_donor(
    inputs: &FillInputs<'_>,
    state: &mut WellFillState,
    log: LogKind,
    ctx: &Ctx<'_>,
    sigma: f64,
    ncfg: &NeighborsConfig,
    cfg: &FillConfig,
    report: &mut Vec<FillReportRow>,
) -> SxResult<usize> {
    let (values, prov) = curve(&state.logs, log)?;
    let mut groups: BTreeMap<Option<i64>, Vec<usize>> = BTreeMap::new();
    for (i, (&v, &p)) in values.iter().zip(prov).enumerate() {
        if !is_open(v, p) {
            continue;
        }
        if ctx.coord.is_none() {
            groups.entry(None).or_default().push(i);
        } else if let Some(k) = ctx.interval_key(i) {
            groups.entry(Some(k)).or_default().push(i);
        }
    }
    if groups.is_empty() {
        return Ok(0);
    }
    let pool = eligible_donors(
        &inputs.nbr,
        ctx.id,
        log,
        Some(cfg.ladder.donor_max_dist_m),
        Some(cfg.ladder.real_donor_max_n),
        ncfg,
    )?;
    if pool.is_empty() {
        return Ok(0);
    }
    let n = state.logs.grid.len;
    let step = FillStep::RealDonor;
    let mut rejected: BTreeMap<RejectReason, Vec<usize>> = BTreeMap::new();
    let mut total = 0;
    for (key, rows) in groups {
        let interval = match key {
            Some(k) => match u16::try_from(k) {
                Ok(i) => Some(i),
                Err(_) => continue,
            },
            None => None,
        };
        let mut best: Option<(WellId, f64)> = None;
        for &(d, _) in &pool {
            let s = donor_score(&inputs.nbr, ctx.id, d, log, interval, ncfg)?;
            if s > 0.0 && best.is_none_or(|(_, b)| s > b) {
                best = Some((d, s));
            }
        }
        let Some((donor, score)) = best else {
            continue;
        };
        let mut cand = vec![f32::NAN; n];
        for run in consecutive_runs(&rows) {
            let vals = sample_donor(&inputs.nbr, ctx.id, donor, log, run.clone(), ncfg)?;
            for (slot, v) in cand.get_mut(run).into_iter().flatten().zip(vals) {
                *slot = v;
            }
        }
        let res = copy_write(state, log, &cand, sigma, step, cfg)?;
        push_writes(
            report,
            state,
            ctx,
            log,
            step,
            &res.written,
            Some((donor, score)),
        );
        for (reason, r) in res.rejected {
            rejected.entry(reason).or_default().extend(r);
        }
        total += res.written.len();
    }
    push_rejects(report, state, ctx, log, step, &rejected);
    Ok(total)
}

/// Split ascending rows into runs of consecutive rows.
fn consecutive_runs(rows: &[usize]) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for &i in rows {
        match out.last_mut() {
            Some(r) if r.end == i => r.end = i + 1,
            _ => out.push(i..i + 1),
        }
    }
    out
}

/// One `write` row per interval of `rows`.
fn push_writes(
    report: &mut Vec<FillReportRow>,
    state: &WellFillState,
    ctx: &Ctx<'_>,
    log: LogKind,
    step: FillStep,
    rows: &[usize],
    donor: Option<(WellId, f64)>,
) {
    let mut by_interval: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for &i in rows {
        by_interval.entry(ctx.interval(i)).or_default().push(i);
    }
    for (interval, rs) in by_interval {
        let mut r = FillReportRow::over_rows(ReportKind::Write, state, &ctx.name, log, step, &rs);
        r.code = step.provenance().code();
        r.interval = interval;
        if let Some((d, s)) = donor {
            r.donor_well_id = d.0;
            r.score = s;
        }
        report.push(r);
    }
}

/// One `reject` row per reason.
fn push_rejects(
    report: &mut Vec<FillReportRow>,
    state: &WellFillState,
    ctx: &Ctx<'_>,
    log: LogKind,
    step: FillStep,
    rejected: &BTreeMap<RejectReason, Vec<usize>>,
) {
    for (reason, rows) in rejected {
        let mut r = FillReportRow::over_rows(ReportKind::Reject, state, &ctx.name, log, step, rows);
        r.reason = reason.name().to_string();
        report.push(r);
    }
}

/// A gap with its anchors.
struct GapView {
    gap: Range<usize>,
    above: Range<usize>,
    below: Range<usize>,
    anchor_above: Option<f64>,
    anchor_below: Option<f64>,
}

impl GapView {
    fn new(
        values: &[f32],
        grid: GridRange,
        coverage: Option<crate::neighbors::Coverage>,
        gap: Range<usize>,
        anchor_rows: usize,
    ) -> Self {
        let ok = |i: usize| {
            in_coverage(grid, coverage, i) && values.get(i).is_some_and(|v| v.is_finite())
        };
        let mut a0 = gap.start;
        while a0 > 0 && gap.start - a0 < anchor_rows && ok(a0 - 1) {
            a0 -= 1;
        }
        let mut b1 = gap.end;
        while b1 - gap.end < anchor_rows && ok(b1) {
            b1 += 1;
        }
        let above = a0..gap.start;
        let below = gap.end..b1;
        Self {
            anchor_above: mean_finite(values.get(above.clone()).unwrap_or(&[])),
            anchor_below: mean_finite(values.get(below.clone()).unwrap_or(&[])),
            gap,
            above,
            below,
        }
    }

    fn ext(&self) -> Range<usize> {
        self.above.start..self.below.end
    }
}

/// The chosen shaped candidate of one gap.
struct Chosen {
    step: FillStep,
    values: Vec<f32>,
    donor_well_id: i32,
    score: f64,
}

impl Chosen {
    fn new(step: FillStep, values: Vec<f32>) -> Self {
        Self {
            step,
            values,
            donor_well_id: -1,
            score: f64::NAN,
        }
    }
}

/// Tilt, guard and written-line check of one gap.
struct Shaper<'s> {
    values: &'s [f32],
    prov: &'s [u8],
    view: &'s GapView,
    log: LogKind,
    scale: f64,
    nbr_gap: &'s [f32],
    cfg: &'s FillConfig,
}

impl Shaper<'_> {
    /// "Try": fully finite, then tilt (`None` = untilted; `Some((ref_above,
    /// ref_below))`, a `None` ref = edge ref), then guard, then the written-line check.
    fn attempt(
        &self,
        mut c: Vec<f64>,
        tilt: Option<(Option<f64>, Option<f64>)>,
        allow_quiet: bool,
        exact_only: bool,
    ) -> Option<Vec<f32>> {
        if c.is_empty() || c.len() != self.view.gap.len() || c.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if let Some((ra, rb)) = tilt {
            if !self.tilt(&mut c, ra, rb) {
                return None;
            }
        }
        let out = self.guard(&c, allow_quiet, exact_only)?;
        if gap_joins_line(
            self.values,
            self.prov,
            self.view.gap.clone(),
            &out,
            self.cfg,
        ) {
            return None;
        }
        Some(out)
    }

    /// False if an offset exceeds `tilt_max_scale × scale`.
    fn tilt(&self, c: &mut [f64], ref_above: Option<f64>, ref_below: Option<f64>) -> bool {
        let (Some(&first), Some(&last)) = (c.first(), c.last()) else {
            return false;
        };
        let d_lo = self
            .view
            .anchor_above
            .map(|a| a - ref_above.unwrap_or(first));
        let d_hi = self
            .view
            .anchor_below
            .map(|b| b - ref_below.unwrap_or(last));
        let lim = self.cfg.ladder.tilt_max_scale * self.scale;
        if d_lo
            .into_iter()
            .chain(d_hi)
            .any(|d| d.is_nan() || d.abs() > lim)
        {
            return false;
        }
        let n1 = (c.len() + 1) as f64;
        match (d_lo, d_hi) {
            (Some(lo), Some(hi)) => {
                for (j, v) in c.iter_mut().enumerate() {
                    *v += lo + (hi - lo) * (j + 1) as f64 / n1;
                }
            }
            (Some(d), None) | (None, Some(d)) => {
                for v in c.iter_mut() {
                    *v += d;
                }
            }
            (None, None) => {}
        }
        true
    }

    /// `guard(c, allow_quiet, exact_only)`; the clipped candidate if it passes.
    fn guard(&self, c: &[f64], allow_quiet: bool, exact_only: bool) -> Option<Vec<f32>> {
        let n = c.len();
        if c.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let b = working_bounds(self.log);
        let n_out = c.iter().filter(|&&v| !b.contains(v)).count();
        if n_out as f64 > self.cfg.ladder.physical_max_out_frac * n as f64 {
            return None;
        }
        let w: Vec<f32> = c.iter().map(|&v| v.max(b.lo).min(b.hi) as f32).collect();
        let w64: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
        let mn = w64.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = w64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if n >= self.cfg.ladder.exact_flat_min_rows && mx - mn <= self.cfg.exact_flat_tol {
            return None;
        }
        let line_rows = n.min(self.cfg.line_min_rows.max(3));
        if n >= 3 && !exact_line_stretches(&w, line_rows, self.cfg.line_rel_tol).is_empty() {
            return None;
        }
        if !exact_only
            && !allow_quiet
            && pop_std(&w64) < self.cfg.ladder.guard_quiet_frac * self.scale
            && !is_quiet(Some(self.nbr_gap), n, self.scale, self.cfg)
        {
            return None;
        }
        Some(w)
    }
}

/// On one gap; `None` = nothing passed.
#[allow(clippy::too_many_arguments)]
fn fill_gap(
    inputs: &FillInputs<'_>,
    state: &WellFillState,
    log: LogKind,
    ctx: &Ctx<'_>,
    gap: Range<usize>,
    scale: f64,
    donors: &mut Option<Vec<(WellId, f64)>>,
    ncfg: &NeighborsConfig,
    cfg: &FillConfig,
) -> SxResult<Option<Chosen>> {
    let (values, prov) = curve(&state.logs, log)?;
    let grid = state.logs.grid;
    let n = gap.len();
    let hole = cfg.ladder.max_hole_rows;
    let view = GapView::new(
        values,
        grid,
        state.coverage,
        gap.clone(),
        cfg.ladder.anchor_rows,
    );
    let nbr_full: Option<&[f32]> = ctx.nbr.map(|c| c.value.as_slice());
    let over_gap = |src: Option<&[f32]>| -> Vec<f32> {
        src.and_then(|s| s.get(gap.clone()))
            .map_or_else(|| vec![f32::NAN; n], |s| hole_fill(s, hole))
    };
    let nbr_gap = over_gap(nbr_full);
    let guess_gap = over_gap(state.model_guess.get(&log).map(Vec::as_slice));
    let guess_ok = guess_gap.iter().all(|v| v.is_finite());
    let sh = Shaper {
        values,
        prov,
        view: &view,
        log,
        scale,
        nbr_gap: &nbr_gap,
        cfg,
    };
    let refs_from = |src: &[f32]| {
        (
            mean_finite(src.get(view.above.clone()).unwrap_or(&[])),
            mean_finite(src.get(view.below.clone()).unwrap_or(&[])),
        )
    };

    if let Some(full) = nbr_full {
        if let Some(v) = sh.attempt(to_f64(&nbr_gap), Some(refs_from(full)), true, false) {
            return Ok(Some(Chosen::new(FillStep::NeighborShape, v)));
        }
    }

    if let Some((ext_pred, r2)) = cross_log_fit(state, log, ctx, &view, cfg)? {
        let off = view.above.len();
        let part = |r: Range<usize>| ext_pred.get(r).unwrap_or(&[]).to_vec();
        let c = part(off..off + n);
        let ra = mean_f64(&part(0..off));
        let rb = mean_f64(&part(off + n..ext_pred.len()));
        if let Some(v) = sh.attempt(c, Some((ra, rb)), false, false) {
            let mut ch = Chosen::new(FillStep::CrossLogFit, v);
            ch.score = r2;
            return Ok(Some(ch));
        }
    }

    if donors.is_none() {
        *donors = Some(eligible_donors(
            &inputs.nbr,
            ctx.id,
            log,
            Some(cfg.ladder.donor_max_dist_m),
            Some(cfg.ladder.donor_max_n),
            ncfg,
        )?);
    }
    for &(d, _) in donors.as_deref().unwrap_or(&[]) {
        let s = sample_donor(&inputs.nbr, ctx.id, d, log, view.ext(), ncfg)?;
        let off = view.above.len();
        let d_gap = hole_fill(s.get(off..off + n).unwrap_or(&[]), hole);
        let ra = mean_finite(s.get(0..off).unwrap_or(&[]));
        let rb = mean_finite(s.get(off + n..s.len()).unwrap_or(&[]));
        if let Some(v) = sh.attempt(to_f64(&d_gap), Some((ra, rb)), true, false) {
            let mut ch = Chosen::new(FillStep::DonorShape, v);
            ch.donor_well_id = d.0;
            return Ok(Some(ch));
        }
    }

    if guess_ok {
        let window = cfg.ladder.texture_window;
        if let Some(c) = own_texture(values, prov, grid, gap.clone(), &guess_gap, window) {
            if let Some(v) = sh.attempt(c, Some((None, None)), false, false) {
                return Ok(Some(Chosen::new(FillStep::OwnTexture, v)));
            }
        }

        let c = to_f64(&guess_gap);
        let tol = cfg.ladder.anchor_tol_scale * scale;
        let near = |a: Option<f64>, end: Option<&f64>| {
            a.is_none_or(|a| end.is_some_and(|e| (a - e).abs() <= tol))
        };
        let untilted = near(view.anchor_above, c.first()) && near(view.anchor_below, c.last());
        let tilt = (!untilted).then_some((None, None));
        if let Some(v) = sh.attempt(c, tilt, true, true) {
            return Ok(Some(Chosen::new(FillStep::SmoothedModel, v)));
        }
    }
    Ok(None)
}

/// Least-squares fit with intercept on standardized predictors.
pub(crate) struct LinearFit {
    mean: Vec<f64>,
    std: Vec<f64>,
    beta: Vec<f64>,
    /// In-sample `R²`.
    pub(crate) r2: f64,
}

impl LinearFit {
    /// `cols[j][r]` = predictor `j` at fit row `r`. `None` if empty, ragged,
    /// singular, or `SS_tot <= 0`. Std below `min_std` counts as 1.
    pub(crate) fn fit(cols: &[Vec<f64>], y: &[f64], min_std: f64) -> Option<Self> {
        let n = y.len();
        if n == 0 || cols.iter().any(|c| c.len() != n) {
            return None;
        }
        let m = cols.len() + 1;
        let mut mean = Vec::with_capacity(cols.len());
        let mut std = Vec::with_capacity(cols.len());
        for c in cols {
            let mu = c.iter().sum::<f64>() / n as f64;
            let sd = pop_std(c);
            mean.push(mu);
            std.push(if sd.is_nan() || sd < min_std { 1.0 } else { sd });
        }
        let mut a = vec![0.0; m * m];
        let mut b = vec![0.0; m];
        let mut x = vec![0.0; m];
        for (r, &yr) in y.iter().enumerate() {
            x[0] = 1.0;
            for ((xj, c), (mu, sd)) in x.iter_mut().skip(1).zip(cols).zip(mean.iter().zip(&std)) {
                *xj = (c.get(r).copied().unwrap_or(f64::NAN) - mu) / sd;
            }
            for (bi, &xi) in b.iter_mut().zip(&x) {
                *bi += xi * yr;
            }
            for (row, &xi) in a.chunks_mut(m).zip(&x) {
                for (aij, &xj) in row.iter_mut().zip(&x) {
                    *aij += xi * xj;
                }
            }
        }
        let beta = cholesky_solve(&a, &b, m)?;
        let fit = Self {
            mean,
            std,
            beta,
            r2: f64::NAN,
        };
        let ybar = y.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
        if ss_tot.is_nan() || ss_tot <= 0.0 {
            return None;
        }
        let mut ss_res = 0.0;
        let mut row = vec![0.0; cols.len()];
        for (r, &yr) in y.iter().enumerate() {
            for (slot, c) in row.iter_mut().zip(cols) {
                *slot = c.get(r).copied().unwrap_or(f64::NAN);
            }
            let e = yr - fit.predict(&row);
            ss_res += e * e;
        }
        Some(Self {
            r2: 1.0 - ss_res / ss_tot,
            ..fit
        })
    }

    /// Prediction at raw predictor values `x` (one per column).
    pub(crate) fn predict(&self, x: &[f64]) -> f64 {
        let mut out = self.beta.first().copied().unwrap_or(f64::NAN);
        for (((&b, &xj), &mu), &sd) in self
            .beta
            .iter()
            .skip(1)
            .zip(x)
            .zip(&self.mean)
            .zip(&self.std)
        {
            out += b * (xj - mu) / sd;
        }
        out
    }
}

/// Solve `A x = b` (`A` symmetric, row-major `m × m`) by Cholesky; `None` if a
/// pivot is below `1e-12 ×` the largest diagonal entry.
fn cholesky_solve(a: &[f64], b: &[f64], m: usize) -> Option<Vec<f64>> {
    if a.len() != m * m || b.len() != m {
        return None;
    }
    let max_diag = (0..m).map(|i| a[i * m + i]).fold(0.0, f64::max);
    if max_diag.is_nan() || max_diag <= 0.0 {
        return None;
    }
    let tol = 1e-12 * max_diag;
    let mut l = vec![0.0; m * m];
    for j in 0..m {
        let mut d = a[j * m + j];
        for k in 0..j {
            d -= l[j * m + k] * l[j * m + k];
        }
        if d.is_nan() || d < tol {
            return None;
        }
        let ljj = d.sqrt();
        l[j * m + j] = ljj;
        for i in j + 1..m {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= l[i * m + k] * l[j * m + k];
            }
            l[i * m + j] = s / ljj;
        }
    }
    let mut z = vec![0.0; m];
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * m + k] * z[k];
        }
        z[i] = s / l[i * m + i];
    }
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = z[i];
        for k in i + 1..m {
            s -= l[k * m + i] * x[k];
        }
        x[i] = s / l[i * m + i];
    }
    Some(x)
}

/// Predictions over the extended window (NaN where a
/// predictor is not measured) and the in-sample `R²`; `None` = skip.
fn cross_log_fit(
    state: &WellFillState,
    log: LogKind,
    ctx: &Ctx<'_>,
    view: &GapView,
    cfg: &FillConfig,
) -> SxResult<Option<(Vec<f64>, f64)>> {
    let grid = state.logs.grid;
    let gap = view.gap.clone();
    let measured = |v: &[f32], p: &[u8], i: usize| {
        p.get(i) == Some(&Provenance::Measured.code()) && v.get(i).is_some_and(|x| x.is_finite())
    };
    let mut preds: Vec<(&[f32], &[u8])> = Vec::new();
    for other in LogKind::ALL {
        if other == log {
            continue;
        }
        let Ok((v, p)) = curve(&state.logs, other) else {
            continue;
        };
        if gap.clone().all(|i| measured(v, p, i)) {
            preds.push((v, p));
        }
    }
    if preds.is_empty() || gap.is_empty() {
        return Ok(None);
    }
    let gap_last = gap.end - 1;
    let (tv, tp) = curve(&state.logs, log)?;
    let gap_ids: BTreeSet<i64> = gap
        .clone()
        .filter_map(|i| ctx.interval_key(i))
        .filter(|&k| k >= 0)
        .collect();
    let w = cfg.ladder.crosslog_window_ft;
    let (lo, hi) = (grid.md_ft(gap.start) - w, grid.md_ft(gap_last) + w);
    let fit_rows: Vec<usize> = (0..grid.len)
        .filter(|&i| {
            measured(tv, tp, i)
                && preds.iter().all(|(v, p)| measured(v, p, i))
                && ({
                    let md = grid.md_ft(i);
                    lo <= md && md <= hi
                } || ctx.interval_key(i).is_some_and(|k| gap_ids.contains(&k)))
        })
        .collect();
    if fit_rows.len() < cfg.ladder.crosslog_min_rows {
        return Ok(None);
    }
    let value = |v: &[f32], i: usize| v.get(i).map_or(f64::NAN, |&x| f64::from(x));
    let cols: Vec<Vec<f64>> = preds
        .iter()
        .map(|(v, _)| fit_rows.iter().map(|&i| value(v, i)).collect())
        .collect();
    let y: Vec<f64> = fit_rows.iter().map(|&i| value(tv, i)).collect();
    let Some(fit) = LinearFit::fit(&cols, &y, cfg.ladder.crosslog_min_std) else {
        return Ok(None);
    };
    if fit.r2.is_nan() || fit.r2 < cfg.ladder.crosslog_min_r2 {
        return Ok(None);
    }
    let mut x = vec![0.0; preds.len()];
    let pred = view
        .ext()
        .map(|i| {
            if !preds.iter().all(|(v, p)| measured(v, p, i)) {
                return f64::NAN;
            }
            for (slot, (v, _)) in x.iter_mut().zip(&preds) {
                *slot = value(v, i);
            }
            fit.predict(&x)
        })
        .collect();
    Ok(Some((pred, fit.r2)))
}

/// Candidate (before the edge-ref tilt); `None` = no measured run.
fn own_texture(
    values: &[f32],
    prov: &[u8],
    grid: GridRange,
    gap: Range<usize>,
    guess_gap: &[f32],
    window: usize,
) -> Option<Vec<f64>> {
    let n = gap.len();
    let measured: Vec<bool> = values
        .iter()
        .zip(prov)
        .map(|(v, &p)| p == Provenance::Measured.code() && v.is_finite())
        .collect();
    let mut best: Option<(f64, bool, Range<usize>)> = None;
    for r in true_runs(&measured) {
        if r.len() < n || n == 0 {
            continue;
        }
        let (dist, above) = if r.end <= gap.start {
            (grid.md_ft(gap.start) - grid.md_ft(r.end - 1), true)
        } else if r.start >= gap.end {
            (grid.md_ft(r.start) - grid.md_ft(gap.end - 1), false)
        } else {
            continue;
        };
        let better = match &best {
            None => true,
            Some((bd, ba, _)) => dist < *bd || (dist == *bd && above && !*ba),
        };
        if better {
            best = Some((dist, above, r));
        }
    }
    let (_, above, run) = best?;
    let seg = if above {
        values.get(run.end - n..run.end)?
    } else {
        values.get(run.start..run.start + n)?
    };
    let cm = centered_mean_f64(seg, window);
    let mut texture: Vec<f64> = seg
        .iter()
        .zip(&cm)
        .map(|(&s, &m)| f64::from(s) - m)
        .collect();
    if !above {
        texture.reverse();
    }
    let level = centered_mean_f64(guess_gap, window);
    Some(level.iter().zip(&texture).map(|(l, t)| l + t).collect())
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Mean of the finite values; `None` if there are none.
fn mean_finite(v: &[f32]) -> Option<f64> {
    mean_f64(&to_f64(v))
}

fn mean_f64(v: &[f64]) -> Option<f64> {
    let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    (!f.is_empty()).then(|| f.iter().sum::<f64>() / f.len() as f64)
}
