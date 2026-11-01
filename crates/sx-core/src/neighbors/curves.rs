// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tiered search, validation, range map and the trusted
//! flag.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use tracing::debug;

use crate::config::NeighborsConfig;
use crate::error::{SxError, SxResult};
use crate::neighbors::mapping::{interval_of, sample_row, Measured, WellView};
use crate::neighbors::median::{aggregate, Contrib};
use crate::neighbors::search::Pool;
use crate::neighbors::weights::RowWeights;
use crate::neighbors::{Coverage, NeighborCurve, WellNeighborCurves};
use crate::types::{GridRange, LogKind, WellId, CODE_NONE, GRID_STEP_FT};
use crate::units::working_bounds;
use crate::util::percentiles;

impl NeighborCurve {
    /// Curve of `len` rows with no value anywhere.
    pub(crate) fn empty(len: usize) -> Self {
        Self {
            value: vec![f32::NAN; len],
            spread: vec![f32::NAN; len],
            n_donors: vec![0; len],
            dist_m: vec![f64::NAN; len],
            quality: vec![f32::NAN; len],
            tier: vec![CODE_NONE; len],
            trusted: vec![false; len],
        }
    }

    /// Clear row `i`.
    fn blank(&mut self, i: usize) {
        if let Some(v) = self.value.get_mut(i) {
            *v = f32::NAN;
        }
        if let Some(v) = self.spread.get_mut(i) {
            *v = f32::NAN;
        }
        if let Some(v) = self.n_donors.get_mut(i) {
            *v = 0;
        }
        if let Some(v) = self.dist_m.get_mut(i) {
            *v = f64::NAN;
        }
        if let Some(v) = self.quality.get_mut(i) {
            *v = f32::NAN;
        }
        if let Some(v) = self.tier.get_mut(i) {
            *v = CODE_NONE;
        }
        if let Some(v) = self.trusted.get_mut(i) {
            *v = false;
        }
    }
}

/// Stack, open rows and per-row contributors of one target and one log.
struct Stack<'p, 'a> {
    pool: &'p Pool<'a>,
    target: &'p WellView<'a>,
    log: LogKind,
    contrib: Vec<Vec<Contrib>>,
    open: Vec<usize>,
    stacked: BTreeSet<WellId>,
    curve: NeighborCurve,
    scratch: Vec<(f64, f64)>,
}

impl<'p, 'a> Stack<'p, 'a> {
    fn new(pool: &'p Pool<'a>, target: &'p WellView<'a>, log: LogKind) -> Self {
        let len = target.logs.grid.len;
        Self {
            pool,
            target,
            log,
            contrib: vec![Vec::new(); len],
            open: (0..len).collect(),
            stacked: BTreeSet::new(),
            curve: NeighborCurve::empty(len),
            scratch: Vec::new(),
        }
    }

    /// Add one donor; sample it at the open rows only.
    fn add(&mut self, donor: WellId, dist_m: f64) {
        let pool = self.pool;
        let cfg = pool.cfg;
        let (Some(d), Some(stats)) = (pool.wells.get(&donor), pool.stats.get(&(donor, self.log)))
        else {
            return;
        };
        let Some(measured) = Measured::new(d.logs, self.log) else {
            return;
        };
        self.stacked.insert(donor);
        let mut weights = RowWeights::new(dist_m, cfg);
        for &r in &self.open {
            let Some((value, key)) = sample_row(self.target, d, &measured, r, cfg) else {
                continue;
            };
            let (weight, quality) = weights.get(key, self.target, d, stats, pool.inputs.pairs, cfg);
            if value.is_finite() && weight > 0.0 {
                if let Some(list) = self.contrib.get_mut(r) {
                    list.push(Contrib {
                        value,
                        weight,
                        dist_m,
                        quality,
                    });
                }
            }
        }
    }

    fn add_all(&mut self, donors: &[(WellId, f64)]) {
        for &(id, d) in donors {
            self.add(id, d);
        }
    }

    /// Accept every open row with at least `min_n` (and at least 1) contributors.
    fn accept(&mut self, min_n: usize, tier: u8) {
        let min_n = min_n.max(1);
        let mut still = Vec::with_capacity(self.open.len());
        for &r in &self.open {
            let Some(list) = self.contrib.get_mut(r) else {
                continue;
            };
            if list.len() < min_n {
                still.push(r);
                continue;
            }
            let s = aggregate(list, &mut self.scratch);
            *list = Vec::new();
            let c = &mut self.curve;
            if let Some(v) = c.value.get_mut(r) {
                *v = s.value as f32;
            }
            if let Some(v) = c.spread.get_mut(r) {
                *v = s.spread as f32;
            }
            if let Some(v) = c.n_donors.get_mut(r) {
                *v = u32::try_from(s.n).unwrap_or(u32::MAX);
            }
            if let Some(v) = c.dist_m.get_mut(r) {
                *v = s.dist_m;
            }
            if let Some(v) = c.quality.get_mut(r) {
                *v = s.quality as f32;
            }
            if let Some(v) = c.tier.get_mut(r) {
                *v = tier;
            }
        }
        self.open = still;
    }

    fn is_done(&self) -> bool {
        self.open.is_empty()
    }
}

/// Tiers 1–5 for one log.
fn tiered_curve(
    pool: &Pool<'_>,
    target: &WellView<'_>,
    log: LogKind,
    capped: &[(WellId, f64)],
    rest: &[(WellId, f64)],
    all_sorted: &mut Option<Vec<(WellId, f64)>>,
) -> NeighborCurve {
    let cfg = pool.cfg;
    let batch = cfg.batch_size.max(1);
    let mut s = Stack::new(pool, target, log);
    let (primary, more) = capped.split_at(cfg.k_primary.min(capped.len()));
    s.add_all(primary);
    s.accept(cfg.quorum, 1);
    for chunk in more.chunks(batch) {
        if s.is_done() {
            break;
        }
        s.add_all(chunk);
        s.accept(cfg.quorum, 2);
    }
    if !s.is_done() {
        s.accept(1, 3);
    }
    for chunk in rest.chunks(batch) {
        if s.is_done() {
            break;
        }
        s.add_all(chunk);
        s.accept(1, 4);
    }
    if !s.is_done() {
        let all = all_sorted.get_or_insert_with(|| {
            target
                .xy
                .map_or_else(Vec::new, |q| pool.index.all_sorted(q))
        });
        let eps = cfg.interval_eps;
        let restrict: Option<BTreeSet<i64>> = target.coord.as_ref().and_then(|c| {
            s.open
                .iter()
                .map(|&r| c.at(r))
                .all(f64::is_finite)
                .then(|| {
                    s.open
                        .iter()
                        .map(|&r| interval_of(c.at(r), eps))
                        .filter(|&i| i >= 0)
                        .collect()
                })
        });
        let walk: Vec<(WellId, f64)> = all
            .iter()
            .copied()
            .filter(|(id, _)| {
                !s.stacked.contains(id)
                    && pool
                        .eligible(target.id, *id, log)
                        .is_some_and(|st| restrict.as_ref().is_none_or(|set| st.covers(set)))
            })
            .collect();
        for chunk in walk.chunks(batch) {
            if s.is_done() {
                break;
            }
            s.add_all(chunk);
            s.accept(1, 5);
        }
    }
    s.curve
}

/// On the stored curve.
pub(crate) fn validate_curve(curve: &mut NeighborCurve, log: LogKind, cfg: &NeighborsConfig) {
    let b = working_bounds(log);
    let out_of_bounds: Vec<usize> = curve
        .value
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite() && !b.contains(f64::from(**v)))
        .map(|(i, _)| i)
        .collect();
    for i in out_of_bounds {
        curve.blank(i);
    }
    let mut pieces: Vec<Vec<usize>> = Vec::new();
    let mut prev: Option<usize> = None;
    for (i, v) in curve.value.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        match (prev, pieces.last_mut()) {
            (Some(p), Some(piece)) if GRID_STEP_FT * (i - p) as f64 <= cfg.validate_split_ft => {
                piece.push(i);
            }
            _ => pieces.push(vec![i]),
        }
        prev = Some(i);
    }
    let sat_tol = cfg.saturation_rel_tol * (b.hi - b.lo).abs().max(1.0);
    for piece in pieces {
        if piece.len() < cfg.validate_min_run {
            continue;
        }
        let vals: Vec<f64> = piece
            .iter()
            .filter_map(|&i| curve.value.get(i).map(|&v| f64::from(v)))
            .collect();
        let (mn, mx, max_abs) = vals.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY, 0.0f64),
            |(mn, mx, ma), &v| (mn.min(v), mx.max(v), ma.max(v.abs())),
        );
        let flat = mx - mn <= cfg.flat_rel_tol * max_abs.max(1.0);
        let n_sat = vals
            .iter()
            .filter(|&&v| (v - b.lo).abs() <= sat_tol || (v - b.hi).abs() <= sat_tol)
            .count();
        let saturated = n_sat as f64 >= cfg.saturation_frac * vals.len() as f64;
        if flat || saturated {
            for &i in &piece {
                curve.blank(i);
            }
        }
    }
}

fn set_trusted(curve: &mut NeighborCurve, cfg: &NeighborsConfig) {
    let rows = curve
        .value
        .iter()
        .zip(&curve.n_donors)
        .zip(&curve.dist_m)
        .zip(&curve.quality);
    for (t, (((v, &n), &d), &q)) in curve.trusted.iter_mut().zip(rows) {
        *t = v.is_finite()
            && n as usize >= cfg.trusted_min_n
            && d.is_finite()
            && d <= cfg.trusted_max_dist_m
            && f64::from(q) >= cfg.trusted_min_quality;
    }
}

/// Curves of one target: tiered search, validation and trusted flag for every
/// log in suite order (no range map yet).
pub(crate) fn build_target(pool: &Pool<'_>, target: WellId) -> SxResult<WellNeighborCurves> {
    let cfg = pool.cfg;
    let t = pool
        .wells
        .get(&target)
        .ok_or_else(|| SxError::NotFound(format!("target {} has no logs", target.0)))?;
    let knn =
        t.xy.map_or_else(Vec::new, |q| pool.index.knn(q, cfg.knn_list_len));
    let target_set: Option<BTreeSet<i64>> = t.coord.as_ref().map(|c| {
        c.finite_values()
            .map(|u| interval_of(u, cfg.interval_eps))
            .filter(|&i| i >= 0)
            .collect()
    });
    let mut all_sorted = None;
    let mut curves = BTreeMap::new();
    let mut tally = String::new();
    for log in LogKind::ALL {
        let (capped, rest) = pool.candidates(target, &knn, log, target_set.as_ref());
        let mut curve = tiered_curve(pool, t, log, &capped, &rest, &mut all_sorted);
        validate_curve(&mut curve, log, cfg);
        set_trusted(&mut curve, cfg);
        let mut counts = [0usize; 6];
        for &code in &curve.tier {
            let slot = if (1..=5).contains(&code) {
                usize::from(code - 1)
            } else {
                5
            };
            if let Some(c) = counts.get_mut(slot) {
                *c += 1;
            }
        }
        let _ = write!(tally, "{}={counts:?} ", log.name());
        curves.insert(log, curve);
    }
    debug!(
        well = target.0,
        tiers = tally.trim_end(),
        "neighbor curves: tier counts [1, 2, 3, 4, 5, none]"
    );
    Ok(WellNeighborCurves {
        id: target,
        grid: t.logs.grid,
        curves,
    })
}

/// Two-point map `(gain, offset)` of `(c_low, c_high)` onto `(ref_low, ref_high)`
///; `None` when the curve range is degenerate.
pub(crate) fn range_map_params(
    c_low: f64,
    c_high: f64,
    ref_low: f64,
    ref_high: f64,
    cfg: &NeighborsConfig,
) -> Option<(f64, f64)> {
    let den = c_high - c_low;
    if !den.is_finite() || den.abs() <= 1e-10 {
        return None;
    }
    let span = ref_high - ref_low;
    let gain = if den.abs() < cfg.min_span_frac * span.abs() {
        1.0
    } else {
        let g = span / den;
        let sign = if g > 0.0 {
            1.0
        } else if g < 0.0 {
            -1.0
        } else {
            0.0
        };
        sign * g.abs().max(1.0 / cfg.max_gain).min(cfg.max_gain)
    };
    let offset = (ref_low + ref_high) / 2.0 - gain * (c_low + c_high) / 2.0;
    Some((gain, offset))
}

/// For one well: every log of `range_map_logs` with a reference pair.
///
/// The window is the rows the reference pair describes: coverage rows with a
/// finite layer coordinate `coord` (normalization's layer scope), or all coverage
/// rows when that leaves fewer than `range_map_min_n` values.
///
/// Errors: `InvalidInput` if a configured percentile is outside [0, 100].
pub(crate) fn range_map_well(
    w: &mut WellNeighborCurves,
    coverage: Option<&Coverage>,
    coord: Option<&[f64]>,
    ref_ranges: &BTreeMap<(WellId, LogKind), (f64, f64)>,
    cfg: &NeighborsConfig,
) -> SxResult<()> {
    let Some(cov) = coverage else {
        return Ok(());
    };
    let logs: BTreeSet<LogKind> = cfg.range_map_logs.iter().copied().collect();
    for log in logs {
        let Some(&(ref_low, ref_high)) = ref_ranges.get(&(w.id, log)) else {
            continue;
        };
        if !ref_low.is_finite() || !ref_high.is_finite() {
            continue;
        }
        let Some(curve) = w.curves.get_mut(&log) else {
            continue;
        };
        range_map_curve(curve, w.grid, cov, coord, log, (ref_low, ref_high), cfg)?;
    }
    Ok(())
}

fn range_map_curve(
    curve: &mut NeighborCurve,
    grid: GridRange,
    cov: &Coverage,
    coord: Option<&[f64]>,
    log: LogKind,
    (ref_low, ref_high): (f64, f64),
    cfg: &NeighborsConfig,
) -> SxResult<()> {
    let in_cov = |i: usize, v: f32| {
        let k = grid.k(i);
        v.is_finite() && cov.k_first <= k && k <= cov.k_last
    };
    let in_layer = |i: usize| coord.is_some_and(|c| c.get(i).is_some_and(|u| u.is_finite()));
    let pick = |layer_only: bool| -> Vec<f64> {
        curve
            .value
            .iter()
            .enumerate()
            .filter(|&(i, &v)| in_cov(i, v) && (!layer_only || in_layer(i)))
            .map(|(_, &v)| f64::from(v))
            .collect()
    };
    let mut window = pick(true);
    if window.len() < cfg.range_map_min_n {
        window = pick(false);
    }
    if window.len() < cfg.range_map_min_n {
        return Ok(());
    }
    let p = percentiles(&window, &[cfg.range_p_low, cfg.range_p_high])?;
    let (Some(&c_low), Some(&c_high)) = (p.first(), p.get(1)) else {
        return Ok(());
    };
    let Some((gain, offset)) = range_map_params(c_low, c_high, ref_low, ref_high, cfg) else {
        return Ok(());
    };
    let b = working_bounds(log);
    for (v, s) in curve.value.iter_mut().zip(curve.spread.iter_mut()) {
        if !v.is_finite() {
            continue;
        }
        let mapped = gain * f64::from(*v) + offset;
        if b.contains(mapped) {
            *v = mapped as f32;
            *s = (f64::from(*s) * gain.abs()) as f32;
        }
    }
    Ok(())
}
