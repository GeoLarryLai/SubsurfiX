// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Layer coordinate, match rows, robust sparse solve, monotonicity, inverse
//! lookup and output tables.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, MatMut, Par, Side};
use rayon::prelude::*;

use crate::config::CorrelateSolveConfig;
use crate::correlate::graph::WellGraph;
use crate::correlate::pairs::PairPaths;
use crate::correlate::{interval_solvable, interval_usable, WellBoundaries};
use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;
use crate::types::{GridRange, WellId, WellLogs, GRID_STEP_FT};
use crate::util::{isotonic_increasing, median, MAD_TO_SIGMA};

const LAYER_TIME_TAG: &str = "subsurfix.layer_time/1";
const SOLVE_REPORT_TAG: &str = "subsurfix.solve_report/1";

/// Floor of the robust scale `s` (layer-time fraction).
const SCALE_FLOOR: f64 = 1e-3;

/// Final layer time of one well on its grid.
#[derive(Clone, Debug)]
pub struct LayerTimeWell {
    /// Well id.
    pub well: WellId,
    /// Grid range (the well's logs grid).
    pub grid: GridRange,
    /// Length `grid.len`. NaN where undefined.
    pub layer_time: Vec<f64>,
}

/// Kind of a `solve_report` row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportRowKind {
    /// One row per interval.
    Interval,
    /// One row per active edge with a path.
    Edge,
}

/// One row of the `solve_report` table.
#[derive(Clone, Debug)]
pub struct SolveReportRow {
    /// Pass 1 or 2.
    pub pass: u8,
    /// Interval or edge row.
    pub kind: ReportRowKind,
    /// Interval index.
    pub interval: u16,
    /// Lower id (edge rows).
    pub well_a: Option<WellId>,
    /// Higher id (edge rows).
    pub well_b: Option<WellId>,
    /// Edge length, m (NaN on interval rows).
    pub length_m: f64,
    /// Path quality (NaN on interval rows).
    pub quality: f64,
    /// Kept rows.
    pub n_rows: usize,
    /// Kept rows with `w_robust = 0` in the final solve.
    pub n_zero_weight: usize,
    /// Signed median residual of the final solution (layer-time fraction).
    pub median_residual: f64,
    /// Robust scale `s` of the final weights (NaN on edge rows, with no
    /// robust passes, or with no rows).
    pub scale: f64,
    /// Unknowns solved (0 on edge rows).
    pub n_unknowns: usize,
    /// Wells with free knots that no kept row touches, ascending.
    pub unconstrained: Vec<WellId>,
    /// Failed active pairs of the interval (0 on edge rows).
    pub n_dtw_failed: usize,
    /// DTW ran for this edge in this pass (set by the caller of
    /// [`solve_pass`]; `false` from [`solve_pass`] itself).
    pub path_recomputed: bool,
}

/// Result of one solve pass.
#[derive(Clone, Debug)]
pub struct PassOutput {
    /// One entry per well of `bounds`, sorted by well.
    pub layer_time: Vec<LayerTimeWell>,
    /// Per interval `0..=n`: the interval row, then its edge rows.
    pub report: Vec<SolveReportRow>,
}

/// Check the `[correlate.solve]` keys ([`SxError::Config`]).
pub(crate) fn check_solve_config(cfg: &CorrelateSolveConfig) -> SxResult<()> {
    let bad = |key: &str, rule: &str| {
        Err(SxError::Config(format!(
            "correlate.solve.{key} must be {rule}"
        )))
    };
    if cfg.knot_step_ft.is_nan() || cfg.knot_step_ft <= 0.0 {
        return bad("knot_step_ft", "> 0");
    }
    if cfg.match_step < 1 {
        return bad("match_step", ">= 1");
    }
    if !cfg.lambda_smooth.is_finite() || cfg.lambda_smooth < 0.0 {
        return bad("lambda_smooth", ">= 0");
    }
    if !cfg.lambda_ridge.is_finite() || cfg.lambda_ridge <= 0.0 {
        return bad("lambda_ridge", "> 0");
    }
    if cfg.tukey_c.is_nan() || cfg.tukey_c <= 0.0 {
        return bad("tukey_c", "> 0");
    }
    Ok(())
}

/// Usable interval and fraction of a sample at `md`.
///
/// **Choice:** The interval index is returned with the fraction, never
/// recovered from `floor(u)`.
fn locate(b: &WellBoundaries, md: f64) -> Option<(usize, f64)> {
    let mut at_base: Option<(usize, f64)> = None;
    for i in 0..b.state.len() {
        if !interval_usable(b, i) {
            continue;
        }
        let (Some(&top), Some(&base)) = (b.md_ft.get(i), b.md_ft.get(i + 1)) else {
            continue;
        };
        if top <= md && md < base {
            return Some((i, (md - top) / (base - top)));
        }
        if md == base && at_base.is_none() {
            at_base = Some((i, 1.0));
        }
    }
    at_base
}

/// Rough layer coordinate `u = i + f` on the well's grid. NaN where undefined.
///
/// Sample `md = 0.5 k` belongs to usable interval `i` when
/// `md_ft[i] ≤ md < md_ft[i+1]`, or with `f = 1` when it equals the base of a
/// usable interval and no interval holds it. Usable = state not `Unknown`,
/// finite boundaries, positive thickness.
///
/// Units: coordinate units (interval + fraction). NaN rule: NaN outside
/// every usable interval. Errors: none. Complexity: `O(samples × intervals)`.
pub fn layer_coordinate(grid: &GridRange, b: &WellBoundaries) -> Vec<f64> {
    (0..grid.len)
        .map(|s| locate(b, grid.md_ft(s)).map_or(f64::NAN, |(i, f)| i as f64 + f))
        .collect()
}

/// Smallest MD (ft) where `tau >= t`, linear between grid samples. NaN if not covered.
/// Works for any non-decreasing coordinate (layer time or rough `u`).
///
/// First local sample `m` with finite `tau[m] ≥ t`: exact hit returns its MD;
/// otherwise sample `m − 1` must be finite and the MD is interpolated.
/// Samples beyond `grid.len` are ignored.
///
/// NaN rule: NaN `t` gives NaN; NaN `tau` samples are skipped as candidates
/// and block interpolation. Errors: none. Complexity: `O(samples)`.
pub fn md_at_layer_time(grid: &GridRange, tau: &[f64], t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let Some(m) = tau
        .iter()
        .take(grid.len)
        .position(|&v| v.is_finite() && v >= t)
    else {
        return f64::NAN;
    };
    let Some(&tm) = tau.get(m) else {
        return f64::NAN;
    };
    if tm == t {
        return grid.md_ft(m);
    }
    let Some(prev) = m.checked_sub(1) else {
        return f64::NAN;
    };
    match tau.get(prev) {
        Some(&tp) if tp.is_finite() => grid.md_ft(prev) + GRID_STEP_FT * (t - tp) / (tm - tp),
        _ => f64::NAN,
    }
}

/// One match row of the interval system: `Σ c δ = rhs` with a weight.
#[derive(Clone, Debug)]
pub(crate) struct MatchRow {
    /// (unknown index, coefficient), indices `< n_unknowns`.
    pub terms: Vec<(usize, f64)>,
    /// Right-hand side `f_b − f_a`.
    pub rhs: f64,
    /// Row weight `w_edge × w_robust`.
    pub weight: f64,
}

fn add_outer(lower: &mut BTreeMap<(usize, usize), f64>, terms: &[(usize, f64)], w: f64) {
    for &(p, cp) in terms {
        for &(q, cq) in terms {
            if p >= q {
                *lower.entry((p, q)).or_insert(0.0) += w * cp * cq;
            }
        }
    }
}

/// Solve one interval system for `δ`.
///
/// `N = Σ w c cᵀ + λ_s Σ d dᵀ + λ_r I`, `rhs = Σ w c b`, lower triangle
/// accumulated in a `BTreeMap` (rows in order, then `smooth` vectors, then
/// the ridge), sparse LLᵀ in `faer` with AMD ordering and `Par::Seq`.
///
/// **Choice:** Shape: explicit rows (`terms`, `rhs`, `weight`), the
/// smoothness vectors `d` already restricted to unknowns, and the two
/// weights. No unknowns gives an empty vector.
///
/// NaN rule: error. Errors: [`SxError::InvalidInput`] (term index out of
/// range), [`SxError::Numerical`] naming the interval (matrix build,
/// factorization or a non-finite solution). Complexity: sparse Cholesky.
pub(crate) fn solve_interval_rows(
    interval: u16,
    n_unknowns: usize,
    rows: &[MatchRow],
    smooth: &[Vec<(usize, f64)>],
    lambda_smooth: f64,
    lambda_ridge: f64,
) -> SxResult<Vec<f64>> {
    if n_unknowns == 0 {
        return Ok(Vec::new());
    }
    let numerical = |what: &str| SxError::Numerical(format!("interval {interval}: {what}"));
    let mut lower: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut rhs = vec![0.0f64; n_unknowns];
    for row in rows {
        add_outer(&mut lower, &row.terms, row.weight);
        for &(p, c) in &row.terms {
            let slot = rhs.get_mut(p).ok_or_else(|| {
                SxError::InvalidInput(format!("interval {interval}: unknown {p} out of range"))
            })?;
            *slot += row.weight * c * row.rhs;
        }
    }
    for d in smooth {
        add_outer(&mut lower, d, lambda_smooth);
    }
    for p in 0..n_unknowns {
        *lower.entry((p, p)).or_insert(0.0) += lambda_ridge;
    }
    if lower.keys().any(|&(p, _)| p >= n_unknowns) {
        return Err(SxError::InvalidInput(format!(
            "interval {interval}: unknown index out of range"
        )));
    }
    let triplets: Vec<Triplet<usize, usize, f64>> = lower
        .iter()
        .map(|(&(r, c), &v)| Triplet::new(r, c, v))
        .collect();
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n_unknowns, n_unknowns, &triplets)
        .map_err(|e| numerical(&format!("cannot build normal matrix: {e:?}")))?;
    let symbolic = factorize_symbolic_cholesky(
        a.symbolic(),
        Side::Lower,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .map_err(|e| numerical(&format!("symbolic Cholesky failed: {e:?}")))?;
    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let mut mem = MemBuffer::try_new(StackReq::any_of(&[
        symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Default::default()),
        symbolic.solve_in_place_scratch::<f64>(1, Par::Seq),
    ]))
    .map_err(|e| numerical(&format!("cannot allocate solver workspace: {e:?}")))?;
    let stack = MemStack::new(&mut mem);
    let llt = symbolic
        .factorize_numeric_llt::<f64>(
            &mut l_values,
            a.as_ref(),
            Side::Lower,
            LltRegularization::default(),
            Par::Seq,
            stack,
            Default::default(),
        )
        .map_err(|e| numerical(&format!("Cholesky factorization failed: {e:?}")))?;
    let mut x = rhs;
    llt.solve_in_place_with_conj(
        Conj::No,
        MatMut::from_column_major_slice_mut(&mut x, n_unknowns, 1),
        Par::Seq,
        stack,
    );
    if x.iter().any(|v| !v.is_finite()) {
        return Err(numerical("solution is not finite"));
    }
    Ok(x)
}

/// Pool-adjacent-violators (non-decreasing, equal weights), then clip to
/// `[0, 1]`.
pub(crate) fn monotone_knots(g: &[f64]) -> SxResult<Vec<f64>> {
    Ok(isotonic_increasing(g, None)?
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0))
        .collect())
}

/// Knot `j` and weight `θ` on knot `j + 1` at fraction `f`, `m ≥ 1` segments.
fn knot_pos(f: f64, m: usize) -> (usize, f64) {
    let fm = f * m as f64;
    let j = (fm.floor().max(0.0) as usize).min(m.saturating_sub(1));
    (j, fm - j as f64)
}

/// Piecewise-linear knot curve at `f`.
fn eval_knots(g: &[f64], m: usize, f: f64) -> f64 {
    let (j, theta) = knot_pos(f, m);
    let at = |k: usize| g.get(k).copied().unwrap_or(f64::NAN);
    if theta == 0.0 {
        at(j)
    } else {
        (1.0 - theta) * at(j) + theta * at(j + 1)
    }
}

/// Knots of one well in one interval.
#[derive(Clone, Copy, Debug)]
struct Knots {
    m: usize,
    top: f64,
    thick: f64,
    /// First and last free knot index (inclusive); none free when `hi < lo`.
    lo: usize,
    hi: usize,
    /// First unknown index when the well is in the system.
    offset: Option<usize>,
}

impl Knots {
    fn is_free(&self, j: usize) -> bool {
        j >= self.lo && j <= self.hi
    }

    fn n_free(&self) -> usize {
        (self.hi + 1).saturating_sub(self.lo)
    }

    fn unknown(&self, j: usize) -> Option<usize> {
        if self.is_free(j) {
            Some(self.offset? + (j - self.lo))
        } else {
            None
        }
    }
}

/// Match row before unknowns are numbered: (bounds position, knot, coefficient).
struct RawRow {
    edge: usize,
    terms: Vec<(usize, usize, f64)>,
    rhs: f64,
    w_edge: f64,
}

/// Solved interval: monotone knots per bounds position (`None` → `τ = u`)
/// and the report rows.
struct IntervalOut {
    knots: Vec<Option<(usize, Vec<f64>)>>,
    report: Vec<SolveReportRow>,
}

struct PassCtx<'a> {
    pass: u8,
    graph: &'a WellGraph,
    bounds: &'a [WellBoundaries],
    paths: &'a PairPaths,
    cfg: &'a CorrelateSolveConfig,
    n_int: usize,
    /// Bounds positions of each edge's wells.
    edge_pos: Vec<Option<(usize, usize)>>,
}

fn residuals(rows: &[MatchRow], delta: &[f64]) -> Vec<f64> {
    rows.iter()
        .map(|r| {
            r.terms
                .iter()
                .map(|&(p, c)| c * delta.get(p).copied().unwrap_or(f64::NAN))
                .sum::<f64>()
                - r.rhs
        })
        .collect()
}

fn median_or_nan(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        median(v)
    }
}

fn well_knots(b: &WellBoundaries, i: usize, last: usize, step: f64) -> SxResult<Option<Knots>> {
    if !interval_solvable(b, i) {
        return Ok(None);
    }
    let (Some(&top), Some(&base)) = (b.md_ft.get(i), b.md_ft.get(i + 1)) else {
        return Ok(None);
    };
    let thick = base - top;
    let m = (thick / step).round().max(1.0);
    if m.is_nan() || m > f64::from(u32::MAX) {
        return Err(SxError::InvalidInput(format!(
            "well {} interval {i}: too many knots ({m})",
            b.well.0
        )));
    }
    let m = m as usize;
    Ok(Some(Knots {
        m,
        top,
        thick,
        lo: if i == 0 { 0 } else { 1 },
        hi: if i == last { m } else { m - 1 },
        offset: None,
    }))
}

/// Grid-index targets of one edge and the first path pair at each.
fn sample_path(path: &[(i32, i32)], top_a: f64, base_a: f64, step: usize) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    let k0 = (top_a / GRID_STEP_FT).ceil();
    if !k0.is_finite() {
        return out;
    }
    let mut k = k0 as i64;
    let step = i64::try_from(step).unwrap_or(i64::MAX);
    let mut p = 0usize;
    while GRID_STEP_FT * (k as f64) <= base_a {
        while path.get(p).is_some_and(|&(ka, _)| i64::from(ka) < k) {
            p += 1;
        }
        match path.get(p) {
            Some(&pair) if i64::from(pair.0) == k => out.push(pair),
            Some(_) => {}
            None => break,
        }
        k = k.saturating_add(step);
    }
    out
}

fn interp_terms(pos: usize, kn: &Knots, f: f64, sign: f64, out: &mut Vec<(usize, usize, f64)>) {
    let (j, theta) = knot_pos(f, kn.m);
    for (jj, w) in [(j, 1.0 - theta), (j + 1, theta)] {
        if w != 0.0 && kn.is_free(jj) {
            out.push((pos, jj, sign * w));
        }
    }
}

fn solve_one_interval(ctx: &PassCtx<'_>, i: usize) -> SxResult<IntervalOut> {
    let iv =
        u16::try_from(i).map_err(|_| SxError::Shape(format!("interval index {i} exceeds u16")))?;
    let cfg = ctx.cfg;
    let last = ctx.n_int.saturating_sub(1);
    let mut knots: Vec<Option<Knots>> = ctx
        .bounds
        .iter()
        .map(|b| well_knots(b, i, last, cfg.knot_step_ft))
        .collect::<SxResult<_>>()?;

    // Match rows, by edge index then target.
    let mut raw: Vec<RawRow> = Vec::new();
    let mut edges_with_path: Vec<usize> = Vec::new();
    let mut n_failed = 0usize;
    for (e, edge) in ctx.graph.edges.iter().enumerate() {
        let Some(Some((pa, pb))) = ctx.edge_pos.get(e).copied() else {
            continue;
        };
        let (Some(Some(ka)), Some(Some(kb))) = (knots.get(pa), knots.get(pb)) else {
            continue;
        };
        if ctx.paths.failed.contains(&(iv, e)) {
            n_failed += 1;
        }
        let Some(path) = ctx.paths.paths.get(&(iv, e)) else {
            continue;
        };
        edges_with_path.push(e);
        let w_edge = ctx.graph.edge_weight(edge.length_m, path.quality);
        if w_edge.is_nan() || w_edge <= 0.0 {
            continue;
        }
        let base_a = ka.top + ka.thick;
        for (k_a, k_b) in sample_path(&path.path, ka.top, base_a, cfg.match_step) {
            let f_a = ((GRID_STEP_FT * f64::from(k_a) - ka.top) / ka.thick).clamp(0.0, 1.0);
            let f_b = ((GRID_STEP_FT * f64::from(k_b) - kb.top) / kb.thick).clamp(0.0, 1.0);
            let mut terms = Vec::with_capacity(4);
            interp_terms(pa, ka, f_a, 1.0, &mut terms);
            interp_terms(pb, kb, f_b, -1.0, &mut terms);
            if terms.is_empty() {
                continue;
            }
            raw.push(RawRow {
                edge: e,
                terms,
                rhs: f_b - f_a,
                w_edge,
            });
        }
    }

    // Unknowns: touched wells with free knots, wells ascending, then knot.
    let touched: BTreeSet<usize> = raw
        .iter()
        .flat_map(|r| r.terms.iter().map(|t| t.0))
        .collect();
    let mut unconstrained = Vec::new();
    let mut n_unknowns = 0usize;
    for (pos, slot) in knots.iter_mut().enumerate() {
        let Some(kn) = slot else {
            continue;
        };
        if kn.n_free() == 0 {
            continue;
        }
        if touched.contains(&pos) {
            kn.offset = Some(n_unknowns);
            n_unknowns += kn.n_free();
        } else if let Some(b) = ctx.bounds.get(pos) {
            unconstrained.push(b.well);
        }
    }
    let unknown_of = |pos: usize, j: usize| -> Option<usize> {
        knots.get(pos).copied().flatten().and_then(|k| k.unknown(j))
    };
    let mut rows: Vec<MatchRow> = Vec::with_capacity(raw.len());
    for r in &raw {
        let mut terms = Vec::with_capacity(r.terms.len());
        for &(pos, j, c) in &r.terms {
            let p = unknown_of(pos, j).ok_or_else(|| {
                SxError::Numerical(format!("interval {i}: row touches an unnumbered knot"))
            })?;
            terms.push((p, c));
        }
        rows.push(MatchRow {
            terms,
            rhs: r.rhs,
            weight: r.w_edge,
        });
    }
    let mut smooth: Vec<Vec<(usize, f64)>> = Vec::new();
    for (pos, kn) in knots.iter().enumerate() {
        let Some(kn) = kn else {
            continue;
        };
        if kn.offset.is_none() {
            continue;
        }
        for j in 1..kn.m {
            let d: Vec<(usize, f64)> = [(j - 1, 1.0), (j, -2.0), (j + 1, 1.0)]
                .into_iter()
                .filter_map(|(jj, c)| Some((unknown_of(pos, jj)?, c)))
                .collect();
            if !d.is_empty() {
                smooth.push(d);
            }
        }
    }

    // Robust solve.
    let mut w_rob = vec![1.0f64; rows.len()];
    let mut scale = f64::NAN;
    let mut delta = Vec::new();
    let mut resid = Vec::new();
    if n_unknowns > 0 {
        let solve = |rows: &[MatchRow]| {
            solve_interval_rows(
                iv,
                n_unknowns,
                rows,
                &smooth,
                cfg.lambda_smooth,
                cfg.lambda_ridge,
            )
        };
        delta = solve(&rows)?;
        if !rows.is_empty() {
            for _ in 0..cfg.robust_passes {
                let r = residuals(&rows, &delta);
                let med = median(&r);
                let dev: Vec<f64> = r.iter().map(|v| (v - med).abs()).collect();
                let s = (MAD_TO_SIGMA * median(&dev)).max(SCALE_FLOOR);
                let cs = cfg.tukey_c * s;
                for ((w, row), (raw_row, rv)) in w_rob
                    .iter_mut()
                    .zip(rows.iter_mut())
                    .zip(raw.iter().zip(&r))
                {
                    *w = if rv.abs() < cs {
                        let q = rv / cs;
                        (1.0 - q * q) * (1.0 - q * q)
                    } else {
                        0.0
                    };
                    row.weight = raw_row.w_edge * *w;
                }
                scale = s;
                delta = solve(&rows)?;
            }
        }
        resid = residuals(&rows, &delta);
    }

    // Report rows.
    let mut report = vec![SolveReportRow {
        pass: ctx.pass,
        kind: ReportRowKind::Interval,
        interval: iv,
        well_a: None,
        well_b: None,
        length_m: f64::NAN,
        quality: f64::NAN,
        n_rows: rows.len(),
        n_zero_weight: w_rob.iter().filter(|&&w| w == 0.0).count(),
        median_residual: median_or_nan(&resid),
        scale,
        n_unknowns,
        unconstrained: unconstrained.clone(),
        n_dtw_failed: n_failed,
        path_recomputed: false,
    }];
    let mut rows_of_edge: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, r) in raw.iter().enumerate() {
        rows_of_edge.entry(r.edge).or_default().push(k);
    }
    for &e in &edges_with_path {
        let (Some(edge), Some(path)) = (ctx.graph.edges.get(e), ctx.paths.paths.get(&(iv, e)))
        else {
            continue;
        };
        let idx: &[usize] = rows_of_edge.get(&e).map_or(&[], Vec::as_slice);
        let edge_resid: Vec<f64> = idx.iter().filter_map(|&k| resid.get(k).copied()).collect();
        report.push(SolveReportRow {
            pass: ctx.pass,
            kind: ReportRowKind::Edge,
            interval: iv,
            well_a: Some(edge.a),
            well_b: Some(edge.b),
            length_m: edge.length_m,
            quality: f64::from(path.quality),
            n_rows: idx.len(),
            n_zero_weight: idx
                .iter()
                .filter(|&&k| w_rob.get(k).is_some_and(|&w| w == 0.0))
                .count(),
            median_residual: median_or_nan(&edge_resid),
            scale: f64::NAN,
            n_unknowns: 0,
            unconstrained: Vec::new(),
            n_dtw_failed: 0,
            path_recomputed: false,
        });
    }

    // Monotone knot values; unconstrained wells keep `τ = u`.
    let mut out_knots = Vec::with_capacity(knots.len());
    for (pos, kn) in knots.iter().enumerate() {
        let Some(kn) = kn else {
            out_knots.push(None);
            continue;
        };
        if kn.n_free() > 0 && kn.offset.is_none() {
            out_knots.push(None);
            continue;
        }
        let g: Vec<f64> = (0..=kn.m)
            .map(|j| {
                let d = unknown_of(pos, j)
                    .and_then(|p| delta.get(p).copied())
                    .unwrap_or(0.0);
                j as f64 / kn.m as f64 + d
            })
            .collect();
        out_knots.push(Some((kn.m, monotone_knots(&g)?)));
    }
    Ok(IntervalOut {
        knots: out_knots,
        report,
    })
}

/// One solve pass over all intervals.
///
/// Per interval (rayon, collected in interval order): knots
/// `M = max(1, round(T / knot_step_ft))` for wells where the interval is
/// usable and not `Pinched`; free knots `1..M−1` plus the outer pad ends;
/// match rows sampled every `match_step` grid samples of well `a`; robust
/// Tukey reweighting (`robust_passes`); monotone knots (PAV then clip). Layer
/// time `τ = i + G(f)`; `Pinched` intervals and unconstrained wells keep
/// `τ = u`; `Unknown` or no interval gives NaN. `path_recomputed` is left
/// `false`.
///
/// Units: ft, layer-time units. NaN rule: NaN boundaries make intervals
/// unusable (NaN layer time there). Errors: [`SxError::Config`] (bad solve
/// keys), [`SxError::Shape`] (bounds not sorted by well, unequal interval
/// counts, `md_ft.len() != state.len() + 1`, a well missing from `logs`),
/// [`SxError::Numerical`] (a factorization fails, naming the interval).
/// Empty `bounds` gives empty output. Complexity: per interval one sparse
/// Cholesky per robust pass plus `O(rows)`, and `O(samples × intervals)`.
pub fn solve_pass(
    pass: u8,
    graph: &WellGraph,
    logs: &[WellLogs],
    bounds: &[WellBoundaries],
    paths: &PairPaths,
    cfg: &CorrelateSolveConfig,
) -> SxResult<PassOutput> {
    check_solve_config(cfg)?;
    if bounds
        .windows(2)
        .any(|p| matches!(p, [x, y] if x.well >= y.well))
    {
        return Err(SxError::Shape(
            "solve: boundaries must be sorted by well, unique".into(),
        ));
    }
    let n_int = bounds.first().map_or(0, |b| b.state.len());
    if bounds
        .iter()
        .any(|b| b.state.len() != n_int || b.md_ft.len() != n_int + 1)
    {
        return Err(SxError::Shape(
            "solve: every well needs the same intervals and n + 2 boundaries".into(),
        ));
    }
    let logs_by_id: BTreeMap<WellId, &WellLogs> = logs.iter().map(|l| (l.id, l)).collect();
    let mut grids = Vec::with_capacity(bounds.len());
    for b in bounds {
        let l = logs_by_id
            .get(&b.well)
            .ok_or_else(|| SxError::Shape(format!("solve: no logs for well {}", b.well.0)))?;
        grids.push(l.grid);
    }
    let pos_of = |w: WellId| bounds.binary_search_by_key(&w, |b| b.well).ok();
    let ctx = PassCtx {
        pass,
        graph,
        bounds,
        paths,
        cfg,
        n_int,
        edge_pos: graph
            .edges
            .iter()
            .map(|e| Some((pos_of(e.a)?, pos_of(e.b)?)))
            .collect(),
    };
    let solved: Vec<SxResult<IntervalOut>> = (0..n_int)
        .into_par_iter()
        .map(|i| solve_one_interval(&ctx, i))
        .collect();
    let mut outs = Vec::with_capacity(n_int);
    for r in solved {
        outs.push(r?);
    }

    let mut layer_time = Vec::with_capacity(bounds.len());
    for (pos, (b, grid)) in bounds.iter().zip(&grids).enumerate() {
        let mut tau: Vec<f64> = (0..grid.len)
            .map(|s| match locate(b, grid.md_ft(s)) {
                None => f64::NAN,
                Some((i, f)) => {
                    let g = outs
                        .get(i)
                        .and_then(|o| o.knots.get(pos))
                        .and_then(Option::as_ref);
                    match g {
                        Some((m, g)) => i as f64 + eval_knots(g, *m, f),
                        None => i as f64 + f,
                    }
                }
            })
            .collect();
        // Knots are monotone; interpolation round-off on a flat stretch can still
        // step back by one ulp, which the neighbor mapping rejects. Clamp it.
        let mut prev = f64::NEG_INFINITY;
        for t in tau.iter_mut().filter(|t| t.is_finite()) {
            if *t < prev {
                *t = prev;
            }
            prev = *t;
        }
        layer_time.push(LayerTimeWell {
            well: b.well,
            grid: *grid,
            layer_time: tau,
        });
    }
    let report = outs.into_iter().flat_map(|o| o.report).collect();
    Ok(PassOutput { layer_time, report })
}

fn finish(cols: Vec<(&str, ArrayRef)>, tag: &str) -> SxResult<RecordBatch> {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(name, a)| Field::new(*name, a.data_type().clone(), false))
        .collect();
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, tag)
}

/// `layer_time` product table.
///
/// One row per `(well_id, k)` over each well's grid, sorted by `well_id`
/// then `k` (row-order contract). Columns `well_id`, `k` int32, `layer_time`
/// float64. Schema tag `subsurfix.layer_time/1`.
///
/// NaN rule: NaN layer times are stored as is. Errors: [`SxError::Shape`]
/// (`layer_time.len() != grid.len`), [`SxError::InvalidInput`] (batch build).
/// Complexity: `O(samples + wells log wells)`.
pub fn layer_time_batch(rows: &[LayerTimeWell]) -> SxResult<arrow::record_batch::RecordBatch> {
    let mut sorted: Vec<&LayerTimeWell> = rows.iter().collect();
    sorted.sort_by_key(|r| r.well);
    let total: usize = sorted.iter().map(|r| r.grid.len).sum();
    let (mut well_id, mut k, mut tau) = (
        Vec::with_capacity(total),
        Vec::with_capacity(total),
        Vec::with_capacity(total),
    );
    for r in sorted {
        if r.layer_time.len() != r.grid.len {
            return Err(SxError::Shape(format!(
                "well {}: layer_time has {} samples, grid has {}",
                r.well.0,
                r.layer_time.len(),
                r.grid.len
            )));
        }
        for (s, &t) in r.layer_time.iter().enumerate() {
            well_id.push(r.well.0);
            k.push(r.grid.k(s));
            tau.push(t);
        }
    }
    finish(
        vec![
            ("well_id", Arc::new(Int32Array::from(well_id))),
            ("k", Arc::new(Int32Array::from(k))),
            ("layer_time", Arc::new(Float64Array::from(tau))),
        ],
        LAYER_TIME_TAG,
    )
}

fn count_i64(v: usize) -> SxResult<i64> {
    i64::try_from(v).map_err(|_| SxError::InvalidInput(format!("count {v} exceeds int64")))
}

/// `solve_report` table.
///
/// Rows in input order (Pass 1 then Pass 2 from [`super::run_correlation`]).
/// Columns `pass`, `interval` int32; `row_kind` utf8 (`interval` / `edge`);
/// `well_a`, `well_b` int32 (−1 on interval rows); `length_m`,
/// `quality_frac` float64; `n_rows`, `n_zero_weight` int64;
/// `zero_weight_share_frac` (NaN with no rows), `median_residual_frac`,
/// `scale_frac` float64; `n_unknowns`, `n_unconstrained` int64;
/// `unconstrained_wells` utf8 (ids joined by `;`); `n_dtw_failed` int64;
/// `path_recomputed` bool. Schema tag `subsurfix.solve_report/1`.
///
/// NaN rule: NaN floats are stored as is. Errors: [`SxError::InvalidInput`]
/// (count overflow, batch build). Complexity: `O(rows)`.
pub fn solve_report_batch(rows: &[SolveReportRow]) -> SxResult<arrow::record_batch::RecordBatch> {
    let ints = |f: &dyn Fn(&SolveReportRow) -> usize| -> SxResult<ArrayRef> {
        let v: Vec<i64> = rows
            .iter()
            .map(|r| count_i64(f(r)))
            .collect::<SxResult<_>>()?;
        Ok(Arc::new(Int64Array::from(v)))
    };
    let f64s = |f: &dyn Fn(&SolveReportRow) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(rows.iter().map(f)))
    };
    let cols: Vec<(&str, ArrayRef)> = vec![
        (
            "pass",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| i32::from(r.pass)),
            )),
        ),
        (
            "row_kind",
            Arc::new(StringArray::from_iter_values(rows.iter().map(
                |r| match r.kind {
                    ReportRowKind::Interval => "interval",
                    ReportRowKind::Edge => "edge",
                },
            ))),
        ),
        (
            "interval",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| i32::from(r.interval)),
            )),
        ),
        (
            "well_a",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.well_a.map_or(-1, |w| w.0)),
            )),
        ),
        (
            "well_b",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.well_b.map_or(-1, |w| w.0)),
            )),
        ),
        ("length_m", f64s(&|r| r.length_m)),
        ("quality_frac", f64s(&|r| r.quality)),
        ("n_rows", ints(&|r| r.n_rows)?),
        ("n_zero_weight", ints(&|r| r.n_zero_weight)?),
        (
            "zero_weight_share_frac",
            f64s(&|r| {
                if r.n_rows == 0 {
                    f64::NAN
                } else {
                    r.n_zero_weight as f64 / r.n_rows as f64
                }
            }),
        ),
        ("median_residual_frac", f64s(&|r| r.median_residual)),
        ("scale_frac", f64s(&|r| r.scale)),
        ("n_unknowns", ints(&|r| r.n_unknowns)?),
        ("n_unconstrained", ints(&|r| r.unconstrained.len())?),
        (
            "unconstrained_wells",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|r| {
                r.unconstrained
                    .iter()
                    .map(|w| w.0.to_string())
                    .collect::<Vec<_>>()
                    .join(";")
            }))),
        ),
        ("n_dtw_failed", ints(&|r| r.n_dtw_failed)?),
        (
            "path_recomputed",
            Arc::new(BooleanArray::from(
                rows.iter().map(|r| r.path_recomputed).collect::<Vec<_>>(),
            )),
        ),
    ];
    finish(cols, SOLVE_REPORT_TAG)
}
