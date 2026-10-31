// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tops check between network-solve Pass 1 and Pass 2, the `tops` table and
//! the suspect list.
//!
//! A user pick is never moved, deleted or rewritten in any output table: a
//! disagreeing user pick only gets `suspect = true`, a rank and a
//! `suggested_md_ft`. Depths are feet (MD), missing values NaN.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int32Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::config::{CorrelateTopsConfig, DtwConfig};
use crate::correlate::graph::WellGraph;
use crate::correlate::WellBoundaries;
use crate::dtw::{window_path, SpanSide};
use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;
use crate::types::{
    IntervalState, PairPath, TopSet, TopSource, WellGeometry, WellHeader, WellId, WellLogs,
    GRID_STEP_FT,
};
use crate::util::{median, MAD_TO_SIGMA};

const TOPS_TAG: &str = "subsurfix.tops/1";
const SUSPECTS_TAG: &str = "subsurfix.suspects/1";

/// Floor of the spread in the rank score, ft.
const RANK_SPREAD_FLOOR_FT: f64 = 1.0;

/// Gap kept by the order constraint, ft.
const ORDER_GAP_FT: f64 = 0.5;

/// Outcome of the window check for one top in one well.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckState {
    /// Votes were cast and compared.
    Checked,
    /// A touching interval is `Pinched` or `Unknown`, or the check did not run.
    NotChecked,
    /// Fewer voters than `min_votes`.
    TooFewVotes,
}

/// Check result of one top in one well.
#[derive(Clone, Debug)]
pub struct TopCheckRow {
    /// Well id.
    pub well: WellId,
    /// Top index `j` (boundary `j + 1`).
    pub top: u16,
    /// Check state.
    pub state: CheckState,
    /// Voters used.
    pub votes: u32,
    /// `z*`, ft. NaN if not computed.
    pub z_star_md_ft: f64,
    /// `s`, ft. NaN if not computed.
    pub spread_ft: f64,
    /// `Δ = p − z*` for user picks, ft. NaN otherwise.
    pub offset_ft: f64,
    /// User pick is suspect.
    pub suspect: bool,
    /// 1-based suspect rank, 0 if not suspect.
    pub rank: u32,
    /// Boundary moves for Pass 2.
    pub moved: bool,
    /// Re-estimated boundary was clamped.
    pub clamped: bool,
}

/// Tops check result.
#[derive(Clone, Debug)]
pub struct TopsCheck {
    /// One row per (well, top), sorted by (well, top).
    pub rows: Vec<TopCheckRow>,
    /// Boundaries for Pass 2.
    pub boundaries: Vec<WellBoundaries>,
}

/// One row of the `tops` table.
#[derive(Clone, Debug)]
pub struct TopRow {
    /// Well id.
    pub well: WellId,
    /// Well name from the header.
    pub well_name: String,
    /// Top index `j`.
    pub top: u16,
    /// Top name from the `TopSet`.
    pub top_name: String,
    /// User pick if present, else the Pass 2 boundary. ft.
    pub md_ft: f64,
    /// Elevation at `md_ft`, ft (positive up). NaN outside the geometry.
    pub elev_ft: f64,
    /// `User` or `Suggested` only (pads are not rows).
    pub source: TopSource,
    /// Check state.
    pub state: CheckState,
    /// User pick is suspect.
    pub suspect: bool,
    /// `z*` before clamping, ft. NaN if not computed.
    pub suggested_md_ft: f64,
    /// `Δ = p − z*` for user picks, ft. NaN otherwise.
    pub offset_ft: f64,
    /// Voters used.
    pub votes: u32,
    /// Vote spread `s`, ft. NaN if not computed.
    pub spread_ft: f64,
    /// 1-based suspect rank, 0 if not suspect.
    pub rank: u32,
    /// Re-estimated boundary was clamped.
    pub clamped: bool,
}

/// Check the `[correlate.tops]` keys ([`SxError::Config`]).
pub(crate) fn check_tops_config(cfg: &CorrelateTopsConfig) -> SxResult<()> {
    let bad = |key: &str, rule: &str| {
        Err(SxError::Config(format!(
            "correlate.tops.{key} must be {rule}"
        )))
    };
    if cfg.k_check < 1 {
        return bad("k_check", ">= 1");
    }
    if cfg.min_votes < 1 || cfg.min_votes > cfg.k_check {
        return bad("min_votes", "in 1 ..= k_check");
    }
    if cfg.tol_ft.is_nan() || cfg.tol_ft < 0.0 {
        return bad("tol_ft", ">= 0");
    }
    if cfg.spread_mult.is_nan() || cfg.spread_mult < 0.0 {
        return bad("spread_mult", ">= 0");
    }
    let share_ok = cfg.side_share > 0.0 && cfg.side_share <= 1.0;
    if !share_ok {
        return bad("side_share", "in (0, 1]");
    }
    Ok(())
}

/// Bounds sorted by well, unique, with `n + 2` boundaries and `n + 1` states.
fn check_bounds_layout(bounds: &[WellBoundaries]) -> SxResult<()> {
    if bounds
        .windows(2)
        .any(|p| matches!(p, [x, y] if x.well >= y.well))
    {
        return Err(SxError::Shape(
            "tops check: boundaries must be sorted by well, unique".into(),
        ));
    }
    for b in bounds {
        let nb = b.md_ft.len();
        if nb < 2 || b.source.len() != nb || b.state.len() + 1 != nb {
            return Err(SxError::Shape(format!(
                "tops check: well {} needs n + 2 boundaries and sources and n + 1 states",
                b.well.0
            )));
        }
        if nb - 2 > usize::from(u16::MAX) + 1 {
            return Err(SxError::Shape(format!(
                "tops check: well {} has more tops than u16 holds",
                b.well.0
            )));
        }
    }
    Ok(())
}

/// Top `j` can be checked in this well: intervals `j` and
/// `j + 1` are neither `Pinched` nor `Unknown`, boundaries `j ..= j + 2` are
/// finite, and the top lies inside the well's logged MD range `cov` (a top
/// below or above the logs has no curve to match).
fn checkable(b: &WellBoundaries, j: usize, cov: Option<(f64, f64)>) -> bool {
    let state_ok = |i: usize| {
        b.state
            .get(i)
            .is_some_and(|s| !matches!(s, IntervalState::Pinched | IntervalState::Unknown))
    };
    let finite = |i: usize| b.md_ft.get(i).is_some_and(|v| v.is_finite());
    let covered = b
        .md_ft
        .get(j + 1)
        .is_some_and(|&m| cov.is_some_and(|(lo, hi)| m >= lo && m <= hi));
    state_ok(j) && state_ok(j + 1) && finite(j) && finite(j + 1) && finite(j + 2) && covered
}

/// MD range (ft) of samples where any log is finite; `None` without data.
fn logged_range(logs: Option<&WellLogs>) -> Option<(f64, f64)> {
    let l = logs?;
    let any_finite = |i: usize| {
        l.curves
            .values()
            .any(|v| v.get(i).is_some_and(|x| x.is_finite()))
    };
    let first = (0..l.grid.len).find(|&i| any_finite(i))?;
    let last = (0..l.grid.len).rev().find(|&i| any_finite(i))?;
    Some((l.grid.md_ft(first), l.grid.md_ft(last)))
}

fn pos_of(bounds: &[WellBoundaries], w: WellId) -> Option<usize> {
    bounds.binary_search_by_key(&w, |b| b.well).ok()
}

/// One chosen voter: position in `bounds` and edge length (m).
#[derive(Clone, Copy, Debug)]
struct Voter {
    pos: usize,
    length_m: f64,
}

/// Voters of top `j` in well `p`: eligible neighbors
/// nearest first, user picks up to `k_check`, then other non-pad sources until
/// `min_votes`.
fn choose_voters(
    bounds: &[WellBoundaries],
    cov: &[Option<(f64, f64)>],
    nbrs: &[(WellId, f64)],
    p: usize,
    j: usize,
    cfg: &CorrelateTopsConfig,
) -> Vec<Voter> {
    let cov_at = |q: usize| cov.get(q).copied().flatten();
    if !bounds.get(p).is_some_and(|b| checkable(b, j, cov_at(p))) {
        return Vec::new();
    }
    let src = |q: usize| bounds.get(q).and_then(|b| b.source.get(j + 1)).copied();
    let eligible: Vec<Voter> = nbrs
        .iter()
        .filter_map(|&(id, length_m)| {
            let pos = pos_of(bounds, id)?;
            bounds.get(pos).filter(|b| checkable(b, j, cov_at(pos)))?;
            Some(Voter { pos, length_m })
        })
        .collect();
    let mut chosen: Vec<Voter> = eligible
        .iter()
        .filter(|v| src(v.pos) == Some(TopSource::User))
        .take(cfg.k_check)
        .copied()
        .collect();
    if chosen.len() < cfg.min_votes {
        for v in eligible
            .iter()
            .filter(|v| !matches!(src(v.pos), Some(TopSource::User | TopSource::Pad) | None))
        {
            if chosen.len() >= cfg.min_votes {
                break;
            }
            chosen.push(*v);
        }
    }
    chosen
}

/// Map a voter's top MD into the other well through a window path. NaN when a needed pair is missing (the voter is dropped).
///
/// `q = md / 0.5`, `k0 = floor(q)`, `t = q − k0`; the first pair whose voter
/// side is `k0` gives `k1`, and for `t > 0` the first pair at `k0 + 1` gives
/// `k2`; result `0.5 · ((1 − t) k1 + t k2)` ft. Relies on the `PairPath`
/// contract (pairs non-decreasing in both wells) to find the first pair by
/// binary search.
pub(crate) fn map_through_path(path: &PairPath, voter_is_a: bool, md_ft: f64) -> f64 {
    let q = md_ft / GRID_STEP_FT;
    if !q.is_finite() {
        return f64::NAN;
    }
    let k0f = q.floor();
    if k0f < f64::from(i32::MIN) || k0f + 1.0 > f64::from(i32::MAX) {
        return f64::NAN;
    }
    let t = q - k0f;
    let k0 = k0f as i32;
    let side = |p: &(i32, i32)| if voter_is_a { p.0 } else { p.1 };
    let other = |k: i32| -> Option<i32> {
        let i = path.path.partition_point(|p| side(p) < k);
        let pair = path.path.get(i)?;
        (side(pair) == k).then_some(if voter_is_a { pair.1 } else { pair.0 })
    };
    let Some(k1) = other(k0) else {
        return f64::NAN;
    };
    if t > 0.0 {
        let Some(k2) = other(k0 + 1) else {
            return f64::NAN;
        };
        GRID_STEP_FT * ((1.0 - t) * f64::from(k1) + t * f64::from(k2))
    } else {
        GRID_STEP_FT * f64::from(k1)
    }
}

/// Weighted median of `(value, weight, tie id)`.
///
/// Sort by (value, id) ascending; `W = Σ w` (weight 1 for all when `W` is not
/// finite or `≤ 0`); the first value whose cumulative weight reaches
/// `0.5 W`. Empty input gives NaN.
pub(crate) fn weighted_median(values: &[(f64, f64, WellId)]) -> f64 {
    let mut v: Vec<(f64, f64, WellId)> = values.to_vec();
    v.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.2.cmp(&y.2)));
    let total: f64 = v.iter().map(|x| x.1).sum();
    let unit = !total.is_finite() || total <= 0.0;
    let half = 0.5 * if unit { v.len() as f64 } else { total };
    let mut cum = 0.0;
    for &(z, w, _) in &v {
        cum += if unit { 1.0 } else { w };
        if cum >= half {
            return z;
        }
    }
    v.last().map_or(f64::NAN, |x| x.0)
}

/// `(z*, s)`: weighted median and `MAD_TO_SIGMA · median(|z_b − z*|)`.
fn vote(zs: &[(f64, f64, WellId)]) -> (f64, f64) {
    let z_star = weighted_median(zs);
    let dev: Vec<f64> = zs.iter().map(|z| (z.0 - z_star).abs()).collect();
    (z_star, MAD_TO_SIGMA * median(&dev))
}

/// Suspect rule for a user pick `p` with `votes ≥ min_votes`:
/// `|Δ| > max(tol_ft, spread_mult · s)` and the share of `z_b`
/// strictly on the same side of `p` as `z*` is `≥ side_share`.
fn is_suspect(p: f64, z_star: f64, s: f64, zs: &[f64], cfg: &CorrelateTopsConfig) -> bool {
    let d = p - z_star;
    let threshold = cfg.tol_ft.max(cfg.spread_mult * s);
    let beyond = d.abs() > threshold;
    if !beyond || zs.is_empty() {
        return false;
    }
    let same = zs
        .iter()
        .filter(|&&z| if z_star > p { z > p } else { z < p })
        .count();
    same as f64 / zs.len() as f64 >= cfg.side_share
}

/// Rank score `|Δ| / max(s, 1.0)`.
fn rank_score(offset_ft: f64, spread_ft: f64) -> f64 {
    offset_ft.abs() / spread_ft.max(RANK_SPREAD_FLOOR_FT)
}

fn check_state(can_check: bool, votes: usize, min_votes: usize) -> CheckState {
    if !can_check {
        CheckState::NotChecked
    } else if votes < min_votes {
        CheckState::TooFewVotes
    } else {
        CheckState::Checked
    }
}

/// Window path of one (pair, top) reduced to what the votes read.
#[derive(Clone, Copy, Debug)]
struct WindowMap {
    quality: f32,
    /// Lower-id well's top mapped into the higher-id well, ft (NaN = dropped).
    lo_into_hi: f64,
    /// Higher-id well's top mapped into the lower-id well, ft (NaN = dropped).
    hi_into_lo: f64,
}

/// (lower bounds position, higher bounds position, top `j`).
type WindowKey = (usize, usize, usize);

/// Check row of top `j` in well `p` before rank and order.
fn vote_row(
    bounds: &[WellBoundaries],
    cov: &[Option<(f64, f64)>],
    graph: &WellGraph,
    windows: &BTreeMap<WindowKey, Option<WindowMap>>,
    cfg: &CorrelateTopsConfig,
    (p, j): (usize, usize),
    voters: &[Voter],
) -> SxResult<TopCheckRow> {
    let b = bounds
        .get(p)
        .ok_or_else(|| SxError::NotFound(format!("tops check: boundaries at {p}")))?;
    let top = u16::try_from(j)
        .map_err(|_| SxError::Shape(format!("tops check: top index {j} exceeds u16")))?;
    let mut zs: Vec<(f64, f64, WellId)> = Vec::with_capacity(voters.len());
    for v in voters {
        let Some(vb) = bounds.get(v.pos) else {
            continue;
        };
        let Some(Some(m)) = windows.get(&(p.min(v.pos), p.max(v.pos), j)) else {
            continue;
        };
        let z = if v.pos < p {
            m.lo_into_hi
        } else {
            m.hi_into_lo
        };
        if z.is_nan() {
            continue;
        }
        zs.push((z, graph.edge_weight(v.length_m, m.quality), vb.well));
    }
    let votes = zs.len();
    let (z_star, spread) = if votes >= 1 {
        vote(&zs)
    } else {
        (f64::NAN, f64::NAN)
    };
    let user = b.source.get(j + 1) == Some(&TopSource::User);
    let pick = b.md_ft.get(j + 1).copied().unwrap_or(f64::NAN);
    let offset_ft = if user && votes >= 1 {
        pick - z_star
    } else {
        f64::NAN
    };
    let z_only: Vec<f64> = zs.iter().map(|z| z.0).collect();
    let suspect = user && votes >= cfg.min_votes && is_suspect(pick, z_star, spread, &z_only, cfg);
    Ok(TopCheckRow {
        well: b.well,
        top,
        state: check_state(
            checkable(b, j, cov.get(p).copied().flatten()),
            votes,
            cfg.min_votes,
        ),
        votes: u32::try_from(votes)
            .map_err(|_| SxError::InvalidInput(format!("vote count {votes} exceeds u32")))?,
        z_star_md_ft: z_star,
        spread_ft: spread,
        offset_ft,
        suspect,
        rank: 0,
        moved: false,
        clamped: false,
    })
}

/// Order-constrained boundaries of one well.
#[derive(Clone, Debug)]
pub(crate) struct Ordered {
    /// Boundary MD with moved values written in, ft.
    pub md_ft: Vec<f64>,
    /// Target fits and moves.
    pub moved: Vec<bool>,
    /// Target was clamped or had no room.
    pub clamped: Vec<bool>,
}

/// Place the targeted boundaries of one well between the fixed ones.
///
/// `target[b] = Some(z)` marks boundary `b` as re-estimated to `z`; every
/// other boundary is fixed. Targets come in runs between the nearest finite
/// fixed boundaries `F_lo` / `F_hi` (`∓∞` when none). A run of `r` targets
/// with `F_hi − F_lo < 0.5 (r + 1)` does not move (all clamped); otherwise
/// clamp into `(F_lo, F_hi)` by 0.5 ft, forward pass `v_m ≥ v_{m−1} + 0.5`,
/// backward pass from `F_hi − 0.5`; changed values are clamped, all targets
/// of the run move.
pub(crate) fn order_boundaries(md_ft: &[f64], target: &[Option<f64>]) -> Ordered {
    let n = md_ft.len();
    let mut out = Ordered {
        md_ft: md_ft.to_vec(),
        moved: vec![false; n],
        clamped: vec![false; n],
    };
    let mut run: Vec<(usize, f64)> = Vec::new();
    let mut f_lo = f64::NEG_INFINITY;
    for (b, &md) in md_ft.iter().enumerate() {
        if let Some(z) = target.get(b).copied().flatten() {
            run.push((b, z));
        } else if md.is_finite() {
            place_run(&mut out, &run, f_lo, md);
            run.clear();
            f_lo = md;
        }
    }
    place_run(&mut out, &run, f_lo, f64::INFINITY);
    out
}

fn place_run(out: &mut Ordered, run: &[(usize, f64)], f_lo: f64, f_hi: f64) {
    let r = run.len();
    if r == 0 {
        return;
    }
    if f_hi - f_lo < ORDER_GAP_FT * (r + 1) as f64 {
        for &(b, _) in run {
            if let Some(c) = out.clamped.get_mut(b) {
                *c = true;
            }
        }
        return;
    }
    let mut v: Vec<f64> = run
        .iter()
        .map(|&(_, z)| {
            if z <= f_lo {
                f_lo + ORDER_GAP_FT
            } else if z >= f_hi {
                f_hi - ORDER_GAP_FT
            } else {
                z
            }
        })
        .collect();
    for m in 1..r {
        v[m] = v[m].max(v[m - 1] + ORDER_GAP_FT);
    }
    if let Some(last) = v.last_mut() {
        if *last >= f_hi {
            *last = f_hi - ORDER_GAP_FT;
        }
    }
    for m in (0..r - 1).rev() {
        v[m] = v[m].min(v[m + 1] - ORDER_GAP_FT);
    }
    for (&(b, z), &nv) in run.iter().zip(&v) {
        if let Some(slot) = out.md_ft.get_mut(b) {
            *slot = nv;
        }
        if let Some(slot) = out.moved.get_mut(b) {
            *slot = true;
        }
        if let Some(slot) = out.clamped.get_mut(b) {
            *slot = nv != z;
        }
    }
}

/// Window check of every top against graph neighbors.
///
/// Per (well, top): voters are eligible graph neighbors (nearest first, user
/// picks up to `k_check`, then other non-pad sources up to `min_votes`; the
/// well's own pick never votes). Each voter's top is mapped into the well
/// through `dtw::window_path` over boundaries `j ..= j + 2` of both wells
/// (lower id first, `dtw.window` settings). `z*` is the weighted median
/// (weight `graph.edge_weight(length, quality)`), `s = 1.4826 · median|z − z*|`.
/// User picks far from `z*` with a one-sided vote are suspect and ranked.
/// Estimated/suggested tops with votes and suspect user tops are re-estimated
/// to `z*` under the order constraint; `boundaries` are the Pass 1 boundaries
/// with moved values written in (sources and states unchanged).
///
/// **Choice:** Window paths are computed once per (pair, top), in
/// parallel, and kept only as the two mapped depths and the quality; results
/// equal uncached calls. A DTW error drops the voters of that (pair, top)
/// (logged with `debug!`).
///
/// Units: ft, m. NaN rule: NaN boundaries make a top `NotChecked`; NaN
/// floats in a row mean "not computed". Errors: [`SxError::Config`] (bad
/// `[correlate.tops]` keys), [`SxError::Shape`] (bounds not sorted by well or
/// inconsistent lengths), [`SxError::NotFound`] (a voting well missing from
/// `logs`). Empty `bounds` gives no rows. Complexity:
/// `O(wells × tops × k_check)` plus one window DTW per (pair, top).
pub fn check_tops(
    graph: &WellGraph,
    logs: &[WellLogs],
    bounds: &[WellBoundaries],
    cfg: &CorrelateTopsConfig,
    dtw: &DtwConfig,
) -> SxResult<TopsCheck> {
    check_tops_config(cfg)?;
    check_bounds_layout(bounds)?;

    let nbrs: Vec<Vec<(WellId, f64)>> = bounds
        .par_iter()
        .map(|b| graph.neighbors_by_distance(b.well))
        .collect();
    let mut logs_map: BTreeMap<WellId, &WellLogs> = BTreeMap::new();
    for l in logs {
        logs_map.entry(l.id).or_insert(l);
    }
    let cov: Vec<Option<(f64, f64)>> = bounds
        .iter()
        .map(|b| logged_range(logs_map.get(&b.well).copied()))
        .collect();
    let tasks: Vec<(usize, usize)> = bounds
        .iter()
        .enumerate()
        .flat_map(|(p, b)| (0..b.md_ft.len().saturating_sub(2)).map(move |j| (p, j)))
        .collect();
    let voters: Vec<Vec<Voter>> = tasks
        .par_iter()
        .map(|&(p, j)| {
            let nb = nbrs.get(p).map_or(&[][..], Vec::as_slice);
            choose_voters(bounds, &cov, nb, p, j, cfg)
        })
        .collect();

    // One window path per (pair, top).
    let mut key_set: BTreeSet<WindowKey> = BTreeSet::new();
    for (&(p, j), vs) in tasks.iter().zip(&voters) {
        for v in vs {
            key_set.insert((p.min(v.pos), p.max(v.pos), j));
        }
    }
    let keys: Vec<WindowKey> = key_set.into_iter().collect();
    let side = |q: usize, j: usize| -> SxResult<(SpanSide<'_>, f64)> {
        let b = bounds
            .get(q)
            .ok_or_else(|| SxError::NotFound(format!("tops check: boundaries at {q}")))?;
        let l = logs_map
            .get(&b.well)
            .copied()
            .ok_or_else(|| SxError::NotFound(format!("logs of well {}", b.well.0)))?;
        let md = |i: usize| b.md_ft.get(i).copied().unwrap_or(f64::NAN);
        Ok((
            SpanSide {
                logs: l,
                top_md_ft: md(j),
                base_md_ft: md(j + 2),
            },
            md(j + 1),
        ))
    };
    let mut jobs = Vec::with_capacity(keys.len());
    for &(lo, hi, j) in &keys {
        jobs.push((side(lo, j)?, side(hi, j)?));
    }
    let maps: Vec<Option<WindowMap>> = jobs
        .par_iter()
        .zip(keys.par_iter())
        .map(
            |(((a, top_a), (b, top_b)), &(_, _, j))| match window_path(a, b, &dtw.window) {
                Ok(path) => Some(WindowMap {
                    quality: path.quality,
                    lo_into_hi: map_through_path(&path, true, *top_a),
                    hi_into_lo: map_through_path(&path, false, *top_b),
                }),
                Err(err) => {
                    tracing::debug!(
                        well_a = a.logs.id.0,
                        well_b = b.logs.id.0,
                        top = j,
                        %err,
                        "window DTW failed, voter dropped"
                    );
                    None
                }
            },
        )
        .collect();
    let windows: BTreeMap<WindowKey, Option<WindowMap>> = keys.into_iter().zip(maps).collect();

    let mut rows: Vec<TopCheckRow> = tasks
        .par_iter()
        .zip(voters.par_iter())
        .map(|(&task, vs)| vote_row(bounds, &cov, graph, &windows, cfg, task, vs))
        .collect::<SxResult<_>>()?;

    // Rank: score descending, then (well, top) = row order.
    let mut order: Vec<(f64, usize)> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.suspect)
        .map(|(k, r)| (rank_score(r.offset_ft, r.spread_ft), k))
        .collect();
    order.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    for (rank, &(_, k)) in order.iter().enumerate() {
        if let Some(r) = rows.get_mut(k) {
            r.rank = u32::try_from(rank + 1)
                .map_err(|_| SxError::InvalidInput("suspect rank exceeds u32".into()))?;
        }
    }

    // Order constraint per well; rows of well `p` are contiguous.
    let mut boundaries = bounds.to_vec();
    let mut start = 0usize;
    for (b, out_b) in bounds.iter().zip(boundaries.iter_mut()) {
        let n = b.md_ft.len().saturating_sub(2);
        let Some(well_rows) = rows.get_mut(start..start + n) else {
            return Err(SxError::Shape(
                "tops check: rows do not cover the boundaries".into(),
            ));
        };
        start += n;
        let mut target: Vec<Option<f64>> = vec![None; b.md_ft.len()];
        for (j, r) in well_rows.iter().enumerate() {
            let re_estimate = matches!(
                b.source.get(j + 1),
                Some(TopSource::Estimated | TopSource::Suggested)
            ) && r.votes >= 1;
            if (re_estimate || r.suspect) && r.z_star_md_ft.is_finite() {
                if let Some(t) = target.get_mut(j + 1) {
                    *t = Some(r.z_star_md_ft);
                }
            }
        }
        let ord = order_boundaries(&b.md_ft, &target);
        for (j, r) in well_rows.iter_mut().enumerate() {
            r.moved = ord.moved.get(j + 1).copied().unwrap_or(false);
            r.clamped = ord.clamped.get(j + 1).copied().unwrap_or(false);
        }
        out_b.md_ft = ord.md_ft;
    }
    tracing::info!(
        rows = rows.len(),
        suspects = order.len(),
        moved = rows.iter().filter(|r| r.moved).count(),
        "tops check done"
    );
    Ok(TopsCheck { rows, boundaries })
}

/// Elevation (ft) at an MD, linear between grid samples. Past either end of
/// the geometry grid it continues along the end segment (a pick below the
/// logs still gets an elevation). NaN for NaN MD or an empty grid.
///
/// Inside: `q = md / 0.5`, `k0 = floor(q)`, `t = q − k0`,
/// `elev = (1 − t) e[k0] + t e[k0 + 1]`. NaN rule: propagate. Errors: none.
/// Complexity: `O(1)`.
pub fn elev_at_md(geometry: &WellGeometry, md_ft: f64) -> f64 {
    let n = geometry.grid.len;
    let q = md_ft / GRID_STEP_FT;
    if !q.is_finite() || n == 0 {
        return f64::NAN;
    }
    let md = |i: usize| geometry.grid.md_ft(i);
    let e = |i: usize| geometry.elev_ft.get(i).copied().unwrap_or(f64::NAN);
    // Past either end, continue along the end segment (vertical if one sample).
    let slope = |a: usize, b: usize| {
        if a == b {
            -1.0
        } else {
            (e(b) - e(a)) / (md(b) - md(a))
        }
    };
    if md_ft <= md(0) {
        return e(0) + (md_ft - md(0)) * slope(0, 1.min(n - 1));
    }
    if md_ft >= md(n - 1) {
        let a = n.saturating_sub(2);
        return e(n - 1) + (md_ft - md(n - 1)) * slope(a, n - 1);
    }
    let k0f = q.floor();
    if k0f < f64::from(i32::MIN) || k0f > f64::from(i32::MAX) {
        return f64::NAN;
    }
    let t = q - k0f;
    let Some(i0) = geometry.grid.index_of_k(k0f as i32) else {
        return f64::NAN;
    };
    if t > 0.0 && i0 + 1 < n {
        (1.0 - t) * e(i0) + t * e(i0 + 1)
    } else {
        e(i0)
    }
}

/// Build the `tops` table after Pass 2.
///
/// One row per (well, top), wells in header order, tops `0..n`. Pass 1
/// source `User` → `md_ft` is the Pass 1 boundary (the user pick, untouched)
/// and `source = User`; otherwise `md_ft` is the Pass 2 boundary and
/// `source = Suggested`. Check fields copy the [`TopCheckRow`];
/// `suggested_md_ft = z*` before clamping; `offset_ft` is NaN for non-user
/// rows.
///
/// Units: ft. NaN rule: a NaN boundary gives NaN `md_ft` and `elev_ft`.
/// Errors: [`SxError::Shape`] when headers, geometry, check rows and both
/// boundary sets do not cover the same wells in the same order, or a well
/// does not have `n + 2` boundaries and `n` check rows. Empty headers give an
/// empty table. Complexity: `O(wells × tops)`.
pub fn build_tops_table(
    headers: &[WellHeader],
    geometry: &[WellGeometry],
    tops: &TopSet,
    check: &TopsCheck,
    bounds_pass1: &[WellBoundaries],
    bounds_pass2: &[WellBoundaries],
) -> SxResult<Vec<TopRow>> {
    let n = tops.len();
    let nw = headers.len();
    let shape = |what: &str| SxError::Shape(format!("tops table: {what}"));
    if geometry.len() != nw
        || bounds_pass1.len() != nw
        || bounds_pass2.len() != nw
        || nw.checked_mul(n) != Some(check.rows.len())
    {
        return Err(shape(
            "headers, geometry, check rows and boundaries cover different wells",
        ));
    }
    let mut out = Vec::with_capacity(check.rows.len());
    for (wi, (((h, g), b1), b2)) in headers
        .iter()
        .zip(geometry)
        .zip(bounds_pass1)
        .zip(bounds_pass2)
        .enumerate()
    {
        if g.id != h.id || b1.well != h.id || b2.well != h.id {
            return Err(shape("wells differ in id or order"));
        }
        if b1.md_ft.len() != n + 2 || b1.source.len() != n + 2 || b2.md_ft.len() != n + 2 {
            return Err(shape(&format!("well {} needs n + 2 boundaries", h.id.0)));
        }
        for (j, name) in tops.names.iter().enumerate() {
            let row = check
                .rows
                .get(wi * n + j)
                .filter(|r| r.well == h.id && usize::from(r.top) == j)
                .ok_or_else(|| shape(&format!("well {}: no check row for top {j}", h.id.0)))?;
            let user = b1.source.get(j + 1) == Some(&TopSource::User);
            let (md_ft, source) = if user {
                (
                    b1.md_ft.get(j + 1).copied().unwrap_or(f64::NAN),
                    TopSource::User,
                )
            } else {
                (
                    b2.md_ft.get(j + 1).copied().unwrap_or(f64::NAN),
                    TopSource::Suggested,
                )
            };
            out.push(TopRow {
                well: h.id,
                well_name: h.name.clone(),
                top: row.top,
                top_name: name.clone(),
                md_ft,
                elev_ft: elev_at_md(g, md_ft),
                source,
                state: row.state,
                suspect: row.suspect,
                suggested_md_ft: row.z_star_md_ft,
                offset_ft: if user { row.offset_ft } else { f64::NAN },
                votes: row.votes,
                spread_ft: row.spread_ft,
                rank: row.rank,
                clamped: row.clamped,
            });
        }
    }
    Ok(out)
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

fn u32_to_i32(v: u32, what: &str) -> SxResult<i32> {
    i32::try_from(v).map_err(|_| SxError::InvalidInput(format!("{what} {v} exceeds int32")))
}

fn rows_batch(rows: &[&TopRow], tag: &str) -> SxResult<RecordBatch> {
    let source: Vec<&str> = rows
        .iter()
        .map(|r| match r.source {
            TopSource::User => Ok("user"),
            TopSource::Suggested => Ok("suggested"),
            other => Err(SxError::InvalidInput(format!(
                "tops table: well {} top {} has source {other:?}; rows are user or suggested",
                r.well.0, r.top
            ))),
        })
        .collect::<SxResult<_>>()?;
    let votes: Vec<i32> = rows
        .iter()
        .map(|r| u32_to_i32(r.votes, "votes"))
        .collect::<SxResult<_>>()?;
    let rank: Vec<i32> = rows
        .iter()
        .map(|r| u32_to_i32(r.rank, "rank"))
        .collect::<SxResult<_>>()?;
    let f64s = |f: &dyn Fn(&TopRow) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| f(r))))
    };
    let bools = |f: &dyn Fn(&TopRow) -> bool| -> ArrayRef {
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| f(r)).collect::<Vec<_>>(),
        ))
    };
    let cols: Vec<(&str, ArrayRef)> = vec![
        (
            "well_id",
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.well.0))),
        ),
        (
            "well_name",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.well_name.as_str()),
            )),
        ),
        (
            "top_index",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| i32::from(r.top)),
            )),
        ),
        (
            "top_name",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.top_name.as_str()),
            )),
        ),
        ("md_ft", f64s(&|r| r.md_ft)),
        ("elev_ft", f64s(&|r| r.elev_ft)),
        ("source", Arc::new(StringArray::from_iter_values(source))),
        (
            "check_state",
            Arc::new(StringArray::from_iter_values(rows.iter().map(
                |r| match r.state {
                    CheckState::Checked => "checked",
                    CheckState::NotChecked => "not_checked",
                    CheckState::TooFewVotes => "too_few_votes",
                },
            ))),
        ),
        ("suspect", bools(&|r| r.suspect)),
        ("suggested_md_ft", f64s(&|r| r.suggested_md_ft)),
        ("offset_ft", f64s(&|r| r.offset_ft)),
        ("votes", Arc::new(Int32Array::from(votes))),
        ("spread_ft", f64s(&|r| r.spread_ft)),
        ("rank", Arc::new(Int32Array::from(rank))),
        ("clamped", bools(&|r| r.clamped)),
    ];
    finish(cols, tag)
}

/// `tops` table (Parquet and CSV share it).
///
/// Rows in input order (from [`build_tops_table`]: sorted by (well, top)).
/// Columns `well_id` int32, `well_name` utf8, `top_index` int32, `top_name`
/// utf8, `md_ft`, `elev_ft` float64, `source` utf8 (`user` / `suggested`),
/// `check_state` utf8 (`checked` / `not_checked` / `too_few_votes`),
/// `suspect` bool, `suggested_md_ft`, `offset_ft` float64, `votes` int32,
/// `spread_ft` float64, `rank` int32, `clamped` bool. Schema tag
/// `subsurfix.tops/1`.
///
/// NaN rule: NaN floats are stored as is. Errors: [`SxError::InvalidInput`]
/// (a source other than `User` / `Suggested`, a count above int32, batch
/// build). Empty rows give a header-only batch. Complexity: `O(rows)`.
pub fn tops_batch(rows: &[TopRow]) -> SxResult<arrow::record_batch::RecordBatch> {
    let refs: Vec<&TopRow> = rows.iter().collect();
    rows_batch(&refs, TOPS_TAG)
}

/// `suspects` table: suspect rows only, sorted by rank.
///
/// Same columns as [`tops_batch`]; ties in rank keep (well, top) order.
/// Schema tag `subsurfix.suspects/1`. NaN rule and errors as
/// [`tops_batch`]. No suspect gives a header-only batch. Complexity:
/// `O(rows log rows)`.
pub fn suspects_batch(rows: &[TopRow]) -> SxResult<arrow::record_batch::RecordBatch> {
    let mut refs: Vec<&TopRow> = rows.iter().filter(|r| r.suspect).collect();
    refs.sort_by_key(|r| (r.rank, r.well, r.top));
    rows_batch(&refs, SUSPECTS_TAG)
}

#[cfg(test)]
mod tests {
    use arrow::array::AsArray;
    use arrow::datatypes::Float64Type;

    use super::*;
    use crate::types::GridRange;

    #[test]
    fn elevation_continues_past_the_log_grid() {
        // Vertical well, KB 2000 ft, grid 1000.0..=1002.0 ft MD.
        let grid = GridRange::new(2000, 5).unwrap();
        let md: Vec<f64> = (0..5).map(|i| grid.md_ft(i)).collect();
        let g = WellGeometry {
            id: WellId(1),
            grid,
            tvd_ft: md.clone(),
            elev_ft: md.iter().map(|m| 2000.0 - m).collect(),
            x_m: vec![0.0; 5],
            y_m: vec![0.0; 5],
        };
        close(elev_at_md(&g, 1001.25), 998.75, 1e-12);
        close(elev_at_md(&g, 1302.0), 698.0, 1e-12);
        close(elev_at_md(&g, 900.0), 1100.0, 1e-12);
    }

    #[test]
    fn top_outside_the_logs_is_not_checkable() {
        let b = WellBoundaries {
            well: WellId(1),
            md_ft: vec![800.0, 1000.0, 1300.0, 1500.0],
            source: vec![
                TopSource::Pad,
                TopSource::User,
                TopSource::User,
                TopSource::Pad,
            ],
            state: vec![
                IntervalState::Pad,
                IntervalState::Observed,
                IntervalState::Pad,
            ],
        };
        assert!(checkable(&b, 0, Some((900.0, 1200.0))));
        assert!(!checkable(&b, 1, Some((900.0, 1200.0))));
        assert!(!checkable(&b, 0, None));
    }

    fn close(got: f64, want: f64, rel: f64) {
        assert!(
            (got - want).abs() <= rel + rel * want.abs(),
            "got {got}, want {want}"
        );
    }

    fn votes(z: &[f64], w: &[f64]) -> Vec<(f64, f64, WellId)> {
        z.iter()
            .zip(w)
            .enumerate()
            .map(|(k, (&z, &w))| (z, w, WellId(k as i32 + 1)))
            .collect()
    }

    /// Worked example: user pick 5,000 ft, five equal-weight voters.
    #[test]
    fn correlate_tops_suspect_example() {
        let cfg = CorrelateTopsConfig::default();
        let zs = votes(&[5021.0, 5018.0, 5023.0, 5020.0, 4998.0], &[1.0; 5]);
        let (z_star, s) = vote(&zs);
        assert_eq!(z_star, 5020.0);
        close(s, 2.9652, 1e-4);
        let z_only: Vec<f64> = zs.iter().map(|z| z.0).collect();
        assert!(is_suspect(5000.0, z_star, s, &z_only, &cfg));
        assert_eq!(5000.0 - z_star, -20.0);
        close(rank_score(5000.0 - z_star, s), 6.745, 1e-4);
    }

    /// Weights 0.1 / 0.2 / 0.5 put `z*` at the deepest value.
    #[test]
    fn correlate_tops_weighted_median_example() {
        let zs = votes(&[5010.0, 5020.0, 5030.0], &[0.1, 0.2, 0.5]);
        assert_eq!(weighted_median(&zs), 5030.0);
    }

    /// Two votes on an estimated top, too few votes but still moved.
    #[test]
    fn correlate_tops_two_vote_estimated_top() {
        let zs = votes(&[5004.0, 5012.0], &[1.0, 1.0]);
        let (z_star, s) = vote(&zs);
        assert_eq!(z_star, 5004.0);
        close(s, 5.9304, 1e-4);
        assert_eq!(check_state(true, 2, 3), CheckState::TooFewVotes);
        let ord = order_boundaries(&[4000.0, 5010.0, 6000.0], &[None, Some(z_star), None]);
        assert_eq!(ord.md_ft, vec![4000.0, 5004.0, 6000.0]);
        assert_eq!(ord.moved, vec![false, true, false]);
        assert_eq!(ord.clamped, vec![false, false, false]);
    }

    /// A suspect user pick keeps its MD in the tops and suspects tables even
    /// though its Pass 2 boundary moved to `z*`.
    #[test]
    fn correlate_tops_user_pick_never_moved() {
        let well = WellId(7);
        let grid = GridRange::new(9000, 4000).unwrap();
        let md: Vec<f64> = (0..grid.len).map(|i| grid.md_ft(i)).collect();
        let headers = vec![WellHeader {
            id: well,
            name: "W7".into(),
            x_m: 0.0,
            y_m: 0.0,
            kb_elev_ft: 2000.0,
        }];
        let geometry = vec![WellGeometry {
            id: well,
            grid,
            tvd_ft: md.clone(),
            elev_ft: md.iter().map(|m| 2000.0 - m).collect(),
            x_m: vec![0.0; grid.len],
            y_m: vec![0.0; grid.len],
        }];
        let tops = TopSet::new(vec!["A".into(), "B".into()]).unwrap();
        let b1 = WellBoundaries {
            well,
            md_ft: vec![4600.0, 5000.0, 5500.0, 6400.0],
            source: vec![
                TopSource::Pad,
                TopSource::User,
                TopSource::Estimated,
                TopSource::Pad,
            ],
            state: vec![
                IntervalState::Pad,
                IntervalState::Estimated,
                IntervalState::Pad,
            ],
        };
        let mut b2 = b1.clone();
        b2.md_ft[1] = 5020.0;
        b2.md_ft[2] = 5504.0;
        let row = |top: u16, z: f64, offset: f64, suspect: bool| TopCheckRow {
            well,
            top,
            state: CheckState::Checked,
            votes: 5,
            z_star_md_ft: z,
            spread_ft: 2.9652,
            offset_ft: offset,
            suspect,
            rank: u32::from(suspect),
            moved: true,
            clamped: false,
        };
        let check = TopsCheck {
            rows: vec![row(0, 5020.0, -20.0, true), row(1, 5504.0, f64::NAN, false)],
            boundaries: vec![b2.clone()],
        };
        let rows = build_tops_table(&headers, &geometry, &tops, &check, &[b1], &[b2]).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].md_ft, 5000.0);
        assert_eq!(rows[0].source, TopSource::User);
        assert_eq!(rows[0].elev_ft, -3000.0);
        assert_eq!(rows[0].suggested_md_ft, 5020.0);
        assert_eq!(rows[0].offset_ft, -20.0);
        assert!(rows[0].suspect && rows[0].rank == 1);
        assert_eq!(rows[1].md_ft, 5504.0);
        assert_eq!(rows[1].source, TopSource::Suggested);
        assert!(rows[1].offset_ft.is_nan());

        let t = tops_batch(&rows).unwrap();
        let md_col = t.column(4).as_primitive::<Float64Type>();
        assert_eq!(md_col.values().to_vec(), vec![5000.0, 5504.0]);
        assert_eq!(t.column(6).as_string::<i32>().value(0), "user");
        let s = suspects_batch(&rows).unwrap();
        assert_eq!(s.num_rows(), 1);
        assert_eq!(s.column(4).as_primitive::<Float64Type>().value(0), 5000.0);
    }
}
