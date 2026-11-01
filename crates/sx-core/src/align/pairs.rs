// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Pair selection, interval mode, depth mode and pair-map assembly.
//!
//! Nearest-well search: brute force over all candidates with a
//! partial selection, ordered by `(distance, WellId)`, so ties follow the
//! brute-force definition exactly.

use rayon::prelude::*;
use tracing::debug;

use crate::align::depth::{elev_at_md, md_at_elev};
use crate::align::intervals::IntervalInfo;
use crate::config::{AlignParams, PairParams};
use crate::dtw::{
    band_half_width, banded_dtw, interval_path, interval_plan, pairs_from_md, resample_curve,
    rescale_unit, row_mean_map, shared_nodes, unit_cost, working_values, IntervalMatch,
    IntervalParams, IntervalSide, MatchMethod,
};
use crate::error::SxResult;
use crate::types::{IntervalState, LogKind, PairPath, WellGeometry, WellHeader, WellId, WellLogs};

/// Status of a directed pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PairStatus {
    /// Interval mode with at least one accepted interval.
    IntervalMap = 0,
    /// Depth mode (a well without a layer coordinate).
    DepthMap = 1,
    /// No map.
    Failed = 2,
}

/// Match map from a target well (a) to one of its nearest donors (b).
#[derive(Debug, Clone, PartialEq)]
pub struct PairMap {
    /// a = target, b = donor, `interval = None`; `path` holds the assembled
    /// grid pairs (`k_a` strictly increasing, `k_b` non-decreasing; empty when
    /// failed); `cost` = total DTW cost (NaN when no DTW ran); `quality` per
    pub path: PairPath,
    /// Rank of the donor, 1 = nearest.
    pub rank: u16,
    /// Surface distance, m.
    pub distance_m: f64,
    /// Interval map, depth map or failed.
    pub status: PairStatus,
    /// Node-weighted mean thickness ratio donor / target; 1.0 for a depth map,
    /// NaN when failed.
    pub span_ratio: f64,
    /// Node-weighted mean of the DTW intervals' mean cost; NaN when no DTW ran
    ///
    pub mean_cost: f64,
    /// Accepted intervals matched by DTW.
    pub n_dtw_intervals: u16,
    /// Accepted intervals matched linearly.
    pub n_linear_intervals: u16,
    /// Accepted intervals, ascending.
    pub intervals: Vec<u16>,
    /// Same length as `intervals`: method factor × sqrt(confidence product).
    pub interval_quality: Vec<f32>,
    /// Same length as `intervals`: 0 = DTW, 1 = linear.
    pub interval_method: Vec<u8>,
}

/// One well as seen by the pair stage.
pub(crate) struct PairWell<'a> {
    pub header: &'a WellHeader,
    pub geom: &'a WellGeometry,
    pub logs: &'a WellLogs,
    /// Layer coordinate on `logs.grid` when the well has one.
    pub coord: Option<&'a [f64]>,
    /// Interval table of the well, interval `0..=n`.
    pub intervals: &'a [IntervalInfo],
}

/// Euclidean surface distance, m.
pub(crate) fn distance_m(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let (dx, dy) = (x1 - x0, y1 - y0);
    (dx * dx + dy * dy).sqrt()
}

/// The `k` items with the smallest `(distance, id)`, sorted by distance then
/// id (brute force with a partial selection).
pub(crate) fn k_nearest<T>(mut items: Vec<(f64, i32, T)>, k: usize) -> Vec<(f64, i32, T)> {
    let order = |a: &(f64, i32, T), b: &(f64, i32, T)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    let k = k.min(items.len());
    if k == 0 {
        return Vec::new();
    }
    if k < items.len() {
        items.select_nth_unstable_by(k - 1, order);
        items.truncate(k);
    }
    items.sort_by(order);
    items
}

fn has_finite_log(logs: &WellLogs) -> bool {
    logs.curves
        .values()
        .any(|v| v.iter().any(|x| x.is_finite()))
}

/// Depth-mode axis `x(k) = −elev_at_md(0.5 k)` on the well's logs grid.
fn depth_axis(w: &PairWell<'_>) -> Vec<f64> {
    (0..w.logs.grid.len)
        .map(|i| -elev_at_md(w.header, w.geom, w.logs.grid.md_ft(i)))
        .collect()
}

/// All directed pairs, sorted by `(target, rank)`. Targets run in
/// parallel; results are collected in candidate order.
pub(crate) fn build_pairs(
    wells: &[PairWell<'_>],
    p: &AlignParams,
    dtw: &IntervalParams,
) -> SxResult<Vec<PairMap>> {
    let cands: Vec<usize> = wells
        .iter()
        .enumerate()
        .filter(|(_, w)| {
            w.header.x_m.is_finite() && w.header.y_m.is_finite() && has_finite_log(w.logs)
        })
        .map(|(i, _)| i)
        .collect();
    if cands.len() < 2 {
        return Ok(Vec::new());
    }
    let need_axis = cands
        .iter()
        .any(|&i| wells.get(i).is_some_and(|w| w.coord.is_none()));
    let axes: Vec<Vec<f64>> = if need_axis {
        wells.par_iter().map(depth_axis).collect()
    } else {
        vec![Vec::new(); wells.len()]
    };
    let per_target: Vec<SxResult<Vec<PairMap>>> = cands
        .par_iter()
        .map(|&ti| {
            let (Some(a), Some(axis_a)) = (wells.get(ti), axes.get(ti)) else {
                return Ok(Vec::new());
            };
            let others: Vec<(f64, i32, usize)> = cands
                .iter()
                .filter(|&&j| j != ti)
                .filter_map(|&j| {
                    let b = wells.get(j)?;
                    let d = distance_m(a.header.x_m, a.header.y_m, b.header.x_m, b.header.y_m);
                    Some((d, b.header.id.0, j))
                })
                .collect();
            let maps: SxResult<Vec<PairMap>> = k_nearest(others, p.k_pairs)
                .into_iter()
                .enumerate()
                .filter_map(|(r, (d, _, j))| Some((r, d, wells.get(j)?, axes.get(j)?)))
                .map(|(r, d, b, axis_b)| {
                    let rank = u16::try_from(r + 1).unwrap_or(u16::MAX);
                    align_pair(a, b, axis_a, axis_b, rank, d, p, dtw)
                })
                .collect();
            if let Ok(m) = &maps {
                debug!(
                    well = a.header.id.0,
                    n_pairs = m.len(),
                    "rough alignment: pairs"
                );
            }
            maps
        })
        .collect();
    let mut out = Vec::new();
    for maps in per_target {
        out.extend(maps?);
    }
    Ok(out)
}

/// Failed pair.
fn failed(a: WellId, b: WellId, rank: u16, distance_m: f64, quality: f32) -> PairMap {
    PairMap {
        path: PairPath {
            a,
            b,
            interval: None,
            path: Vec::new(),
            cost: f64::NAN,
            quality,
        },
        rank,
        distance_m,
        status: PairStatus::Failed,
        span_ratio: f64::NAN,
        mean_cost: f64::NAN,
        n_dtw_intervals: 0,
        n_linear_intervals: 0,
        intervals: Vec::new(),
        interval_quality: Vec::new(),
        interval_method: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn align_pair(
    a: &PairWell<'_>,
    b: &PairWell<'_>,
    axis_a: &[f64],
    axis_b: &[f64],
    rank: u16,
    distance: f64,
    p: &AlignParams,
    dtw: &IntervalParams,
) -> SxResult<PairMap> {
    let (ida, idb) = (a.header.id, b.header.id);
    match (a.coord, b.coord) {
        (Some(ca), Some(cb)) => {
            let accepted = interval_mode(a, b, ca, cb, &p.pairs, dtw)?;
            if accepted.is_empty() {
                debug!(
                    a = ida.0,
                    b = idb.0,
                    "rough alignment: no accepted interval"
                );
                return Ok(failed(ida, idb, rank, distance, p.pairs.fail_quality));
            }
            Ok(assemble(ida, idb, rank, distance, &accepted))
        }
        _ => {
            let fail = failed(ida, idb, rank, distance, p.pairs.fail_quality_no_coord);
            depth_mode(a, b, axis_a, axis_b, fail, &p.pairs, dtw)
        }
    }
}

fn is_eligible(info: &IntervalInfo) -> bool {
    !matches!(
        info.row.state,
        IntervalState::Pinched | IntervalState::Unknown
    ) && info.thickness_ft.is_finite()
        && info.thickness_ft > 0.0
}

/// One well's side of an interval-mode call.
fn side<'a>(w: &PairWell<'a>, coord: &'a [f64], info: &IntervalInfo) -> IntervalSide<'a> {
    IntervalSide {
        logs: w.logs,
        layer_coord: coord,
        top_md_ft: info.row.top_md_ft,
        base_md_ft: info.row.base_md_ft,
        confidence: info.confidence,
        is_pad: info.row.state == IntervalState::Pad,
    }
}

/// Interval mode: accepted `(interval, match)` in ascending
/// interval order.
fn interval_mode(
    a: &PairWell<'_>,
    b: &PairWell<'_>,
    coord_a: &[f64],
    coord_b: &[f64],
    p: &PairParams,
    dtw: &IntervalParams,
) -> SxResult<Vec<(u16, IntervalMatch)>> {
    let eligible: Vec<(u16, &IntervalInfo, &IntervalInfo)> = a
        .intervals
        .iter()
        .zip(b.intervals)
        .filter(|(ia, ib)| is_eligible(ia) && is_eligible(ib))
        .map(|(ia, ib)| (ia.row.interval, ia, ib))
        .collect();
    if eligible.is_empty() {
        return Ok(Vec::new());
    }
    let n_pad = eligible
        .iter()
        .filter(|(_, ia, ib)| {
            ia.row.state == IntervalState::Pad || ib.row.state == IntervalState::Pad
        })
        .count();
    let plan = interval_plan(eligible.len(), n_pad, dtw);
    let mut accepted = Vec::new();
    for (i, ia, ib) in eligible {
        let ratio = ib.thickness_ft / ia.thickness_ft;
        if !(p.min_span_ratio..=p.max_span_ratio).contains(&ratio) {
            debug!(
                interval = i,
                ratio, "rough alignment: span ratio out of range"
            );
            continue;
        }
        let m = interval_path(&side(a, coord_a, ia), &side(b, coord_b, ib), i, plan, dtw)?;
        if m.path.quality > 0.0 {
            accepted.push((i, m));
        }
    }
    Ok(accepted)
}

/// Pair-map assembly and quality. `accepted` is non-empty and in
/// ascending interval order.
fn assemble(
    a: WellId,
    b: WellId,
    rank: u16,
    distance_m: f64,
    accepted: &[(u16, IntervalMatch)],
) -> PairMap {
    let mut path: Vec<(i32, i32)> = Vec::new();
    let mut last_ka: Option<i32> = None;
    let mut max_kb = i32::MIN;
    for (_, m) in accepted {
        for &(ka, kb) in &m.path.path {
            if last_ka.is_none_or(|l| ka > l) {
                max_kb = max_kb.max(kb);
                path.push((ka, max_kb));
                last_ka = Some(ka);
            }
        }
    }
    let (mut sum_n, mut sum_q, mut sum_r) = (0.0f64, 0.0f64, 0.0f64);
    let (mut dtw_n, mut dtw_cost, mut dtw_mean) = (0.0f64, 0.0f64, 0.0f64);
    let (mut n_dtw, mut n_linear) = (0u16, 0u16);
    for (_, m) in accepted {
        let n = m.n_nodes as f64;
        sum_n += n;
        sum_q += n * f64::from(m.path.quality);
        sum_r += n * m.span_ratio;
        match m.method {
            MatchMethod::Dtw => {
                n_dtw = n_dtw.saturating_add(1);
                dtw_n += n;
                dtw_cost += m.path.cost;
                dtw_mean += n * m.mean_cost;
            }
            MatchMethod::Linear => n_linear = n_linear.saturating_add(1),
        }
    }
    let (total_cost, mean_cost) = if n_dtw > 0 {
        (dtw_cost, dtw_mean / dtw_n)
    } else {
        (f64::NAN, f64::NAN)
    };
    PairMap {
        path: PairPath {
            a,
            b,
            interval: None,
            path,
            cost: total_cost,
            quality: (sum_q / sum_n) as f32,
        },
        rank,
        distance_m,
        status: PairStatus::IntervalMap,
        span_ratio: sum_r / sum_n,
        mean_cost,
        n_dtw_intervals: n_dtw,
        n_linear_intervals: n_linear,
        intervals: accepted.iter().map(|(i, _)| *i).collect(),
        interval_quality: accepted.iter().map(|(_, m)| m.path.quality).collect(),
        interval_method: accepted.iter().map(|(_, m)| m.method as u8).collect(),
    }
}

/// Min and max of `axis` over samples where any of `values` is finite.
fn axis_range(axis: &[f64], values: &[Vec<f32>]) -> Option<(f64, f64)> {
    axis.iter()
        .enumerate()
        .filter(|(k, x)| {
            x.is_finite()
                && values
                    .iter()
                    .any(|v| v.get(*k).is_some_and(|y| y.is_finite()))
        })
        .fold(None, |acc, (_, &x)| match acc {
            None => Some((x, x)),
            Some((lo, hi)) => Some((lo.min(x), hi.max(x))),
        })
}

fn finite_pairs(axis: &[f64], values: &[f32]) -> usize {
    axis.iter()
        .zip(values)
        .filter(|(x, v)| x.is_finite() && v.is_finite())
        .count()
}

/// Depth mode on the elevation axis.
/// Returns `fail` when a step finds no usable match.
fn depth_mode(
    a: &PairWell<'_>,
    b: &PairWell<'_>,
    axis_a: &[f64],
    axis_b: &[f64],
    fail: PairMap,
    p: &PairParams,
    dtw: &IntervalParams,
) -> SxResult<PairMap> {
    let reject = |reason: &str, fail: PairMap| {
        debug!(
            a = fail.path.a.0,
            b = fail.path.b.0,
            reason,
            "rough alignment: depth map failed"
        );
        Ok(fail)
    };
    let shared: Vec<(Vec<f32>, Vec<f32>)> = LogKind::ALL
        .into_iter()
        .filter_map(|log| {
            let va = a.logs.curves.get(&log)?;
            let vb = b.logs.curves.get(&log)?;
            Some((working_values(log, va), working_values(log, vb)))
        })
        .collect();
    let (wa, wb): (Vec<Vec<f32>>, Vec<Vec<f32>>) = shared.into_iter().unzip();
    let (Some((lo_a, hi_a)), Some((lo_b, hi_b))) =
        (axis_range(axis_a, &wa), axis_range(axis_b, &wb))
    else {
        return reject("no shared log", fail);
    };
    let (lo, hi) = (lo_a.max(lo_b), hi_a.min(hi_b));
    let span = hi - lo;
    if span.is_nan() || span <= 0.0 {
        return reject("no depth overlap", fail);
    }
    let n = p.depth_nodes;
    if n < 2 {
        return reject("fewer than 2 nodes", fail);
    }
    let last = (n - 1) as f64;
    let node_x = |pos: f64| {
        if pos >= last {
            hi
        } else {
            lo + span * pos / last
        }
    };
    let nodes: Vec<f64> = (0..n).map(|m| node_x(m as f64)).collect();
    let gap = dtw.max_gap_steps * span / last;
    let mut rows_a: Vec<Vec<f64>> = Vec::new();
    let mut rows_b: Vec<Vec<f64>> = Vec::new();
    for (va, vb) in wa.iter().zip(&wb) {
        if finite_pairs(axis_a, va) < dtw.min_channel_samples
            || finite_pairs(axis_b, vb) < dtw.min_channel_samples
        {
            continue;
        }
        let mut ra = resample_curve(axis_a, va, &nodes, gap)?;
        let mut rb = resample_curve(axis_b, vb, &nodes, gap)?;
        if rescale_unit(&mut ra, dtw) && rescale_unit(&mut rb, dtw) {
            rows_a.push(ra);
            rows_b.push(rb);
        }
    }
    if rows_a.is_empty() {
        return reject("no channel", fail);
    }
    if shared_nodes(&rows_a, &rows_b) < p.depth_min_shared {
        return reject("too few shared nodes", fail);
    }
    let w = band_half_width(n, p.depth_band_frac)?;
    let dtw_path = banded_dtw(n, n, w, |i, j| {
        unit_cost(&rows_a, &rows_b, i, j, dtw.gap_cost)
    })?;
    let mapped = row_mean_map(&dtw_path, n);
    let pairs_md: Vec<(f64, f64)> = nodes
        .iter()
        .zip(&mapped)
        .map(|(&xa, &mb)| {
            (
                md_at_elev(a.header, a.geom, -xa),
                md_at_elev(b.header, b.geom, -node_x(mb)),
            )
        })
        .collect();
    let path = match pairs_from_md(&pairs_md) {
        Ok(path) => path,
        Err(_) => return reject("node MD not finite", fail),
    };
    Ok(PairMap {
        path: PairPath {
            path,
            cost: dtw_path.total_cost,
            quality: p.depth_quality,
            ..fail.path
        },
        status: PairStatus::DepthMap,
        span_ratio: 1.0,
        mean_cost: dtw_path.mean_cost(),
        ..fail
    })
}
