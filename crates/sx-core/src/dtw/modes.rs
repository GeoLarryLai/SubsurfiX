// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Interval, full and window modes.

use crate::config::{IntervalParams, SpanParams};
use crate::dtw::cost::{pooled_scale, shared_nodes, unit_cost, CappedCost, CostLog};
use crate::dtw::engine::{band_half_width, banded_dtw};
use crate::dtw::path::{pairs_from_md, row_mean_map};
use crate::dtw::resample::{resample_curve, rescale_unit, working_values_floor};
use crate::error::{SxError, SxResult};
use crate::types::{LogKind, PairPath, WellLogs, GRID_STEP_FT};

/// Node plan of one pair in interval mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalPlan {
    /// Nodes `N` per interval (nodes).
    pub n_nodes: usize,
    /// Shared nodes needed before DTW is tried (nodes).
    pub min_shared: usize,
}

/// Node plan for interval mode.
///
/// `n_shared` = intervals eligible in both wells, `n_pad` = of those, the ones
/// that are a pad in either well. `n_dtw = max(1, n_shared − n_pad)`,
/// `N = clamp(grid_budget / n_dtw, min_nodes, max_nodes)` (integer division),
/// `min_shared = max(floor, min(cap, N / div))`. Counts are intervals, results
/// are nodes. NaN rule: not applicable. Errors: none (a zero divisor is treated
/// as 1; `max_nodes < min_nodes` yields `min_nodes`). Complexity: `O(1)`.
pub fn interval_plan(n_shared: usize, n_pad: usize, p: &IntervalParams) -> IntervalPlan {
    let n_dtw = n_shared.saturating_sub(n_pad).max(1);
    let n_nodes = (p.grid_budget / n_dtw).min(p.max_nodes).max(p.min_nodes);
    let min_shared = p
        .min_shared_floor
        .max(p.min_shared_cap.min(n_nodes / p.min_shared_div.max(1)));
    IntervalPlan {
        n_nodes,
        min_shared,
    }
}

/// One well's side of an interval-mode call.
#[derive(Debug, Clone)]
pub struct IntervalSide<'a> {
    /// QC curves; RESD in ohm·m (converted inside).
    pub logs: &'a WellLogs,
    /// Layer coordinate of each grid sample of `logs.grid` (coordinate units).
    pub layer_coord: &'a [f64],
    /// Interval top, MD in feet.
    pub top_md_ft: f64,
    /// Interval base, MD in feet.
    pub base_md_ft: f64,
    /// Interval confidence in `[0, 1]`.
    pub confidence: f32,
    /// True if this interval is a pad in this well.
    pub is_pad: bool,
}

/// How an interval match was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatchMethod {
    /// Banded DTW on the shared channels.
    Dtw = 0,
    /// Node-for-node (tops-linear) match.
    Linear = 1,
}

/// Result of [`interval_path`].
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalMatch {
    /// Grid pairs; `interval = Some(i)`; `cost` is NaN for a linear match.
    pub path: PairPath,
    /// DTW or linear.
    pub method: MatchMethod,
    /// `total_cost / path_length` for DTW, NaN for linear (reported only).
    pub mean_cost: f64,
    /// `thickness_b / thickness_a` (thicknesses in ft).
    pub span_ratio: f64,
    /// Nodes `N` used.
    pub n_nodes: usize,
}

/// One well's side of a full- or window-mode call.
#[derive(Debug, Clone)]
pub struct SpanSide<'a> {
    /// Complete curves; RESD in ohm·m (converted inside).
    pub logs: &'a WellLogs,
    /// Span top, MD in feet.
    pub top_md_ft: f64,
    /// Span base, MD in feet.
    pub base_md_ft: f64,
}

/// `top + (base − top) × pos / last`; exactly `base` when `pos ≥ last`.
fn node_md(top: f64, base: f64, pos: f64, last: f64) -> f64 {
    if pos >= last {
        base
    } else {
        top + (base - top) * pos / last
    }
}

fn check_shapes(logs: &WellLogs, axis_len: Option<usize>) -> SxResult<()> {
    let n = logs.grid.len;
    if let Some(len) = axis_len {
        if len != n {
            return Err(SxError::Shape(format!(
                "well {}: layer_coord has {len} samples, grid has {n}",
                logs.id.0
            )));
        }
    }
    for (log, values) in &logs.curves {
        if values.len() != n {
            return Err(SxError::Shape(format!(
                "well {}: {} has {} samples, grid has {n}",
                logs.id.0,
                log.name(),
                values.len()
            )));
        }
    }
    Ok(())
}

/// Align one interval of one pair.
///
/// Linear match (node `m` of a to node `m` of b) is used, in this order, when
/// either side is a pad, no log survives the channel rules, or the shared nodes
/// are `< plan.min_shared`; the reason is logged with `tracing::debug!`. Any
/// failure inside the DTW also falls back to the linear match, so this never
/// returns [`SxError::Infeasible`]. Otherwise the DTW at `p.band_frac` gives
/// the node map (row means, running maximum, pinned ends). Quality is
/// `factor × sqrt(max(c_a, 0) × max(c_b, 0))` (`dtw_factor` or `linear_factor`).
///
/// Units: depths in ft, `layer_coord` in layer-coordinate units; `n_nodes`
/// nodes. NaN rule: skip for curve and coordinate values; error for
/// non-finite boundaries or confidences. Errors: [`SxError::InvalidInput`]
/// (non-finite boundary or confidence, `plan.n_nodes < 2`), [`SxError::Shape`]
/// (`layer_coord` or a curve length differs from `logs.grid.len`).
///
/// Complexity: `O(n_wells_samples × n_logs + N × w × n_logs)`.
pub fn interval_path(
    a: &IntervalSide<'_>,
    b: &IntervalSide<'_>,
    interval: u16,
    plan: IntervalPlan,
    p: &IntervalParams,
) -> SxResult<IntervalMatch> {
    for (name, v) in [
        ("a top", a.top_md_ft),
        ("a base", a.base_md_ft),
        ("b top", b.top_md_ft),
        ("b base", b.base_md_ft),
    ] {
        if !v.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "interval boundary {name} is not finite"
            )));
        }
    }
    if !a.confidence.is_finite() || !b.confidence.is_finite() {
        return Err(SxError::InvalidInput(
            "interval confidence is not finite".into(),
        ));
    }
    let n = plan.n_nodes;
    if n < 2 {
        return Err(SxError::InvalidInput(
            "interval plan needs at least 2 nodes".into(),
        ));
    }
    check_shapes(a.logs, Some(a.layer_coord.len()))?;
    check_shapes(b.logs, Some(b.layer_coord.len()))?;

    let last = (n - 1) as f64;
    let mut matched: Option<(Vec<f64>, f64, f64)> = None; // (mapped, total, mean)
    let mut reason = "";
    if a.is_pad || b.is_pad {
        reason = "pad interval";
    } else {
        let (rows_a, rows_b) = channel_rows(a, b, interval, n, p)?;
        if rows_a.is_empty() {
            reason = "no usable log";
        } else if shared_nodes(&rows_a, &rows_b) < plan.min_shared {
            reason = "too few shared nodes";
        } else {
            match run_interval_dtw(&rows_a, &rows_b, n, p) {
                Ok(m) => matched = Some(m),
                Err(err) => {
                    tracing::debug!(interval, %err, "interval DTW failed, using linear match");
                    reason = "dtw failed";
                }
            }
        }
    }

    let (method, mapped, total_cost, mean_cost, factor) = match matched {
        Some((mapped, total, mean)) => (MatchMethod::Dtw, mapped, total, mean, p.dtw_factor),
        None => {
            tracing::debug!(interval, reason, "interval linear match");
            let mapped: Vec<f64> = (0..n).map(|m| m as f64).collect();
            (
                MatchMethod::Linear,
                mapped,
                f64::NAN,
                f64::NAN,
                p.linear_factor,
            )
        }
    };

    let pairs_md: Vec<(f64, f64)> = mapped
        .iter()
        .enumerate()
        .map(|(m, &mb)| {
            (
                node_md(a.top_md_ft, a.base_md_ft, m as f64, last),
                node_md(b.top_md_ft, b.base_md_ft, mb, last),
            )
        })
        .collect();
    let path = pairs_from_md(&pairs_md)?;
    let quality =
        factor * (f64::from(a.confidence).max(0.0) * f64::from(b.confidence).max(0.0)).sqrt();
    Ok(IntervalMatch {
        path: PairPath {
            a: a.logs.id,
            b: b.logs.id,
            interval: Some(interval),
            path,
            cost: total_cost,
            quality: quality as f32,
        },
        method,
        mean_cost,
        span_ratio: (b.base_md_ft - b.top_md_ft) / (a.base_md_ft - a.top_md_ft),
        n_nodes: n,
    })
}

type NodeRows = Vec<Vec<f64>>;

/// Surviving channels of one interval: node rows per used log, in
/// suite order, one set per well.
fn channel_rows(
    a: &IntervalSide<'_>,
    b: &IntervalSide<'_>,
    interval: u16,
    n: usize,
    p: &IntervalParams,
) -> SxResult<(NodeRows, NodeRows)> {
    let last = (n - 1) as f64;
    let nodes: Vec<f64> = (0..n)
        .map(|m| {
            if m + 1 == n {
                f64::from(interval) + 1.0
            } else {
                f64::from(interval) + m as f64 / last
            }
        })
        .collect();
    let max_gap = p.max_gap_steps / last;
    let mut rows_a: NodeRows = Vec::new();
    let mut rows_b: NodeRows = Vec::new();
    for log in LogKind::ALL {
        let (Some(ca), Some(cb)) = (a.logs.curves.get(&log), b.logs.curves.get(&log)) else {
            continue;
        };
        let wa = working_values_floor(log, ca, p.resd_floor_ohmm);
        let wb = working_values_floor(log, cb, p.resd_floor_ohmm);
        if finite_pairs(a.layer_coord, &wa) < p.min_channel_samples
            || finite_pairs(b.layer_coord, &wb) < p.min_channel_samples
        {
            continue;
        }
        let mut ra = resample_curve(a.layer_coord, &wa, &nodes, max_gap)?;
        let mut rb = resample_curve(b.layer_coord, &wb, &nodes, max_gap)?;
        if rescale_unit(&mut ra, p) && rescale_unit(&mut rb, p) {
            rows_a.push(ra);
            rows_b.push(rb);
        }
    }
    Ok((rows_a, rows_b))
}

fn finite_pairs(axis: &[f64], values: &[f32]) -> usize {
    axis.iter()
        .zip(values.iter())
        .filter(|(x, v)| x.is_finite() && v.is_finite())
        .count()
}

fn run_interval_dtw(
    rows_a: &[Vec<f64>],
    rows_b: &[Vec<f64>],
    n: usize,
    p: &IntervalParams,
) -> SxResult<(Vec<f64>, f64, f64)> {
    let w = band_half_width(n, p.band_frac)?;
    let path = banded_dtw(n, n, w, |i, j| unit_cost(rows_a, rows_b, i, j, p.gap_cost))?;
    let mapped = row_mean_map(&path, n);
    Ok((mapped, path.total_cost, path.mean_cost()))
}

/// Full mode: DTW of one interval between the current boundaries, MD axis
///
/// Both wells are resampled onto `N = clamp(max(⌊thick/0.5⌋ + 1), min_nodes,
/// max_nodes)` nodes, costs are the GR-first capped robust-z costs, the DTW
/// band is `p.band_frac`. `PairPath.interval = Some(interval)`,
/// `cost = total_cost`, `quality = 1 / (1 + mean_cost)`.
///
/// Units: depths in ft. NaN rule: skip for curve values; error for non-finite
/// boundaries. Errors: [`SxError::InvalidInput`] (non-finite or `≤ 0`
/// thickness, band, or a non-finite cost), [`SxError::Shape`] (curve length
/// differs from the grid), [`SxError::EmptyInput`] (no log usable in both
/// wells), [`SxError::Infeasible`] only from the core.
///
/// Complexity: `O(N × w × n_logs)`.
pub fn full_path(
    a: &SpanSide<'_>,
    b: &SpanSide<'_>,
    interval: u16,
    p: &SpanParams,
) -> SxResult<PairPath> {
    span_path(a, b, Some(interval), p)
}

/// Window mode: like [`full_path`] over the caller's window with only the
/// Window ends pinned.
///
/// Pass [`SpanParams::window_default`] for the standard band 0.25.
/// `PairPath.interval = None`. Units, NaN rule, errors and complexity as
/// [`full_path`] (`O(N × w × n_logs)`).
pub fn window_path(a: &SpanSide<'_>, b: &SpanSide<'_>, p: &SpanParams) -> SxResult<PairPath> {
    span_path(a, b, None, p)
}

fn span_nodes(side: &SpanSide<'_>, n: usize) -> Vec<f64> {
    let last = (n - 1) as f64;
    (0..n)
        .map(|m| node_md(side.top_md_ft, side.base_md_ft, m as f64, last))
        .collect()
}

fn span_path(
    a: &SpanSide<'_>,
    b: &SpanSide<'_>,
    interval: Option<u16>,
    p: &SpanParams,
) -> SxResult<PairPath> {
    for side in [a, b] {
        let thick = side.base_md_ft - side.top_md_ft;
        if !side.top_md_ft.is_finite()
            || !side.base_md_ft.is_finite()
            || !thick.is_finite()
            || thick <= 0.0
        {
            return Err(SxError::InvalidInput(format!(
                "span boundaries must be finite with base > top, got {} .. {}",
                side.top_md_ft, side.base_md_ft
            )));
        }
        check_shapes(side.logs, None)?;
    }
    let n_span =
        |s: &SpanSide<'_>| ((s.base_md_ft - s.top_md_ft) / GRID_STEP_FT).floor() as usize + 1;
    let n = n_span(a)
        .max(n_span(b))
        .min(p.max_nodes)
        .max(p.min_nodes)
        .max(2);
    let last = (n - 1) as f64;
    let nodes_a = span_nodes(a, n);
    let nodes_b = span_nodes(b, n);
    let gap = |s: &SpanSide<'_>| {
        (p.max_gap_steps * (s.base_md_ft - s.top_md_ft) / last).max(p.min_gap_ft)
    };
    let (gap_a, gap_b) = (gap(a), gap(b));
    let axis = |s: &SpanSide<'_>| -> Vec<f64> {
        (0..s.logs.grid.len).map(|i| s.logs.grid.md_ft(i)).collect()
    };
    let (axis_a, axis_b) = (axis(a), axis(b));

    let mut used: Vec<CostLog> = Vec::new();
    for log in LogKind::ALL {
        let (Some(ca), Some(cb)) = (a.logs.curves.get(&log), b.logs.curves.get(&log)) else {
            continue;
        };
        let wa = working_values_floor(log, ca, p.resd_floor_ohmm);
        let wb = working_values_floor(log, cb, p.resd_floor_ohmm);
        let ra = resample_curve(&axis_a, &wa, &nodes_a, gap_a)?;
        let rb = resample_curve(&axis_b, &wb, &nodes_b, gap_b)?;
        if let Some(scale) = pooled_scale(&ra, &rb, p) {
            used.push(CostLog {
                is_gr: log == LogKind::Gr,
                scale,
                a: ra,
                b: rb,
            });
        }
    }
    if used.is_empty() {
        return Err(SxError::EmptyInput("no log is usable in both wells".into()));
    }
    let cost = CappedCost::new(used, p);
    let w = band_half_width(n, p.band_frac)?;
    let dtw = banded_dtw(n, n, w, |i, j| cost.cell(i, j))?;
    let pairs_md: Vec<(f64, f64)> = dtw
        .cells
        .iter()
        .filter_map(|&(i, j)| Some((*nodes_a.get(i)?, *nodes_b.get(j)?)))
        .collect();
    let path = pairs_from_md(&pairs_md)?;
    Ok(PairPath {
        a: a.logs.id,
        b: b.logs.id,
        interval,
        path,
        cost: dtw.total_cost,
        quality: (1.0 / (1.0 + dtw.mean_cost())) as f32,
    })
}
