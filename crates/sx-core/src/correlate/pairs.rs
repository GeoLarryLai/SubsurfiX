// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Active pairs and pair DTW per edge and interval.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::config::DtwConfig;
use crate::correlate::graph::WellGraph;
use crate::correlate::{interval_solvable, WellBoundaries};
use crate::dtw::{full_path, SpanSide};
use crate::error::{SxError, SxResult};
use crate::types::{PairPath, WellId, WellLogs};

/// Boundary match tolerance between passes, ft.
const MOVE_TOL_FT: f64 = 1e-9;

/// Pair DTW results of one pass.
#[derive(Clone, Debug)]
pub struct PairPaths {
    /// Key (interval, edge index). Path `a` = edge `a` (lower id).
    pub paths: BTreeMap<(u16, usize), PairPath>,
    /// Active (interval, edge) where the DTW call returned an error.
    pub failed: BTreeSet<(u16, usize)>,
}

/// Boundaries by well; the first entry wins on a repeated id.
fn by_well(bounds: &[WellBoundaries]) -> BTreeMap<WellId, &WellBoundaries> {
    let mut map = BTreeMap::new();
    for b in bounds {
        map.entry(b.well).or_insert(b);
    }
    map
}

/// Active (interval, edge index) pairs, sorted by interval then edge index.
///
/// Edge `(a, b)` is active in interval `i` when the interval is usable in
/// both wells and neither state is `Pinched` or `Unknown`; edges with a well
/// missing from `bounds` are never active. NaN rule: a NaN boundary makes the
/// interval unusable. Errors: none. Complexity: `O(intervals × edges)`.
pub fn active_pairs(graph: &WellGraph, bounds: &[WellBoundaries]) -> Vec<(u16, usize)> {
    let map = by_well(bounds);
    let ends: Vec<Option<(&WellBoundaries, &WellBoundaries)>> = graph
        .edges
        .iter()
        .map(|e| Some((*map.get(&e.a)?, *map.get(&e.b)?)))
        .collect();
    let n_int = bounds.iter().map(|b| b.state.len()).max().unwrap_or(0);
    let mut out = Vec::new();
    for i in 0..n_int {
        let Ok(iv) = u16::try_from(i) else {
            break;
        };
        for (e, pair) in ends.iter().enumerate() {
            if let Some((ba, bb)) = pair {
                if interval_solvable(ba, i) && interval_solvable(bb, i) {
                    out.push((iv, e));
                }
            }
        }
    }
    out
}

/// Two boundary values differ by more than 1e-9 ft (NaN equals NaN).
fn moved(x: Option<&f64>, y: Option<&f64>) -> bool {
    match (x, y) {
        (Some(x), Some(y)) => {
            if x.is_nan() && y.is_nan() {
                false
            } else {
                let d = (x - y).abs();
                d.is_nan() || d > MOVE_TOL_FT
            }
        }
        (None, None) => false,
        _ => true,
    }
}

/// Active pairs whose interval top or base moved in either well between passes.
///
/// Active on `after`; a well missing from `before` counts as moved. Units: ft.
/// NaN rule: NaN equals NaN, NaN vs finite is a move. Errors: none.
/// Complexity: `O(intervals × edges)`.
pub fn changed_pairs(
    graph: &WellGraph,
    before: &[WellBoundaries],
    after: &[WellBoundaries],
) -> Vec<(u16, usize)> {
    let (map0, map1) = (by_well(before), by_well(after));
    let well_moved = |w: WellId, i: usize| -> bool {
        let (Some(b0), Some(b1)) = (map0.get(&w), map1.get(&w)) else {
            return true;
        };
        moved(b0.md_ft.get(i), b1.md_ft.get(i)) || moved(b0.md_ft.get(i + 1), b1.md_ft.get(i + 1))
    };
    active_pairs(graph, after)
        .into_iter()
        .filter(|&(i, e)| {
            let i = usize::from(i);
            graph
                .edges
                .get(e)
                .is_some_and(|edge| well_moved(edge.a, i) || well_moved(edge.b, i))
        })
        .collect()
}

/// Run `dtw::full_path` for each listed pair, in parallel.
///
/// Well `a` (lower id) first, well `b` second, complete curves, interval `i`
/// top and base of each well, `dtw.full` settings. Rayon over `todo`,
/// results collected in list order. A DTW error puts `(i, e)` into `failed`
/// (logged with `debug!`); other pairs continue.
///
/// Units: ft. NaN rule: as [`full_path`] (curve NaNs skipped). Errors:
/// [`SxError::NotFound`] (edge index out of range, or a well missing from
/// `logs` or `bounds`). Empty `todo` gives empty results. Complexity:
/// `O(todo × N × w × logs)`.
pub fn compute_pair_paths(
    graph: &WellGraph,
    logs: &[WellLogs],
    bounds: &[WellBoundaries],
    todo: &[(u16, usize)],
    dtw: &DtwConfig,
) -> SxResult<PairPaths> {
    let mut logs_map: BTreeMap<WellId, &WellLogs> = BTreeMap::new();
    for l in logs {
        logs_map.entry(l.id).or_insert(l);
    }
    let bounds_map = by_well(bounds);
    let find_logs = |w: WellId| -> SxResult<&WellLogs> {
        logs_map
            .get(&w)
            .copied()
            .ok_or_else(|| SxError::NotFound(format!("logs of well {}", w.0)))
    };
    let find_b = |w: WellId| -> SxResult<&WellBoundaries> {
        bounds_map
            .get(&w)
            .copied()
            .ok_or_else(|| SxError::NotFound(format!("boundaries of well {}", w.0)))
    };
    let mut jobs: Vec<(u16, SpanSide<'_>, SpanSide<'_>)> = Vec::with_capacity(todo.len());
    for &(i, e) in todo {
        let edge = graph
            .edges
            .get(e)
            .ok_or_else(|| SxError::NotFound(format!("graph edge {e}")))?;
        let iu = usize::from(i);
        let side = |w: WellId| -> SxResult<SpanSide<'_>> {
            let b = find_b(w)?;
            Ok(SpanSide {
                logs: find_logs(w)?,
                top_md_ft: b.md_ft.get(iu).copied().unwrap_or(f64::NAN),
                base_md_ft: b.md_ft.get(iu + 1).copied().unwrap_or(f64::NAN),
            })
        };
        jobs.push((i, side(edge.a)?, side(edge.b)?));
    }
    let results: Vec<SxResult<PairPath>> = jobs
        .par_iter()
        .map(|(i, a, b)| full_path(a, b, *i, &dtw.full))
        .collect();
    let mut out = PairPaths {
        paths: BTreeMap::new(),
        failed: BTreeSet::new(),
    };
    for (&(i, e), res) in todo.iter().zip(results) {
        match res {
            Ok(path) => {
                out.paths.insert((i, e), path);
            }
            Err(err) => {
                tracing::debug!(interval = i, edge = e, %err, "pair DTW failed");
                out.failed.insert((i, e));
            }
        }
    }
    Ok(out)
}
