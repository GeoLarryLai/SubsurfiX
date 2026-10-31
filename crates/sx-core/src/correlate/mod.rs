// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Full correlation.
//!
//! Builds the well graph ([`graph`]), runs pair DTW per active edge and
//! interval ([`pairs`]), and solves the robust sparse network for one
//! monotone layer time per well ([`solve`]), in two passes around the tops
//! check ([`tops`]). Depths are feet (MD), distances meters, missing values
//! NaN. Writing files belongs to the pipeline runner.

pub mod graph;
pub mod pairs;
pub mod solve;
pub mod subhorizons;
pub mod tops;

use std::collections::BTreeSet;

use crate::config::{CorrelateGraphConfig, CorrelateSolveConfig, CorrelateTopsConfig, DtwConfig};
use crate::error::{SxError, SxResult};
use crate::types::{
    IntervalRow, IntervalState, TopPick, TopSet, TopSource, WellGeometry, WellHeader, WellId,
    WellLogs,
};

/// Boundary match tolerance between neighboring interval rows, ft.
const BOUNDARY_TOL_FT: f64 = 1e-9;

/// User-pick match tolerance against its boundary, ft.
const USER_PICK_TOL_FT: f64 = 1e-6;

/// Boundary depths and sources of one well, `TopSet` layout.
/// `md_ft.len() == source.len() == n + 2`, `state.len() == n + 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct WellBoundaries {
    /// Well id.
    pub well: WellId,
    /// Boundary MD, ft. NaN = cannot be placed.
    pub md_ft: Vec<f64>,
    /// Origin of each boundary. Index 0 and n+1 are `Pad`.
    pub source: Vec<TopSource>,
    /// Interval states. Same in Pass 1 and Pass 2.
    pub state: Vec<IntervalState>,
}

/// Inputs of [`run_correlation`].
#[derive(Clone, Debug)]
pub struct CorrelationInputs<'a> {
    /// Sorted by id, unique.
    pub headers: &'a [WellHeader],
    /// Same ids, same order.
    pub geometry: &'a [WellGeometry],
    /// Complete curves after gap filling. Same ids, same order.
    pub logs: &'a [WellLogs],
    /// Ordered top names.
    pub tops: &'a TopSet,
    /// `n + 1` rows per well, sorted by (well, interval).
    pub intervals: &'a [IntervalRow],
    /// User picks and estimated tops from rough alignment.
    pub picks: &'a [TopPick],
}

/// Config sections used by [`run_correlation`].
#[derive(Clone, Debug)]
pub struct CorrelationConfig<'a> {
    /// `[correlate.graph]`.
    pub graph: &'a CorrelateGraphConfig,
    /// `[correlate.solve]`.
    pub solve: &'a CorrelateSolveConfig,
    /// `[correlate.tops]`, passed to the tops check.
    pub tops: &'a CorrelateTopsConfig,
    /// `[dtw]`; pair DTW uses `dtw.full`.
    pub dtw: &'a DtwConfig,
}

/// Result of [`run_correlation`].
#[derive(Clone, Debug)]
pub struct CorrelationOutput {
    /// Well graph.
    pub graph: graph::WellGraph,
    /// Pass 1 boundaries.
    pub boundaries_pass1: Vec<WellBoundaries>,
    /// Pass 2 boundaries (from the tops check).
    pub boundaries_pass2: Vec<WellBoundaries>,
    /// Tops check between the passes.
    pub tops_check: tops::TopsCheck,
    /// Final (Pass 2) layer time, sorted by well.
    pub layer_time: Vec<solve::LayerTimeWell>,
    /// Pass 1 rows, then Pass 2 rows.
    pub solve_report: Vec<solve::SolveReportRow>,
}

/// Interval `i` is usable: state not `Unknown`, both boundaries finite,
/// positive thickness.
pub(crate) fn interval_usable(b: &WellBoundaries, i: usize) -> bool {
    match (b.state.get(i), b.md_ft.get(i), b.md_ft.get(i + 1)) {
        (Some(&s), Some(&top), Some(&base)) => {
            s != IntervalState::Unknown && top.is_finite() && base.is_finite() && base > top
        }
        _ => false,
    }
}

/// Interval `i` is usable and neither `Pinched` nor `Unknown` in this well.
pub(crate) fn interval_solvable(b: &WellBoundaries, i: usize) -> bool {
    interval_usable(b, i)
        && !matches!(
            b.state.get(i),
            Some(IntervalState::Pinched) | Some(IntervalState::Unknown) | None
        )
}

/// One boundary from the base of the row above and the top of the row below.
fn merge_boundary(well: WellId, b: usize, above: f64, below: f64) -> SxResult<f64> {
    match (above.is_finite(), below.is_finite()) {
        (true, true) => {
            if (above - below).abs() > BOUNDARY_TOL_FT {
                Err(SxError::InvalidInput(format!(
                    "well {}: boundary {b} is {above} ft as a base and {below} ft as a top",
                    well.0
                )))
            } else {
                Ok(below)
            }
        }
        (true, false) => Ok(above),
        (false, true) => Ok(below),
        (false, false) => Ok(f64::NAN),
    }
}

/// Build per-well boundaries from the interval table and picks.
///
/// `md_ft[i]` = row `i` top, `md_ft[n+1]` = row `n` base; a row base and the
/// next row top must agree within 1e-9 ft when both are finite, else the
/// finite one is used. Sources: `Pad` at both ends; for top `j` `User` if a
/// user pick exists, else the source of the first other pick in input order,
/// else `Estimated`. States copy the rows.
///
/// **Choice:** Picks for wells absent from `intervals`, or with a top
/// index `≥ n`, are ignored.
///
/// Units: ft. NaN rule: a non-finite MD is "cannot be placed" (NaN).
/// Errors: [`SxError::Shape`] (rows not sorted by (well, interval), or a
/// well without exactly the rows `0..=n`), [`SxError::InvalidInput`]
/// (boundary disagreement, finite boundaries decreasing with index, two user
/// picks for one top, a user pick more than 1e-6 ft from its boundary).
/// Empty `intervals` gives an empty vector. Complexity:
/// `O(rows + picks log wells)`.
pub fn build_boundaries(
    tops: &TopSet,
    intervals: &[IntervalRow],
    picks: &[TopPick],
) -> SxResult<Vec<WellBoundaries>> {
    let n = tops.len();
    if intervals
        .windows(2)
        .any(|p| matches!(p, [x, y] if (x.well, x.interval) >= (y.well, y.interval)))
    {
        return Err(SxError::Shape(
            "interval rows must be sorted by (well, interval), unique".into(),
        ));
    }
    let mut out: Vec<WellBoundaries> = Vec::new();
    for rows in intervals.chunk_by(|x, y| x.well == y.well) {
        let Some(first) = rows.first() else {
            continue;
        };
        let well = first.well;
        if rows.len() != n + 1
            || rows
                .iter()
                .enumerate()
                .any(|(i, r)| usize::from(r.interval) != i)
        {
            return Err(SxError::Shape(format!(
                "well {}: interval rows must be exactly 0..={n}",
                well.0
            )));
        }
        let mut md_ft = Vec::with_capacity(n + 2);
        let mut prev_base = f64::NAN;
        for (i, r) in rows.iter().enumerate() {
            md_ft.push(if i == 0 {
                if r.top_md_ft.is_finite() {
                    r.top_md_ft
                } else {
                    f64::NAN
                }
            } else {
                merge_boundary(well, i, prev_base, r.top_md_ft)?
            });
            prev_base = r.base_md_ft;
        }
        md_ft.push(if prev_base.is_finite() {
            prev_base
        } else {
            f64::NAN
        });
        let mut last = f64::NEG_INFINITY;
        for &md in md_ft.iter().filter(|m| m.is_finite()) {
            if md < last {
                return Err(SxError::InvalidInput(format!(
                    "well {}: finite boundaries decrease with index",
                    well.0
                )));
            }
            last = md;
        }
        let mut source = vec![TopSource::Estimated; n + 2];
        if let Some(s) = source.first_mut() {
            *s = TopSource::Pad;
        }
        if let Some(s) = source.last_mut() {
            *s = TopSource::Pad;
        }
        out.push(WellBoundaries {
            well,
            md_ft,
            source,
            state: rows.iter().map(|r| r.state).collect(),
        });
    }

    let mut user_seen: BTreeSet<(WellId, u16)> = BTreeSet::new();
    let mut other_seen: BTreeSet<(WellId, u16)> = BTreeSet::new();
    for p in picks {
        let Ok(w) = out.binary_search_by_key(&p.well, |b| b.well) else {
            continue;
        };
        let j = usize::from(p.top);
        if j >= n {
            continue;
        }
        let Some(b) = out.get_mut(w) else {
            continue;
        };
        let (Some(slot), Some(&md)) = (b.source.get_mut(j + 1), b.md_ft.get(j + 1)) else {
            continue;
        };
        if p.source == TopSource::User {
            if !user_seen.insert((p.well, p.top)) {
                return Err(SxError::InvalidInput(format!(
                    "well {}: two user picks for top {j}",
                    p.well.0
                )));
            }
            let off = (p.md_ft - md).abs();
            if off.is_nan() || off > USER_PICK_TOL_FT {
                return Err(SxError::InvalidInput(format!(
                    "well {}: user pick for top {j} at {} ft does not match boundary {md} ft",
                    p.well.0, p.md_ft
                )));
            }
            *slot = TopSource::User;
        } else if other_seen.insert((p.well, p.top)) && !user_seen.contains(&(p.well, p.top)) {
            *slot = p.source;
        }
    }
    Ok(out)
}

/// Check that the boundaries of the tops check keep wells, lengths and states.
fn check_same_layout(b1: &[WellBoundaries], b2: &[WellBoundaries]) -> SxResult<()> {
    let same = b1.len() == b2.len()
        && b1.iter().zip(b2).all(|(x, y)| {
            x.well == y.well
                && x.md_ft.len() == y.md_ft.len()
                && x.source.len() == y.source.len()
                && x.state == y.state
        });
    if same {
        Ok(())
    } else {
        Err(SxError::Shape(
            "tops check boundaries differ from Pass 1 in wells, lengths or states".into(),
        ))
    }
}

/// Set `path_recomputed` on edge rows whose (interval, edge) is in `todo`.
fn mark_recomputed(
    report: &mut [solve::SolveReportRow],
    graph: &graph::WellGraph,
    todo: &[(u16, usize)],
) {
    let todo: BTreeSet<(u16, usize)> = todo.iter().copied().collect();
    for row in report.iter_mut() {
        if row.kind != solve::ReportRowKind::Edge {
            continue;
        }
        if let (Some(a), Some(b)) = (row.well_a, row.well_b) {
            if let Some(e) = graph.edge_index(a, b) {
                row.path_recomputed = todo.contains(&(row.interval, e));
            }
        }
    }
}

/// Full correlation: Pass 1, tops check, boundary update, Pass 2.
///
/// Fixed sequence, no loop: boundaries and graph; Pass 1 pair DTW on all
/// active pairs and solve; [`tops::check_tops`]; Pass 2 recomputes DTW only
/// for pairs whose boundaries moved ([`pairs::changed_pairs`]), reuses the
/// rest (including Pass 1 failures), and solves again. Output layer time is
/// Pass 2; the report is Pass 1 rows then Pass 2 rows.
///
/// Units: ft, m. NaN rule: as the parts. Errors: [`SxError::Config`] (bad
/// graph or solve keys), [`SxError::EmptyInput`] (no wells),
/// [`SxError::Shape`] (headers, geometry, logs or interval wells disagree in
/// ids or order; a `validate()` failure; tops-check boundaries change
/// layout), plus the errors of [`build_boundaries`],
/// [`graph::build_well_graph`], [`solve::solve_pass`] and
/// [`tops::check_tops`]. Complexity: dominated by pair DTW,
/// `O(pairs × N × w × logs)`.
pub fn run_correlation(
    inp: &CorrelationInputs<'_>,
    cfg: &CorrelationConfig<'_>,
) -> SxResult<CorrelationOutput> {
    graph::check_graph_config(cfg.graph)?;
    solve::check_solve_config(cfg.solve)?;
    if inp.headers.is_empty() {
        return Err(SxError::EmptyInput("correlation: no wells".into()));
    }
    if inp
        .headers
        .windows(2)
        .any(|p| matches!(p, [x, y] if x.id >= y.id))
    {
        return Err(SxError::Shape(
            "correlation: headers must be sorted by id, unique".into(),
        ));
    }
    let ids: Vec<WellId> = inp.headers.iter().map(|h| h.id).collect();
    let geo_ids: Vec<WellId> = inp.geometry.iter().map(|g| g.id).collect();
    let log_ids: Vec<WellId> = inp.logs.iter().map(|l| l.id).collect();
    if geo_ids != ids || log_ids != ids {
        return Err(SxError::Shape(
            "correlation: headers, geometry and logs must have the same ids in the same order"
                .into(),
        ));
    }
    for g in inp.geometry {
        g.validate()?;
    }
    for l in inp.logs {
        l.validate()?;
    }

    let b1 = build_boundaries(inp.tops, inp.intervals, inp.picks)?;
    let b_ids: Vec<WellId> = b1.iter().map(|b| b.well).collect();
    if b_ids != ids {
        return Err(SxError::Shape(
            "correlation: interval table wells differ from the headers".into(),
        ));
    }
    let graph = graph::build_well_graph(inp.headers, cfg.graph)?;
    tracing::info!(
        wells = graph.wells.len(),
        edges = graph.edges.len(),
        "correlation graph built"
    );

    let todo1 = pairs::active_pairs(&graph, &b1);
    let paths1 = pairs::compute_pair_paths(&graph, inp.logs, &b1, &todo1, cfg.dtw)?;
    let mut p1 = solve::solve_pass(1, &graph, inp.logs, &b1, &paths1, cfg.solve)?;
    mark_recomputed(&mut p1.report, &graph, &todo1);
    tracing::info!(
        pairs = todo1.len(),
        failed = paths1.failed.len(),
        "correlation pass 1 solved"
    );

    let check = tops::check_tops(&graph, inp.logs, &b1, cfg.tops, cfg.dtw)?;
    let b2 = check.boundaries.clone();
    check_same_layout(&b1, &b2)?;

    let todo2 = pairs::changed_pairs(&graph, &b1, &b2);
    let fresh = pairs::compute_pair_paths(&graph, inp.logs, &b2, &todo2, cfg.dtw)?;
    let changed: BTreeSet<(u16, usize)> = todo2.iter().copied().collect();
    let mut paths2 = pairs::PairPaths {
        paths: Default::default(),
        failed: Default::default(),
    };
    for key in pairs::active_pairs(&graph, &b2) {
        let src = if changed.contains(&key) {
            &fresh
        } else {
            &paths1
        };
        if let Some(p) = src.paths.get(&key) {
            paths2.paths.insert(key, p.clone());
        } else if src.failed.contains(&key) {
            paths2.failed.insert(key);
        }
    }
    let mut p2 = solve::solve_pass(2, &graph, inp.logs, &b2, &paths2, cfg.solve)?;
    mark_recomputed(&mut p2.report, &graph, &todo2);
    tracing::info!(recomputed = todo2.len(), "correlation pass 2 solved");

    let mut solve_report = p1.report;
    solve_report.extend(p2.report);
    Ok(CorrelationOutput {
        graph,
        boundaries_pass1: b1,
        boundaries_pass2: b2,
        tops_check: check,
        layer_time: p2.layer_time,
        solve_report,
    })
}
