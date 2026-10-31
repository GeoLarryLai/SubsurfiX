// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Who may donate and the candidate lists.
//!
//! The k-d tree is kiddo 6's `ImmutableKdTree<f64, 2>` (its
//! default `kiddo::KdTree` for a tree built once) over surface `(x_m, y_m)`
//! with squared Euclidean distance; the item is the index into the id-sorted
//! well list; it is built once per call. A nearest-`n` query gives a radius, a
//! radius query (slightly widened) collects every well within it, and the
//! result is re-sorted by exact `(distance, WellId)` and cut at `n`, so ties at
//! the `n`-th distance go to the lower `WellId`. If the tree cannot be built,
//! and for full-list walks (tier 5, [`eligible_donors`]), all wells are sorted
//! by brute force instead. Every path equals sorting all wells by
//! `(distance, WellId)`.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;

use crate::config::NeighborsConfig;
use crate::error::{SxError, SxResult};
use crate::neighbors::mapping::{interval_of, Measured, WellView};
use crate::neighbors::NeighborInputs;
use crate::types::{LogKind, WellId, WellLogs};

/// `donor` may donate `log` to `target` under.
pub(crate) fn may_donate(
    inputs: &NeighborInputs<'_>,
    target: WellId,
    donor: WellId,
    log: LogKind,
) -> bool {
    donor != target
        && !inputs.exclude.contains(&(donor, log))
        && !inputs.donor_exclude_wells.contains(&donor)
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    (dx * dx + dy * dy).sqrt()
}

/// Surface distance between two wells' header points, m; NaN if either is
/// missing or not finite.
pub(crate) fn surface_distance_m(inputs: &NeighborInputs<'_>, a: WellId, b: WellId) -> f64 {
    match (inputs.headers.get(&a), inputs.headers.get(&b)) {
        (Some(ha), Some(hb)) => distance([ha.x_m, ha.y_m], [hb.x_m, hb.y_m]),
        _ => f64::NAN,
    }
}

/// `(well, surface distance m)` sorted by `(distance, WellId)`.
pub(crate) type DonorList = Vec<(WellId, f64)>;

fn rank_order(a: &(WellId, f64), b: &(WellId, f64)) -> Ordering {
    a.1.total_cmp(&b.1).then(a.0.cmp(&b.0))
}

/// Measured finite samples of `log` (working units); `None` without a curve.
fn count_measured(logs: &WellLogs, log: LogKind) -> Option<usize> {
    let m = Measured::new(logs, log)?;
    Some((0..m.len()).filter(|&j| m.get(j).is_some()).count())
}

/// Per well and log: measured count and per-interval row / measured counts.
pub(crate) struct LogStats {
    n_measured: usize,
    /// interval id → (rows with that id, rows with a measured finite value).
    per_interval: BTreeMap<i64, (usize, usize)>,
}

impl LogStats {
    /// `None` when the well has no curve for `log`.
    pub(crate) fn new(w: &WellView<'_>, log: LogKind, cfg: &NeighborsConfig) -> Option<Self> {
        let m = Measured::new(w.logs, log)?;
        let mut n_measured = 0usize;
        let mut per_interval: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
        for j in 0..m.len() {
            let measured = m.get(j).is_some();
            n_measured += usize::from(measured);
            let u = w.coord.as_ref().map_or(f64::NAN, |c| c.at(j));
            if u.is_finite() {
                let e = per_interval
                    .entry(interval_of(u, cfg.interval_eps))
                    .or_insert((0, 0));
                e.0 += 1;
                e.1 += usize::from(measured);
            }
        }
        Some(Self {
            n_measured,
            per_interval,
        })
    }

    /// `cov_d(i)`: share of rows in interval `i` with a measured finite value.
    pub(crate) fn cov(&self, i: i64) -> f64 {
        match self.per_interval.get(&i) {
            Some(&(rows, measured)) if rows > 0 => measured as f64 / rows as f64,
            _ => 0.0,
        }
    }

    /// Test: a measured finite sample on a row whose interval is in `set`.
    pub(crate) fn covers(&self, set: &BTreeSet<i64>) -> bool {
        self.per_interval
            .iter()
            .any(|(i, &(_, measured))| measured > 0 && set.contains(i))
    }
}

/// Surface points of all wells with a finite header point.
pub(crate) struct WellIndex {
    ids: Vec<WellId>,
    xy: Vec<[f64; 2]>,
    tree: Option<ImmutableKdTree<f64, 2>>,
}

impl WellIndex {
    /// `points` sorted by id.
    fn new(points: Vec<(WellId, [f64; 2])>) -> Self {
        let (ids, xy): (Vec<WellId>, Vec<[f64; 2]>) = points.into_iter().unzip();
        let tree = if xy.is_empty() {
            None
        } else {
            ImmutableKdTree::new_from_slice(&xy).ok()
        };
        Self { ids, xy, tree }
    }

    fn entry(&self, i: u32, q: [f64; 2]) -> Option<(WellId, f64)> {
        let i = usize::try_from(i).ok()?;
        Some((*self.ids.get(i)?, distance(q, *self.xy.get(i)?)))
    }

    /// Every well sorted by `(distance, WellId)`.
    pub(crate) fn all_sorted(&self, q: [f64; 2]) -> Vec<(WellId, f64)> {
        let mut out: Vec<(WellId, f64)> = self
            .ids
            .iter()
            .zip(&self.xy)
            .map(|(&id, &p)| (id, distance(q, p)))
            .collect();
        out.sort_by(rank_order);
        out
    }

    /// The first `n` wells by `(distance, WellId)`, the query well included.
    pub(crate) fn knn(&self, q: [f64; 2], n: usize) -> Vec<(WellId, f64)> {
        let (Some(tree), Some(qty)) = (&self.tree, NonZero::new(n)) else {
            let mut all = self.all_sorted(q);
            all.truncate(n);
            return all;
        };
        if n >= self.ids.len() {
            return self.all_sorted(q);
        }
        let radius = tree
            .query(&q)
            .nearest_n::<SquaredEuclidean<f64>>(qty)
            .execute()
            .into_iter()
            .filter_map(|nn| self.entry(nn.item, q))
            .fold(0.0f64, |acc, (_, d)| acc.max(d));
        let r2 = radius * radius * (1.0 + 1e-9) + 1e-9;
        let mut out: Vec<(WellId, f64)> = tree
            .query(&q)
            .within::<SquaredEuclidean<f64>>(r2)
            .unsorted()
            .execute()
            .into_iter()
            .filter_map(|nn| self.entry(nn.item, q))
            .collect();
        out.sort_by(rank_order);
        out.truncate(n);
        out
    }
}

/// Prepared per-call view of every well, with log statistics and the tree.
pub(crate) struct Pool<'a> {
    pub inputs: &'a NeighborInputs<'a>,
    pub cfg: &'a NeighborsConfig,
    pub wells: BTreeMap<WellId, WellView<'a>>,
    pub stats: BTreeMap<(WellId, LogKind), LogStats>,
    pub index: WellIndex,
}

impl<'a> Pool<'a> {
    /// Errors: `Shape` / `InvalidInput` from any well's input checks.
    pub(crate) fn new(inputs: &'a NeighborInputs<'a>, cfg: &'a NeighborsConfig) -> SxResult<Self> {
        let ids: Vec<WellId> = inputs.logs.keys().copied().collect();
        let views: Vec<SxResult<WellView<'a>>> = ids
            .par_iter()
            .map(|&id| WellView::new(inputs, id))
            .collect();
        let mut list = Vec::with_capacity(views.len());
        for v in views {
            list.push(v?);
        }
        let stats: Vec<Vec<((WellId, LogKind), LogStats)>> = list
            .par_iter()
            .map(|w| {
                LogKind::ALL
                    .iter()
                    .filter_map(|&log| LogStats::new(w, log, cfg).map(|s| ((w.id, log), s)))
                    .collect()
            })
            .collect();
        let index = WellIndex::new(list.iter().filter_map(|w| Some((w.id, w.xy?))).collect());
        Ok(Self {
            inputs,
            cfg,
            wells: list.into_iter().map(|w| (w.id, w)).collect(),
            stats: stats.into_iter().flatten().collect(),
            index,
        })
    }

    /// Statistics of `donor`'s `log` if it is eligible for `target`
    /// (a finite surface point is also required).
    pub(crate) fn eligible(
        &self,
        target: WellId,
        donor: WellId,
        log: LogKind,
    ) -> Option<&LogStats> {
        if !may_donate(self.inputs, target, donor, log) {
            return None;
        }
        self.wells.get(&donor)?.xy?;
        self.stats
            .get(&(donor, log))
            .filter(|s| s.n_measured >= self.cfg.min_finite_donor)
    }

    /// Capped and rest lists from the target's nearest-well list.
    /// `target_set` is the target interval set when it has a layer coordinate.
    pub(crate) fn candidates(
        &self,
        target: WellId,
        knn: &[(WellId, f64)],
        log: LogKind,
        target_set: Option<&BTreeSet<i64>>,
    ) -> (DonorList, DonorList) {
        let mut capped: DonorList = knn
            .iter()
            .copied()
            .filter(|&(id, d)| {
                d <= self.cfg.max_dist_m
                    && self
                        .eligible(target, id, log)
                        .is_some_and(|s| target_set.is_none_or(|set| s.covers(set)))
            })
            .collect();
        let rest = capped.split_off(self.cfg.search_cap.min(capped.len()));
        (capped, rest)
    }
}

/// Eligible donors for `target` and `log`,
/// sorted by (distance, WellId), optionally cut at `max_dist_m` and `max_n`.
/// Returns (donor, surface distance m).
///
/// Donors are drawn from every well in `inputs.logs` with a finite header
/// point; the target itself, excluded channels, holdout wells, wells without a
/// curve for `log` and wells with fewer than `min_finite_donor` measured
/// finite samples are left out. NaN rule: skip (no finite distance, no
/// donor). Errors: `NotFound` if `target` is missing from `headers`; `Shape`
/// for a donor's log length mismatch. Complexity: `O(wells × grid + wells log wells)`.
pub fn eligible_donors(
    inputs: &NeighborInputs<'_>,
    target: WellId,
    log: LogKind,
    max_dist_m: Option<f64>,
    max_n: Option<usize>,
    cfg: &NeighborsConfig,
) -> SxResult<Vec<(WellId, f64)>> {
    if !inputs.headers.contains_key(&target) {
        return Err(SxError::NotFound(format!(
            "eligible donors: target {} has no header",
            target.0
        )));
    }
    let mut out = Vec::new();
    for (&id, logs) in inputs.logs {
        if !may_donate(inputs, target, id, log) {
            continue;
        }
        let d = surface_distance_m(inputs, target, id);
        if !d.is_finite() || max_dist_m.is_some_and(|m| m.is_nan() || d > m) {
            continue;
        }
        logs.validate()?;
        if count_measured(logs, log).is_some_and(|n| n >= cfg.min_finite_donor) {
            out.push((id, d));
        }
    }
    out.sort_by(rank_order);
    if let Some(n) = max_n {
        out.truncate(n);
    }
    Ok(out)
}
