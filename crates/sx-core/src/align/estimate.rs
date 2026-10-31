// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Missing-top estimation, ordering fix-up and per-well acceptance.

use std::cmp::Ordering;

use rayon::prelude::*;
use tracing::debug;

use crate::align::breaks::build_breaks;
use crate::align::depth::md_at_elev;
use crate::align::pairs::{distance_m, k_nearest};
use crate::align::run::BoundaryStatus;
use crate::config::TopsParams;
use crate::types::{WellGeometry, WellHeader};

/// Cleaned user picks of one well over all canonical names.
#[derive(Debug, Clone)]
pub(crate) struct WellPicks {
    /// Pick MD per canonical name (ft); NaN = absent or not a depth.
    pub md_ft: Vec<f64>,
    /// `elev_at_md(md_ft)` per name (ft); NaN when `md_ft` is NaN.
    pub elev_ft: Vec<f64>,
    /// A pick row exists for the name.
    pub present: Vec<bool>,
}

/// Result of one `(well, top)` estimate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Estimate {
    /// Estimated MD, ft (NaN when `Missing`).
    pub md_ft: f64,
    /// Estimated elevation, ft (one donor's pick elevation; NaN when `Missing`).
    pub elev_ft: f64,
    /// Donors kept within `max_dist_m` (0 when `Missing`).
    pub n_donors: u16,
    /// `Kept` (accepted), `Missing`, `RejectedOrder` or `RejectedWell`.
    pub status: BoundaryStatus,
}

/// A well that donates its user pick of one top.
struct Donor {
    x_m: f64,
    y_m: f64,
    id: i32,
    elev_ft: f64,
}

fn has_xy(h: &WellHeader) -> bool {
    h.x_m.is_finite() && h.y_m.is_finite()
}

/// Estimates for every well (same order as `headers`) and canonical name
///. `None` where no estimate was attempted: the top is
/// present, fewer than `quorum` donors exist, or the well has no finite
/// coordinates. Wells run in parallel; each well's result depends only on its
/// own inputs and the donor lists.
pub(crate) fn estimate_tops(
    headers: &[WellHeader],
    geometry: &[WellGeometry],
    picks: &[WellPicks],
    n_names: usize,
    p: &TopsParams,
) -> Vec<Vec<Option<Estimate>>> {
    let donors: Vec<Vec<Donor>> = (0..n_names)
        .map(|t| {
            headers
                .iter()
                .zip(picks)
                .filter(|(h, _)| has_xy(h))
                .filter_map(|(h, wp)| {
                    let md = *wp.md_ft.get(t)?;
                    let elev_ft = *wp.elev_ft.get(t)?;
                    (md.is_finite() && elev_ft.is_finite()).then_some(Donor {
                        x_m: h.x_m,
                        y_m: h.y_m,
                        id: h.id.0,
                        elev_ft,
                    })
                })
                .collect()
        })
        .collect();
    headers
        .par_iter()
        .zip(geometry)
        .zip(picks)
        .map(|((h, g), wp)| {
            if !has_xy(h) {
                return vec![None; n_names];
            }
            let est = estimate_well(h, g, wp, &donors, p);
            let n_kept = est
                .iter()
                .flatten()
                .filter(|e| e.status == BoundaryStatus::Kept)
                .count();
            debug!(well = h.id.0, n_kept, "rough alignment: estimates");
            est
        })
        .collect()
}

fn estimate_well(
    h: &WellHeader,
    g: &WellGeometry,
    wp: &WellPicks,
    donors: &[Vec<Donor>],
    p: &TopsParams,
) -> Vec<Option<Estimate>> {
    let mut est: Vec<Option<Estimate>> = donors
        .iter()
        .enumerate()
        .map(|(t, d)| {
            let present = wp.present.get(t).copied().unwrap_or(false);
            (!present && d.len() >= p.quorum).then(|| estimate_one(h, g, wp, d, p))
        })
        .collect();
    fix_up(&mut est, wp);
    accept(&mut est, wp, p.min_own);
    est
}

/// For one `(well, top)`.
fn estimate_one(
    h: &WellHeader,
    g: &WellGeometry,
    wp: &WellPicks,
    donors: &[Donor],
    p: &TopsParams,
) -> Estimate {
    let missing = Estimate {
        md_ft: f64::NAN,
        elev_ft: f64::NAN,
        n_donors: 0,
        status: BoundaryStatus::Missing,
    };
    let all: Vec<(f64, i32, f64)> = donors
        .iter()
        .map(|d| (distance_m(h.x_m, h.y_m, d.x_m, d.y_m), d.id, d.elev_ft))
        .collect();
    let mut near = k_nearest(all, p.k);
    near.retain(|d| d.0 <= p.max_dist_m);
    if near.is_empty() || near.len() < p.quorum {
        return missing;
    }
    let Some(elev_ft) = weighted_lower_median(&near, p) else {
        return missing;
    };
    let md_ft = own_md_at_elev(wp, elev_ft).unwrap_or_else(|| md_at_elev(h, g, elev_ft));
    if !md_ft.is_finite() {
        return missing;
    }
    Estimate {
        md_ft,
        elev_ft,
        n_donors: u16::try_from(near.len()).unwrap_or(u16::MAX),
        status: BoundaryStatus::Kept,
    }
}

/// Weighted lower median of donor elevations.
///
/// `near` = `(distance_m, id, elev_ft)` in distance order (tie lower id).
/// Weights `1 / max(d, min_dist_m)^power`; donors are stable-sorted by
/// elevation descending (shallowest first) and the first donor whose
/// cumulative weight reaches half the total gives the estimate. The result is
/// always one donor's elevation.
fn weighted_lower_median(near: &[(f64, i32, f64)], p: &TopsParams) -> Option<f64> {
    let mut items: Vec<(f64, f64)> = near
        .iter()
        .map(|&(d, _, e)| (e, 1.0 / d.max(p.min_dist_m).powf(p.power)))
        .collect();
    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    let total: f64 = items.iter().map(|it| it.1).sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let half = 0.5 * total;
    let mut cum = 0.0;
    for &(elev, w) in &items {
        cum += w;
        if cum >= half {
            return Some(elev);
        }
    }
    items.last().map(|it| it.0)
}

/// MD of the well's own user pick at exactly elevation `elev_ft` (smallest MD
/// if several), so equal elevations give equal MDs.
fn own_md_at_elev(wp: &WellPicks, elev_ft: f64) -> Option<f64> {
    wp.md_ft
        .iter()
        .zip(&wp.elev_ft)
        .filter(|(md, e)| md.is_finite() && **e == elev_ft)
        .map(|(md, _)| *md)
        .reduce(f64::min)
}

/// Ordering fix-up against the well's own user picks only.
fn fix_up(est: &mut [Option<Estimate>], wp: &WellPicks) {
    let own: Vec<f64> = wp
        .md_ft
        .iter()
        .zip(&wp.elev_ft)
        .map(|(md, e)| if md.is_finite() { *e } else { f64::NAN })
        .collect();
    for (t, slot) in est.iter_mut().enumerate() {
        let Some(e) = slot.as_mut().filter(|e| e.status == BoundaryStatus::Kept) else {
            continue;
        };
        let shallower = own
            .iter()
            .take(t)
            .copied()
            .filter(|v| v.is_finite())
            .reduce(f64::min);
        let deeper = own
            .iter()
            .skip(t + 1)
            .copied()
            .filter(|v| v.is_finite())
            .reduce(f64::max);
        if shallower.is_some_and(|m| e.elev_ft > m) || deeper.is_some_and(|m| e.elev_ft < m) {
            e.status = BoundaryStatus::RejectedOrder;
        }
    }
}

/// Per-well acceptance: accept the surviving estimates only if the
/// merged set (own user picks + survivors, all names, no pads) has at least
/// `max(min_own, 2)` break groups and at least one survivor. Otherwise the
/// survivors become `RejectedWell` (already `RejectedOrder` ones stay).
fn accept(est: &mut [Option<Estimate>], wp: &WellPicks, min_own: usize) {
    let is_kept = |e: &Estimate| e.status == BoundaryStatus::Kept;
    if !est.iter().flatten().any(is_kept) {
        return;
    }
    let merged: Vec<f64> = est
        .iter()
        .enumerate()
        .map(|(t, slot)| {
            let own = wp.md_ft.get(t).copied().unwrap_or(f64::NAN);
            match slot {
                _ if own.is_finite() => own,
                Some(e) if is_kept(e) => e.md_ft,
                _ => f64::NAN,
            }
        })
        .collect();
    if build_breaks(&merged).md_ft.len() < min_own.max(2) {
        for e in est.iter_mut().flatten().filter(|e| is_kept(e)) {
            e.status = BoundaryStatus::RejectedWell;
        }
    }
}
