// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! The rough-alignment stage: input checks, canonical order, estimation,
//! boundaries, interval tables, layer coordinates and pairs.

use std::collections::{BTreeMap, BTreeSet};
use std::iter::once;

use rayon::prelude::*;
use tracing::{debug, info};

use crate::align::breaks::{build_breaks, interval_of_coord, layer_coord};
use crate::align::depth::elev_at_md;
use crate::align::estimate::{estimate_tops, Estimate, WellPicks};
use crate::align::intervals::{interval_table, IntervalInfo};
use crate::align::pairs::{build_pairs, PairMap, PairStatus, PairWell};
use crate::align::picks::{canonical_order, clean_picks, study_range, UserPick};
use crate::config::AlignParams;
use crate::dtw::IntervalParams;
use crate::error::{SxError, SxResult};
use crate::types::{GridRange, TopSet, TopSource, WellGeometry, WellHeader, WellId, WellLogs};

/// Inputs of [`rough_align`].
///
/// All slices sorted by `WellId`; `headers`, `geometry` and `logs` have the
/// same ids in the same order.
#[derive(Debug, Clone, Copy)]
pub struct AlignInput<'a> {
    /// Well headers (surface `x_m`, `y_m`, `kb_elev_ft`).
    pub headers: &'a [WellHeader],
    /// Per-sample geometry of each well.
    pub geometry: &'a [WellGeometry],
    /// QC curves (stage 2), before normalization; RESD in ohm·m.
    pub logs: &'a [WellLogs],
    /// User picks, any order.
    pub picks: &'a [UserPick],
    /// Top order set by the user (`ingest.top_names`), shallow to deep; empty =
    /// order by median pick elevation.
    pub order: &'a [String],
}

/// Status of one boundary row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundaryStatus {
    /// A boundary of the well (user pick, accepted estimate or pad).
    Kept = 0,
    /// Dropped by the break rule; the shallower boundary in depth won.
    OrderConflict = 1,
    /// No value.
    Missing = 2,
    /// Estimate dropped by the ordering fix-up.
    RejectedOrder = 3,
    /// Estimate dropped because the well was not accepted.
    RejectedWell = 4,
}

/// One row of `tops_estimated` (one per well per boundary `0..=n+1`).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryRow {
    /// Well id.
    pub well: WellId,
    /// Boundary index `0..=n+1` (0 = upper pad, `n + 1` = lower pad).
    pub boundary: u16,
    /// MD of the pick, estimate or pad, ft; NaN if none. Rejected estimates
    /// keep their computed MD.
    pub md_ft: f64,
    /// `elev_at_md(md_ft)`, ft; NaN if none.
    pub elev_ft: f64,
    /// Origin of `md_ft`; `None` = no value (255 in tables).
    pub source: Option<TopSource>,
    /// Kept, order conflict, missing or rejected.
    pub status: BoundaryStatus,
    /// Donors kept within `max_dist_m` for an estimate; 0 otherwise.
    pub n_donors: u16,
}

/// Where a well's layer coordinate comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CoordSource {
    /// No coordinate (fewer than 2 breaks).
    None = 0,
    /// Coordinate with at least one kept estimated boundary.
    WithEstimates = 1,
    /// Coordinate from user picks and pads only.
    Own = 2,
}

/// Layer coordinate of one well on its logs grid.
#[derive(Debug, Clone, PartialEq)]
pub struct WellCoord {
    /// Well id.
    pub well: WellId,
    /// The well's logs grid.
    pub grid: GridRange,
    /// Interval + fraction per sample; NaN where undefined.
    pub layer_coord: Vec<f64>,
    /// Interval per sample; −1 = none.
    pub interval: Vec<i32>,
    /// Coordinate source, same for every sample.
    pub source: CoordSource,
}

/// Output of [`rough_align`].
#[derive(Debug, Clone, PartialEq)]
pub struct RoughAlignment {
    /// Tops of the study range, shallow to deep.
    pub top_set: TopSet,
    /// Sorted by `(well, boundary)`.
    pub boundaries: Vec<BoundaryRow>,
    /// Sorted by `(well, interval)`.
    pub intervals: Vec<IntervalInfo>,
    /// Sorted by `(target, rank)`.
    pub pairs: Vec<PairMap>,
    /// Sorted by well.
    pub coords: Vec<WellCoord>,
}

fn check_input(input: &AlignInput<'_>) -> SxResult<()> {
    let n = input.headers.len();
    if n == 0 {
        return Err(SxError::EmptyInput("rough alignment: no wells".into()));
    }
    if input.geometry.len() != n || input.logs.len() != n {
        return Err(SxError::Shape(format!(
            "rough alignment: {n} headers, {} geometries, {} logs",
            input.geometry.len(),
            input.logs.len()
        )));
    }
    for (i, ((h, g), l)) in input
        .headers
        .iter()
        .zip(input.geometry)
        .zip(input.logs)
        .enumerate()
    {
        if g.id != h.id || l.id != h.id {
            return Err(SxError::InvalidInput(format!(
                "rough alignment: well ids differ at position {i}: header {}, geometry {}, logs {}",
                h.id.0, g.id.0, l.id.0
            )));
        }
        g.validate()?;
        l.validate()?;
    }
    if input.headers.windows(2).any(|w| match w {
        [a, b] => a.id >= b.id,
        _ => false,
    }) {
        return Err(SxError::InvalidInput(
            "rough alignment: well ids are not strictly ascending".into(),
        ));
    }
    Ok(())
}

/// Per-well results before pairing.
struct WellOut {
    boundaries: Vec<BoundaryRow>,
    intervals: Vec<IntervalInfo>,
    coord: WellCoord,
}

/// Boundaries, break table, interval table and coordinate of one well.
/// `study = (first, n)`: canonical index of the study top and number of tops
/// in the study range.
fn align_well(
    h: &WellHeader,
    g: &WellGeometry,
    logs: &WellLogs,
    wp: &WellPicks,
    est: &[Option<Estimate>],
    study: (usize, usize),
    p: &AlignParams,
) -> SxResult<WellOut> {
    let (first, n) = study;
    let est_at = |b: usize| -> Option<Estimate> {
        b.checked_sub(1)
            .filter(|&t| t < n)
            .and_then(|t| est.get(first + t).copied().flatten())
    };
    let tops: Vec<(f64, Option<TopSource>)> = (1..=n)
        .map(|b| {
            let user = wp.md_ft.get(first + b - 1).copied().unwrap_or(f64::NAN);
            match est_at(b) {
                _ if user.is_finite() => (user, Some(TopSource::User)),
                Some(e) if e.status == BoundaryStatus::Kept => {
                    (e.md_ft, Some(TopSource::Estimated))
                }
                _ => (f64::NAN, None),
            }
        })
        .collect();
    let pad = |top: Option<&(f64, Option<TopSource>)>, offset: f64| match top {
        Some(&(md, _)) if md.is_finite() => (md + offset, Some(TopSource::Pad)),
        _ => (f64::NAN, None),
    };
    let upper = pad(tops.first(), -p.pad_ft);
    let lower = pad(tops.last(), p.pad_ft);
    let (md, src): (Vec<f64>, Vec<Option<TopSource>>) =
        once(upper).chain(tops).chain(once(lower)).unzip();

    let breaks = build_breaks(&md);
    let is_kept = |b: usize| breaks.kept.get(b).copied().unwrap_or(false);
    let kept_md: Vec<f64> = (0..md.len())
        .map(|b| match md.get(b) {
            Some(&m) if is_kept(b) => m,
            _ => f64::NAN,
        })
        .collect();
    let kept_src: Vec<Option<TopSource>> = (0..src.len())
        .map(|b| src.get(b).copied().flatten().filter(|_| is_kept(b)))
        .collect();
    let intervals = interval_table(h.id, &kept_md, &kept_src, &p.confidence)?;

    let len = logs.grid.len;
    let coord = if breaks.has_coord() {
        let u = layer_coord(logs.grid, &breaks);
        let interval = u.iter().map(|&x| interval_of_coord(x, n)).collect();
        let source = if kept_src.contains(&Some(TopSource::Estimated)) {
            CoordSource::WithEstimates
        } else {
            CoordSource::Own
        };
        WellCoord {
            well: h.id,
            grid: logs.grid,
            layer_coord: u,
            interval,
            source,
        }
    } else {
        WellCoord {
            well: h.id,
            grid: logs.grid,
            layer_coord: vec![f64::NAN; len],
            interval: vec![-1; len],
            source: CoordSource::None,
        }
    };

    let mut boundaries = Vec::with_capacity(md.len());
    for (b, (&m, &s)) in md.iter().zip(&src).enumerate() {
        let boundary = u16::try_from(b)
            .map_err(|_| SxError::InvalidInput("rough alignment: too many boundaries".into()))?;
        let row = |md_ft: f64, source, status, n_donors| BoundaryRow {
            well: h.id,
            boundary,
            md_ft,
            elev_ft: if md_ft.is_finite() {
                elev_at_md(h, g, md_ft)
            } else {
                f64::NAN
            },
            source,
            status,
            n_donors,
        };
        let est_b = est_at(b);
        boundaries.push(if m.is_finite() {
            let status = if is_kept(b) {
                BoundaryStatus::Kept
            } else {
                BoundaryStatus::OrderConflict
            };
            let n_donors = match (s, est_b) {
                (Some(TopSource::Estimated), Some(e)) => e.n_donors,
                _ => 0,
            };
            row(m, s, status, n_donors)
        } else {
            match est_b {
                Some(e)
                    if matches!(
                        e.status,
                        BoundaryStatus::RejectedOrder | BoundaryStatus::RejectedWell
                    ) =>
                {
                    row(e.md_ft, Some(TopSource::Estimated), e.status, e.n_donors)
                }
                _ => row(f64::NAN, None, BoundaryStatus::Missing, 0),
            }
        });
    }
    debug!(
        well = h.id.0,
        n_breaks = breaks.md_ft.len(),
        coord_source = coord.source as u8,
        "rough alignment: well"
    );
    Ok(WellOut {
        boundaries,
        intervals,
        coord,
    })
}

/// Rough alignment of all wells.
///
/// Steps: input checks; pick cleaning (unknown wells dropped, smallest MD per
/// `(well, name)`); canonical top order by median pick elevation; missing-top
/// estimation over all canonical names (weighted lower median of donor
/// elevations, ordering fix-up, per-well acceptance); study range and MD pads;
/// break table, interval table and layer coordinate per well; nearest-well
/// pairs matched in interval mode (both wells have a coordinate) or depth mode.
/// Wells and pair targets run in parallel; outputs are collected in `WellId`
/// order, so they do not depend on the thread count.
///
/// Units: depths in ft, distances in m. NaN rule: skip (non-finite pick MDs
/// are not depths; NaN curve values are skipped by the matching).
///
/// Errors: [`SxError::EmptyInput`] (no wells, or no top name with a finite
/// pick elevation); [`SxError::Shape`] (slice lengths differ, a geometry or
/// logs vector length differs from its grid, or a curve has no provenance);
/// [`SxError::InvalidInput`] (ids differ between slices or are not strictly
/// ascending, or a blank top name lies in the study range);
/// [`SxError::Config`] (`pad_ft ≤ 0` or not finite, unknown or reversed study
/// names); errors of the DTW calls.
///
/// Complexity: `O(n_tops × n_wells × n_donors)` for estimation plus
/// `O(n_wells × k_pairs)` interval-mode or depth-mode DTW calls.
pub fn rough_align(
    input: &AlignInput<'_>,
    p: &AlignParams,
    dtw: &IntervalParams,
) -> SxResult<RoughAlignment> {
    check_input(input)?;
    if !p.pad_ft.is_finite() || p.pad_ft <= 0.0 {
        return Err(SxError::Config(format!(
            "align.pad_ft must be finite and > 0, got {}",
            p.pad_ft
        )));
    }
    let (headers, geometry, logs) = (input.headers, input.geometry, input.logs);

    let cleaned = clean_picks(headers, input.picks);
    let mut all_names: BTreeSet<&str> = BTreeSet::new();
    let mut pick_elevations: Vec<(String, f64)> = Vec::new();
    for ((h, g), picks) in headers.iter().zip(geometry).zip(&cleaned) {
        for (name, &md) in picks {
            all_names.insert(name.as_str());
            if md.is_finite() {
                pick_elevations.push((name.clone(), elev_at_md(h, g, md)));
            }
        }
    }
    let names = if input.order.is_empty() {
        canonical_order(&pick_elevations)
    } else {
        let have: BTreeSet<&str> = pick_elevations.iter().map(|(n, _)| n.as_str()).collect();
        input
            .order
            .iter()
            .filter(|n| have.contains(n.as_str()))
            .cloned()
            .collect()
    };
    if names.is_empty() {
        return Err(SxError::EmptyInput(
            "rough alignment: no top name has a finite pick".into(),
        ));
    }
    if all_names.len() > names.len() {
        info!(
            n_dropped = all_names.len() - names.len(),
            "rough alignment: top names without a finite pick dropped"
        );
    }
    let (first, last) = study_range(&names, &p.study_top, &p.study_base)?;
    let top_set = TopSet::new(names.get(first..=last).unwrap_or_default().to_vec())?;
    let n = top_set.len();
    info!(
        n_wells = headers.len(),
        n_names = names.len(),
        n_tops = n,
        "rough alignment: canonical order"
    );

    let index: BTreeMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let wpicks: Vec<WellPicks> = headers
        .iter()
        .zip(geometry)
        .zip(&cleaned)
        .map(|((h, g), picks)| {
            let mut wp = WellPicks {
                md_ft: vec![f64::NAN; names.len()],
                elev_ft: vec![f64::NAN; names.len()],
                present: vec![false; names.len()],
            };
            for (name, &md) in picks {
                let Some(&t) = index.get(name.as_str()) else {
                    continue;
                };
                if let (Some(m), Some(e), Some(pr)) = (
                    wp.md_ft.get_mut(t),
                    wp.elev_ft.get_mut(t),
                    wp.present.get_mut(t),
                ) {
                    *pr = true;
                    if md.is_finite() {
                        *m = md;
                        *e = elev_at_md(h, g, md);
                    }
                }
            }
            wp
        })
        .collect();

    let estimates = estimate_tops(headers, geometry, &wpicks, names.len(), &p.tops);
    let count = |s: BoundaryStatus| {
        estimates
            .iter()
            .flatten()
            .flatten()
            .filter(|e| e.status == s)
            .count()
    };
    let wells_estimated = estimates
        .iter()
        .filter(|w| w.iter().flatten().any(|e| e.status == BoundaryStatus::Kept))
        .count();
    info!(
        wells_estimated,
        kept = count(BoundaryStatus::Kept),
        missing = count(BoundaryStatus::Missing),
        rejected_order = count(BoundaryStatus::RejectedOrder),
        rejected_well = count(BoundaryStatus::RejectedWell),
        "rough alignment: missing-top estimation"
    );

    let per_well: Vec<SxResult<WellOut>> = (0..headers.len())
        .into_par_iter()
        .map(|w| {
            match (
                headers.get(w),
                geometry.get(w),
                logs.get(w),
                wpicks.get(w),
                estimates.get(w),
            ) {
                (Some(h), Some(g), Some(l), Some(wp), Some(est)) => {
                    align_well(h, g, l, wp, est, (first, n), p)
                }
                _ => Err(SxError::Shape(
                    "rough alignment: per-well inputs differ".into(),
                )),
            }
        })
        .collect();
    let per_well: Vec<WellOut> = per_well.into_iter().collect::<SxResult<_>>()?;
    let order_conflicts = per_well
        .iter()
        .flat_map(|w| &w.boundaries)
        .filter(|b| b.status == BoundaryStatus::OrderConflict)
        .count();
    let wells_with_coord = per_well
        .iter()
        .filter(|w| w.coord.source != CoordSource::None)
        .count();
    info!(
        order_conflicts,
        wells_with_coord, "rough alignment: boundaries and coordinates"
    );

    let pair_wells: Vec<PairWell<'_>> = headers
        .iter()
        .zip(geometry)
        .zip(logs)
        .zip(&per_well)
        .map(|(((h, g), l), w)| PairWell {
            header: h,
            geom: g,
            logs: l,
            coord: (w.coord.source != CoordSource::None).then_some(w.coord.layer_coord.as_slice()),
            intervals: &w.intervals,
        })
        .collect();
    let mut pairs = build_pairs(&pair_wells, p, dtw)?;
    pairs.sort_by_key(|m| (m.path.a, m.rank));
    let by_status = |s: PairStatus| pairs.iter().filter(|m| m.status == s).count();
    info!(
        interval_map = by_status(PairStatus::IntervalMap),
        depth_map = by_status(PairStatus::DepthMap),
        failed = by_status(PairStatus::Failed),
        "rough alignment: pairs"
    );

    let mut boundaries = Vec::new();
    let mut intervals = Vec::new();
    let mut coords = Vec::with_capacity(per_well.len());
    for w in per_well {
        boundaries.extend(w.boundaries);
        intervals.extend(w.intervals);
        coords.push(w.coord);
    }
    Ok(RoughAlignment {
        top_set,
        boundaries,
        intervals,
        pairs,
        coords,
    })
}
