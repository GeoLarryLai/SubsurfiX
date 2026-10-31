// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Interval facts, pair qualities, row weights and the donor score.

use std::collections::{BTreeMap, BTreeSet};

use crate::align::{PairMap, PairStatus};
use crate::config::NeighborsConfig;
use crate::error::{SxError, SxResult};
use crate::neighbors::mapping::WellView;
use crate::neighbors::search::{may_donate, surface_distance_m, LogStats};
use crate::neighbors::NeighborInputs;
use crate::types::{
    IntervalRow, IntervalState, LogKind, TopPick, TopSet, TopSource, WellGeometry, WellId,
    GRID_STEP_FT,
};

/// Facts of one interval of one well.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalFacts {
    /// State from the interval table.
    pub state: IntervalState,
    /// TVD thickness, feet; NaN if unknown.
    pub tvd_thickness_ft: f64,
    /// Confidence in [0, 1].
    pub confidence: f64,
}

impl IntervalFacts {
    /// Not `Pinched` / `Unknown`, and a finite TVD thickness `> 0`.
    pub(crate) fn usable(&self) -> bool {
        !matches!(self.state, IntervalState::Pinched | IntervalState::Unknown)
            && self.tvd_thickness_ft.is_finite()
            && self.tvd_thickness_ft > 0.0
    }
}

/// TVD at `md_ft`, linear between grid samples of `geom.tvd_ft`; NaN if `md_ft`
/// is NaN or outside the grid.
fn tvd_at_md(geom: &WellGeometry, md_ft: f64) -> f64 {
    if !md_ft.is_finite() || geom.grid.len == 0 {
        return f64::NAN;
    }
    let pos = md_ft / GRID_STEP_FT - f64::from(geom.grid.k_start);
    let last = (geom.grid.len - 1) as f64;
    if !(0.0..=last).contains(&pos) {
        return f64::NAN;
    }
    let j = pos.floor();
    let frac = pos - j;
    let j = j as usize;
    let Some(&a) = geom.tvd_ft.get(j) else {
        return f64::NAN;
    };
    if frac == 0.0 {
        return a;
    }
    match geom.tvd_ft.get(j + 1) {
        Some(&b) => a + frac * (b - a),
        None => f64::NAN,
    }
}

/// One entry per interval `0..=n` (index = interval id).
///
/// State from `rows`; TVD thickness = `tvd(base_md_ft) − tvd(top_md_ft)` with
/// `tvd` linear between grid samples of `geom.tvd_ft` (NaN if either MD is NaN
/// or outside the grid); confidence by boundary source (`Observed` → `conf_observed`;
/// `Estimated` with one or more `User` boundary → `conf_estimated_one_user`, none
/// → `conf_estimated_no_user`; `Pad` → the adjacent non-pad interval's
/// confidence if `> 0`, else `conf_pad_floor`; `Pinched` / `Unknown` → 0).
/// Boundary `b` in `1..=n` is top `b − 1`; it is a `User` boundary if `picks`
/// holds a `User` pick of that top for well `geom.id`. Units: ft (TVD).
/// NaN rule: propagate (NaN MD gives NaN thickness). Errors: `Shape` if `rows`
/// does not hold exactly one row per interval of `tops`, or `geom` fails its
/// length check. Complexity: `O(n + picks)`.
pub fn interval_facts(
    rows: &[IntervalRow],
    picks: &[TopPick],
    tops: &TopSet,
    geom: &WellGeometry,
    cfg: &NeighborsConfig,
) -> SxResult<Vec<IntervalFacts>> {
    geom.validate()?;
    let n_int = tops.n_intervals();
    let n = tops.len();
    if rows.len() != n_int {
        return Err(SxError::Shape(format!(
            "well {}: {} interval rows for {n_int} intervals",
            geom.id.0,
            rows.len()
        )));
    }
    let mut by_interval: Vec<Option<&IntervalRow>> = vec![None; n_int];
    for row in rows {
        match by_interval.get_mut(usize::from(row.interval)) {
            Some(slot @ None) => *slot = Some(row),
            _ => {
                return Err(SxError::Shape(format!(
                    "well {}: interval {} repeated or outside 0..={}",
                    geom.id.0, row.interval, n
                )))
            }
        }
    }
    let user_tops: BTreeSet<u16> = picks
        .iter()
        .filter(|p| p.well == geom.id && p.source == TopSource::User)
        .map(|p| p.top)
        .collect();
    let is_user = |b: usize| {
        (1..=n).contains(&b) && u16::try_from(b - 1).is_ok_and(|t| user_tops.contains(&t))
    };
    let mut facts = Vec::with_capacity(n_int);
    for (i, row) in by_interval.iter().enumerate() {
        let Some(row) = row else {
            return Err(SxError::Shape(format!(
                "well {}: interval {i} has no row",
                geom.id.0
            )));
        };
        let thickness = tvd_at_md(geom, row.base_md_ft) - tvd_at_md(geom, row.top_md_ft);
        let n_user = usize::from(is_user(i)) + usize::from(is_user(i + 1));
        let confidence = match row.state {
            IntervalState::Observed => cfg.conf_observed,
            IntervalState::Estimated if n_user >= 1 => cfg.conf_estimated_one_user,
            IntervalState::Estimated => cfg.conf_estimated_no_user,
            IntervalState::Pad => cfg.conf_pad_floor,
            IntervalState::Pinched | IntervalState::Unknown => 0.0,
        };
        facts.push(IntervalFacts {
            state: row.state,
            tvd_thickness_ft: thickness,
            confidence,
        });
    }
    let pad_conf = |i: usize, facts: &[IntervalFacts]| -> f64 {
        let adjacent = if i == 0 {
            Some(1)
        } else if i == n {
            n.checked_sub(1)
        } else {
            None
        };
        match adjacent.and_then(|a| facts.get(a).map(|f| (a, f))) {
            Some((a, f))
                if a != 0 && a != n && f.state != IntervalState::Pad && f.confidence > 0.0 =>
            {
                f.confidence
            }
            _ => cfg.conf_pad_floor,
        }
    };
    let pads: Vec<(usize, f64)> = facts
        .iter()
        .enumerate()
        .filter(|(_, f)| f.state == IntervalState::Pad)
        .map(|(i, _)| (i, pad_conf(i, &facts)))
        .collect();
    for (i, c) in pads {
        if let Some(f) = facts.get_mut(i) {
            f.confidence = c;
        }
    }
    Ok(facts)
}

/// Pair qualities keyed by (a, b, interval).
///
/// Interval keys come from `PairMap.intervals` and `interval_quality`.
/// A depth-map pair (no intervals) is stored with `interval = None` and the
/// pair's own `path.quality`. Failed pairs are not stored (they count as absent).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairQuality {
    map: BTreeMap<(WellId, WellId, Option<u16>), f32>,
}

impl PairQuality {
    /// Build from the rough-alignment pair maps.
    ///
    /// NaN rule: error (a NaN quality is outside [0, 1]). Errors: `InvalidInput`
    /// on a repeated (a, b, interval) key or a quality outside [0, 1]; `Shape` if
    /// `intervals` and `interval_quality` differ in length. Empty input gives an
    /// empty table. Complexity: `O(k log k)` for `k` keys.
    pub fn from_pairs(pairs: &[PairMap]) -> SxResult<Self> {
        let mut map = BTreeMap::new();
        let mut insert = |key: (WellId, WellId, Option<u16>), q: f32| -> SxResult<()> {
            if !(0.0..=1.0).contains(&q) {
                return Err(SxError::InvalidInput(format!(
                    "pair ({}, {}, {:?}): quality {q} outside [0, 1]",
                    key.0 .0, key.1 .0, key.2
                )));
            }
            if map.insert(key, q).is_some() {
                return Err(SxError::InvalidInput(format!(
                    "pair ({}, {}, {:?}) repeated",
                    key.0 .0, key.1 .0, key.2
                )));
            }
            Ok(())
        };
        for m in pairs {
            let (a, b) = (m.path.a, m.path.b);
            match m.status {
                PairStatus::Failed => {}
                PairStatus::DepthMap => insert((a, b, None), m.path.quality)?,
                PairStatus::IntervalMap => {
                    if m.intervals.len() != m.interval_quality.len() {
                        return Err(SxError::Shape(format!(
                            "pair ({}, {}): {} intervals but {} interval qualities",
                            a.0,
                            b.0,
                            m.intervals.len(),
                            m.interval_quality.len()
                        )));
                    }
                    for (&i, &q) in m.intervals.iter().zip(&m.interval_quality) {
                        insert((a, b, Some(i)), q)?;
                    }
                }
            }
        }
        Ok(Self { map })
    }

    /// (target, donor, interval), else (donor, target, interval), else `None`.
    ///
    /// Unitless quality in [0, 1]. NaN rule: none (stored qualities are finite).
    /// Errors: none. Complexity: `O(log k)`.
    pub fn get(&self, target: WellId, donor: WellId, interval: Option<u16>) -> Option<f32> {
        self.map
            .get(&(target, donor, interval))
            .or_else(|| self.map.get(&(donor, target, interval)))
            .copied()
    }
}

/// `1 / max(dist_m, min_dist_m)^idw_power`.
pub(crate) fn dist_factor(dist_m: f64, cfg: &NeighborsConfig) -> f64 {
    1.0 / dist_m.max(cfg.min_dist_m).powf(cfg.idw_power)
}

/// `iq` of a coordinate-mapped row in interval `i`.
pub(crate) fn interval_iq(
    target: &WellView<'_>,
    donor: &WellView<'_>,
    stats: &LogStats,
    i: i64,
) -> f64 {
    let (Some(ft), Some(fd)) = (target.fact(i), donor.fact(i)) else {
        return 0.0;
    };
    if !ft.usable() || !fd.usable() {
        return 0.0;
    }
    let (ht, hd) = (ft.tvd_thickness_ft, fd.tvd_thickness_ft);
    let thick_match = ht.min(hd) / ht.max(hd);
    thick_match * (ft.confidence.max(0.0) * fd.confidence.max(0.0)).sqrt() * stats.cov(i)
}

/// `aq` for interval `Some(i)` (coordinate-mapped) or `None` (elevation-matched).
pub(crate) fn pair_aq(
    pairs: &PairQuality,
    target: WellId,
    donor: WellId,
    interval: Option<i64>,
    cfg: &NeighborsConfig,
) -> f64 {
    let key = match interval {
        Some(i) => match u16::try_from(i) {
            Ok(i) => Some(i),
            Err(_) => return cfg.aq_unaligned,
        },
        None => None,
    };
    pairs
        .get(target, donor, key)
        .map_or(cfg.aq_unaligned, f64::from)
}

/// Row weights of one stacked donor, cached per interval key.
pub(crate) struct RowWeights {
    factor: f64,
    cache: BTreeMap<Option<i64>, (f64, f64)>,
}

impl RowWeights {
    pub(crate) fn new(dist_m: f64, cfg: &NeighborsConfig) -> Self {
        Self {
            factor: dist_factor(dist_m, cfg),
            cache: BTreeMap::new(),
        }
    }

    /// `(w, q)` for a row with interval key `key` (`None` = elevation-matched).
    pub(crate) fn get(
        &mut self,
        key: Option<i64>,
        target: &WellView<'_>,
        donor: &WellView<'_>,
        stats: &LogStats,
        pairs: &PairQuality,
        cfg: &NeighborsConfig,
    ) -> (f64, f64) {
        if let Some(&wq) = self.cache.get(&key) {
            return wq;
        }
        let iq = match key {
            Some(i) => interval_iq(target, donor, stats, i),
            None => cfg.iq_no_coord,
        };
        let aq = pair_aq(pairs, target.id, donor.id, key, cfg).max(0.0);
        let q = iq * aq;
        let wq = (self.factor * q, q);
        self.cache.insert(key, wq);
        wq
    }
}

/// Whole-interval donor score `1/max(d,1)^p × iq × max(aq,0)` for interval `Some(i)`,
/// or the no-coordinate score (`iq_no_coord`, pair quality with interval `None`) for `None`.
///
/// `d` is the surface distance (m); the floor and power are `min_dist_m` and
/// `idw_power`. A donor that may not donate `log` to `target` (the target itself,
/// an excluded channel, a holdout well) or with a non-finite distance scores 0.
/// NaN rule: skip (missing facts or coverage give `iq = 0`). Errors: `NotFound`
/// if `target` or `donor` is missing from `headers` or `logs`; `Shape` /
/// `InvalidInput` from the donor's input checks. Complexity: `O(donor grid)`.
pub fn donor_score(
    inputs: &NeighborInputs<'_>,
    target: WellId,
    donor: WellId,
    log: LogKind,
    interval: Option<u16>,
    cfg: &NeighborsConfig,
) -> SxResult<f64> {
    for id in [target, donor] {
        if !inputs.headers.contains_key(&id) || !inputs.logs.contains_key(&id) {
            return Err(SxError::NotFound(format!(
                "donor score: well {} missing from headers or logs",
                id.0
            )));
        }
    }
    if !may_donate(inputs, target, donor, log) {
        return Ok(0.0);
    }
    let dist = surface_distance_m(inputs, target, donor);
    if !dist.is_finite() {
        return Ok(0.0);
    }
    let factor = dist_factor(dist, cfg);
    let (iq, key) = match interval {
        Some(i) => {
            let t = WellView::new(inputs, target)?;
            let d = WellView::new(inputs, donor)?;
            let iq = match LogStats::new(&d, log, cfg) {
                Some(stats) => interval_iq(&t, &d, &stats, i64::from(i)),
                None => 0.0,
            };
            (iq, Some(i64::from(i)))
        }
        None => (cfg.iq_no_coord, None),
    };
    let aq = pair_aq(inputs.pairs, target, donor, key, cfg).max(0.0);
    Ok(factor * iq * aq)
}
