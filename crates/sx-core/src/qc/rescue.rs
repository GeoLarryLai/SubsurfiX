// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Unit rescue: plan which channels were stored in the wrong unit and apply
//! the fixed factor.
//!
//! Rescue is the only value change in QC. The plan runs per log that has a
//! factor in `QcConfig::rescue_factors`; values are linear (the curve's
//! stored unit), taken after the tail step.

use std::collections::BTreeMap;

use rayon::prelude::*;

use super::stats::mean_std;
use crate::config::QcConfig;
use crate::types::{LogKind, WellId};
use crate::util::median;

/// Why a channel was or was not rescued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RescueReason {
    /// The log has no rescue factor.
    NoFactor,
    /// Fewer than `rescue_min_count` finite samples.
    TooFew,
    /// Fewer than `fleet_min_wells` wells qualify, or the fleet median is unusable.
    NoFleet,
    /// The level is not off in the direction the factor can fix.
    NotCandidate,
    /// `std / |mean|` is below `rescue_min_cv` (or NaN).
    DeadChannel,
    /// The converted level is outside the accept band.
    FactorOff,
    /// The channel is multiplied by the factor.
    Rescued,
}

impl RescueReason {
    /// Text used in `qc_report.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            RescueReason::NoFactor => "no_factor",
            RescueReason::TooFew => "too_few",
            RescueReason::NoFleet => "no_fleet",
            RescueReason::NotCandidate => "not_candidate",
            RescueReason::DeadChannel => "dead_channel",
            RescueReason::FactorOff => "factor_off",
            RescueReason::Rescued => "rescued",
        }
    }
}

/// One rescue decision. Numbers are linear units; NaN where not computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RescueRow {
    /// Well id.
    pub well: WellId,
    /// Log.
    pub log: LogKind,
    /// Decision.
    pub reason: RescueReason,
    /// Factor of this log (NaN if none).
    pub factor: f64,
    /// Finite samples.
    pub n: usize,
    /// Median of the finite samples (NaN if not computed).
    pub median: f64,
    /// `std / |mean|` (NaN if not computed).
    pub cv: f64,
    /// `median / fleet_med` (NaN if not computed).
    pub level_ratio: f64,
    /// Reference median used for the conversion check (NaN if not computed).
    pub ref_median: f64,
    /// Normal wells behind `ref_median`; 0 = fleet reference.
    pub n_ref: usize,
}

/// The rescue factor configured for `log`, matched case-insensitively on the
/// log name. `None` if the log has no valid factor.
pub fn rescue_factor(cfg: &QcConfig, log: LogKind) -> Option<f64> {
    cfg.rescue_factors
        .iter()
        .find(|(k, _)| LogKind::from_name(k).is_ok_and(|l| l == log))
        .map(|(_, &f)| f)
}

struct Own {
    n: usize,
    median: f64,
    mean: f64,
    std: f64,
}

/// Plan the rescue of one log across the fleet.
///
/// `channels` holds each well's linear values (NaN = missing). `xy` holds
/// well coordinates in meters; distances are Euclidean and neighbor ranking
/// sorts by `(distance, WellId)`. One row per input well, sorted by `WellId`.
///
/// Steps ([`RescueReason`] in order): no factor → `NoFactor`; `n <
/// rescue_min_count` → `TooFew`; fleet median (median of the medians of wells
/// with enough samples; needs `fleet_min_wells` such wells and a finite
/// positive value) else `NoFleet`; `level_ratio = median / fleet_med`;
/// normal wells have the ratio inside `rescue_accept_band` and known XY; a
/// candidate has `factor > 1` and ratio below the candidate band's low end,
/// or `factor < 1` and ratio above its high end, otherwise `NotCandidate`;
/// the reference is the median of the medians of the `rescue_ref_wells`
/// nearest normal wells when the candidate has XY and at least
/// `rescue_min_ref_wells` normal wells exist, else the fleet median
/// (`n_ref = 0`); then `DeadChannel` (`cv < rescue_min_cv` or NaN),
/// `FactorOff` (converted level outside the accept band), or `Rescued`.
///
/// NaN rule: skip. Duplicate ids in `channels` are not checked.
pub fn plan_rescue(
    log: LogKind,
    factor: Option<f64>,
    channels: &[(WellId, &[f64])],
    xy: &BTreeMap<WellId, (f64, f64)>,
    cfg: &QcConfig,
) -> Vec<RescueRow> {
    let mut order: Vec<usize> = (0..channels.len()).collect();
    order.sort_by_key(|&i| channels.get(i).map(|c| c.0));
    let blank = |well: WellId, reason: RescueReason, f: f64| RescueRow {
        well,
        log,
        reason,
        factor: f,
        n: 0,
        median: f64::NAN,
        cv: f64::NAN,
        level_ratio: f64::NAN,
        ref_median: f64::NAN,
        n_ref: 0,
    };
    let factor = match factor {
        Some(f) => f,
        None => {
            return order
                .iter()
                .filter_map(|&i| channels.get(i))
                .map(|(id, _)| blank(*id, RescueReason::NoFactor, f64::NAN))
                .collect();
        }
    };

    // Step 1: per-well own stats.
    let owns: Vec<(WellId, Own)> = order
        .par_iter()
        .filter_map(|&i| channels.get(i))
        .map(|(id, values)| {
            let (n, mean, std) = mean_std(values);
            let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
            (
                *id,
                Own {
                    n,
                    median: median(&finite),
                    mean,
                    std,
                },
            )
        })
        .collect();

    // Step 2: fleet median.
    let enough: Vec<f64> = owns
        .iter()
        .filter(|(_, o)| o.n >= cfg.rescue_min_count)
        .map(|(_, o)| o.median)
        .collect();
    let fleet_med = if enough.len() >= cfg.fleet_min_wells {
        median(&enough)
    } else {
        f64::NAN
    };
    let fleet_ok = fleet_med.is_finite() && fleet_med > 0.0;

    let [acc_lo, acc_hi] = cfg.rescue_accept_band;
    let [cand_lo, cand_hi] = cfg.rescue_candidate_band;

    // Step 3: normal wells (median, xy) for the local reference.
    let normal: Vec<(WellId, f64, f64, f64)> = if fleet_ok {
        owns.iter()
            .filter(|(_, o)| o.n >= cfg.rescue_min_count)
            .filter_map(|(id, o)| {
                let ratio = o.median / fleet_med;
                let (x, y) = *xy.get(id)?;
                (acc_lo <= ratio && ratio <= acc_hi && x.is_finite() && y.is_finite())
                    .then_some((*id, o.median, x, y))
            })
            .collect()
    } else {
        Vec::new()
    };

    owns.iter()
        .map(|(id, o)| {
            let mut row = blank(*id, RescueReason::NoFactor, factor);
            row.n = o.n;
            if o.n < cfg.rescue_min_count {
                row.reason = RescueReason::TooFew;
                return row;
            }
            row.median = o.median;
            row.cv = if o.mean == 0.0 {
                f64::NAN
            } else {
                o.std / o.mean.abs()
            };
            if !fleet_ok {
                row.reason = RescueReason::NoFleet;
                return row;
            }
            row.level_ratio = o.median / fleet_med;
            let candidate = (factor > 1.0 && row.level_ratio < cand_lo)
                || (factor < 1.0 && row.level_ratio > cand_hi);
            if !candidate {
                row.reason = RescueReason::NotCandidate;
                return row;
            }
            // Reference.
            let own_xy = xy
                .get(id)
                .copied()
                .filter(|(x, y)| x.is_finite() && y.is_finite());
            match own_xy {
                Some((x, y)) if normal.len() >= cfg.rescue_min_ref_wells => {
                    let mut ranked: Vec<(f64, WellId, f64)> = normal
                        .iter()
                        .map(|&(nid, med, nx, ny)| ((nx - x).hypot(ny - y), nid, med))
                        .collect();
                    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    let take = cfg.rescue_ref_wells.min(ranked.len());
                    let meds: Vec<f64> = ranked.iter().take(take).map(|r| r.2).collect();
                    row.ref_median = median(&meds);
                    row.n_ref = meds.len();
                }
                _ => {
                    row.ref_median = fleet_med;
                    row.n_ref = 0;
                }
            }
            // Decision.
            if row.cv.is_nan() || row.cv < cfg.rescue_min_cv {
                row.reason = RescueReason::DeadChannel;
                return row;
            }
            let conv = o.median * factor / row.ref_median;
            row.reason = if acc_lo <= conv && conv <= acc_hi {
                RescueReason::Rescued
            } else {
                RescueReason::FactorOff
            };
            row
        })
        .collect()
}

/// Multiply every finite sample by `factor`: `v = (v as f64 * factor) as f32`.
///
/// NaN rule: skip. The caller re-checks physical bounds afterwards
/// (`bounds_mask` with `QC_OUT_OF_BOUNDS`).
pub fn apply_rescue(values: &mut [f32], factor: f64) {
    for v in values.iter_mut() {
        if v.is_finite() {
            *v = (f64::from(*v) * factor) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Chan = (WellId, Vec<f64>);
    type Xy = BTreeMap<WellId, (f64, f64)>;

    /// 101 evenly spaced samples with median `m` and cv about 0.346.
    fn chan(m: f64) -> Vec<f64> {
        (0..=100)
            .map(|i| m * (0.4 + 1.2 * f64::from(i) / 100.0))
            .collect()
    }

    /// Example 5 scenario: fleet median 60; A at the origin, B next to it,
    /// six nearest normal wells with medians [55, 58, 62, 60, 70, 52].
    fn scenario() -> (Vec<Chan>, Xy) {
        let mut chans = Vec::new();
        let mut xy = BTreeMap::new();
        chans.push((WellId(1), chan(4.0))); // A
        xy.insert(WellId(1), (0.0, 0.0));
        chans.push((WellId(2), chan(1.5))); // B
        xy.insert(WellId(2), (0.0, -0.5));
        for (j, m) in [55.0, 58.0, 62.0, 60.0, 70.0, 52.0].iter().enumerate() {
            let id = WellId(10 + j as i32);
            chans.push((id, chan(*m)));
            xy.insert(id, (1.0 + j as f64, 0.0));
        }
        for j in 0..4 {
            let id = WellId(20 + j);
            chans.push((id, chan(60.0)));
            xy.insert(id, (100.0 + f64::from(j), 0.0));
        }
        (chans, xy)
    }

    fn plan(
        chans: &[(WellId, Vec<f64>)],
        xy: &BTreeMap<WellId, (f64, f64)>,
        factor: Option<f64>,
    ) -> Vec<RescueRow> {
        let view: Vec<(WellId, &[f64])> = chans.iter().map(|(i, v)| (*i, v.as_slice())).collect();
        plan_rescue(LogKind::Gr, factor, &view, xy, &QcConfig::default())
    }

    #[test]
    fn example_5() {
        let (chans, xy) = scenario();
        let rows = plan(&chans, &xy, Some(14.0));
        assert_eq!(rows.len(), chans.len());
        let a = &rows[0];
        let b = &rows[1];
        assert_eq!(a.well, WellId(1));
        assert_eq!(a.reason, RescueReason::Rescued);
        assert_eq!(a.n_ref, 6);
        assert!((a.ref_median - 59.0).abs() < 1e-9);
        assert!((a.level_ratio - 4.0 / 60.0).abs() < 1e-9);
        assert!((a.median * 14.0 / a.ref_median - 56.0 / 59.0).abs() < 1e-9);
        assert!((a.cv - 0.357).abs() < 0.02, "{}", a.cv);
        assert_eq!(b.reason, RescueReason::FactorOff);
        assert!((b.median * 14.0 / b.ref_median - 21.0 / 59.0).abs() < 1e-9);
        for r in &rows[2..] {
            assert_eq!(r.reason, RescueReason::NotCandidate);
        }
    }

    #[test]
    fn six_times_too_high_is_not_candidate() {
        let (mut chans, xy) = scenario();
        chans.push((WellId(30), chan(360.0)));
        let mut xy = xy;
        xy.insert(WellId(30), (5.0, 5.0));
        let rows = plan(&chans, &xy, Some(14.0));
        let r = rows.iter().find(|r| r.well == WellId(30)).unwrap();
        assert_eq!(r.reason, RescueReason::NotCandidate);
        // a well 14x too high is fixed by the inverse factor
        chans.pop();
        chans.push((WellId(30), chan(840.0)));
        let rows = plan(&chans, &xy, Some(1.0 / 14.0));
        let r = rows.iter().find(|r| r.well == WellId(30)).unwrap();
        assert_eq!(r.reason, RescueReason::Rescued);
    }

    #[test]
    fn other_reasons() {
        let (chans, xy) = scenario();
        let rows = plan(&chans, &xy, None);
        assert!(rows.iter().all(|r| r.reason == RescueReason::NoFactor));
        assert!(rows.iter().all(|r| r.factor.is_nan()));

        // too few samples
        let mut c2 = chans.clone();
        c2[0].1.truncate(50);
        let rows = plan(&c2, &xy, Some(14.0));
        assert_eq!(rows[0].reason, RescueReason::TooFew);
        assert_eq!(rows[0].n, 50);

        // no fleet: only two wells with enough samples
        let rows = plan(&chans[..2], &xy, Some(14.0));
        assert!(rows.iter().all(|r| r.reason == RescueReason::NoFleet));

        // dead channel: flat channel at a low level
        let mut c3 = chans.clone();
        c3[0].1 = vec![4.0; 101];
        let rows = plan(&c3, &xy, Some(14.0));
        assert_eq!(rows[0].reason, RescueReason::DeadChannel);

        // no xy -> fleet reference (n_ref = 0): 56/60 is accepted
        let rows = plan(&chans, &BTreeMap::new(), Some(14.0));
        assert_eq!(rows[0].reason, RescueReason::Rescued);
        assert_eq!(rows[0].n_ref, 0);
        assert!((rows[0].ref_median - 60.0).abs() < 1e-9);
    }

    #[test]
    fn apply() {
        let mut v = [4.0f32, f32::NAN, 0.5];
        apply_rescue(&mut v, 14.0);
        assert_eq!(v[0], 56.0);
        assert!(v[1].is_nan());
        assert_eq!(v[2], 7.0);
    }

    #[test]
    fn factor_lookup() {
        let cfg = QcConfig::default();
        assert_eq!(rescue_factor(&cfg, LogKind::Gr), Some(14.0));
        assert_eq!(rescue_factor(&cfg, LogKind::Nphi), Some(100.0));
        assert_eq!(rescue_factor(&cfg, LogKind::Rhob), None);
    }
}
