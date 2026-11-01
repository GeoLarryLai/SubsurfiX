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
