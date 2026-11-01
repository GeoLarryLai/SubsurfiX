// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Sigma test with two rings and the unit-mismatch flag.
//!
//! Each well's channel stats are compared with a local baseline (its nearest
//! wells) or the fleet baseline. A second, wider ring guards against a
//! biased near cluster. All statistics are native space.

use std::collections::BTreeMap;

use rayon::prelude::*;

use super::stats::{fleet_summary, mad_n, ChannelStats, FleetSummary};
use crate::config::QcConfig;
use crate::types::{LogKind, WellId};
use crate::util::median;

/// Own bit: mean deviates.
pub const SIGMA_MEAN: u8 = 1;
/// Own bit: lower percentile deviates.
pub const SIGMA_P_LOW: u8 = 2;
/// Own bit: upper percentile deviates.
pub const SIGMA_P_HIGH: u8 = 4;
/// The well sits in a near cluster found biased by the outer ring.
pub const SIGMA_CLUSTER: u8 = 8;
/// The median is off by a unit-like factor and the mean deviates.
pub const SIGMA_UNIT: u8 = 16;

/// Which baseline a row was tested against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    /// Median / MAD of the nearest wells.
    Local,
    /// Fleet summary.
    Fleet,
    /// No fleet summary was possible; nothing was tested.
    None,
}

impl Baseline {
    /// Text used in `qc_report.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            Baseline::Local => "local",
            Baseline::Fleet => "fleet",
            Baseline::None => "none",
        }
    }
}

/// Ring-test result of one well and log.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingRow {
    /// Well id.
    pub well: WellId,
    /// Log.
    pub log: LogKind,
    /// `SIGMA_*` bits.
    pub bits: u8,
    /// Largest of the three own deviations (MAD units, before clearing); NaN if untested.
    pub max_dev: f64,
    /// Largest cluster deviation pointing at this well (MAD units); NaN if none.
    pub cluster_dev: f64,
    /// Baseline used.
    pub baseline: Baseline,
    /// Median ratio to the fleet median (linear ratio; `10^diff` for RESD); NaN if undefined.
    pub unit_ratio: f64,
}

/// Per-row intermediate result of phases 3 to 5.
struct Phase {
    own_bits: u8,
    dev_mean: f64,
    max_dev: f64,
    baseline: Baseline,
    cleared: bool,
    /// Wells to receive `cluster_dev` (indices into the sorted rows) and the value.
    cluster: Option<(Vec<usize>, f64)>,
}

fn dev(x: f64, med: f64, mad: f64) -> f64 {
    (x - med).abs() / mad
}

/// Run the sigma test of one log.
///
/// `rows` are the stats rows (native space); wells are sorted by `WellId`
/// internally and one row is returned per input row in that order. `xy`
/// gives meters; distances are Euclidean and rankings sort by
/// `(distance, WellId)`, counting only wells that have stats for this log.
///
/// 1. No fleet summary (`fleet_min_wells`): every row has `Baseline::None`,
///    bits 0, NaN numbers.
/// 2. Fewer than `near_min + 1` rows with XY: everyone uses the fleet baseline.
/// 3. A well without XY uses the fleet baseline. Otherwise the `near_count`
///    nearest other wells with XY form `near`; fewer than `near_min` → fleet
///    baseline, else the local baseline is the median and `MAD_n` (floored)
///    of their mean, `p_low`, `p_high`.
/// 4. `dev_x = |x_w - median_x| / mad_x`; bits `SIGMA_MEAN`, `SIGMA_P_LOW`,
///    `SIGMA_P_HIGH` where `dev_x > dev_limit`.
/// 5. Two-scale consensus: with a local baseline, own bits, and at least
///    `ring_min` wells at ranks `ring_first..=ring_last` (1-based, clipped),
///    `c = |median(near means) - median(ring means)| / max(MAD_n(ring
///    means), mad_floor)`. If `c > dev_limit` the well is cleared (own bits
///    dropped) and every near well gets `cluster_dev = max(.., c)`.
/// 6. Final bits: own bits unless cleared, plus `SIGMA_CLUSTER` where a
///    cluster deviation points at the well, plus `SIGMA_UNIT`. All phases run
///    for all wells first, so the result does not depend on input order.
/// 7. `SIGMA_UNIT` when `unit_ratio` is finite and strictly outside
///    `unit_band` and the own `dev_mean > dev_limit`.
///
/// NaN rule: skip (NaN stats make the dev NaN, which never exceeds the limit).
/// O(n² log n) for `n` wells (parallel over wells).
pub fn ring_test(
    log: LogKind,
    rows: &[ChannelStats],
    xy: &BTreeMap<WellId, (f64, f64)>,
    cfg: &QcConfig,
) -> Vec<RingRow> {
    let mut sorted: Vec<ChannelStats> = rows.to_vec();
    sorted.sort_by_key(|r| r.well);
    let n = sorted.len();
    let fleet = fleet_summary(log, &sorted, cfg.fleet_min_wells, cfg.mad_floor);
    let fleet = match fleet {
        Some(f) => f,
        None => {
            return sorted
                .iter()
                .map(|r| RingRow {
                    well: r.well,
                    log,
                    bits: 0,
                    max_dev: f64::NAN,
                    cluster_dev: f64::NAN,
                    baseline: Baseline::None,
                    unit_ratio: f64::NAN,
                })
                .collect();
        }
    };

    // Rows with XY (indices into `sorted`, ascending = WellId order).
    let pos: Vec<Option<(f64, f64)>> = sorted
        .iter()
        .map(|r| {
            xy.get(&r.well)
                .copied()
                .filter(|(x, y)| x.is_finite() && y.is_finite())
        })
        .collect();
    let with_xy: Vec<usize> = (0..n)
        .filter(|&i| pos.get(i).is_some_and(Option::is_some))
        .collect();
    let fleet_mode = with_xy.len() < cfg.near_min + 1;

    let phases: Vec<Phase> = (0..n)
        .into_par_iter()
        .map(|i| phase(i, &sorted, &pos, &with_xy, fleet_mode, &fleet, cfg))
        .collect();

    // Reduce cluster deviations (max is order independent).
    let mut cluster_dev = vec![f64::NAN; n];
    for p in &phases {
        if let Some((targets, c)) = &p.cluster {
            for &u in targets {
                if let Some(slot) = cluster_dev.get_mut(u) {
                    if slot.is_nan() || *c > *slot {
                        *slot = *c;
                    }
                }
            }
        }
    }

    // Unit flag.
    let medians: Vec<f64> = sorted.iter().map(|r| r.median).collect();
    let fleet_median = median(&medians);

    sorted
        .iter()
        .zip(phases.iter())
        .zip(cluster_dev.iter())
        .map(|((r, p), &cd)| {
            let unit_ratio = unit_ratio(log, r.median, fleet_median);
            let [u_lo, u_hi] = cfg.unit_band;
            let unit_flag = unit_ratio.is_finite()
                && (unit_ratio < u_lo || unit_ratio > u_hi)
                && p.dev_mean > cfg.dev_limit;
            let mut bits = if p.cleared { 0 } else { p.own_bits };
            if !cd.is_nan() {
                bits |= SIGMA_CLUSTER;
            }
            if unit_flag {
                bits |= SIGMA_UNIT;
            }
            RingRow {
                well: r.well,
                log,
                bits,
                max_dev: p.max_dev,
                cluster_dev: cd,
                baseline: p.baseline,
                unit_ratio,
            }
        })
        .collect()
}

fn unit_ratio(log: LogKind, median_w: f64, fleet_median: f64) -> f64 {
    if crate::units::is_log10(log) {
        if median_w.is_finite() && fleet_median.is_finite() {
            10f64.powf(median_w - fleet_median)
        } else {
            f64::NAN
        }
    } else if median_w > 0.0
        && fleet_median > 0.0
        && median_w.is_finite()
        && fleet_median.is_finite()
    {
        median_w / fleet_median
    } else {
        f64::NAN
    }
}

/// Phases 3 to 5 for row `i`.
fn phase(
    i: usize,
    sorted: &[ChannelStats],
    pos: &[Option<(f64, f64)>],
    with_xy: &[usize],
    fleet_mode: bool,
    fleet: &FleetSummary,
    cfg: &QcConfig,
) -> Phase {
    let fleet_phase = |row: &ChannelStats| {
        let d_mean = dev(row.mean, fleet.mean_med, fleet.mean_mad);
        let d_low = dev(row.p_low, fleet.low_med, fleet.low_mad);
        let d_high = dev(row.p_high, fleet.high_med, fleet.high_mad);
        finish(Baseline::Fleet, d_mean, d_low, d_high, cfg)
    };
    let row = match sorted.get(i) {
        Some(r) => r,
        None => return finish(Baseline::None, f64::NAN, f64::NAN, f64::NAN, cfg),
    };
    let own_pos = match pos.get(i).copied().flatten() {
        Some(p) if !fleet_mode => p,
        _ => return fleet_phase(row),
    };
    // R = rows with XY minus self, ranked by (distance, WellId); `sorted`
    // is in WellId order so the index breaks distance ties by WellId.
    let mut ranked: Vec<(f64, usize)> = with_xy
        .iter()
        .filter(|&&j| j != i)
        .filter_map(|&j| {
            let (x, y) = pos.get(j).copied().flatten()?;
            Some(((x - own_pos.0).hypot(y - own_pos.1), j))
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let near: Vec<usize> = ranked
        .iter()
        .take(cfg.near_count)
        .map(|&(_, j)| j)
        .collect();
    if near.len() < cfg.near_min {
        return fleet_phase(row);
    }
    let col = |idx: &[usize], f: fn(&ChannelStats) -> f64| -> Vec<f64> {
        idx.iter().filter_map(|&j| sorted.get(j)).map(f).collect()
    };
    let near_mean = col(&near, |r| r.mean);
    let near_low = col(&near, |r| r.p_low);
    let near_high = col(&near, |r| r.p_high);
    let med_mean = median(&near_mean);
    let d_mean = dev(row.mean, med_mean, mad_n(&near_mean, cfg.mad_floor));
    let d_low = dev(
        row.p_low,
        median(&near_low),
        mad_n(&near_low, cfg.mad_floor),
    );
    let d_high = dev(
        row.p_high,
        median(&near_high),
        mad_n(&near_high, cfg.mad_floor),
    );
    let mut ph = finish(Baseline::Local, d_mean, d_low, d_high, cfg);

    // Two-scale consensus.
    if ph.own_bits != 0 {
        let lo = cfg.ring_first.max(1) - 1;
        let hi = cfg.ring_last.min(ranked.len());
        let ring: Vec<usize> = if lo < hi {
            ranked
                .get(lo..hi)
                .unwrap_or(&[])
                .iter()
                .map(|&(_, j)| j)
                .collect()
        } else {
            Vec::new()
        };
        if ring.len() >= cfg.ring_min {
            let ring_mean = col(&ring, |r| r.mean);
            let c = (med_mean - median(&ring_mean)).abs()
                / mad_n(&ring_mean, cfg.mad_floor).max(cfg.mad_floor);
            if c > cfg.dev_limit {
                ph.cleared = true;
                ph.cluster = Some((near, c));
            }
        }
    }
    ph
}

fn finish(baseline: Baseline, d_mean: f64, d_low: f64, d_high: f64, cfg: &QcConfig) -> Phase {
    let mut own_bits = 0u8;
    if d_mean > cfg.dev_limit {
        own_bits |= SIGMA_MEAN;
    }
    if d_low > cfg.dev_limit {
        own_bits |= SIGMA_P_LOW;
    }
    if d_high > cfg.dev_limit {
        own_bits |= SIGMA_P_HIGH;
    }
    let max_dev = [d_mean, d_low, d_high]
        .into_iter()
        .filter(|d| !d.is_nan())
        .fold(f64::NAN, |a, d| if a.is_nan() || d > a { d } else { a });
    Phase {
        own_bits,
        dev_mean: d_mean,
        max_dev,
        baseline,
        cleared: false,
        cluster: None,
    }
}
