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

#[cfg(test)]
mod tests {
    use super::*;

    fn st(id: i32, mean: f64) -> ChannelStats {
        ChannelStats {
            well: WellId(id),
            log: LogKind::Gr,
            n: 1000,
            mean,
            median: mean,
            std: 5.0,
            p_low: mean - 8.0,
            p_high: mean + 8.0,
        }
    }

    /// Deterministic spread around `centre`, MAD-friendly (no exact ties).
    fn spread(i: usize) -> f64 {
        let k = (i as u64 * 7919 + 13) % 101;
        (k as f64 - 50.0) / 50.0
    }

    fn line_xy(n: usize) -> BTreeMap<WellId, (f64, f64)> {
        (0..n)
            .map(|i| (WellId(i as i32), (i as f64 * 100.0, 0.0)))
            .collect()
    }

    /// Example 4 in the ring-test: a well with mean 56 among near means
    /// [50, 52, 48, 51, 49] has dev_mean 4.0469.
    #[test]
    fn example_4() {
        let means = [50.0, 52.0, 48.0, 51.0, 49.0];
        let mut rows: Vec<ChannelStats> = means
            .iter()
            .enumerate()
            .map(|(i, m)| st(i as i32 + 1, *m))
            .collect();
        rows.push(st(0, 56.0));
        // well 0 sits at the origin; the five near wells are its nearest.
        let mut xy = BTreeMap::new();
        xy.insert(WellId(0), (0.0, 0.0));
        for i in 1..=5 {
            xy.insert(WellId(i), (f64::from(i) * 10.0, 0.0));
        }
        let out = ring_test(LogKind::Gr, &rows, &xy, &QcConfig::default());
        let w0 = out.iter().find(|r| r.well == WellId(0)).unwrap();
        assert_eq!(w0.baseline, Baseline::Local);
        // mean dev = 6 / 1.482602218505602, p_low/p_high deviate equally
        assert!(
            (w0.max_dev - 4.04693850117649).abs() < 1e-9,
            "{}",
            w0.max_dev
        );
        assert_eq!(w0.bits & SIGMA_MEAN, SIGMA_MEAN);
        // the other five see 6 near wells (incl. well 0): no flags
        for r in out.iter().filter(|r| r.well != WellId(0)) {
            assert_eq!(r.bits, 0, "{:?}", r);
        }
    }

    #[test]
    fn no_fleet_gives_none() {
        let rows = vec![st(1, 50.0), st(2, 51.0)];
        let out = ring_test(LogKind::Gr, &rows, &BTreeMap::new(), &QcConfig::default());
        assert_eq!(out.len(), 2);
        for r in out {
            assert_eq!(r.baseline, Baseline::None);
            assert_eq!(r.bits, 0);
            assert!(r.max_dev.is_nan() && r.unit_ratio.is_nan() && r.cluster_dev.is_nan());
        }
    }

    #[test]
    fn planted_level_outlier() {
        let n = 30;
        let mut rows: Vec<ChannelStats> = (0..n)
            .map(|i| st(i as i32, 60.0 + 3.0 * spread(i)))
            .collect();
        rows[11] = st(11, 140.0);
        let out = ring_test(LogKind::Gr, &rows, &line_xy(n), &QcConfig::default());
        for r in &out {
            if r.well == WellId(11) {
                assert_eq!(r.bits & SIGMA_MEAN, SIGMA_MEAN);
                assert_eq!(r.baseline, Baseline::Local);
            } else {
                assert_eq!(r.bits & SIGMA_MEAN, 0, "{:?}", r);
            }
        }
        // fleet baseline when wells have no XY
        let out2 = ring_test(LogKind::Gr, &rows, &BTreeMap::new(), &QcConfig::default());
        assert!(out2.iter().all(|r| r.baseline == Baseline::Fleet));
        assert_eq!(out2[11].bits & SIGMA_MEAN, SIGMA_MEAN);
    }

    /// A seed well at the regional level, its 16 nearest wells shifted, the
    /// next 48 unshifted: the seed is cleared, its near wells get the cluster bit.
    #[test]
    fn two_ring_consensus_and_order_independence() {
        // Layout on a line: seed at x=0; 16 shifted wells at x=10..160; 48
        // regional wells at x=1000..; positions use distinct distances.
        let mut rows = Vec::new();
        let mut xy = BTreeMap::new();
        rows.push(st(0, 60.0));
        xy.insert(WellId(0), (0.0, 0.0));
        for i in 1..=16 {
            rows.push(st(i, 90.0 + 2.0 * spread(i as usize)));
            xy.insert(WellId(i), (f64::from(i) * 10.0, 0.0));
        }
        for i in 17..=64 {
            rows.push(st(i, 60.0 + 2.0 * spread(i as usize)));
            xy.insert(WellId(i), (1000.0 + f64::from(i) * 10.0, 0.0));
        }
        let cfg = QcConfig::default();
        let out = ring_test(LogKind::Gr, &rows, &xy, &cfg);
        let seed = out.iter().find(|r| r.well == WellId(0)).unwrap();
        assert_eq!(
            seed.bits & (SIGMA_MEAN | SIGMA_P_LOW | SIGMA_P_HIGH),
            0,
            "{seed:?}"
        );
        assert!(seed.max_dev > 3.0);
        assert_eq!(seed.bits & SIGMA_CLUSTER, 0);
        for i in 1..=16 {
            let r = out.iter().find(|r| r.well == WellId(i)).unwrap();
            assert_eq!(r.bits & SIGMA_CLUSTER, SIGMA_CLUSTER, "well {i}: {r:?}");
            assert!(r.cluster_dev > 3.0);
        }
        // reverse order: identical result
        let mut rev = rows.clone();
        rev.reverse();
        let out_rev = ring_test(LogKind::Gr, &rev, &xy, &cfg);
        // NaN-aware comparison (cluster_dev is NaN where unset)
        assert_eq!(format!("{out:?}"), format!("{out_rev:?}"));
    }

    /// An NPHI well in percent is flagged; a local cluster of low-porosity
    /// wells (its members' nearest wells are all in the cluster) is not.
    #[test]
    fn unit_flag() {
        let n = 100usize;
        let mut xy = BTreeMap::new();
        let mut rows: Vec<ChannelStats> = Vec::new();
        for i in 0..n {
            let (m, x) = if i < 40 {
                (0.02 + 0.002 * spread(i), i as f64)
            } else {
                (0.15 + 0.01 * spread(i), 10000.0 + 100.0 * i as f64)
            };
            let w = if i < 40 { 0.01 } else { 0.06 };
            rows.push(ChannelStats {
                well: WellId(i as i32),
                log: LogKind::Nphi,
                n: 1000,
                mean: m,
                median: m,
                std: 0.05,
                p_low: m - w,
                p_high: m + w,
            });
            xy.insert(WellId(i as i32), (x, 0.0));
        }
        // well 55 in percent
        rows[55] = ChannelStats {
            mean: 15.0,
            median: 15.0,
            p_low: 9.0,
            p_high: 21.0,
            ..rows[55]
        };
        let out = ring_test(LogKind::Nphi, &rows, &xy, &QcConfig::default());
        let pct = out.iter().find(|r| r.well == WellId(55)).unwrap();
        assert_eq!(pct.bits & SIGMA_UNIT, SIGMA_UNIT, "{pct:?}");
        assert!(pct.unit_ratio > 5.0);
        for r in out.iter().filter(|r| r.well.0 < 40) {
            assert_eq!(r.bits & SIGMA_UNIT, 0, "{r:?}");
            assert!(r.unit_ratio < 0.2, "ratio {}", r.unit_ratio);
        }
    }

    #[test]
    fn resd_unit_ratio() {
        assert!((unit_ratio(LogKind::Resd, 2.0, 1.0) - 10.0).abs() < 1e-12);
        assert!(unit_ratio(LogKind::Gr, -1.0, 5.0).is_nan());
        assert!(unit_ratio(LogKind::Gr, 5.0, 0.0).is_nan());
        assert_eq!(unit_ratio(LogKind::Gr, 10.0, 5.0), 2.0);
    }
}
