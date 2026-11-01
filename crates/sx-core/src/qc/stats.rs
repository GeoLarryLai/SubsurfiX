// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Per-well channel statistics, fleet summaries, and the low-variance and
//! post-normalization flat checks.
//!
//! All statistics are in native space (log10 ohm·m for RESD). NaN rule for
//! every function here: skip (NaN samples are ignored).

use std::collections::BTreeSet;

use crate::types::{LogKind, WellId};
use crate::util::{median, percentiles, MAD_TO_SIGMA};

/// `(n, mean, std)` of the finite values, std with ddof 1 (NaN if `n < 2`),
/// sums in f64 (two-pass). `n = 0` gives `(0, NaN, NaN)`.
pub(crate) fn mean_std(values: &[f64]) -> (usize, f64, f64) {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return (0, f64::NAN, f64::NAN);
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (n, mean, f64::NAN);
    }
    let ss: f64 = finite.iter().map(|v| (v - mean) * (v - mean)).sum();
    (n, mean, (ss / (n as f64 - 1.0)).sqrt())
}

/// `MAD_n(x) = 1.482602218505602 * median(|x - median(x)|)`, floored at
/// `floor`. NaN inputs are skipped; no finite input gives NaN.
pub(crate) fn mad_n(values: &[f64], floor: f64) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    let m = median(&finite);
    let dev: Vec<f64> = finite.iter().map(|v| (v - m).abs()).collect();
    (MAD_TO_SIGMA * median(&dev)).max(floor)
}

/// Statistics of one channel of one well, native space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelStats {
    /// Well id.
    pub well: WellId,
    /// Log.
    pub log: LogKind,
    /// Finite samples.
    pub n: usize,
    /// Mean (native).
    pub mean: f64,
    /// Median (native).
    pub median: f64,
    /// Standard deviation, ddof 1 (native).
    pub std: f64,
    /// Lower percentile (`stats_pct[0]`, native).
    pub p_low: f64,
    /// Upper percentile (`stats_pct[1]`, native).
    pub p_high: f64,
}

/// Stats of the finite native values, or `None` if fewer than `min_count`
/// (or none) are finite or `pct` is outside `[0, 100]`.
///
/// NaN rule: skip. O(n log n).
pub fn channel_stats(
    well: WellId,
    log: LogKind,
    native: &[f64],
    min_count: usize,
    pct: [f64; 2],
) -> Option<ChannelStats> {
    let finite: Vec<f64> = native.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || finite.len() < min_count {
        return None;
    }
    let (n, mean, std) = mean_std(&finite);
    let ps = percentiles(&finite, &pct).ok()?;
    let (p_low, p_high) = match ps.as_slice() {
        [a, b] => (*a, *b),
        _ => return None,
    };
    Some(ChannelStats {
        well,
        log,
        n,
        mean,
        median: median(&finite),
        std,
        p_low,
        p_high,
    })
}

/// Fleet-wide robust summary of the channel stats of one log (native space).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FleetSummary {
    /// Log.
    pub log: LogKind,
    /// Number of wells (rows).
    pub n_wells: usize,
    /// Median over wells of the mean.
    pub mean_med: f64,
    /// Median over wells of `p_low`.
    pub low_med: f64,
    /// Median over wells of `p_high`.
    pub high_med: f64,
    /// Median over wells of the std.
    pub std_med: f64,
    /// `MAD_n` over wells of the mean, floored at `mad_floor`.
    pub mean_mad: f64,
    /// `MAD_n` over wells of `p_low`, floored.
    pub low_mad: f64,
    /// `MAD_n` over wells of `p_high`, floored.
    pub high_mad: f64,
}

/// Fleet summary, or `None` if there are fewer than `min_wells` rows (or none).
///
/// NaN rule: skip (NaN entries are ignored inside each median / MAD).
pub fn fleet_summary(
    log: LogKind,
    rows: &[ChannelStats],
    min_wells: usize,
    mad_floor: f64,
) -> Option<FleetSummary> {
    if rows.is_empty() || rows.len() < min_wells {
        return None;
    }
    let col = |f: fn(&ChannelStats) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
    let mean = col(|r| r.mean);
    let low = col(|r| r.p_low);
    let high = col(|r| r.p_high);
    let std = col(|r| r.std);
    Some(FleetSummary {
        log,
        n_wells: rows.len(),
        mean_med: median(&mean),
        low_med: median(&low),
        high_med: median(&high),
        std_med: median(&std),
        mean_mad: mad_n(&mean, mad_floor),
        low_mad: mad_n(&low, mad_floor),
        high_mad: mad_n(&high, mad_floor),
    })
}

/// Wells whose std is below `ratio` times the median std of the rows.
///
/// Needs at least `min_wells` rows (and at least one), else empty. NaN rule:
/// a NaN std is never flagged.
pub fn low_variance_wells(rows: &[ChannelStats], ratio: f64, min_wells: usize) -> BTreeSet<WellId> {
    if rows.is_empty() || rows.len() < min_wells {
        return BTreeSet::new();
    }
    let stds: Vec<f64> = rows.iter().map(|r| r.std).collect();
    let limit = ratio * median(&stds);
    rows.iter()
        .filter(|r| r.std < limit)
        .map(|r| r.well)
        .collect()
}

/// Post-normalization flat check: wells whose std is below `ratio` times the
/// median std over wells.
///
/// Only wells with at least `min_count` finite values take part (std with
/// ddof 1); needs at least `min_wells` such wells, else empty. NaN rule: skip.
pub fn post_flat_wells(
    channels: &[(WellId, &[f64])],
    ratio: f64,
    min_count: usize,
    min_wells: usize,
) -> BTreeSet<WellId> {
    let mut stds: Vec<(WellId, f64)> = Vec::new();
    for (id, values) in channels {
        let (n, _, std) = mean_std(values);
        if n >= min_count.max(1) && std.is_finite() {
            stds.push((*id, std));
        }
    }
    if stds.is_empty() || stds.len() < min_wells {
        return BTreeSet::new();
    }
    let all: Vec<f64> = stds.iter().map(|(_, s)| *s).collect();
    let limit = ratio * median(&all);
    stds.iter()
        .filter(|(_, s)| *s < limit)
        .map(|(id, _)| *id)
        .collect()
}
