// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Closed-caliper RHOB rescue, `[qc.closed_caliper]`.
//!
//! Physics: the caliper arm is what presses the density pad against the
//! borehole wall. With the arm closed (caliper below `qc.cal_open_min_in`,
//! typically a fast pass above the zone of interest) the density tool still
//! records, but through a mud standoff: RHOB reads low and spiky and the density
//! correction is large. The bad-hole rule drops these samples; `mode = "rescue"`
//! takes some back:
//!
//! 1. Candidates: RHOB samples blanked by the bad-hole rule where the caliper
//!    reads closed (raw value from `logs_raw`).
//! 2. Despike: drop samples further than `max(mad_k · MAD·1.4826, spike_floor)`
//!    from a centered rolling median (`despike_window`; MAD over `mad_window`),
//!    with |density correction| > `drho_limit`, or RHOB < `min_rhob`.
//! 3. Envelope: the `k_neighbors` nearest wells sharing at least
//!    `neighbor_min_bins` elevation bins (`bin_ft`, elevation = KB − MD, vertical
//!    well assumed) of valid (QC-passed) RHOB with the candidates; each
//!    neighbor curve is the bin median, gaps of up to 5 bins bridged linearly,
//!    then a 5-bin rolling median. Per bin, P10 / P50 / P90 across neighbors
//!    where at least `min_neighbors` have a value.
//! 4. Bias: per well, the median of (despiked closed-caliper RHOB − envelope
//!    P50) over every despiked candidate that has an envelope (closed-vs-envelope
//!    overlap, not open-caliper overlap). Fewer than `min_overlap` such samples:
//!    no bias, every candidate stays rejected (status `too_little_overlap`).
//! 5. Keep: corrected value (raw − bias) inside [P10, P90]. Kept samples carry
//!    the corrected value in `logs_qc` and are listed with raw, bias and
//!    envelope in `rhob_closed_caliper.parquet`; candidates without an envelope
//!    (no nearby well with valid RHOB at that depth) stay rejected.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::QcClosedCalConfig;
use crate::error::SxResult;
use crate::types::GridRange;
use crate::util::stats::{median, percentiles, MAD_TO_SIGMA};

/// Bins bridged by linear interpolation in a neighbor curve.
const MAX_GAP_BINS: i64 = 5;
/// Window of the rolling median that smooths a neighbor curve (bins).
const SMOOTH_BINS: usize = 5;

/// Centered rolling median over finite values; NaN where fewer than
/// `min_count` finite values fall in the window.
#[must_use]
pub fn rolling_median(values: &[f64], window: usize, min_count: usize) -> Vec<f64> {
    let n = values.len();
    let h = window / 2;
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(h);
            let hi = (i + h + 1).min(n);
            let w: Vec<f64> = values
                .get(lo..hi)
                .unwrap_or(&[])
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            if w.len() < min_count.max(1) {
                f64::NAN
            } else {
                median(&w)
            }
        })
        .collect()
}

/// Spike mask (true = spike) of `values` (NaN = not a candidate, never a spike).
#[must_use]
pub fn despike_mask(values: &[f64], c: &QcClosedCalConfig) -> Vec<bool> {
    let min_count = (c.despike_window / 2).max(1);
    let med = rolling_median(values, c.despike_window, min_count);
    let dev: Vec<f64> = values
        .iter()
        .zip(&med)
        .map(|(v, m)| (v - m).abs())
        .collect();
    let mad = rolling_median(&dev, c.mad_window, min_count);
    values
        .iter()
        .zip(med.iter().zip(&mad))
        .map(|(&v, (&m, &d))| {
            if !(v.is_finite() && m.is_finite()) {
                return false;
            }
            let s = d * MAD_TO_SIGMA;
            let thr = if s.is_finite() {
                (c.mad_k * s).max(c.spike_floor)
            } else {
                c.spike_floor
            };
            (v - m).abs() > thr
        })
        .collect()
}

/// Elevation bin of `elev_ft`.
#[must_use]
pub fn bin_of(elev_ft: f64, bin_ft: f64) -> i64 {
    (elev_ft / bin_ft).floor() as i64
}

/// Median of the finite `vals` per elevation bin.
#[must_use]
pub fn bin_curve(elev_ft: &[f64], vals: &[f64], bin_ft: f64) -> BTreeMap<i64, f64> {
    let mut groups: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (&e, &v) in elev_ft.iter().zip(vals) {
        if e.is_finite() && v.is_finite() {
            groups.entry(bin_of(e, bin_ft)).or_default().push(v);
        }
    }
    groups.into_iter().map(|(b, v)| (b, median(&v))).collect()
}

/// Neighbor curve on bins `lo..=hi`: gaps of up to 5 bins bridged linearly,
/// then a 5-bin centered rolling median (at least 2 values).
#[must_use]
pub fn smooth_curve(raw: &BTreeMap<i64, f64>, lo: i64, hi: i64) -> BTreeMap<i64, f64> {
    let (a, b) = (lo - MAX_GAP_BINS, hi + MAX_GAP_BINS);
    if b < a {
        return BTreeMap::new();
    }
    let mut v: Vec<f64> = (a..=b)
        .map(|k| raw.get(&k).copied().unwrap_or(f64::NAN))
        .collect();
    let mut last: Option<usize> = None;
    for i in 0..v.len() {
        if !v[i].is_finite() {
            continue;
        }
        if let Some(j) = last {
            let gap = i - j - 1;
            if gap > 0 && gap as i64 <= MAX_GAP_BINS {
                let (y0, y1) = (v[j], v[i]);
                for g in 1..=gap {
                    v[j + g] = y0 + (y1 - y0) * g as f64 / (gap + 1) as f64;
                }
            }
        }
        last = Some(i);
    }
    let s = rolling_median(&v, SMOOTH_BINS, 2);
    (a..=b)
        .zip(s)
        .filter(|(k, x)| x.is_finite() && (lo..=hi).contains(k))
        .collect()
}

/// Envelope of the neighbor curves at one bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    /// Lower percentile (g/cm³).
    pub lo: f64,
    /// Median (g/cm³).
    pub mid: f64,
    /// Upper percentile (g/cm³).
    pub hi: f64,
    /// Neighbors with a value at this bin.
    pub n: usize,
}

/// Envelope at `bin`, or `None` with fewer than `min_n` neighbor values.
///
/// # Errors
/// From [`percentiles`] for a percent outside [0, 100].
pub fn envelope_at(
    curves: &[BTreeMap<i64, f64>],
    bin: i64,
    min_n: usize,
    pct: [f64; 2],
) -> SxResult<Option<Envelope>> {
    let v: Vec<f64> = curves.iter().filter_map(|c| c.get(&bin).copied()).collect();
    if v.len() < min_n.max(1) {
        return Ok(None);
    }
    let p = percentiles(&v, &[pct[0], 50.0, pct[1]])?;
    Ok(Some(Envelope {
        lo: p[0],
        mid: p[1],
        hi: p[2],
        n: v.len(),
    }))
}

/// Per-well bias: median of the finite `diffs` (closed-caliper RHOB minus
/// envelope median), or `None` with fewer than `min_overlap` of them.
#[must_use]
pub fn estimate_bias(diffs: &[f64], min_overlap: usize) -> Option<f64> {
    let d: Vec<f64> = diffs.iter().copied().filter(|v| v.is_finite()).collect();
    if d.len() < min_overlap.max(1) {
        None
    } else {
        Some(median(&d))
    }
}

/// True when `raw - bias` lies inside the envelope.
#[must_use]
pub fn inside_envelope(raw: f64, bias: f64, env: &Envelope) -> bool {
    let c = raw - bias;
    c.is_finite() && c >= env.lo && c <= env.hi
}

/// Outcome of the rescue in one well.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosedCalStatus {
    /// `mode = "reject"`: nothing rescued.
    RejectMode,
    /// No despiked candidate has an envelope (no nearby well with valid RHOB).
    NoNeighbors,
    /// Fewer than `min_overlap` despiked candidates with an envelope.
    TooLittleOverlap,
    /// Bias estimated; samples inside the envelope kept.
    Rescued,
}

impl ClosedCalStatus {
    /// Text used in the reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RejectMode => "reject_mode",
            Self::NoNeighbors => "no_neighbors",
            Self::TooLittleOverlap => "too_little_overlap",
            Self::Rescued => "rescued",
        }
    }
}

/// One closed-caliper well for [`rescue_well`].
#[derive(Debug, Clone, Copy)]
pub struct ClosedCalTarget<'a> {
    /// KB elevation (ft).
    pub kb_ft: f64,
    /// Grid of the well.
    pub grid: GridRange,
    /// Raw RHOB (g/cm³).
    pub raw: &'a [f32],
    /// Candidates: closed caliper and dropped by the bad-hole rule.
    pub candidate: &'a [bool],
    /// Density correction (g/cm³; empty or NaN = none).
    pub drho: &'a [f32],
}

/// Result of [`rescue_well`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedCalResult {
    /// Outcome.
    pub status: ClosedCalStatus,
    /// Candidates left after the despike.
    pub n_despiked: usize,
    /// Despiked candidates with an envelope.
    pub n_with_envelope: usize,
    /// Neighbor wells used.
    pub n_neighbors: usize,
    /// Bias (g/cm³; NaN when not estimated).
    pub bias: f64,
    /// Kept samples: (index, corrected value, envelope).
    pub kept: Vec<(usize, f64, Envelope)>,
}

/// Rescue one well against `neighbors` (binned valid RHOB of the other wells,
/// nearest first; see [`bin_curve`]).
///
/// # Errors
/// From [`envelope_at`].
pub fn rescue_well(
    t: &ClosedCalTarget<'_>,
    neighbors: &[&BTreeMap<i64, f64>],
    c: &QcClosedCalConfig,
) -> SxResult<ClosedCalResult> {
    let n = t.raw.len();
    let cand: Vec<f64> = (0..n)
        .map(|i| {
            if t.candidate.get(i).copied().unwrap_or(false) {
                f64::from(t.raw[i])
            } else {
                f64::NAN
            }
        })
        .collect();
    let spike = despike_mask(&cand, c);
    let ok: Vec<usize> = (0..n)
        .filter(|&i| {
            let v = cand[i];
            let d = t.drho.get(i).map_or(f64::NAN, |&d| f64::from(d));
            v.is_finite() && !spike[i] && v >= c.min_rhob && (d.is_nan() || d.abs() <= c.drho_limit)
        })
        .collect();
    let mut res = ClosedCalResult {
        status: ClosedCalStatus::NoNeighbors,
        n_despiked: ok.len(),
        n_with_envelope: 0,
        n_neighbors: 0,
        bias: f64::NAN,
        kept: Vec::new(),
    };
    if c.mode == "reject" {
        res.status = ClosedCalStatus::RejectMode;
        return Ok(res);
    }
    if ok.is_empty() {
        return Ok(res);
    }
    let bin = |i: usize| bin_of(t.kb_ft - t.grid.md_ft(i), c.bin_ft);
    let bins: BTreeSet<i64> = ok.iter().map(|&i| bin(i)).collect();
    let (Some(&lo), Some(&hi)) = (bins.first(), bins.last()) else {
        return Ok(res);
    };
    let curves: Vec<BTreeMap<i64, f64>> = neighbors
        .iter()
        .filter(|nb| bins.iter().filter(|b| nb.contains_key(b)).count() >= c.neighbor_min_bins)
        .take(c.k_neighbors)
        .map(|nb| smooth_curve(nb, lo, hi))
        .collect();
    res.n_neighbors = curves.len();
    let mut env: BTreeMap<i64, Option<Envelope>> = BTreeMap::new();
    for &b in &bins {
        env.insert(b, envelope_at(&curves, b, c.min_neighbors, c.envelope_pct)?);
    }
    let with_env: Vec<(usize, Envelope)> = ok
        .iter()
        .filter_map(|&i| env.get(&bin(i)).copied().flatten().map(|e| (i, e)))
        .collect();
    res.n_with_envelope = with_env.len();
    if with_env.is_empty() {
        return Ok(res);
    }
    let diffs: Vec<f64> = with_env.iter().map(|&(i, e)| cand[i] - e.mid).collect();
    let Some(bias) = estimate_bias(&diffs, c.min_overlap) else {
        res.status = ClosedCalStatus::TooLittleOverlap;
        return Ok(res);
    };
    res.status = ClosedCalStatus::Rescued;
    res.bias = bias;
    res.kept = with_env
        .into_iter()
        .filter(|(i, e)| inside_envelope(cand[*i], bias, e))
        .map(|(i, e)| (i, cand[i] - bias, e))
        .collect();
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QcClosedCalConfig {
        QcClosedCalConfig {
            min_overlap: 20,
            neighbor_min_bins: 10,
            ..QcClosedCalConfig::default()
        }
    }

    #[test]
    fn despike_flags_isolated_spikes_only() {
        let mut v: Vec<f64> = (0..60).map(|i| 2.4 + 0.01 * ((i * 7) % 5) as f64).collect();
        v[30] = 1.3;
        v[10] = f64::NAN;
        let m = despike_mask(&v, &cfg());
        assert!(m[30]);
        assert_eq!(m.iter().filter(|&&b| b).count(), 1);
        assert!(!m[10]);
    }

    #[test]
    fn smooth_bridges_short_gaps_only() {
        let mut raw = BTreeMap::new();
        for k in 0..10 {
            raw.insert(k, 2.0);
        }
        for k in 13..20 {
            raw.insert(k, 2.0);
        }
        for k in 40..50 {
            raw.insert(k, 2.5);
        }
        let s = smooth_curve(&raw, 0, 49);
        assert!(s.contains_key(&11) && s.contains_key(&12));
        assert!(!s.contains_key(&30));
        assert!((s[&45] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn bias_estimate_and_minimum_overlap() {
        let d = vec![-0.2, -0.1, -0.3, f64::NAN];
        assert_eq!(estimate_bias(&d, 3), Some(-0.2));
        assert_eq!(estimate_bias(&d, 4), None);
    }

    #[test]
    fn envelope_needs_min_neighbors_and_keep_uses_bias() {
        let curves: Vec<BTreeMap<i64, f64>> = [2.3, 2.4, 2.5]
            .iter()
            .map(|&v| [(7i64, v)].into_iter().collect())
            .collect();
        assert!(envelope_at(&curves, 7, 4, [10.0, 90.0]).unwrap().is_none());
        assert!(envelope_at(&curves, 8, 1, [10.0, 90.0]).unwrap().is_none());
        let e = envelope_at(&curves, 7, 3, [10.0, 90.0]).unwrap().unwrap();
        assert!(
            (e.lo - 2.32).abs() < 1e-9 && (e.mid - 2.4).abs() < 1e-9 && (e.hi - 2.48).abs() < 1e-9
        );
        assert!(!inside_envelope(2.2, 0.0, &e));
        assert!(inside_envelope(2.2, -0.2, &e));
    }

    fn target_well(n: usize, offset: f32) -> (Vec<f32>, Vec<bool>, Vec<f32>) {
        let raw: Vec<f32> = (0..n)
            .map(|i| 2.45 + offset + 0.02 * ((i % 7) as f32 - 3.0) / 3.0)
            .collect();
        (raw, vec![true; n], vec![0.05; n])
    }

    fn nbr(n: usize, kb: f64, v: f64, c: &QcClosedCalConfig) -> BTreeMap<i64, f64> {
        let g = GridRange::new(0, n).unwrap();
        let e: Vec<f64> = (0..n).map(|i| kb - g.md_ft(i)).collect();
        bin_curve(&e, &vec![v; n], c.bin_ft)
    }

    #[test]
    fn rescue_shifts_by_bias_and_keeps_inside() {
        let c = cfg();
        let n = 400;
        let (raw, cand, drho) = target_well(n, -0.2);
        let t = ClosedCalTarget {
            kb_ft: 1000.0,
            grid: GridRange::new(0, n).unwrap(),
            raw: &raw,
            candidate: &cand,
            drho: &drho,
        };
        let nb: Vec<BTreeMap<i64, f64>> = [2.40, 2.45, 2.50, 2.44]
            .iter()
            .map(|&v| nbr(n, 1000.0, v, &c))
            .collect();
        let refs: Vec<&BTreeMap<i64, f64>> = nb.iter().collect();
        let r = rescue_well(&t, &refs, &c).unwrap();
        assert_eq!(r.status, ClosedCalStatus::Rescued);
        assert_eq!(r.n_neighbors, 4);
        assert!((r.bias + 0.2).abs() < 0.03, "{}", r.bias);
        assert!(!r.kept.is_empty());
        for (i, v, e) in &r.kept {
            assert!((*v - (f64::from(raw[*i]) - r.bias)).abs() < 1e-9);
            assert!(*v >= e.lo && *v <= e.hi);
        }
    }

    #[test]
    fn rescue_falls_back_without_neighbors_or_overlap() {
        let c = cfg();
        let n = 400;
        let (raw, cand, drho) = target_well(n, 0.0);
        let t = ClosedCalTarget {
            kb_ft: 1000.0,
            grid: GridRange::new(0, n).unwrap(),
            raw: &raw,
            candidate: &cand,
            drho: &drho,
        };
        // No neighbors.
        let r = rescue_well(&t, &[], &c).unwrap();
        assert_eq!(r.status, ClosedCalStatus::NoNeighbors);
        assert!(r.kept.is_empty());
        // Neighbors only far below the closed interval: no shared bins.
        let far: Vec<BTreeMap<i64, f64>> = (0..3).map(|_| nbr(n, -5000.0, 2.4, &c)).collect();
        let refs: Vec<&BTreeMap<i64, f64>> = far.iter().collect();
        assert_eq!(
            rescue_well(&t, &refs, &c).unwrap().status,
            ClosedCalStatus::NoNeighbors
        );
        // Three neighbors, but the overlap is too short for the bias.
        let short: Vec<BTreeMap<i64, f64>> = (0..3).map(|_| nbr(30, 1000.0, 2.45, &c)).collect();
        let refs: Vec<&BTreeMap<i64, f64>> = short.iter().collect();
        let c2 = QcClosedCalConfig {
            neighbor_min_bins: 5,
            min_overlap: 100,
            ..cfg()
        };
        let r = rescue_well(&t, &refs, &c2).unwrap();
        assert_eq!(r.status, ClosedCalStatus::TooLittleOverlap);
        assert!(r.kept.is_empty() && r.bias.is_nan());
        // Reject mode never keeps anything.
        let c3 = QcClosedCalConfig {
            mode: "reject".into(),
            ..cfg()
        };
        let nb: Vec<BTreeMap<i64, f64>> = (0..4).map(|_| nbr(n, 1000.0, 2.45, &c)).collect();
        let refs: Vec<&BTreeMap<i64, f64>> = nb.iter().collect();
        let r = rescue_well(&t, &refs, &c3).unwrap();
        assert_eq!(r.status, ClosedCalStatus::RejectMode);
        assert!(r.kept.is_empty());
    }

    #[test]
    fn low_and_bad_correction_samples_are_never_kept() {
        let c = cfg();
        let n = 400;
        let (mut raw, cand, mut drho) = target_well(n, 0.0);
        raw[100] = 1.5;
        drho[200] = 0.5;
        let t = ClosedCalTarget {
            kb_ft: 1000.0,
            grid: GridRange::new(0, n).unwrap(),
            raw: &raw,
            candidate: &cand,
            drho: &drho,
        };
        let nb: Vec<BTreeMap<i64, f64>> = [2.40, 2.45, 2.50]
            .iter()
            .map(|&v| nbr(n, 1000.0, v, &c))
            .collect();
        let refs: Vec<&BTreeMap<i64, f64>> = nb.iter().collect();
        let r = rescue_well(&t, &refs, &c).unwrap();
        assert!(r.kept.iter().all(|(i, _, _)| *i != 100 && *i != 200));
    }
}
