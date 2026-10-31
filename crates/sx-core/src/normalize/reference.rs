// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Regional references of stage 4: the fleet pair, the
//! inverse-distance pair, the second pick of voters, the depth-aware pair by
//! layer coordinate, and the inverse-distance mean.
//!
//! Distances are Euclidean on the header `x_m`, `y_m` (m); every ranking is a
//! brute-force sort by `(distance, WellId)`. Values are native: the log's own
//! unit, log10(ohm·m) for RESD.

use std::collections::BTreeSet;

use super::transform::{gain_offset, GainOffset};
use crate::config::{NormalizeConfig, QcConfig};
use crate::types::{LogKind, WellId};
use crate::util::{median, percentiles};

/// Floor of the comparison span in the sanity ratio (native units).
const SPAN_FLOOR: f64 = 1e-10;
/// Epsilon of [`layer_bin`].
const BIN_EPS: f64 = 1e-9;
/// Widest target bin range looked up through a dense table (bins); wider
/// ranges use the bin set.
const MAX_DENSE_BINS: usize = 1 << 24;

/// Which samples a channel's statistics use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Finite samples with a finite layer coordinate.
    Layer,
    /// All finite samples.
    All,
}

impl Scope {
    /// Text used in the CSV reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Layer => "layer",
            Scope::All => "all",
        }
    }
}

/// Anchor of one well and normalized log: percentiles at `qc.stats_pct` of
/// its scope samples (native units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    /// Well id.
    pub well: WellId,
    /// Log.
    pub log: LogKind,
    /// Scope samples.
    pub n: usize,
    /// Lower anchor (native).
    pub p_low: f64,
    /// Upper anchor (native).
    pub p_high: f64,
    /// Scope used.
    pub scope: Scope,
}

/// Where a fleet pair came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FleetSource {
    /// Medians over the eligible voters.
    Voters,
    /// Medians over all anchored wells.
    All,
    /// Too few wells: `(NaN, NaN)`.
    None,
}

impl FleetSource {
    /// Text used in `norm_anchors.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            FleetSource::Voters => "voters",
            FleetSource::All => "all",
            FleetSource::None => "none",
        }
    }
}

/// Where a reference pair came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefSource {
    /// Depth-aware pair by layer coordinate.
    Depth,
    /// Whole-interval inverse-distance pair.
    Whole,
    /// Fleet pair.
    Fleet,
    /// No reference (identity transform).
    None,
}

impl RefSource {
    /// Text used in `norm_anchors.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            RefSource::Depth => "depth",
            RefSource::Whole => "whole",
            RefSource::Fleet => "fleet",
            RefSource::None => "none",
        }
    }
}

/// Reference pair of one well and log (native units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefPair {
    /// Lower reference (NaN when `source` is `None`).
    pub low: f64,
    /// Upper reference (NaN when `source` is `None`).
    pub high: f64,
    /// Source.
    pub source: RefSource,
    /// Voters in the inverse-distance sum; 0 when no sum ran.
    pub n_voters: usize,
    /// The sum ran but failed the span sanity check, so the fleet pair is used.
    pub reverted: bool,
}

impl RefPair {
    /// The fleet pair as a reference: source `Fleet` when both ends are
    /// finite, else `None` with NaN ends.
    fn from_fleet(fleet: (f64, f64)) -> Self {
        let finite = fleet.0.is_finite() && fleet.1.is_finite();
        Self {
            low: if finite { fleet.0 } else { f64::NAN },
            high: if finite { fleet.1 } else { f64::NAN },
            source: if finite {
                RefSource::Fleet
            } else {
                RefSource::None
            },
            n_voters: 0,
            reverted: false,
        }
    }
}

/// An eligible voter: surface XY (m) and its pair (native units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voter {
    /// Well id.
    pub well: WellId,
    /// Surface x, y (m).
    pub xy: (f64, f64),
    /// Lower value (native).
    pub low: f64,
    /// Upper value (native).
    pub high: f64,
}

/// Why an anchored well is not a voter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoterExclusion {
    /// Eligible voter.
    None,
    /// No anchor.
    NoAnchor,
    /// Official ring-test bits are not 0.
    Sigma,
    /// Divergence outlier.
    Jsd,
    /// Unit rescue in stage 2.
    Rescued,
    /// An anchor end is not finite or outside the native bounds.
    Implausible,
    /// No surface XY.
    NoXy,
    /// Second pick: fewer than `pick_min_count` anchor samples.
    TooShort,
    /// Second pick: stretch to the build-1 reference outside the gain cap.
    StretchOut,
}

impl VoterExclusion {
    /// Text used in `norm_anchors.csv` (empty for an eligible voter).
    pub fn as_str(self) -> &'static str {
        match self {
            VoterExclusion::None => "",
            VoterExclusion::NoAnchor => "no_anchor",
            VoterExclusion::Sigma => "sigma",
            VoterExclusion::Jsd => "jsd",
            VoterExclusion::Rescued => "rescued",
            VoterExclusion::Implausible => "implausible",
            VoterExclusion::NoXy => "no_xy",
            VoterExclusion::TooShort => "too_short",
            VoterExclusion::StretchOut => "stretch_out",
        }
    }
}

/// Outcome of the depth-aware pass (4.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthReason {
    /// The depth-aware pair is the final reference.
    LayerWindow,
    /// Cover at or above `depth_max_cover`: whole-interval reference.
    WideCover,
    /// Scope `all` or no layer bin: whole-interval reference.
    NoLayer,
    /// No XY, too few qualifying voters, or failed sanity: whole-interval reference.
    WholeFallback,
    /// Not run (channel blanked before the pass).
    NotRun,
}

impl DepthReason {
    /// Text used in `norm_anchors.csv`.
    pub fn as_str(self) -> &'static str {
        match self {
            DepthReason::LayerWindow => "layer_window",
            DepthReason::WideCover => "wide_cover",
            DepthReason::NoLayer => "no_layer",
            DepthReason::WholeFallback => "whole_fallback",
            DepthReason::NotRun => "not_run",
        }
    }
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    (dx * dx + dy * dy).sqrt()
}

fn finite_xy(p: Option<(f64, f64)>) -> Option<(f64, f64)> {
    p.filter(|(x, y)| x.is_finite() && y.is_finite())
}

/// Candidates other than `target` with finite XY, sorted by `(distance, WellId)`.
fn rank<T>(
    target: WellId,
    at: (f64, f64),
    items: impl Iterator<Item = (WellId, (f64, f64), T)>,
) -> Vec<(f64, WellId, T)> {
    let mut out: Vec<(f64, WellId, T)> = items
        .filter(|(w, p, _)| *w != target && p.0.is_finite() && p.1.is_finite())
        .map(|(w, p, t)| (distance(at, p), w, t))
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

/// Weights `1 / max(d, ref_dist_floor_m)^ref_power` of the first
/// `min(ref_k, n)` distances, normalized to sum 1.
fn weights(dists: impl Iterator<Item = f64>, cfg: &NormalizeConfig) -> Vec<f64> {
    let raw: Vec<f64> = dists
        .take(cfg.ref_k)
        .map(|d| 1.0 / d.max(cfg.ref_dist_floor_m).powf(cfg.ref_power))
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.into_iter().map(|w| w / sum).collect()
}

/// Inverse-distance pair over the first `min(ref_k, n)` ranked voters:
/// `(low, high, voters used)`, same weights for both ends.
fn idw_sum(ranked: &[(f64, WellId, (f64, f64))], cfg: &NormalizeConfig) -> (f64, f64, usize) {
    let w = weights(ranked.iter().map(|r| r.0), cfg);
    let (mut low, mut high) = (0.0f64, 0.0f64);
    for (wi, (_, _, (l, h))) in w.iter().zip(ranked) {
        low += wi * l;
        high += wi * h;
    }
    (low, high, w.len())
}

/// Span sanity: false when `against` is finite and `|high − low| /
/// max(|against span|, 1e-10) > ratio_max`.
fn sane(low: f64, high: f64, against: (f64, f64), ratio_max: f64) -> bool {
    if !(against.0.is_finite() && against.1.is_finite()) {
        return true;
    }
    let ratio = (high - low).abs() / (against.1 - against.0).abs().max(SPAN_FLOOR);
    ratio.is_nan() || ratio <= ratio_max
}

/// Fleet pair of one log (4.3): `(median p_low, median p_high)` over `voters`
/// when there are at least `min_wells` of them (`Voters`), else over
/// `anchored` when at least `min_wells` (`All`), else `(NaN, NaN, None)`.
///
/// Units: native. NaN rule: skip (inside each median). Errors: none.
/// O(n log n).
pub fn fleet_pair(
    voters: &[Anchor],
    anchored: &[Anchor],
    min_wells: usize,
) -> (f64, f64, FleetSource) {
    let need = min_wells.max(1);
    let medians = |rows: &[Anchor]| {
        let lo: Vec<f64> = rows.iter().map(|a| a.p_low).collect();
        let hi: Vec<f64> = rows.iter().map(|a| a.p_high).collect();
        (median(&lo), median(&hi))
    };
    if voters.len() >= need {
        let (lo, hi) = medians(voters);
        (lo, hi, FleetSource::Voters)
    } else if anchored.len() >= need {
        let (lo, hi) = medians(anchored);
        (lo, hi, FleetSource::All)
    } else {
        (f64::NAN, f64::NAN, FleetSource::None)
    }
}

/// Whole-interval reference of `target` (4.4).
///
/// 1. No (finite) `target_xy` → the fleet pair (`Fleet`, or `None` when the
///    fleet pair is not finite).
/// 2. Voters other than `target` (finite XY and pair); fewer than
///    `ref_min_voters` → the fleet pair.
/// 3. The `min(ref_k, count)` nearest by `(distance, WellId)`, weights
///    `1 / max(d, ref_dist_floor_m)^ref_power` normalized to sum 1; `low =
///    Σ w·low_i`, `high = Σ w·high_i`. Source `Whole`.
/// 4. If `fleet` is finite and `|high − low| / max(|fleet span|, 1e-10) >
///    ref_span_ratio_max`: the fleet pair with `reverted = true`.
///
/// `n_voters` counts the voters of the sum (0 when it did not run). Units:
/// native, distances m. NaN rule: skip (voters with a NaN end are ignored).
/// Errors: none. O(v log v) for `v` voters.
pub fn idw_pair(
    target: WellId,
    target_xy: Option<(f64, f64)>,
    voters: &[Voter],
    fleet: (f64, f64),
    cfg: &NormalizeConfig,
) -> RefPair {
    let fallback = RefPair::from_fleet(fleet);
    let Some(at) = finite_xy(target_xy) else {
        return fallback;
    };
    let ranked = rank(
        target,
        at,
        voters
            .iter()
            .filter(|v| v.low.is_finite() && v.high.is_finite())
            .map(|v| (v.well, v.xy, (v.low, v.high))),
    );
    if ranked.is_empty() || ranked.len() < cfg.ref_min_voters {
        return fallback;
    }
    let (low, high, n) = idw_sum(&ranked, cfg);
    if !sane(low, high, fleet, cfg.ref_span_ratio_max) {
        return RefPair {
            n_voters: n,
            reverted: true,
            ..fallback
        };
    }
    RefPair {
        low,
        high,
        source: RefSource::Whole,
        n_voters: n,
        reverted: false,
    }
}

/// Inverse-distance mean of one value per voter (4.8): the weights of
/// [`idw_pair`] over the `min(ref_k, count)` nearest voters other than
/// `target`. `target` without (finite) XY or fewer than `ref_min_voters`
/// voters → `fallback` (the caller passes the median of the other voters'
/// values when there are at least `qc.fleet_min_wells` of them, else NaN).
///
/// Units: native (log10 ohm·m for the decade check), distances m. NaN rule:
/// skip (voters with a NaN value are ignored). Errors: none. O(v log v).
pub fn idw_mean(
    target: WellId,
    target_xy: Option<(f64, f64)>,
    voters: &[(WellId, (f64, f64), f64)],
    fallback: f64,
    cfg: &NormalizeConfig,
) -> f64 {
    let Some(at) = finite_xy(target_xy) else {
        return fallback;
    };
    let ranked = rank(
        target,
        at,
        voters.iter().copied().filter(|v| v.2.is_finite()),
    );
    if ranked.is_empty() || ranked.len() < cfg.ref_min_voters {
        return fallback;
    }
    let w = weights(ranked.iter().map(|r| r.0), cfg);
    w.iter().zip(&ranked).map(|(wi, r)| wi * r.2).sum()
}

/// Layer bin of a layer coordinate: `floor(l · bins_per_interval + 1e-9)`;
/// `None` when `l` is not finite or the bin does not fit in `i64`.
///
/// Units: `l` in layer-coordinate units (interval + fraction). NaN rule:
/// `None`. Errors: none.
pub fn layer_bin(l: f64, bins_per_interval: usize) -> Option<i64> {
    if !l.is_finite() {
        return None;
    }
    let b = (l * bins_per_interval as f64 + BIN_EPS).floor();
    (b.is_finite() && b >= i64::MIN as f64 && b < i64::MAX as f64).then_some(b as i64)
}

/// Stretch of an anchor pair to a reference: `(high − low) / (p_high −
/// p_low)`; NaN when the reference is `None` or either span is not `> 0`.
pub(crate) fn stretch_need(r: &RefPair, p_low: f64, p_high: f64) -> f64 {
    let (rs, ds) = (r.high - r.low, p_high - p_low);
    if r.source == RefSource::None || rs.is_nan() || ds.is_nan() || rs <= 0.0 || ds <= 0.0 {
        f64::NAN
    } else {
        rs / ds
    }
}

/// Second pick of one eligible voter (4.5) against its build-1 reference
/// `w1`: `TooShort` when `n < pick_min_count`, `StretchOut` when the stretch
/// is finite and outside `[1/gain_cap, gain_cap]`, else `None` (kept; a NaN
/// stretch keeps the voter).
pub(crate) fn second_pick(anchor: &Anchor, w1: &RefPair, cfg: &NormalizeConfig) -> VoterExclusion {
    if anchor.n < cfg.pick_min_count {
        return VoterExclusion::TooShort;
    }
    let need = stretch_need(w1, anchor.p_low, anchor.p_high);
    if need.is_finite() && (need > cfg.gain_cap || need < 1.0 / cfg.gain_cap) {
        VoterExclusion::StretchOut
    } else {
        VoterExclusion::None
    }
}

/// Target of the depth-aware pass.
pub(crate) struct DepthTarget<'a> {
    pub well: WellId,
    pub xy: Option<(f64, f64)>,
    pub scope: Scope,
    /// `B_t`: layer bins of the target's scope samples.
    pub bins: &'a BTreeSet<i64>,
    /// `|B_fleet|`.
    pub fleet_bins: usize,
}

/// One final eligible voter as seen by the depth-aware pass.
pub(crate) struct DepthVoter<'a> {
    pub well: WellId,
    pub xy: (f64, f64),
    /// Native values after pileups, on the voter's grid.
    pub values: &'a [f32],
    /// Layer bin per sample of the voter's grid.
    pub bins: &'a [Option<i64>],
    pub p_low: f64,
    pub p_high: f64,
    /// The voter's own whole-interval reference `W_v`.
    pub whole: RefPair,
}

/// Result of the depth-aware pass of one target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DepthResult {
    pub reason: DepthReason,
    /// `|B_t| / |B_fleet|`; NaN unless computed.
    pub cover: f64,
    /// Inverse-distance pair over the qualifying voters; NaN unless computed.
    pub low: f64,
    pub high: f64,
    /// Qualifying voters.
    pub n_voters: usize,
}

/// Depth-aware reference by layer coordinate (4.7).
///
/// `NoLayer` for scope `all` or an empty `B_t`; `WideCover` when `cover >=
/// depth_max_cover`; `WholeFallback` without target XY. Otherwise the
/// voters other than the target, nearest first, at most `depth_candidates`:
/// each needs `qc.stats_min_count` finite samples in the bins of `B_t`; its
/// percentiles at `qc.stats_pct` are mapped with its own transform onto
/// `W_v` (identity when `W_v` is `None`) and swapped if reversed. Fewer than
/// `ref_min_voters` qualifying, or a pair failing the span sanity against the
/// medians of all qualifying pairs → `WholeFallback`; else `LayerWindow`.
pub(crate) fn depth_pair(
    t: &DepthTarget<'_>,
    voters: &[DepthVoter<'_>],
    qc: &QcConfig,
    cfg: &NormalizeConfig,
) -> DepthResult {
    let mut out = DepthResult {
        reason: DepthReason::NoLayer,
        cover: f64::NAN,
        low: f64::NAN,
        high: f64::NAN,
        n_voters: 0,
    };
    let (Some(&b_min), Some(&b_max)) = (t.bins.first(), t.bins.last()) else {
        return out;
    };
    if t.scope == Scope::All || t.fleet_bins == 0 {
        return out;
    }
    out.cover = t.bins.len() as f64 / t.fleet_bins as f64;
    if out.cover >= cfg.depth_max_cover {
        out.reason = DepthReason::WideCover;
        return out;
    }
    out.reason = DepthReason::WholeFallback;
    let Some(at) = finite_xy(t.xy) else {
        return out;
    };
    let offset = |b: i64| b.checked_sub(b_min).and_then(|d| usize::try_from(d).ok());
    let dense: Option<Vec<bool>> = offset(b_max)
        .and_then(|d| d.checked_add(1))
        .filter(|&span| span <= MAX_DENSE_BINS)
        .map(|span| {
            let mut member = vec![false; span];
            for &b in t.bins {
                if let Some(m) = offset(b).and_then(|i| member.get_mut(i)) {
                    *m = true;
                }
            }
            member
        });
    let in_bt = |b: i64| match &dense {
        Some(member) => offset(b)
            .and_then(|i| member.get(i))
            .copied()
            .unwrap_or(false),
        None => t.bins.contains(&b),
    };

    let ranked = rank(t.well, at, voters.iter().map(|v| (v.well, v.xy, v)));
    let mut qualifying: Vec<(f64, WellId, (f64, f64))> = Vec::new();
    for (d, id, v) in ranked.into_iter().take(cfg.depth_candidates) {
        let sel: Vec<f64> = v
            .values
            .iter()
            .zip(v.bins)
            .filter(|(x, b)| x.is_finite() && b.is_some_and(in_bt))
            .map(|(x, _)| f64::from(*x))
            .collect();
        if sel.is_empty() || sel.len() < qc.stats_min_count {
            continue;
        }
        let (q_low, q_high) = match percentiles(&sel, &qc.stats_pct).as_deref() {
            Ok([a, b]) => (*a, *b),
            _ => continue,
        };
        let go = if v.whole.source == RefSource::None {
            GainOffset::identity(cfg.gain_cap)
        } else {
            gain_offset(
                v.p_low,
                v.p_high,
                v.whole.low,
                v.whole.high,
                cfg.gain_cap,
                cfg.shift_only_frac,
                cfg.degenerate_span,
            )
        };
        let lo = go.gain * q_low + go.offset;
        let hi = go.gain * q_high + go.offset;
        if !(lo.is_finite() && hi.is_finite()) {
            continue;
        }
        qualifying.push((d, id, if hi < lo { (hi, lo) } else { (lo, hi) }));
    }
    out.n_voters = qualifying.len();
    if qualifying.is_empty() || qualifying.len() < cfg.ref_min_voters {
        return out;
    }
    let lows: Vec<f64> = qualifying.iter().map(|q| q.2 .0).collect();
    let highs: Vec<f64> = qualifying.iter().map(|q| q.2 .1).collect();
    let (low, high, _) = idw_sum(&qualifying, cfg);
    out.low = low;
    out.high = high;
    if sane(
        low,
        high,
        (median(&lows), median(&highs)),
        cfg.ref_span_ratio_max,
    ) {
        out.reason = DepthReason::LayerWindow;
    }
    out
}
