// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Consistency jury: a strictly leave-one-out reference curve per
//! well and log, a sliding-window judge, and the blanking of convictions.
//!
//! Nearby wells (jurors) are mapped onto the judged well through the
//! rough-alignment pair paths, or through the layer coordinate when no path
//! is usable, and combined row by row with a weighted lower median. The judged
//! well is never a juror and its own values are never mixed into its
//! reference. Values are judged in native space (RESD in log10; the caller
//! passes native values); indices are local samples on each well's MD grid (0.5 ft). Every
//! channel is judged on the same input; blanking happens only in
//! [`apply_jury`], so the result does not depend on order.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
use tracing::{debug, info};

use super::{apply_mask, blank_channel, QcFlags, QC_JURY_SEGMENT};
use crate::align::{PairMap, PairStatus};
use crate::config::QcJuryConfig;
use crate::error::{SxError, SxResult};
use crate::types::{
    GridRange, IntervalRow, IntervalState, LogKind, WellGeometry, WellHeader, WellId, WellLogs,
    GRID_STEP_FT,
};
use crate::util::weighted_median;

/// Epsilon of the interval of a layer coordinate: `floor(L + 1e-9)`.
const LAYER_EPS: f64 = 1e-9;

/// Inputs of the jury. Read only; wells are matched by `WellId`, so slice
/// order does not matter.
#[derive(Debug, Clone, Copy)]
pub struct JuryInput<'a> {
    /// Well headers; surface `x_m`, `y_m` (m) give the juror distances.
    pub wells: &'a [WellHeader],
    /// Native values (RESD in log10), one entry per well, grids as in
    /// `logs_qc`. The jury does not convert them again. Read only.
    pub logs: &'a [WellLogs],
    /// Per-sample geometry; `tvd_ft` gives the TVD thickness of intervals.
    pub geometry: &'a [WellGeometry],
    /// Interval rows of every well (boundary MDs in ft, state).
    pub intervals: &'a [IntervalRow],
    /// Layer coordinate per sample on each well's grid (interval + fraction,
    /// NaN where undefined). A well without an entry is all NaN.
    pub layer: &'a BTreeMap<WellId, Vec<f64>>,
    /// pairs_rough: one `PairMap` per directed pair, absolute grid indices.
    pub pairs: &'a [PairMap],
    /// (well, log) channels flagged by the provisional ring test in stage 2
    /// (any sigma bit): still judged, never jurors for that log.
    pub off_jury: &'a BTreeSet<(WellId, LogKind)>,
}

/// Verdict of one channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JuryVerdict {
    /// Judged; convicted share below `bad_share` (segments may remain).
    Ok = 0,
    /// Convicted share at least `bad_share`: the whole channel is blanked.
    Bad = 1,
    /// Pool below quorum, no XY, no jurors, or no scoreable window.
    NoVerdict = 2,
    /// Fewer than `min_count` finite samples.
    TooFew = 3,
}

/// Convicted range, local sample indices on the well's grid, inclusive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JurySegment {
    /// First convicted sample.
    pub i_lo: usize,
    /// Last convicted sample.
    pub i_hi: usize,
    /// Smallest finite window correlation in the segment; NaN if none.
    pub corr_min: f64,
    /// At least one window in the segment was flagged by the flat rule.
    pub flat: bool,
}

/// Leave-one-out reference of one channel, on the judged well's grid.
#[derive(Debug, Clone, PartialEq)]
pub struct JuryReference {
    /// Reference value per sample (native space); NaN where the target is NaN
    /// or no tier reaches the quorum.
    pub values: Vec<f64>,
    /// Contributions used per sample (capped at 255); 0 where `values` is NaN.
    pub n_contrib: Vec<u8>,
    /// Distinct jurors used at one or more samples, ascending `WellId`.
    pub jurors: Vec<WellId>,
}

/// Window-judge result of one channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Judgement {
    /// Pearson correlation over all joint rows; NaN below `corr_min_count`
    /// joint rows or with a zero std.
    pub corr: f64,
    /// Scoreable windows over both scales.
    pub n_windows: usize,
    /// Flagged windows over both scales.
    pub n_flagged: usize,
    /// Merged convicted segments, ascending.
    pub segments: Vec<JurySegment>,
    /// Finite samples inside the segments / finite samples (0 when none).
    pub convicted_frac: f64,
}

/// Jury result of one (well, log) channel.
#[derive(Debug, Clone, PartialEq)]
pub struct JuryResult {
    /// Judged well.
    pub well: WellId,
    /// Judged log.
    pub log: LogKind,
    /// Verdict.
    pub verdict: JuryVerdict,
    /// Jury correlation (see [`Judgement::corr`]); NaN when not judged.
    pub corr: f64,
    /// Convicted share (see [`Judgement::convicted_frac`]); NaN when not judged.
    pub convicted_frac: f64,
    /// Scoreable windows.
    pub n_windows: usize,
    /// Flagged windows.
    pub n_flagged: usize,
    /// Distinct jurors used.
    pub n_jurors: usize,
    /// Convicted segments (kept for `Bad`, applied for `Ok`).
    pub segments: Vec<JurySegment>,
}

/// Samples blanked by [`apply_jury`], per acted-on channel.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JuryCounts {
    /// Samples blanked with `QC_JURY_SEGMENT` (verdict `Ok` with segments).
    pub n_segment_samples: BTreeMap<(WellId, LogKind), usize>,
    /// Samples blanked with `QC_CHANNEL` (verdict `Bad`).
    pub n_channel_samples: BTreeMap<(WellId, LogKind), usize>,
}

/// Leave-one-out reference for one channel. Errors: NotFound if `target` is not in `input.logs`.
///
/// Jurors: the pool (wells with XY, at least `min_count` finite values of
/// `log`, not in `off_jury`) minus `target`, within `max_dist_m`, ranked by
/// `(distance, WellId)`, first `max_jurors`; tier 1 = first `near_jurors`.
/// Each juror is mapped onto the target per row through a usable pair path,
/// else the layer map (factor `unaligned_weight`); rows without a layer
/// coordinate use a depth-map path (factor = path quality, interval factor 1;
/// preference as for interval paths). Weight `q × g / max(d, dist_floor_m)^dist_power`.
/// Reference = lower weighted median (ties by lower `WellId`) of tier 1 if it
/// gives `>= quorum` contributions, else of tier 2, else NaN. The target's own
/// values only decide which rows get a reference. TVD outside a well's
/// geometry grid extends vertically.
///
/// Units: native (RESD in log10). NaN rule: skip. Other errors: `Shape` when a
/// curve, layer or geometry vector differs from its grid; `InvalidInput` for a
/// repeated well in `logs`. Complexity: O(total samples) to index the input,
/// plus O(R × J log J) for R target rows and J ≤ `max_jurors` jurors.
pub fn jury_reference(
    target: WellId,
    log: LogKind,
    input: &JuryInput<'_>,
    cfg: &QcJuryConfig,
) -> SxResult<JuryReference> {
    let ctx = Ctx::new(input)?;
    let ti = ctx
        .index
        .get(&target)
        .copied()
        .ok_or_else(|| SxError::NotFound(format!("jury: well {} not in logs", target.0)))?;
    let pool = ctx.pool(log, cfg);
    let tv = ctx.wells.get(ti).map(|t| t.native(log)).unwrap_or_default();
    ctx.build_reference(ti, log, &tv, &pool, cfg)
}

/// Window judge. `values` and `reference` have equal length (Shape error otherwise). NaN rule: skip.
///
/// Windows start at the first finite value: per scale `ft` in `window_ft`,
/// `W = max(window_min, round(ft / 0.5))`, hop `max(hop_min, W / 2)`, while
/// the window ends at or before the last finite value. A window is scoreable
/// with `>= max(joint_floor, ceil(joint_share × W))` joint finite rows and
/// flagged when `corr < anticorr_limit` or the values are flat
/// (`std_v < flat_std_share × std_r`, population stds, `std_r > 0`). A streak
/// of `>= min_streak` flagged windows in one scale is convicted; an
/// unflagged or unscoreable window breaks a streak. Segments of both scales
/// are merged when they overlap.
///
/// Units: any (same for both inputs). Errors: `Shape`. No finite value: no
/// windows, `corr` NaN, `convicted_frac` 0. Complexity: O(n) per scale.
pub fn judge_channel(values: &[f64], reference: &[f64], cfg: &QcJuryConfig) -> SxResult<Judgement> {
    if values.len() != reference.len() {
        return Err(SxError::Shape(format!(
            "judge_channel: values {}, reference {}",
            values.len(),
            reference.len()
        )));
    }
    let finite = |i: &usize| values.get(*i).is_some_and(|v| v.is_finite());
    let (Some(first), Some(last)) = (
        (0..values.len()).find(finite),
        (0..values.len()).rev().find(finite),
    ) else {
        return Ok(Judgement {
            corr: f64::NAN,
            n_windows: 0,
            n_flagged: 0,
            segments: Vec::new(),
            convicted_frac: 0.0,
        });
    };

    let mut segments = Vec::new();
    let (mut n_windows, mut n_flagged) = (0usize, 0usize);
    for ft in cfg.window_ft {
        let w = window_len(ft, cfg.window_min);
        let hop = cfg.hop_min.max(w / 2).max(1);
        let need = cfg.joint_floor.max(ceil_usize(cfg.joint_share * w as f64));
        let mut streak: Option<Streak> = None;
        let mut s = first;
        while s + w <= last + 1 {
            let (Some(v), Some(r)) = (values.get(s..s + w), reference.get(s..s + w)) else {
                break;
            };
            let win = joint_stats(v, r);
            let mut flagged = false;
            if win.n >= need {
                n_windows += 1;
                let flat = win.std_r > 0.0 && win.std_v < cfg.flat_std_share * win.std_r;
                if win.corr < cfg.anticorr_limit || flat {
                    flagged = true;
                    n_flagged += 1;
                    let st = streak.get_or_insert(Streak {
                        s_first: s,
                        s_last: s,
                        n: 0,
                        corr_min: f64::NAN,
                        flat: false,
                    });
                    st.s_last = s;
                    st.n += 1;
                    st.corr_min = st.corr_min.min(win.corr);
                    st.flat |= flat;
                }
            }
            if !flagged {
                close_streak(streak.take(), w, cfg.min_streak, &mut segments);
            }
            s += hop;
        }
        close_streak(streak.take(), w, cfg.min_streak, &mut segments);
    }

    segments.sort_by_key(|seg| seg.i_lo);
    let mut merged: Vec<JurySegment> = Vec::new();
    for seg in segments {
        match merged.last_mut() {
            Some(cur) if seg.i_lo <= cur.i_hi => {
                cur.i_hi = cur.i_hi.max(seg.i_hi);
                cur.corr_min = cur.corr_min.min(seg.corr_min);
                cur.flat |= seg.flat;
            }
            _ => merged.push(seg),
        }
    }

    let count_finite = |v: &[f64]| v.iter().filter(|x| x.is_finite()).count();
    let convicted: usize = merged
        .iter()
        .map(|seg| values.get(seg.i_lo..=seg.i_hi).map_or(0, count_finite))
        .sum();
    let all = joint_stats(
        values.get(first..=last).unwrap_or_default(),
        reference.get(first..=last).unwrap_or_default(),
    );
    Ok(Judgement {
        corr: if all.n >= cfg.corr_min_count {
            all.corr
        } else {
            f64::NAN
        },
        n_windows,
        n_flagged,
        segments: merged,
        convicted_frac: convicted as f64 / count_finite(values) as f64,
    })
}

/// All wells × all six logs; sorted by (WellId, suite order).
///
/// Per channel: fewer than `min_count` finite values → `TooFew`; pool below
/// `quorum` or target without XY → `NoVerdict`; else the reference
/// ([`jury_reference`]) is judged ([`judge_channel`]): no jurors or no
/// scoreable window → `NoVerdict`, `convicted_frac >= bad_share` → `Bad`,
/// else `Ok`. Unjudged channels have NaN `corr` and `convicted_frac`.
///
/// Empty `logs` → empty result. Errors as [`jury_reference`]. Parallel over
/// (well, log) with rayon; results are collected, then sorted, so they do not
/// depend on the thread count. Complexity: see [`jury_reference`], per channel.
pub fn run_jury(input: &JuryInput<'_>, cfg: &QcJuryConfig) -> SxResult<Vec<JuryResult>> {
    if input.logs.is_empty() {
        return Ok(Vec::new());
    }
    let ctx = Ctx::new(input)?;
    let pools: Vec<(LogKind, Vec<usize>)> = LogKind::ALL
        .into_iter()
        .map(|log| (log, ctx.pool(log, cfg)))
        .collect();
    let tasks: Vec<(usize, LogKind, &[usize])> = (0..ctx.wells.len())
        .flat_map(|ti| {
            pools
                .iter()
                .map(move |(log, pool)| (ti, *log, pool.as_slice()))
        })
        .collect();
    let per_task: Vec<SxResult<JuryResult>> = tasks
        .par_iter()
        .map(|&(ti, log, pool)| ctx.judge_one(ti, log, pool, cfg))
        .collect();
    let mut out = per_task.into_iter().collect::<SxResult<Vec<_>>>()?;
    out.sort_by_key(|r| (r.well, r.log));
    let count = |v: JuryVerdict| out.iter().filter(|r| r.verdict == v).count();
    info!(
        n_channels = out.len(),
        ok = count(JuryVerdict::Ok),
        bad = count(JuryVerdict::Bad),
        no_verdict = count(JuryVerdict::NoVerdict),
        too_few = count(JuryVerdict::TooFew),
        "jury: verdicts"
    );
    Ok(out)
}

/// Blanks convictions in the stage-4 working copy and ORs bits into the flags.
///
/// `Bad` → [`blank_channel`] (bit `QC_CHANNEL`), count in
/// `n_channel_samples`. `Ok` with segments → finite samples inside the
/// segments become NaN with bit `QC_JURY_SEGMENT`, count in
/// `n_segment_samples`. Other results are not touched and get no count entry.
/// Wells are matched by `WellId`.
///
/// NaN rule: skip (NaN samples keep their bits). Errors: `NotFound` when an
/// acted-on channel has no curve in `logs` or no bits in `flags`; `Shape` when
/// a curve and its bits differ in length. Complexity: O(acted-on samples).
pub fn apply_jury(
    results: &[JuryResult],
    logs: &mut [WellLogs],
    flags: &mut [QcFlags],
) -> SxResult<JuryCounts> {
    let log_at: BTreeMap<WellId, usize> = logs.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
    let flag_at: BTreeMap<WellId, usize> =
        flags.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
    let mut counts = JuryCounts::default();
    for r in results {
        let whole = match r.verdict {
            JuryVerdict::Bad => true,
            JuryVerdict::Ok if !r.segments.is_empty() => false,
            _ => continue,
        };
        let missing = |what: &str| {
            SxError::NotFound(format!(
                "jury: well {} {} has no {what}",
                r.well.0,
                r.log.name()
            ))
        };
        let values = log_at
            .get(&r.well)
            .and_then(|&i| logs.get_mut(i))
            .and_then(|l| l.curves.get_mut(&r.log))
            .ok_or_else(|| missing("curve"))?;
        let bits = flag_at
            .get(&r.well)
            .and_then(|&i| flags.get_mut(i))
            .and_then(|f| f.bits.get_mut(&r.log))
            .ok_or_else(|| missing("QC bits"))?;
        if whole {
            let n = blank_channel(values, bits)?;
            *counts.n_channel_samples.entry((r.well, r.log)).or_default() += n;
        } else {
            let mut mask = vec![false; values.len()];
            for seg in &r.segments {
                for m in mask
                    .iter_mut()
                    .take(seg.i_hi.saturating_add(1))
                    .skip(seg.i_lo)
                {
                    *m = true;
                }
            }
            let n = apply_mask(values, bits, &mask, QC_JURY_SEGMENT)?;
            *counts.n_segment_samples.entry((r.well, r.log)).or_default() += n;
        }
    }
    Ok(counts)
}

/// Interval of a layer coordinate, `min(floor(L + 1e-9), cap)` (the rough
/// alignment rule: a sample exactly on the lower pad boundary stays in the
/// lower pad); `None` when `L` is not finite or the floor is outside
/// `0..=u16::MAX`. `cap` = the last interval index of the project, if known.
fn interval_of(l: f64, cap: Option<u16>) -> Option<u16> {
    let f = (l + LAYER_EPS).floor();
    let iv = (f.is_finite() && (0.0..=f64::from(u16::MAX)).contains(&f)).then_some(f as u16);
    iv.map(|v| cap.map_or(v, |c| v.min(c)))
}

/// TVD (ft) at `md_ft`: linear between geometry samples, vertical extension
/// beyond the first and last sample; NaN for a NaN depth or empty geometry.
fn tvd_at_md(geom: &WellGeometry, md_ft: f64) -> f64 {
    let n = geom.grid.len.min(geom.tvd_ft.len());
    let (Some(&tvd_first), Some(&tvd_last)) = (
        geom.tvd_ft.first(),
        n.checked_sub(1).and_then(|i| geom.tvd_ft.get(i)),
    ) else {
        return f64::NAN;
    };
    if md_ft.is_nan() {
        return f64::NAN;
    }
    let (md_first, md_last) = (geom.grid.md_ft(0), geom.grid.md_ft(n - 1));
    if md_ft <= md_first {
        return tvd_first - (md_first - md_ft);
    }
    if md_ft >= md_last {
        return tvd_last + (md_ft - md_last);
    }
    let x = (md_ft - md_first) / GRID_STEP_FT;
    let i = x.floor();
    let f = x - i;
    let i = i as usize;
    match (geom.tvd_ft.get(i), geom.tvd_ft.get(i + 1)) {
        (Some(&a), _) if f == 0.0 => a,
        (Some(&a), Some(&b)) => a + f * (b - a),
        _ => f64::NAN,
    }
}

/// Window length in samples: `max(window_min, round(ft / 0.5))`, at least 1.
fn window_len(ft: f64, window_min: usize) -> usize {
    let r = (ft / GRID_STEP_FT).round();
    let r = if r.is_finite() && r > 0.0 {
        r as usize
    } else {
        0
    };
    window_min.max(r).max(1)
}

fn ceil_usize(x: f64) -> usize {
    let c = x.ceil();
    if c.is_finite() && c > 0.0 {
        c as usize
    } else {
        0
    }
}

/// Run of consecutive flagged windows in one scale.
struct Streak {
    s_first: usize,
    s_last: usize,
    n: usize,
    corr_min: f64,
    flat: bool,
}

fn close_streak(streak: Option<Streak>, w: usize, min_streak: usize, out: &mut Vec<JurySegment>) {
    if let Some(st) = streak.filter(|st| st.n >= min_streak) {
        out.push(JurySegment {
            i_lo: st.s_first,
            i_hi: st.s_last + w - 1,
            corr_min: st.corr_min,
            flat: st.flat,
        });
    }
}

/// Pearson correlation and population stds over rows where both are finite.
struct Joint {
    n: usize,
    corr: f64,
    std_v: f64,
    std_r: f64,
}

fn joint_stats(v: &[f64], r: &[f64]) -> Joint {
    let rows = || {
        v.iter()
            .zip(r)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
    };
    let (mut n, mut sum_v, mut sum_r) = (0usize, 0.0f64, 0.0f64);
    for (a, b) in rows() {
        n += 1;
        sum_v += a;
        sum_r += b;
    }
    if n == 0 {
        return Joint {
            n,
            corr: f64::NAN,
            std_v: f64::NAN,
            std_r: f64::NAN,
        };
    }
    let nf = n as f64;
    let (mean_v, mean_r) = (sum_v / nf, sum_r / nf);
    let (mut svv, mut srr, mut svr) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in rows() {
        let (da, db) = (a - mean_v, b - mean_r);
        svv += da * da;
        srr += db * db;
        svr += da * db;
    }
    let corr = if svv > 0.0 && srr > 0.0 {
        svr / (svv * srr).sqrt()
    } else {
        f64::NAN
    };
    Joint {
        n,
        corr,
        std_v: (svv / nf).sqrt(),
        std_r: (srr / nf).sqrt(),
    }
}

/// Donor value at fractional global grid index `kf`:
/// exact sample when `kf` is whole, else linear between two finite samples;
/// NaN otherwise or outside the donor grid.
fn donor_value(values: &[f64], grid: GridRange, kf: f64) -> f64 {
    if !kf.is_finite() {
        return f64::NAN;
    }
    let j = kf.floor();
    let f = kf - j;
    let local = j - f64::from(grid.k_start);
    if local < 0.0 || local >= values.len() as f64 {
        return f64::NAN;
    }
    let local = local as usize;
    let Some(&a) = values.get(local) else {
        return f64::NAN;
    };
    if f == 0.0 {
        return a;
    }
    match values.get(local + 1) {
        Some(&b) if a.is_finite() && b.is_finite() => (1.0 - f) * a + f * b,
        _ => f64::NAN,
    }
}

/// A pair path oriented from the target: distinct target indices `k_t`,
/// ascending, each with the mean of its donor indices `k_d`.
struct OMap<'a> {
    pm: &'a PairMap,
    from_target: bool,
    keys: Vec<(i32, f64)>,
}

impl<'a> OMap<'a> {
    fn new(pm: &'a PairMap, target: WellId) -> Self {
        let from_target = pm.path.a == target;
        let mut pts: Vec<(i32, i32)> = pm
            .path
            .path
            .iter()
            .map(|&(a, b)| if from_target { (a, b) } else { (b, a) })
            .collect();
        pts.sort_by_key(|p| p.0);
        let keys = pts
            .chunk_by(|x, y| x.0 == y.0)
            .filter_map(|run| {
                let first = run.first()?;
                let sum: f64 = run.iter().map(|p| f64::from(p.1)).sum();
                Some((first.0, sum / run.len() as f64))
            })
            .collect();
        Self {
            pm,
            from_target,
            keys,
        }
    }

    fn covers(&self, k: i32) -> bool {
        matches!(
            (self.keys.first(), self.keys.last()),
            (Some(lo), Some(hi)) if lo.0 <= k && k <= hi.0
        )
    }

    /// `kf` at target index `k`: the key's value, else linear between the two
    /// neighboring keys; `None` outside the path.
    fn sample(&self, k: i32) -> Option<f64> {
        let p = self.keys.partition_point(|e| e.0 < k);
        let &(k_hi, f_hi) = self.keys.get(p)?;
        if k_hi == k {
            return Some(f_hi);
        }
        let &(k_lo, f_lo) = self.keys.get(p.checked_sub(1)?)?;
        let t = (f64::from(k) - f64::from(k_lo)) / (f64::from(k_hi) - f64::from(k_lo));
        Some(f_lo + t * (f_hi - f_lo))
    }
}

/// Alignment factor of an interval-mode path for interval `iv`.
fn interval_factor(m: &OMap<'_>, iv: u16) -> Option<f64> {
    if m.pm.status != PairStatus::IntervalMap {
        return None;
    }
    let pos = m.pm.intervals.iter().position(|&x| x == iv)?;
    m.pm.interval_quality.get(pos).map(|&q| f64::from(q))
}

/// Alignment factor of a depth-mode path.
fn depth_factor(m: &OMap<'_>) -> Option<f64> {
    (m.pm.status == PairStatus::DepthMap).then_some(f64::from(m.pm.path.quality))
}

/// Usable map at target index `k` with its factor: `factor(m)` must exist and
/// be `>= min_q`, and the map must cover `k`. Preference: path from the
/// target, then higher factor, then input order.
fn best_map<'m, 'a>(
    maps: &'m [OMap<'a>],
    k: i32,
    min_q: f64,
    factor: impl Fn(&OMap<'a>) -> Option<f64>,
) -> Option<(&'m OMap<'a>, f64)> {
    let mut best: Option<(&OMap<'a>, f64)> = None;
    for m in maps {
        let Some(q) = factor(m) else { continue };
        if q.is_nan() || q < min_q || !m.covers(k) {
            continue;
        }
        let better = match best {
            None => true,
            Some((b, bq)) => {
                (m.from_target && !b.from_target) || (m.from_target == b.from_target && q > bq)
            }
        };
        if better {
            best = Some((m, q));
        }
    }
    best
}

/// One layer-map segment of a donor: local index `j`, `lo = L[j] < hi = L[j + 1]`.
#[derive(Debug, Clone, Copy)]
struct Seg {
    j: usize,
    lo: f64,
    hi: f64,
}

/// One well as seen by the jury.
struct JWell<'a> {
    id: WellId,
    xy: Option<(f64, f64)>,
    logs: &'a WellLogs,
    layer: Option<&'a [f64]>,
    segs: Vec<Seg>,
    /// Segments ascending and non-overlapping, so a binary search finds the
    /// lowest containing segment.
    segs_sorted: bool,
    n_finite: BTreeMap<LogKind, usize>,
    /// `cov(i)` per (log, interval).
    cov: BTreeMap<(LogKind, u16), f64>,
    /// Last interval index of the project (cap for [`interval_of`]).
    last_iv: Option<u16>,
}

impl<'a> JWell<'a> {
    fn new(
        id: WellId,
        xy: Option<(f64, f64)>,
        logs: &'a WellLogs,
        layer: Option<&'a [f64]>,
        last_iv: Option<u16>,
    ) -> Self {
        let coord = layer.unwrap_or_default();
        let ivs: Vec<Option<u16>> = coord.iter().map(|&l| interval_of(l, last_iv)).collect();
        let mut totals: BTreeMap<u16, usize> = BTreeMap::new();
        for &iv in ivs.iter().flatten() {
            *totals.entry(iv).or_default() += 1;
        }
        let mut n_finite = BTreeMap::new();
        let mut cov = BTreeMap::new();
        for (&log, values) in &logs.curves {
            let mut n = 0usize;
            let mut fin: BTreeMap<u16, usize> = BTreeMap::new();
            for (i, &v) in values.iter().enumerate() {
                if v.is_finite() {
                    n += 1;
                    if let Some(&Some(iv)) = ivs.get(i) {
                        *fin.entry(iv).or_default() += 1;
                    }
                }
            }
            n_finite.insert(log, n);
            for (&iv, &tot) in &totals {
                let f = fin.get(&iv).copied().unwrap_or(0);
                cov.insert((log, iv), f as f64 / tot as f64);
            }
        }
        let segs: Vec<Seg> = coord
            .windows(2)
            .enumerate()
            .filter_map(|(j, w)| match *w {
                [lo, hi] if lo.is_finite() && hi.is_finite() && hi > lo => Some(Seg { j, lo, hi }),
                _ => None,
            })
            .collect();
        let segs_sorted = segs.windows(2).all(|p| matches!(p, [a, b] if a.hi <= b.lo));
        Self {
            id,
            xy,
            logs,
            layer,
            segs,
            segs_sorted,
            n_finite,
            cov,
            last_iv,
        }
    }

    fn n_finite(&self, log: LogKind) -> usize {
        self.n_finite.get(&log).copied().unwrap_or(0)
    }

    /// Native values of `log` on the well's grid (all NaN if absent).
    fn native(&self, log: LogKind) -> Vec<f64> {
        match self.logs.curves.get(&log) {
            Some(v) => v.iter().map(|&x| f64::from(x)).collect(),
            None => vec![f64::NAN; self.logs.grid.len],
        }
    }

    /// Layer map: global fractional grid index of layer
    /// coordinate `x` in the lowest segment `L[j] <= x <= L[j + 1]`.
    fn layer_map(&self, x: f64) -> Option<f64> {
        let seg = if self.segs_sorted {
            let p = self.segs.partition_point(|s| s.hi < x);
            self.segs.get(p).filter(|s| s.lo <= x)
        } else {
            self.segs.iter().find(|s| s.lo <= x && x <= s.hi)
        }?;
        Some(f64::from(self.logs.grid.k(seg.j)) + (x - seg.lo) / (seg.hi - seg.lo))
    }
}

/// One juror of one target and log.
struct Juror<'c, 'a> {
    id: WellId,
    tier1: bool,
    /// `max(d, dist_floor_m)^dist_power`.
    dist_div: f64,
    well: &'c JWell<'a>,
    values: Vec<f64>,
    /// Interval factor `g` per interval index (0 where absent).
    g: Vec<f64>,
    /// Interval-mode and depth-mode paths of the pair, input order.
    maps: Vec<OMap<'a>>,
}

impl Juror<'_, '_> {
    /// Contribution `(value, weight)` at target grid index `k` with layer
    /// coordinate `lt` and its interval `iv`.
    fn contribution(
        &self,
        k: i32,
        lt: f64,
        iv: Option<u16>,
        cfg: &QcJuryConfig,
    ) -> Option<(f64, f64)> {
        let (kf, q, g) = if lt.is_finite() {
            let iv = iv?;
            let g = self.g.get(usize::from(iv)).copied().unwrap_or(0.0);
            if g <= 0.0 {
                return None;
            }
            match best_map(&self.maps, k, cfg.path_min_quality, |m| {
                interval_factor(m, iv)
            }) {
                Some((m, q)) => (m.sample(k)?, q, g),
                None => (self.well.layer_map(lt)?, cfg.unaligned_weight, g),
            }
        } else {
            let (m, q) = best_map(&self.maps, k, cfg.path_min_quality, depth_factor)?;
            (m.sample(k)?, q, 1.0)
        };
        let v = donor_value(&self.values, self.well.logs.grid, kf);
        let w = q * g / self.dist_div;
        (v.is_finite() && w.is_finite() && w > 0.0).then_some((v, w))
    }
}

/// Contributions of one tier at one row, in juror (`WellId`) order.
#[derive(Default)]
struct Tier {
    values: Vec<f64>,
    weights: Vec<f64>,
    who: Vec<usize>,
}

impl Tier {
    fn clear(&mut self) {
        self.values.clear();
        self.weights.clear();
        self.who.clear();
    }

    fn push(&mut self, v: f64, w: f64, who: usize) {
        self.values.push(v);
        self.weights.push(w);
        self.who.push(who);
    }
}

/// Indexed input shared by every channel.
struct Ctx<'a> {
    /// Wells of `logs`, ascending `WellId`.
    wells: Vec<JWell<'a>>,
    index: BTreeMap<WellId, usize>,
    /// Pair maps per unordered well pair `(lower, higher)`, input order.
    pairs: BTreeMap<(WellId, WellId), Vec<&'a PairMap>>,
    /// TVD thickness per (well, interval); `None` when the state is
    /// `Pinched` / `Unknown` or the thickness is not finite and `> 0`.
    thick: BTreeMap<(WellId, u16), Option<f64>>,
    off_jury: &'a BTreeSet<(WellId, LogKind)>,
}

impl<'a> Ctx<'a> {
    fn new(input: &JuryInput<'a>) -> SxResult<Self> {
        let mut headers: BTreeMap<WellId, &WellHeader> = BTreeMap::new();
        for h in input.wells {
            headers.entry(h.id).or_insert(h);
        }
        let mut geoms: BTreeMap<WellId, &WellGeometry> = BTreeMap::new();
        for g in input.geometry {
            g.validate()?;
            geoms.entry(g.id).or_insert(g);
        }
        let mut by_id: BTreeMap<WellId, &WellLogs> = BTreeMap::new();
        for l in input.logs {
            for (log, v) in &l.curves {
                if v.len() != l.grid.len {
                    return Err(SxError::Shape(format!(
                        "jury: well {} {} has {} samples, grid has {}",
                        l.id.0,
                        log.name(),
                        v.len(),
                        l.grid.len
                    )));
                }
            }
            if by_id.insert(l.id, l).is_some() {
                return Err(SxError::InvalidInput(format!(
                    "jury: well {} appears twice in logs",
                    l.id.0
                )));
            }
        }
        let mut entries = Vec::with_capacity(by_id.len());
        for (&id, &logs) in &by_id {
            let layer = match input.layer.get(&id) {
                Some(v) if v.len() != logs.grid.len => {
                    return Err(SxError::Shape(format!(
                        "jury: well {} layer coordinate has {} samples, grid has {}",
                        id.0,
                        v.len(),
                        logs.grid.len
                    )))
                }
                Some(v) => Some(v.as_slice()),
                None => None,
            };
            let xy = headers
                .get(&id)
                .filter(|h| h.x_m.is_finite() && h.y_m.is_finite())
                .map(|h| (h.x_m, h.y_m));
            entries.push((id, xy, logs, layer));
        }
        let last_iv = input.intervals.iter().map(|r| r.interval).max();
        let wells: Vec<JWell<'a>> = entries
            .into_par_iter()
            .map(|(id, xy, logs, layer)| JWell::new(id, xy, logs, layer, last_iv))
            .collect();
        let index: BTreeMap<WellId, usize> =
            wells.iter().enumerate().map(|(i, w)| (w.id, i)).collect();

        let mut pairs: BTreeMap<(WellId, WellId), Vec<&'a PairMap>> = BTreeMap::new();
        for pm in input.pairs {
            let (a, b) = (pm.path.a, pm.path.b);
            if !index.contains_key(&a) || !index.contains_key(&b) {
                debug!(
                    a = a.0,
                    b = b.0,
                    "jury: pair map names an unknown well, ignored"
                );
                continue;
            }
            if a != b {
                pairs.entry((a.min(b), a.max(b))).or_default().push(pm);
            }
        }

        let mut thick = BTreeMap::new();
        for row in input.intervals {
            let th = match row.state {
                IntervalState::Pinched | IntervalState::Unknown => None,
                _ => geoms.get(&row.well).and_then(|g| {
                    let th = tvd_at_md(g, row.base_md_ft) - tvd_at_md(g, row.top_md_ft);
                    (th.is_finite() && th > 0.0).then_some(th)
                }),
            };
            thick.entry((row.well, row.interval)).or_insert(th);
        }

        Ok(Self {
            wells,
            index,
            pairs,
            thick,
            off_jury: input.off_jury,
        })
    }

    /// Juror pool of `log`: indices of wells with XY, at least `min_count`
    /// finite values, and not off the jury.
    fn pool(&self, log: LogKind, cfg: &QcJuryConfig) -> Vec<usize> {
        self.wells
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                w.xy.is_some()
                    && w.n_finite(log) >= cfg.min_count
                    && !self.off_jury.contains(&(w.id, log))
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Jurors of target `t`, sorted by `WellId`; empty when `t` has no XY.
    fn jurors(
        &self,
        t: &JWell<'a>,
        log: LogKind,
        pool: &[usize],
        cfg: &QcJuryConfig,
    ) -> Vec<Juror<'_, 'a>> {
        let Some((tx, ty)) = t.xy else {
            return Vec::new();
        };
        let mut cands: Vec<(f64, WellId, &JWell<'a>)> = pool
            .iter()
            .filter_map(|&wi| self.wells.get(wi))
            .filter(|w| w.id != t.id)
            .filter_map(|w| {
                let (x, y) = w.xy?;
                let (dx, dy) = (x - tx, y - ty);
                let d = (dx * dx + dy * dy).sqrt();
                (d <= cfg.max_dist_m).then_some((d, w.id, w))
            })
            .collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        cands.truncate(cfg.max_jurors);
        let t_thick: Vec<(u16, f64)> = self
            .thick
            .range((t.id, 0)..=(t.id, u16::MAX))
            .filter_map(|(&(_, iv), th)| th.map(|th| (iv, th)))
            .collect();
        let mut jurors: Vec<Juror<'_, 'a>> = cands
            .into_iter()
            .enumerate()
            .map(|(rank, (d, id, w))| {
                let maps = self
                    .pairs
                    .get(&(t.id.min(id), t.id.max(id)))
                    .map(|v| {
                        v.iter()
                            .filter(|pm| {
                                matches!(pm.status, PairStatus::IntervalMap | PairStatus::DepthMap)
                            })
                            .map(|pm| OMap::new(pm, t.id))
                            .collect()
                    })
                    .unwrap_or_default();
                Juror {
                    id,
                    tier1: rank < cfg.near_jurors,
                    dist_div: d.max(cfg.dist_floor_m).powf(cfg.dist_power),
                    well: w,
                    values: w.native(log),
                    g: self.g_table(&t_thick, w, log),
                    maps,
                }
            })
            .collect();
        jurors.sort_by_key(|j| j.id);
        jurors
    }

    /// Interval factor `g(t, d, i)` per interval index, from the target's
    /// valid thicknesses `t_thick`.
    fn g_table(&self, t_thick: &[(u16, f64)], d: &JWell<'_>, log: LogKind) -> Vec<f64> {
        let len = t_thick
            .iter()
            .map(|&(iv, _)| usize::from(iv) + 1)
            .max()
            .unwrap_or(0);
        let mut g = vec![0.0; len];
        for &(iv, th_t) in t_thick {
            let Some(&Some(th_d)) = self.thick.get(&(d.id, iv)) else {
                continue;
            };
            let cov = d.cov.get(&(log, iv)).copied().unwrap_or(0.0);
            if let Some(slot) = g.get_mut(usize::from(iv)) {
                *slot = th_t.min(th_d) / th_t.max(th_d) * cov;
            }
        }
        g
    }

    /// Reference of well `ti`; `tv` are its native values (they only select
    /// the rows that get a reference).
    fn build_reference(
        &self,
        ti: usize,
        log: LogKind,
        tv: &[f64],
        pool: &[usize],
        cfg: &QcJuryConfig,
    ) -> SxResult<JuryReference> {
        let t = self
            .wells
            .get(ti)
            .ok_or_else(|| SxError::NotFound(format!("jury: well index {ti}")))?;
        let n = t.logs.grid.len;
        let mut values = vec![f64::NAN; n];
        let mut n_contrib = vec![0u8; n];
        let jurors = self.jurors(t, log, pool, cfg);
        let mut used = vec![false; jurors.len()];
        let (mut near, mut all) = (Tier::default(), Tier::default());
        for (i, (out, nc)) in values.iter_mut().zip(n_contrib.iter_mut()).enumerate() {
            if !tv.get(i).is_some_and(|v| v.is_finite()) {
                continue;
            }
            let k = t.logs.grid.k(i);
            let lt = t.layer.and_then(|l| l.get(i).copied()).unwrap_or(f64::NAN);
            let iv = interval_of(lt, t.last_iv);
            near.clear();
            all.clear();
            for (ji, j) in jurors.iter().enumerate() {
                if let Some((v, w)) = j.contribution(k, lt, iv, cfg) {
                    if j.tier1 {
                        near.push(v, w, ji);
                    }
                    all.push(v, w, ji);
                }
            }
            let tier = if near.values.len() >= cfg.quorum {
                &near
            } else if all.values.len() >= cfg.quorum {
                &all
            } else {
                continue;
            };
            *out = weighted_median(&tier.values, &tier.weights, 0)?;
            *nc = u8::try_from(tier.values.len()).unwrap_or(u8::MAX);
            for &ji in &tier.who {
                if let Some(u) = used.get_mut(ji) {
                    *u = true;
                }
            }
        }
        let jurors = jurors
            .iter()
            .zip(&used)
            .filter(|(_, &u)| u)
            .map(|(j, _)| j.id)
            .collect();
        Ok(JuryReference {
            values,
            n_contrib,
            jurors,
        })
    }

    /// Judge one channel.
    fn judge_one(
        &self,
        ti: usize,
        log: LogKind,
        pool: &[usize],
        cfg: &QcJuryConfig,
    ) -> SxResult<JuryResult> {
        let t = self
            .wells
            .get(ti)
            .ok_or_else(|| SxError::NotFound(format!("jury: well index {ti}")))?;
        let mut res = JuryResult {
            well: t.id,
            log,
            verdict: JuryVerdict::TooFew,
            corr: f64::NAN,
            convicted_frac: f64::NAN,
            n_windows: 0,
            n_flagged: 0,
            n_jurors: 0,
            segments: Vec::new(),
        };
        if t.n_finite(log) < cfg.min_count {
            return Ok(res);
        }
        res.verdict = JuryVerdict::NoVerdict;
        if pool.len() < cfg.quorum || t.xy.is_none() {
            return Ok(res);
        }
        let tv = t.native(log);
        let reference = self.build_reference(ti, log, &tv, pool, cfg)?;
        let judged = judge_channel(&tv, &reference.values, cfg)?;
        res.verdict = if reference.jurors.is_empty() || judged.n_windows == 0 {
            JuryVerdict::NoVerdict
        } else if judged.convicted_frac >= cfg.bad_share {
            JuryVerdict::Bad
        } else {
            JuryVerdict::Ok
        };
        res.corr = judged.corr;
        res.convicted_frac = judged.convicted_frac;
        res.n_windows = judged.n_windows;
        res.n_flagged = judged.n_flagged;
        res.n_jurors = reference.jurors.len();
        res.segments = judged.segments;
        debug!(
            well = t.id.0,
            log = log.name(),
            verdict = res.verdict as u8,
            n_jurors = res.n_jurors,
            "jury: channel"
        );
        Ok(res)
    }
}
