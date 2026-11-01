// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Within-file splice: the primary candidate, filled from shifted and
//! calibrated secondaries.

use std::cmp::Ordering;

use super::families::curve_family;
use super::load::Candidate;
use super::resample::interp_gap;
use crate::types::LogKind;
use crate::units::{physical_bounds, Bounds};
use crate::util::{percentile, percentiles};

/// Smallest overlap (ft) needed to search for a depth shift.
const MIN_OVERLAP_FT: f64 = 50.0;
/// Largest depth shift tried, integer feet, both directions.
const MAX_SHIFT_FT: i32 = 20;
/// Fewest valid samples for a shift search.
const MIN_VALID: usize = 30;
/// Fewest finite shifted samples that score a shift.
const MIN_PAIRS: usize = 10;
/// Percentiles used to calibrate a secondary onto the primary.
const CALIB_PCT: (f64, f64) = (5.0, 95.0);
/// Largest residual-to-scale ratio a calibrated secondary may have.
const MAX_RESID_RATIO: f64 = 1.0;
/// Spread below which a calibration is not possible.
const TINY: f64 = 1e-10;
/// Fewest overlapping rows to calibrate.
const MIN_CALIB_ROWS: usize = 30;

/// Result of splicing one log of one file.
#[derive(Debug, Clone)]
pub(crate) struct Spliced {
    /// Spliced values on the file's depth, NaN = missing.
    pub out: Vec<f32>,
    /// Position of the primary in the candidate slice.
    pub primary: usize,
    /// Secondaries that filled, in order.
    pub merged_from: Vec<String>,
    /// Their shifts, integer feet, same order.
    pub shifts_ft: Vec<i32>,
    /// `MNEM:resid` / `MNEM:cross_family`.
    pub skipped: Vec<String>,
    /// Valid primary rows / rows.
    pub coverage_before_frac: f64,
    /// Finite output rows / rows.
    pub coverage_after_frac: f64,
}

/// Splice outcome of one log of one file.
#[derive(Debug, Clone)]
pub(crate) struct SpliceOutcome {
    /// `None` when no accepted candidate has a valid sample.
    pub spliced: Option<Spliced>,
    /// Positions (in the candidate slice) of accepted candidates with no valid
    /// sample (reason `no_valid`).
    pub no_valid: Vec<usize>,
}

/// One accepted candidate with its ranking facts.
struct Prep<'a> {
    pos: usize,
    col: usize,
    priority: usize,
    mnemonic: &'a str,
    values: &'a [f32],
    span: f64,
    frac: f64,
    family: &'static str,
}

fn is_valid(v: f32, bounds: Bounds) -> bool {
    v.is_finite() && bounds.contains(f64::from(v))
}

fn is_valid64(v: f64, bounds: Bounds) -> bool {
    v.is_finite() && bounds.contains(v)
}

fn pct50(values: &[f64]) -> f64 {
    percentile(values, 50.0).unwrap_or(f64::NAN)
}

/// Valid `(z, value)` points of a series, values as `f64`.
fn valid_points(z: &[f64], v: &[f32], bounds: Bounds) -> (Vec<f64>, Vec<f64>) {
    let mut zs = Vec::new();
    let mut vs = Vec::new();
    for (&zi, &vi) in z.iter().zip(v) {
        if is_valid(vi, bounds) {
            zs.push(zi);
            vs.push(f64::from(vi));
        }
    }
    (zs, vs)
}

/// Integer-foot shift of the secondary that best matches `out` (rule 4.8 step
/// 4.1); `+sh` moves the secondary deeper.
fn best_shift(z: &[f64], out: &[f32], s: &[f32], bounds: Bounds, max_gap_ft: f64) -> i32 {
    let (p_z, p_v) = valid_points(z, out, bounds);
    let (s_z, s_v) = valid_points(z, s, bounds);
    if p_z.len() < MIN_VALID || s_z.len() < MIN_VALID {
        return 0;
    }
    let (Some(&p_lo), Some(&p_hi), Some(&s_lo), Some(&s_hi)) =
        (p_z.first(), p_z.last(), s_z.first(), s_z.last())
    else {
        return 0;
    };
    let lo = p_lo.max(s_lo);
    let hi = p_hi.min(s_hi);
    if hi - lo < MIN_OVERLAP_FT {
        return 0;
    }
    let mut t = Vec::new();
    let mut tv = Vec::new();
    for (&zi, &vi) in p_z.iter().zip(&p_v) {
        if zi >= lo && zi <= hi {
            t.push(zi);
            tv.push(vi);
        }
    }
    if t.len() < MIN_VALID {
        return 0;
    }
    let mut best: Option<(f64, i32)> = None;
    for sh in -MAX_SHIFT_FT..=MAX_SHIFT_FT {
        let x: Vec<f64> = s_z.iter().map(|&v| v + f64::from(sh)).collect();
        let y = interp_gap(&t, &x, &s_v, max_gap_ft);
        let resid: Vec<f64> = tv
            .iter()
            .zip(&y)
            .filter(|(_, y)| y.is_finite())
            .map(|(a, y)| (a - y).abs())
            .collect();
        if resid.len() < MIN_PAIRS {
            continue;
        }
        let score = pct50(&resid);
        if best.is_none_or(|(b, _)| score < b) {
            best = Some((score, sh));
        }
    }
    best.map_or(0, |(_, sh)| sh)
}

/// Splice the accepted candidates of one log of one file.
///
/// `z` is the file's cleaned depth (strictly ascending, ft); every accepted
/// candidate holds one value per row. Rejected candidates are ignored.
pub(crate) fn splice_log(
    log: LogKind,
    z: &[f64],
    cands: &[Candidate],
    max_gap_ft: f64,
) -> SpliceOutcome {
    let bounds = physical_bounds(log);
    let n = z.len();
    let mut no_valid = Vec::new();
    let mut preps: Vec<Prep<'_>> = Vec::new();
    for (pos, c) in cands.iter().enumerate() {
        let Ok(acc) = &c.result else { continue };
        let (vz, _) = valid_points(z, &acc.values, bounds);
        let (Some(&first), Some(&last)) = (vz.first(), vz.last()) else {
            no_valid.push(pos);
            continue;
        };
        preps.push(Prep {
            pos,
            col: c.col,
            priority: c.priority,
            mnemonic: &c.mnemonic,
            values: &acc.values,
            span: last - first,
            frac: vz.len() as f64 / n as f64,
            family: curve_family(log, &c.mnemonic),
        });
    }
    preps.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then_with(|| b.span.partial_cmp(&a.span).unwrap_or(Ordering::Equal))
            .then_with(|| b.frac.partial_cmp(&a.frac).unwrap_or(Ordering::Equal))
            .then_with(|| a.col.cmp(&b.col))
    });
    let mut iter = preps.into_iter();
    let Some(primary) = iter.next() else {
        return SpliceOutcome {
            spliced: None,
            no_valid,
        };
    };
    let mut out: Vec<f32> = primary
        .values
        .iter()
        .map(|&v| if is_valid(v, bounds) { v } else { f32::NAN })
        .collect();
    let coverage_before_frac = out.iter().filter(|v| v.is_finite()).count() as f64 / n as f64;
    // Secondaries of the primary's family first, otherwise in rank order.
    let rest: Vec<Prep<'_>> = iter.collect();
    let (same, other): (Vec<_>, Vec<_>) =
        rest.into_iter().partition(|p| p.family == primary.family);
    let mut merged_from = Vec::new();
    let mut shifts_ft = Vec::new();
    let mut skipped = Vec::new();
    for s in same.into_iter().chain(other) {
        let shift = best_shift(z, &out, s.values, bounds, max_gap_ft);
        let (s_z, s_v) = valid_points(z, s.values, bounds);
        let sx: Vec<f64> = s_z.iter().map(|&v| v + f64::from(shift)).collect();
        let g = interp_gap(z, &sx, &s_v, max_gap_ft);
        let ov: Vec<usize> = (0..n)
            .filter(|&i| {
                out.get(i).is_some_and(|v| v.is_finite())
                    && g.get(i).is_some_and(|&v| is_valid64(v, bounds))
            })
            .collect();
        let out_ov: Vec<f64> = ov
            .iter()
            .map(|&i| out.get(i).map_or(f64::NAN, |&v| f64::from(v)))
            .collect();
        let g_ov: Vec<f64> = ov
            .iter()
            .map(|&i| g.get(i).copied().unwrap_or(f64::NAN))
            .collect();
        let span_ov = match (ov.first(), ov.last()) {
            (Some(&a), Some(&b)) => {
                z.get(b).copied().unwrap_or(f64::NAN) - z.get(a).copied().unwrap_or(f64::NAN)
            }
            _ => 0.0,
        };
        let mut calibrated = false;
        let mut c = g.clone();
        if ov.len() >= MIN_CALIB_ROWS && span_ov >= MIN_OVERLAP_FT {
            let ps = [CALIB_PCT.0, CALIB_PCT.1];
            let p = percentiles(&out_ov, &ps).unwrap_or_default();
            let q = percentiles(&g_ov, &ps).unwrap_or_default();
            if let ([p_lo, p_hi], [s_lo, s_hi]) = (p.as_slice(), q.as_slice()) {
                if (s_hi - s_lo).abs() > TINY {
                    let k = (p_hi - p_lo) / (s_hi - s_lo);
                    c = g.iter().map(|&v| p_lo + (v - s_lo) * k).collect();
                    calibrated = true;
                }
            }
        }
        if calibrated {
            let resid: Vec<f64> = ov
                .iter()
                .map(|&i| {
                    let o = out.get(i).map_or(f64::NAN, |&v| f64::from(v));
                    (o - c.get(i).copied().unwrap_or(f64::NAN)).abs()
                })
                .collect();
            let resid = pct50(&resid);
            let centre = pct50(&out_ov);
            let dev: Vec<f64> = out_ov.iter().map(|v| (v - centre).abs()).collect();
            let scale = pct50(&dev);
            if scale > TINY && resid > MAX_RESID_RATIO * scale {
                skipped.push(format!("{}:resid", s.mnemonic));
                continue;
            }
        } else if s.family != primary.family {
            skipped.push(format!("{}:cross_family", s.mnemonic));
            continue;
        }
        for (o, &cv) in out.iter_mut().zip(&c) {
            if o.is_nan() && is_valid64(cv, bounds) {
                *o = cv as f32;
            }
        }
        merged_from.push(s.mnemonic.to_string());
        shifts_ft.push(shift);
    }
    let coverage_after_frac = out.iter().filter(|v| v.is_finite()).count() as f64 / n as f64;
    SpliceOutcome {
        spliced: Some(Spliced {
            out,
            primary: primary.pos,
            merged_from,
            shifts_ft,
            skipped,
            coverage_before_frac,
            coverage_after_frac,
        }),
        no_valid,
    }
}
