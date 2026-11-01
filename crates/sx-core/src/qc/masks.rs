// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Sample masks: bounds, constant channel, tails, exact repeats, soft
//! plateaus, short runs, pileups, and the functions that apply them.
//!
//! Every mask function is pure. NaN rule for all of them: skip (a NaN sample
//! is never masked and a NaN breaks runs). The returned `Vec<bool>` always has
//! the length of the input; `true` marks a sample to blank.

use std::collections::VecDeque;

use super::stats::mean_std;
use super::QC_CHANNEL;
use crate::error::{SxError, SxResult};
use crate::util::{median, percentiles};

/// Out-of-bounds mask (bit `QC_OUT_OF_BOUNDS`): `true` where a finite value
/// has `v < lo` or `v > hi`.
///
/// Unit: any, same as `lo`/`hi`. NaN rule: skip. Infinite values are not
/// finite and are never masked. O(n).
pub fn bounds_mask(values: &[f64], lo: f64, hi: f64) -> Vec<bool> {
    values
        .iter()
        .map(|&v| v.is_finite() && (v < lo || v > hi))
        .collect()
}

/// True when the channel is constant (linear space).
///
/// Needs at least `min_count` finite samples (and at least one), else `false`.
/// True when `max - min <= rel_tol * (|median| + 1)`. NaN rule: skip. O(n log n).
pub fn constant_channel(values: &[f64], min_count: usize, rel_tol: f64) -> bool {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || finite.len() < min_count {
        return false;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in &finite {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    hi - lo <= rel_tol * (median(&finite).abs() + 1.0)
}

/// Tail mask (bit `QC_TAIL`), native space.
///
/// Needs `min_count` finite samples, else no mask. `p_lo`, `p_hi` are the
/// percentiles at `pct`; `pad = pad_frac * (p_hi - p_lo)`. Masks finite
/// `v < p_lo - pad` or `v > p_hi + pad`. `pad_frac = 0` is a plain percentile
/// clip. A percentile pair outside `[0, 100]` gives no mask. NaN rule: skip.
/// O(n log n).
pub fn tail_mask(values: &[f64], pct: [f64; 2], pad_frac: f64, min_count: usize) -> Vec<bool> {
    let none = vec![false; values.len()];
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || finite.len() < min_count {
        return none;
    }
    let ps = match percentiles(&finite, &pct) {
        Ok(p) => p,
        Err(_) => return none,
    };
    let (p_lo, p_hi) = match ps.as_slice() {
        [a, b] => (*a, *b),
        _ => return none,
    };
    let pad = pad_frac * (p_hi - p_lo);
    let lo = p_lo - pad;
    let hi = p_hi + pad;
    values
        .iter()
        .map(|&v| v.is_finite() && (v < lo || v > hi))
        .collect()
}

/// Exact-repeat mask (bit `QC_REPEAT`), linear space, MD order.
///
/// Needs `min_count` finite samples, else no mask.
/// `tol = rel_tol * (|median(finite)| + 1)`. For each finite position `r`
/// let `l(r)` be the smallest index such that every sample in `l(r)..=r` is
/// finite and `max - min` over it is `<= tol`. Mask `l(r)..=r` whenever
/// `r - l(r) + 1 >= min_len`; the mask is the union. A NaN resets the window.
/// O(n) with two monotone deques.
pub fn repeat_mask(values: &[f64], min_len: usize, rel_tol: f64, min_count: usize) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![false; n];
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || finite.len() < min_count {
        return mask;
    }
    let tol = rel_tol * (median(&finite).abs() + 1.0);
    let min_len = min_len.max(1);
    let mut max_q: VecDeque<usize> = VecDeque::new();
    let mut min_q: VecDeque<usize> = VecDeque::new();
    let mut l = 0usize;
    let mut marked = 0usize; // everything below `marked` is already decided
    for (r, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            max_q.clear();
            min_q.clear();
            l = r + 1;
            continue;
        }
        while let Some(&b) = max_q.back() {
            if values.get(b).is_some_and(|&x| x <= v) {
                max_q.pop_back();
            } else {
                break;
            }
        }
        max_q.push_back(r);
        while let Some(&b) = min_q.back() {
            if values.get(b).is_some_and(|&x| x >= v) {
                min_q.pop_back();
            } else {
                break;
            }
        }
        min_q.push_back(r);
        while l < r {
            let hi = max_q.front().and_then(|&i| values.get(i)).copied();
            let lo = min_q.front().and_then(|&i| values.get(i)).copied();
            match (hi, lo) {
                (Some(h), Some(w)) if h - w > tol => {
                    l += 1;
                    while max_q.front().is_some_and(|&i| i < l) {
                        max_q.pop_front();
                    }
                    while min_q.front().is_some_and(|&i| i < l) {
                        min_q.pop_front();
                    }
                }
                _ => break,
            }
        }
        if r + 1 - l >= min_len {
            let from = l.max(marked);
            for m in mask.iter_mut().take(r + 1).skip(from) {
                *m = true;
            }
            marked = r + 1;
        }
    }
    mask
}

/// Parameters of [`plateau_mask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlateauParams {
    /// Rolling-std window (samples).
    pub window: usize,
    /// Finite values a window needs to give a std (samples).
    pub window_min: usize,
    /// Shortest quiet run that is masked (samples).
    pub min_len: usize,
    /// Finite samples needed before plateaus are tested (samples).
    pub min_count: usize,
    /// Rolling std below this share of the channel std is quiet.
    pub std_share: f64,
}

/// Soft-plateau mask (bit `QC_PLATEAU`), native space, MD order.
///
/// Gate: finite count `>= max(min_count, 3 * window)`, else no mask; the
/// channel std `s` (ddof 1) must be finite and `> 0`. Threshold
/// `t = std_share * s`. The rolling std at `i` covers samples
/// `i - W/2 .. i - W/2 + W - 1` (integer division, clipped to the channel);
/// with `m` finite values inside it is the ddof-1 std if `m >= window_min`
/// (and `m >= 2`), else NaN. `quiet[i]` = sample finite and rolling std
/// finite and `< t`. Every run of quiet positions of length `>= min_len` is
/// masked. NaN rule: skip. O(n * window).
pub fn plateau_mask(values: &[f64], p: &PlateauParams) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![false; n];
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let gate = p.min_count.max(p.window.saturating_mul(3));
    if finite.is_empty() || finite.len() < gate || p.window == 0 {
        return mask;
    }
    let (_, _, s) = mean_std(&finite);
    if !s.is_finite() || s <= 0.0 {
        return mask;
    }
    let t = p.std_share * s;
    let half = p.window / 2;
    let mut quiet = vec![false; n];
    for (i, q) in quiet.iter_mut().enumerate() {
        if !values.get(i).is_some_and(|v| v.is_finite()) {
            continue;
        }
        let start = i.saturating_sub(half);
        let end = (i + p.window).saturating_sub(half).min(n); // exclusive
        let win: Vec<f64> = values
            .get(start..end)
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if win.len() >= p.window_min.max(2) {
            let (_, _, sd) = mean_std(&win);
            *q = sd.is_finite() && sd < t;
        }
    }
    mark_runs(&quiet, p.min_len.max(1), &mut mask);
    mask
}

/// Set `mask` over every run of `true` in `flags` that is at least `min_len` long.
fn mark_runs(flags: &[bool], min_len: usize, mask: &mut [bool]) {
    let n = flags.len();
    let mut i = 0usize;
    while i < n {
        if flags.get(i).copied().unwrap_or(false) {
            let mut e = i;
            while flags.get(e).copied().unwrap_or(false) {
                e += 1;
            }
            if e - i >= min_len {
                for m in mask.iter_mut().take(e).skip(i) {
                    *m = true;
                }
            }
            i = e;
        } else {
            i += 1;
        }
    }
}

/// Short-run mask (bit `QC_SHORT_RUN`).
///
/// A run is a maximal block of consecutive finite samples. The longest run
/// (first one on ties) is never masked; every other run shorter than
/// `min_len` is. NaN rule: skip (NaN breaks runs). O(n).
pub fn short_run_mask(values: &[f64], min_len: usize) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![false; n];
    let mut runs: Vec<(usize, usize)> = Vec::new(); // (start, end exclusive)
    let mut i = 0usize;
    while i < n {
        if values.get(i).is_some_and(|v| v.is_finite()) {
            let mut e = i;
            while values.get(e).is_some_and(|v| v.is_finite()) {
                e += 1;
            }
            runs.push((i, e));
            i = e;
        } else {
            i += 1;
        }
    }
    let mut longest: Option<usize> = None; // index into runs
    let mut best = 0usize;
    for (idx, &(s, e)) in runs.iter().enumerate() {
        if e - s > best {
            best = e - s;
            longest = Some(idx);
        }
    }
    for (idx, &(s, e)) in runs.iter().enumerate() {
        if Some(idx) != longest && e - s < min_len {
            for m in mask.iter_mut().take(e).skip(s) {
                *m = true;
            }
        }
    }
    mask
}

/// Saturation / clipping pileup mask (bit `QC_PILEUP`), native space, MD order.
///
/// No mask if fewer than `min_len` finite samples. `tol = rel_tol *
/// max(max|v|, 1)`. Greedy scan: `i = 0`; while `i < n`: a NaN advances `i`;
/// otherwise extend `e` from `i + 1` while `v[e]` is finite and
/// `|v[e] - v[i]| <= tol`. If `e - i >= min_len` and the share of finite
/// samples of the channel with `|v - v[i]| <= tol` is `>= min_share`, mask
/// `i..e` and set `i = e`; else `i += 1`. NaN rule: skip. O(n log n + n * L)
/// where `L` is the longest candidate run below the share limit.
pub fn pileup_mask(values: &[f64], min_len: usize, min_share: f64, rel_tol: f64) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![false; n];
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() || sorted.len() < min_len {
        return mask;
    }
    sorted.sort_by(f64::total_cmp);
    let max_abs = sorted.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    let tol = rel_tol * max_abs;
    let total = sorted.len() as f64;
    let mut i = 0usize;
    while i < n {
        let vi = match values.get(i) {
            Some(&v) if v.is_finite() => v,
            _ => {
                i += 1;
                continue;
            }
        };
        let mut e = i + 1;
        while values
            .get(e)
            .is_some_and(|&v| v.is_finite() && (v - vi).abs() <= tol)
        {
            e += 1;
        }
        if e - i >= min_len {
            let lo = sorted.partition_point(|&x| vi - x > tol);
            let hi = sorted.partition_point(|&x| x - vi <= tol);
            let share = hi.saturating_sub(lo) as f64 / total;
            if share >= min_share {
                for m in mask.iter_mut().take(e).skip(i) {
                    *m = true;
                }
                i = e;
                continue;
            }
        }
        i += 1;
    }
    mask
}

/// Blank masked finite samples and record `bit`.
///
/// Sets every finite sample with `mask == true` to NaN and ORs `bit` into
/// `bits` there. Returns the number blanked. NaN rule: skip (NaN samples are
/// untouched and get no bit).
///
/// # Errors
/// `SxError::Shape` if the three lengths differ.
pub fn apply_mask(values: &mut [f32], bits: &mut [u8], mask: &[bool], bit: u8) -> SxResult<usize> {
    if values.len() != bits.len() || values.len() != mask.len() {
        return Err(SxError::Shape(format!(
            "apply_mask: values {}, bits {}, mask {}",
            values.len(),
            bits.len(),
            mask.len()
        )));
    }
    let mut count = 0usize;
    for ((v, b), &m) in values.iter_mut().zip(bits.iter_mut()).zip(mask) {
        if m && v.is_finite() {
            *v = f32::NAN;
            *b |= bit;
            count += 1;
        }
    }
    Ok(count)
}

/// Bad-hole thresholds (see `[qc]` `cal_*` and `drho_limit`).
#[derive(Debug, Clone, Copy)]
pub struct BadHoleParams {
    /// Caliper below this is closed / cased hole (in).
    pub open_min_in: f64,
    /// Undergauge margin below the nominal hole size (in).
    pub undergauge_in: f64,
    /// Washout margin above the nominal hole size (in).
    pub washout_in: f64,
    /// |density correction| limit (g/cm³).
    pub drho_limit: f64,
}

/// Bad-hole mask from a caliper (in) and a density-correction curve (g/cm³).
///
/// Nominal hole size = median caliper over open samples (>= `open_min_in`).
/// [`badhole_eval`] uses the bit size instead where the file has one.
/// A sample is bad when the caliper is closed, undergauge or washed out, or
/// when |drho| exceeds the limit. Guards: a caliper with fewer than 10 % open
/// samples is ignored (unit trouble, not a closed tool everywhere); a nominal
/// above 30 is read as millimetres; a drho curve whose median |value| is above
/// half the limit does not look like a correction and is ignored.
/// NaN rule: a NaN aux sample never flags.
pub fn badhole_mask(cal_in: &[f32], drho: &[f32], p: BadHoleParams) -> Vec<bool> {
    badhole_eval(cal_in, drho, &[], p).0
}

/// [`badhole_mask`] plus a per-sample "hole verified" mask: true where a usable
/// caliper or density-correction sample exists (so a good-hole verdict is backed
/// by a measurement, not by a missing curve). `bs_in` is the bit size (in; empty
/// or NaN = none): where a sample has an open bit size (>= `open_min_in`; a
/// median above 30 is read as millimetres) it is the nominal hole size for that
/// sample, else the median caliper is.
pub fn badhole_eval(
    cal_in: &[f32],
    drho: &[f32],
    bs_in: &[f32],
    p: BadHoleParams,
) -> (Vec<bool>, Vec<bool>) {
    let n = cal_in.len().max(drho.len());
    let mut mask = vec![false; n];
    let mut verified = vec![false; n];
    let cal: Vec<f64> = cal_in.iter().map(|&v| f64::from(v)).collect();
    let finite: Vec<f64> = cal.iter().copied().filter(|v| v.is_finite()).collect();
    if !finite.is_empty() {
        let mut scale = 1.0;
        let mut open: Vec<f64> = finite
            .iter()
            .copied()
            .filter(|&v| v >= p.open_min_in)
            .collect();
        let mut nominal = median(&open);
        if nominal > 30.0 {
            scale = 1.0 / 25.4;
            let o: Vec<f64> = finite
                .iter()
                .map(|v| v * scale)
                .filter(|&v| v >= p.open_min_in)
                .collect();
            nominal = median(&o);
            open = o;
        }
        let bs: Vec<f64> = bs_in.iter().map(|&v| f64::from(v)).collect();
        let bs_fin: Vec<f64> = bs.iter().copied().filter(|v| v.is_finite()).collect();
        let bs_scale = if median(&bs_fin) > 30.0 {
            1.0 / 25.4
        } else {
            1.0
        };
        if open.len() * 10 >= finite.len() && nominal.is_finite() {
            for (i, ((m, v), &c)) in mask
                .iter_mut()
                .zip(verified.iter_mut())
                .zip(&cal)
                .enumerate()
            {
                let c = c * scale;
                let b = bs.get(i).map_or(f64::NAN, |b| b * bs_scale);
                let nom = if b.is_finite() && b >= p.open_min_in {
                    b
                } else {
                    nominal
                };
                *v |= c.is_finite();
                if c.is_finite()
                    && (c < p.open_min_in || c < nom - p.undergauge_in || c > nom + p.washout_in)
                {
                    *m = true;
                }
            }
        }
    }
    let abs: Vec<f64> = drho
        .iter()
        .map(|&v| f64::from(v).abs())
        .filter(|v| v.is_finite())
        .collect();
    if median(&abs) <= 0.5 * p.drho_limit {
        for ((m, v), &d) in mask.iter_mut().zip(verified.iter_mut()).zip(drho) {
            *v |= d.is_finite();
            if f64::from(d).abs() > p.drho_limit {
                *m = true;
            }
        }
    }
    (mask, verified)
}

/// Mask of finite samples outside `[lo, hi]`.
pub fn range_mask(values: &[f32], lo: f64, hi: f64) -> Vec<bool> {
    values
        .iter()
        .map(|&v| v.is_finite() && (f64::from(v) < lo || f64::from(v) > hi))
        .collect()
}

/// Blank every finite sample with `QC_CHANNEL`. Returns the number blanked.
///
/// NaN rule: skip.
///
/// # Errors
/// `SxError::Shape` if `values` and `bits` differ in length.
pub fn blank_channel(values: &mut [f32], bits: &mut [u8]) -> SxResult<usize> {
    let mask = vec![true; values.len()];
    apply_mask(values, bits, &mask, QC_CHANNEL)
}

/// Closed-caliper mask (true = caliper reads below `open_min_in`), with the
/// same millimetre detection and "caliper mostly open" guard as
/// [`badhole_eval`]: a caliper that is open in fewer than 10% of its finite
/// samples is ignored (all false).
#[must_use]
pub fn closed_caliper_mask(cal_in: &[f32], open_min_in: f64) -> Vec<bool> {
    let cal: Vec<f64> = cal_in.iter().map(|&v| f64::from(v)).collect();
    let finite: Vec<f64> = cal.iter().copied().filter(|v| v.is_finite()).collect();
    let mut scale = 1.0;
    let open: Vec<f64> = finite
        .iter()
        .copied()
        .filter(|&v| v >= open_min_in)
        .collect();
    let mut n_open = open.len();
    if median(&open) > 30.0 {
        scale = 1.0 / 25.4;
        n_open = finite.iter().filter(|&&v| v * scale >= open_min_in).count();
    }
    if finite.is_empty() || n_open * 10 < finite.len() {
        return vec![false; cal.len()];
    }
    cal.iter()
        .map(|&c| c.is_finite() && c * scale < open_min_in)
        .collect()
}
