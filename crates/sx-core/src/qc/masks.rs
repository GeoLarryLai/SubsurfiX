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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qc::{QC_OUT_OF_BOUNDS, QC_TAIL};

    fn idx(mask: &[bool]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn badhole_rules() {
        let p = BadHoleParams {
            open_min_in: 5.0,
            undergauge_in: 1.0,
            washout_in: 2.0,
            drho_limit: 0.15,
        };
        let cal = [8.5f32, 8.6, 8.4, 2.0, 11.5, 7.0, 8.5, f32::NAN, 8.5, 8.5];
        let drho = [0.01f32, 0.02, 0.3, 0.0, 0.0, 0.0, -0.2, 0.0, f32::NAN, 0.0];
        let m = badhole_mask(&cal, &drho, p);
        assert_eq!(idx(&m), vec![2, 3, 4, 5, 6]);
        // Millimetre caliper.
        let mm: Vec<f32> = cal.iter().map(|v| v * 25.4).collect();
        assert_eq!(idx(&badhole_mask(&mm, &[], p)), vec![3, 4, 5]);
        // Caliper almost never open: ignored.
        assert!(badhole_mask(&[0.3f32; 20], &[], p).iter().all(|&b| !b));
        // drho that is not a correction: ignored.
        assert!(badhole_mask(&[], &[2.4f32; 5], p).iter().all(|&b| !b));
        // Two hole sections: the median caliper calls the 8.5 in section undergauge,
        // the bit size does not; NaN bit size falls back to the median.
        let two: Vec<f32> = [[12.2f32; 6], [8.5; 6]].concat();
        assert_eq!(
            idx(&badhole_mask(&two, &[], p)),
            (6..12).collect::<Vec<_>>()
        );
        let bs: Vec<f32> = [[12.25f32; 6], [8.5; 6]].concat();
        let (m, v) = badhole_eval(&two, &[], &bs, p);
        assert!(m.iter().all(|&b| !b) && v.iter().all(|&b| b));
        let mut bs_part = bs.clone();
        bs_part[11] = f32::NAN;
        assert_eq!(idx(&badhole_eval(&two, &[], &bs_part, p).0), vec![11]);
        let bs_mm: Vec<f32> = bs.iter().map(|v| v * 25.4).collect();
        assert!(badhole_eval(&two, &[], &bs_mm, p).0.iter().all(|&b| !b));
        assert_eq!(
            idx(&range_mask(&[1.2f32, 2.5, f32::NAN, 3.3], 1.65, 3.1)),
            vec![0, 3]
        );
    }

    #[test]
    fn bounds() {
        let v = [f64::NAN, -1.0, 0.0, 250.0, 500.0, 500.1];
        assert_eq!(
            bounds_mask(&v, 0.0, 500.0),
            vec![false, true, false, false, false, true]
        );
    }

    #[test]
    fn constant() {
        assert!(constant_channel(&[3.0; 10], 5, 1e-9));
        assert!(!constant_channel(&[3.0; 4], 5, 1e-9));
        assert!(!constant_channel(&[], 0, 1e-9));
        let mut v = vec![3.0; 10];
        v[4] = 3.001;
        assert!(!constant_channel(&v, 5, 1e-9));
        // NaN samples are skipped
        let mut w = vec![f64::NAN; 3];
        w.extend([7.0; 6]);
        assert!(constant_channel(&w, 5, 1e-9));
    }

    /// Example 1: tail pad.
    #[test]
    fn tail_pad_example() {
        let mut v: Vec<f64> = (0..400).map(|i| 50.0 + 0.1 * f64::from(i)).collect();
        v.push(480.0);
        let m = tail_mask(&v, [0.25, 99.75], 0.10, 5);
        assert_eq!(idx(&m), vec![400]);
        // plain clip also drops 50.0
        let m0 = tail_mask(&v, [0.25, 99.75], 0.0, 5);
        assert!(m0[0]);
        assert!(m0[400]);
        // too few samples: no mask
        assert!(idx(&tail_mask(&[1.0, 2.0, 100.0], [0.25, 99.75], 0.0, 5)).is_empty());
    }

    /// Example 2: repeats inside a drift.
    #[test]
    fn repeat_drift_example() {
        let mut v = vec![49.80, 49.84, 49.88, 49.92, 49.96];
        v.extend(std::iter::repeat_n(50.00, 22));
        v.extend([50.04, 50.08, 50.12, 50.16]);
        assert_eq!(v.len(), 31);
        let m = repeat_mask(&v, 20, 1e-3, 5);
        assert_eq!(idx(&m), (4..=27).collect::<Vec<_>>());
    }

    #[test]
    fn repeat_lengths() {
        let mut v: Vec<f64> = (0..60).map(|i| 10.0 + f64::from(i) * 5.0).collect();
        for x in v.iter_mut().skip(10).take(20) {
            *x = 100.0;
        }
        let m = repeat_mask(&v, 20, 1e-3, 5);
        assert_eq!(idx(&m), (10..30).collect::<Vec<_>>());
        let mut w: Vec<f64> = (0..60).map(|i| 10.0 + f64::from(i) * 5.0).collect();
        for x in w.iter_mut().skip(10).take(19) {
            *x = 100.0;
        }
        assert!(idx(&repeat_mask(&w, 20, 1e-3, 5)).is_empty());
    }

    #[test]
    fn repeat_slow_ramp_not_masked() {
        // step 0.01, tol = 1e-3 * 51 = 0.051: a window of 6 stays within tol,
        // but the run never reaches 20 samples.
        let v: Vec<f64> = (0..200).map(|i| 50.0 + 0.01 * f64::from(i)).collect();
        assert!(idx(&repeat_mask(&v, 20, 1e-3, 5)).is_empty());
    }

    #[test]
    fn repeat_nan_breaks() {
        let mut v = vec![5.0; 30];
        v[15] = f64::NAN;
        assert!(idx(&repeat_mask(&v, 20, 1e-3, 5)).is_empty());
        let mut w = vec![5.0; 41];
        w[20] = f64::NAN;
        let m = repeat_mask(&w, 20, 1e-3, 5);
        assert_eq!(idx(&m), (0..20).chain(21..41).collect::<Vec<_>>());
    }

    fn pp() -> PlateauParams {
        PlateauParams {
            window: 20,
            window_min: 10,
            min_len: 20,
            min_count: 60,
            std_share: 0.02,
        }
    }

    fn noise(i: usize) -> f64 {
        // deterministic pseudo-noise with std ~10
        let x = ((i as u64).wrapping_mul(2654435761) % 1000) as f64 / 1000.0;
        (x - 0.5) * 34.64
    }

    #[test]
    fn plateau_quiet_stretch() {
        let mut v: Vec<f64> = (0..300).map(noise).collect();
        for (j, x) in v.iter_mut().enumerate().skip(100).take(40) {
            *x = 5.0 + 0.001 * (j % 2) as f64;
        }
        let m = plateau_mask(&v, &pp());
        let got = idx(&m);
        assert!(!got.is_empty());
        assert!(got.iter().all(|&i| (100..140).contains(&i)), "{got:?}");
        // the interior of the stretch is masked
        assert!((110..130).all(|i| m[i]));
    }

    #[test]
    fn plateau_noise_not_masked() {
        let v: Vec<f64> = (0..300).map(noise).collect();
        assert!(idx(&plateau_mask(&v, &pp())).is_empty());
    }

    #[test]
    fn plateau_gates() {
        // fewer than 60 finite samples
        let v: Vec<f64> = (0..59)
            .map(|i| if i < 30 { 1.0 } else { noise(i) })
            .collect();
        assert!(idx(&plateau_mask(&v, &pp())).is_empty());
        // constant channel: std 0 -> no mask
        assert!(idx(&plateau_mask(&[1.0; 100], &pp())).is_empty());
    }

    /// Example 3: short runs.
    #[test]
    fn short_runs_examples() {
        let nan = f64::NAN;
        let mut v = vec![1.0; 5];
        v.push(nan);
        v.extend(vec![2.0; 30]);
        v.push(nan);
        v.extend(vec![3.0; 12]);
        let m = short_run_mask(&v, 20);
        let want: Vec<usize> = (0..5).chain(37..49).collect();
        assert_eq!(idx(&m), want);

        let mut w = vec![1.0; 8];
        w.push(nan);
        w.extend(vec![2.0; 12]);
        w.push(nan);
        w.extend(vec![3.0; 12]);
        let m = short_run_mask(&w, 20);
        let want: Vec<usize> = (0..8).chain(22..34).collect();
        assert_eq!(idx(&m), want);
    }

    #[test]
    fn short_run_longest_survives() {
        let nan = f64::NAN;
        let v = [1.0, 1.0, 1.0, nan, 2.0, 2.0];
        let m = short_run_mask(&v, 20);
        assert_eq!(idx(&m), vec![4, 5]);
        assert!(idx(&short_run_mask(&[nan, nan], 20)).is_empty());
    }

    #[test]
    fn pileup_basic() {
        let mut v: Vec<f64> = (0..200).map(|i| 10.0 + f64::from(i) * 0.37).collect();
        for x in v.iter_mut().skip(50).take(30) {
            *x = 100.0;
        }
        let m = pileup_mask(&v, 20, 0.01, 1e-8);
        assert_eq!(idx(&m), (50..80).collect::<Vec<_>>());
        // run shorter than min_len
        let mut w: Vec<f64> = (0..200).map(|i| 10.0 + f64::from(i) * 0.37).collect();
        for x in w.iter_mut().skip(50).take(19) {
            *x = 100.0;
        }
        assert!(idx(&pileup_mask(&w, 20, 0.01, 1e-8)).is_empty());
        // too few finite samples
        assert!(idx(&pileup_mask(&[1.0; 10], 20, 0.01, 1e-8)).is_empty());
    }

    #[test]
    fn pileup_share_gate() {
        // 25-sample run in 5000 samples: share 0.005 < 0.01 -> not masked
        let mut v: Vec<f64> = (0..5000).map(|i| 10.0 + f64::from(i) * 0.01).collect();
        for x in v.iter_mut().skip(100).take(25) {
            *x = 3.0;
        }
        assert!(idx(&pileup_mask(&v, 20, 0.01, 1e-8)).is_empty());
        assert_eq!(idx(&pileup_mask(&v, 20, 0.004, 1e-8)).len(), 25);
    }

    #[test]
    fn apply_and_blank() {
        let mut v = [1.0f32, f32::NAN, 3.0, 4.0];
        let mut b = [0u8; 4];
        let n = apply_mask(&mut v, &mut b, &[true, true, false, true], QC_TAIL).unwrap();
        assert_eq!(n, 2);
        assert!(v[0].is_nan() && v[1].is_nan() && v[3].is_nan());
        assert_eq!(v[2], 3.0);
        assert_eq!(b, [QC_TAIL, 0, 0, QC_TAIL]);
        let n = blank_channel(&mut v, &mut b).unwrap();
        assert_eq!(n, 1);
        assert_eq!(b, [QC_TAIL, 0, QC_CHANNEL, QC_TAIL]);
        // OR-ing keeps earlier bits only on samples finite before: none here
        let mut v2 = [1.0f32];
        let mut b2 = [QC_OUT_OF_BOUNDS];
        apply_mask(&mut v2, &mut b2, &[true], QC_TAIL).unwrap();
        assert_eq!(b2, [QC_OUT_OF_BOUNDS | QC_TAIL]);
        assert!(matches!(
            apply_mask(&mut [1.0], &mut [0u8; 2], &[true], 1),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            blank_channel(&mut [1.0], &mut [0u8; 2]),
            Err(SxError::Shape(_))
        ));
    }
}
