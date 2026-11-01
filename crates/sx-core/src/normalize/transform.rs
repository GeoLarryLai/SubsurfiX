// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Two-point transform of stage 4: gain and offset from an anchor
//! pair and a reference pair, the stretch cap, and the application to one
//! channel. Values are native (the log's own unit).

use crate::config::NormalizeConfig;
use crate::error::{SxError, SxResult};
use crate::qc::{JuryVerdict, QC_OUT_OF_BOUNDS};

/// Gain and offset of one channel: `y = gain · x + offset` (native units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainOffset {
    /// Gain (unitless).
    pub gain: f64,
    /// Offset (native units).
    pub offset: f64,
    /// The anchor span was degenerate or under `shift_only_frac` of the
    /// reference span, so the gain is exactly 1.
    pub shift_only: bool,
    /// Gain cap used (×).
    pub cap: f64,
}

impl GainOffset {
    /// Identity transform (`gain = 1`, `offset = 0`) carrying `cap`.
    pub(crate) fn identity(cap: f64) -> Self {
        Self {
            gain: 1.0,
            offset: 0.0,
            shift_only: false,
            cap,
        }
    }
}

/// Gain and offset that map the anchor pair `(p_low, p_high)` onto the
/// reference pair `(ref_low, ref_high)`.
///
/// `den = p_high − p_low`, `rs = ref_high − ref_low`. When `|den| <=
/// degenerate_span` or `|den| < shift_only_frac · |rs|` the gain is 1
/// (`shift_only`); else `g = rs / den` and `gain = sign(g) · clamp(|g|,
/// 1/cap, cap)` with `sign(0) = +1`, so `|gain|` always lies in `[1/cap,
/// cap]` (`cap >= 1`; a `cap < 1` is read as `1/cap`). `offset = (ref_low +
/// ref_high)/2 − gain · (p_low + p_high)/2`.
///
/// Units: native. NaN rule: propagate (any non-finite input gives NaN gain
/// and offset). Errors: none.
pub fn gain_offset(
    p_low: f64,
    p_high: f64,
    ref_low: f64,
    ref_high: f64,
    cap: f64,
    shift_only_frac: f64,
    degenerate_span: f64,
) -> GainOffset {
    if ![p_low, p_high, ref_low, ref_high, cap]
        .iter()
        .all(|v| v.is_finite())
    {
        return GainOffset {
            gain: f64::NAN,
            offset: f64::NAN,
            shift_only: false,
            cap,
        };
    }
    let den = p_high - p_low;
    let rs = ref_high - ref_low;
    let shift_only = den.abs() <= degenerate_span || den.abs() < shift_only_frac * rs.abs();
    let gain = if shift_only {
        1.0
    } else {
        let g = rs / den;
        let (lo, hi) = (cap.min(1.0 / cap), cap.max(1.0 / cap));
        let sign = if g < 0.0 { -1.0 } else { 1.0 };
        sign * g.abs().max(lo).min(hi)
    };
    let offset = (ref_low + ref_high) / 2.0 - gain * ((p_low + p_high) / 2.0);
    GainOffset {
        gain,
        offset,
        shift_only,
        cap,
    }
}

/// Apply `y = gain · x + offset` (f64) to every finite sample.
///
/// A result inside `[lo, hi]` (native bounds) replaces the sample; any other
/// result (including NaN) blanks it with `QC_OUT_OF_BOUNDS`. Values are never
/// clipped. Returns the number blanked. NaN rule: skip (NaN samples are
/// untouched and get no bit). O(n).
///
/// # Errors
/// `SxError::Shape` if `values` and `bits` differ in length.
pub fn transform_channel(
    values: &mut [f32],
    bits: &mut [u8],
    go: &GainOffset,
    lo: f64,
    hi: f64,
) -> SxResult<usize> {
    if values.len() != bits.len() {
        return Err(SxError::Shape(format!(
            "transform_channel: values {}, bits {}",
            values.len(),
            bits.len()
        )));
    }
    let mut blanked = 0usize;
    for (v, b) in values.iter_mut().zip(bits.iter_mut()) {
        if !v.is_finite() {
            continue;
        }
        let y = go.gain * f64::from(*v) + go.offset;
        if (lo..=hi).contains(&y) {
            *v = y as f32;
        } else {
            *v = f32::NAN;
            *b |= QC_OUT_OF_BOUNDS;
            blanked += 1;
        }
    }
    Ok(blanked)
}

/// Gain cap of one channel (4.9): `(gain_cap_extended, true)` when `need` is
/// finite and outside `[1/gain_cap, gain_cap]`, the jury verdict is `Ok` and
/// `corr >= stretch_min_corr`; else `(gain_cap, false)`. NaN `corr` never
/// extends.
pub(crate) fn stretch_cap(
    need: f64,
    verdict: JuryVerdict,
    corr: f64,
    cfg: &NormalizeConfig,
) -> (f64, bool) {
    let out = need.is_finite() && (need > cfg.gain_cap || need < 1.0 / cfg.gain_cap);
    if out && verdict == JuryVerdict::Ok && corr >= cfg.stretch_min_corr {
        (cfg.gain_cap_extended, true)
    } else {
        (cfg.gain_cap, false)
    }
}
