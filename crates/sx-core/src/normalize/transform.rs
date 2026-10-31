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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::reference::{idw_pair, RefSource, Voter};
    use crate::qc::native_bounds;
    use crate::types::{LogKind, WellId};

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs {b}");
    }

    /// GR worked example: W = (21, 111), gain 1.5, offset −24.
    #[test]
    fn gr_worked_example() {
        let cfg = NormalizeConfig::default();
        let voters = [
            Voter {
                well: WellId(1),
                xy: (1000.0, 0.0),
                low: 20.0,
                high: 110.0,
            },
            Voter {
                well: WellId(2),
                xy: (0.0, 2000.0),
                low: 20.0,
                high: 110.0,
            },
            Voter {
                well: WellId(3),
                xy: (4000.0, 0.0),
                low: 41.0,
                high: 131.0,
            },
        ];
        let w = idw_pair(WellId(0), Some((0.0, 0.0)), &voters, (25.0, 105.0), &cfg);
        assert_eq!(w.source, RefSource::Whole);
        assert_eq!(w.n_voters, 3);
        assert!(!w.reverted);
        close(w.low, 21.0);
        close(w.high, 111.0);

        let go = gain_offset(
            30.0,
            90.0,
            w.low,
            w.high,
            cfg.gain_cap,
            cfg.shift_only_frac,
            cfg.degenerate_span,
        );
        assert!(!go.shift_only);
        close(go.gain, 1.5);
        close(go.offset, -24.0);

        let (lo, hi) = native_bounds(LogKind::Gr);
        let mut v = [10.0f32, 30.0, 90.0, 150.0, 360.0];
        let mut b = [0u8; 5];
        assert_eq!(transform_channel(&mut v, &mut b, &go, lo, hi).unwrap(), 2);
        assert!(v[0].is_nan() && v[4].is_nan());
        for (got, want) in v[1..4].iter().zip([21.0f32, 111.0, 201.0]) {
            assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
        }
        assert_eq!(b, [QC_OUT_OF_BOUNDS, 0, 0, 0, QC_OUT_OF_BOUNDS]);
    }

    /// narrow well (40, 60) against W = (21, 111): need 4.5, gain
    /// capped at 2; with jury `ok` and corr >= 0.3 the cap is 4.
    #[test]
    fn narrow_well_gain_cap_and_extended_cap() {
        let cfg = NormalizeConfig::default();
        let (r_lo, r_hi) = (21.0, 111.0);
        let need = (r_hi - r_lo) / (60.0 - 40.0);
        close(need, 4.5);
        let apply = |cap: f64| {
            let go = gain_offset(
                40.0,
                60.0,
                r_lo,
                r_hi,
                cap,
                cfg.shift_only_frac,
                cfg.degenerate_span,
            );
            let mut v = [40.0f32, 60.0];
            let mut b = [0u8; 2];
            transform_channel(&mut v, &mut b, &go, 0.0, 500.0).unwrap();
            (go, v)
        };

        let (cap, ext) = stretch_cap(need, JuryVerdict::NoVerdict, f64::NAN, &cfg);
        assert_eq!((cap, ext), (2.0, false));
        assert_eq!(stretch_cap(need, JuryVerdict::Ok, 0.29, &cfg), (2.0, false));
        let (go, v) = apply(cap);
        close(go.gain, 2.0);
        close(go.offset, -34.0);
        assert_eq!(v, [46.0, 86.0]);

        let (cap, ext) = stretch_cap(need, JuryVerdict::Ok, 0.3, &cfg);
        assert_eq!((cap, ext), (4.0, true));
        let (go, v) = apply(cap);
        close(go.gain, 4.0);
        close(go.offset, -134.0);
        assert_eq!(v, [26.0, 106.0]);
    }
}
