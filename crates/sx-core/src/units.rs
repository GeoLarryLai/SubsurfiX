// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Canonical units, header-unit conversion, and plausibility ranges.
//!
//! `WellLogs` curves hold canonical linear units. Deep resistivity is stored
//! as ohm·m; callers convert with [`to_working`] when they do math.

use crate::error::{SxError, SxResult};
use crate::types::LogKind;

/// Meters per international foot (exact).
///
/// Unit: m/ft. This is a definition, not a sample, so there is no NaN rule.
/// Errors: none.
pub const M_PER_FT: f64 = 0.3048;

/// International feet per meter (`1 / 0.3048` = 3.280839895013123...).
///
/// Unit: ft/m. This is a definition, not a sample, so there is no NaN rule.
/// Errors: none.
pub const FT_PER_M: f64 = 1.0 / 0.3048;

/// Inclusive closed range `[lo, hi]`.
///
/// `lo` and `hi` use the caller's unit (canonical linear, or working units
/// for resistivity). Table ranges in this module are finite. NaN rule: see
/// [`Bounds::contains`]. Errors: none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Lower end, inclusive, in the range's unit. Not a sample and not NaN
    /// for ranges this module returns. Errors: none.
    pub lo: f64,
    /// Upper end, inclusive, in the range's unit. Not a sample and not NaN
    /// for ranges this module returns. Errors: none.
    pub hi: f64,
}

impl Bounds {
    /// True when `lo <= v <= hi`.
    ///
    /// Unit: the same unit as `lo` and `hi`. NaN rule: a NaN `v` is outside
    /// the range. Errors: none.
    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }
}

/// Result of looking up a curve's header unit string for one log.
///
/// Not a sample value. NaN rule: none of the variants carries a sample; a
/// [`HeaderUnit::Divisor`] is finite and `> 0`. Errors: none. Lookup reports
/// the class only; curve ingest decides what to do with blank or unrecognized
/// units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeaderUnit {
    /// The unit string is empty, or one of the "no unit" words (see
    /// [`header_unit`]).
    ///
    /// Not a sample. No NaN rule. Errors: none.
    Blank,
    /// Known unit. Canonical value = raw value / divisor.
    ///
    /// `divisor` is finite and `> 0`, in "header-unit per canonical unit"
    /// (raw = canonical * divisor). Not a sample, so it is never NaN.
    /// Errors: none.
    Divisor(f64),
    /// Non-blank string that is not in this log's table.
    ///
    /// Not a sample. No NaN rule. Errors: none.
    Unrecognized,
}

/// Header unit → divisor. Keys are already in [`normalize_unit`] form and are
/// unique within a log. DTC µs/m uses [`FT_PER_M`]: slowness per foot =
/// slowness per meter / feet-per-meter.
const HEADER_DIVISORS: &[(LogKind, &str, f64)] = &[
    (LogKind::Gr, "GAPI", 1.0),
    (LogKind::Gr, "API", 1.0),
    (LogKind::Gr, "APIGR", 1.0),
    (LogKind::Rhob, "G/CM3", 1.0),
    (LogKind::Rhob, "G/C3", 1.0),
    (LogKind::Rhob, "GM/CC", 1.0),
    (LogKind::Rhob, "G/CC", 1.0),
    (LogKind::Rhob, "GCC", 1.0),
    (LogKind::Rhob, "KG/M3", 1000.0),
    (LogKind::Nphi, "V/V", 1.0),
    (LogKind::Nphi, "M3/M3", 1.0),
    (LogKind::Nphi, "FT3/FT3", 1.0),
    (LogKind::Nphi, "VV", 1.0),
    (LogKind::Nphi, "FRAC", 1.0),
    (LogKind::Nphi, "FRACTION", 1.0),
    (LogKind::Nphi, "DEC", 1.0),
    (LogKind::Nphi, "PU", 100.0),
    (LogKind::Nphi, "%", 100.0),
    (LogKind::Nphi, "PERCENT", 100.0),
    (LogKind::Nphi, "PCT", 100.0),
    (LogKind::Nphi, "NAPI", 1000.0),
    (LogKind::Dtc, "US/FT", 1.0),
    (LogKind::Dtc, "USEC/FT", 1.0),
    (LogKind::Dtc, "USFT", 1.0),
    (LogKind::Dtc, "US/F", 1.0),
    (LogKind::Dtc, "US/M", FT_PER_M),
    (LogKind::Dtc, "USEC/M", FT_PER_M),
    (LogKind::Dtc, "USM", FT_PER_M),
    (LogKind::Resd, "OHMM", 1.0),
    (LogKind::Resd, "OHM", 1.0),
    (LogKind::Resd, "MOHMM", 1000.0),
    (LogKind::Pef, "B/E", 1.0),
    (LogKind::Pef, "BARNS/E", 1.0),
    (LogKind::Pef, "BARNS/ELECTRON", 1.0),
    (LogKind::Pef, "PE", 1.0),
];

/// Canonical unit text for the log (the unit of [`LogKind::column`]).
///
/// Unit: the text names the canonical linear unit of stored curve samples.
/// Resistivity is ohm·m, never log10. Not a sample. No NaN rule. Errors: none.
pub fn canonical_unit(log: LogKind) -> &'static str {
    match log {
        LogKind::Gr => "gAPI",
        LogKind::Rhob => "g/cm3",
        LogKind::Nphi => "v/v",
        LogKind::Dtc => "us/ft",
        LogKind::Resd => "ohm.m",
        LogKind::Pef => "b/e",
    }
}

/// Normalize a raw unit string to its lookup key.
///
/// Steps, in order:
/// 1. U+00B5 and U+03BC become `U`; U+03A9 and U+2126 become `OHM`; U+00B3
///    becomes `3`; U+00B7 is removed.
/// 2. ASCII-uppercase (`char::to_ascii_uppercase`). Other non-ASCII
///    characters stay as they are.
/// 3. Remove whitespace and `.` `-` `_` `(` `)` `[` `]`. `/` and `%` stay.
///
/// The result is a lookup key, not a sample. It may be empty. No NaN rule.
/// Errors: none.
pub fn normalize_unit(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            '\u{00B5}' | '\u{03BC}' => out.push('U'),
            '\u{03A9}' | '\u{2126}' => out.push_str("OHM"),
            '\u{00B3}' => out.push('3'),
            '\u{00B7}' => {}
            _ => {
                let c = c.to_ascii_uppercase();
                if c.is_whitespace() || matches!(c, '.' | '-' | '_' | '(' | ')' | '[' | ']') {
                    continue;
                }
                out.push(c);
            }
        }
    }
    out
}

/// Look up a raw header unit string for `log`.
///
/// `normalize_unit` first. An empty key, `NONE`, or `UNKNOWN` is
/// [`HeaderUnit::Blank`]. A key in this log's table is
/// [`HeaderUnit::Divisor`] (canonical = raw / divisor). Anything else is
/// [`HeaderUnit::Unrecognized`].
///
/// DTC in µs/m divides by [`FT_PER_M`] (about 3.28084 ft/m): one meter is
/// that many feet, so slowness per foot = slowness per meter / [`FT_PER_M`].
///
/// Unit of a divisor: header units per canonical unit. NaN rule: the input is
/// text; a returned divisor is finite and `> 0`. Errors: none. An empty
/// string is [`HeaderUnit::Blank`].
pub fn header_unit(log: LogKind, raw: &str) -> HeaderUnit {
    let key = normalize_unit(raw);
    if key.is_empty() || key == "NONE" || key == "UNKNOWN" {
        return HeaderUnit::Blank;
    }
    for &(row_log, row_key, divisor) in HEADER_DIVISORS {
        if row_log == log && row_key == key {
            return HeaderUnit::Divisor(divisor);
        }
    }
    HeaderUnit::Unrecognized
}

/// Divide every finite value in place by `divisor`, computing in f64:
/// `v = (f64::from(v) / divisor) as f32`.
///
/// Unit: `v` is an f32 curve sample in the header unit; the stored result is
/// that sample in canonical units when `divisor` comes from [`header_unit`].
/// NaN rule: skip. NaN and ±infinity are left unchanged.
/// Empty `values`: `Ok(())`, nothing to do.
/// Errors: [`SxError::InvalidInput`] if `divisor` is not finite or `<= 0`,
/// with the message `divisor must be finite and > 0, got {divisor}`.
pub fn apply_divisor(values: &mut [f32], divisor: f64) -> SxResult<()> {
    if !divisor.is_finite() || divisor <= 0.0 {
        return Err(SxError::InvalidInput(format!(
            "divisor must be finite and > 0, got {divisor}"
        )));
    }
    for v in values.iter_mut() {
        if v.is_finite() {
            *v = (f64::from(*v) / divisor) as f32;
        }
    }
    Ok(())
}

/// Physical bounds in canonical linear units.
///
/// Inclusive. Unit: [`canonical_unit`] of `log`. NaN rule: the bounds are
/// finite; NaN is outside ([`Bounds::contains`] is false). Errors: none.
pub fn physical_bounds(log: LogKind) -> Bounds {
    match log {
        LogKind::Gr => Bounds { lo: 0.0, hi: 500.0 },
        LogKind::Rhob => Bounds { lo: 1.0, hi: 3.5 },
        LogKind::Nphi => Bounds { lo: -0.15, hi: 1.0 },
        LogKind::Dtc => Bounds {
            lo: 30.0,
            hi: 250.0,
        },
        LogKind::Resd => Bounds {
            lo: 0.01,
            hi: 50000.0,
        },
        LogKind::Pef => Bounds { lo: 0.5, hi: 10.0 },
    }
}

/// Physical bounds in working units.
///
/// Same as [`physical_bounds`], except resistivity is
/// `[log10(0.01), log10(50000)]` = `[-2.0, 4.698970004336019]` (ohm·m in
/// `log10`). Other logs stay in canonical linear units. NaN rule: the bounds
/// are finite; NaN is outside. Errors: none.
pub fn working_bounds(log: LogKind) -> Bounds {
    let bounds = physical_bounds(log);
    if is_log10(log) {
        Bounds {
            lo: bounds.lo.log10(),
            hi: bounds.hi.log10(),
        }
    } else {
        bounds
    }
}

/// True only for resistivity: math on this log happens in `log10(ohm·m)`.
///
/// Other logs are false. Not a sample. No NaN rule. Errors: none.
pub fn is_log10(log: LogKind) -> bool {
    matches!(log, LogKind::Resd)
}

/// Canonical linear value → working value.
///
/// Resistivity: `v > 0` and finite → `v.log10()`; `v == +∞` → `+∞`; `v <= 0`
/// or NaN → NaN. Unit of the result: `log10(ohm·m)`.
/// Other logs: `v` unchanged (NaN propagates). Unit: canonical linear.
/// Errors: none.
pub fn to_working(log: LogKind, v: f64) -> f64 {
    if !is_log10(log) {
        return v;
    }
    if v.is_nan() || v <= 0.0 {
        f64::NAN
    } else if v.is_infinite() {
        f64::INFINITY
    } else {
        v.log10()
    }
}

/// Working value → canonical linear value.
///
/// Resistivity: `10^w` via `10f64.powf` (NaN → NaN; other non-finite values
/// follow `powf`). Unit of the result: ohm·m.
/// Other logs: `w` unchanged (NaN propagates). Unit: canonical linear.
/// Errors: none.
pub fn from_working(log: LogKind, w: f64) -> f64 {
    if is_log10(log) {
        10f64.powf(w)
    } else {
        w
    }
}

/// Plausible range of a curve's median in canonical linear units.
///
/// Inclusive. Unit: [`canonical_unit`] of `log`. NaN rule: the bounds are
/// finite; a NaN median is outside ([`Bounds::contains`] is false). Errors:
/// none.
///
/// [`PLAUSIBLE_MIN_SAMPLES`] and [`PLAUSIBLE_MIN_IN_BOUNDS_FRAC`] gate the
/// check. The check that uses them (median of all finite samples compared
/// with this range; at least 5 % of finite samples inside the physical bounds;
/// skipped below 30 finite samples) belongs to curve ingest.
pub fn plausible_median(log: LogKind) -> Bounds {
    match log {
        LogKind::Gr => Bounds { lo: 1.0, hi: 400.0 },
        LogKind::Rhob => Bounds { lo: 1.5, hi: 3.2 },
        LogKind::Nphi => Bounds {
            lo: -0.05,
            hi: 0.60,
        },
        LogKind::Dtc => Bounds {
            lo: 40.0,
            hi: 190.0,
        },
        LogKind::Resd => Bounds {
            lo: 0.05,
            hi: 5000.0,
        },
        LogKind::Pef => Bounds { lo: 1.0, hi: 8.0 },
    }
}

/// Minimum finite samples before the plausibility check is applied.
///
/// Unit: count of finite samples. Not a curve sample. No NaN rule. Errors:
/// none. The check itself belongs to curve ingest; see [`plausible_median`].
pub const PLAUSIBLE_MIN_SAMPLES: usize = 30;

/// Minimum share of finite samples that must lie inside the physical bounds.
///
/// Unit: fraction of finite samples (dimensionless). Not a curve sample. No
/// NaN rule. Errors: none. The check itself belongs to curve ingest; see
/// [`plausible_median`].
pub const PLAUSIBLE_MIN_IN_BOUNDS_FRAC: f64 = 0.05;
