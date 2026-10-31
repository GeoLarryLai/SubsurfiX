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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SxError;
    use crate::types::LogKind;

    #[test]
    fn normalize_examples() {
        let rows = [
            (" ohm-m ", "OHMM"),
            ("OHM M", "OHMM"),
            ("\u{03A9}\u{00B7}m", "OHMM"),
            ("P.U.", "PU"),
            ("\u{00B5}s/m", "US/M"),
            ("g/cm\u{00B3}", "G/CM3"),
            ("(gAPI)", "GAPI"),
            ("[us/ft]", "US/FT"),
            ("MOHM.M", "MOHMM"),
        ];
        assert_eq!(rows.len(), 9);
        for (raw, key) in rows {
            assert_eq!(normalize_unit(raw), key, "{raw:?}");
        }
        // The other code points named in the replacement step.
        assert_eq!(normalize_unit("\u{03BC}s/m"), "US/M");
        assert_eq!(normalize_unit("\u{2126}\u{00B7}m"), "OHMM");
        // Non-ASCII that step 1 does not list stays; ASCII is uppercased.
        assert_eq!(normalize_unit("µα/m"), "Uα/M");
        assert_eq!(normalize_unit(""), "");
        assert_eq!(normalize_unit("   .-_"), "");
    }

    #[test]
    fn every_table_key_round_trips() {
        let rows: &[(LogKind, &str, f64)] = &[
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
        assert_eq!(rows.len(), 35);
        for (i, &(log, key, divisor)) in rows.iter().enumerate() {
            assert_eq!(normalize_unit(key), key, "{key}");
            assert_eq!(header_unit(log, key), HeaderUnit::Divisor(divisor));
            for &(log2, key2, _) in rows.iter().skip(i + 1) {
                assert!(
                    log != log2 || key != key2,
                    "duplicate key {key} for {log:?}"
                );
            }
        }
    }

    #[test]
    fn canonical_is_divisor_one() {
        let rows = [
            (LogKind::Gr, "gAPI", "gr_api"),
            (LogKind::Rhob, "g/cm3", "rhob_gcc"),
            (LogKind::Nphi, "v/v", "nphi_frac"),
            (LogKind::Dtc, "us/ft", "dtc_usft"),
            (LogKind::Resd, "ohm.m", "resd_ohmm"),
            (LogKind::Pef, "b/e", "pef_bfrac"),
        ];
        assert_eq!(rows.len(), LogKind::ALL.len());
        for (log, unit, column) in rows {
            assert_eq!(canonical_unit(log), unit);
            assert_eq!(log.column(), column);
            assert_eq!(
                header_unit(log, canonical_unit(log)),
                HeaderUnit::Divisor(1.0)
            );
        }
    }

    #[test]
    fn blank_words() {
        for log in LogKind::ALL {
            for raw in ["", "   ", "none", "NONE", "unknown"] {
                assert_eq!(header_unit(log, raw), HeaderUnit::Blank, "{log:?} {raw:?}");
            }
        }
    }

    #[test]
    fn unrecognized_examples() {
        let rows = [
            ("cps", LogKind::Gr),
            ("us/ft", LogKind::Gr),
            ("API", LogKind::Rhob),
        ];
        for (raw, log) in rows {
            assert_eq!(
                header_unit(log, raw),
                HeaderUnit::Unrecognized,
                "{log:?} {raw:?}"
            );
        }
    }

    #[test]
    fn dtc_us_per_m() {
        assert_eq!(M_PER_FT, 0.3048);
        assert_eq!(FT_PER_M, 1.0 / 0.3048);
        assert_eq!(FT_PER_M, 3.280839895013123);
        assert_eq!(
            header_unit(LogKind::Dtc, "us/m"),
            HeaderUnit::Divisor(3.280839895013123)
        );
        assert_eq!(
            header_unit(LogKind::Dtc, "\u{00B5}s/m"),
            HeaderUnit::Divisor(FT_PER_M)
        );
        assert_eq!(
            header_unit(LogKind::Dtc, "\u{03BC}s/m"),
            HeaderUnit::Divisor(FT_PER_M)
        );
        let mut values = [250.0f32];
        apply_divisor(&mut values, FT_PER_M).unwrap();
        assert_eq!(values, [76.2f32]);
    }

    #[test]
    fn apply_divisor_skips_non_finite() {
        let mut values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 10.0];
        apply_divisor(&mut values, 10.0).unwrap();
        assert!(values[0].is_nan());
        assert_eq!(values[1], f32::INFINITY);
        assert_eq!(values[2], f32::NEG_INFINITY);
        assert_eq!(values[3], 1.0);
    }

    #[test]
    fn apply_divisor_rejects_bad() {
        for divisor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut values = [f32::NAN, 1.0];
            match apply_divisor(&mut values, divisor) {
                Err(SxError::InvalidInput(msg)) => {
                    assert_eq!(
                        msg,
                        format!("divisor must be finite and > 0, got {divisor}")
                    );
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
            assert!(values[0].is_nan());
            assert_eq!(values[1], 1.0);
        }
    }

    #[test]
    fn apply_divisor_empty_ok() {
        let mut values = [];
        assert!(apply_divisor(&mut values, 2.0).is_ok());
    }

    #[test]
    fn bounds_tables() {
        let physical = [
            (LogKind::Gr, 0.0, 500.0),
            (LogKind::Rhob, 1.0, 3.5),
            (LogKind::Nphi, -0.15, 1.0),
            (LogKind::Dtc, 30.0, 250.0),
            (LogKind::Resd, 0.01, 50000.0),
            (LogKind::Pef, 0.5, 10.0),
        ];
        let plausible = [
            (LogKind::Gr, 1.0, 400.0),
            (LogKind::Rhob, 1.5, 3.2),
            (LogKind::Nphi, -0.05, 0.60),
            (LogKind::Dtc, 40.0, 190.0),
            (LogKind::Resd, 0.05, 5000.0),
            (LogKind::Pef, 1.0, 8.0),
        ];
        assert_eq!(physical.len(), LogKind::ALL.len());
        assert_eq!(plausible.len(), LogKind::ALL.len());
        for (rows, got) in [
            (
                physical.as_slice(),
                physical_bounds as fn(LogKind) -> Bounds,
            ),
            (
                plausible.as_slice(),
                plausible_median as fn(LogKind) -> Bounds,
            ),
        ] {
            for log in LogKind::ALL {
                assert!(rows.iter().any(|(k, _, _)| *k == log));
            }
            for &(log, lo, hi) in rows {
                let bounds = got(log);
                assert_eq!(bounds, Bounds { lo, hi });
                assert!(bounds.contains(lo));
                assert!(bounds.contains(hi));
                assert!(!bounds.contains(f64::NAN));
                assert!(!bounds.contains(lo - (lo.abs() + 1.0)));
                assert!(!bounds.contains(hi + (hi.abs() + 1.0)));
            }
        }

        let nphi = physical_bounds(LogKind::Nphi);
        assert!(nphi.contains(-0.15));
        assert!(!nphi.contains(1.0001));
        assert!(!nphi.contains(f64::NAN));

        for log in LogKind::ALL {
            if log == LogKind::Resd {
                assert_eq!(
                    working_bounds(log),
                    Bounds {
                        lo: -2.0,
                        hi: 4.698970004336019,
                    }
                );
            } else {
                assert_eq!(working_bounds(log), physical_bounds(log));
            }
        }

        assert_eq!(PLAUSIBLE_MIN_SAMPLES, 30);
        assert_eq!(PLAUSIBLE_MIN_IN_BOUNDS_FRAC, 0.05);
    }

    #[test]
    fn working_round_trip() {
        assert!(is_log10(LogKind::Resd));
        for log in LogKind::ALL {
            if log == LogKind::Resd {
                continue;
            }
            assert!(!is_log10(log));
            assert_eq!(to_working(log, 12.5), 12.5);
            assert_eq!(from_working(log, -4.0), -4.0);
            assert!(to_working(log, f64::NAN).is_nan());
            assert!(from_working(log, f64::NAN).is_nan());
            assert_eq!(to_working(log, f64::INFINITY), f64::INFINITY);
        }

        for v in [0.01_f64, 1.0, 50000.0] {
            let back = from_working(LogKind::Resd, to_working(LogKind::Resd, v));
            let rel = (back - v).abs() / v.abs();
            assert!(rel <= 1e-12, "v={v} back={back} rel={rel}");
        }

        assert_eq!(to_working(LogKind::Resd, 100.0), 2.0);
        assert!(to_working(LogKind::Resd, 0.0).is_nan());
        assert!(to_working(LogKind::Resd, -5.0).is_nan());
        assert!(to_working(LogKind::Resd, f64::NAN).is_nan());
        assert_eq!(to_working(LogKind::Resd, f64::INFINITY), f64::INFINITY);
        assert!(to_working(LogKind::Resd, f64::NEG_INFINITY).is_nan());
        assert_eq!(from_working(LogKind::Resd, 2.0), 100.0);
        assert!(from_working(LogKind::Resd, f64::NAN).is_nan());
    }

    #[test]
    fn worked_examples() {
        assert_eq!(
            header_unit(LogKind::Dtc, "us/m"),
            HeaderUnit::Divisor(3.280839895013123)
        );
        let mut dtc = [250.0f32];
        apply_divisor(&mut dtc, FT_PER_M).unwrap();
        assert_eq!(dtc, [76.2f32]);

        assert_eq!(
            header_unit(LogKind::Nphi, "P.U."),
            HeaderUnit::Divisor(100.0)
        );
        let mut nphi = [25.0f32];
        apply_divisor(&mut nphi, 100.0).unwrap();
        assert_eq!(nphi, [0.25]);

        assert_eq!(
            header_unit(LogKind::Rhob, "kg/m3"),
            HeaderUnit::Divisor(1000.0)
        );
        let mut rhob = [2450.0f32];
        apply_divisor(&mut rhob, 1000.0).unwrap();
        assert_eq!(rhob, [2.45f32]);

        assert_eq!(
            header_unit(LogKind::Resd, " ohm-m "),
            HeaderUnit::Divisor(1.0)
        );
        assert_eq!(
            header_unit(LogKind::Resd, "MOHM.M"),
            HeaderUnit::Divisor(1000.0)
        );

        assert_eq!(header_unit(LogKind::Gr, ""), HeaderUnit::Blank);
        assert_eq!(header_unit(LogKind::Gr, " none "), HeaderUnit::Blank);
        assert_eq!(header_unit(LogKind::Gr, "Unknown"), HeaderUnit::Blank);

        assert_eq!(header_unit(LogKind::Gr, "cps"), HeaderUnit::Unrecognized);
        assert_eq!(header_unit(LogKind::Gr, "us/ft"), HeaderUnit::Unrecognized);

        match apply_divisor(&mut [f32::NAN, 1.0], 0.0) {
            Err(SxError::InvalidInput(_)) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }

        let nphi_bounds = physical_bounds(LogKind::Nphi);
        assert!(nphi_bounds.contains(-0.15));
        assert!(!nphi_bounds.contains(1.0001));
        assert!(!nphi_bounds.contains(f64::NAN));

        assert_eq!(
            working_bounds(LogKind::Resd),
            Bounds {
                lo: -2.0,
                hi: 4.698970004336019,
            }
        );
        assert_eq!(to_working(LogKind::Resd, 100.0), 2.0);
        assert!(to_working(LogKind::Resd, 0.0).is_nan());
        assert_eq!(from_working(LogKind::Resd, 2.0), 100.0);
    }
}
