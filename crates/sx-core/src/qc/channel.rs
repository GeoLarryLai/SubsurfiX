// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Whole-channel decisions used by stage 4: GR low span and the
//! resistivity decade check.

/// GR low-span decision.
///
/// True when `flagged` and both spans are finite, `ref_span > 0` and
/// `span < frac * ref_span`. Unit: any (both spans in the same unit). NaN
/// rule: a NaN span gives `false`.
pub fn gr_low_span(span: f64, ref_span: f64, flagged: bool, frac: f64) -> bool {
    flagged && span.is_finite() && ref_span.is_finite() && ref_span > 0.0 && span < frac * ref_span
}

/// Resistivity decade check.
///
/// True when both inputs are finite and `|median_log10 - ref_log10| >=
/// limit_log10`. A well with no reference (NaN) is kept (`false`). Unit:
/// log10(ohm·m).
pub fn resd_decade_off(median_log10: f64, ref_log10: f64, limit_log10: f64) -> bool {
    median_log10.is_finite()
        && ref_log10.is_finite()
        && (median_log10 - ref_log10).abs() >= limit_log10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_span() {
        assert!(gr_low_span(10.0, 100.0, true, 0.2));
        assert!(!gr_low_span(20.0, 100.0, true, 0.2)); // not strictly below
        assert!(!gr_low_span(10.0, 100.0, false, 0.2));
        assert!(!gr_low_span(f64::NAN, 100.0, true, 0.2));
        assert!(!gr_low_span(10.0, f64::NAN, true, 0.2));
        assert!(!gr_low_span(10.0, 0.0, true, 0.2));
    }

    #[test]
    fn decade() {
        assert!(resd_decade_off(1.0, 0.25, 0.75)); // exact boundary: >=
        assert!(resd_decade_off(0.25, 1.0, 0.7));
        assert!(!resd_decade_off(1.0, 0.5, 0.7));
        assert!(!resd_decade_off(1.0, f64::NAN, 0.7));
        assert!(!resd_decade_off(f64::NAN, 1.0, 0.7));
    }
}
