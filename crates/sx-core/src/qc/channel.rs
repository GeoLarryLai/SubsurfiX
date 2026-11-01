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
