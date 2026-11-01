// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Working values, resampling onto nodes, and unit rescaling.

use crate::config::IntervalParams;
use crate::error::{SxError, SxResult};
use crate::types::LogKind;

/// Default resistivity floor, ohm·m (`resd_floor_ohmm`).
const RESD_FLOOR_OHMM: f64 = 1e-6;

/// Curve values in the units used for math.
///
/// RESD (ohm·m): a value `v ≤ 0` or non-finite becomes NaN, otherwise
/// `log10(max(v, 1e-6))` (unit `log10(ohm·m)`). Every other log is returned as
/// stored. NaN rule: NaN stays NaN. Errors: none. Complexity: `O(n)`.
pub fn working_values(log: LogKind, values: &[f32]) -> Vec<f32> {
    working_values_floor(log, values, RESD_FLOOR_OHMM)
}

/// [`working_values`] with an explicit RESD floor (ohm·m).
pub(crate) fn working_values_floor(log: LogKind, values: &[f32], resd_floor_ohmm: f64) -> Vec<f32> {
    if log != LogKind::Resd {
        return values.to_vec();
    }
    values
        .iter()
        .map(|&v| {
            if v.is_finite() && v > 0.0 {
                f64::from(v).max(resd_floor_ohmm).log10() as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// Resample one curve onto node positions.
///
/// `axis[k]` is the axis value of sample `k` (layer coordinate or MD in ft),
/// `values[k]` the working value, `nodes` the node positions (same unit as the
/// axis) and `max_gap` the widest bridged gap (axis unit). Samples whose axis
/// or value is not finite are skipped; the rest are stable-sorted by axis. A
/// node outside the kept range is NaN; a node equal to a kept axis value takes
/// the first such sample; otherwise the nearest kept samples `L ≤ x < R` give a
/// linear value, unless `x_R − x_L > max_gap` (strict), which gives NaN.
///
/// NaN rule: skip (NaN samples are ignored; a NaN node gives NaN). Errors:
/// [`SxError::Shape`] if `axis` and `values` differ in length. Complexity:
/// `O(n log n + N log n)`.
pub fn resample_curve(
    axis: &[f64],
    values: &[f32],
    nodes: &[f64],
    max_gap: f64,
) -> SxResult<Vec<f64>> {
    if axis.len() != values.len() {
        return Err(SxError::Shape(format!(
            "axis has {} samples, values has {}",
            axis.len(),
            values.len()
        )));
    }
    let mut kept: Vec<(f64, f64)> = axis
        .iter()
        .zip(values.iter())
        .filter(|(x, v)| x.is_finite() && v.is_finite())
        .map(|(&x, &v)| (x, f64::from(v)))
        .collect();
    // Stable sort: equal axis values keep sample order.
    kept.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(nodes
        .iter()
        .map(|&xm| interpolate_node(&kept, xm, max_gap))
        .collect())
}

fn interpolate_node(kept: &[(f64, f64)], xm: f64, max_gap: f64) -> f64 {
    let (Some(first), Some(last)) = (kept.first(), kept.last()) else {
        return f64::NAN;
    };
    if xm.is_nan() || xm < first.0 || xm > last.0 {
        return f64::NAN;
    }
    // First kept sample with x >= xm.
    let hi = kept.partition_point(|s| s.0 < xm);
    let Some(&(x_r, v_r)) = kept.get(hi) else {
        return f64::NAN;
    };
    if x_r == xm {
        return v_r;
    }
    let Some(&(x_l, v_l)) = hi.checked_sub(1).and_then(|lo| kept.get(lo)) else {
        return f64::NAN;
    };
    if x_r - x_l > max_gap {
        return f64::NAN;
    }
    v_l + (xm - x_l) / (x_r - x_l) * (v_r - v_l)
}

/// Rescale one node row to `[0, 1]` by its own percentiles
///
///
/// Needs at least `min_channel_nodes` finite nodes; `p1`, `p99` are the
/// `pct_low` / `pct_high` percentiles of the finite nodes (linear rule); a
/// percentile span below `min_span` or not finite drops the row. Otherwise
/// every value becomes `clamp((x − p1) / span, 0, 1)`. Returns `false` when
/// the row is dropped (the row is then left unchanged).
///
/// NaN rule: skip (NaN stays NaN). Errors: none. Complexity: `O(n log n)`.
pub fn rescale_unit(row: &mut [f64], p: &IntervalParams) -> bool {
    let mut sorted: Vec<f64> = row.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.len() < p.min_channel_nodes || sorted.is_empty() {
        return false;
    }
    sorted.sort_by(f64::total_cmp);
    let p1 = percentile_sorted(&sorted, p.pct_low);
    let p99 = percentile_sorted(&sorted, p.pct_high);
    let span = p99 - p1;
    if !span.is_finite() || span < p.min_span {
        return false;
    }
    for v in row.iter_mut() {
        if !v.is_nan() {
            *v = ((*v - p1) / span).clamp(0.0, 1.0);
        }
    }
    true
}

/// Linear-rule percentile of a sorted, non-empty slice:
/// `h = (p / 100)(n − 1)`, `s[⌊h⌋] + (h − ⌊h⌋)(s[⌊h⌋+1] − s[⌊h⌋])`.
pub(crate) fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (p / 100.0) * (n - 1) as f64;
    let lo = (h.floor().max(0.0) as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let (Some(&a), Some(&b)) = (sorted.get(lo), sorted.get(hi)) else {
        return f64::NAN;
    };
    a + (h - lo as f64) * (b - a)
}

/// Median of a sorted slice: middle value, or mean of the two middle values
/// for an even count. NaN for an empty slice.
pub(crate) fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted.get(n / 2).copied().unwrap_or(f64::NAN)
    } else {
        match (sorted.get(n / 2 - 1), sorted.get(n / 2)) {
            (Some(&a), Some(&b)) => (a + b) / 2.0,
            _ => f64::NAN,
        }
    }
}
