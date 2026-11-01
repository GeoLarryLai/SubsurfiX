// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Local costs: interval-mode unit cost and the GR-first capped
//! cost with pooled robust scale used by full and window modes.

use crate::config::SpanParams;
use crate::dtw::resample::median_sorted;

/// Interval-mode local cost of cell `(i, j)`.
///
/// `a[l]` and `b[l]` are the node rows of used log `l` (same log order in both),
/// already rescaled to `[0, 1]`. The cost is the mean of `|a_l[i] − b_l[j]|`
/// over the logs finite at a's node `i` and b's node `j`, summed in f64. When no
/// log is finite at both, the cost is `gap_cost` (cost units).
///
/// NaN rule: skip. Errors: none (an out-of-range index counts as NaN).
/// Complexity: `O(n_logs)`.
pub fn unit_cost(a: &[Vec<f64>], b: &[Vec<f64>], i: usize, j: usize, gap_cost: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        if let (Some(&x), Some(&y)) = (row_a.get(i), row_b.get(j)) {
            if x.is_finite() && y.is_finite() {
                sum += (x - y).abs();
                count += 1;
            }
        }
    }
    if count == 0 {
        gap_cost
    } else {
        sum / count as f64
    }
}

/// Number of node indices `m` where at least one used log is finite in both
/// wells at node `m` (same index in both).
///
/// NaN rule: skip. Errors: none. Complexity: `O(n_logs × N)`.
pub fn shared_nodes(a: &[Vec<f64>], b: &[Vec<f64>]) -> usize {
    let n = a.iter().chain(b.iter()).map(Vec::len).min().unwrap_or(0);
    (0..n)
        .filter(|&m| {
            a.iter().zip(b.iter()).any(|(row_a, row_b)| {
                matches!(
                    (row_a.get(m), row_b.get(m)),
                    (Some(x), Some(y)) if x.is_finite() && y.is_finite()
                )
            })
        })
        .count()
}

/// Pooled robust scale of one log over both wells' node values.
///
/// Pools the finite values of both rows, `scale = max(mad_factor × MAD,
/// scale_floor)` in working units. `None` when fewer than 2 values are pooled
/// (the log is then not used).
pub(crate) fn pooled_scale(a: &[f64], b: &[f64], p: &SpanParams) -> Option<f64> {
    let mut pooled: Vec<f64> = a
        .iter()
        .chain(b.iter())
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if pooled.len() < 2 {
        return None;
    }
    pooled.sort_by(f64::total_cmp);
    let med = median_sorted(&pooled);
    let mut dev: Vec<f64> = pooled.iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let mad = median_sorted(&dev);
    Some((p.mad_factor * mad).max(p.scale_floor))
}

/// One log's node rows and scale for the capped GR-first cost.
pub(crate) struct CostLog {
    /// True for GR (gets `gr_weight`).
    pub is_gr: bool,
    /// Pooled robust scale, working units.
    pub scale: f64,
    /// Well a node values.
    pub a: Vec<f64>,
    /// Well b node values.
    pub b: Vec<f64>,
}

/// GR-first capped local cost over the used logs.
pub(crate) struct CappedCost {
    logs: Vec<CostLog>,
    gr_weight: f64,
    z_cap: f64,
    sqrt_cap: f64,
}

impl CappedCost {
    /// Build from the used logs (suite order) and the span parameters.
    pub(crate) fn new(logs: Vec<CostLog>, p: &SpanParams) -> Self {
        Self {
            logs,
            gr_weight: p.gr_weight,
            z_cap: p.z_cap,
            sqrt_cap: p.z_cap.sqrt(),
        }
    }

    /// Cost of cell `(i, j)`, in `[0, 1]`; `1.0` when no log is valid there.
    ///
    /// `term = sqrt(min(|z|, z_cap)) / sqrt(z_cap)`, `z = (a_i − b_j) / scale`.
    /// GR (if valid) has weight `gr_weight`; each valid non-GR log gets
    /// `(1 − gr_weight) / n_other`; the result is `Σ w term / Σ w`.
    pub(crate) fn cell(&self, i: usize, j: usize) -> f64 {
        let mut gr_term: Option<f64> = None;
        let mut other_sum = 0.0;
        let mut n_other = 0usize;
        for log in &self.logs {
            let (Some(&x), Some(&y)) = (log.a.get(i), log.b.get(j)) else {
                continue;
            };
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            let z = (x - y) / log.scale;
            let term = z.abs().min(self.z_cap).sqrt() / self.sqrt_cap;
            if log.is_gr {
                gr_term = Some(term);
            } else {
                other_sum += term;
                n_other += 1;
            }
        }
        let mut num = 0.0;
        let mut den = 0.0;
        if let Some(t) = gr_term {
            num += self.gr_weight * t;
            den += self.gr_weight;
        }
        if n_other > 0 {
            let w_other = (1.0 - self.gr_weight) / n_other as f64;
            num += w_other * other_sum;
            den += w_other * n_other as f64;
        }
        if gr_term.is_none() && n_other == 0 {
            return 1.0;
        }
        if den > 0.0 {
            num / den
        } else {
            // Degenerate weights (gr_weight = 1 with no GR, or 0 with only GR):
            // fall back to the plain mean of the valid terms.
            let n = n_other + usize::from(gr_term.is_some());
            (other_sum + gr_term.unwrap_or(0.0)) / n as f64
        }
    }
}
