// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Core banded DTW.
//!
//! Three steps (diagonal, left, up), weight 1, no slope limit, tie-break in the
//! order diagonal, left, up with strict `<`. Memory is banded: `D` keeps two
//! rows of `2w + 1` values, the winning steps keep `n_rows × (2w + 1)` bytes.
//! layout: cell `(i, j)` of the band lives at offset
//! `j + w − i` inside its row, so the diagonal predecessor shares the offset
//! in the previous row, the left predecessor is offset − 1 in the same row and
//! the up predecessor is offset + 1 in the previous row.

use crate::error::{SxError, SxResult};

const STEP_NONE: u8 = 0;
const STEP_DIAG: u8 = 1;
const STEP_LEFT: u8 = 2;
const STEP_UP: u8 = 3;

/// Result of the core recursion.
#[derive(Debug, Clone, PartialEq)]
pub struct DtwPath {
    /// Cells `(i, j)` (row = well a node, column = well b node), from `(0, 0)`
    /// to `(n_rows - 1, n_cols - 1)`.
    pub cells: Vec<(usize, usize)>,
    /// `D` at the end cell (sum of the local costs along the path).
    pub total_cost: f64,
}

impl DtwPath {
    /// `total_cost / cells.len()`; NaN if the path has no cells.
    pub fn mean_cost(&self) -> f64 {
        if self.cells.is_empty() {
            f64::NAN
        } else {
            self.total_cost / self.cells.len() as f64
        }
    }
}

/// Band half-width `w = round_half_even(max(band_frac, 2 / n) × n) − 1` for a
/// square matrix of side `n`.
///
/// `band_frac` is a fraction of `n` (unitless); `w` is in matrix cells. The
/// band admits cell `(i, j)` when `|i − j| ≤ w`.
///
/// NaN rule: error. Errors: [`SxError::InvalidInput`] if `n = 0` or
/// `band_frac` is not finite or `≤ 0`; [`SxError::Infeasible`] if the rounded
/// width `r` is 0. Complexity: `O(1)`.
pub fn band_half_width(n: usize, band_frac: f64) -> SxResult<usize> {
    if n == 0 {
        return Err(SxError::InvalidInput("band needs n > 0".into()));
    }
    if !band_frac.is_finite() || band_frac <= 0.0 {
        return Err(SxError::InvalidInput(format!(
            "band_frac must be finite and > 0, got {band_frac}"
        )));
    }
    let n_f = n as f64;
    let r = (band_frac.max(2.0 / n_f) * n_f).round_ties_even();
    if r < 1.0 {
        return Err(SxError::Infeasible("DTW band is empty".into()));
    }
    // `as` saturates for huge values; the core clamps the width to the matrix.
    Ok((r as usize).saturating_sub(1))
}

/// Core banded DTW with path.
///
/// `cost(i, j)` is called exactly once per in-band cell inside the matrix, in
/// recursion order (rows ascending, then columns ascending). A band cell is
/// `|i − j| ≤ half_width`. Rectangular input is allowed. The path starts at
/// `(0, 0)` and ends at `(n_rows − 1, n_cols − 1)`.
///
/// NaN rule: error (a NaN cost is invalid). Errors:
/// [`SxError::EmptyInput`] for a zero dimension; [`SxError::InvalidInput`] for a
/// NaN, negative or infinite cost; [`SxError::Infeasible`] when
/// `|n_rows − n_cols| > half_width` or the end cell is unreachable.
///
/// Complexity: `O(n_rows × (2w + 1))` time; memory `O(n_rows × (2w + 1))` bytes
/// for the steps plus `O(w)` for `D`. No dense matrix is built.
pub fn banded_dtw<F>(
    n_rows: usize,
    n_cols: usize,
    half_width: usize,
    mut cost: F,
) -> SxResult<DtwPath>
where
    F: FnMut(usize, usize) -> f64,
{
    if n_rows == 0 || n_cols == 0 {
        return Err(SxError::EmptyInput("DTW matrix has no cells".into()));
    }
    // Widening the band past the matrix changes nothing; clamping keeps the
    // storage bounded by the matrix size.
    let w = half_width.min(n_rows.max(n_cols) - 1);
    if n_rows.abs_diff(n_cols) > w {
        return Err(SxError::Infeasible(format!(
            "end cell ({}, {}) lies outside the band of half-width {w}",
            n_rows - 1,
            n_cols - 1
        )));
    }
    let stride = 2 * w + 1;
    let total = n_rows
        .checked_mul(stride)
        .ok_or_else(|| SxError::InvalidInput("DTW band too large".into()))?;
    let mut steps = vec![STEP_NONE; total];
    let mut prev = vec![f64::INFINITY; stride];
    let mut cur = vec![f64::INFINITY; stride];

    for i in 0..n_rows {
        cur.fill(f64::INFINITY);
        let j_lo = i.saturating_sub(w);
        let j_hi = (i + w).min(n_cols - 1);
        let row_steps = steps.get_mut(i * stride..(i + 1) * stride);
        let Some(row_steps) = row_steps else {
            return Err(SxError::Numerical("DTW step storage mismatch".into()));
        };
        for j in j_lo..=j_hi {
            let c = cost(i, j);
            if !c.is_finite() || c < 0.0 {
                return Err(SxError::InvalidInput(format!(
                    "DTW cost at ({i}, {j}) must be finite and >= 0, got {c}"
                )));
            }
            // j + w >= i always holds inside the band.
            let off = j + w - i;
            if i == 0 && j == 0 {
                if let Some(slot) = cur.get_mut(off) {
                    *slot = c;
                }
                continue;
            }
            let diag = if i > 0 && j > 0 {
                prev.get(off).copied().unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            };
            let left = if j > 0 && off > 0 {
                cur.get(off - 1).copied().unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            };
            let up = if i > 0 {
                prev.get(off + 1).copied().unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            };
            let mut best = diag;
            let mut step = STEP_DIAG;
            if left < best {
                best = left;
                step = STEP_LEFT;
            }
            if up < best {
                best = up;
                step = STEP_UP;
            }
            if best.is_finite() {
                if let Some(slot) = cur.get_mut(off) {
                    *slot = c + best;
                }
                if let Some(s) = row_steps.get_mut(off) {
                    *s = step;
                }
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    // After the last swap `prev` holds the final row.
    let end_off = (n_cols - 1) + w - (n_rows - 1);
    let total_cost = prev.get(end_off).copied().unwrap_or(f64::INFINITY);
    if !total_cost.is_finite() {
        return Err(SxError::Infeasible("DTW end cell is unreachable".into()));
    }

    let mut cells = Vec::with_capacity(n_rows + n_cols);
    let (mut i, mut j) = (n_rows - 1, n_cols - 1);
    loop {
        cells.push((i, j));
        if i == 0 && j == 0 {
            break;
        }
        let off = j + w - i;
        let step = steps.get(i * stride + off).copied().unwrap_or(STEP_NONE);
        match step {
            STEP_DIAG if i > 0 && j > 0 => {
                i -= 1;
                j -= 1;
            }
            STEP_LEFT if j > 0 => j -= 1,
            STEP_UP if i > 0 => i -= 1,
            _ => {
                return Err(SxError::Numerical("DTW backtrack lost its path".into()));
            }
        }
    }
    cells.reverse();
    Ok(DtwPath { cells, total_cost })
}

/// [`banded_dtw`] on a dense row-major cost matrix (tests, small inputs).
///
/// `cost` has `n_rows × n_cols` values; cells outside the band are ignored.
///
/// NaN rule and errors as [`banded_dtw`], plus [`SxError::Shape`] when
/// `cost.len() != n_rows × n_cols`. Complexity as [`banded_dtw`].
pub fn banded_dtw_dense(
    cost: &[f64],
    n_rows: usize,
    n_cols: usize,
    half_width: usize,
) -> SxResult<DtwPath> {
    if n_rows.checked_mul(n_cols) != Some(cost.len()) {
        return Err(SxError::Shape(format!(
            "cost has {} values, want {n_rows} x {n_cols}",
            cost.len()
        )));
    }
    banded_dtw(n_rows, n_cols, half_width, |i, j| {
        cost.get(i * n_cols + j).copied().unwrap_or(f64::NAN)
    })
}
