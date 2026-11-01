// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Piecewise-linear interpolation.

use crate::error::{SxError, SxResult};

/// What to do when a query falls outside the node range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfRange {
    /// The query becomes NaN.
    Nan,
    /// Below the first node yields `y[0]`; above the last yields `y[last]`.
    Clamp,
    /// Extend the nearest end segment. A single node repeats `y[0]`.
    Extrapolate,
}

/// Piecewise-linear interpolation of `(x, y)` at the query points `xq`.
///
/// `x` and `xq` share one coordinate; `y` and the result share the value's
/// unit. NaN in a query is NaN. NaN in `y` propagates through the segment
/// formula; an exact hit on a node returns that node's value.
///
/// Empty `xq` returns an empty vector. Empty `x` is [`SxError::EmptyInput`].
///
/// Errors, in order: `x` and `y` lengths differ ([`SxError::Shape`]), empty
/// `x`, a non-finite `x` node ([`SxError::InvalidInput`]), `x` not strictly
/// increasing.
///
/// Complexity: `O(m log n)` for `m` queries and `n` nodes.
pub fn interp_linear(x: &[f64], y: &[f64], xq: &[f64], out: OutOfRange) -> SxResult<Vec<f64>> {
    validate(x, y)?;
    let mut values = Vec::with_capacity(xq.len());
    for &q in xq {
        values.push(interp_one(x, y, q, out)?);
    }
    Ok(values)
}

fn validate(x: &[f64], y: &[f64]) -> SxResult<()> {
    if x.len() != y.len() {
        return Err(SxError::Shape(format!(
            "x length {} does not match y length {}",
            x.len(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(SxError::EmptyInput("x is empty".into()));
    }
    for &xi in x {
        if !xi.is_finite() {
            return Err(SxError::InvalidInput("x is not finite".into()));
        }
    }
    if let Some(&first) = x.first() {
        let mut prev = first;
        for &xi in x.iter().skip(1) {
            if xi <= prev {
                return Err(SxError::InvalidInput("x is not strictly increasing".into()));
            }
            prev = xi;
        }
    }
    Ok(())
}

fn interp_one(x: &[f64], y: &[f64], q: f64, out: OutOfRange) -> SxResult<f64> {
    if q.is_nan() {
        return Ok(f64::NAN);
    }
    let n = x.len();
    let x_first = at(x, 0)?;
    let x_last = at(x, n.saturating_sub(1))?;
    if q < x_first || q > x_last {
        return match out {
            OutOfRange::Nan => Ok(f64::NAN),
            OutOfRange::Clamp => {
                if q < x_first {
                    at(y, 0)
                } else {
                    at(y, n.saturating_sub(1))
                }
            }
            OutOfRange::Extrapolate => {
                if n == 1 {
                    at(y, 0)
                } else if q < x_first {
                    segment(x, y, 0, 1, q)
                } else {
                    segment(x, y, n.saturating_sub(2), n.saturating_sub(1), q)
                }
            }
        };
    }
    let right = x.partition_point(|v| *v <= q);
    let Some(j) = right.checked_sub(1) else {
        return Err(SxError::Numerical(
            "interpolation partition was empty".into(),
        ));
    };
    let xj = at(x, j)?;
    if q == xj {
        return at(y, j);
    }
    segment(x, y, j, j.saturating_add(1), q)
}

fn segment(x: &[f64], y: &[f64], a: usize, b: usize, q: f64) -> SxResult<f64> {
    let x_a = at(x, a)?;
    let x_b = at(x, b)?;
    let y_a = at(y, a)?;
    let y_b = at(y, b)?;
    let slope = (y_b - y_a) / (x_b - x_a);
    Ok(slope * (q - x_a) + y_a)
}

fn at(values: &[f64], index: usize) -> SxResult<f64> {
    values.get(index).copied().ok_or_else(|| {
        SxError::Numerical(format!(
            "index {index} out of range for length {}",
            values.len()
        ))
    })
}
