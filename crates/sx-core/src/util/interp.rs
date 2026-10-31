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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SxError;

    fn assert_close_slice(got: &[f64], expect: &[f64]) {
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect).enumerate() {
            if e.is_nan() {
                assert!(g.is_nan(), "index {i}: got {g}, expected NaN");
            } else if e.is_infinite() {
                assert_eq!(*g, *e, "index {i}");
            } else {
                let diff = (g - e).abs();
                assert!(
                    diff <= 1e-12,
                    "index {i}: got {g}, expected {e}, diff {diff}"
                );
            }
        }
    }

    #[test]
    fn example_nan_clamp_extrapolate() {
        let x = [1.0, 2.0, 4.0];
        let y = [10.0, 20.0, 40.0];
        let xq = [0.5, 1.0, 1.5, 3.0, 4.0, 5.0];
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Nan).unwrap(),
            &[f64::NAN, 10.0, 15.0, 30.0, 40.0, f64::NAN],
        );
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Clamp).unwrap(),
            &[10.0, 10.0, 15.0, 30.0, 40.0, 40.0],
        );
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Extrapolate).unwrap(),
            &[5.0, 10.0, 15.0, 30.0, 40.0, 50.0],
        );
    }

    #[test]
    fn example_nan_in_y() {
        let y = interp_linear(
            &[1.0, 2.0, 4.0],
            &[10.0, f64::NAN, 40.0],
            &[1.0, 1.5, 2.0, 3.0, 4.0],
            OutOfRange::Nan,
        )
        .unwrap();
        assert_close_slice(&y, &[10.0, f64::NAN, f64::NAN, f64::NAN, 40.0]);
    }

    #[test]
    fn query_nan_propagates_for_every_mode() {
        let x = [1.0, 2.0];
        let y = [10.0, 20.0];
        for mode in [OutOfRange::Nan, OutOfRange::Clamp, OutOfRange::Extrapolate] {
            let got = interp_linear(&x, &y, &[f64::NAN], mode).unwrap();
            assert!(got[0].is_nan(), "{mode:?}");
        }
    }

    #[test]
    fn infinite_query_follows_out_of_range_mode() {
        let x = [1.0, 2.0];
        let y = [10.0, 20.0];
        let xq = [f64::NEG_INFINITY, f64::INFINITY];
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Nan).unwrap(),
            &[f64::NAN, f64::NAN],
        );
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Clamp).unwrap(),
            &[10.0, 20.0],
        );
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Extrapolate).unwrap(),
            &[f64::NEG_INFINITY, f64::INFINITY],
        );
    }

    #[test]
    fn single_node_out_of_range() {
        let x = [2.0];
        let y = [5.0];
        let xq = [0.0, 2.0, 9.0];
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Extrapolate).unwrap(),
            &[5.0, 5.0, 5.0],
        );
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Clamp).unwrap(),
            &[5.0, 5.0, 5.0],
        );
        assert_close_slice(
            &interp_linear(&x, &y, &xq, OutOfRange::Nan).unwrap(),
            &[f64::NAN, 5.0, f64::NAN],
        );
        let hit = interp_linear(&[2.0], &[f64::NAN], &[2.0], OutOfRange::Clamp).unwrap();
        assert!(hit[0].is_nan());
    }

    #[test]
    fn empty_xq_is_empty_and_empty_x_is_an_error() {
        assert_eq!(
            interp_linear(&[1.0], &[2.0], &[], OutOfRange::Nan).unwrap(),
            Vec::<f64>::new()
        );
        assert!(matches!(
            interp_linear(&[], &[], &[], OutOfRange::Nan),
            Err(SxError::EmptyInput(_))
        ));
        assert!(matches!(
            interp_linear(&[], &[1.0], &[], OutOfRange::Clamp),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            interp_linear(&[1.0], &[1.0, 2.0], &[1.0], OutOfRange::Nan),
            Err(SxError::Shape(_))
        ));
    }

    #[test]
    fn non_finite_x_is_reported_before_order() {
        let err = interp_linear(
            &[1.0, f64::NAN, 0.0],
            &[0.0, 1.0, 2.0],
            &[1.0],
            OutOfRange::Nan,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not finite"), "{err}");
        assert!(matches!(
            interp_linear(&[0.0, f64::INFINITY], &[1.0, 2.0], &[0.0], OutOfRange::Nan),
            Err(SxError::InvalidInput(_))
        ));
    }

    #[test]
    fn x_must_be_strictly_increasing() {
        assert!(matches!(
            interp_linear(&[2.0, 1.0], &[0.0, 1.0], &[1.5], OutOfRange::Nan),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            interp_linear(&[1.0, 1.0], &[0.0, 1.0], &[1.0], OutOfRange::Nan),
            Err(SxError::InvalidInput(_))
        ));
    }
}
