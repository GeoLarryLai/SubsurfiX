// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Isotonic regression by the pool-adjacent-violators algorithm.

use crate::error::{SxError, SxResult};

/// Weighted least-squares fit that is non-decreasing.
///
/// `weights == None` uses weight `1` for every sample. The result is in the
/// same units as `y`. Adjacent blocks with equal means are left unmerged.
///
/// NaN rule: error. Empty `y` returns an empty vector and does not inspect
/// `weights`.
///
/// Errors, in order: weight length differs ([`SxError::Shape`]), a non-finite
/// `y`, then a weight that is non-finite or `<= 0` ([`SxError::InvalidInput`]).
///
/// Complexity: `O(n)`.
pub fn isotonic_increasing(y: &[f64], weights: Option<&[f64]>) -> SxResult<Vec<f64>> {
    if y.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(w) = weights {
        if w.len() != y.len() {
            return Err(SxError::Shape(format!(
                "isotonic weights length {} does not match y length {}",
                w.len(),
                y.len()
            )));
        }
    }
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "isotonic y[{i}] is not finite"
            )));
        }
    }
    if let Some(w) = weights {
        for (i, &wi) in w.iter().enumerate() {
            if !wi.is_finite() || wi <= 0.0 {
                return Err(SxError::InvalidInput(format!(
                    "isotonic weight[{i}] must be finite and > 0"
                )));
            }
        }
    }

    let mut stack = Vec::new();
    for (i, &yi) in y.iter().enumerate() {
        let w = match weights {
            Some(ws) => match ws.get(i) {
                Some(&w) => w,
                None => return Err(SxError::Shape("isotonic weight is missing".into())),
            },
            None => 1.0,
        };
        stack.push(Block {
            sum_wy: w * yi,
            sum_w: w,
            len: 1,
        });
        pool(&mut stack);
    }

    let mut out = Vec::with_capacity(y.len());
    for block in &stack {
        let mean = block.mean();
        for _ in 0..block.len {
            out.push(mean);
        }
    }
    if out.len() != y.len() {
        return Err(SxError::Numerical(
            "isotonic output length does not match y".into(),
        ));
    }
    Ok(out)
}

struct Block {
    sum_wy: f64,
    sum_w: f64,
    len: usize,
}

impl Block {
    fn mean(&self) -> f64 {
        self.sum_wy / self.sum_w
    }
}

fn pool(stack: &mut Vec<Block>) {
    loop {
        let n = stack.len();
        if n < 2 {
            break;
        }
        let mean_prev = match stack.get(n - 2) {
            Some(block) => block.mean(),
            None => break,
        };
        let mean_last = match stack.get(n - 1) {
            Some(block) => block.mean(),
            None => break,
        };
        if mean_prev <= mean_last {
            break;
        }
        let Some(last) = stack.pop() else { break };
        let Some(prev) = stack.pop() else {
            stack.push(last);
            break;
        };
        stack.push(Block {
            sum_wy: prev.sum_wy + last.sum_wy,
            sum_w: prev.sum_w + last.sum_w,
            len: prev.len + last.len,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SxError;

    fn assert_close_slice(got: &[f64], expect: &[f64]) {
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= 1e-12,
                "index {i}: got {g}, expected {e}, diff {diff}"
            );
        }
    }

    #[test]
    fn examples() {
        assert_close_slice(
            &isotonic_increasing(&[1.0, 3.0, 2.0, 4.0], None).unwrap(),
            &[1.0, 2.5, 2.5, 4.0],
        );
        assert_close_slice(
            &isotonic_increasing(&[1.0, 3.0, 2.0], Some(&[1.0, 1.0, 3.0])).unwrap(),
            &[1.0, 2.25, 2.25],
        );
        assert_close_slice(
            &isotonic_increasing(&[3.0, 2.0, 1.0], None).unwrap(),
            &[2.0, 2.0, 2.0],
        );
        assert_close_slice(
            &isotonic_increasing(&[0.0, 0.6, 0.4, 0.5, 1.0, 0.9], None).unwrap(),
            &[0.0, 0.5, 0.5, 0.5, 0.95, 0.95],
        );
        let y = [1.0, 3.0, 2.0, 4.0];
        assert_eq!(
            isotonic_increasing(&y, None).unwrap(),
            isotonic_increasing(&y, Some(&[1.0, 1.0, 1.0, 1.0])).unwrap()
        );
    }

    #[test]
    fn empty_input_skips_weight_checks() {
        assert_eq!(isotonic_increasing(&[], None).unwrap(), Vec::<f64>::new());
        assert_eq!(
            isotonic_increasing(&[], Some(&[1.0, 2.0])).unwrap(),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn errors() {
        assert!(matches!(
            isotonic_increasing(&[1.0, 2.0], Some(&[1.0])),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            isotonic_increasing(&[1.0, f64::INFINITY], None),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            isotonic_increasing(&[1.0, f64::NAN], None),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            isotonic_increasing(&[1.0], Some(&[0.0])),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            isotonic_increasing(&[1.0], Some(&[-0.5])),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            isotonic_increasing(&[1.0], Some(&[f64::NAN])),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            isotonic_increasing(&[1.0], Some(&[f64::INFINITY])),
            Err(SxError::InvalidInput(_))
        ));
        let err = isotonic_increasing(&[f64::NAN], Some(&[-1.0])).unwrap_err();
        assert!(err.to_string().contains("y["), "{err}");
    }

    proptest::proptest! {
        #[test]
        fn isotonic_is_nondecreasing_and_preserves_weighted_sum(
            y in proptest::collection::vec(-100.0f64..100.0, 0..80),
            weights in proptest::collection::vec(0.05f64..20.0, 0..80),
        ) {
            let n = y.len().min(weights.len());
            let y = &y[..n];
            let weights = &weights[..n];
            let out = isotonic_increasing(y, Some(weights)).unwrap_or_else(|err| panic!("{err}"));
            proptest::prop_assert_eq!(out.len(), n);
            for i in 1..n {
                proptest::prop_assert!(out[i] >= out[i - 1]);
            }
            let in_sum: f64 = y.iter().zip(weights).map(|(yi, wi)| yi * wi).sum();
            let out_sum: f64 = out.iter().zip(weights).map(|(yi, wi)| yi * wi).sum();
            let diff = (in_sum - out_sum).abs();
            let scale = in_sum.abs().max(out_sum.abs());
            proptest::prop_assert!(
                diff <= 1e-9 * scale.max(1.0),
                "in {in_sum} out {out_sum} diff {diff}"
            );
        }
    }
}
