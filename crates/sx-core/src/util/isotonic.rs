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
