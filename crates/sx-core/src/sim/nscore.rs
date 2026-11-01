// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Normal-score transform: data → standard normal scores by rank, and back.

use crate::error::{SxError, SxResult};

/// Standard normal CDF Φ(x) (erfc approximation, |error| < 1.2e-7).
pub fn norm_cdf(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let erfc = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        1.0 - 0.5 * erfc
    } else {
        0.5 * erfc
    }
}

/// Inverse standard normal CDF Φ⁻¹(p) (Acklam, relative error < 1.2e-9).
/// Returns ±∞ at 0 and 1, NaN outside [0, 1].
pub fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let pl = 0.024_25;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Fitted normal-score table: sorted data values and their cumulative
/// probabilities `(rank + 0.5) / n` (ties share the mean rank).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalScore {
    /// Sorted distinct data values.
    pub values: Vec<f64>,
    /// Cumulative probability of each value.
    pub probs: Vec<f64>,
}

impl NormalScore {
    /// Fit on the finite values.
    ///
    /// # Errors
    /// [`SxError::EmptyInput`] with fewer than 2 distinct finite values.
    pub fn fit(data: &[f64]) -> SxResult<Self> {
        let mut v: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        v.sort_by(f64::total_cmp);
        let n = v.len() as f64;
        let (mut values, mut probs) = (vec![], vec![]);
        let mut i = 0;
        while i < v.len() {
            let mut j = i;
            while j + 1 < v.len() && v[j + 1] == v[i] {
                j += 1;
            }
            values.push(v[i]);
            probs.push(((i + j) as f64 / 2.0 + 0.5) / n);
            i = j + 1;
        }
        if values.len() < 2 {
            return Err(SxError::EmptyInput(
                "normal score needs ≥ 2 distinct values".into(),
            ));
        }
        Ok(Self { values, probs })
    }

    /// Normal score of `x` (linear in probability between table values,
    /// clamped to the table ends). NaN stays NaN.
    pub fn forward(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        norm_inv(interp(&self.values, &self.probs, x))
    }

    /// Value of normal score `y` (linear between table points; tails clamp
    /// to the data minimum and maximum).
    pub fn back(&self, y: f64) -> f64 {
        if y.is_nan() {
            return f64::NAN;
        }
        interp(&self.probs, &self.values, norm_cdf(y))
    }
}

/// Piecewise-linear `y(x)` on increasing `xs`, clamped at the ends.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let i = xs.partition_point(|v| *v <= x);
    let (x0, x1, y0, y1) = (xs[i - 1], xs[i], ys[i - 1], ys[i]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}
