// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Isotropic semivariogram: experimental bins and model fitting.
//!
//! γ(h) = ½·mean[(z(x) − z(x + h))²] over pairs whose distance falls in a
//! lag bin. Anisotropy is not handled yet; scale
//! the coordinates first (for example z by a horizontal/vertical ratio).

use serde::{Deserialize, Serialize};

use crate::error::{SxError, SxResult};

/// One lag bin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VariogramBin {
    /// Mean pair distance in the bin.
    pub lag: f64,
    /// Semivariance.
    pub gamma: f64,
    /// Number of pairs.
    pub pairs: usize,
}

/// Model shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    /// Reaches the sill exactly at `range`.
    Spherical,
    /// Practical range: 95% of the sill at `range`.
    Exponential,
    /// Practical range: 95% of the sill at `range`.
    Gaussian,
}

impl ModelKind {
    /// Parse "spherical" / "exponential" / "gaussian".
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for other names.
    pub fn parse(s: &str) -> SxResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "spherical" | "sph" => Ok(Self::Spherical),
            "exponential" | "exp" => Ok(Self::Exponential),
            "gaussian" | "gau" => Ok(Self::Gaussian),
            _ => Err(SxError::InvalidInput(format!(
                "unknown variogram model {s:?}"
            ))),
        }
    }

    /// Unit-sill shape `f(h/range)` in [0, 1].
    fn shape(self, h: f64, range: f64) -> f64 {
        let r = h / range;
        match self {
            Self::Spherical => {
                if r >= 1.0 {
                    1.0
                } else {
                    1.5 * r - 0.5 * r * r * r
                }
            }
            Self::Exponential => 1.0 - (-3.0 * r).exp(),
            Self::Gaussian => 1.0 - (-3.0 * r * r).exp(),
        }
    }
}

/// Fitted model: γ(h) = nugget + psill·f(h/range) for h > 0, γ(0) = 0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VariogramModel {
    /// Shape.
    pub kind: ModelKind,
    /// Nugget ≥ 0.
    pub nugget: f64,
    /// Partial sill ≥ 0 (total sill = nugget + psill).
    pub psill: f64,
    /// Range > 0 (practical range for exponential and Gaussian).
    pub range: f64,
}

impl VariogramModel {
    /// Semivariance at distance `h`.
    pub fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            0.0
        } else {
            self.nugget + self.psill * self.kind.shape(h, self.range)
        }
    }

    /// Covariance `sill − γ(h)` (with C(0) = sill).
    pub fn covariance(&self, h: f64) -> f64 {
        self.nugget + self.psill - self.gamma(h)
    }
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Experimental variogram with `n_lags` bins of width `lag_width` (bin `b`
/// holds distances in `(b·w, (b+1)·w]`). NaN values are skipped; empty bins
/// are left out. `O(n²)` pairs: subsample large sets first.
///
/// # Errors
/// [`SxError::Shape`] on a length mismatch; [`SxError::InvalidInput`] for a
/// non-positive width or zero bins; [`SxError::EmptyInput`] with no pairs.
pub fn experimental_variogram(
    points: &[[f64; 3]],
    values: &[f64],
    lag_width: f64,
    n_lags: usize,
) -> SxResult<Vec<VariogramBin>> {
    if points.len() != values.len() {
        return Err(SxError::Shape(format!(
            "{} points vs {} values",
            points.len(),
            values.len()
        )));
    }
    if !(lag_width > 0.0) || n_lags == 0 {
        return Err(SxError::InvalidInput(
            "lag_width must be > 0 and n_lags ≥ 1".into(),
        ));
    }
    let mut sum = vec![(0.0, 0.0, 0usize); n_lags];
    let ok: Vec<usize> = (0..values.len())
        .filter(|&i| values[i].is_finite())
        .collect();
    for (a, &i) in ok.iter().enumerate() {
        for &j in &ok[a + 1..] {
            let h = dist(&points[i], &points[j]);
            if h <= 0.0 {
                continue;
            }
            let b = (h / lag_width).ceil() as usize;
            if b == 0 || b > n_lags {
                continue;
            }
            let s = &mut sum[b - 1];
            s.0 += h;
            s.1 += 0.5 * (values[i] - values[j]).powi(2);
            s.2 += 1;
        }
    }
    let bins: Vec<VariogramBin> = sum
        .into_iter()
        .filter(|s| s.2 > 0)
        .map(|(h, g, n)| VariogramBin {
            lag: h / n as f64,
            gamma: g / n as f64,
            pairs: n,
        })
        .collect();
    if bins.is_empty() {
        return Err(SxError::EmptyInput(
            "no point pairs within the lag bins".into(),
        ));
    }
    Ok(bins)
}

/// Fit `nugget`, `psill`, `range` to bins by weighted least squares, weight
/// = pair count. For each of 200 trial ranges between the first lag/2 and 2×
/// the last lag, nugget and psill come from a 2-variable non-negative linear
/// least-squares solve; the best range wins.
///
/// # Errors
/// [`SxError::EmptyInput`] with no bins.
pub fn fit_variogram(bins: &[VariogramBin], kind: ModelKind) -> SxResult<VariogramModel> {
    if bins.is_empty() {
        return Err(SxError::EmptyInput("no variogram bins".into()));
    }
    let lo = bins.iter().map(|b| b.lag).fold(f64::INFINITY, f64::min) * 0.5;
    let hi = bins.iter().map(|b| b.lag).fold(0.0, f64::max) * 2.0;
    let mut best: Option<(f64, VariogramModel)> = None;
    for t in 0..200 {
        let range = lo * (hi / lo).powf(t as f64 / 199.0);
        // Normal equations for γ ≈ n + p·f, weights w.
        let (mut sw, mut sf, mut sff, mut sg, mut sfg) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for b in bins {
            let w = b.pairs as f64;
            let f = kind.shape(b.lag, range);
            sw += w;
            sf += w * f;
            sff += w * f * f;
            sg += w * b.gamma;
            sfg += w * f * b.gamma;
        }
        let det = sw * sff - sf * sf;
        let mut cands = vec![
            (0.0, if sff > 0.0 { (sfg / sff).max(0.0) } else { 0.0 }),
            ((sg / sw).max(0.0), 0.0),
        ];
        if det.abs() > 1e-12 {
            let n = (sg * sff - sf * sfg) / det;
            let p = (sw * sfg - sf * sg) / det;
            if n >= 0.0 && p >= 0.0 {
                cands.push((n, p));
            }
        }
        for (nugget, psill) in cands {
            let m = VariogramModel {
                kind,
                nugget,
                psill,
                range,
            };
            let sse: f64 = bins
                .iter()
                .map(|b| b.pairs as f64 * (m.gamma(b.lag) - b.gamma).powi(2))
                .sum();
            if best.as_ref().is_none_or(|(e, _)| sse < *e) {
                best = Some((sse, m));
            }
        }
    }
    Ok(best.map(|(_, m)| m).expect("at least one candidate"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_values() {
        let m = VariogramModel {
            kind: ModelKind::Spherical,
            nugget: 0.1,
            psill: 0.9,
            range: 10.0,
        };
        assert_eq!(m.gamma(0.0), 0.0);
        assert!((m.gamma(5.0) - (0.1 + 0.9 * (0.75 - 0.0625))).abs() < 1e-12);
        assert!((m.gamma(20.0) - 1.0).abs() < 1e-12);
        assert!((m.covariance(0.0) - 1.0).abs() < 1e-12);
        let e = VariogramModel {
            kind: ModelKind::Exponential,
            ..m
        };
        assert!((e.gamma(10.0) - (0.1 + 0.9 * (1.0 - (-3f64).exp()))).abs() < 1e-12);
    }

    #[test]
    fn fit_recovers_a_known_model() {
        let truth = VariogramModel {
            kind: ModelKind::Spherical,
            nugget: 0.2,
            psill: 1.0,
            range: 300.0,
        };
        let bins: Vec<VariogramBin> = (1..=20)
            .map(|k| VariogramBin {
                lag: 25.0 * k as f64,
                gamma: truth.gamma(25.0 * k as f64),
                pairs: 50,
            })
            .collect();
        let m = fit_variogram(&bins, ModelKind::Spherical).unwrap();
        assert!((m.nugget - 0.2).abs() < 0.02, "{m:?}");
        assert!((m.psill - 1.0).abs() < 0.03, "{m:?}");
        assert!((m.range - 300.0).abs() < 10.0, "{m:?}");
    }

    #[test]
    fn experimental_bins_by_hand() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let v = [0.0, 1.0, 3.0];
        let b = experimental_variogram(&p, &v, 1.0, 2).unwrap();
        // lag 1: pairs (0,1) 0.5, (1,2) 2.0 → 1.25; lag 2: (0,2) 4.5
        assert_eq!(b[0].pairs, 2);
        assert!((b[0].gamma - 1.25).abs() < 1e-12);
        assert!((b[1].gamma - 4.5).abs() < 1e-12);
    }
}
