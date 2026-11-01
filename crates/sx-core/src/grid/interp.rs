// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Inverse distance weighting (IDW) and ordinary kriging (OK).
//!
//! Neighbour search is brute force (`O(n_data)` per target): fine for well
//! counts in the hundreds and grids up to about 10⁶ cells. A k-d tree search
//! is a later speed-up.

use crate::error::{SxError, SxResult};
use crate::grid::variogram::VariogramModel;

/// IDW settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdwParams {
    /// Distance power (2 is common).
    pub power: f64,
    /// Use at most this many nearest data (0 = all).
    pub max_neighbors: usize,
    /// Ignore data farther than this (∞ = no limit).
    pub max_distance: f64,
}

impl Default for IdwParams {
    fn default() -> Self {
        Self {
            power: 2.0,
            max_neighbors: 12,
            max_distance: f64::INFINITY,
        }
    }
}

/// OK settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KrigingParams {
    /// Variogram model.
    pub model: VariogramModel,
    /// Nearest data per target (≥ 1).
    pub max_neighbors: usize,
    /// Ignore data farther than this.
    pub max_distance: f64,
}

/// OK output per target: estimate and kriging variance (NaN where no data
/// are in reach).
#[derive(Debug, Clone, PartialEq)]
pub struct KrigingResult {
    /// Estimates.
    pub estimate: Vec<f64>,
    /// Kriging variances (≥ 0 up to round-off).
    pub variance: Vec<f64>,
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn check(points: &[[f64; 3]], values: &[f64]) -> SxResult<Vec<usize>> {
    if points.len() != values.len() {
        return Err(SxError::Shape(format!(
            "{} points vs {} values",
            points.len(),
            values.len()
        )));
    }
    let ok: Vec<usize> = (0..values.len())
        .filter(|&i| values[i].is_finite() && points[i].iter().all(|c| c.is_finite()))
        .collect();
    if ok.is_empty() {
        return Err(SxError::EmptyInput("no finite data".into()));
    }
    Ok(ok)
}

/// Nearest `k` of `ok` within `dmax` of `t`, as (distance, index), sorted.
pub(crate) fn nearest(
    points: &[[f64; 3]],
    ok: &[usize],
    t: &[f64; 3],
    k: usize,
    dmax: f64,
) -> Vec<(f64, usize)> {
    let mut d: Vec<(f64, usize)> = ok
        .iter()
        .map(|&i| (dist(&points[i], t), i))
        .filter(|(h, _)| *h <= dmax)
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    if k > 0 {
        d.truncate(k);
    }
    d
}

/// IDW estimates at `targets`. A target on a datum takes its value. NaN
/// where no datum is in reach.
///
/// # Errors
/// [`SxError::Shape`], [`SxError::EmptyInput`], or
/// [`SxError::InvalidInput`] for a negative or non-finite power.
pub fn idw(
    points: &[[f64; 3]],
    values: &[f64],
    targets: &[[f64; 3]],
    p: &IdwParams,
) -> SxResult<Vec<f64>> {
    let ok = check(points, values)?;
    if !(p.power.is_finite() && p.power >= 0.0) {
        return Err(SxError::InvalidInput(format!("IDW power {}", p.power)));
    }
    Ok(targets
        .iter()
        .map(|t| {
            let nb = nearest(points, &ok, t, p.max_neighbors, p.max_distance);
            if let Some(&(h, i)) = nb.first() {
                if h == 0.0 {
                    return values[i];
                }
            }
            let (mut sw, mut sv) = (0.0, 0.0);
            for (h, i) in nb {
                let w = h.powf(-p.power);
                sw += w;
                sv += w * values[i];
            }
            if sw > 0.0 {
                sv / sw
            } else {
                f64::NAN
            }
        })
        .collect())
}

/// Solve `a·x = b` in place (Gaussian elimination, partial pivoting).
pub(crate) fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for c in 0..n {
        let piv = (c..n).max_by(|&r, &s| a[r * n + c].abs().total_cmp(&a[s * n + c].abs()))?;
        if a[piv * n + c].abs() < 1e-12 {
            return None;
        }
        if piv != c {
            for k in 0..n {
                a.swap(c * n + k, piv * n + k);
            }
            b.swap(c, piv);
        }
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    a[r * n + k] -= f * a[c * n + k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r * n + k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r * n + r];
    }
    Some(x)
}

/// Polynomial drift (trend) in the horizontal coordinates x, y.
/// Limitation: no drift in z, no external drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drift {
    /// Constant mean only (ordinary kriging).
    None,
    /// 1, x, y.
    Linear,
    /// 1, x, y, x², xy, y².
    Quadratic,
}

impl Drift {
    /// Parse "none" / "linear" / "quadratic".
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for other names.
    pub fn parse(s: &str) -> SxResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "constant" | "ordinary" => Ok(Self::None),
            "linear" => Ok(Self::Linear),
            "quadratic" => Ok(Self::Quadratic),
            _ => Err(SxError::InvalidInput(format!("unknown drift {s:?}"))),
        }
    }

    /// Number of basis functions.
    pub fn n_terms(self) -> usize {
        match self {
            Self::None => 1,
            Self::Linear => 3,
            Self::Quadratic => 6,
        }
    }

    fn basis(self, u: f64, v: f64) -> [f64; 6] {
        [1.0, u, v, u * u, u * v, v * v]
    }
}

/// Least-squares polynomial trend in x, y (global), for detrend-then-krige.
#[derive(Debug, Clone, PartialEq)]
pub struct Trend {
    /// Drift form.
    pub drift: Drift,
    /// Centre used to scale coordinates.
    pub center: [f64; 2],
    /// Scale used for coordinates.
    pub scale: f64,
    /// Coefficients, `drift.n_terms()` of them.
    pub coef: Vec<f64>,
}

impl Trend {
    /// Trend value at `p`.
    pub fn eval(&self, p: &[f64; 3]) -> f64 {
        let b = self.drift.basis(
            (p[0] - self.center[0]) / self.scale,
            (p[1] - self.center[1]) / self.scale,
        );
        self.coef.iter().zip(b).map(|(c, f)| c * f).sum()
    }
}

/// Fit a polynomial trend by least squares (NaN values skipped).
///
/// # Errors
/// [`SxError::Shape`], [`SxError::EmptyInput`], or [`SxError::Numerical`]
/// when there are too few distinct points for the drift.
pub fn fit_trend(points: &[[f64; 3]], values: &[f64], drift: Drift) -> SxResult<Trend> {
    let ok = check(points, values)?;
    let m = drift.n_terms();
    let n = ok.len() as f64;
    let cx = ok.iter().map(|&i| points[i][0]).sum::<f64>() / n;
    let cy = ok.iter().map(|&i| points[i][1]).sum::<f64>() / n;
    let scale = ok
        .iter()
        .map(|&i| (points[i][0] - cx).abs().max((points[i][1] - cy).abs()))
        .fold(0.0, f64::max)
        .max(1e-12);
    let mut a = vec![0.0; m * m];
    let mut b = vec![0.0; m];
    for &i in &ok {
        let f = drift.basis((points[i][0] - cx) / scale, (points[i][1] - cy) / scale);
        for r in 0..m {
            b[r] += f[r] * values[i];
            for c in 0..m {
                a[r * m + c] += f[r] * f[c];
            }
        }
    }
    let coef = solve(a, b, m)
        .ok_or_else(|| SxError::Numerical("trend fit is singular (too few points)".into()))?;
    Ok(Trend {
        drift,
        center: [cx, cy],
        scale,
        coef,
    })
}

/// Ordinary kriging with the nearest `max_neighbors` data. Weights sum to 1
/// (Lagrange multiplier μ). Variance = Σ λᵢ·γ(dᵢ) + μ. A target on a datum
/// returns it with variance 0. Singular systems (duplicate points) fall back
/// to the nearest datum's value with NaN variance.
///
/// # Errors
/// [`SxError::Shape`], [`SxError::EmptyInput`], or
/// [`SxError::InvalidInput`] for `max_neighbors = 0`.
pub fn ordinary_kriging(
    points: &[[f64; 3]],
    values: &[f64],
    targets: &[[f64; 3]],
    p: &KrigingParams,
) -> SxResult<KrigingResult> {
    universal_kriging(points, values, targets, p, Drift::None)
}

/// Universal kriging (kriging with a polynomial drift in x, y). With
/// [`Drift::None`] this is ordinary kriging. Drift terms use coordinates
/// centred on the target and scaled by the farthest neighbour, for a
/// well-conditioned system. The variogram should be that of the residuals
/// from the drift (for example from [`fit_trend`]). Variance =
/// Σ λᵢ·γ(dᵢ) + Σ μₗ·fₗ(x₀). Too few neighbours for the drift fall back to
/// ordinary kriging for that target.
///
/// # Errors
/// As [`ordinary_kriging`].
pub fn universal_kriging(
    points: &[[f64; 3]],
    values: &[f64],
    targets: &[[f64; 3]],
    p: &KrigingParams,
    drift: Drift,
) -> SxResult<KrigingResult> {
    let ok = check(points, values)?;
    if p.max_neighbors == 0 {
        return Err(SxError::InvalidInput("max_neighbors must be ≥ 1".into()));
    }
    let g = |h: f64| p.model.gamma(h);
    let mut est = Vec::with_capacity(targets.len());
    let mut var = Vec::with_capacity(targets.len());
    for t in targets {
        let nb = nearest(points, &ok, t, p.max_neighbors, p.max_distance);
        match nb.first() {
            None => {
                est.push(f64::NAN);
                var.push(f64::NAN);
                continue;
            }
            Some(&(0.0, i)) => {
                est.push(values[i]);
                var.push(0.0);
                continue;
            }
            _ => {}
        }
        let (e, v) = krige_one(points, values, t, &nb, &g, drift)
            .or_else(|| krige_one(points, values, t, &nb, &g, Drift::None))
            .unwrap_or((values[nb[0].1], f64::NAN));
        est.push(e);
        var.push(v);
    }
    Ok(KrigingResult {
        estimate: est,
        variance: var,
    })
}

/// Leave-one-out cross-validation of (universal) kriging on the data:
/// returns `(rmse, msse)`, where MSSE is the mean of
/// (error² / kriging variance). A well-calibrated variance gives MSSE ≈ 1.
/// Points without a usable estimate or with zero variance are skipped.
///
/// # Errors
/// As [`universal_kriging`].
pub fn kriging_loo(
    points: &[[f64; 3]],
    values: &[f64],
    p: &KrigingParams,
    drift: Drift,
) -> SxResult<(f64, f64)> {
    check(points, values)?;
    let n = points.len();
    let (mut se, mut zs, mut k) = (0.0, 0.0, 0usize);
    for i in 0..n {
        let (mut pp, mut vv) = (points.to_vec(), values.to_vec());
        let (t, truth) = (pp.remove(i), vv.remove(i));
        let r = universal_kriging(&pp, &vv, &[t], p, drift)?;
        let (e, v) = (r.estimate[0], r.variance[0]);
        if e.is_finite() && v.is_finite() && v > 0.0 && truth.is_finite() {
            se += (e - truth).powi(2);
            zs += (e - truth).powi(2) / v;
            k += 1;
        }
    }
    if k == 0 {
        return Err(SxError::EmptyInput("no leave-one-out estimate".into()));
    }
    Ok(((se / k as f64).sqrt(), zs / k as f64))
}

/// Rescale a variogram's nugget and partial sill so that leave-one-out
/// kriging on the data has MSSE = 1 (the cross-validation sill correction).
/// Kriging weights, and so estimates, do not change with the scale; only the
/// kriging variance does. Use it when the variogram was fitted on residuals
/// of a trend (universal kriging or detrend-then-OK), which biases the sill
/// low. Returns the scaled model and the scale factor.
///
/// # Errors
/// As [`kriging_loo`].
pub fn calibrate_sill(
    points: &[[f64; 3]],
    values: &[f64],
    p: &KrigingParams,
    drift: Drift,
) -> SxResult<(VariogramModel, f64)> {
    let (_, msse) = kriging_loo(points, values, p, drift)?;
    let f = if msse.is_finite() && msse > 0.0 {
        msse
    } else {
        1.0
    };
    let mut m = p.model;
    m.nugget *= f;
    m.psill *= f;
    Ok((m, f))
}

/// One kriging system; `None` if singular or too few neighbours.
pub(crate) fn krige_one(
    points: &[[f64; 3]],
    values: &[f64],
    t: &[f64; 3],
    nb: &[(f64, usize)],
    g: &dyn Fn(f64) -> f64,
    drift: Drift,
) -> Option<(f64, f64)> {
    let n = nb.len();
    let l = drift.n_terms();
    if n < l {
        return None;
    }
    let s = nb
        .iter()
        .map(|&(_, i)| (points[i][0] - t[0]).abs().max((points[i][1] - t[1]).abs()))
        .fold(0.0, f64::max)
        .max(1e-12);
    let m = n + l;
    let mut a = vec![0.0; m * m];
    let mut b = vec![0.0; m];
    for (r, &(dr, ir)) in nb.iter().enumerate() {
        for (c, &(_, ic)) in nb.iter().enumerate() {
            a[r * m + c] = g(dist(&points[ir], &points[ic]));
        }
        let f = drift.basis((points[ir][0] - t[0]) / s, (points[ir][1] - t[1]) / s);
        for q in 0..l {
            a[r * m + n + q] = f[q];
            a[(n + q) * m + r] = f[q];
        }
        b[r] = g(dr);
    }
    let f0 = drift.basis(0.0, 0.0);
    b[n..m].copy_from_slice(&f0[..l]);
    let x = solve(a, b.clone(), m)?;
    let e = (0..n).map(|r| x[r] * values[nb[r].1]).sum();
    let v = (0..m).map(|r| x[r] * b[r]).sum::<f64>();
    Some((e, v.max(0.0)))
}
