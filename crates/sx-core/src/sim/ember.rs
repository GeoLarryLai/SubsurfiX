// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! EMBER: embedded-model estimation and simulation with a decision forest
//! ([Daly (2021)](https://doi.org/10.3389/frai.2021.624697),
//! [Daly (2022)](https://doi.org/10.1007/s11004-021-09972-8)).
//!
//! 1. Embedded models: simple kriging of the target at two (or more)
//!    ranges, short and long by default. For each tree the estimates at
//!    the training rows use only that tree's bootstrap sample, leaving the
//!    row's whole group (well) out by default, or only the row itself
//!    ([`EmbedHoldout::Sample`], as in the paper); at the targets all data
//!    are used.
//! 2. Envelope: the forest weights `ωᵢ(x)` give the conditional
//!    distribution `F(z | x) = Σ ωᵢ 1[Zᵢ ≤ z]` (Meinshausen 2006). The
//!    EMBER estimate is its mean `Σ ωᵢ Zᵢ`; quantiles are read off `F`.
//!    Splits use one random value per candidate feature. Quantile LightGBM
//!    is an alternative envelope (quantile grid, embedded estimates
//!    computed once).
//! 3. Sampling variogram: covariance of the rescaled residuals
//!    `(Z(x) − m(x)) / σ(x)` at the data, with `m`, `σ` from held-out
//!    envelopes (out-of-bag trees for the forest, group folds for
//!    LightGBM), fitted with an exponential model and scaled to sill 1.
//! 4. Simulation: at each datum the Gaussian value `X` is restricted to
//!    `[Φ⁻¹(F(z⁻)), Φ⁻¹(F(z))]` under the datum's envelope and drawn jointly
//!    by a Gibbs sampler (truncated Gaussian, Freulon 1992); the targets
//!    follow by sequential Gaussian simulation conditioned on those values,
//!    and `z = F⁻¹(Φ(X))`.
//!
//! Neighbours for the embedded models come from the nearest groups by
//! distance between group centres, at most `max_per_group` points per
//! group. Coordinates are divided by `anisotropy` before any distance.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{SxError, SxResult};
use crate::grid::interp::solve;
use crate::grid::variogram::{experimental_variogram, fit_variogram, ModelKind, VariogramModel};
use crate::sim::nscore::{norm_cdf, norm_inv};
use crate::sim::propagation::{lgbm_quantiles, PropagationModel, PropagationParams};
use crate::sim::qrf::{QrfParams, QuantileForest};
use crate::sim::{auto_cell, gauss, random_path, Buckets};

/// What an embedded estimate at a training row leaves out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedHoldout {
    /// The row itself (Daly 2022, §2.3).
    Sample,
    /// The row's whole group (well).
    Group,
}

/// EMBER settings.
#[derive(Debug, Clone, PartialEq)]
pub struct EmberParams {
    /// Envelope model and its settings (`quantiles`, `seed` and the model
    /// settings are used).
    pub propagation: PropagationParams,
    /// Coordinate divisors (x, y, z) before distances; for example
    /// `[1, 1, 1/50]` makes 1 ft vertical count as 50 ft horizontal.
    pub anisotropy: [f64; 3],
    /// Use the scaled coordinates as forest inputs.
    pub use_coordinates: bool,
    /// Add embedded kriging estimates (off = plain forest ensemble).
    pub embed: bool,
    /// Embedded kriging ranges in scaled units (empty = short and long
    /// from the group spacing and the extent).
    pub embed_ranges: Vec<f64>,
    /// Hold-out unit for the embedded estimates at training rows
    /// (default: the whole group).
    pub embed_holdout: EmbedHoldout,
    /// Neighbours per embedded kriging estimate.
    pub max_neighbors: usize,
    /// Most neighbours from one group.
    pub max_per_group: usize,
    /// One random split value per candidate feature (forest envelope).
    pub random_split: bool,
    /// Realizations (0 = envelope only).
    pub n_realizations: usize,
    /// Sampling variogram of the Gaussian field (sill 1); `None` = fitted.
    pub sampling_variogram: Option<VariogramModel>,
    /// Neighbours in the Gibbs sampler and the simulation.
    pub sim_neighbors: usize,
    /// Gibbs sweeps over the data per realization.
    pub gibbs_sweeps: usize,
    /// Recalibrate the envelope levels with the held-out PIT values (off =
    /// the published method).
    pub calibrate: bool,
    /// Spread of the simulated Gaussian deviations at the targets.
    pub spread: Spread,
    /// Block size (scaled x/y units) for the training folds used by
    /// `calibrate` and [`Spread::Auto`]; 0 = folds of whole groups.
    pub calib_block: f64,
}

/// Scale of the simulated Gaussian deviation at the targets
/// (`x = m + s sqrt(v) e`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spread {
    /// s = 1 (the published method).
    None,
    /// Fixed s.
    Fixed(f64),
    /// s chosen on held-out folds of the training data so that the
    /// realization P10-P90 interval covers 80 % of the held-out values.
    Auto,
    /// As [`Spread::Auto`] but never below 1: widens only when the
    /// training folds show under-coverage (the default).
    Inflate,
}

impl Spread {
    fn fitted(self) -> bool {
        matches!(self, Spread::Auto | Spread::Inflate)
    }
}

impl Default for EmberParams {
    fn default() -> Self {
        Self {
            propagation: PropagationParams::default(),
            anisotropy: [1.0, 1.0, 1.0],
            use_coordinates: true,
            embed: true,
            embed_ranges: vec![],
            embed_holdout: EmbedHoldout::Group,
            max_neighbors: 16,
            max_per_group: 4,
            random_split: true,
            n_realizations: 0,
            sampling_variogram: None,
            sim_neighbors: 16,
            gibbs_sweeps: 100,
            calibrate: false,
            spread: Spread::Inflate,
            calib_block: 0.0,
        }
    }
}

/// EMBER input. `secondary` rows may be empty (no secondary inputs) but
/// must have the same length for training and targets.
#[derive(Debug, Clone, Copy)]
pub struct EmberInput<'a> {
    /// Data locations.
    pub train_xyz: &'a [[f64; 3]],
    /// Data values (finite).
    pub train_y: &'a [f64],
    /// Group per datum (usually the well).
    pub train_groups: &'a [i32],
    /// Secondary inputs at the data, `[row][var]`.
    pub train_secondary: &'a [Vec<f64>],
    /// Target locations.
    pub pred_xyz: &'a [[f64; 3]],
    /// Secondary inputs at the targets.
    pub pred_secondary: &'a [Vec<f64>],
}

/// EMBER result.
#[derive(Debug, Clone, PartialEq)]
pub struct EmberResult {
    /// Envelope quantiles `[target][level]` at the requested levels.
    pub quantiles: Vec<Vec<f64>>,
    /// Envelope mean (the EMBER estimate).
    pub mean: Vec<f64>,
    /// Envelope standard deviation.
    pub std: Vec<f64>,
    /// Realizations `[realization][target]`.
    pub realizations: Vec<Vec<f64>>,
    /// Embedded estimates at the targets, `[target][model]`.
    pub embedded: Vec<Vec<f64>>,
    /// Sampling variogram used (sill 1).
    pub sampling_variogram: VariogramModel,
    /// Embedded kriging ranges (scaled units).
    pub embed_ranges: Vec<f64>,
}

/// Conditional distribution at one location.
#[derive(Debug, Clone)]
enum Envelope {
    /// Forest: sorted distinct values `z` with cumulative weights `c`
    /// (`c` ends at 1).
    Steps { z: Vec<f64>, c: Vec<f64> },
    /// Quantile grid: values `q` at `levels` (both ascending, levels from 0
    /// to 1), linear in between.
    Grid { levels: Vec<f64>, q: Vec<f64> },
}

impl Envelope {
    fn from_weights(w: &[(f64, f64)]) -> Option<Self> {
        let tot: f64 = w.iter().map(|a| a.1).sum();
        if w.is_empty() || !(tot > 0.0) {
            return None;
        }
        let (mut z, mut c): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
        let mut acc = 0.0;
        for &(v, wi) in w {
            acc += wi / tot;
            if z.last() == Some(&v) {
                *c.last_mut().unwrap() = acc;
            } else {
                z.push(v);
                c.push(acc);
            }
        }
        *c.last_mut().unwrap() = 1.0;
        Some(Self::Steps { z, c })
    }

    /// `F⁻¹(u)`: smallest value whose CDF reaches `u`.
    fn quantile(&self, u: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        match self {
            Self::Steps { z, c } => z[c.partition_point(|v| *v < u - 1e-12).min(z.len() - 1)],
            Self::Grid { levels, q } => {
                let i = levels
                    .partition_point(|l| *l < u)
                    .clamp(1, levels.len() - 1);
                let (l0, l1) = (levels[i - 1], levels[i]);
                let t = if l1 > l0 { (u - l0) / (l1 - l0) } else { 0.0 };
                q[i - 1] + t * (q[i] - q[i - 1])
            }
        }
    }

    /// `[F(z⁻), F(z)]`: the levels `u` with `F⁻¹(u) = z` (a point when `z`
    /// is off the support).
    fn invert(&self, zv: f64) -> (f64, f64) {
        match self {
            Self::Steps { z, c } => {
                let k = z.partition_point(|v| *v < zv);
                let below = if k == 0 { 0.0 } else { c[k - 1] };
                if k < z.len() && z[k] == zv {
                    (below, c[k])
                } else {
                    (below, below)
                }
            }
            Self::Grid { levels, q } => {
                let n = q.len();
                if zv <= q[0] {
                    return (0.0, levels[q.iter().rposition(|v| *v <= q[0]).unwrap_or(0)]);
                }
                if zv >= q[n - 1] {
                    let j = q.iter().position(|v| *v >= q[n - 1]).unwrap_or(n - 1);
                    return (levels[j], 1.0);
                }
                let lo_i = q.partition_point(|v| *v < zv);
                let hi_i = q.partition_point(|v| *v <= zv);
                let at = |i: usize| {
                    let (q0, q1) = (q[i - 1], q[i]);
                    let t = if q1 > q0 { (zv - q0) / (q1 - q0) } else { 0.0 };
                    levels[i - 1] + t * (levels[i] - levels[i - 1])
                };
                let lo = at(lo_i.max(1));
                let hi = if hi_i > lo_i { levels[hi_i - 1] } else { lo };
                (lo, hi.max(lo))
            }
        }
    }

    fn moments(&self) -> (f64, f64) {
        let (mut m, mut m2) = (0.0, 0.0);
        match self {
            Self::Steps { z, c } => {
                let mut prev = 0.0;
                for (v, ci) in z.iter().zip(c) {
                    let w = ci - prev;
                    prev = *ci;
                    m += w * v;
                    m2 += w * v * v;
                }
            }
            Self::Grid { levels, q } => {
                for i in 1..levels.len() {
                    let du = levels[i] - levels[i - 1];
                    let (a, b) = (q[i - 1], q[i]);
                    m += du * (a + b) / 2.0;
                    m2 += du * (a * a + a * b + b * b) / 3.0;
                }
            }
        }
        (m, (m2 - m * m).max(0.0).sqrt())
    }

    /// Same distribution with its levels remapped by `g` (increasing on
    /// [0, 1]): `F_new⁻¹(u) = F⁻¹(g(u))`.
    fn remap(&self, g: &Calibration) -> Self {
        match self {
            Self::Steps { z, c } => {
                let c2: Vec<f64> = c.iter().map(|v| g.cdf(*v)).collect();
                Self::Steps {
                    z: z.clone(),
                    c: c2,
                }
            }
            Self::Grid { levels, q } => {
                let l2: Vec<f64> = levels.iter().map(|v| g.cdf(*v)).collect();
                Self::Grid {
                    levels: l2,
                    q: q.clone(),
                }
            }
        }
    }
}

/// Monotone level map from held-out PIT values: `forward(u)` is their
/// empirical CDF, so `forward(PIT)` is uniform.
#[derive(Debug, Clone)]
struct Calibration {
    pit: Vec<f64>,
}

impl Calibration {
    fn new(mut pit: Vec<f64>) -> Option<Self> {
        pit.retain(|v| v.is_finite());
        if pit.len() < 20 {
            return None;
        }
        pit.sort_by(f64::total_cmp);
        Some(Self { pit })
    }
    /// Empirical CDF of the PIT values (linear between order statistics).
    fn cdf(&self, u: f64) -> f64 {
        let n = self.pit.len() as f64;
        let k = self.pit.partition_point(|v| *v < u) as f64;
        let j = self.pit.partition_point(|v| *v <= u) as f64;
        (0.5 * (k + j) / n).clamp(0.0, 1.0)
    }
}

fn scaled(p: &[f64; 3], a: &[f64; 3]) -> [f64; 3] {
    [p[0] / a[0], p[1] / a[1], p[2] / a[2]]
}

fn d3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Data grouped for the embedded-model search.
struct Groups {
    id: Vec<i32>,
    centre: Vec<[f64; 3]>,
    /// Member rows per group, sorted by scaled z.
    rows: Vec<Vec<usize>>,
}

impl Groups {
    fn new(pts: &[[f64; 3]], groups: &[i32]) -> Self {
        Self::of_rows(pts, groups, 0..pts.len())
    }

    fn of_rows(pts: &[[f64; 3]], groups: &[i32], rows: impl Iterator<Item = usize>) -> Self {
        let mut map: std::collections::BTreeMap<i32, Vec<usize>> = Default::default();
        for i in rows {
            map.entry(groups[i]).or_default().push(i);
        }
        let (mut id, mut centre, mut rows) = (vec![], vec![], vec![]);
        for (g, mut r) in map {
            r.sort_by(|a, b| pts[*a][2].total_cmp(&pts[*b][2]));
            let n = r.len() as f64;
            let mut c = [0.0; 3];
            for &i in &r {
                for a in 0..3 {
                    c[a] += pts[i][a] / n;
                }
            }
            id.push(g);
            centre.push(c);
            rows.push(r);
        }
        Self { id, centre, rows }
    }

    /// Up to `k` neighbours of `t`, at most `per` per group, skipping group
    /// `skip` and row `skip_row`; (distance, row) sorted by distance.
    fn neighbours(
        &self,
        pts: &[[f64; 3]],
        t: &[f64; 3],
        k: usize,
        per: usize,
        skip: Option<i32>,
        skip_row: Option<usize>,
    ) -> Vec<(f64, usize)> {
        let h = |c: &[f64; 3]| ((c[0] - t[0]).powi(2) + (c[1] - t[1]).powi(2)).sqrt();
        let mut order: Vec<(f64, usize)> = (0..self.id.len())
            .filter(|&g| Some(self.id[g]) != skip)
            .map(|g| (h(&self.centre[g]), g))
            .collect();
        let want = k.div_ceil(per.max(1)) * 2;
        if order.len() > want {
            order.select_nth_unstable_by(want, |a, b| a.0.total_cmp(&b.0));
            order.truncate(want);
        }
        let mut out = vec![];
        for &(_, g) in &order {
            let r: Vec<usize> = self.rows[g]
                .iter()
                .copied()
                .filter(|i| Some(*i) != skip_row)
                .collect();
            let mut j = r.partition_point(|&i| pts[i][2] < t[2]);
            let mut i = j;
            let mut picked = vec![];
            while picked.len() < per && (i > 0 || j < r.len()) {
                let up = if i > 0 {
                    (t[2] - pts[r[i - 1]][2]).abs()
                } else {
                    f64::INFINITY
                };
                let dn = if j < r.len() {
                    (pts[r[j]][2] - t[2]).abs()
                } else {
                    f64::INFINITY
                };
                if up <= dn {
                    i -= 1;
                    picked.push(r[i]);
                } else {
                    picked.push(r[j]);
                    j += 1;
                }
            }
            out.extend(picked.into_iter().map(|q| (d3(&pts[q], t), q)));
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out.truncate(k);
        out
    }
}

/// Simple kriging weights and variance for neighbours `nb` (distance to
/// the target, index into `pts`).
fn sk_weights(pts: &[[f64; 3]], nb: &[(f64, usize)], c: &dyn Fn(f64) -> f64) -> (Vec<f64>, f64) {
    let n = nb.len();
    let c0 = c(0.0);
    if n == 0 {
        return (vec![], c0);
    }
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    for (r, &(dr, ir)) in nb.iter().enumerate() {
        for (cc, &(_, ic)) in nb.iter().enumerate() {
            a[r * n + cc] = c(d3(&pts[ir], &pts[ic]));
        }
        a[r * n + r] += 1e-9 * c0;
        b[r] = c(dr);
    }
    match solve(a, b.clone(), n) {
        Some(w) => {
            let v = c0 - (0..n).map(|r| w[r] * b[r]).sum::<f64>();
            (w, v.max(0.0))
        }
        None => (vec![0.0; n], c0),
    }
}

/// Simple kriging estimate and variance with covariance `c`, mean `m`.
fn simple_krige(
    pts: &[[f64; 3]],
    vals: &[f64],
    nb: &[(f64, usize)],
    c: &dyn Fn(f64) -> f64,
    m: f64,
) -> (f64, f64) {
    let (w, v) = sk_weights(pts, nb, c);
    let e = m + w
        .iter()
        .zip(nb)
        .map(|(w, (_, i))| w * (vals[*i] - m))
        .sum::<f64>();
    (e, v)
}

fn exp_cov(sill: f64, range: f64) -> impl Fn(f64) -> f64 {
    move |h: f64| sill * (-3.0 * h / range).exp()
}

/// Short and long embedded ranges: twice the median nearest group-centre
/// distance, and half the largest centre extent.
fn auto_ranges(g: &Groups) -> Vec<f64> {
    let c = &g.centre;
    let mut nn: Vec<f64> = (0..c.len())
        .map(|i| {
            (0..c.len())
                .filter(|&j| j != i)
                .map(|j| ((c[i][0] - c[j][0]).powi(2) + (c[i][1] - c[j][1]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
        })
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    nn.sort_by(f64::total_cmp);
    let ext = (0..2)
        .map(|a| {
            let (lo, hi) = c
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
                    (l.min(p[a]), h.max(p[a]))
                });
            hi - lo
        })
        .fold(0.0, f64::max);
    let short = nn.get(nn.len() / 2).copied().unwrap_or(1.0) * 2.0;
    let long = (0.5 * ext).max(2.0 * short);
    vec![short.max(1e-6), long.max(1e-6)]
}

fn check(inp: &EmberInput<'_>, p: &EmberParams) -> SxResult<()> {
    let n = inp.train_xyz.len();
    if n == 0 {
        return Err(SxError::EmptyInput("EMBER needs data".into()));
    }
    if inp.train_y.len() != n || inp.train_groups.len() != n || inp.train_secondary.len() != n {
        return Err(SxError::Shape("data columns differ in length".into()));
    }
    if inp.pred_secondary.len() != inp.pred_xyz.len() {
        return Err(SxError::Shape("target columns differ in length".into()));
    }
    let ns = inp.train_secondary[0].len();
    if inp
        .train_secondary
        .iter()
        .chain(inp.pred_secondary)
        .any(|r| r.len() != ns)
    {
        return Err(SxError::Shape(
            "secondary rows must share one length".into(),
        ));
    }
    if inp.train_y.iter().any(|v| !v.is_finite()) {
        return Err(SxError::InvalidInput(
            "EMBER data values must be finite".into(),
        ));
    }
    if p.anisotropy.iter().any(|a| !(*a > 0.0)) || p.max_neighbors == 0 || p.max_per_group == 0 {
        return Err(SxError::InvalidInput(
            "anisotropy > 0 and neighbour counts ≥ 1 are required".into(),
        ));
    }
    if p.embed_ranges.iter().any(|r| !(*r > 0.0)) {
        return Err(SxError::InvalidInput("embedded ranges must be > 0".into()));
    }
    if !p.use_coordinates && !p.embed && ns == 0 {
        return Err(SxError::InvalidInput(
            "EMBER needs coordinates, embedded models or secondary inputs".into(),
        ));
    }
    let q = &p.propagation.quantiles;
    if q.is_empty() || q.iter().any(|v| !(*v > 0.0 && *v < 1.0)) {
        return Err(SxError::InvalidInput("quantiles must lie in (0, 1)".into()));
    }
    Ok(())
}

/// Fitted envelope pieces shared by [`ember`] and [`ember_cv`].
struct Fitted {
    /// Envelope at the data (all data in the embedded models).
    data_env: Vec<Envelope>,
    /// Envelope at the targets.
    pred_env: Vec<Envelope>,
    /// Held-out envelope at the data (`None` where no tree left it out).
    held_env: Vec<Option<Envelope>>,
    /// Envelope at the data from training folds (empty unless `calibrate`
    /// or [`Spread::Auto`]).
    cv_env: Vec<Option<Envelope>>,
    embedded: Vec<Vec<f64>>,
    ranges: Vec<f64>,
}

/// LightGBM quantile grid with the tails clamped to the data range.
fn lgbm_envelopes(
    xt: &[Vec<f64>],
    y: &[f64],
    xp: &[Vec<f64>],
    pr: &PropagationParams,
) -> SxResult<Vec<Envelope>> {
    let levels: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
    let inner = &levels[1..levels.len() - 1];
    let lo = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(lgbm_quantiles(xt, y, xp, inner, pr)?
        .into_iter()
        .map(|mut r| {
            r.sort_by(f64::total_cmp);
            let mut q = vec![lo.min(r[0])];
            q.extend(r.iter().copied());
            q.push(hi.max(*r.last().unwrap_or(&hi)));
            Envelope::Grid {
                levels: levels.clone(),
                q,
            }
        })
        .collect())
}

/// Five folds of whole groups, or of x/y blocks of groups when `block > 0`:
/// (training rows, held-out rows).
fn group_folds(g: &Groups, gid: &[i32], seed: u64, block: f64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let key = |k: usize| -> (i64, i64) {
        if block > 0.0 {
            (
                (g.centre[k][0] / block).floor() as i64,
                (g.centre[k][1] / block).floor() as i64,
            )
        } else {
            (i64::from(g.id[k]), 0)
        }
    };
    let mut keys: Vec<(i64, i64)> = (0..g.id.len()).map(key).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xf01d);
    for i in (1..keys.len()).rev() {
        let j = rng.random_range(0..=i);
        keys.swap(i, j);
    }
    let kf: std::collections::HashMap<(i64, i64), usize> =
        keys.iter().enumerate().map(|(k, c)| (*c, k % 5)).collect();
    let fold: std::collections::HashMap<i32, usize> =
        (0..g.id.len()).map(|k| (g.id[k], kf[&key(k)])).collect();
    (0..5.min(keys.len()))
        .filter_map(|f| {
            let (te, tr): (Vec<usize>, Vec<usize>) =
                (0..gid.len()).partition(|i| fold[&gid[*i]] == f);
            (!te.is_empty() && !tr.is_empty()).then_some((tr, te))
        })
        .collect()
}

fn fit_envelope(inp: &EmberInput<'_>, p: &EmberParams) -> SxResult<Fitted> {
    let a = p.anisotropy;
    let tp: Vec<[f64; 3]> = inp.train_xyz.iter().map(|q| scaled(q, &a)).collect();
    let pp: Vec<[f64; 3]> = inp.pred_xyz.iter().map(|q| scaled(q, &a)).collect();
    let y = inp.train_y;
    let gid = inp.train_groups;
    let n = y.len();
    let mean = y.iter().sum::<f64>() / n as f64;
    let var = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).max(1e-12);
    let all = Groups::new(&tp, gid);
    let ranges = if !p.embed {
        vec![]
    } else if p.embed_ranges.is_empty() {
        auto_ranges(&all)
    } else {
        p.embed_ranges.clone()
    };
    let covs: Vec<_> = ranges.iter().map(|&r| exp_cov(var, r)).collect();
    // Embedded estimates at point `t` from the data in `g`.
    let embed =
        |g: &Groups, t: &[f64; 3], skip: Option<i32>, skip_row: Option<usize>| -> Vec<f64> {
            if covs.is_empty() {
                return vec![];
            }
            let nb = g.neighbours(&tp, t, p.max_neighbors, p.max_per_group, skip, skip_row);
            covs.iter()
                .map(|c| simple_krige(&tp, y, &nb, c, mean).0)
                .collect()
        };
    let holdout = |i: usize| -> (Option<i32>, Option<usize>) {
        match p.embed_holdout {
            EmbedHoldout::Sample => (None, Some(i)),
            EmbedHoldout::Group => (Some(gid[i]), None),
        }
    };
    let (emb_d, emb_p): (Vec<Vec<f64>>, Vec<Vec<f64>>) = crate::runtime::install(|| {
        (
            tp.par_iter().map(|t| embed(&all, t, None, None)).collect(),
            pp.par_iter().map(|t| embed(&all, t, None, None)).collect(),
        )
    });
    let base = |xyz: &[f64; 3], sec: &[f64]| -> Vec<f64> {
        let mut r = vec![];
        if p.use_coordinates {
            r.extend_from_slice(xyz);
        }
        r.extend_from_slice(sec);
        r
    };
    let join = |mut b: Vec<f64>, e: &[f64]| -> Vec<f64> {
        b.extend_from_slice(e);
        b
    };
    let xb: Vec<Vec<f64>> = (0..n)
        .map(|i| base(&tp[i], &inp.train_secondary[i]))
        .collect();
    let xd: Vec<Vec<f64>> = (0..n).map(|i| join(xb[i].clone(), &emb_d[i])).collect();
    let xp: Vec<Vec<f64>> = (0..pp.len())
        .map(|i| join(base(&pp[i], &inp.pred_secondary[i]), &emb_p[i]))
        .collect();
    let pr = &p.propagation;
    let (data_env, pred_env, held_env, cv_env) = match pr.model {
        PropagationModel::QuantileForest => {
            let qp = QrfParams {
                seed: pr.seed,
                random_split: p.random_split,
                ..pr.qrf
            };
            let per_tree = |bag: &[u32]| -> Vec<Vec<f64>> {
                let mut u: Vec<usize> = bag.iter().map(|&r| r as usize).collect();
                u.sort_unstable();
                u.dedup();
                let g = Groups::of_rows(&tp, gid, u.into_iter());
                (0..n)
                    .map(|i| {
                        let (s, r) = holdout(i);
                        embed(&g, &tp[i], s, r)
                    })
                    .collect()
            };
            let f = if ranges.is_empty() {
                QuantileForest::fit(&xb, y, &qp)?
            } else {
                QuantileForest::fit_embedded(&xb, y, &qp, ranges.len(), &per_tree)?
            };
            let env = |w: Vec<Vec<(f64, f64)>>| -> Vec<Option<Envelope>> {
                w.iter().map(|w| Envelope::from_weights(w)).collect()
            };
            let some = |v: Vec<Option<Envelope>>| -> SxResult<Vec<Envelope>> {
                v.into_iter()
                    .map(|e| e.ok_or_else(|| SxError::Model("empty forest envelope".into())))
                    .collect()
            };
            let held = env(f.predict_weights_oob(&xb)?);
            let mut cv: Vec<Option<Envelope>> = vec![];
            if p.calibrate || (p.spread.fitted() && p.n_realizations > 0) {
                // Held-out envelopes from training folds of whole groups
                // (or blocks), as far from their training data as at
                // prediction.
                cv = vec![None; n];
                for (tr, te) in group_folds(&all, gid, pr.seed, p.calib_block) {
                    let per_tree_f = |bag: &[u32]| -> Vec<Vec<f64>> {
                        let g = Groups::of_rows(&tp, gid, bag.iter().map(|&r| tr[r as usize]));
                        tr.iter()
                            .map(|&i| {
                                let (s, r) = holdout(i);
                                embed(&g, &tp[i], s, r)
                            })
                            .collect()
                    };
                    let xbt: Vec<Vec<f64>> = tr.iter().map(|&i| xb[i].clone()).collect();
                    let yt: Vec<f64> = tr.iter().map(|&i| y[i]).collect();
                    let ff = if ranges.is_empty() {
                        QuantileForest::fit(&xbt, &yt, &qp)?
                    } else {
                        QuantileForest::fit_embedded(&xbt, &yt, &qp, ranges.len(), &per_tree_f)?
                    };
                    let gt = Groups::of_rows(&tp, gid, tr.iter().copied());
                    let xe: Vec<Vec<f64>> = te
                        .iter()
                        .map(|&i| join(xb[i].clone(), &embed(&gt, &tp[i], None, None)))
                        .collect();
                    for (k, e) in env(ff.predict_weights(&xe)?).into_iter().enumerate() {
                        cv[te[k]] = e;
                    }
                }
            }
            (
                some(env(f.predict_weights(&xd)?))?,
                some(env(f.predict_weights(&xp)?))?,
                held,
                cv,
            )
        }
        PropagationModel::LightGbm => {
            // Embedded estimates computed once with the hold-out rule.
            let emb_t: Vec<Vec<f64>> = crate::runtime::install(|| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let (s, r) = holdout(i);
                        embed(&all, &tp[i], s, r)
                    })
                    .collect()
            });
            let xt: Vec<Vec<f64>> = (0..n).map(|i| join(xb[i].clone(), &emb_t[i])).collect();
            let both: Vec<Vec<f64>> = xd.iter().chain(&xp).cloned().collect();
            let mut envs = lgbm_envelopes(&xt, y, &both, pr)?;
            let pred = envs.split_off(n);
            // Held-out envelopes from five group folds.
            let mut held: Vec<Option<Envelope>> = vec![None; n];
            for (tr, te) in group_folds(&all, gid, pr.seed, p.calib_block) {
                let sub = |ix: &[usize]| ix.iter().map(|&i| xt[i].clone()).collect::<Vec<_>>();
                let ytr: Vec<f64> = tr.iter().map(|&i| y[i]).collect();
                let e = lgbm_envelopes(&sub(&tr), &ytr, &sub(&te), pr)?;
                for (k, &i) in te.iter().enumerate() {
                    held[i] = Some(e[k].clone());
                }
            }
            let cv = held.clone();
            (envs, pred, held, cv)
        }
    };
    let (data_env, pred_env) = if p.calibrate {
        let pit: Vec<f64> = cv_env
            .iter()
            .zip(y)
            .map(|(e, z)| {
                e.as_ref().map_or(f64::NAN, |e| {
                    let (u0, u1) = e.invert(*z);
                    0.5 * (u0 + u1)
                })
            })
            .collect();
        match Calibration::new(pit) {
            Some(c) => (
                data_env.iter().map(|e| e.remap(&c)).collect(),
                pred_env.iter().map(|e| e.remap(&c)).collect(),
            ),
            None => (data_env, pred_env),
        }
    } else {
        (data_env, pred_env)
    };
    Ok(Fitted {
        data_env,
        pred_env,
        held_env,
        cv_env,
        embedded: emb_p,
        ranges,
    })
}

/// Sampling variogram (Daly 2022, eq. 3): exponential model fitted to the
/// rescaled residuals `(z − m) / σ` under the held-out envelopes, scaled to
/// sill 1.
fn sampling_variogram(
    tp: &[[f64; 3]],
    y: &[f64],
    held: &[Option<Envelope>],
    seed: u64,
) -> SxResult<VariogramModel> {
    let r: Vec<f64> = held
        .iter()
        .zip(y)
        .map(|(e, z)| match e {
            Some(e) => {
                let (m, s) = e.moments();
                if s > 0.0 {
                    (z - m) / s
                } else {
                    f64::NAN
                }
            }
            None => f64::NAN,
        })
        .collect();
    let mut idx: Vec<usize> = (0..r.len()).filter(|&i| r[i].is_finite()).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
    for i in 0..idx.len().min(3000) {
        let j = rng.random_range(i..idx.len());
        idx.swap(i, j);
    }
    idx.truncate(3000);
    if idx.len() < 10 {
        return Err(SxError::EmptyInput(
            "too few data for the sampling variogram".into(),
        ));
    }
    let pts: Vec<[f64; 3]> = idx.iter().map(|&i| tp[i]).collect();
    let v: Vec<f64> = idx.iter().map(|&i| r[i]).collect();
    let mut ext: f64 = 0.0;
    for p in &pts {
        for q in pts.iter().take(200) {
            ext = ext.max(d3(p, q));
        }
    }
    let bins = experimental_variogram(&pts, &v, (ext / 2.0 / 15.0).max(1e-9), 15)?;
    let mut m = fit_variogram(&bins, ModelKind::Exponential)?;
    let s = m.nugget + m.psill;
    if s > 0.0 {
        m.nugget /= s;
        m.psill /= s;
    } else {
        m = VariogramModel {
            kind: ModelKind::Exponential,
            nugget: 0.0,
            psill: 1.0,
            range: m.range.max(1e-6),
        };
    }
    Ok(m)
}

/// Draw from N(m, s²) truncated to [a, b].
fn trunc_normal(m: f64, s: f64, a: f64, b: f64, rng: &mut ChaCha8Rng) -> f64 {
    if s < 1e-12 || b - a < 1e-12 {
        return m.clamp(a, b);
    }
    let (pa, pb) = (norm_cdf((a - m) / s), norm_cdf((b - m) / s));
    if pb - pa < 1e-12 {
        return if m < a { a } else { b };
    }
    (m + s * norm_inv((pa + rng.random::<f64>() * (pb - pa)).clamp(1e-12, 1.0 - 1e-12))).clamp(a, b)
}

/// Gibbs sampler for the Gaussian values at the data, each restricted to
/// its interval (Freulon 1992). Conditionals use simple kriging from the
/// `k` nearest other data (precomputed weights).
struct Gibbs {
    nb: Vec<Vec<usize>>,
    w: Vec<Vec<f64>>,
    s: Vec<f64>,
}

impl Gibbs {
    fn new(tp: &[[f64; 3]], vg: &VariogramModel, k: usize, cell: f64) -> Self {
        let mut b = Buckets::new(cell);
        for q in tp {
            b.push(*q);
        }
        let cov = |h: f64| vg.covariance(h);
        let rows: Vec<(Vec<usize>, Vec<f64>, f64)> = crate::runtime::install(|| {
            (0..tp.len())
                .into_par_iter()
                .map(|i| {
                    let nb: Vec<(f64, usize)> = b
                        .nearest(&tp[i], k + 1, f64::INFINITY)
                        .into_iter()
                        .filter(|x| x.1 != i)
                        .take(k)
                        .collect();
                    let (w, v) = sk_weights(tp, &nb, &cov);
                    (nb.iter().map(|x| x.1).collect(), w, v.sqrt())
                })
                .collect()
        });
        let (mut nb, mut w, mut s) = (vec![], vec![], vec![]);
        for (a, b, c) in rows {
            nb.push(a);
            w.push(b);
            s.push(c);
        }
        Self { nb, w, s }
    }

    fn sample(&self, bounds: &[(f64, f64)], sweeps: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut x: Vec<f64> = bounds
            .iter()
            .map(|&(a, b)| trunc_normal(0.0, 1.0, a, b, rng))
            .collect();
        for _ in 0..sweeps {
            for i in random_path(x.len(), rng) {
                let m: f64 = self.nb[i]
                    .iter()
                    .zip(&self.w[i])
                    .map(|(j, w)| w * x[*j])
                    .sum();
                x[i] = trunc_normal(m, self.s[i], bounds[i].0, bounds[i].1, rng);
            }
        }
        x
    }
}

/// Spread factor from training folds: with Gibbs values at the data, the
/// simple-kriging mean m and variance v of the Gaussian at each held-out
/// datum from the other folds give r = |Phi^-1(PIT) - m| / sqrt(v); the
/// factor is the 0.8 quantile of r over Phi^-1(0.9), so that P10-P90 of
/// the realizations covers 80 % of the held-out data.
#[allow(clippy::too_many_arguments)]
fn auto_spread(
    tp: &[[f64; 3]],
    y: &[f64],
    cv_env: &[Option<Envelope>],
    folds: &[(Vec<usize>, Vec<usize>)],
    gibbs: &Gibbs,
    bounds: &[(f64, f64)],
    vg: &VariogramModel,
    k: usize,
    cell: f64,
    p: &EmberParams,
    seed: u64,
) -> f64 {
    let cov = |h: f64| vg.covariance(h);
    let lim = norm_inv(1e-6).abs();
    let mut r: Vec<f64> = vec![];
    for d in 0..3u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5bd0 ^ d);
        let xd = gibbs.sample(bounds, p.gibbs_sweeps, &mut rng);
        for (tr, te) in folds {
            let mut b = Buckets::new(cell);
            for &i in tr {
                b.push(tp[i]);
            }
            let pts: Vec<[f64; 3]> = tr.iter().map(|&i| tp[i]).collect();
            let vals: Vec<f64> = tr.iter().map(|&i| xd[i]).collect();
            for &i in te {
                let Some(e) = cv_env.get(i).and_then(Option::as_ref) else {
                    continue;
                };
                let (u0, u1) = e.invert(y[i]);
                let u = (0.5 * (u0 + u1)).clamp(1e-6, 1.0 - 1e-6);
                let g = norm_inv(u).clamp(-lim, lim);
                let nb = b.nearest(&tp[i], k, f64::INFINITY);
                let (m, v) = simple_krige(&pts, &vals, &nb, &cov, 0.0);
                if v > 1e-10 {
                    r.push((g - m).abs() / v.sqrt());
                }
            }
        }
    }
    if r.len() < 10 {
        return 1.0;
    }
    r.sort_by(f64::total_cmp);
    let q = r[((r.len() - 1) as f64 * 0.8).round() as usize];
    (q / norm_inv(0.9)).clamp(0.5, 4.0)
}

/// Gaussian values at the targets by sequential simulation conditioned on
/// the data values `xd`.
#[allow(clippy::too_many_arguments)]
fn simulate_targets(
    tp: &[[f64; 3]],
    xd: &[f64],
    pp: &[[f64; 3]],
    vg: &VariogramModel,
    k: usize,
    cell: f64,
    spread: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let cov = |h: f64| vg.covariance(h);
    let mut b = Buckets::new(cell);
    let mut pts: Vec<[f64; 3]> = tp.to_vec();
    let mut vals: Vec<f64> = xd.to_vec();
    for q in tp {
        b.push(*q);
    }
    let mut out = vec![0.0; pp.len()];
    for i in random_path(pp.len(), rng) {
        let nb = b.nearest(&pp[i], k, f64::INFINITY);
        let (m, v) = simple_krige(&pts, &vals, &nb, &cov, 0.0);
        let x = if v < 1e-10 {
            m
        } else {
            m + spread * v.sqrt() * gauss(rng)
        };
        out[i] = x;
        b.push(pp[i]);
        pts.push(pp[i]);
        vals.push(x);
    }
    out
}

/// Run EMBER: envelope quantiles, mean and std at the targets, and
/// `n_realizations` conditional realizations.
///
/// # Errors
/// [`SxError::Shape`] / [`SxError::InvalidInput`] / [`SxError::EmptyInput`]
/// on bad input; [`SxError::Model`] if LightGBM fails.
pub fn ember(inp: &EmberInput<'_>, p: &EmberParams) -> SxResult<EmberResult> {
    check(inp, p)?;
    let f = fit_envelope(inp, p)?;
    let a = p.anisotropy;
    let tp: Vec<[f64; 3]> = inp.train_xyz.iter().map(|q| scaled(q, &a)).collect();
    let pp: Vec<[f64; 3]> = inp.pred_xyz.iter().map(|q| scaled(q, &a)).collect();
    let quantiles = f
        .pred_env
        .iter()
        .map(|e| {
            p.propagation
                .quantiles
                .iter()
                .map(|&u| e.quantile(u))
                .collect()
        })
        .collect();
    let (mean, std): (Vec<f64>, Vec<f64>) = f.pred_env.iter().map(Envelope::moments).unzip();
    let vg = match p.sampling_variogram {
        Some(v) => v,
        None => sampling_variogram(&tp, inp.train_y, &f.held_env, p.propagation.seed)?,
    };
    let mut realizations = vec![];
    if p.n_realizations > 0 {
        let lim = norm_inv(1e-6).abs();
        let bounds: Vec<(f64, f64)> = f
            .data_env
            .iter()
            .zip(inp.train_y)
            .map(|(e, z)| {
                let (u0, u1) = e.invert(*z);
                let g = |u: f64| {
                    if u <= 0.0 {
                        -lim
                    } else if u >= 1.0 {
                        lim
                    } else {
                        norm_inv(u).clamp(-lim, lim)
                    }
                };
                (g(u0), g(u1))
            })
            .collect();
        let all: Vec<[f64; 3]> = tp.iter().chain(&pp).copied().collect();
        let cell = auto_cell(&all);
        let k = p.sim_neighbors.max(1);
        let gibbs = Gibbs::new(&tp, &vg, k, cell);
        let seed = p.propagation.seed;
        let spread = match p.spread {
            Spread::None => 1.0,
            Spread::Fixed(v) => v,
            Spread::Auto | Spread::Inflate => {
                let g = Groups::new(&tp, inp.train_groups);
                let folds = group_folds(&g, inp.train_groups, seed, p.calib_block);
                auto_spread(
                    &tp,
                    inp.train_y,
                    &f.cv_env,
                    &folds,
                    &gibbs,
                    &bounds,
                    &vg,
                    k,
                    cell,
                    p,
                    seed,
                )
                .max(if p.spread == Spread::Inflate {
                    1.0
                } else {
                    0.0
                })
            }
        };
        realizations = crate::runtime::install(|| {
            (0..p.n_realizations)
                .into_par_iter()
                .map(|r| {
                    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1 + r as u64));
                    let xd = gibbs.sample(&bounds, p.gibbs_sweeps, &mut rng);
                    let x = simulate_targets(&tp, &xd, &pp, &vg, k, cell, spread, &mut rng);
                    x.iter()
                        .zip(&f.pred_env)
                        .map(|(g, e)| e.quantile(norm_cdf(*g)))
                        .collect()
                })
                .collect()
        });
    }
    Ok(EmberResult {
        quantiles,
        mean,
        std,
        realizations,
        embedded: f.embedded,
        sampling_variogram: vg,
        embed_ranges: f.ranges,
    })
}

/// Blocked cross-validation scores of one method.
#[derive(Debug, Clone, PartialEq)]
pub struct CvScore {
    /// "ember", "ember_realizations" or "ml_propagation".
    pub method: String,
    /// Mean absolute error of the point estimate (envelope mean, mean of the
    /// realizations, or P50).
    pub mae: f64,
    /// Root mean square error of the point estimate.
    pub rmse: f64,
    /// Share of held-out values inside the P10–P90 band.
    pub coverage_p10_p90: f64,
    /// Mean P10–P90 width.
    pub width_p10_p90: f64,
    /// Held-out rows scored.
    pub n: usize,
}

/// Blocked spatial cross-validation of EMBER against plain
/// `ml_propagation` (same coordinates and secondary inputs, no embedded
/// models). Groups are assigned to folds by square map blocks of side
/// `block` (scaled units; 0 = one block per group), so whole wells and
/// their close neighbours are held out together.
///
/// # Errors
/// As [`ember`]; [`SxError::InvalidInput`] for fewer than two folds.
pub fn ember_cv(
    xyz: &[[f64; 3]],
    y: &[f64],
    groups: &[i32],
    secondary: &[Vec<f64>],
    p: &EmberParams,
    n_folds: usize,
    block: f64,
) -> SxResult<Vec<CvScore>> {
    let empty: Vec<Vec<f64>> = vec![];
    check(
        &EmberInput {
            train_xyz: xyz,
            train_y: y,
            train_groups: groups,
            train_secondary: secondary,
            pred_xyz: &[],
            pred_secondary: &empty,
        },
        p,
    )?;
    let a = p.anisotropy;
    let tp: Vec<[f64; 3]> = xyz.iter().map(|q| scaled(q, &a)).collect();
    let g = Groups::new(&tp, groups);
    let mut blocks: std::collections::BTreeMap<(i64, i64), Vec<i32>> = Default::default();
    for (i, c) in g.centre.iter().enumerate() {
        let key = if block > 0.0 {
            ((c[0] / block).floor() as i64, (c[1] / block).floor() as i64)
        } else {
            (i as i64, 0)
        };
        blocks.entry(key).or_default().push(g.id[i]);
    }
    if n_folds < 2 || n_folds > blocks.len() {
        return Err(SxError::InvalidInput(format!(
            "n_folds must be in 2..={}",
            blocks.len()
        )));
    }
    let mut keys: Vec<(i64, i64)> = blocks.keys().copied().collect();
    let mut rng = ChaCha8Rng::seed_from_u64(p.propagation.seed ^ 0xc0ff);
    for i in (1..keys.len()).rev() {
        let j = rng.random_range(0..=i);
        keys.swap(i, j);
    }
    let mut fold_of: std::collections::HashMap<i32, usize> = Default::default();
    for (i, k) in keys.iter().enumerate() {
        for gid in &blocks[k] {
            fold_of.insert(*gid, i % n_folds);
        }
    }
    let qs = [0.1, 0.5, 0.9];
    let mut pe = p.clone();
    pe.propagation.quantiles = qs.to_vec();
    if pe.calib_block <= 0.0 {
        pe.calib_block = block;
    }
    // (estimate, p10, p90, truth) per method
    let mut acc: [Vec<(f64, f64, f64, f64)>; 3] = [vec![], vec![], vec![]];
    for f in 0..n_folds {
        let (te, tr): (Vec<usize>, Vec<usize>) =
            (0..y.len()).partition(|&i| fold_of[&groups[i]] == f);
        if te.is_empty() || tr.is_empty() {
            continue;
        }
        let pick3 = |ix: &[usize]| ix.iter().map(|&i| xyz[i]).collect::<Vec<_>>();
        let pick = |ix: &[usize]| ix.iter().map(|&i| secondary[i].clone()).collect::<Vec<_>>();
        let (trx, tex) = (pick3(&tr), pick3(&te));
        let (trs, tes) = (pick(&tr), pick(&te));
        let try_: Vec<f64> = tr.iter().map(|&i| y[i]).collect();
        let trg: Vec<i32> = tr.iter().map(|&i| groups[i]).collect();
        let r = ember(
            &EmberInput {
                train_xyz: &trx,
                train_y: &try_,
                train_groups: &trg,
                train_secondary: &trs,
                pred_xyz: &tex,
                pred_secondary: &tes,
            },
            &pe,
        )?;
        for (j, &i) in te.iter().enumerate() {
            acc[0].push((r.mean[j], r.quantiles[j][0], r.quantiles[j][2], y[i]));
            if !r.realizations.is_empty() {
                let mut v: Vec<f64> = r.realizations.iter().map(|s| s[j]).collect();
                v.sort_by(f64::total_cmp);
                // Order-statistic quantile at position u (n + 1) (Hyndman-Fan type 6),
                // so a calibrated ensemble covers P10-P90 80 % of the time.
                let at = |u: f64| {
                    let x = (u * (v.len() + 1) as f64 - 1.0).clamp(0.0, (v.len() - 1) as f64);
                    let (i, f) = (x.floor() as usize, x.fract());
                    v[i] + f * (v[(i + 1).min(v.len() - 1)] - v[i])
                };
                let m = v.iter().sum::<f64>() / v.len() as f64;
                acc[1].push((m, at(0.1), at(0.9), y[i]));
            }
        }
        // Plain ml_propagation on coordinates + secondary.
        let feat = |x: &[[f64; 3]], s: &[Vec<f64>]| -> Vec<Vec<f64>> {
            x.iter()
                .zip(s)
                .map(|(q, s)| {
                    let mut r = scaled(q, &a).to_vec();
                    r.extend_from_slice(s);
                    r
                })
                .collect()
        };
        let mut pp = p.propagation.clone();
        pp.quantiles = qs.to_vec();
        let q = super::propagation::ml_propagation_grouped(
            &feat(&trx, &trs),
            &try_,
            Some(&trg),
            &feat(&tex, &tes),
            &pp,
        )?;
        for (j, &i) in te.iter().enumerate() {
            acc[2].push((q[j][1], q[j][0], q[j][2], y[i]));
        }
    }
    let names = ["ember", "ember_realizations", "ml_propagation"];
    Ok(acc
        .iter()
        .zip(names)
        .filter(|(v, _)| !v.is_empty())
        .map(|(v, name)| {
            let n = v.len() as f64;
            CvScore {
                method: name.to_string(),
                mae: v.iter().map(|r| (r.0 - r.3).abs()).sum::<f64>() / n,
                rmse: (v.iter().map(|r| (r.0 - r.3).powi(2)).sum::<f64>() / n).sqrt(),
                coverage_p10_p90: v.iter().filter(|r| r.3 >= r.1 && r.3 <= r.2).count() as f64 / n,
                width_p10_p90: v.iter().map(|r| r.2 - r.1).sum::<f64>() / n,
                n: v.len(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wells on a 10 × 10 grid sampling a smooth field plus noise.
    fn field(n_wells: usize, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>, Vec<i32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (mut p, mut y, mut g) = (vec![], vec![], vec![]);
        for w in 0..n_wells {
            let (x0, y0) = (rng.random::<f64>() * 1000.0, rng.random::<f64>() * 1000.0);
            for k in 0..20 {
                let z = k as f64;
                p.push([x0, y0, z]);
                y.push((x0 / 200.0).sin() + (y0 / 300.0).cos() + 0.05 * z + 0.2 * gauss(&mut rng));
                g.push(w as i32);
            }
        }
        (p, y, g)
    }

    fn params() -> EmberParams {
        EmberParams {
            propagation: PropagationParams {
                qrf: QrfParams {
                    n_trees: 60,
                    min_leaf: 5,
                    ..QrfParams::default()
                },
                ..PropagationParams::default()
            },
            anisotropy: [1.0, 1.0, 0.1],
            embed_holdout: EmbedHoldout::Group,
            ..EmberParams::default()
        }
    }

    #[test]
    fn envelope_inverts_and_has_moments() {
        let lv: Vec<f64> = (0..=4).map(|i| i as f64 / 4.0).collect();
        let e = Envelope::Grid {
            levels: lv,
            q: vec![0.0, 1.0, 1.0, 2.0, 4.0],
        };
        assert_eq!(e.quantile(0.125), 0.5);
        assert_eq!(e.invert(0.5), (0.125, 0.125));
        assert_eq!(e.invert(1.0), (0.25, 0.5));
        assert_eq!(e.invert(9.0).1, 1.0);
        let (m, s) = e.moments();
        assert!((m - 1.5).abs() < 1e-12 && s > 0.0);
        // Forest weights: F(z) = Σ ω 1[Z ≤ z].
        let f =
            Envelope::from_weights(&[(1.0, 0.25), (2.0, 0.25), (2.0, 0.25), (5.0, 0.25)]).unwrap();
        assert_eq!(f.invert(2.0), (0.25, 0.75));
        assert_eq!(f.invert(3.0), (0.75, 0.75));
        assert_eq!(f.quantile(0.5), 2.0);
        assert_eq!(f.quantile(0.8), 5.0);
        assert!((f.moments().0 - 2.5).abs() < 1e-12);
        let c = Calibration::new((0..100).map(|i| 0.5 + 0.004 * i as f64).collect()).unwrap();
        assert!(c.cdf(0.4) == 0.0 && c.cdf(0.95) == 1.0);
    }

    #[test]
    fn realizations_honour_data_and_cv_runs() {
        let (p, y, g) = field(40, 1);
        let sec = vec![vec![]; y.len()];
        let targets: Vec<[f64; 3]> = p.iter().step_by(7).copied().collect();
        let tsec = vec![vec![]; targets.len()];
        let r = ember(
            &EmberInput {
                train_xyz: &p,
                train_y: &y,
                train_groups: &g,
                train_secondary: &sec,
                pred_xyz: &targets,
                pred_secondary: &tsec,
            },
            &EmberParams {
                n_realizations: 4,
                ..params()
            },
        )
        .unwrap();
        assert_eq!(r.realizations.len(), 4);
        assert_eq!(r.embed_ranges.len(), 2);
        // Targets on data points get the datum back in every realization.
        for real in &r.realizations {
            for (j, t) in targets.iter().enumerate() {
                let i = p.iter().position(|q| q == t).unwrap();
                assert_eq!(real[j], y[i]);
            }
        }
        assert!(r.quantiles.iter().all(|q| q[0] <= q[1] && q[1] <= q[2]));
        assert!((0.0..=1.0)
            .contains(&(r.sampling_variogram.nugget + r.sampling_variogram.psill - 1.0).abs()));
        let cv = ember_cv(
            &p,
            &y,
            &g,
            &sec,
            &EmberParams {
                n_realizations: 5,
                ..params()
            },
            4,
            0.0,
        )
        .unwrap();
        let get = |m: &str| cv.iter().find(|s| s.method == m).unwrap();
        let (e, ml) = (get("ember"), get("ml_propagation"));
        // Embedded kriging helps on a smooth field sampled by wells.
        assert!(e.mae < ml.mae, "ember {} vs ml {}", e.mae, ml.mae);
        assert!(
            (0.6..0.97).contains(&e.coverage_p10_p90),
            "coverage {}",
            e.coverage_p10_p90
        );
        assert!(get("ember_realizations").n == e.n);
    }

    #[test]
    fn spread_inflation_keeps_data_and_finite() {
        let (p, y, g) = field(40, 1);
        let sec = vec![vec![]; y.len()];
        let targets: Vec<[f64; 3]> = p.iter().step_by(5).copied().collect();
        let tsec = vec![vec![]; targets.len()];
        for spread in [Spread::Auto, Spread::Inflate, Spread::Fixed(1.5)] {
            let r = ember(
                &EmberInput {
                    train_xyz: &p,
                    train_y: &y,
                    train_groups: &g,
                    train_secondary: &sec,
                    pred_xyz: &targets,
                    pred_secondary: &tsec,
                },
                &EmberParams {
                    n_realizations: 3,
                    spread,
                    calib_block: 0.0,
                    ..params()
                },
            )
            .unwrap();
            for real in &r.realizations {
                assert!(real.iter().all(|v| v.is_finite()));
                for (j, t) in targets.iter().enumerate() {
                    let i = p.iter().position(|q| q == t).unwrap();
                    assert_eq!(real[j], y[i]);
                }
            }
        }
    }

    #[test]
    fn bad_input_is_rejected() {
        let p = [[0.0; 3]];
        let s = vec![vec![]];
        let inp = EmberInput {
            train_xyz: &p,
            train_y: &[f64::NAN],
            train_groups: &[0],
            train_secondary: &s,
            pred_xyz: &p,
            pred_secondary: &s,
        };
        assert!(ember(&inp, &EmberParams::default()).is_err());
    }
}
