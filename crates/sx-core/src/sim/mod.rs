// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stochastic property modelling.
//!
//! - [`nscore`]: normal-score transform and back-transform.
//! - [`sgs`]: sequential Gaussian simulation (SGS) of a continuous property
//!   (porosity, or permeability as log10), ordinary-kriging based.
//! - [`sis`]: sequential indicator simulation (SIS) of facies codes.
//! - [`qrf`]: quantile regression forest (Meinshausen 2006), written in house.
//! - [`propagation`]: `ml_propagation`: quantile models (quantile forest
//!   by default, LightGBM as option) from location/stratigraphic features
//!   to P10/P50/P90 at target points.
//! - [`ember`]: EMBER: embedded kriging inputs, a quantile-forest envelope,
//!   well conditioning, spatially correlated realizations and blocked
//!   cross-validation.
//!
//! Limitations for the whole module: neighbour search is brute
//! force; anisotropy only by scaling coordinates before the call; no
//! multiple-grid search, no collocated secondary data, no trend in SGS.

pub mod ember;
pub mod nscore;
pub mod propagation;
pub mod qrf;
pub mod sgs;
pub mod sis;

pub use ember::{
    ember, ember_cv, CvScore, EmbedHoldout, EmberInput, EmberParams, EmberResult, Spread,
};
pub use nscore::{norm_cdf, norm_inv, NormalScore};
pub use propagation::{
    conformal_offsets, ml_propagation, ml_propagation_grouped, PropagationModel, PropagationParams,
};
pub use qrf::{QrfParams, QuantileForest};
pub use sgs::{sgs, SgsParams};
pub use sis::{sis, SisParams};

use rand::RngExt;
use rand_chacha::ChaCha8Rng;

/// Standard normal draw (Box–Muller).
pub(crate) fn gauss(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Random visiting order of `n` targets (Fisher–Yates).
pub(crate) fn random_path(n: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        p.swap(i, j);
    }
    p
}

pub(crate) fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Bucketed neighbour index over a growing point set (cubic cells of side
/// `cell`); `nearest` widens the search ring until `k` points are found and
/// the ring radius covers the k-th distance.
pub(crate) struct Buckets {
    cell: f64,
    map: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
    pts: Vec<[f64; 3]>,
    lo: [i64; 3],
    hi: [i64; 3],
}

impl Buckets {
    pub(crate) fn new(cell: f64) -> Self {
        Self {
            cell: cell.max(1e-9),
            map: Default::default(),
            pts: vec![],
            lo: [i64::MAX; 3],
            hi: [i64::MIN; 3],
        }
    }
    fn key(&self, p: &[f64; 3]) -> (i64, i64, i64) {
        let f = |v: f64| (v / self.cell).floor() as i64;
        (f(p[0]), f(p[1]), f(p[2]))
    }
    /// Add a point; returns its index.
    pub(crate) fn push(&mut self, p: [f64; 3]) -> usize {
        let k = self.key(&p);
        for (a, v) in [k.0, k.1, k.2].into_iter().enumerate() {
            self.lo[a] = self.lo[a].min(v);
            self.hi[a] = self.hi[a].max(v);
        }
        self.pts.push(p);
        self.map.entry(k).or_default().push(self.pts.len() - 1);
        self.pts.len() - 1
    }
    pub(crate) fn point(&self, i: usize) -> [f64; 3] {
        self.pts[i]
    }
    /// Up to `k` nearest (distance, index) within `dmax`, sorted.
    pub(crate) fn nearest(&self, t: &[f64; 3], k: usize, dmax: f64) -> Vec<(f64, usize)> {
        if self.pts.is_empty() || k == 0 {
            return vec![];
        }
        let c = self.key(t);
        let span = (0..3)
            .map(|a| (self.hi[a] - self.lo[a]).max(0))
            .max()
            .unwrap_or(0)
            + [c.0, c.1, c.2]
                .iter()
                .enumerate()
                .map(|(a, v)| (v - self.lo[a]).abs().max((v - self.hi[a]).abs()))
                .max()
                .unwrap_or(0);
        let mut found: Vec<(f64, usize)> = vec![];
        let mut r: i64 = 0;
        loop {
            // Visit the shell at Chebyshev radius r.
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        if let Some(v) = self.map.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                            for &i in v {
                                let d = dist(&self.pts[i], t);
                                if d <= dmax {
                                    found.push((d, i));
                                }
                            }
                        }
                    }
                }
            }
            // Everything within r·cell of t has been seen.
            let covered = r as f64 * self.cell;
            if found.len() >= k {
                found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                if found[k - 1].0 <= covered {
                    found.truncate(k);
                    return found;
                }
            }
            if r > span || covered > dmax {
                found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                found.truncate(k);
                return found;
            }
            r += 1;
        }
    }
}

/// Bucket side giving about 8 points per occupied cell over the bounding box
/// of `pts` (axes with no extent are ignored).
pub(crate) fn auto_cell(pts: &[[f64; 3]]) -> f64 {
    let mut ext = vec![];
    for a in 0..3 {
        let (lo, hi) = pts
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
                (l.min(p[a]), h.max(p[a]))
            });
        if hi - lo > 1e-9 {
            ext.push(hi - lo);
        }
    }
    if ext.is_empty() || pts.is_empty() {
        return 1.0;
    }
    let vol: f64 = ext.iter().product();
    (vol / pts.len() as f64 * 8.0).powf(1.0 / ext.len() as f64)
}
