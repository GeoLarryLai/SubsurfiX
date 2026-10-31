// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Quantile regression forest (QRF; Meinshausen 2006, JMLR 7:983–999),
//! written in house (no extra dependency).
//!
//! Trees are grown as in a random forest (bootstrap rows, `mtry` random
//! features per split, variance-reduction splits, `min_leaf` rows per
//! leaf). Unlike a plain forest, each leaf keeps its training rows. For a
//! new row, training row `i` gets weight `wᵢ = (1/T)·Σₜ 1[i ∈ leafₜ(x)] /
//! |leafₜ(x)|` (in-bag rows, with bootstrap multiplicity), and quantile
//! `q` is the smallest `yᵢ` whose weighted CDF reaches `q`.
//! Missing features (NaN) go to the right child.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{SxError, SxResult};

/// QRF settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QrfParams {
    /// Trees.
    pub n_trees: usize,
    /// Features tried per split (0 = max(1, p/3)).
    pub mtry: usize,
    /// Minimum rows per leaf.
    pub min_leaf: usize,
    /// Maximum depth (0 = unlimited).
    pub max_depth: usize,
    /// Bootstrap sample size as a fraction of the rows.
    pub sample_frac: f64,
    /// Seed; tree `t` uses `seed + t`.
    pub seed: u64,
}

impl Default for QrfParams {
    fn default() -> Self {
        Self {
            n_trees: 200,
            mtry: 0,
            min_leaf: 10,
            max_depth: 0,
            sample_frac: 1.0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
enum Node {
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    /// Range into the tree's `members`.
    Leaf { start: usize, len: usize },
}

#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
    members: Vec<u32>,
}

impl Tree {
    fn leaf(&self, x: &[f64]) -> &[u32] {
        let mut n = 0;
        loop {
            match &self.nodes[n] {
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    n = if x[*feature] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
                Node::Leaf { start, len } => return &self.members[*start..*start + *len],
            }
        }
    }
}

/// Fitted quantile regression forest.
#[derive(Debug, Clone)]
pub struct QuantileForest {
    trees: Vec<Tree>,
    y: Vec<f64>,
    n_features: usize,
}

struct Grow<'a> {
    x: &'a [Vec<f64>],
    y: &'a [f64],
    p: QrfParams,
    mtry: usize,
    rng: ChaCha8Rng,
    nodes: Vec<Node>,
    members: Vec<u32>,
}

impl Grow<'_> {
    fn build(&mut self, rows: &mut [u32], depth: usize) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::Leaf { start: 0, len: 0 });
        let split = if rows.len() >= 2 * self.p.min_leaf
            && (self.p.max_depth == 0 || depth < self.p.max_depth)
        {
            self.best_split(rows)
        } else {
            None
        };
        match split {
            Some((feature, threshold)) => {
                // Partition: left = x <= thr (NaN goes right).
                let mut i = 0;
                let mut j = rows.len();
                while i < j {
                    if self.x[rows[i] as usize][feature] <= threshold {
                        i += 1;
                    } else {
                        j -= 1;
                        rows.swap(i, j);
                    }
                }
                let (l, r) = rows.split_at_mut(i);
                let left = self.build(l, depth + 1);
                let right = self.build(r, depth + 1);
                self.nodes[id] = Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                };
            }
            None => {
                let start = self.members.len();
                self.members.extend_from_slice(rows);
                self.nodes[id] = Node::Leaf {
                    start,
                    len: rows.len(),
                };
            }
        }
        id
    }

    fn best_split(&mut self, rows: &[u32]) -> Option<(usize, f64)> {
        let nf = self.x[0].len();
        let mut feats: Vec<usize> = (0..nf).collect();
        for i in 0..self.mtry.min(nf) {
            let j = self.rng.random_range(i..nf);
            feats.swap(i, j);
        }
        let n = rows.len() as f64;
        let tot: f64 = rows.iter().map(|&r| self.y[r as usize]).sum();
        let mut best: Option<(f64, usize, f64)> = None;
        let min_leaf = self.p.min_leaf.max(1);
        for &f in &feats[..self.mtry.min(nf)] {
            let mut v: Vec<(f64, f64)> = rows
                .iter()
                .map(|&r| (self.x[r as usize][f], self.y[r as usize]))
                .filter(|(a, _)| !a.is_nan())
                .collect();
            if v.len() < 2 * min_leaf {
                continue;
            }
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            // NaN rows sit on the right with the rest of the right sum.
            let mut ls = 0.0;
            for k in 0..v.len() - 1 {
                ls += v[k].1;
                let nl = (k + 1) as f64;
                let nr = n - nl;
                if k + 1 < min_leaf || (nr as usize) < min_leaf || v[k].0 == v[k + 1].0 {
                    continue;
                }
                // Maximize the between-node sum of squares.
                let gain = ls * ls / nl + (tot - ls) * (tot - ls) / nr;
                if best.is_none_or(|b| gain > b.0) {
                    best = Some((gain, f, 0.5 * (v[k].0 + v[k + 1].0)));
                }
            }
        }
        let parent = tot * tot / n;
        best.filter(|b| b.0 > parent * (1.0 + 1e-12) + 1e-12)
            .map(|b| (b.1, b.2))
    }
}

impl QuantileForest {
    /// Fit on rows `x` (equal lengths) and finite targets `y`.
    ///
    /// # Errors
    /// [`SxError::Shape`] on ragged or mismatched input; [`SxError::EmptyInput`]
    /// with no rows; [`SxError::InvalidInput`] for non-finite `y` or zero trees.
    pub fn fit(x: &[Vec<f64>], y: &[f64], p: &QrfParams) -> SxResult<Self> {
        if x.len() != y.len() {
            return Err(SxError::Shape(format!(
                "{} rows vs {} targets",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(SxError::EmptyInput("QRF needs training rows".into()));
        }
        let nf = x[0].len();
        if nf == 0 || x.iter().any(|r| r.len() != nf) {
            return Err(SxError::Shape(
                "QRF rows must have the same, non-zero length".into(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) || p.n_trees == 0 {
            return Err(SxError::InvalidInput(
                "QRF targets must be finite and n_trees ≥ 1".into(),
            ));
        }
        let mtry = if p.mtry == 0 {
            (nf / 3).max(1)
        } else {
            p.mtry.min(nf)
        };
        let m = ((x.len() as f64 * p.sample_frac).round() as usize).max(1);
        let trees = crate::runtime::install(|| {
            (0..p.n_trees)
                .into_par_iter()
                .map(|t| {
                    let mut rng = ChaCha8Rng::seed_from_u64(p.seed.wrapping_add(t as u64));
                    let mut rows: Vec<u32> = (0..m)
                        .map(|_| rng.random_range(0..x.len()) as u32)
                        .collect();
                    let mut g = Grow {
                        x,
                        y,
                        p: *p,
                        mtry,
                        rng,
                        nodes: vec![],
                        members: vec![],
                    };
                    g.build(&mut rows, 0);
                    Tree {
                        nodes: g.nodes,
                        members: g.members,
                    }
                })
                .collect()
        });
        Ok(Self {
            trees,
            y: y.to_vec(),
            n_features: nf,
        })
    }

    /// Quantiles `qs` (each in [0, 1]) for every row: `[row][q]`.
    ///
    /// # Errors
    /// [`SxError::Shape`] if a row has the wrong length; [`SxError::InvalidInput`]
    /// for a quantile outside [0, 1].
    pub fn predict_quantiles(&self, x: &[Vec<f64>], qs: &[f64]) -> SxResult<Vec<Vec<f64>>> {
        if x.iter().any(|r| r.len() != self.n_features) {
            return Err(SxError::Shape(format!(
                "QRF rows need {} features",
                self.n_features
            )));
        }
        if qs.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(SxError::InvalidInput("quantiles must lie in [0, 1]".into()));
        }
        let mut order: Vec<usize> = (0..self.y.len()).collect();
        order.sort_by(|a, b| self.y[*a].total_cmp(&self.y[*b]));
        let t = self.trees.len() as f64;
        Ok(crate::runtime::install(|| {
            x.par_iter()
                .map(|row| {
                    let mut w = vec![0.0f64; self.y.len()];
                    for tree in &self.trees {
                        let leaf = tree.leaf(row);
                        let inc = 1.0 / (t * leaf.len() as f64);
                        for &i in leaf {
                            w[i as usize] += inc;
                        }
                    }
                    let mut out = vec![f64::NAN; qs.len()];
                    let mut qi: Vec<usize> = (0..qs.len()).collect();
                    qi.sort_by(|a, b| qs[*a].total_cmp(&qs[*b]));
                    let mut acc = 0.0;
                    let mut k = 0;
                    let mut last = f64::NAN;
                    for &i in &order {
                        if w[i] == 0.0 {
                            continue;
                        }
                        acc += w[i];
                        last = self.y[i];
                        while k < qi.len() && acc >= qs[qi[k]] - 1e-12 {
                            out[qi[k]] = self.y[i];
                            k += 1;
                        }
                    }
                    for q in qi[k..].iter() {
                        out[*q] = last;
                    }
                    out
                })
                .collect()
        }))
    }

    /// Mean prediction (weighted mean of the training targets).
    pub fn predict_mean(&self, row: &[f64]) -> f64 {
        let t = self.trees.len() as f64;
        self.trees
            .iter()
            .map(|tree| {
                let leaf = tree.leaf(row);
                leaf.iter().map(|&i| self.y[i as usize]).sum::<f64>() / leaf.len() as f64
            })
            .sum::<f64>()
            / t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heteroscedastic_coverage() {
        // y = 2x + noise with spread growing in x.
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut gen = |n: usize| {
            let mut xs = vec![];
            let mut ys = vec![];
            for _ in 0..n {
                let x0: f64 = rng.random::<f64>();
                let x1: f64 = rng.random::<f64>(); // irrelevant feature
                let e: f64 = rng.random::<f64>() - 0.5;
                xs.push(vec![x0, x1]);
                ys.push(2.0 * x0 + e * (0.2 + 2.0 * x0));
            }
            (xs, ys)
        };
        let (xt, yt) = gen(2000);
        let (xv, yv) = gen(1000);
        let f = QuantileForest::fit(
            &xt,
            &yt,
            &QrfParams {
                n_trees: 100,
                seed: 1,
                ..QrfParams::default()
            },
        )
        .unwrap();
        let q = f.predict_quantiles(&xv, &[0.1, 0.5, 0.9]).unwrap();
        let inside = q
            .iter()
            .zip(&yv)
            .filter(|(r, y)| **y >= r[0] && **y <= r[2])
            .count() as f64
            / yv.len() as f64;
        assert!((0.72..0.88).contains(&inside), "coverage {inside}");
        // Interval is wider where the noise is larger.
        let w = |x0: f64| {
            let r = &f.predict_quantiles(&[vec![x0, 0.5]], &[0.1, 0.9]).unwrap()[0];
            r[1] - r[0]
        };
        assert!(w(0.9) > 2.0 * w(0.1));
        assert!(q.iter().all(|r| r[0] <= r[1] && r[1] <= r[2]));
        assert!((f.predict_mean(&[0.5, 0.5]) - 1.0).abs() < 0.15);
    }
}
