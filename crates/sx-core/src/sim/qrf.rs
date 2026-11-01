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
//!
//! Options used by EMBER: `random_split` draws one split value per
//! candidate feature (uniform between the node's min and max) and keeps the
//! feature with the smallest within-node variance (Daly 2022, §2.2);
//! [`QuantileForest::fit_embedded`] adds per-tree feature columns computed
//! from that tree's bootstrap sample.

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
    /// One random split value per candidate feature instead of the best one.
    pub random_split: bool,
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
            random_split: false,
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
    /// Bit set of the rows drawn into this tree's bootstrap sample.
    in_bag: Vec<u64>,
    /// Per-tree feature columns for every training row (`n × extra`), used
    /// for out-of-bag prediction; empty without per-tree features.
    extra: Vec<f64>,
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
    n_extra: usize,
}

/// Per-tree feature columns: given the tree's bootstrap rows (with
/// repeats), the extra columns for every training row, `[row][col]`.
pub type TreeFeatures<'a> = dyn Fn(&[u32]) -> Vec<Vec<f64>> + Sync + 'a;

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
            if self.p.random_split {
                let (lo, hi) = (v[0].0, v[v.len() - 1].0);
                if hi <= lo {
                    continue;
                }
                let thr = lo + self.rng.random::<f64>() * (hi - lo);
                let k = v.partition_point(|a| a.0 <= thr);
                if k < min_leaf || (n as usize) - k < min_leaf {
                    continue;
                }
                let ls: f64 = v[..k].iter().map(|a| a.1).sum();
                let (nl, nr) = (k as f64, n - k as f64);
                let gain = ls * ls / nl + (tot - ls) * (tot - ls) / nr;
                if best.is_none_or(|b| gain > b.0) {
                    best = Some((gain, f, thr));
                }
                continue;
            }
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
        Self::fit_inner(x, y, p, None)
    }

    /// Fit with per-tree feature columns appended to `x`: `extra(bag)`
    /// returns the columns for every row given the tree's bootstrap rows.
    /// Prediction rows must carry the base and extra columns.
    ///
    /// # Errors
    /// As [`QuantileForest::fit`]; [`SxError::Shape`] if `extra` returns a
    /// wrong shape.
    pub fn fit_embedded(
        x: &[Vec<f64>],
        y: &[f64],
        p: &QrfParams,
        n_extra: usize,
        extra: &TreeFeatures<'_>,
    ) -> SxResult<Self> {
        Self::fit_inner(x, y, p, Some((n_extra, extra)))
    }

    fn fit_inner(
        x: &[Vec<f64>],
        y: &[f64],
        p: &QrfParams,
        extra: Option<(usize, &TreeFeatures<'_>)>,
    ) -> SxResult<Self> {
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
        let nb = x[0].len();
        let ne = extra.map_or(0, |e| e.0);
        let nf = nb + ne;
        if nf == 0 || x.iter().any(|r| r.len() != nb) {
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
        let trees: Vec<SxResult<Tree>> = crate::runtime::install(|| {
            (0..p.n_trees)
                .into_par_iter()
                .map(|t| {
                    let mut rng = ChaCha8Rng::seed_from_u64(p.seed.wrapping_add(t as u64));
                    let mut rows: Vec<u32> = (0..m)
                        .map(|_| rng.random_range(0..x.len()) as u32)
                        .collect();
                    let (full, flat): (Option<Vec<Vec<f64>>>, Vec<f64>) = match extra {
                        None => (None, vec![]),
                        Some((ne, f)) => {
                            let e = f(&rows);
                            if e.len() != x.len() || e.iter().any(|r| r.len() != ne) {
                                return Err(SxError::Shape(
                                    "per-tree features have the wrong shape".into(),
                                ));
                            }
                            let full = x
                                .iter()
                                .zip(&e)
                                .map(|(a, b)| a.iter().chain(b).copied().collect())
                                .collect();
                            (Some(full), e.concat())
                        }
                    };
                    let xs: &[Vec<f64>] = full.as_deref().unwrap_or(x);
                    let mut g = Grow {
                        x: xs,
                        y,
                        p: *p,
                        mtry,
                        rng,
                        nodes: vec![],
                        members: vec![],
                    };
                    let mut in_bag = vec![0u64; x.len().div_ceil(64)];
                    for &r in &rows {
                        in_bag[r as usize / 64] |= 1 << (r % 64);
                    }
                    g.build(&mut rows, 0);
                    Ok(Tree {
                        nodes: g.nodes,
                        members: g.members,
                        in_bag,
                        extra: flat,
                    })
                })
                .collect()
        });
        Ok(Self {
            trees: trees.into_iter().collect::<SxResult<_>>()?,
            y: y.to_vec(),
            n_features: nf,
            n_extra: ne,
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
        Ok(self.quantiles_with(x, qs, false))
    }

    /// Out-of-bag quantiles at the training rows `x` (the rows passed to
    /// [`QuantileForest::fit`], same order): each row is predicted only by
    /// the trees whose bootstrap sample left it out. Rows that are in every
    /// tree's sample get NaN.
    ///
    /// # Errors
    /// [`SxError::Shape`] if `x` is not the training set shape;
    /// [`SxError::InvalidInput`] for a quantile outside [0, 1].
    pub fn predict_quantiles_oob(&self, x: &[Vec<f64>], qs: &[f64]) -> SxResult<Vec<Vec<f64>>> {
        if x.len() != self.y.len() || x.iter().any(|r| r.len() != self.n_features) {
            return Err(SxError::Shape(
                "OOB prediction needs the training rows".into(),
            ));
        }
        if qs.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(SxError::InvalidInput("quantiles must lie in [0, 1]".into()));
        }
        Ok(self.quantiles_with(x, qs, true))
    }

    fn quantiles_with(&self, x: &[Vec<f64>], qs: &[f64], oob: bool) -> Vec<Vec<f64>> {
        let mut order: Vec<usize> = (0..self.y.len()).collect();
        order.sort_by(|a, b| self.y[*a].total_cmp(&self.y[*b]));
        let mut qi: Vec<usize> = (0..qs.len()).collect();
        qi.sort_by(|a, b| qs[*a].total_cmp(&qs[*b]));
        crate::runtime::install(|| {
            x.par_iter()
                .enumerate()
                .map(|(r, row)| {
                    let mut w = vec![0.0f64; self.y.len()];
                    let mut used = 0usize;
                    for tree in &self.trees {
                        if oob && tree.in_bag[r / 64] >> (r % 64) & 1 == 1 {
                            continue;
                        }
                        used += 1;
                        let leaf = tree.leaf(row);
                        let inc = 1.0 / leaf.len() as f64;
                        for &i in leaf {
                            w[i as usize] += inc;
                        }
                    }
                    let mut out = vec![f64::NAN; qs.len()];
                    if used == 0 {
                        return out;
                    }
                    let t = used as f64;
                    let mut acc = 0.0;
                    let mut k = 0;
                    let mut last = f64::NAN;
                    for &i in &order {
                        if w[i] == 0.0 {
                            continue;
                        }
                        acc += w[i] / t;
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
        })
    }

    /// Training targets in ascending order with their forest weights
    /// `ωᵢ(x)` (Meinshausen 2006): the estimated conditional distribution
    /// is `F(z | x) = Σ ωᵢ 1[Zᵢ ≤ z]`. Zero weights are dropped.
    ///
    /// # Errors
    /// [`SxError::Shape`] if a row has the wrong length.
    pub fn predict_weights(&self, x: &[Vec<f64>]) -> SxResult<Vec<Vec<(f64, f64)>>> {
        if x.iter().any(|r| r.len() != self.n_features) {
            return Err(SxError::Shape(format!(
                "QRF rows need {} features",
                self.n_features
            )));
        }
        Ok(self.weights_with(x, false))
    }

    /// As [`QuantileForest::predict_weights`] at the training rows, from the
    /// trees whose bootstrap sample left the row out. For a forest fitted
    /// with per-tree features, `x` holds the base columns and each tree uses
    /// its own extra columns for the row. Rows in every sample get no
    /// weights.
    ///
    /// # Errors
    /// [`SxError::Shape`] if `x` is not the training set shape.
    pub fn predict_weights_oob(&self, x: &[Vec<f64>]) -> SxResult<Vec<Vec<(f64, f64)>>> {
        let nb = self.n_features - self.n_extra;
        if x.len() != self.y.len() || x.iter().any(|r| r.len() != nb) {
            return Err(SxError::Shape(
                "OOB prediction needs the training rows".into(),
            ));
        }
        Ok(self.weights_with(x, true))
    }

    fn weights_with(&self, x: &[Vec<f64>], oob: bool) -> Vec<Vec<(f64, f64)>> {
        let ne = self.n_extra;
        crate::runtime::install(|| {
            x.par_iter()
                .enumerate()
                .map(|(r, row)| {
                    let mut w: std::collections::HashMap<u32, f64> = Default::default();
                    let mut used = 0usize;
                    let mut full = row.clone();
                    for tree in &self.trees {
                        if oob && tree.in_bag[r / 64] >> (r % 64) & 1 == 1 {
                            continue;
                        }
                        used += 1;
                        let leaf = if oob && ne > 0 {
                            full.truncate(row.len());
                            full.extend_from_slice(&tree.extra[r * ne..(r + 1) * ne]);
                            tree.leaf(&full)
                        } else {
                            tree.leaf(row)
                        };
                        let inc = 1.0 / leaf.len() as f64;
                        for &i in leaf {
                            *w.entry(i).or_default() += inc;
                        }
                    }
                    let t = used.max(1) as f64;
                    let mut out: Vec<(f64, f64)> = w
                        .into_iter()
                        .map(|(i, v)| (self.y[i as usize], v / t))
                        .collect();
                    out.sort_by(|a, b| a.0.total_cmp(&b.0));
                    out
                })
                .collect()
        })
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
