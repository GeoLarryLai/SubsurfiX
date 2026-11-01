// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Log separability: how many top-level lithology classes a set of logs can
//! tell apart, and which logs matter.
//!
//! With labels:
//! 1. Classes with too few rows are dropped. The fine classes are merged one
//!    pair at a time, always the pair the logs confuse most (cross-validated
//!    confusion on the merge log set), giving one class list per level `k`.
//! 2. Every level and every log combination is scored with grouped
//!    cross-validation (whole wells held out): balanced accuracy, smallest
//!    class recall, balanced kappa, and the effective number of classes
//!    `2^I`, with `I` the mutual information (bits) between true and
//!    predicted class under equal class weights.
//! 3. Recommended `k`: the largest level whose best log combination reaches
//!    the target balanced accuracy and the minimum class recall. Best logs:
//!    the fewest logs within `tol` of the best score at that level.
//!
//! Without labels: k-means clusters per log combination and `k`, scored by
//! prediction strength (Tibshirani and Walther 2005) on random halves of the
//! wells. Recommended `k`: the largest `k` whose prediction strength reaches
//! the threshold, beats one Gaussian cloud with the same mean and covariance
//! by a margin, and whose clusters are split by density gaps (a split of one
//! unimodal cloud is stable but has no gap).
//!
//! Classifiers: Gaussian (one mean and shrunk covariance per class, equal
//! priors) or rules (a cutoff tree with at most `k` splits, classes weighted
//! equally).

use std::collections::{BTreeMap, BTreeSet};

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{SxError, SxResult};

type Split = Option<(f64, usize, f64)>;

/// Classifier used to score labelled classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classifier {
    /// Gaussian class densities (shrunk covariance), equal priors.
    Gaussian,
    /// Cutoff tree with at most `k` splits.
    Rules,
}

impl Classifier {
    /// Parse "gaussian" / "rules".
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for other names.
    pub fn parse(s: &str) -> SxResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gaussian" | "qda" => Ok(Self::Gaussian),
            "rules" | "tree" | "cutoffs" => Ok(Self::Rules),
            _ => Err(SxError::InvalidInput(format!("unknown classifier {s:?}"))),
        }
    }
}

/// Settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparabilityParams {
    /// Classifier for the labelled mode.
    pub classifier: Classifier,
    /// Cross-validation folds (groups held out together).
    pub n_folds: usize,
    /// Seed.
    pub seed: u64,
    /// Classes with fewer rows are dropped.
    pub min_rows_class: usize,
    /// Rows kept per class (labelled) or in total (clusters); random subsample.
    pub max_rows: usize,
    /// Largest log combination tried (0 = all logs).
    pub max_logs: usize,
    /// A log combination needs this share of rows with every log present.
    pub min_coverage: f64,
    /// Balanced accuracy a level needs to be recommended.
    pub target_accuracy: f64,
    /// Smallest class recall a level needs to be recommended.
    pub min_recall: f64,
    /// Best logs: fewest logs within this score of the best.
    pub tol: f64,
    /// Largest number of clusters (no labels).
    pub k_max: usize,
    /// Prediction strength a cluster count needs (no labels).
    pub ps_threshold: f64,
    /// Prediction strength must beat that of one Gaussian cloud with the
    /// same mean and covariance by this much (no labels).
    pub ps_margin: f64,
    /// Largest allowed density ratio between the gap and the lower peak of
    /// any two clusters along the line through their centres (no labels).
    pub valley_max: f64,
    /// Random halves per prediction-strength score.
    pub ps_splits: usize,
    /// Covariance shrinkage toward the diagonal (Gaussian classifier).
    pub shrink: f64,
    /// Fewest rows at one end of a cutoff rule.
    pub rule_min_leaf: usize,
}

impl Default for SeparabilityParams {
    fn default() -> Self {
        Self {
            classifier: Classifier::Gaussian,
            n_folds: 5,
            seed: 0,
            min_rows_class: 200,
            max_rows: 5000,
            max_logs: 0,
            min_coverage: 0.3,
            target_accuracy: 0.7,
            min_recall: 0.5,
            tol: 0.02,
            k_max: 8,
            ps_threshold: 0.8,
            ps_margin: 0.05,
            valley_max: 0.7,
            ps_splits: 10,
            shrink: 0.1,
            rule_min_leaf: 50,
        }
    }
}

/// Input rows.
#[derive(Debug, Clone, Copy)]
pub struct SeparabilityInput<'a> {
    /// Log values per row (`[row][log]`, NaN = missing).
    pub logs: &'a [Vec<f64>],
    /// Log names.
    pub names: &'a [String],
    /// Class code per row (negative = unlabelled); `None` for the cluster mode.
    pub labels: Option<&'a [i32]>,
    /// Group per row (usually the well); groups are held out together.
    pub groups: &'a [i32],
}

/// One merge level: `k` classes, each a list of input codes.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeLevel {
    /// Number of classes.
    pub k: usize,
    /// Input codes in each class.
    pub classes: Vec<Vec<i32>>,
}

/// Score of one log combination at one level.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreRow {
    /// Number of classes (or clusters).
    pub k: usize,
    /// Log indices.
    pub logs: Vec<usize>,
    /// Rows used.
    pub n_rows: usize,
    /// Share of rows with every log present.
    pub coverage: f64,
    /// Balanced accuracy (NaN in the cluster mode).
    pub balanced_accuracy: f64,
    /// Smallest class recall (NaN in the cluster mode).
    pub min_recall: f64,
    /// Balanced kappa (NaN in the cluster mode).
    pub kappa: f64,
    /// `2^I` with `I` the mutual information in bits (NaN in the cluster mode).
    pub effective_classes: f64,
    /// Prediction strength (NaN in the labelled mode).
    pub prediction_strength: f64,
    /// Prediction strength of one Gaussian cloud with the same mean and
    /// covariance (NaN in the labelled mode).
    pub prediction_strength_null: f64,
    /// Worst gap-to-peak density ratio over cluster pairs (NaN in the
    /// labelled mode).
    pub valley_ratio: f64,
}

/// Drop in score when one log is removed from the reference combination.
#[derive(Debug, Clone, PartialEq)]
pub struct LogImportance {
    /// Log index.
    pub log: usize,
    /// Score of the reference minus score without this log.
    pub drop: f64,
}

/// Result.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparabilityReport {
    /// "labels" or "clusters".
    pub mode: String,
    /// Log names.
    pub names: Vec<String>,
    /// Merge levels (labelled mode), finest first.
    pub levels: Vec<MergeLevel>,
    /// Codes dropped for too few rows.
    pub dropped_classes: Vec<i32>,
    /// Logs used to choose merges (labelled) or for importance (both).
    pub reference_logs: Vec<usize>,
    /// Every score.
    pub scores: Vec<ScoreRow>,
    /// Recommended number of classes.
    pub recommended_k: usize,
    /// Fewest logs within `tol` of the best score at the recommended `k`.
    pub best_logs: Vec<usize>,
    /// Log importance at the recommended `k` (reference combination).
    pub importance: Vec<LogImportance>,
    /// Row-normalized confusion at the recommended `k` with the best logs
    /// (labelled mode).
    pub confusion: Vec<Vec<f64>>,
}

/// Run the separability analysis.
///
/// # Errors
/// [`SxError::Shape`] on ragged or mismatched input; [`SxError::EmptyInput`]
/// with no usable rows; [`SxError::InvalidInput`] for bad settings or fewer
/// than two classes or groups.
pub fn separability(
    input: &SeparabilityInput<'_>,
    p: &SeparabilityParams,
) -> SxResult<SeparabilityReport> {
    let n = input.logs.len();
    let nl = input.names.len();
    if nl == 0 || input.logs.iter().any(|r| r.len() != nl) {
        return Err(SxError::Shape(
            "every row needs one value per log name".into(),
        ));
    }
    if input.groups.len() != n || input.labels.is_some_and(|l| l.len() != n) {
        return Err(SxError::Shape(
            "labels and groups need one value per row".into(),
        ));
    }
    if p.n_folds < 2 || p.max_rows < 10 || !(0.0..=1.0).contains(&p.min_coverage) {
        return Err(SxError::InvalidInput(
            "n_folds ≥ 2, max_rows ≥ 10 and min_coverage in [0, 1] are required".into(),
        ));
    }
    let subsets = log_subsets(nl, p.max_logs);
    match input.labels {
        Some(labels) => labelled(input, labels, &subsets, p),
        None => clusters(input, &subsets, p),
    }
}

fn log_subsets(nl: usize, max_logs: usize) -> Vec<Vec<usize>> {
    let cap = if max_logs == 0 { nl } else { max_logs.min(nl) };
    let mut out: Vec<Vec<usize>> = (1u32..(1u32 << nl))
        .map(|m| (0..nl).filter(|i| m & (1 << i) != 0).collect::<Vec<_>>())
        .filter(|s: &Vec<usize>| s.len() <= cap)
        .collect();
    out.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    out
}

fn complete(row: &[f64], logs: &[usize]) -> bool {
    logs.iter().all(|&j| row[j].is_finite())
}

/// Log set for merging and importance: logs added by coverage while every
/// log is present on at least `min_cov` of the rows (at least one log).
fn reference_set(rows: &[&[f64]], nl: usize, min_cov: f64) -> Vec<usize> {
    let n = rows.len().max(1) as f64;
    let mut order: Vec<usize> = (0..nl).collect();
    let cov = |s: &[usize]| rows.iter().filter(|r| complete(r, s)).count() as f64 / n;
    order.sort_by(|a, b| cov(&[*b]).total_cmp(&cov(&[*a])).then(a.cmp(b)));
    let mut set = vec![order[0]];
    for &j in &order[1..] {
        let mut t = set.clone();
        t.push(j);
        if cov(&t) >= min_cov.max(0.5) {
            set = t;
        }
    }
    set.sort_unstable();
    set
}

fn group_folds(groups: &[i32], k: usize, seed: u64) -> SxResult<BTreeMap<i32, usize>> {
    let mut u: Vec<i32> = groups
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if u.len() < 2 {
        return Err(SxError::InvalidInput(
            "need at least two groups (wells)".into(),
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
    for i in (1..u.len()).rev() {
        let j = rng.random_range(0..=i);
        u.swap(i, j);
    }
    let k = k.min(u.len());
    Ok(u.into_iter().enumerate().map(|(i, g)| (g, i % k)).collect())
}

// ---------------------------------------------------------------- labelled

struct Labelled<'a> {
    rows: Vec<&'a [f64]>,
    class: Vec<usize>, // index into fine classes
    fold: Vec<usize>,
    n_folds: usize,
}

fn labelled(
    input: &SeparabilityInput<'_>,
    labels: &[i32],
    subsets: &[Vec<usize>],
    p: &SeparabilityParams,
) -> SxResult<SeparabilityReport> {
    let mut count: BTreeMap<i32, usize> = BTreeMap::new();
    for &l in labels.iter().filter(|l| **l >= 0) {
        *count.entry(l).or_default() += 1;
    }
    let codes: Vec<i32> = count
        .iter()
        .filter(|(_, c)| **c >= p.min_rows_class)
        .map(|(k, _)| *k)
        .collect();
    let dropped: Vec<i32> = count
        .keys()
        .filter(|k| !codes.contains(k))
        .copied()
        .collect();
    if codes.len() < 2 {
        return Err(SxError::InvalidInput(
            "need at least two classes with enough rows".into(),
        ));
    }
    let idx: BTreeMap<i32, usize> = codes.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    // Subsample each class to max_rows.
    let mut by_class: Vec<Vec<usize>> = vec![vec![]; codes.len()];
    for (i, l) in labels.iter().enumerate() {
        if let Some(&c) = idx.get(l) {
            by_class[c].push(i);
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
    let mut keep = vec![];
    for rows in &mut by_class {
        for i in 0..rows.len().min(p.max_rows) {
            let j = rng.random_range(i..rows.len());
            rows.swap(i, j);
        }
        rows.truncate(p.max_rows);
        keep.extend_from_slice(rows);
    }
    keep.sort_unstable();
    let folds = group_folds(input.groups, p.n_folds, p.seed)?;
    let n_folds = p.n_folds.min(folds.len());
    let data = Labelled {
        rows: keep.iter().map(|&i| input.logs[i].as_slice()).collect(),
        class: keep.iter().map(|&i| idx[&labels[i]]).collect(),
        fold: keep.iter().map(|&i| folds[&input.groups[i]]).collect(),
        n_folds,
    };
    let nl = input.names.len();
    let reference = reference_set(&data.rows, nl, p.min_coverage);

    // Merge levels on the reference logs.
    let mut map: Vec<usize> = (0..codes.len()).collect(); // fine -> merged
    let mut levels = vec![];
    let mut k = codes.len();
    loop {
        levels.push((k, map.clone()));
        if k <= 2 {
            break;
        }
        let conf = cv_confusion(&data, &map, k, &reference, p)
            .ok_or_else(|| SxError::EmptyInput("no rows with the reference logs".into()))?;
        let rate = row_normalize(&conf);
        let mut best = (f64::NEG_INFINITY, 0, 1);
        #[allow(clippy::needless_range_loop)] // symmetric pair table
        for a in 0..k {
            for b in a + 1..k {
                let s = rate[a][b] + rate[b][a];
                if s > best.0 {
                    best = (s, a, b);
                }
            }
        }
        let (_, a, b) = best;
        for m in &mut map {
            if *m == b {
                *m = a;
            } else if *m > b {
                *m -= 1;
            }
        }
        k -= 1;
    }

    // Score every level × log combination.
    let n_all = data.rows.len() as f64;
    let jobs: Vec<(usize, usize)> = (0..levels.len())
        .flat_map(|l| (0..subsets.len()).map(move |s| (l, s)))
        .collect();
    let scores: Vec<Option<ScoreRow>> = crate::runtime::install(|| {
        jobs.par_iter()
            .map(|&(l, s)| {
                let (k, map) = &levels[l];
                let logs = &subsets[s];
                let n_rows = data.rows.iter().filter(|r| complete(r, logs)).count();
                let coverage = n_rows as f64 / n_all;
                if coverage < p.min_coverage {
                    return None;
                }
                let conf = cv_confusion(&data, map, *k, logs, p)?;
                Some(score_row(*k, logs.clone(), n_rows, coverage, &conf))
            })
            .collect()
    });
    let scores: Vec<ScoreRow> = scores.into_iter().flatten().collect();

    // Recommendation.
    let best_at = |k: usize| {
        scores
            .iter()
            .filter(|r| r.k == k)
            .max_by(|a, b| a.balanced_accuracy.total_cmp(&b.balanced_accuracy))
    };
    let mut recommended = 2;
    for (k, _) in &levels {
        if let Some(r) = best_at(*k) {
            if r.balanced_accuracy >= p.target_accuracy
                && r.min_recall >= p.min_recall
                && *k > recommended
            {
                recommended = *k;
            }
        }
    }
    let top = best_at(recommended).map_or(f64::NAN, |r| r.balanced_accuracy);
    let best_logs = scores
        .iter()
        .filter(|r| r.k == recommended && r.balanced_accuracy >= top - p.tol)
        .min_by(|a, b| {
            a.logs
                .len()
                .cmp(&b.logs.len())
                .then(b.balanced_accuracy.total_cmp(&a.balanced_accuracy))
        })
        .map(|r| r.logs.clone())
        .unwrap_or_default();
    let lvl = levels.iter().find(|(k, _)| *k == recommended).cloned();
    let (importance, confusion) = match lvl {
        Some((k, map)) => {
            let get = |logs: &[usize]| {
                scores
                    .iter()
                    .find(|r| r.k == k && r.logs == logs)
                    .map(|r| r.balanced_accuracy)
            };
            let base = get(&reference);
            let imp = reference
                .iter()
                .filter_map(|&j| {
                    let rest: Vec<usize> = reference.iter().copied().filter(|&x| x != j).collect();
                    let without = if rest.is_empty() {
                        Some(1.0 / k as f64)
                    } else {
                        get(&rest)
                    };
                    Some(LogImportance {
                        log: j,
                        drop: base? - without?,
                    })
                })
                .collect();
            let conf = cv_confusion(&data, &map, k, &best_logs, p)
                .map(|c| row_normalize(&c))
                .unwrap_or_default();
            (imp, conf)
        }
        None => (vec![], vec![]),
    };
    Ok(SeparabilityReport {
        mode: "labels".into(),
        names: input.names.to_vec(),
        levels: levels
            .iter()
            .map(|(k, map)| MergeLevel {
                k: *k,
                classes: (0..*k)
                    .map(|c| {
                        codes
                            .iter()
                            .zip(map)
                            .filter(|(_, m)| **m == c)
                            .map(|(code, _)| *code)
                            .collect()
                    })
                    .collect(),
            })
            .collect(),
        dropped_classes: dropped,
        reference_logs: reference,
        scores,
        recommended_k: recommended,
        best_logs,
        importance,
        confusion,
    })
}

fn row_normalize(c: &[Vec<f64>]) -> Vec<Vec<f64>> {
    c.iter()
        .map(|r| {
            let s: f64 = r.iter().sum();
            r.iter()
                .map(|v| if s > 0.0 { v / s } else { 0.0 })
                .collect()
        })
        .collect()
}

fn score_row(
    k: usize,
    logs: Vec<usize>,
    n_rows: usize,
    coverage: f64,
    conf: &[Vec<f64>],
) -> ScoreRow {
    let rate = row_normalize(conf);
    let present: Vec<usize> = (0..k)
        .filter(|&c| conf[c].iter().sum::<f64>() > 0.0)
        .collect();
    let kk = present.len().max(1) as f64;
    let recalls: Vec<f64> = present.iter().map(|&c| rate[c][c]).collect();
    let bacc = recalls.iter().sum::<f64>() / kk;
    let minr = recalls.iter().copied().fold(f64::INFINITY, f64::min);
    let kappa = if kk > 1.0 {
        (bacc - 1.0 / kk) / (1.0 - 1.0 / kk)
    } else {
        f64::NAN
    };
    // Mutual information with equal class weights.
    let mut col = vec![0.0; k];
    for &c in &present {
        for j in 0..k {
            col[j] += rate[c][j] / kk;
        }
    }
    let mut mi = 0.0;
    for &c in &present {
        for j in 0..k {
            let pj = rate[c][j] / kk;
            if pj > 0.0 && col[j] > 0.0 {
                mi += pj * (pj / ((1.0 / kk) * col[j])).log2();
            }
        }
    }
    ScoreRow {
        k,
        logs,
        n_rows,
        coverage,
        balanced_accuracy: bacc,
        min_recall: minr,
        kappa,
        effective_classes: 2f64.powf(mi),
        prediction_strength: f64::NAN,
        prediction_strength_null: f64::NAN,
        valley_ratio: f64::NAN,
    }
}

/// Cross-validated confusion counts `[true][pred]` for `k` merged classes.
fn cv_confusion(
    d: &Labelled<'_>,
    map: &[usize],
    k: usize,
    logs: &[usize],
    p: &SeparabilityParams,
) -> Option<Vec<Vec<f64>>> {
    let rows: Vec<usize> = (0..d.rows.len())
        .filter(|&i| complete(d.rows[i], logs))
        .collect();
    if rows.is_empty() {
        return None;
    }
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|&i| logs.iter().map(|&j| d.rows[i][j]).collect())
        .collect();
    let y: Vec<usize> = rows.iter().map(|&i| map[d.class[i]]).collect();
    let f: Vec<usize> = rows.iter().map(|&i| d.fold[i]).collect();
    let mut conf = vec![vec![0.0; k]; k];
    for fold in 0..d.n_folds {
        let tr: Vec<usize> = (0..x.len()).filter(|&i| f[i] != fold).collect();
        let te: Vec<usize> = (0..x.len()).filter(|&i| f[i] == fold).collect();
        if tr.is_empty() || te.is_empty() {
            continue;
        }
        let pred: Vec<usize> = match p.classifier {
            Classifier::Gaussian => {
                let m = Gaussian::fit(&x, &y, &tr, k, p.shrink);
                te.iter().map(|&i| m.predict(&x[i])).collect()
            }
            Classifier::Rules => {
                let t = RuleTree::fit(&x, &y, &tr, k, k, p.rule_min_leaf);
                te.iter().map(|&i| t.predict(&x[i])).collect()
            }
        };
        for (&i, &pc) in te.iter().zip(&pred) {
            conf[y[i]][pc] += 1.0;
        }
    }
    Some(conf)
}

// ---------------------------------------------------------------- Gaussian

pub(crate) struct Gaussian {
    classes: Vec<(Vec<f64>, Vec<f64>, f64)>, // mean, Cholesky factor (row-major), log det
    dim: usize,
}

fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= a[j * n + k] * a[j * n + k];
        }
        if s <= 0.0 {
            return false;
        }
        let d = s.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut t = a[i * n + j];
            for k in 0..j {
                t -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = t / d;
        }
        for k in j + 1..n {
            a[j * n + k] = 0.0;
        }
    }
    true
}

impl Gaussian {
    pub(crate) fn fit(x: &[Vec<f64>], y: &[usize], tr: &[usize], k: usize, shrink: f64) -> Self {
        let dim = x[0].len();
        // Pooled scale for the ridge.
        let mut gvar = vec![0.0; dim];
        let mut gmean = vec![0.0; dim];
        for &i in tr {
            for d in 0..dim {
                gmean[d] += x[i][d];
            }
        }
        for v in &mut gmean {
            *v /= tr.len() as f64;
        }
        for &i in tr {
            for d in 0..dim {
                gvar[d] += (x[i][d] - gmean[d]).powi(2);
            }
        }
        for v in &mut gvar {
            *v = (*v / tr.len() as f64).max(1e-12);
        }
        let classes = (0..k)
            .map(|c| {
                let rows: Vec<usize> = tr.iter().copied().filter(|&i| y[i] == c).collect();
                if rows.len() < 2 {
                    return (vec![f64::NAN; dim], vec![], f64::NAN);
                }
                let nr = rows.len() as f64;
                let mut m = vec![0.0; dim];
                for &i in &rows {
                    for d in 0..dim {
                        m[d] += x[i][d] / nr;
                    }
                }
                let mut s = vec![0.0; dim * dim];
                for &i in &rows {
                    for a in 0..dim {
                        let da = x[i][a] - m[a];
                        for b in 0..=a {
                            s[a * dim + b] += da * (x[i][b] - m[b]) / nr;
                        }
                    }
                }
                for a in 0..dim {
                    for b in 0..a {
                        s[b * dim + a] = s[a * dim + b];
                    }
                }
                for a in 0..dim {
                    for b in 0..dim {
                        if a != b {
                            s[a * dim + b] *= 1.0 - shrink;
                        }
                    }
                    s[a * dim + a] += 1e-4 * gvar[a];
                }
                if !cholesky(&mut s, dim) {
                    return (vec![f64::NAN; dim], vec![], f64::NAN);
                }
                let logdet = 2.0 * (0..dim).map(|a| s[a * dim + a].ln()).sum::<f64>();
                (m, s, logdet)
            })
            .collect();
        Self { classes, dim }
    }

    pub(crate) fn predict(&self, row: &[f64]) -> usize {
        let n = self.dim;
        let mut best = (f64::NEG_INFINITY, 0);
        let mut z = vec![0.0; n];
        for (c, (m, l, logdet)) in self.classes.iter().enumerate() {
            if l.is_empty() {
                continue;
            }
            for i in 0..n {
                let mut t = row[i] - m[i];
                for k in 0..i {
                    t -= l[i * n + k] * z[k];
                }
                z[i] = t / l[i * n + i];
            }
            let ll = -0.5 * (logdet + z.iter().map(|v| v * v).sum::<f64>());
            if ll > best.0 {
                best = (ll, c);
            }
        }
        best.1
    }
}

// ---------------------------------------------------------------- rules

enum RNode {
    Split {
        f: usize,
        t: f64,
        l: usize,
        r: usize,
    },
    Leaf(usize),
}

struct RuleTree {
    nodes: Vec<RNode>,
}

impl RuleTree {
    /// Best-first tree with at most `max_splits` splits, class-balanced Gini.
    fn fit(
        x: &[Vec<f64>],
        y: &[usize],
        tr: &[usize],
        k: usize,
        max_splits: usize,
        min_leaf: usize,
    ) -> Self {
        let mut cnt = vec![0.0; k];
        for &i in tr {
            cnt[y[i]] += 1.0;
        }
        let w: Vec<f64> = cnt
            .iter()
            .map(|c| if *c > 0.0 { 1.0 / c } else { 0.0 })
            .collect();
        let majority = |rows: &[usize]| {
            let mut h = vec![0.0; k];
            for &i in rows {
                h[y[i]] += w[y[i]];
            }
            (0..k).max_by(|a, b| h[*a].total_cmp(&h[*b])).unwrap_or(0)
        };
        let gini = |h: &[f64]| {
            let s: f64 = h.iter().sum();
            if s <= 0.0 {
                0.0
            } else {
                s - h.iter().map(|v| v * v).sum::<f64>() / s
            }
        };
        let best_split = |rows: &[usize]| -> Option<(f64, usize, f64)> {
            let mut tot = vec![0.0; k];
            for &i in rows {
                tot[y[i]] += w[y[i]];
            }
            let parent = gini(&tot);
            let mut best: Option<(f64, usize, f64)> = None;
            #[allow(clippy::needless_range_loop)] // f indexes every row
            for f in 0..x[0].len() {
                let mut v: Vec<(f64, usize)> = rows.iter().map(|&i| (x[i][f], y[i])).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut left = vec![0.0; k];
                for s in 0..v.len().saturating_sub(1) {
                    left[v[s].1] += w[v[s].1];
                    if s + 1 < min_leaf || v.len() - s - 1 < min_leaf || v[s].0 == v[s + 1].0 {
                        continue;
                    }
                    let right: Vec<f64> = tot.iter().zip(&left).map(|(a, b)| a - b).collect();
                    let gain = parent - gini(&left) - gini(&right);
                    if best.is_none_or(|b| gain > b.0) {
                        best = Some((gain, f, 0.5 * (v[s].0 + v[s + 1].0)));
                    }
                }
            }
            best.filter(|b| b.0 > 1e-12)
        };
        let mut nodes = vec![RNode::Leaf(majority(tr))];
        // (node, rows, best split as (gain, log, cutoff))
        let mut open: Vec<(usize, Vec<usize>, Split)> = vec![(0, tr.to_vec(), best_split(tr))];
        for _ in 0..max_splits {
            let Some(pos) = open
                .iter()
                .enumerate()
                .filter(|(_, o)| o.2.is_some())
                .max_by(|a, b| a.1 .2.unwrap().0.total_cmp(&b.1 .2.unwrap().0))
                .map(|(i, _)| i)
            else {
                break;
            };
            let (id, rows, split) = open.swap_remove(pos);
            let (_, f, t) = split.unwrap_or((0.0, 0, 0.0));
            let (lr, rr): (Vec<usize>, Vec<usize>) = rows.iter().partition(|&&i| x[i][f] <= t);
            let (l, r) = (nodes.len(), nodes.len() + 1);
            nodes.push(RNode::Leaf(majority(&lr)));
            nodes.push(RNode::Leaf(majority(&rr)));
            nodes[id] = RNode::Split { f, t, l, r };
            let sl = best_split(&lr);
            let sr = best_split(&rr);
            open.push((l, lr, sl));
            open.push((r, rr, sr));
        }
        Self { nodes }
    }

    fn predict(&self, row: &[f64]) -> usize {
        let mut n = 0;
        loop {
            match &self.nodes[n] {
                RNode::Split { f, t, l, r } => n = if row[*f] <= *t { *l } else { *r },
                RNode::Leaf(c) => return *c,
            }
        }
    }
}

// ---------------------------------------------------------------- clusters

fn standardize(x: &mut [Vec<f64>]) {
    let dim = x.first().map_or(0, Vec::len);
    for d in 0..dim {
        let mut v: Vec<f64> = x.iter().map(|r| r[d]).collect();
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        let med = q(0.5);
        let s = ((q(0.75) - q(0.25)) / 1.349).max(1e-9);
        for r in x.iter_mut() {
            r[d] = (r[d] - med) / s;
        }
    }
}

fn d2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn nearest_center(c: &[Vec<f64>], r: &[f64]) -> usize {
    (0..c.len())
        .min_by(|&a, &b| d2(&c[a], r).total_cmp(&d2(&c[b], r)))
        .unwrap_or(0)
}

/// k-means++ with Lloyd iterations; best of `starts` by inertia.
fn kmeans(x: &[&Vec<f64>], k: usize, rng: &mut ChaCha8Rng, starts: usize) -> Vec<Vec<f64>> {
    let mut best: (f64, Vec<Vec<f64>>) = (f64::INFINITY, vec![]);
    for _ in 0..starts {
        let mut c: Vec<Vec<f64>> = vec![x[rng.random_range(0..x.len())].clone()];
        let mut dist: Vec<f64> = x.iter().map(|r| d2(r, &c[0])).collect();
        while c.len() < k {
            let tot: f64 = dist.iter().sum();
            let mut u = rng.random::<f64>() * tot;
            let mut pick = x.len() - 1;
            for (i, d) in dist.iter().enumerate() {
                u -= d;
                if u <= 0.0 {
                    pick = i;
                    break;
                }
            }
            c.push(x[pick].clone());
            for (i, r) in x.iter().enumerate() {
                dist[i] = dist[i].min(d2(r, &c[c.len() - 1]));
            }
        }
        let dim = x[0].len();
        let mut lab = vec![usize::MAX; x.len()];
        for _ in 0..40 {
            let mut changed = false;
            for (i, r) in x.iter().enumerate() {
                let a = nearest_center(&c, r);
                if a != lab[i] {
                    lab[i] = a;
                    changed = true;
                }
            }
            let mut s = vec![vec![0.0; dim]; k];
            let mut m = vec![0usize; k];
            for (i, r) in x.iter().enumerate() {
                m[lab[i]] += 1;
                for d in 0..dim {
                    s[lab[i]][d] += r[d];
                }
            }
            for j in 0..k {
                if m[j] > 0 {
                    c[j] = s[j].iter().map(|v| v / m[j] as f64).collect();
                }
            }
            if !changed {
                break;
            }
        }
        let inertia: f64 = x.iter().map(|r| d2(r, &c[nearest_center(&c, r)])).sum();
        if inertia < best.0 {
            best = (inertia, c);
        }
    }
    best.1
}

/// Worst ratio of the density at the gap to the lower of the two peaks,
/// over all cluster pairs, measured along the line through the centres.
fn valley_ratio(x: &[&Vec<f64>], c: &[Vec<f64>]) -> f64 {
    let lab: Vec<usize> = x.iter().map(|r| nearest_center(c, r)).collect();
    let mut worst: f64 = 0.0;
    for a in 0..c.len() {
        for b in a + 1..c.len() {
            let dir: Vec<f64> = c[b].iter().zip(&c[a]).map(|(p, q)| p - q).collect();
            let len = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
            if len == 0.0 {
                return 1.0;
            }
            let proj: Vec<f64> = x
                .iter()
                .zip(&lab)
                .filter(|(_, &l)| l == a || l == b)
                .map(|(r, _)| {
                    r.iter()
                        .zip(&c[a])
                        .zip(&dir)
                        .map(|((v, o), d)| (v - o) * d)
                        .sum::<f64>()
                        / len
                })
                .collect();
            if proj.len() < 20 {
                return 1.0;
            }
            let lo = proj.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let nb = ((proj.len() as f64).sqrt() / 2.0).clamp(10.0, 40.0) as usize;
            let w = (hi - lo) / nb as f64;
            if w <= 0.0 {
                return 1.0;
            }
            let mut h = vec![0.0; nb];
            for v in &proj {
                h[(((v - lo) / w) as usize).min(nb - 1)] += 1.0;
            }
            let sm: Vec<f64> = (0..nb)
                .map(|i| {
                    let (s, e) = (i.saturating_sub(1), (i + 2).min(nb));
                    h[s..e].iter().sum::<f64>() / (e - s) as f64
                })
                .collect();
            // Bin of the midpoint between the two centres.
            let mid = (((len / 2.0 - lo) / w) as usize).min(nb - 1);
            let ia = (0..=mid)
                .max_by(|&i, &j| sm[i].total_cmp(&sm[j]))
                .unwrap_or(0);
            let ib = (mid..nb)
                .max_by(|&i, &j| sm[i].total_cmp(&sm[j]))
                .unwrap_or(nb - 1);
            let peak = sm[ia].min(sm[ib]);
            let gap = (ia..=ib).map(|i| sm[i]).fold(f64::INFINITY, f64::min);
            if peak > 0.0 {
                worst = worst.max(gap / peak);
            }
        }
    }
    worst
}

/// Rows drawn from one Gaussian with the mean and covariance of `x`.
fn gaussian_like(x: &[Vec<f64>], rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
    let n = x.len() as f64;
    let dim = x[0].len();
    let mut m = vec![0.0; dim];
    for r in x {
        for d in 0..dim {
            m[d] += r[d] / n;
        }
    }
    let mut c = vec![0.0; dim * dim];
    for r in x {
        for a in 0..dim {
            for b in 0..dim {
                c[a * dim + b] += (r[a] - m[a]) * (r[b] - m[b]) / n;
            }
        }
    }
    for a in 0..dim {
        c[a * dim + a] += 1e-9;
    }
    if !cholesky(&mut c, dim) {
        return x.to_vec();
    }
    (0..x.len())
        .map(|_| {
            let z: Vec<f64> = (0..dim).map(|_| crate::sim::gauss(rng)).collect();
            (0..dim)
                .map(|a| m[a] + (0..=a).map(|b| c[a * dim + b] * z[b]).sum::<f64>())
                .collect()
        })
        .collect()
}

/// Prediction strength of `k` clusters on one split.
fn prediction_strength(tr: &[&Vec<f64>], te: &[&Vec<f64>], k: usize, rng: &mut ChaCha8Rng) -> f64 {
    let ca = kmeans(tr, k, rng, 6);
    let cb = kmeans(te, k, rng, 6);
    let mut tab = vec![vec![0usize; k]; k];
    for r in te {
        tab[nearest_center(&cb, r)][nearest_center(&ca, r)] += 1;
    }
    tab.iter()
        .filter_map(|row| {
            let n: usize = row.iter().sum();
            if n < 2 {
                return None;
            }
            let same: usize = row.iter().map(|m| m * m.saturating_sub(1) / 2).sum();
            Some(same as f64 / (n * (n - 1) / 2) as f64)
        })
        .fold(1.0, f64::min)
}

fn clusters(
    input: &SeparabilityInput<'_>,
    subsets: &[Vec<usize>],
    p: &SeparabilityParams,
) -> SxResult<SeparabilityReport> {
    if p.k_max < 2 || p.ps_splits == 0 {
        return Err(SxError::InvalidInput(
            "k_max ≥ 2 and ps_splits ≥ 1 are required".into(),
        ));
    }
    let n = input.logs.len();
    if n == 0 {
        return Err(SxError::EmptyInput("no rows".into()));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
    let mut pick: Vec<usize> = (0..n).collect();
    for i in 0..n.min(p.max_rows) {
        let j = rng.random_range(i..n);
        pick.swap(i, j);
    }
    pick.truncate(p.max_rows);
    pick.sort_unstable();
    let rows: Vec<&[f64]> = pick.iter().map(|&i| input.logs[i].as_slice()).collect();
    let groups: Vec<i32> = pick.iter().map(|&i| input.groups[i]).collect();
    let ug: Vec<i32> = groups
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if ug.len() < 2 {
        return Err(SxError::InvalidInput(
            "need at least two groups (wells)".into(),
        ));
    }
    let nl = input.names.len();
    let reference = reference_set(&rows, nl, p.min_coverage);
    // Random halves of the groups, shared by all combinations.
    let halves: Vec<BTreeSet<i32>> = (0..p.ps_splits)
        .map(|_| {
            let mut g = ug.clone();
            for i in (1..g.len()).rev() {
                let j = rng.random_range(0..=i);
                g.swap(i, j);
            }
            g[..g.len() / 2].iter().copied().collect()
        })
        .collect();
    let n_all = rows.len() as f64;
    let jobs: Vec<(usize, usize)> = (0..subsets.len())
        .flat_map(|s| (2..=p.k_max).map(move |k| (s, k)))
        .collect();
    let scores: Vec<Option<ScoreRow>> = crate::runtime::install(|| {
        jobs.par_iter()
            .map(|&(s, k)| {
                let logs = &subsets[s];
                let idx: Vec<usize> = (0..rows.len())
                    .filter(|&i| complete(rows[i], logs))
                    .collect();
                let coverage = idx.len() as f64 / n_all;
                if coverage < p.min_coverage || idx.len() < 4 * k {
                    return None;
                }
                let mut x: Vec<Vec<f64>> = idx
                    .iter()
                    .map(|&i| logs.iter().map(|&j| rows[i][j]).collect())
                    .collect();
                standardize(&mut x);
                let mut r = ChaCha8Rng::seed_from_u64(p.seed ^ ((s as u64) << 8) ^ k as u64);
                let null = gaussian_like(&x, &mut r);
                let (mut ps, mut ps0) = (vec![], vec![]);
                for h in &halves {
                    let (a, b): (Vec<usize>, Vec<usize>) =
                        (0..idx.len()).partition(|&i| h.contains(&groups[idx[i]]));
                    if a.len() < 2 * k || b.len() < 2 * k {
                        continue;
                    }
                    for (src, out) in [(&x, &mut ps), (&null, &mut ps0)] {
                        let ta: Vec<&Vec<f64>> = a.iter().map(|&i| &src[i]).collect();
                        let tb: Vec<&Vec<f64>> = b.iter().map(|&i| &src[i]).collect();
                        out.push(prediction_strength(&ta, &tb, k, &mut r));
                    }
                }
                if ps.is_empty() {
                    return None;
                }
                let all: Vec<&Vec<f64>> = x.iter().collect();
                let centres = kmeans(&all, k, &mut r, 6);
                let valley = valley_ratio(&all, &centres);
                Some(ScoreRow {
                    k,
                    logs: logs.clone(),
                    n_rows: idx.len(),
                    coverage,
                    balanced_accuracy: f64::NAN,
                    min_recall: f64::NAN,
                    kappa: f64::NAN,
                    effective_classes: f64::NAN,
                    prediction_strength: ps.iter().sum::<f64>() / ps.len() as f64,
                    prediction_strength_null: ps0.iter().sum::<f64>() / ps0.len() as f64,
                    valley_ratio: valley,
                })
            })
            .collect()
    });
    let scores: Vec<ScoreRow> = scores.into_iter().flatten().collect();
    let recommended = scores
        .iter()
        .filter(|r| {
            r.prediction_strength >= p.ps_threshold
                && r.prediction_strength - r.prediction_strength_null >= p.ps_margin
                && r.valley_ratio <= p.valley_max
        })
        .map(|r| r.k)
        .max()
        .unwrap_or(1);
    let at_k: Vec<&ScoreRow> = scores
        .iter()
        .filter(|r| r.k == recommended.max(2))
        .collect();
    let top = at_k
        .iter()
        .map(|r| r.prediction_strength)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_logs = at_k
        .iter()
        .filter(|r| r.prediction_strength >= top - p.tol)
        .min_by(|a, b| {
            a.logs
                .len()
                .cmp(&b.logs.len())
                .then(b.prediction_strength.total_cmp(&a.prediction_strength))
        })
        .map(|r| r.logs.clone())
        .unwrap_or_default();
    let get = |logs: &[usize]| {
        at_k.iter()
            .find(|r| r.logs == logs)
            .map(|r| r.prediction_strength)
    };
    let importance = match get(&reference) {
        Some(base) if reference.len() > 1 => reference
            .iter()
            .filter_map(|&j| {
                let rest: Vec<usize> = reference.iter().copied().filter(|&x| x != j).collect();
                get(&rest).map(|v| LogImportance {
                    log: j,
                    drop: base - v,
                })
            })
            .collect(),
        _ => vec![],
    };
    Ok(SeparabilityReport {
        mode: "clusters".into(),
        names: input.names.to_vec(),
        levels: vec![],
        dropped_classes: vec![],
        reference_logs: reference,
        scores,
        recommended_k: recommended,
        best_logs,
        importance,
        confusion: vec![],
    })
}
