// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Distribution check against spatial neighbors (the "JSD check").
//!
//! The statistic is a symmetrized k-NN Kullback–Leibler estimate (no bins).
//! Called by in stage 4 (flag only). Values are native space.

use std::collections::{BTreeMap, BTreeSet};

use rand::Rng;
use rayon::prelude::*;

use super::stats::{mad_n, mean_std};
use crate::config::QcConfig;
use crate::hash::stream_seed;
use crate::types::{LogKind, WellId};
use crate::util::median;

/// Distance to the `k`-th nearest value of `sorted` from `p` (`k >= 1`),
/// skipping the entry at index `exclude`. `None` if fewer than `k` candidates.
///
/// O(log n + k): binary search, then a two-sided merge outwards.
fn kth_nearest(sorted: &[f64], p: f64, k: usize, exclude: Option<usize>) -> Option<f64> {
    if k == 0 {
        return None;
    }
    let start = sorted.partition_point(|&x| x < p);
    let mut left = start; // next left candidate is left - 1
    let mut right = start; // next right candidate is right
    let mut taken = 0usize;
    loop {
        if exclude == Some(right) {
            right += 1;
            continue;
        }
        let dl = if left > 0 {
            sorted.get(left - 1).map(|&x| p - x)
        } else {
            None
        };
        let dr = sorted.get(right).map(|&x| x - p);
        let d = match (dl, dr) {
            (Some(a), Some(b)) => {
                if a <= b {
                    left -= 1;
                    a
                } else {
                    right += 1;
                    b
                }
            }
            (Some(a), None) => {
                left -= 1;
                a
            }
            (None, Some(b)) => {
                right += 1;
                b
            }
            (None, None) => return None,
        };
        taken += 1;
        if taken == k {
            return Some(d);
        }
    }
}

/// 1-D k-NN Kullback–Leibler estimate `KL(X‖Y)`.
///
/// Inputs are sorted ascending and finite. For each `x_i`, `r_i` is the
/// `k`-th smallest `|x_i - x_j|` over `j != i`; `s_i` is the `k`-th smallest
/// `|x_i - y_j|` over all `j`; both are floored at `floor`.
/// `KL = mean_i ln(s_i / r_i) + ln(|Y| / (|X| - 1))`. Returns NaN if
/// `|X| <= k`, `|Y| < k`, or `k == 0`. Unit: nats. O((|X|) (log n + k)).
pub fn kl_knn_1d(x_sorted: &[f64], y_sorted: &[f64], k: usize, floor: f64) -> f64 {
    let nx = x_sorted.len();
    let ny = y_sorted.len();
    if k == 0 || nx <= k || ny < k {
        return f64::NAN;
    }
    let mut sum = 0.0f64;
    for (i, &x) in x_sorted.iter().enumerate() {
        let r = kth_nearest(x_sorted, x, k, Some(i));
        let s = kth_nearest(y_sorted, x, k, None);
        match (r, s) {
            (Some(r), Some(s)) => sum += (s.max(floor) / r.max(floor)).ln(),
            _ => return f64::NAN,
        }
    }
    sum / nx as f64 + (ny as f64 / (nx as f64 - 1.0)).ln()
}

/// Divergence result of one well and log.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivRow {
    /// Well id.
    pub well: WellId,
    /// Log.
    pub log: LogKind,
    /// Mean of the finite pair scores to the neighbors (z units, nats); NaN if none.
    pub score: f64,
    /// Neighbors with a finite pair score.
    pub n_neighbors: usize,
    /// `n_neighbors >= jsd_min_neighbors`.
    pub valid: bool,
    /// Leave-one-out outlier flag.
    pub outlier: bool,
}

/// Subsample `values` to `m` without replacement (partial Fisher–Yates) and
/// return the picked values in index order.
fn subsample(values: &[f64], m: usize, seed: u64, well: WellId) -> Vec<f64> {
    let n = values.len();
    if n <= m {
        return values.to_vec();
    }
    use rand::SeedableRng;
    let mut rng =
        rand_chacha::ChaCha8Rng::seed_from_u64(stream_seed(seed, "normalize.jsd", well.0));
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..m {
        let j = i + (rng.next_u64() % (n - i) as u64) as usize;
        idx.swap(i, j);
    }
    let mut keep: Vec<usize> = idx.into_iter().take(m).collect();
    keep.sort_unstable();
    keep.iter()
        .filter_map(|&i| values.get(i).copied())
        .collect()
}

/// Distribution check of one log: each well against its spatial neighbors.
///
/// `channels` holds each well's native values (non-finite entries are
/// dropped). One row per input well, sorted by `WellId`.
///
/// 1. A well is scoreable with at least `jsd_k + 5` finite values; a longer
///    vector is subsampled to `jsd_max_count` without replacement (partial
///    Fisher–Yates, `ChaCha8Rng` seeded from `stream_seed(seed,
///    "normalize.jsd", well)`, kept indices sorted).
/// 2. Values are z-scored with the pooled mean and std (ddof 1) of all
///    scoreable wells; if the std is not finite or `<= 0`, no well is valid.
/// 3. `kl_knn_1d` with `k = jsd_k` and floor `jsd_floor`.
/// 4. Pair score = `max(0, (KL(a‖b) + KL(b‖a)) / 2)`, NaN if either is NaN;
///    each unordered pair is computed once.
/// 5. Neighbors: the `jsd_neighbors` nearest scoreable wells with XY
///    (Euclidean, ties by `WellId`). `score` = mean of the finite pair
///    scores, `n_neighbors` = their count, `valid` when `n_neighbors >=
///    jsd_min_neighbors`. Wells without XY: NaN, not valid.
/// 6. Fewer than `jsd_min_wells` valid wells: no outliers.
/// 7. Outlier: `score > median(S) + jsd_mad_mult * max(MAD_n(S), jsd_floor)`
///    where `S` holds the scores of the other valid wells.
///
/// NaN rule: skip. Deterministic for any input order and fixed `seed`.
pub fn jsd_scores(
    log: LogKind,
    channels: &[(WellId, Vec<f64>)],
    xy: &BTreeMap<WellId, (f64, f64)>,
    cfg: &QcConfig,
    seed: u64,
) -> Vec<DivRow> {
    let mut order: Vec<usize> = (0..channels.len()).collect();
    order.sort_by_key(|&i| channels.get(i).map(|c| c.0));
    let wells: Vec<WellId> = order
        .iter()
        .filter_map(|&i| channels.get(i))
        .map(|c| c.0)
        .collect();
    let n = wells.len();
    let mut rows: Vec<DivRow> = wells
        .iter()
        .map(|&w| DivRow {
            well: w,
            log,
            score: f64::NAN,
            n_neighbors: 0,
            valid: false,
            outlier: false,
        })
        .collect();

    // Steps 1-2: scoreable wells, subsampled, pooled z-score.
    let need = cfg.jsd_k + 5;
    let mut samples: Vec<Option<Vec<f64>>> = order
        .par_iter()
        .map(|&i| {
            let (id, values) = channels.get(i)?;
            let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
            if finite.len() < need {
                return None;
            }
            Some(subsample(&finite, cfg.jsd_max_count, seed, *id))
        })
        .collect();
    let pooled: Vec<f64> = samples.iter().flatten().flatten().copied().collect();
    let (_, mean, std) = mean_std(&pooled);
    if !std.is_finite() || std <= 0.0 {
        return rows;
    }
    for s in samples.iter_mut().flatten() {
        for v in s.iter_mut() {
            *v = (*v - mean) / std;
        }
        s.sort_by(f64::total_cmp);
    }

    // Step 5 neighbor lists (indices into `wells`, sorted by (distance, id)).
    let pos: Vec<Option<(f64, f64)>> = wells
        .iter()
        .map(|w| {
            xy.get(w)
                .copied()
                .filter(|(x, y)| x.is_finite() && y.is_finite())
        })
        .collect();
    let scoreable: Vec<usize> = (0..n)
        .filter(|&i| {
            samples.get(i).is_some_and(Option::is_some) && pos.get(i).is_some_and(Option::is_some)
        })
        .collect();
    let neighbors: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let me = match (samples.get(i), pos.get(i).copied().flatten()) {
                (Some(Some(_)), Some(p)) => p,
                _ => return Vec::new(),
            };
            let mut ranked: Vec<(f64, usize)> = scoreable
                .iter()
                .filter(|&&j| j != i)
                .filter_map(|&j| {
                    let (x, y) = pos.get(j).copied().flatten()?;
                    Some(((x - me.0).hypot(y - me.1), j))
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            ranked
                .into_iter()
                .take(cfg.jsd_neighbors)
                .map(|(_, j)| j)
                .collect()
        })
        .collect();

    // Steps 3-4: each unordered pair once.
    let pairs: BTreeSet<(usize, usize)> = neighbors
        .iter()
        .enumerate()
        .flat_map(|(i, nb)| nb.iter().map(move |&j| (i.min(j), i.max(j))))
        .collect();
    let pairs: Vec<(usize, usize)> = pairs.into_iter().collect();
    let scored: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let score = match (
                samples.get(a).and_then(|s| s.as_deref()),
                samples.get(b).and_then(|s| s.as_deref()),
            ) {
                (Some(xa), Some(xb)) => {
                    let kab = kl_knn_1d(xa, xb, cfg.jsd_k, cfg.jsd_floor);
                    let kba = kl_knn_1d(xb, xa, cfg.jsd_k, cfg.jsd_floor);
                    if kab.is_nan() || kba.is_nan() {
                        f64::NAN
                    } else {
                        (0.5 * (kab + kba)).max(0.0)
                    }
                }
                _ => f64::NAN,
            };
            ((a, b), score)
        })
        .collect();
    let table: BTreeMap<(usize, usize), f64> = scored.into_iter().collect();

    for (i, row) in rows.iter_mut().enumerate() {
        let nb = match neighbors.get(i) {
            Some(nb) => nb,
            None => continue,
        };
        let vals: Vec<f64> = nb
            .iter()
            .filter_map(|&j| table.get(&(i.min(j), i.max(j))).copied())
            .filter(|s| s.is_finite())
            .collect();
        row.n_neighbors = vals.len();
        row.valid = vals.len() >= cfg.jsd_min_neighbors && !vals.is_empty();
        if !vals.is_empty() {
            row.score = vals.iter().sum::<f64>() / vals.len() as f64;
        }
    }

    // Steps 6-7: leave-one-out outliers.
    let valid: Vec<usize> = (0..n)
        .filter(|&i| rows.get(i).is_some_and(|r| r.valid))
        .collect();
    if valid.len() >= cfg.jsd_min_wells {
        let flags: Vec<bool> = valid
            .iter()
            .map(|&i| {
                let own = rows.get(i).map_or(f64::NAN, |r| r.score);
                let others: Vec<f64> = valid
                    .iter()
                    .filter(|&&j| j != i)
                    .filter_map(|&j| rows.get(j).map(|r| r.score))
                    .collect();
                own > median(&others)
                    + cfg.jsd_mad_mult * mad_n(&others, cfg.jsd_floor).max(cfg.jsd_floor)
            })
            .collect();
        for (&i, f) in valid.iter().zip(flags) {
            if let Some(r) = rows.get_mut(i) {
                r.outlier = f;
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Example 6.
    #[test]
    fn kl_example() {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = (0..10).map(|i| f64::from(2 * i)).collect();
        let kxy = kl_knn_1d(&x, &y, 5, 1e-12);
        let kyx = kl_knn_1d(&y, &x, 5, 1e-12);
        assert!(rel(kxy, 0.48842127705209) < 1e-9, "{kxy}");
        assert!(rel(kyx, -0.31812394675845) < 1e-9, "{kyx}");
        let pair = (0.5 * (kxy + kyx)).max(0.0);
        assert!(rel(pair, 0.08514866514682) < 1e-9, "{pair}");
        // z-scoring both with a common mean/std leaves it unchanged
        let all: Vec<f64> = x.iter().chain(y.iter()).copied().collect();
        let (_, m, s) = mean_std(&all);
        let zx: Vec<f64> = x.iter().map(|v| (v - m) / s).collect();
        let zy: Vec<f64> = y.iter().map(|v| (v - m) / s).collect();
        let zxy = kl_knn_1d(&zx, &zy, 5, 1e-12);
        let zyx = kl_knn_1d(&zy, &zx, 5, 1e-12);
        assert!(rel(0.5 * (zxy + zyx), 0.08514866514682) < 1e-9);
    }

    #[test]
    fn kl_nan_rules() {
        let x: Vec<f64> = (0..5).map(f64::from).collect();
        let y: Vec<f64> = (0..4).map(f64::from).collect();
        assert!(kl_knn_1d(&x, &y, 5, 1e-12).is_nan()); // |X| <= k
        let x6: Vec<f64> = (0..6).map(f64::from).collect();
        assert!(kl_knn_1d(&x6, &y, 5, 1e-12).is_nan()); // |Y| < k
        assert!(kl_knn_1d(&x6, &x6, 0, 1e-12).is_nan());
    }

    /// Brute-force check of the k-th nearest helper.
    #[test]
    fn kth_matches_brute_force() {
        let v: Vec<f64> = (0..40).map(|i| ((i * 37) % 41) as f64 * 0.5).collect();
        let mut sorted = v.clone();
        sorted.sort_by(f64::total_cmp);
        for (i, &p) in sorted.iter().enumerate() {
            for k in 1..6 {
                let mut d: Vec<f64> = sorted
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, x)| (x - p).abs())
                    .collect();
                d.sort_by(f64::total_cmp);
                assert_eq!(kth_nearest(&sorted, p, k, Some(i)), Some(d[k - 1]));
                let mut d2: Vec<f64> = sorted.iter().map(|x| (x - (p + 0.25)).abs()).collect();
                d2.sort_by(f64::total_cmp);
                assert_eq!(kth_nearest(&sorted, p + 0.25, k, None), Some(d2[k - 1]));
            }
        }
    }

    type Chan = (WellId, Vec<f64>);
    type Xy = BTreeMap<WellId, (f64, f64)>;

    fn lcg_normal(state: &mut u64) -> f64 {
        // sum of 12 uniforms - 6
        let mut s = 0.0;
        for _ in 0..12 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s += (*state >> 11) as f64 / 9007199254740992.0;
        }
        s - 6.0
    }

    fn fleet(n: usize, odd: Option<usize>) -> (Vec<Chan>, Xy) {
        let mut st = 12345u64;
        let mut ch = Vec::new();
        let mut xy = BTreeMap::new();
        for i in 0..n {
            let (mu, sd) = if Some(i) == odd {
                (4.0, 0.4)
            } else {
                (0.0, 1.0)
            };
            let v: Vec<f64> = (0..400).map(|_| mu + sd * lcg_normal(&mut st)).collect();
            ch.push((WellId(i as i32), v));
            xy.insert(
                WellId(i as i32),
                ((i % 5) as f64 * 1000.0, (i / 5) as f64 * 1000.0),
            );
        }
        (ch, xy)
    }

    #[test]
    fn one_distribution_outlier() {
        let (ch, xy) = fleet(20, Some(7));
        let cfg = QcConfig::default();
        let rows = jsd_scores(LogKind::Gr, &ch, &xy, &cfg, 0);
        assert_eq!(rows.len(), 20);
        let out: Vec<i32> = rows
            .iter()
            .filter(|r| r.outlier)
            .map(|r| r.well.0)
            .collect();
        assert_eq!(out, vec![7]);
        assert!(rows.iter().all(|r| r.valid && r.n_neighbors == 10));
        // any input order
        let mut rev = ch.clone();
        rev.reverse();
        let rows_rev = jsd_scores(LogKind::Gr, &rev, &xy, &cfg, 0);
        assert_eq!(rows, rows_rev);
    }

    #[test]
    fn subsample_is_seed_stable() {
        let (mut ch, xy) = fleet(12, None);
        let cfg = QcConfig {
            jsd_max_count: 100,
            ..QcConfig::default()
        };
        let a = jsd_scores(LogKind::Gr, &ch, &xy, &cfg, 7);
        let b = jsd_scores(LogKind::Gr, &ch, &xy, &cfg, 7);
        assert_eq!(a, b);
        let c = jsd_scores(LogKind::Gr, &ch, &xy, &cfg, 8);
        assert_ne!(a, c);
        // draw is also per-well stable: reordering does not change a well's pick
        ch.reverse();
        let d = jsd_scores(LogKind::Gr, &ch, &xy, &cfg, 7);
        assert_eq!(a, d);
        let s1 = subsample(&ch[0].1, 100, 7, ch[0].0);
        let s2 = subsample(&ch[0].1, 100, 7, ch[0].0);
        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 100);
    }

    #[test]
    fn gates() {
        let (mut ch, mut xy) = fleet(8, None);
        let cfg = QcConfig::default();
        // too short well is not scoreable
        ch[0].1.truncate(9);
        // well without xy
        xy.remove(&WellId(1));
        let rows = jsd_scores(LogKind::Gr, &ch, &xy, &cfg, 0);
        assert!(rows[0].score.is_nan() && !rows[0].valid && rows[0].n_neighbors == 0);
        assert!(rows[1].score.is_nan() && !rows[1].valid);
        // fewer than jsd_min_wells valid: no outliers
        let (ch3, xy3) = fleet(4, Some(0));
        let rows3 = jsd_scores(LogKind::Gr, &ch3, &xy3, &cfg, 0);
        assert!(rows3.iter().all(|r| !r.outlier));
        // constant data: pooled std 0 -> nothing valid
        let flat: Vec<(WellId, Vec<f64>)> = (0..8).map(|i| (WellId(i), vec![1.0; 50])).collect();
        let rows_flat = jsd_scores(LogKind::Gr, &flat, &xy3, &cfg, 0);
        assert!(rows_flat.iter().all(|r| !r.valid));
    }
}
