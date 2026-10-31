// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Sequential indicator simulation (SIS) of facies codes.
//!
//! Per target on a random path: for each facies `f`, ordinary indicator
//! kriging of `1[code = f]` from the nearest data and simulated targets
//! (one shared search; the variogram of `f`, or the shared one); clip to
//! [0, 1] and rescale to sum 1 (order-relation correction; global
//! proportions if all are zero); draw a code; add it to the conditioning set.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{SxError, SxResult};
use crate::grid::interp::{krige_one, Drift};
use crate::grid::VariogramModel;
use crate::sim::{auto_cell, random_path, Buckets};

/// SIS settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SisParams {
    /// Indicator variogram per facies code; codes without an entry use
    /// `default_model`.
    pub models: Vec<(i32, VariogramModel)>,
    /// Shared indicator variogram.
    pub default_model: VariogramModel,
    /// Conditioning points per target.
    pub max_neighbors: usize,
    /// Search radius.
    pub max_distance: f64,
    /// Realizations.
    pub n_realizations: usize,
    /// Seed; realization `r` uses `seed + r`.
    pub seed: u64,
}

/// Simulate codes at `targets`: `[realization][target]`. Negative data codes
/// are skipped; targets on a datum take its code.
///
/// # Errors
/// [`SxError::Shape`] on a length mismatch; [`SxError::EmptyInput`] with no
/// valid codes; [`SxError::InvalidInput`] for zero neighbours or realizations.
pub fn sis(
    points: &[[f64; 3]],
    codes: &[i32],
    targets: &[[f64; 3]],
    p: &SisParams,
) -> SxResult<Vec<Vec<i32>>> {
    if points.len() != codes.len() {
        return Err(SxError::Shape(format!(
            "{} points vs {} codes",
            points.len(),
            codes.len()
        )));
    }
    if p.max_neighbors == 0 || p.n_realizations == 0 {
        return Err(SxError::InvalidInput(
            "max_neighbors and n_realizations must be ≥ 1".into(),
        ));
    }
    let data: Vec<([f64; 3], i32)> = points
        .iter()
        .zip(codes)
        .filter(|(q, c)| **c >= 0 && q.iter().all(|v| v.is_finite()))
        .map(|(q, c)| (*q, *c))
        .collect();
    if data.is_empty() {
        return Err(SxError::EmptyInput("no facies data".into()));
    }
    let mut cats: Vec<i32> = data.iter().map(|d| d.1).collect();
    cats.sort_unstable();
    cats.dedup();
    let global: Vec<f64> = cats
        .iter()
        .map(|c| data.iter().filter(|d| d.1 == *c).count() as f64 / data.len() as f64)
        .collect();
    let models: Vec<VariogramModel> = cats
        .iter()
        .map(|c| {
            p.models
                .iter()
                .find(|m| m.0 == *c)
                .map_or(p.default_model, |m| m.1)
        })
        .collect();
    let mut all: Vec<[f64; 3]> = data.iter().map(|d| d.0).collect();
    all.extend_from_slice(targets);
    let cell = auto_cell(&all);
    crate::runtime::install(|| {
        (0..p.n_realizations)
            .into_par_iter()
            .map(|r| {
                let mut rng = ChaCha8Rng::seed_from_u64(p.seed.wrapping_add(r as u64));
                let mut b = Buckets::new(cell);
                let mut val: Vec<i32> = vec![];
                for (q, c) in &data {
                    b.push(*q);
                    val.push(*c);
                }
                let mut out = vec![-1; targets.len()];
                for t in random_path(targets.len(), &mut rng) {
                    let tp = targets[t];
                    let u: f64 = rng.random::<f64>();
                    if tp.iter().any(|c| !c.is_finite()) {
                        continue;
                    }
                    let nb = b.nearest(&tp, p.max_neighbors, p.max_distance);
                    let code = if let Some(&(_, i)) = nb.first().filter(|x| x.0 == 0.0) {
                        val[i]
                    } else {
                        let probs = if nb.is_empty() {
                            global.clone()
                        } else {
                            let pts: Vec<[f64; 3]> = nb.iter().map(|&(_, i)| b.point(i)).collect();
                            let local: Vec<(f64, usize)> =
                                nb.iter().enumerate().map(|(k, &(d, _))| (d, k)).collect();
                            let mut pr: Vec<f64> = cats
                                .iter()
                                .zip(&models)
                                .zip(&global)
                                .map(|((c, m), g0)| {
                                    let ind: Vec<f64> = nb
                                        .iter()
                                        .map(|&(_, i)| f64::from(u8::from(val[i] == *c)))
                                        .collect();
                                    let g = |h: f64| m.gamma(h);
                                    krige_one(&pts, &ind, &tp, &local, &g, Drift::None)
                                        .map_or(*g0, |e| e.0)
                                        .clamp(0.0, 1.0)
                                })
                                .collect();
                            let s: f64 = pr.iter().sum();
                            if s > 0.0 {
                                pr.iter_mut().for_each(|x| *x /= s);
                                pr
                            } else {
                                global.clone()
                            }
                        };
                        let mut acc = 0.0;
                        let mut pick = *cats.last().expect("non-empty");
                        for (c, pr) in cats.iter().zip(&probs) {
                            acc += pr;
                            if u < acc {
                                pick = *c;
                                break;
                            }
                        }
                        pick
                    };
                    b.push(tp);
                    val.push(code);
                    out[t] = code;
                }
                Ok(out)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::ModelKind;

    #[test]
    fn proportions_and_conditioning() {
        // Left half facies 0, right half facies 1.
        let pts: Vec<[f64; 3]> = (0..100)
            .map(|i| [(i % 10) as f64 * 10.0, (i / 10) as f64 * 10.0, 0.0])
            .collect();
        let codes: Vec<i32> = pts.iter().map(|q| i32::from(q[0] >= 50.0)).collect();
        let targets: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                [
                    (i % 20) as f64 * 4.7 + 1.0,
                    (i / 20) as f64 * 9.0 + 2.0,
                    0.0,
                ]
            })
            .chain([pts[7]])
            .collect();
        let m = VariogramModel {
            kind: ModelKind::Spherical,
            nugget: 0.0,
            psill: 0.25,
            range: 60.0,
        };
        let p = SisParams {
            models: vec![],
            default_model: m,
            max_neighbors: 12,
            max_distance: f64::INFINITY,
            n_realizations: 10,
            seed: 4,
        };
        let r = sis(&pts, &codes, &targets, &p).unwrap();
        let mut agree = 0usize;
        let mut total = 0usize;
        for real in &r {
            assert_eq!(real[200], codes[7]);
            for (t, c) in targets.iter().zip(real) {
                if (t[0] - 47.5).abs() > 15.0 {
                    total += 1;
                    agree += usize::from(*c == i32::from(t[0] >= 50.0));
                }
            }
        }
        assert!(agree as f64 / total as f64 > 0.9, "{agree}/{total}");
    }
}
