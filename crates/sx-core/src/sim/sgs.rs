// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Sequential Gaussian simulation (SGS), ordinary-kriging based.
//!
//! 1. Normal-score transform the data.
//! 2. Per realization, visit targets on a random path; at each, krige the
//!    normal score from the nearest data and previously simulated targets
//!    (ordinary kriging, variogram of the normal scores), draw
//!    `mean + σ·N(0, 1)`, and add it to the conditioning set.
//! 3. Back-transform.
//!
//! Permeability: pass log10(k) and take 10^x of the output.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{SxError, SxResult};
use crate::grid::interp::{krige_one, Drift};
use crate::grid::VariogramModel;
use crate::sim::{auto_cell, gauss, random_path, Buckets, NormalScore};

/// SGS settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgsParams {
    /// Variogram of the normal scores (sill near 1).
    pub model: VariogramModel,
    /// Conditioning points per target (data plus simulated).
    pub max_neighbors: usize,
    /// Search radius.
    pub max_distance: f64,
    /// Realizations.
    pub n_realizations: usize,
    /// Seed; realization `r` uses `seed + r`.
    pub seed: u64,
}

/// Simulate at `targets`. Returns `[realization][target]` in data units.
/// Targets on a datum take its value. NaN data are skipped.
///
/// # Errors
/// [`SxError::Shape`] on a length mismatch; [`SxError::EmptyInput`] with
/// fewer than 2 distinct data values; [`SxError::InvalidInput`] for zero
/// neighbours or realizations.
pub fn sgs(
    points: &[[f64; 3]],
    values: &[f64],
    targets: &[[f64; 3]],
    p: &SgsParams,
) -> SxResult<Vec<Vec<f64>>> {
    if points.len() != values.len() {
        return Err(SxError::Shape(format!(
            "{} points vs {} values",
            points.len(),
            values.len()
        )));
    }
    if p.max_neighbors == 0 || p.n_realizations == 0 {
        return Err(SxError::InvalidInput(
            "max_neighbors and n_realizations must be ≥ 1".into(),
        ));
    }
    let ns = NormalScore::fit(values)?;
    let data: Vec<([f64; 3], f64)> = points
        .iter()
        .zip(values)
        .filter(|(q, v)| v.is_finite() && q.iter().all(|c| c.is_finite()))
        .map(|(q, v)| (*q, ns.forward(*v)))
        .collect();
    let data_raw: Vec<f64> = points
        .iter()
        .zip(values)
        .filter(|(q, v)| v.is_finite() && q.iter().all(|c| c.is_finite()))
        .map(|(_, v)| *v)
        .collect();
    let mut all: Vec<[f64; 3]> = data.iter().map(|d| d.0).collect();
    all.extend_from_slice(targets);
    let cell = auto_cell(&all);
    let g = |h: f64| p.model.gamma(h);
    let sill = p.model.nugget + p.model.psill;
    crate::runtime::install(|| {
        (0..p.n_realizations)
            .into_par_iter()
            .map(|r| {
                let mut rng = ChaCha8Rng::seed_from_u64(p.seed.wrapping_add(r as u64));
                let mut b = Buckets::new(cell);
                let mut val: Vec<f64> = Vec::with_capacity(data.len() + targets.len());
                for (q, y) in &data {
                    b.push(*q);
                    val.push(*y);
                }
                let mut out = vec![f64::NAN; targets.len()];
                for t in random_path(targets.len(), &mut rng) {
                    let tp = targets[t];
                    let z = gauss(&mut rng);
                    if tp.iter().any(|c| !c.is_finite()) {
                        continue;
                    }
                    let nb = b.nearest(&tp, p.max_neighbors, p.max_distance);
                    let y = match nb.first() {
                        None => z * sill.sqrt(),
                        Some(&(h, i)) if h == 0.0 && i < data.len() => {
                            out[t] = data_raw[i];
                            b.push(tp);
                            val.push(val[i]);
                            continue;
                        }
                        Some(&(0.0, i)) => val[i],
                        _ => {
                            let pts: Vec<[f64; 3]> =
                                nb.iter().map(|&(_, i)| b_point(&b, i)).collect();
                            let vs: Vec<f64> = nb.iter().map(|&(_, i)| val[i]).collect();
                            let local: Vec<(f64, usize)> =
                                nb.iter().enumerate().map(|(k, &(d, _))| (d, k)).collect();
                            match krige_one(&pts, &vs, &tp, &local, &g, Drift::None) {
                                Some((m, v)) => m + v.max(0.0).sqrt() * z,
                                None => vs[0],
                            }
                        }
                    };
                    b.push(tp);
                    val.push(y);
                    out[t] = ns.back(y);
                }
                Ok(out)
            })
            .collect()
    })
}

fn b_point(b: &Buckets, i: usize) -> [f64; 3] {
    b.point(i)
}
