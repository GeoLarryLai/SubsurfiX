// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stochastic simulation on open Kansas Geological Survey (KGS) data (T34S R11W):
//! 5-fold leave-wells-out comparison of ordinary kriging (OK), sequential
//! Gaussian simulation (SGS), and `ml_propagation` (quantile LightGBM and
//! the in-house quantile regression forest, QRF) for density porosity, plus
//! sequential indicator simulation (SIS) of a two-class GR facies.
//! Coordinates: x, y (m) and depth relative to the Mississippian top.
//! "The source of this material is the Kansas Geological Survey website at
//! http://www.kgs.ku.edu/. All Rights Reserved."
//! Skips with a note when `SX_OPEN_DATA` is not set.

use std::collections::BTreeMap;

use sx_core::config::load_project;
use sx_core::grid::{
    experimental_variogram, fit_variogram, ordinary_kriging, KrigingParams, ModelKind,
};
use sx_core::ingest::run_ingest;
use sx_core::sim::{
    ml_propagation, sgs, sis, NormalScore, PropagationModel, PropagationParams, QrfParams,
    SgsParams, SisParams,
};
use sx_core::types::LogKind;
use sx_testutil::open_data::{kgs_dir, write_kgs_project};

/// Vertical exaggeration: 1 m of depth counts as this many metres laterally.
const ANISO: f64 = 50.0;
const BIN_FT: f64 = 4.0;
const FT: f64 = 0.3048;

struct Sample {
    well: usize,
    x: f64,
    y: f64,
    zrel_ft: f64,
    phi: f64,
    facies: i32,
}

fn r2(pred: &[f64], truth: &[f64]) -> f64 {
    let m = truth.iter().sum::<f64>() / truth.len() as f64;
    let ss_tot: f64 = truth.iter().map(|t| (t - m).powi(2)).sum();
    let ss_res: f64 = pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

#[test]
fn kgs_porosity_and_facies_simulation() {
    let Some(src) = kgs_dir() else {
        eprintln!("SX_OPEN_DATA not set; skipping");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let toml = write_kgs_project(tmp.path(), &src, 200, 10).unwrap();
    let cfg = load_project(&toml).unwrap();
    let out = run_ingest(&cfg.ingest, tmp.path()).unwrap();
    let xy: BTreeMap<i32, (f64, f64)> =
        out.wells.iter().map(|h| (h.id.0, (h.x_m, h.y_m))).collect();
    let top: BTreeMap<i32, f64> = out
        .picks
        .iter()
        .filter(|p| p.top_name == "Mississippian")
        .map(|p| (p.well.0, p.md_ft))
        .collect();

    // Bin RHOB-derived porosity (limestone matrix 2.71, fluid 1.0) and GR facies.
    let mut samples: Vec<Sample> = vec![];
    let mut wells = 0usize;
    for l in &out.logs {
        let Some(&t) = top.get(&l.id.0) else { continue };
        let (Some(rhob), Some(gr)) = (l.curves.get(&LogKind::Rhob), l.curves.get(&LogKind::Gr))
        else {
            continue;
        };
        let mut bins: BTreeMap<i64, (f64, usize, usize, usize)> = BTreeMap::new();
        for i in 0..l.grid.len {
            let z = l.grid.md_ft(i) - t;
            if !(-100.0..200.0).contains(&z) || !rhob[i].is_finite() || !gr[i].is_finite() {
                continue;
            }
            let b = bins.entry((z / BIN_FT).floor() as i64).or_default();
            b.0 += (2.71 - f64::from(rhob[i])) / 1.71;
            b.1 += 1;
            if gr[i] >= 75.0 {
                b.3 += 1
            } else {
                b.2 += 1
            }
        }
        let (x, y) = xy[&l.id.0];
        let before = samples.len();
        for (k, (s, n, clean, shaly)) in bins {
            if n >= 3 {
                samples.push(Sample {
                    well: wells,
                    x,
                    y,
                    zrel_ft: (k as f64 + 0.5) * BIN_FT,
                    phi: s / n as f64,
                    facies: i32::from(shaly > clean),
                });
            }
        }
        if samples.len() > before {
            wells += 1;
        }
    }
    eprintln!(
        "{wells} wells, {} porosity samples (4-ft bins, -100..200 ft around the Mississippian top)",
        samples.len()
    );
    assert!(wells >= 60);

    let pt = |s: &Sample| [s.x, s.y, s.zrel_ft * FT * ANISO];
    let names = [
        "OK",
        "SGS (20 real.)",
        "ml_propagation LightGBM",
        "ml_propagation QRF",
    ];
    let mut pred: Vec<Vec<f64>> = vec![vec![]; 4];
    let mut inside = [0usize; 4];
    let mut width = [0.0f64; 4];
    let mut truth: Vec<f64> = vec![];
    let (mut fac_ok, mut fac_base, mut fac_n, mut fac_prop1) = (0usize, 0usize, 0usize, 0.0f64);
    for fold in 0..5 {
        let (tr, te): (Vec<&Sample>, Vec<&Sample>) =
            samples.iter().partition(|s| s.well % 5 != fold);
        let tp: Vec<[f64; 3]> = tr.iter().map(|s| pt(s)).collect();
        let tv: Vec<f64> = tr.iter().map(|s| s.phi).collect();
        let xp: Vec<[f64; 3]> = te.iter().map(|s| pt(s)).collect();
        let sub = |v: &[f64]| -> (Vec<[f64; 3]>, Vec<f64>) {
            let step = (tp.len() / 1500).max(1);
            (
                (0..tp.len()).step_by(step).map(|i| tp[i]).collect(),
                (0..tp.len()).step_by(step).map(|i| v[i]).collect(),
            )
        };
        // OK on raw porosity.
        let (sp, sv) = sub(&tv);
        let m_raw = fit_variogram(
            &experimental_variogram(&sp, &sv, 250.0, 16).unwrap(),
            ModelKind::Spherical,
        )
        .unwrap();
        let ok = ordinary_kriging(
            &tp,
            &tv,
            &xp,
            &KrigingParams {
                model: m_raw,
                max_neighbors: 16,
                max_distance: f64::INFINITY,
            },
        )
        .unwrap();
        // SGS on normal scores.
        let ns = NormalScore::fit(&tv).unwrap();
        let nv: Vec<f64> = tv.iter().map(|v| ns.forward(*v)).collect();
        let (sp, snv) = sub(&nv);
        let m_ns = fit_variogram(
            &experimental_variogram(&sp, &snv, 250.0, 16).unwrap(),
            ModelKind::Spherical,
        )
        .unwrap();
        let reals = sgs(
            &tp,
            &tv,
            &xp,
            &SgsParams {
                model: m_ns,
                max_neighbors: 16,
                max_distance: f64::INFINITY,
                n_realizations: 20,
                seed: 7 + fold as u64,
            },
        )
        .unwrap();
        // ml_propagation: x, y, depth below the top.
        let fx = |s: &&Sample| vec![s.x, s.y, s.zrel_ft];
        let trx: Vec<Vec<f64>> = tr.iter().map(fx).collect();
        let tex: Vec<Vec<f64>> = te.iter().map(fx).collect();
        let lg = ml_propagation(
            &trx,
            &tv,
            &tex,
            &PropagationParams {
                seed: fold as u64,
                ..PropagationParams::default()
            },
        )
        .unwrap();
        let qp = PropagationParams {
            model: PropagationModel::QuantileForest,
            qrf: QrfParams {
                n_trees: 100,
                min_leaf: 10,
                ..QrfParams::default()
            },
            seed: fold as u64,
            ..PropagationParams::default()
        };
        let qf = ml_propagation(&trx, &tv, &tex, &qp).unwrap();

        for (i, s) in te.iter().enumerate() {
            truth.push(s.phi);
            let sd = ok.variance[i].sqrt();
            let mut r: Vec<f64> = reals.iter().map(|x| x[i]).collect();
            let rows = [
                (
                    ok.estimate[i],
                    ok.estimate[i] - 1.2816 * sd,
                    ok.estimate[i] + 1.2816 * sd,
                ),
                (
                    r.iter().sum::<f64>() / r.len() as f64,
                    quantile(&mut r, 0.1),
                    quantile(&mut r, 0.9),
                ),
                (lg[i][1], lg[i][0], lg[i][2]),
                (qf[i][1], qf[i][0], qf[i][2]),
            ];
            for (m, (p, lo, hi)) in rows.into_iter().enumerate() {
                pred[m].push(p);
                inside[m] += usize::from(s.phi >= lo && s.phi <= hi);
                width[m] += hi - lo;
            }
        }

        // SIS for facies, shared indicator variogram of class 1.
        let codes: Vec<i32> = tr.iter().map(|s| s.facies).collect();
        let ind: Vec<f64> = codes.iter().map(|c| f64::from(*c)).collect();
        let (sp, si) = sub(&ind);
        let m_ind = fit_variogram(
            &experimental_variogram(&sp, &si, 250.0, 16).unwrap(),
            ModelKind::Spherical,
        )
        .unwrap();
        let fr = sis(
            &tp,
            &codes,
            &xp,
            &SisParams {
                models: vec![],
                default_model: m_ind,
                max_neighbors: 16,
                max_distance: f64::INFINITY,
                n_realizations: 20,
                seed: 3 + fold as u64,
            },
        )
        .unwrap();
        let p1 = ind.iter().sum::<f64>() / ind.len() as f64;
        let base = i32::from(p1 > 0.5);
        for (i, s) in te.iter().enumerate() {
            let ones = fr.iter().filter(|r| r[i] == 1).count();
            let mode = i32::from(ones * 2 > fr.len());
            fac_ok += usize::from(mode == s.facies);
            fac_base += usize::from(base == s.facies);
            fac_prop1 += ones as f64 / fr.len() as f64;
            fac_n += 1;
        }
    }
    let n = truth.len() as f64;
    let sd = (truth
        .iter()
        .map(|t| (t - truth.iter().sum::<f64>() / n).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    eprintln!("held-out samples {n}, porosity sd {sd:.4}");
    for m in 0..4 {
        let rmse = (pred[m]
            .iter()
            .zip(&truth)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        eprintln!(
            "SIM {:26} r2 {:5.2}  RMSE {:.4}  P10-P90 coverage {:.2}  mean width {:.4}",
            names[m],
            r2(&pred[m], &truth),
            rmse,
            inside[m] as f64 / n,
            width[m] / n
        );
    }
    let true1 = samples.iter().filter(|s| s.facies == 1).count() as f64 / samples.len() as f64;
    eprintln!(
        "SIM SIS facies: most-frequent accuracy {:.2} (global-majority baseline {:.2}); simulated shaly share {:.2} (data {:.2})",
        fac_ok as f64 / fac_n as f64, fac_base as f64 / fac_n as f64, fac_prop1 / fac_n as f64, true1
    );
    for m in 0..4 {
        let c = inside[m] as f64 / n;
        assert!((0.3..0.99).contains(&c), "{} coverage {c}", names[m]);
    }
    assert!(
        r2(&pred[1], &truth) > 0.0,
        "SGS mean should beat the global mean"
    );
    assert!(fac_ok >= fac_base * 9 / 10);
}
