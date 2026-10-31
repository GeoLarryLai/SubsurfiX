// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Grid checks on open Kansas Geological Survey (KGS) data (T34S R11W):
//! log-to-cell upscaling on a regular grid, variogram fitting, and
//! leave-one-out cross-validation of inverse distance weighting (IDW) vs
//! ordinary kriging (OK) on the Mississippian structure top.
//! "The source of this material is the Kansas Geological Survey website at
//! http://www.kgs.ku.edu/. All Rights Reserved."
//! Skips with a note when `SX_OPEN_DATA` is not set.

use std::collections::BTreeMap;

use sx_core::config::load_project;
use sx_core::grid::{
    calibrate_sill, experimental_variogram, fit_trend, fit_variogram, idw, ordinary_kriging,
    universal_kriging, upscale_continuous, upscale_facies, Averaging, Drift, IdwParams,
    KrigingParams, ModelKind, RegularGrid,
};
use sx_core::ingest::run_ingest;
use sx_core::types::LogKind;
use sx_testutil::open_data::{canonical_top, kgs_dir, kgs_wells, write_kgs_project};

const FT: f64 = 0.3048;

#[test]
fn kgs_upscale_rhob_and_facies() {
    let Some(src) = kgs_dir() else {
        eprintln!("SX_OPEN_DATA not set; skipping");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let toml = write_kgs_project(tmp.path(), &src, 60, 15).unwrap();
    let cfg = load_project(&toml).unwrap();
    let out = run_ingest(&cfg.ingest, tmp.path()).unwrap();
    let xy: BTreeMap<i32, (f64, f64)> =
        out.wells.iter().map(|h| (h.id.0, (h.x_m, h.y_m))).collect();
    // Vertical-well approximation: z = MD in metres (KGS wells here are vertical).
    let (mut pts, mut rhob, mut fac) = (vec![], vec![], vec![]);
    for l in &out.logs {
        let (x, y) = xy[&l.id.0];
        let (r, g) = (&l.curves[&LogKind::Rhob], &l.curves[&LogKind::Gr]);
        for i in 0..l.grid.len {
            pts.push([x, y, l.grid.md_ft(i) * FT]);
            rhob.push(f64::from(r[i]));
            // Two-class facies from GR for the test only: 0 clean (< 75 API), 1 shaly.
            fac.push(if g[i].is_finite() {
                i32::from(g[i] >= 75.0)
            } else {
                -1
            });
        }
    }
    let lo = [-6000.0, -6000.0, 1200.0];
    let hi = [6000.0, 6000.0, 1600.0];
    let grid = RegularGrid::covering(lo, hi, [400.0, 400.0, 2.0]).unwrap();
    let a = upscale_continuous(&grid, &pts, &rhob, Averaging::Arithmetic, 3).unwrap();
    let h = upscale_continuous(&grid, &pts, &rhob, Averaging::Harmonic, 3).unwrap();
    let g = upscale_continuous(&grid, &pts, &rhob, Averaging::Geometric, 3).unwrap();
    let f = upscale_facies(&grid, &pts, &fac, 3).unwrap();
    eprintln!(
        "grid {:?} cells={}; RHOB cells={} facies cells={}",
        grid.dims,
        grid.n_cells(),
        a.len(),
        f.len()
    );
    assert!(a.len() > 1000, "{} upscaled cells", a.len());
    assert_eq!(a.len(), h.len());
    for ((ca, va, _), ((_, vh, _), (_, vg, _))) in a.iter().zip(h.iter().zip(&g)) {
        assert!(
            *vh <= *vg + 1e-12 && *vg <= *va + 1e-12,
            "cell {ca}: H {vh} G {vg} A {va}"
        );
        assert!((1.0..3.3).contains(va), "RHOB {va} (washouts allowed)");
    }
    let shaly = f.iter().filter(|c| c.1 == 1).count() as f64 / f.len() as f64;
    eprintln!("shaly cell fraction {shaly:.2}");
    assert!((0.05..0.95).contains(&shaly));
}

#[test]
fn kgs_mississippian_top_idw_vs_kriging() {
    let Some(src) = kgs_dir() else {
        eprintln!("SX_OPEN_DATA not set; skipping");
        return;
    };
    // Subsea Mississippian top (ft): elevation − top depth, one per well.
    let wells = kgs_wells(&src, 1000).unwrap();
    let by_kid: BTreeMap<_, _> = wells.iter().map(|w| (w.kid.clone(), w)).collect();
    let text = String::from_utf8_lossy(&std::fs::read(src.join("tops.csv")).unwrap()).into_owned();
    let mut lines = text.lines();
    let head: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim_matches('"').to_string())
        .collect();
    let col = |n: &str| head.iter().position(|h| h == n).unwrap();
    let (ck, cf, ct) = (col("KID"), col("FORMATION"), col("TOP"));
    let mut sub: BTreeMap<String, f64> = BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split(',').map(|s| s.trim_matches('"')).collect();
        if f.len() <= ct || canonical_top(f[cf]) != "Mississippian" {
            continue;
        }
        if let (Some(w), Ok(t)) = (by_kid.get(f[ck]), f[ct].parse::<f64>()) {
            sub.entry(f[ck].to_string()).or_insert(w.elev_ft - t);
        }
    }
    let n = wells.len().max(1) as f64;
    let lat0 = wells.iter().map(|w| w.lat_lon.0).sum::<f64>() / n;
    let lon0 = wells.iter().map(|w| w.lat_lon.1).sum::<f64>() / n;
    let (mut p, mut v) = (vec![], vec![]);
    for (kid, z) in &sub {
        let w = by_kid[kid];
        p.push([
            (w.lat_lon.1 - lon0) * 111_320.0 * lat0.to_radians().cos(),
            (w.lat_lon.0 - lat0) * 111_320.0,
            0.0,
        ]);
        v.push(*z);
    }
    let nw = v.len();
    assert!(nw >= 50, "{nw} wells with a Mississippian top");
    let mean = v.iter().sum::<f64>() / nw as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nw as f64).sqrt();

    let bins = experimental_variogram(&p, &v, 500.0, 16).unwrap();
    let model = fit_variogram(&bins, ModelKind::Spherical).unwrap();
    eprintln!("{nw} wells, subsea top mean {mean:.0} ft sd {sd:.1} ft; fitted {model:?}");
    for b in &bins {
        eprintln!(
            "  lag {:7.0} m  gamma {:8.1}  pairs {}",
            b.lag, b.gamma, b.pairs
        );
    }

    // Residual variograms (after a global linear / quadratic trend) for UK
    // and detrend-then-OK.
    let resid_model = |d: Drift| {
        let tr = fit_trend(&p, &v, d).unwrap();
        let r: Vec<f64> = p.iter().zip(&v).map(|(q, z)| z - tr.eval(q)).collect();
        let b = experimental_variogram(&p, &r, 500.0, 16).unwrap();
        fit_variogram(&b, ModelKind::Spherical).unwrap()
    };
    let (m_lin, m_quad) = (resid_model(Drift::Linear), resid_model(Drift::Quadratic));
    eprintln!("residual models: linear {m_lin:?}; quadratic {m_quad:?}");

    // Leave-one-out cross-validation: name, squared error sum, standardized sum.
    // Slots 6-9: the uncertainty fixes. 6 = UK linear with the raw-data
    // variogram; 7-9 = cross-validation sill correction, fitted on the
    // training wells only (nested inside the leave-one-out loop).
    let names = [
        "IDW",
        "OK",
        "UK linear",
        "UK quadratic",
        "detrend linear + OK",
        "detrend quadratic + OK",
        "UK linear, raw variogram",
        "UK linear, CV sill",
        "UK quadratic, CV sill",
        "detrend lin + OK, CV sill",
    ];
    const NS: usize = 10;
    let (mut se, mut zs) = ([0.0f64; NS], [0.0f64; NS]);
    let mut z2: Vec<Vec<f64>> = vec![Vec::new(); NS];
    let kp = |m| KrigingParams {
        model: m,
        max_neighbors: 16,
        max_distance: f64::INFINITY,
    };
    for i in 0..nw {
        let (mut pp, mut vv) = (p.clone(), v.clone());
        let (t, truth) = (pp.remove(i), vv.remove(i));
        let mut est = [f64::NAN; NS];
        let mut var = [f64::NAN; NS];
        est[0] = idw(&pp, &vv, &[t], &IdwParams::default()).unwrap()[0];
        let k = ordinary_kriging(&pp, &vv, &[t], &kp(model)).unwrap();
        (est[1], var[1]) = (k.estimate[0], k.variance[0]);
        for (slot, d, m) in [(2, Drift::Linear, m_lin), (3, Drift::Quadratic, m_quad)] {
            let k = universal_kriging(&pp, &vv, &[t], &kp(m), d).unwrap();
            (est[slot], var[slot]) = (k.estimate[0], k.variance[0]);
        }
        for (slot, d, m) in [(4, Drift::Linear, m_lin), (5, Drift::Quadratic, m_quad)] {
            let tr = fit_trend(&pp, &vv, d).unwrap();
            let r: Vec<f64> = pp.iter().zip(&vv).map(|(q, z)| z - tr.eval(q)).collect();
            let k = ordinary_kriging(&pp, &r, &[t], &kp(m)).unwrap();
            (est[slot], var[slot]) = (k.estimate[0] + tr.eval(&t), k.variance[0]);
        }
        let k = universal_kriging(&pp, &vv, &[t], &kp(model), Drift::Linear).unwrap();
        (est[6], var[6]) = (k.estimate[0], k.variance[0]);
        for (slot, d, m) in [(7, Drift::Linear, m_lin), (8, Drift::Quadratic, m_quad)] {
            let (mc, _) = calibrate_sill(&pp, &vv, &kp(m), d).unwrap();
            let k = universal_kriging(&pp, &vv, &[t], &kp(mc), d).unwrap();
            (est[slot], var[slot]) = (k.estimate[0], k.variance[0]);
        }
        {
            let tr = fit_trend(&pp, &vv, Drift::Linear).unwrap();
            let r: Vec<f64> = pp.iter().zip(&vv).map(|(q, z)| z - tr.eval(q)).collect();
            let (mc, _) = calibrate_sill(&pp, &r, &kp(m_lin), Drift::None).unwrap();
            let k = ordinary_kriging(&pp, &r, &[t], &kp(mc)).unwrap();
            (est[9], var[9]) = (k.estimate[0] + tr.eval(&t), k.variance[0]);
        }
        for s in 0..NS {
            se[s] += (est[s] - truth).powi(2);
            zs[s] += (est[s] - truth).powi(2) / var[s].max(1e-9);
            z2[s].push((est[s] - truth).powi(2) / var[s].max(1e-9));
        }
    }
    let rmse: Vec<f64> = se.iter().map(|e| (e / nw as f64).sqrt()).collect();
    for s in 0..NS {
        if s == 0 {
            eprintln!("LOO {:30} RMSE {:5.1} ft  (no variance)", names[s], rmse[s]);
            continue;
        }
        let mut z = z2[s].clone();
        z.sort_by(f64::total_cmp);
        let cover = z.iter().filter(|q| **q <= 1.6449f64.powi(2)).count() as f64 / z.len() as f64;
        eprintln!(
            "LOO {:30} RMSE {:5.1} ft  MSSE {:.2}  median z² {:.2} (ideal 0.45)  90% interval coverage {:.2}",
            names[s], rmse[s], zs[s] / nw as f64, z[z.len() / 2], cover
        );
    }
    assert!(
        rmse.iter().all(|r| *r < sd),
        "interpolation should beat the mean"
    );
    assert!(rmse[1] <= rmse[0] * 1.2, "OK much worse than IDW");
    assert!(
        rmse[2].min(rmse[4]) <= rmse[1] * 1.05,
        "a linear trend should not hurt"
    );
    for s in 7..NS {
        let m = zs[s] / nw as f64;
        assert!((0.5..2.0).contains(&m), "{}: CV-sill MSSE {m}", names[s]);
    }
    let msse = zs[1] / nw as f64;
    assert!(
        (0.2..5.0).contains(&msse),
        "kriging variance badly calibrated: {msse}"
    );
}
