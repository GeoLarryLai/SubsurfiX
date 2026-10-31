// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Two-well dynamic time warping (DTW) correlation on public FORCE 2020 data.
//!
//! Data: FORCE 2020 machine learning competition, `hidden_test.csv`
//! (Bormann P., Aursand P., Dilib F., Dischington P., Manral S. 2020. 2020
//! FORCE Machine Learning Contest,
//! https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition).
//! Well logs: Norwegian Licence for Open Government Data (NLOD) 2.0; labels:
//! CC-BY-4.0. "Lithofacies data was provided by the FORCE Machine Learning
//! competition with well logs and seismic 2020". The data is not stored in
//! this repository; fetch it with `scripts/fetch_force2020.sh` and set
//! `SX_FORCE2020_CSV`. Without it the test prints a note and passes.
//!
//! Check: the DTW path, built only from gamma ray (GR), should put the
//! Norwegian Offshore Directorate group tops of well A closer to their depths
//! in well B than a linear stretch between the shared top and base does.

use std::collections::BTreeMap;
use sx_core::dtw::{band_half_width, banded_dtw_dense};

struct Well {
    md: Vec<f64>,
    gr: Vec<f64>,
    group: Vec<String>,
}

fn load(path: &str) -> BTreeMap<String, Well> {
    let text = std::fs::read_to_string(path).expect("read csv");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split(';').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).unwrap();
    let (cw, cmd, cgr, cg) = (col("WELL"), col("DEPTH_MD"), col("GR"), col("GROUP"));
    let mut out: BTreeMap<String, Well> = BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split(';').collect();
        let w = out.entry(f[cw].to_string()).or_insert(Well {
            md: vec![],
            gr: vec![],
            group: vec![],
        });
        w.md.push(f[cmd].parse().unwrap_or(f64::NAN));
        w.gr.push(f[cgr].parse().unwrap_or(f64::NAN));
        w.group.push(f[cg].to_string());
    }
    out
}

/// Resample to `step` m by nearest sample, fill gaps by carrying forward, and
/// scale robustly to the 5th–95th percentile (a fresh, simple normalization).
fn prep(w: &Well, step: f64) -> (Vec<f64>, Vec<f64>, Vec<String>) {
    let (lo, hi) = (w.md[0], *w.md.last().unwrap());
    let n = ((hi - lo) / step) as usize;
    let (mut md, mut gr, mut grp) = (vec![], vec![], vec![]);
    let mut k = 0;
    let mut last = f64::NAN;
    for i in 0..n {
        let d = lo + i as f64 * step;
        while k + 1 < w.md.len() && w.md[k + 1] <= d {
            k += 1;
        }
        if w.gr[k].is_finite() {
            last = w.gr[k];
        }
        md.push(d);
        gr.push(last);
        grp.push(w.group[k].clone());
    }
    let first = gr.iter().copied().find(|v| v.is_finite()).unwrap_or(0.0);
    for v in gr.iter_mut() {
        if !v.is_finite() {
            *v = first;
        }
    }
    let mut s = gr.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (p5, p95) = (s[s.len() * 5 / 100], s[s.len() * 95 / 100]);
    // Smooth with a 10-sample centred moving average to suppress bed-scale noise.
    let h = 5usize;
    let sm: Vec<f64> = (0..gr.len())
        .map(|i| {
            let (l, r) = (i.saturating_sub(h), (i + h + 1).min(gr.len()));
            gr[l..r].iter().sum::<f64>() / (r - l) as f64
        })
        .collect();
    let g = sm
        .iter()
        .map(|v| ((v - p5) / (p95 - p5).max(1e-9)).clamp(-0.5, 1.5))
        .collect();
    (md, g, grp)
}

fn tops(md: &[f64], grp: &[String]) -> BTreeMap<String, usize> {
    let mut t = BTreeMap::new();
    for i in 0..md.len() {
        if !grp[i].is_empty() && (i == 0 || grp[i] != grp[i - 1]) {
            t.entry(grp[i].clone()).or_insert(i);
        }
    }
    t
}

#[test]
fn force2020_two_well_dtw() {
    let Ok(path) = std::env::var("SX_FORCE2020_CSV") else {
        eprintln!("SX_FORCE2020_CSV not set; skipping FORCE 2020 DTW check");
        return;
    };
    let wells = load(&path);
    // Pairs fixed in advance; 31/2-10 vs 31/2-21 S is a near tie and is kept
    // in on purpose so the summary is not cherry-picked.
    let pairs = [
        ("35/9-7", "35/11-5"),
        ("31/2-10", "31/2-21 S"),
        ("16/2-7", "16/7-6"),
        ("15/9-23", "16/7-6"),
    ];
    let (mut td, mut tl) = (0.0, 0.0);
    for (x, y) in pairs {
        eprintln!("== {x} vs {y}");
        let (d, l) = pair(&wells[x], &wells[y]);
        td += d;
        tl += l;
    }
    eprintln!(
        "all pairs, mean of per-pair mean top error: DTW {:.1} m, linear {:.1} m",
        td / 4.0,
        tl / 4.0
    );
    assert!(td < tl, "DTW {td} not better than linear {tl}");
}

/// Mean absolute top error (m) of DTW and of linear stretch for one pair.
fn pair(a: &Well, b: &Well) -> (f64, f64) {
    let step = 2.0;
    let (mda, ga, gra) = prep(a, step);
    let (mdb, gb, grb) = prep(b, step);
    let (ta, tb) = (tops(&mda, &gra), tops(&mdb, &grb));
    let shared: Vec<&String> = ta.keys().filter(|k| tb.contains_key(*k)).collect();
    assert!(shared.len() >= 3, "shared groups {shared:?}");

    // Correlate only between the first and last shared tops (pinned ends).
    let (a0, a1) = (
        shared.iter().map(|k| ta[*k]).min().unwrap(),
        shared.iter().map(|k| ta[*k]).max().unwrap(),
    );
    let (b0, b1) = (
        shared.iter().map(|k| tb[*k]).min().unwrap(),
        shared.iter().map(|k| tb[*k]).max().unwrap(),
    );
    let (na, nb) = (a1 - a0 + 1, b1 - b0 + 1);
    let mut cost = vec![0.0; na * nb];
    for i in 0..na {
        for j in 0..nb {
            cost[i * nb + j] = (ga[a0 + i] - gb[b0 + j]).abs();
        }
    }
    let w = band_half_width(na.max(nb), 0.3)
        .unwrap()
        .max(na.abs_diff(nb) + 1);
    let p = banded_dtw_dense(&cost, na, nb, w).unwrap();
    for c in p.cells.windows(2) {
        assert!(c[1].0 >= c[0].0 && c[1].1 >= c[0].1);
    }

    let (mut e_dtw, mut e_lin, mut n) = (0.0, 0.0, 0.0);
    for k in &shared {
        let (ia, ib) = (ta[*k] - a0, tb[*k] - b0);
        let js: Vec<usize> = p.cells.iter().filter(|c| c.0 == ia).map(|c| c.1).collect();
        let jd = js.iter().sum::<usize>() as f64 / js.len() as f64;
        let jl = ia as f64 * (nb - 1) as f64 / (na - 1) as f64;
        e_dtw += (jd - ib as f64).abs() * step;
        e_lin += (jl - ib as f64).abs() * step;
        n += 1.0;
        eprintln!(
            "{k}: B top {:.0} m, DTW {:.0} m, linear {:.0} m",
            mdb[tb[*k]],
            mdb[b0] + jd * step,
            mdb[b0] + jl * step
        );
    }
    eprintln!(
        "mean top error: DTW {:.1} m, linear {:.1} m over {n} tops",
        e_dtw / n,
        e_lin / n
    );
    (e_dtw / n, e_lin / n)
}
