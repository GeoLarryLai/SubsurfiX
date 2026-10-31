// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Log-fill tuning harness on open Kansas Geological Survey (KGS) data.
//! Ignored by default; run with `SX_OPEN_DATA` set and `--ignored`.
//! `SX_TUNE_WELLS` (default 25) and `SX_TUNE` (variant name) pick the setup.
//! "The source of this material is the Kansas Geological Survey website at
//! http://www.kgs.ku.edu/. All Rights Reserved."

use sx_core::config::load_project;
use sx_core::fill::holdout::score_holdout;
use sx_core::pipeline::{run, RunOptions};
use sx_testutil::open_data::{kgs_dir, write_kgs_project};

#[test]
#[ignore = "tuning harness; needs SX_OPEN_DATA"]
fn kgs_fill_tune() {
    let Some(src) = kgs_dir() else { return };
    let n: usize = std::env::var("SX_TUNE_WELLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let variant = std::env::var("SX_TUNE").unwrap_or_default();
    let keep = std::env::var("SX_TUNE_DIR")
        .ok()
        .map(std::path::PathBuf::from);
    let tmp = tempfile::tempdir().unwrap();
    let dir = keep.unwrap_or_else(|| tmp.path().to_path_buf());
    std::fs::create_dir_all(&dir).unwrap();
    let toml = write_kgs_project(&dir, &src, n, 10).unwrap();
    let mut cfg = load_project(&toml).unwrap();
    for kv in variant.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        let l = &mut cfg.fill.lightgbm;
        match k {
            "iters" => l.num_iterations = v.parse().unwrap(),
            "lr" => l.learning_rate = v.parse().unwrap(),
            "leaves" => l.num_leaves = v.parse().unwrap(),
            "depth" => l.max_depth = v.parse().unwrap(),
            "minleaf" => l.min_data_in_leaf = v.parse().unwrap(),
            "ff" => l.feature_fraction = v.parse().unwrap(),
            "l2" => l.lambda_l2 = v.parse().unwrap(),
            "bag" => {
                l.bagging_fraction = v.parse().unwrap();
                l.bagging_freq = 1;
            }
            "obj" => l.objective = v.to_string(),
            "alpha" => l.alpha = v.parse().unwrap(),
            "dropout" => cfg.fill.dropout_frac = v.parse().unwrap(),
            "hold" => cfg.fill.holdout_n_wells = v.parse().unwrap(),
            "qc" if v == "flat" => {
                cfg.qc.low_density_rhob = 0.0;
                cfg.qc.plausible_range.insert("RHOB".into(), [1.65, 3.1]);
            }
            "qc" if v == "legacy" => {
                cfg.qc.badhole_logs.clear();
                cfg.qc.plausible_range.clear();
                cfg.qc.plausible_median.clear();
            }
            "closed" => cfg.qc.closed_caliper.mode = v.to_string(),
            _ => panic!("unknown knob {k}"),
        }
    }
    run(&cfg, &RunOptions::new(&dir)).unwrap();
    let r = score_holdout(&dir, &cfg).unwrap();
    let mut line = format!("TUNE wells={n} [{variant}]");
    for s in &r.summary {
        line += &format!(
            " | {} r2f={:.2} r2m={:.2} unf={:.2} n={}",
            s.log.name(),
            s.final_scope.r2,
            s.model_scope.r2,
            s.unfilled_share,
            s.n_truth
        );
    }
    eprintln!("{line}");
    if std::env::var("SX_TUNE_WELLS_DETAIL").is_ok() {
        for w in &r.scores {
            eprintln!(
                "TUNE  well {:16} {:5} n={:5} r2={:6.2} bias={:7.3} rmse={:6.3} slope={:5.2}",
                w.well_name,
                w.log.name(),
                w.n_truth,
                w.final_scope.r2,
                w.final_scope.bias,
                w.final_scope.rmse,
                w.final_scope.slope
            );
        }
    }
}

/// Per-well median of each log after ingest (data-quality check).
#[test]
#[ignore = "diagnostic; needs SX_OPEN_DATA"]
fn kgs_log_stats() {
    let Some(src) = kgs_dir() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let toml = write_kgs_project(tmp.path(), &src, 200, 10).unwrap();
    let cfg = load_project(&toml).unwrap();
    let out = sx_core::ingest::run_ingest(&cfg.ingest, tmp.path()).unwrap();
    let names: std::collections::BTreeMap<i32, String> =
        out.wells.iter().map(|w| (w.id.0, w.name.clone())).collect();
    for l in &out.logs {
        let mut line = format!("STAT {:>4} {:16}", l.id.0, names[&l.id.0]);
        for (k, v) in &l.curves {
            let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
            f.sort_by(f32::total_cmp);
            if f.is_empty() {
                line += &format!(" {}=-", k.name());
            } else {
                line += &format!(" {}={:.3}[{}]", k.name(), f[f.len() / 2], f.len());
            }
        }
        eprintln!("{line}");
    }
}
