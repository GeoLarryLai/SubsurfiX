// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Validation on synthetic wells: pipeline steps on synthetic wells with known truth,
//! and Rust LightGBM against Python LightGBM.
//!
//! Report only. Every test is `#[ignore]`d, prints its numbers and asserts only
//! that inputs could be built and products read (check 4 also asserts its
//! pass rule). Run once with
//! `cargo test -p sx-testutil --release --test synthetic_validation -- --ignored --nocapture`.
//! Projects, exchange files and the Python script go under `sx_p4_synth/` in the system temp directory.
//! The Python interpreter is `$SX_P4_PYTHON`, else the `work` env.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use serde_json::{json, Value};
use sx_core::config::{load_project, FillLightgbmConfig};
use sx_core::fill::lgbm::{lgbm_seed, LgbmParams, LgbmRegressor};
use sx_core::ingest::write_edges;
use sx_core::io::csv::{batch_to_csv_string, read_csv};
use sx_core::io::parquet::read_batch;
use sx_core::pipeline::{run, RunOptions};
use sx_core::types::{FeatureMatrix, LogKind, Regressor, WellId};
use sx_core::util::rng::{rng_for, standard_normal, uniform_range};
use sx_testutil::synth::{generate, GapConfig, SynthConfig, SynthProject};
use sx_testutil::synth_project::{write_synthetic_project, SynthProjectSpec};

fn root() -> PathBuf {
    std::env::temp_dir().join("sx_p4_synth")
}
const N_WELLS: usize = 25;
const SEED: u64 = 0;
/// Hidden top: this well loses this top's user pick.
const DROP_WELL: i32 = 18;
const DROP_TOP: u16 = 2;
/// Planted bad pick: this well gets one middle top shifted deeper.
const SHIFT_WELL: i32 = 6;
const SHIFT_FT: f64 = 40.0;
const DEFAULT_PYTHON: &str = "python3";
const LGBM_PASS_ABS: f64 = 1e-6;

/// The pipeline runs on the global rayon pool; tests take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------ project setup

/// Same generator call as `write_synthetic_project`.
fn truth_project() -> SynthProject {
    generate(&SynthConfig {
        seed: SEED,
        n_wells: N_WELLS,
        gaps: Some(GapConfig::default()),
        ..SynthConfig::default()
    })
    .unwrap()
}

fn fresh_project(name: &str) -> PathBuf {
    let dir = root().join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    write_synthetic_project(
        &dir,
        &SynthProjectSpec {
            n_wells: N_WELLS,
            seed: SEED,
        },
    )
    .unwrap();
    dir
}

fn run_pipeline(dir: &Path) {
    let t0 = Instant::now();
    let cfg = load_project(&dir.join("project.toml")).unwrap();
    run(&cfg, &RunOptions::new(dir)).unwrap();
    eprintln!(
        "pipeline {}: {:.1} s",
        dir.display(),
        t0.elapsed().as_secs_f64()
    );
}

fn truth_md(truth: &SynthProject, well: i32, top: u16) -> f64 {
    truth
        .tops_truth
        .iter()
        .find(|p| p.well.0 == well && p.top == top)
        .map(|p| p.md_ft)
        .unwrap()
}

/// Middle top (2 or 3) of `SHIFT_WELL` with the thicker interval below it.
fn shift_top(truth: &SynthProject) -> (u16, f64) {
    let below = |j: u16| truth_md(truth, SHIFT_WELL, j + 1) - truth_md(truth, SHIFT_WELL, j);
    let (a, b) = (below(2), below(3));
    if a >= b {
        (2, a)
    } else {
        (3, b)
    }
}

/// Rewrite `edges/` with the hidden top removed and the planted pick shifted.
fn write_edited_tops(dir: &Path, truth: &SynthProject, shift: u16) {
    let headers: Vec<_> = truth.wells.iter().map(|w| w.header.clone()).collect();
    let tops: Vec<(WellId, String, f64, u8)> = truth
        .tops
        .iter()
        .filter(|t| !(t.well.0 == DROP_WELL && t.top == DROP_TOP))
        .map(|t| {
            let md = if t.well.0 == SHIFT_WELL && t.top == shift {
                t.md_ft + SHIFT_FT
            } else {
                t.md_ft
            };
            let name = truth.top_set.names[usize::from(t.top)].clone();
            (t.well, name, md, 0u8)
        })
        .collect();
    assert_eq!(tops.len() + 1, truth.tops.len());
    write_edges(dir, &headers, &BTreeMap::new(), &tops).unwrap();
}

// ------------------------------------------------------------ table reading

struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn csv(path: &Path) -> Self {
        let (header, rows) = read_csv(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Self { header, rows }
    }

    /// Parquet through the engine's CSV text (no arrow dependency here).
    fn parquet(path: &Path) -> Self {
        let batch = read_batch(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let text = batch_to_csv_string(&batch).unwrap();
        let tmp = path.with_extension("p4.csv");
        std::fs::write(&tmp, text).unwrap();
        let t = Self::csv(&tmp);
        std::fs::remove_file(&tmp).unwrap();
        t
    }

    fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name} in {:?}", self.header))
    }

    fn cell<'a>(&self, row: &'a [String], name: &str) -> &'a str {
        row[self.col(name)].as_str()
    }

    fn num(&self, row: &[String], name: &str) -> f64 {
        num(self.cell(row, name))
    }

    fn find(&self, well: i32, key: &str, value: i64) -> Option<&Vec<String>> {
        let (cw, ck) = (self.col("well_id"), self.col(key));
        self.rows.iter().find(|r| {
            r[cw].parse::<i32>().ok() == Some(well) && r[ck].parse::<i64>().ok() == Some(value)
        })
    }
}

fn num(s: &str) -> f64 {
    if s.is_empty() {
        f64::NAN
    } else {
        s.parse().unwrap_or(f64::NAN)
    }
}

/// `well -> k -> value` of a per-sample product.
fn per_sample(t: &Table, value: &str) -> BTreeMap<i32, BTreeMap<i32, f64>> {
    let (cw, ck, cv) = (t.col("well_id"), t.col("k"), t.col(value));
    let mut out: BTreeMap<i32, BTreeMap<i32, f64>> = BTreeMap::new();
    for r in &t.rows {
        out.entry(r[cw].parse().unwrap())
            .or_default()
            .insert(r[ck].parse().unwrap(), num(&r[cv]));
    }
    out
}

// ------------------------------------------------------------ statistics

/// Nearest-rank percentile; NaN for an empty slice.
fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let r = ((p / 100.0) * s.len() as f64).ceil() as usize;
    s[r.clamp(1, s.len()) - 1]
}

#[derive(Default)]
struct Errs {
    units: Vec<f64>,
    ft: Vec<f64>,
    missing: usize,
}

impl Errs {
    fn extend(&mut self, other: &Errs) {
        self.units.extend_from_slice(&other.units);
        self.ft.extend_from_slice(&other.ft);
        self.missing += other.missing;
    }

    fn cells(&self) -> String {
        format!(
            "{:>8.4} {:>8.4} {:>8.4} | {:>7.2} {:>7.2} {:>7.2}",
            pct(&self.units, 50.0),
            pct(&self.units, 95.0),
            pct(&self.units, 100.0),
            pct(&self.ft, 50.0),
            pct(&self.ft, 95.0),
            pct(&self.ft, 100.0),
        )
    }
}

/// `|got − truth|` on samples between the first and last true top
/// (`1 ≤ truth < n_tops`). Pads are skipped: the engine pads are
/// `align.pad_ft` (200 ft) from the outer tops, the generator's are 100 ft, so
/// pad coordinates follow different conventions. Feet = units × true local
/// interval thickness.
fn sample_errors(truth: &SynthProject, well: usize, got: Option<&BTreeMap<i32, f64>>) -> Errs {
    let w = &truth.wells[well];
    let n_tops = truth.top_set.len() as f64;
    let mut e = Errs::default();
    for (i, &t) in w.layer_time.iter().enumerate() {
        if !(1.0..n_tops).contains(&t) {
            continue;
        }
        let k = w.geometry.grid.k_start + i as i32;
        let iv = t.floor() as usize;
        let thick = w.boundaries_elev_ft[iv] - w.boundaries_elev_ft[iv + 1];
        match got.and_then(|m| m.get(&k)) {
            Some(&v) if v.is_finite() => {
                let d = (v - t).abs();
                e.units.push(d);
                e.ft.push(d * thick);
            }
            _ => e.missing += 1,
        }
    }
    e
}

/// Layer time vs rough coordinate against truth. Per-well rows for every
/// well when `focus` is empty, else only for the wells in `focus`.
fn layer_time_report(label: &str, dir: &Path, truth: &SynthProject, focus: &[i32]) {
    let tau = per_sample(
        &Table::parquet(&dir.join("stages/correlate/layer_time.parquet")),
        "layer_time",
    );
    let u = per_sample(
        &Table::parquet(&dir.join("stages/rough_align/layer_coord.parquet")),
        "layer_coord",
    );
    eprintln!("\n[{label}] |value - truth| on samples between the first and last true top");
    eprintln!(
        "  units = layer-time units, ft = units x true local interval thickness; med / P95 / max"
    );
    eprintln!(
        "  {:>4} {:>5} | {:^26} | {:^23} || {:^26} | {:^23} | miss tau/u",
        "well", "n", "tau (correlate) units", "tau ft", "u (rough) units", "u ft"
    );
    let (mut all_tau, mut all_u) = (Errs::default(), Errs::default());
    let mut better = 0usize;
    let mut worse = 0usize;
    for (wi, w) in truth.wells.iter().enumerate() {
        let id = w.header.id.0;
        let et = sample_errors(truth, wi, tau.get(&id));
        let eu = sample_errors(truth, wi, u.get(&id));
        let (pt, pu) = (pct(&et.ft, 95.0), pct(&eu.ft, 95.0));
        if pt + 1e-9 < pu {
            better += 1;
        } else if pt > pu + 1e-9 {
            worse += 1;
        }
        if focus.is_empty() || focus.contains(&id) {
            eprintln!(
                "  {:>4} {:>5} | {} || {} | {}/{}",
                id,
                et.units.len() + et.missing,
                et.cells(),
                eu.cells(),
                et.missing,
                eu.missing
            );
        }
        all_tau.extend(&et);
        all_u.extend(&eu);
    }
    eprintln!(
        "  {:>4} {:>5} | {} || {} | {}/{}",
        "ALL",
        all_tau.units.len() + all_tau.missing,
        all_tau.cells(),
        all_u.cells(),
        all_tau.missing,
        all_u.missing
    );
    eprintln!(
        "  wells where tau P95 ft < u P95 ft: {better}; tau worse: {worse}; equal: {}",
        truth.wells.len() - better - worse
    );
}

/// User rows of `tops.csv` whose `md_ft` differs from the input pick.
fn user_pick_drift(tops: &Table, picks: &BTreeMap<(i32, i64), f64>) -> usize {
    tops.rows
        .iter()
        .filter(|r| tops.cell(r, "source") == "user")
        .filter(|r| {
            let key = (
                tops.cell(r, "well_id").parse().unwrap(),
                tops.cell(r, "top_index").parse().unwrap(),
            );
            picks
                .get(&key)
                .is_none_or(|&md| (md - tops.num(r, "md_ft")).abs() > 1e-9)
        })
        .count()
}

fn spacing_to(truth: &SynthProject, a: i32, b: i32) -> f64 {
    let h = |id: i32| &truth.wells[id as usize].header;
    (h(a).x_m - h(b).x_m).hypot(h(a).y_m - h(b).y_m) / 1000.0
}

fn print_suspects(s: &Table, truth: &SynthProject, skip: Option<(i32, i64)>) -> usize {
    let mut n = 0usize;
    for r in &s.rows {
        let well: i32 = s.cell(r, "well_id").parse().unwrap();
        let top: i64 = s.cell(r, "top_index").parse().unwrap();
        if Some((well, top)) == skip {
            continue;
        }
        n += 1;
        let t = truth_md(truth, well, top as u16);
        eprintln!(
            "    rank {:>3} well {:>2} top {} pick-truth {:>6.2} offset {:>7.2} spread {:>6.2} votes {} suggested-truth {:>7.2} | dist to shift well {:.1}, to drop well {:.1} (x1000 m)",
            s.cell(r, "rank"),
            well,
            top,
            s.num(r, "md_ft") - t,
            s.num(r, "offset_ft"),
            s.num(r, "spread_ft"),
            s.cell(r, "votes"),
            s.num(r, "suggested_md_ft") - t,
            spacing_to(truth, well, SHIFT_WELL),
            spacing_to(truth, well, DROP_WELL),
        );
    }
    n
}

// ------------------------------------------------------------ checks 1-3

#[test]
#[ignore = "slow: 25-well pipeline run"]
fn p4_check1_layer_time_recovery() {
    let _g = serial();
    let truth = truth_project();
    let dir = fresh_project("check1");
    run_pipeline(&dir);
    layer_time_report("check 1, clean picks", &dir, &truth, &[]);

    let tops = Table::csv(&dir.join("stages/correlate/tops.csv"));
    let picks: BTreeMap<(i32, i64), f64> = truth
        .tops
        .iter()
        .map(|p| ((p.well.0, i64::from(p.top)), p.md_ft))
        .collect();
    let suspects = Table::csv(&dir.join("stages/correlate/suspects.csv"));
    eprintln!(
        "\n[check 1] tops.csv rows {}, user rows whose md_ft differs from the input pick: {}",
        tops.rows.len(),
        user_pick_drift(&tops, &picks)
    );
    eprintln!(
        "[check 1] suspects with every pick correct (all false): {}",
        suspects.rows.len()
    );
    print_suspects(&suspects, &truth, None);
}

#[test]
#[ignore = "slow: 25-well pipeline run"]
fn p4_check2_check3_hidden_top_and_bad_pick() {
    let _g = serial();
    let truth = truth_project();
    let (shift, thick_below) = shift_top(&truth);
    assert!(
        thick_below > SHIFT_FT + 5.0,
        "interval below the shifted top is {thick_below} ft"
    );
    let dir = fresh_project("check23");
    write_edited_tops(&dir, &truth, shift);
    eprintln!(
        "\n[check 2/3] hidden: well {DROP_WELL} top {DROP_TOP} ({}); planted: well {SHIFT_WELL} top {shift} ({}) +{SHIFT_FT} ft, interval below {thick_below:.1} ft",
        truth.top_set.names[usize::from(DROP_TOP)],
        truth.top_set.names[usize::from(shift)],
    );
    run_pipeline(&dir);

    // Check 2: hidden top.
    let tops = Table::csv(&dir.join("stages/correlate/tops.csv"));
    let est = Table::parquet(&dir.join("stages/rough_align/tops_estimated.parquet"));
    let t_drop = truth_md(&truth, DROP_WELL, DROP_TOP);
    let r = tops
        .find(DROP_WELL, "top_index", i64::from(DROP_TOP))
        .unwrap();
    let e = est
        .find(DROP_WELL, "boundary", i64::from(DROP_TOP) + 1)
        .unwrap();
    eprintln!("\n[check 2] hidden top, true MD {t_drop:.2} ft");
    eprintln!(
        "  rough estimate (tops_estimated): md {:.2} ft, error {:+.2} ft, source {}, status {}, donors {}",
        est.num(e, "md_ft"),
        est.num(e, "md_ft") - t_drop,
        est.cell(e, "source"),
        est.cell(e, "status"),
        est.cell(e, "n_donors"),
    );
    eprintln!(
        "  tops.csv: md_ft {:.2} ft, error {:+.2} ft, source {}, state {}, votes {}, z* {:.2} (error {:+.2}), spread {:.2}, clamped {}",
        tops.num(r, "md_ft"),
        tops.num(r, "md_ft") - t_drop,
        tops.cell(r, "source"),
        tops.cell(r, "check_state"),
        tops.cell(r, "votes"),
        tops.num(r, "suggested_md_ft"),
        tops.num(r, "suggested_md_ft") - t_drop,
        tops.num(r, "spread_ft"),
        tops.cell(r, "clamped"),
    );
    let n_suggested = tops
        .rows
        .iter()
        .filter(|r| tops.cell(r, "source") == "suggested")
        .count();
    eprintln!("  suggested rows in tops.csv: {n_suggested} (expected 1)");

    // Check 3: planted bad pick.
    let suspects = Table::csv(&dir.join("stages/correlate/suspects.csv"));
    let t_shift = truth_md(&truth, SHIFT_WELL, shift);
    eprintln!(
        "\n[check 3] planted pick: true MD {t_shift:.2} ft, pick {:.2} ft; suspects total {}",
        t_shift + SHIFT_FT,
        suspects.rows.len()
    );
    let tr = tops
        .find(SHIFT_WELL, "top_index", i64::from(shift))
        .unwrap();
    eprintln!(
        "  tops.csv row: md_ft {:.2}, suspect {}, rank {}, state {}, votes {}, z* {:.2} (z*-truth {:+.2}), offset {:+.2}, spread {:.2}",
        tops.num(tr, "md_ft"),
        tops.cell(tr, "suspect"),
        tops.cell(tr, "rank"),
        tops.cell(tr, "check_state"),
        tops.cell(tr, "votes"),
        tops.num(tr, "suggested_md_ft"),
        tops.num(tr, "suggested_md_ft") - t_shift,
        tops.num(tr, "offset_ft"),
        tops.num(tr, "spread_ft"),
    );
    let in_list = suspects.rows.iter().position(|r| {
        suspects.cell(r, "well_id") == SHIFT_WELL.to_string()
            && suspects.cell(r, "top_index") == shift.to_string()
    });
    eprintln!(
        "  in suspects.csv: {}",
        in_list.map_or("no".to_string(), |p| format!(
            "yes, row {} of {}",
            p + 1,
            suspects.rows.len()
        ))
    );
    eprintln!("  other suspects (untouched user picks):");
    let n_false = print_suspects(&suspects, &truth, Some((SHIFT_WELL, i64::from(shift))));
    let n_user = tops
        .rows
        .iter()
        .filter(|r| tops.cell(r, "source") == "user")
        .count();
    eprintln!(
        "  false suspects: {n_false} of {} untouched user picks",
        n_user - 1
    );

    let mut picks: BTreeMap<(i32, i64), f64> = truth
        .tops
        .iter()
        .map(|p| ((p.well.0, i64::from(p.top)), p.md_ft))
        .collect();
    picks.remove(&(DROP_WELL, i64::from(DROP_TOP)));
    if let Some(md) = picks.get_mut(&(SHIFT_WELL, i64::from(shift))) {
        *md += SHIFT_FT;
    }
    eprintln!(
        "  user rows in tops.csv whose md_ft differs from the input pick: {}",
        user_pick_drift(&tops, &picks)
    );

    layer_time_report(
        "check 2/3 run, edited picks",
        &dir,
        &truth,
        &[DROP_WELL, SHIFT_WELL],
    );
}

// ------------------------------------------------------------ check 4

const FEATURES: [&str; 6] = ["f0", "elev_ft", "f2", "layer_coord", "f4", "f5"];

/// `n` rows of the six features and the target `2 f0 + 0.002 (elev + 2000)
/// + sin f2 + 0.3 layer_coord + f4 f5 + 0.1 N(0, 1)`, stored as `f32`.
fn lgbm_rows(stream: i32, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rng_for(4, "p4_lgbm", stream);
    let mut x = Vec::with_capacity(n * FEATURES.len());
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let f = [
            uniform_range(&mut rng, 0.0, 1.0),
            uniform_range(&mut rng, -3000.0, -1000.0),
            standard_normal(&mut rng),
            uniform_range(&mut rng, 0.0, 7.0),
            uniform_range(&mut rng, 0.0, 1.0),
            uniform_range(&mut rng, -1.0, 1.0),
        ];
        let target = 2.0 * f[0]
            + 0.002 * (f[1] + 2000.0)
            + f[2].sin()
            + 0.3 * f[3]
            + f[4] * f[5]
            + 0.1 * standard_normal(&mut rng);
        x.extend(f.iter().map(|&v| v as f32));
        y.push(target as f32);
    }
    (x, y)
}

fn write_rows(path: &Path, x: &[f32], y: &[f32]) {
    let mut text = FEATURES.join(",");
    text.push_str(",y\n");
    for (row, &t) in x.chunks(FEATURES.len()).zip(y) {
        for v in row {
            text.push_str(&format!("{},", f64::from(*v)));
        }
        text.push_str(&format!("{}\n", f64::from(t)));
    }
    std::fs::write(path, text).unwrap();
}

const PY_SCRIPT: &str = r#"import json, sys
import numpy as np
import lightgbm as lgb

d = sys.argv[1]
with open(f"{d}/params.json") as f:
    meta = json.load(f)
names = meta["feature_names"]
nf = len(names)
tr = np.loadtxt(f"{d}/train.csv", delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
te = np.loadtxt(f"{d}/test.csv", delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
X = np.ascontiguousarray(tr[:, :nf].astype(np.float32))
y = tr[:, nf].astype(np.float32)
Xt = np.ascontiguousarray(te[:, :nf].astype(np.float32))
params = dict(meta["booster"])
params.update(meta["dataset"])
ds = lgb.Dataset(X, label=y, feature_name=names, params=dict(params), free_raw_data=False)
bst = lgb.train(params, ds, num_boost_round=meta["num_iterations"])
py_pred = bst.predict(Xt, num_threads=1)
with open(f"{d}/rust_model.txt") as f:
    rust_text = f.read()
py_text = bst.model_to_string()
with open(f"{d}/py_model.txt", "w") as f:
    f.write(py_text)
on_rust = lgb.Booster(model_str=rust_text).predict(Xt, num_threads=1)

def section(t, start, end):
    i = t.find(start)
    j = t.find(end, i)
    return t[i:j] if i >= 0 and j >= 0 else None

trees_py = section(py_text, "Tree=0", "end of trees")
trees_rust = section(rust_text, "Tree=0", "end of trees")
par_py = set((section(py_text, "parameters:", "end of parameters") or "").splitlines())
par_rust = set((section(rust_text, "parameters:", "end of parameters") or "").splitlines())
out = {
    "lightgbm_version": lgb.__version__,
    "py_pred": py_pred.tolist(),
    "py_on_rust_model": on_rust.tolist(),
    "trees_identical": trees_py is not None and trees_py == trees_rust,
    "n_trees_py": bst.num_trees(),
    "params_only_py": sorted(par_py - par_rust),
    "params_only_rust": sorted(par_rust - par_py),
}
with open(f"{d}/py_out.json", "w") as f:
    json.dump(out, f)
"#;

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn f64_list(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Train in Rust and in Python with `params`; returns max |Rust − Python|.
fn lgbm_parity(tag: &str, params: &LgbmParams) -> f64 {
    let dir = root().join("lgbm").join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    let script = root().join("lgbm_parity.py");
    std::fs::write(&script, PY_SCRIPT).unwrap();

    let names: Vec<String> = FEATURES.iter().map(|s| s.to_string()).collect();
    let (x_train, y_train) = lgbm_rows(0, 500);
    let (x_test, y_test) = lgbm_rows(1, 100);
    write_rows(&dir.join("train.csv"), &x_train, &y_train);
    write_rows(&dir.join("test.csv"), &x_test, &y_test);

    let seed = lgbm_seed(0, LogKind::Gr);
    let mut model = LgbmRegressor::new(params.clone(), seed);
    let xm = FeatureMatrix::new(500, names.clone(), x_train).unwrap();
    model.fit(&xm, &y_train, None).unwrap();
    let xt = FeatureMatrix::new(100, names.clone(), x_test).unwrap();
    let rust_pred: Vec<f64> = model
        .predict(&xt)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    std::fs::write(dir.join("rust_model.txt"), model.to_model_text().unwrap()).unwrap();
    std::fs::write(
        dir.join("rust_pred.json"),
        serde_json::to_string(&rust_pred).unwrap(),
    )
    .unwrap();

    let bins: Vec<usize> = names
        .iter()
        .map(|n| {
            if params.wide_bin_features.contains(n) {
                params.max_bin_wide
            } else {
                params.max_bin
            }
        })
        .collect();
    let meta = json!({
        "feature_names": names,
        "num_iterations": params.num_iterations,
        "booster": params.booster_json(seed),
        "dataset": {
            "max_bin": params.max_bin,
            "max_bin_by_feature": bins,
            "bin_construct_sample_cnt": params.bin_construct_sample_cnt,
            "use_missing": true,
            "zero_as_missing": false,
            "data_random_seed": seed,
        },
        "rust_param_string": params.dataset_params(&names, seed),
    });
    std::fs::write(
        dir.join("params.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    let python = std::env::var("SX_P4_PYTHON").unwrap_or_else(|_| DEFAULT_PYTHON.to_string());
    let out = Command::new(&python)
        .arg(&script)
        .arg(&dir)
        .output()
        .unwrap_or_else(|e| panic!("cannot start {python}: {e}"));
    assert!(
        out.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("py_out.json")).unwrap()).unwrap();
    let py_pred = f64_list(&py["py_pred"]);
    let py_on_rust = f64_list(&py["py_on_rust_model"]);
    let d_main = max_abs(&rust_pred, &py_pred);
    let d_train = max_abs(&py_on_rust, &py_pred);
    let d_cast = max_abs(&rust_pred, &py_on_rust);
    let spread = pct(&py_pred, 100.0) - pct(&py_pred, 0.0);
    eprintln!(
        "\n[check 4, {tag}] python lightgbm {}, trees py {}, min_data_in_leaf {}, num_leaves {}",
        py["lightgbm_version"], py["n_trees_py"], params.min_data_in_leaf, params.num_leaves
    );
    eprintln!(
        "  max |rust(f32) - python|               = {d_main:.3e}  (pass < {LGBM_PASS_ABS:e}: {})",
        d_main < LGBM_PASS_ABS
    );
    eprintln!("  max |python on rust model - python|   = {d_train:.3e}  (training parity, f64)");
    eprintln!("  max |rust(f32) - python on rust model| = {d_cast:.3e}  (f32 output cast only)");
    eprintln!(
        "  tree sections identical: {}; prediction range {spread:.3}",
        py["trees_identical"]
    );
    eprintln!(
        "  parameter lines only in python model: {}",
        py["params_only_py"]
    );
    eprintln!(
        "  parameter lines only in rust model:   {}",
        py["params_only_rust"]
    );
    d_main
}

#[test]
#[ignore = "slow: needs Python lightgbm"]
fn p4_check4_lgbm_parity() {
    let _g = serial();
    let mut params = LgbmParams::from_config(&FillLightgbmConfig::default());
    params.num_threads = 1;
    params.num_iterations = 50;
    let d = lgbm_parity("default", &params);
    // Extra, not the pass rule: with 500 rows the default min_data_in_leaf = 200
    // allows at most two leaves per tree, so repeat with deeper trees.
    let mut deep = params.clone();
    deep.min_data_in_leaf = 20;
    lgbm_parity("leaf20", &deep);
    assert!(d < LGBM_PASS_ABS, "max |rust - python| = {d:e}");
}
