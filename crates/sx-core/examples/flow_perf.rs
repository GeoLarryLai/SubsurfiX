// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Flow-core benchmark and regression dump.
//!
//! `flow_perf pressure <nx> <ny> <nz>` / `flow_perf heat <nx> <ny> <nz>` /
//! `flow_perf twophase <nx> <ny> <nz>`: one timed run; prints wall time (s)
//! and peak RSS (MiB, VmHWM). `flow_perf dump <file>`: writes the outputs of
//! the fixed regression cases as little-endian f64 for bit-level comparison.

use std::io::Write;
use std::time::Instant;

#[path = "../tests/common/flow_cases.rs"]
mod flow_cases;
use flow_cases::*;
use sx_core::flow::{run_heat, run_two_phase, solve_pressure, Corey, Physics, ThermalProps, TwoPhaseOptions};

fn hwm_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn dump(path: &str) {
    let out = dump_bytes();
    std::fs::File::create(path).unwrap().write_all(&out).unwrap();
    eprintln!("wrote {} values", out.len() / 8);
}

fn main() {
    if let Some(t) = std::env::var("SX_FLOW_THREADS").ok().and_then(|v| v.parse().ok()) {
        sx_core::flow::set_num_threads(t);
    }
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "dump" {
        return dump(&a[2]);
    }
    let d = |i: usize| a[i].parse::<usize>().unwrap();
    let (nx, ny, nz) = (d(2), d(3), d(4));
    let m = model(nx, ny, nz);
    let n = m.n_cells();
    let ws = wells(nx, ny, nz);
    let fx = fixed(&m, nx);
    let base = hwm_mib();
    let t0 = Instant::now();
    let check = match a[1].as_str() {
        "pressure" => {
            let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &ws, &fx).unwrap();
            sol.p.iter().sum::<f64>()
        }
        "heat" => {
            let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &ws, &fx).unwrap();
            let mut h = heat_inputs(&m, n, 2, 1, false);
            h.times = (1..=10).map(|i| i as f64 * 1e6).collect();
            let r = run_heat(&m, &sol, &ws, &fx, &ThermalProps::default(), &h).unwrap();
            r.temperature.last().unwrap().iter().sum::<f64>()
        }
        "twophase" => {
            let opts = TwoPhaseOptions { max_dt: 2e4, ..TwoPhaseOptions::default() };
            let times: Vec<f64> = (1..=10).map(|i| i as f64 * 2e4).collect();
            let r = run_two_phase(&m, &ws, &fx, &Corey::default(), &Physics::default(), &vec![0.2; n], &vec![1.05e7; n], &times, &opts).unwrap();
            r.sw.last().unwrap().iter().sum::<f64>()
        }
        _ => panic!("mode"),
    };
    let wall = t0.elapsed().as_secs_f64();
    println!("{} cells={n} wall_s={wall:.3} peak_rss_mib={:.1} model_rss_mib={base:.1} check={check:.12e}", a[1], hwm_mib());
}
