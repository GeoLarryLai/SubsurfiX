// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Regression: the 0.1.1 flow core (cached factorizations, CSC assembly,
//! threading) must reproduce the 0.1.0 outputs. `data/flow_golden_v010.bin`
//! was written by `flow_perf dump` built from the 0.1.0 sources.

#[path = "common/flow_cases.rs"]
mod flow_cases;
use flow_cases::*;
use std::sync::Mutex;
use sx_core::flow::{
    run_heat, run_heat_with, run_two_phase, run_two_phase_with, solve_pressure, with_threads, Corey, NpyDirSink,
    Physics, ReportOptions, Snapshot, ThermalProps, TwoPhaseOptions,
};

fn f64s(b: &[u8]) -> Vec<f64> {
    b.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    a == b || (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
}

#[test]
fn matches_v010_golden_outputs() {
    let gold = f64s(include_bytes!("data/flow_golden_v010.bin"));
    let new = f64s(&dump_bytes());
    assert_eq!(gold.len(), new.len(), "output layout changed");
    let mut worst = (0usize, 0.0f64);
    for (i, (a, b)) in gold.iter().zip(&new).enumerate() {
        let r = if a == b { 0.0 } else { (a - b).abs() / a.abs().max(b.abs()) };
        if r > worst.1 {
            worst = (i, r);
        }
    }
    assert!(worst.1 <= 1e-10, "value {} differs: rel {:e} ({} vs {})", worst.0, worst.1, gold[worst.0], new[worst.0]);
}

/// A grid large enough to take the threaded loops and parallel factorization.
#[test]
fn thread_count_does_not_change_results() {
    let (nx, ny, nz) = (60, 60, 8);
    let m = model(nx, ny, nz);
    let n = m.n_cells();
    let ws = wells(nx, ny, nz);
    let fx = fixed(&m, nx);
    let run = |t| {
        with_threads(Some(t), || {
            let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &ws, &fx)?;
            let mut h = heat_inputs(&m, n, 2, 1, false);
            h.times = vec![1e6, 2e6];
            let r = run_heat(&m, &sol, &ws, &fx, &ThermalProps::default(), &h)?;
            Ok((sol.p, r.temperature.last().unwrap().clone()))
        })
        .unwrap()
    };
    let (p1, t1) = run(1);
    let (p4, t4) = run(4);
    for (a, b) in p1.iter().zip(&p4).chain(t1.iter().zip(&t4)) {
        assert!(close(*a, *b, 1e-9), "{a} vs {b}");
    }
}

#[test]
fn report_decimation_and_sink() {
    let (nx, ny, nz) = (12, 10, 3);
    let m = model(nx, ny, nz);
    let n = m.n_cells();
    let ws = wells(nx, ny, nz);
    let fx = fixed(&m, nx);
    let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &ws, &fx).unwrap();
    let mut h = heat_inputs(&m, n, 2, 1, false);
    h.times = (1..=7).map(|i| i as f64 * 1e6).collect();
    let full = run_heat(&m, &sol, &ws, &fx, &ThermalProps::default(), &h).unwrap();
    let seen = Mutex::new(Vec::new());
    let mut sink = |s: &Snapshot<'_>| {
        seen.lock().unwrap().push((s.index, s.time, s.fields[0].1.to_vec()));
        Ok(())
    };
    let mut rep = ReportOptions { every: 3, in_memory: true, sink: Some(&mut sink) };
    let dec = run_heat_with(&m, &sol, &ws, &fx, &ThermalProps::default(), &h, &mut rep).unwrap();
    let kept = [0usize, 3, 6];
    assert_eq!(dec.temperature.len(), kept.len());
    for (k, &i) in kept.iter().enumerate() {
        assert_eq!(dec.temperature[k], full.temperature[i]);
    }
    assert_eq!(dec.energy_change, full.energy_change);
    let seen = seen.into_inner().unwrap();
    assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), kept);
    for (s, &i) in seen.iter().zip(&kept) {
        assert_eq!(s.1, h.times[i]);
        assert_eq!(s.2, full.temperature[i]);
    }

    // Two-phase streamed to .npy, nothing kept in memory.
    let dir = std::env::temp_dir().join(format!("sx_flow_npy_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut npy = NpyDirSink::new(&dir).unwrap();
    let mut w = |s: &Snapshot<'_>| npy.write(s);
    let times = [1e5, 2e5, 3e5, 4e5];
    let opts = TwoPhaseOptions::default();
    let s0 = vec![0.2; n];
    let p0 = vec![1.05e7; n];
    let base = run_two_phase(&m, &ws, &fx, &Corey::default(), &Physics::default(), &s0, &p0, &times, &opts).unwrap();
    let mut rep = ReportOptions { every: 2, in_memory: false, sink: Some(&mut w) };
    let r = run_two_phase_with(&m, &ws, &fx, &Corey::default(), &Physics::default(), &s0, &p0, &times, &opts, &mut rep)
        .unwrap();
    assert!(r.sw.is_empty() && r.p.is_empty());
    assert_eq!(r.bhp, base.bhp);
    for i in [0usize, 2, 3] {
        let b = std::fs::read(dir.join(format!("sw_{i:05}.npy"))).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!(f64s(&b[10 + hl..]), base.sw[i]);
    }
    assert!(!dir.join("sw_00001.npy").exists());
    let csv = std::fs::read_to_string(dir.join("times.csv")).unwrap();
    assert_eq!(csv.lines().count(), 4);
    std::fs::remove_dir_all(&dir).unwrap();
}
