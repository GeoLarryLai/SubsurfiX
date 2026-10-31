// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Shared flow cases for the `flow_perf` example and the regression test.
#![allow(dead_code)]

use sx_core::flow::{
    run_heat, run_two_phase, solve_pressure, Capillary, Corey, FixedPressure, FixedTemperature, FlowModel, FlowWell,
    HeatInputs, Physics, StepControl, ThermalProps, TwoPhaseOptions, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

pub const MU: f64 = 1e-3;

pub fn model(nx: usize, ny: usize, nz: usize) -> FlowModel {
    let g = RegularGrid::new([0.0; 3], [10.0, 10.0, 2.0], [nx, ny, nz]).unwrap();
    let n = nx * ny * nz;
    let perm: Vec<[f64; 3]> = (0..n)
        .map(|c| {
            let f = 1.0 + 0.9 * ((c as f64) * 0.37).sin();
            [1e-13 * f, 1e-13 * f, 1e-14 * f]
        })
        .collect();
    let poro: Vec<f64> = (0..n).map(|c| 0.2 + 0.05 * ((c as f64) * 0.11).cos()).collect();
    FlowModel::from_regular(&g, &perm, &poro).unwrap()
}

pub fn wells(nx: usize, ny: usize, nz: usize) -> Vec<FlowWell> {
    let col = |i: usize, j: usize| (0..nz).map(|k| i + nx * (j + ny * k)).collect::<Vec<_>>();
    vec![
        FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(1e-3 * nz as f64), col(0, 0), 0.1, 0.0),
        FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1e7), col(nx - 1, ny - 1), 0.1, 0.0),
    ]
}

pub fn fixed(m: &FlowModel, nx: usize) -> Vec<FixedPressure> {
    vec![FixedPressure { cell: nx - 1, trans: m.boundary_trans(nx - 1, 0), pressure: 1.1e7 }]
}

pub fn heat_inputs(m: &FlowModel, n: usize, nw: usize, nf: usize, adaptive: bool) -> HeatInputs {
    HeatInputs {
        well_temp: vec![20.0; nw],
        fixed_temp: vec![60.0; nf],
        fixed_t: vec![FixedTemperature { cell: n - 1, conductance: 2.0 * m.boundary_geom(n - 1, 2), temperature: 90.0 }],
        heat_flux: vec![],
        t0: vec![80.0; n],
        times: vec![3.0e6, 1.0e7, 3.0e7],
        dt_max: 1.0e6,
        control: adaptive.then(|| StepControl::new(1e5, 2.0)),
    }
}
pub fn put(out: &mut Vec<u8>, v: &[f64]) {
    out.extend((v.len() as u64).to_le_bytes());
    for x in v {
        out.extend(x.to_le_bytes());
    }
}

/// Outputs of the fixed regression cases as little-endian f64 (length-prefixed blocks).
pub fn dump_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    // Case A: 3D pressure, rate + BHP wells, fixed boundary.
    let (nx, ny, nz) = (24, 20, 6);
    let m = model(nx, ny, nz);
    let n = m.n_cells();
    let ws = wells(nx, ny, nz);
    let fx = fixed(&m, nx);
    let mob: Vec<f64> = (0..n).map(|c| (1.0 + 0.5 * ((c as f64) * 0.05).sin()) / MU).collect();
    let sol = solve_pressure(&m, &mob, 1.0 / MU, &ws, &fx).unwrap();
    put(&mut out, &sol.p);
    put(&mut out, &sol.flux);
    put(&mut out, &sol.boundary_flux);
    for w in &sol.wells {
        put(&mut out, &[w.bhp, w.rate]);
        put(&mut out, &w.cell_rates);
    }
    // Case B: heat, fixed and adaptive steps.
    let sol1 = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &ws, &fx).unwrap();
    for adaptive in [false, true] {
        let r = run_heat(&m, &sol1, &ws, &fx, &ThermalProps::default(), &heat_inputs(&m, n, 2, 1, adaptive)).unwrap();
        for t in &r.temperature {
            put(&mut out, t);
        }
        put(&mut out, &r.energy_change);
        put(&mut out, &r.energy_in);
        put(&mut out, &[r.n_steps as f64, r.n_rejected as f64]);
    }
    // Case B2: pure conduction on a zero-flux field (symmetric system).
    let fz = vec![FixedPressure { cell: 0, trans: m.boundary_trans(0, 0), pressure: 1e7 }];
    let solz = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &[], &fz).unwrap();
    let mut h = heat_inputs(&m, n, 0, 1, false);
    h.t0 = (0..n).map(|c| 20.0 + (c % 7) as f64).collect();
    let r = run_heat(&m, &solz, &[], &fz, &ThermalProps::default(), &h).unwrap();
    for t in &r.temperature {
        put(&mut out, t);
    }
    put(&mut out, &r.energy_change);
    // Case C: two-phase IMPES, gravity + capillary + compressible + rate/BHP.
    let (nx, ny, nz) = (12, 10, 5);
    let m = model(nx, ny, nz);
    let n = m.n_cells();
    let ws = wells(nx, ny, nz);
    let ph = Physics {
        gravity: 9.81,
        capillary: Capillary::BrooksCorey { pd: 5e3, lambda: 2.0 },
        c_rock: 1e-9,
        c_w: 4e-10,
        c_o: 1e-9,
        ..Physics::default()
    };
    let corey = Corey::default();
    let opts = TwoPhaseOptions { max_dt: 1e5, ds_target: 0.2, dp_target: 2e6, ..TwoPhaseOptions::default() };
    let r = run_two_phase(&m, &ws, &[], &corey, &ph, &vec![0.25; n], &vec![1.05e7; n], &[1e5, 1e6, 3e6], &opts).unwrap();
    for (s, p) in r.sw.iter().zip(&r.p) {
        put(&mut out, s);
        put(&mut out, p);
    }
    for v in r.bhp.iter().chain(&r.water_rate).chain(&r.oil_rate) {
        put(&mut out, v);
    }
    put(&mut out, &r.water_in_place);
    put(&mut out, &r.oil_in_place);
    put(&mut out, &[r.n_steps as f64, r.n_rejected as f64]);
    // Case D: incompressible two-phase with fixed boundary.
    let fx = fixed(&m, nx);
    let r = run_two_phase(&m, &ws, &fx, &corey, &Physics::default(), &vec![0.2; n], &vec![1.05e7; n], &[1e5, 1e6], &TwoPhaseOptions::default()).unwrap();
    for (s, p) in r.sw.iter().zip(&r.p) {
        put(&mut out, s);
        put(&mut out, p);
    }
    put(&mut out, &[r.n_steps as f64]);
    out
}
