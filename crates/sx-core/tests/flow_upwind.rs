// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Upwind face mobility in `solve_pressure` / `run_impes`: directions, mass balance, convergence.

use sx_core::flow::{
    run_impes, solve_pressure, Corey, FixedPressure, FlowModel, FlowWell, ImpesOptions, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

const K: f64 = 1e-13;

fn model(nx: usize, ny: usize) -> FlowModel {
    let g = RegularGrid::new([0.0; 3], [10.0, 10.0, 5.0], [nx, ny, 1]).unwrap();
    FlowModel::from_regular(&g, &vec![[K; 3]; nx * ny], &vec![0.2; nx * ny]).unwrap()
}

/// Net outflow per cell from faces, wells and boundaries (should be zero).
fn residual(m: &FlowModel, sol: &sx_core::flow::PressureSolution, wells: &[FlowWell], fixed: &[FixedPressure]) -> Vec<f64> {
    let mut r = vec![0.0; m.n_cells()];
    for (c, q) in m.connections.iter().zip(&sol.flux) {
        r[c.a] -= q;
        r[c.b] += q;
    }
    for (w, res) in wells.iter().zip(&sol.wells) {
        for (&c, q) in w.cells.iter().zip(&res.cell_rates) {
            r[c] += q;
        }
    }
    for (f, q) in fixed.iter().zip(&sol.boundary_flux) {
        r[f.cell] += q;
    }
    r
}

#[test]
fn face_mobility_is_upstream_and_mass_balances() {
    let n = 20;
    let m = model(n, 1);
    // Mobility rises downstream, so upstream and mean differ on every face.
    let mob: Vec<f64> = (0..n).map(|i| 1.0 / 1e-3 * (1.0 + i as f64)).collect();
    let wells = [FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(1e-4), vec![0], 0.1, 0.0)];
    let fixed = [FixedPressure { cell: n - 1, trans: m.boundary_trans(n - 1, 0), pressure: 1e7 }];
    let sol = solve_pressure(&m, &mob, 1e3, &wells, &fixed).unwrap();
    assert!(sol.warnings.is_empty());
    assert!(sol.upwind_iterations >= 1 && sol.upwind_iterations <= 3, "{}", sol.upwind_iterations);
    for (c, q) in m.connections.iter().zip(&sol.flux) {
        assert!(*q > 0.0);
        let expect = c.trans * mob[c.a] * (sol.p[c.a] - sol.p[c.b]);
        assert!((q - expect).abs() <= 1e-9 * q.abs(), "face {}-{}", c.a, c.b);
    }
    let r = residual(&m, &sol, &wells, &fixed);
    let worst = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(worst < 1e-12 * 1e-4 * 1e3, "residual {worst}");
    assert!((sol.wells[0].rate + sol.boundary_flux[0]).abs() < 1e-12);
}

#[test]
fn opposing_wells_converge_in_2d() {
    let (nx, ny) = (15, 15);
    let m = model(nx, ny);
    let n = nx * ny;
    let mob: Vec<f64> = (0..n).map(|c| 1e3 * (1.0 + ((c * 7919) % 13) as f64)).collect();
    let wells = [
        FlowWell::vertical("I", WellKind::Injector, WellControl::Bhp(2e7), vec![0], 0.1, 0.0),
        FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1e7), vec![n - 1], 0.1, 0.0),
        FlowWell::vertical("I2", WellKind::Injector, WellControl::Rate(2e-4), vec![nx - 1], 0.1, 0.0),
    ];
    let sol = solve_pressure(&m, &mob, 5e3, &wells, &[]).unwrap();
    assert!(sol.warnings.is_empty(), "{:?}", sol.warnings);
    assert!(sol.upwind_iterations < sx_core::flow::UPWIND_MAX_ITER);
    let r = residual(&m, &sol, &wells, &[]);
    let worst = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(worst < 1e-10, "residual {worst}");
    let total: f64 = sol.wells.iter().map(|w| w.rate).sum();
    assert!(total.abs() < 1e-12, "{total}");
}

#[test]
fn impes_water_volume_balances_with_upwinding() {
    let n = 30;
    let m = model(n, 1);
    let corey = Corey::default();
    let wells = [
        FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(1e-5), vec![0], 0.1, 0.0),
        FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1e7), vec![n - 1], 0.1, 0.0),
    ];
    let sw0 = vec![corey.swc; n];
    let t_end = 0.3 * m.pore_volume.iter().sum::<f64>() / 1e-5;
    let opts = ImpesOptions::default();
    let r = run_impes(&m, &wells, &[], &corey, &sw0, &[t_end], &opts).unwrap();
    let water: f64 = r.sw[0].iter().zip(&m.pore_volume).map(|(s, v)| (s - corey.swc) * v).sum();
    // No water reaches the producer at 0.3 PV, so all injected water is in place.
    assert!((water / (1e-5 * t_end) - 1.0).abs() < 1e-6, "{}", water / (1e-5 * t_end));
}

#[test]
fn picard_fallback_is_forced_and_still_balances() {
    use sx_core::flow::{solve_pressure_with, PressureOptions};
    let n = 10;
    let m = model(n, 1);
    let mob: Vec<f64> = (0..n).map(|i| 1e3 * (1.0 + i as f64)).collect();
    let wells = [FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(1e-4), vec![0], 0.1, 0.0)];
    let fixed = [FixedPressure { cell: n - 1, trans: m.boundary_trans(n - 1, 0), pressure: 1e7 }];
    let o = PressureOptions { max_upwind_iter: 0, ..PressureOptions::default() };
    let sol = solve_pressure_with(&m, &mob, 1e3, &wells, &fixed, &o).unwrap();
    assert_eq!(sol.warnings.len(), 1);
    assert!(sol.warnings[0].contains("did not settle") && sol.warnings[0].contains("arithmetic-mean"));
    for (c, q) in m.connections.iter().zip(&sol.flux) {
        let mean = 0.5 * (mob[c.a] + mob[c.b]);
        assert!((q - c.trans * mean * (sol.p[c.a] - sol.p[c.b])).abs() <= 1e-9 * q.abs());
    }
    let worst = residual(&m, &sol, &wells, &fixed).iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn boundary_inflow_uses_boundary_mobility() {
    use sx_core::flow::{solve_pressure_with, PressureOptions};
    let n = 10;
    let m = model(n, 1);
    let mob = vec![1e3; n];
    let t = m.boundary_trans(0, 0);
    let fixed = [
        FixedPressure { cell: 0, trans: t, pressure: 2e7 },
        FixedPressure { cell: n - 1, trans: t, pressure: 1e7 },
    ];
    let o = PressureOptions { boundary_inflow_mobility: Some(1e4), ..PressureOptions::default() };
    let sol = solve_pressure_with(&m, &mob, 1e3, &[], &fixed, &o).unwrap();
    let q_in = sol.boundary_flux[0];
    assert!(q_in > 0.0);
    assert!((q_in - t * 1e4 * (2e7 - sol.p[0])).abs() <= 1e-9 * q_in);
    assert!((sol.boundary_flux[1] - t * 1e3 * (1e7 - sol.p[n - 1])).abs() <= 1e-9 * q_in);
    assert!((q_in + sol.boundary_flux[1]).abs() <= 1e-9 * q_in);
    // More mobile inflow fluid -> more flow than with the cell mobility.
    let base = solve_pressure(&m, &mob, 1e3, &[], &fixed).unwrap();
    assert!(q_in > base.boundary_flux[0]);
    assert!(solve_pressure_with(&m, &mob, 1e3, &[], &fixed, &PressureOptions { boundary_inflow_mobility: Some(-1.0), ..o }).is_err());
}
