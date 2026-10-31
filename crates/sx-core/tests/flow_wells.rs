// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Rate control with a bottom-hole pressure (BHP) limit, and per-completion crossflow.

use sx_core::flow::{
    run_two_phase, solve_pressure, Corey, FixedPressure, FlowModel, FlowWell, Physics, PressureSolution,
    TwoPhaseOptions, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

const K: f64 = 1e-13;
const MOB: f64 = 1e3;

fn line(n: usize) -> FlowModel {
    let g = RegularGrid::new([0.0; 3], [10.0, 10.0, 5.0], [n, 1, 1]).unwrap();
    FlowModel::from_regular(&g, &vec![[K; 3]; n], &vec![0.2; n]).unwrap()
}

fn with_limit(kind: WellKind, q: f64, lim: f64, cells: Vec<usize>) -> FlowWell {
    FlowWell { bhp_limit: Some(lim), ..FlowWell::vertical("W", kind, WellControl::Rate(q), cells, 0.1, 0.0) }
}

fn balance(m: &FlowModel, sol: &PressureSolution, wells: &[FlowWell], fixed: &[FixedPressure]) -> f64 {
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
    r.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

#[test]
fn producer_switches_to_bhp_limit_and_back() {
    let m = line(10);
    let fixed = [FixedPressure { cell: 0, trans: m.boundary_trans(0, 0), pressure: 2e7 }];
    let mob = vec![MOB; 10];
    // Small rate: stays on rate control.
    let w = [with_limit(WellKind::Producer, 1e-6, 1.5e7, vec![9])];
    let s = solve_pressure(&m, &mob, MOB, &w, &fixed).unwrap();
    assert!(!s.wells[0].bhp_limited && (s.wells[0].rate + 1e-6).abs() < 1e-15 && s.wells[0].bhp > 1.5e7);
    // Large rate: BHP would fall below the limit, so the well runs at the limit with less rate.
    let w = [with_limit(WellKind::Producer, 1.0, 1.5e7, vec![9])];
    let s = solve_pressure(&m, &mob, MOB, &w, &fixed).unwrap();
    assert!(s.wells[0].bhp_limited && s.wells[0].bhp == 1.5e7);
    assert!(s.wells[0].rate < 0.0 && -s.wells[0].rate < 1.0);
    let bhp_run = solve_pressure(
        &m,
        &mob,
        MOB,
        &[FlowWell::vertical("W", WellKind::Producer, WellControl::Bhp(1.5e7), vec![9], 0.1, 0.0)],
        &fixed,
    )
    .unwrap();
    assert!((s.wells[0].rate / bhp_run.wells[0].rate - 1.0).abs() < 1e-12);
    assert!(balance(&m, &s, &w, &fixed) < 1e-12 * s.wells[0].rate.abs());
    // An injector against a maximum BHP.
    let w = [
        with_limit(WellKind::Injector, 1.0, 2.5e7, vec![5]),
        FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1e7), vec![9], 0.1, 0.0),
    ];
    let s = solve_pressure(&m, &mob, MOB, &w, &fixed).unwrap();
    assert!(s.wells[0].bhp_limited && s.wells[0].bhp == 2.5e7 && !s.wells[1].bhp_limited);
    // A limit that is not finite is refused.
    let bad = [with_limit(WellKind::Producer, 1.0, f64::NAN, vec![9])];
    assert!(solve_pressure(&m, &mob, MOB, &bad, &fixed).is_err());
}

/// Two isolated layers (no vertical transmissibility) held at 20 and 10 MPa
/// by lateral boundaries, with one well completed in both.
fn two_layers() -> (FlowModel, [FixedPressure; 2]) {
    let g = RegularGrid::new([0.0; 3], [10.0, 10.0, 5.0], [1, 1, 2]).unwrap();
    let m = FlowModel::from_regular(&g, &[[K, K, 0.0]; 2], &[0.2; 2]).unwrap();
    assert!(m.connections.is_empty());
    let fixed = [
        FixedPressure { cell: 0, trans: m.boundary_trans(0, 0), pressure: 2e7 },
        FixedPressure { cell: 1, trans: m.boundary_trans(1, 0), pressure: 1e7 },
    ];
    (m, fixed)
}

#[test]
fn producer_crossflow_uses_wellbore_mixture_or_is_shut() {
    let (m, fixed) = two_layers();
    let mob = [MOB, 4.0 * MOB];
    let w = [FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1.6e7), vec![0, 1], 0.1, 0.0)];
    let s = solve_pressure(&m, &mob, MOB, &w, &fixed).unwrap();
    let r = &s.wells[0];
    assert!(r.cell_rates[0] < 0.0 && r.cell_rates[1] > 0.0, "{:?}", r.cell_rates);
    // Crossflow into layer 2 carries the wellbore mixture, i.e. layer 1's mobility.
    let want = r.wi[1] * MOB * (1.6e7 - s.p[1]);
    assert!((r.cell_rates[1] / want - 1.0).abs() < 1e-9);
    assert!(balance(&m, &s, &w, &fixed) < 1e-12 * r.cell_rates[0].abs());
    assert!(s.warnings.is_empty());
    // Crossflow off: the injecting completion is shut.
    let w = [FlowWell { crossflow: false, ..w[0].clone() }];
    let s = solve_pressure(&m, &mob, MOB, &w, &fixed).unwrap();
    assert!(s.wells[0].cell_rates[0] < 0.0 && s.wells[0].cell_rates[1] == 0.0);
    assert!((s.p[1] - 1e7).abs() < 1e-6);
}

#[test]
fn injector_backflow_uses_cell_mobility() {
    let (m, fixed) = two_layers();
    let mob = [2.0 * MOB, MOB];
    // Rate injector whose BHP ends between the layer pressures: layer 1 backflows into the well.
    let w = [FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(1e-6), vec![0, 1], 0.1, 0.0)];
    let s = solve_pressure(&m, &mob, 10.0 * MOB, &w, &fixed).unwrap();
    let r = &s.wells[0];
    assert!(r.bhp < s.p[0] && r.cell_rates[0] < 0.0 && r.cell_rates[1] > 0.0, "{r:?}");
    assert!((r.cell_rates[0] / (r.wi[0] * 2.0 * MOB * (r.bhp - s.p[0])) - 1.0).abs() < 1e-9);
    assert!((r.rate - 1e-6).abs() < 1e-15);
    assert!(balance(&m, &s, &w, &fixed) < 1e-12);
}

#[test]
fn two_phase_producer_respects_bhp_limit_and_crossflow_balances() {
    let n = 20;
    let g = RegularGrid::new([0.0; 3], [5.0, 5.0, 5.0], [n, 1, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n], &vec![0.2; n]).unwrap();
    let corey = Corey::default();
    let ph = Physics { c_rock: 1e-9, ..Physics::default() };
    let wells = [
        FlowWell::vertical("I", WellKind::Injector, WellControl::Bhp(2.2e7), vec![0], 0.1, 0.0),
        with_limit(WellKind::Producer, 1e-3, 1.8e7, vec![n - 1]),
    ];
    let mut sw0 = vec![corey.swc; n];
    sw0[0] = 0.8;
    let r = run_two_phase(&m, &wells, &[], &corey, &ph, &sw0, &vec![2e7; n], &[1e5, 1e6], &TwoPhaseOptions::default())
        .unwrap();
    for (k, lim) in r.bhp_limited.iter().enumerate() {
        assert!(lim[1], "report {k}: producer should be at its limit");
        assert!((r.bhp[k][1] - 1.8e7).abs() < 1e-6);
        assert!(-(r.water_rate[k][1] + r.oil_rate[k][1]) < 1e-3);
    }
    // Two-layer crossflow in a two-phase run: water mass balance closes.
    let gl = RegularGrid::new([0.0; 3], [10.0, 10.0, 5.0], [3, 1, 2]).unwrap();
    let mut perm = vec![[K, K, 0.0]; 6];
    perm[3] = [10.0 * K, 10.0 * K, 0.0];
    let ml = FlowModel::from_regular(&gl, &perm, &[0.2; 6]).unwrap();
    let fixed = [
        FixedPressure { cell: 2, trans: ml.boundary_trans(2, 0), pressure: 2.2e7 },
        FixedPressure { cell: 5, trans: ml.boundary_trans(5, 0), pressure: 1.4e7 },
    ];
    let w = [FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1.8e7), vec![0, 3], 0.1, 0.0)];
    let sw = [0.6, 0.6, 0.6, corey.swc, corey.swc, corey.swc];
    let r = run_two_phase(&ml, &w, &fixed, &corey, &ph, &sw, &[2e7; 6], &[1e5, 1e6], &TwoPhaseOptions { boundary_fw: 0.0, ..TwoPhaseOptions::default() })
        .unwrap();
    let (d_wip, d_in) = (r.water_in_place[1] - r.water_in_place[0], r.water_net_in[1] - r.water_net_in[0]);
    assert!(d_in.abs() > 0.0 && (d_wip - d_in).abs() < 1e-8 * d_in.abs().max(r.water_in_place[0]), "{d_wip} vs {d_in}");
    assert!(r.water_clamp_loss[1] == 0.0);
}
