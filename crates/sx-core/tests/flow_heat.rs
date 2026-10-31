// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Heat transport: Ogata–Banks, pure conduction (erfc) and energy balance.

use sx_core::flow::analytic::{conduction_step, ogata_banks};
use sx_core::flow::{
    run_heat, solve_pressure, FixedPressure, FixedTemperature, FlowModel, HeatFlux, FlowWell, HeatInputs, PressureSolution, StepControl,
    ThermalProps, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

const K: f64 = 1e-12;
const MU: f64 = 1e-3;
const PHI: f64 = 0.2;

fn column(n: usize, dx: f64) -> FlowModel {
    let g = RegularGrid::new([0.0; 3], [dx, 1.0, 1.0], [n, 1, 1]).unwrap();
    FlowModel::from_regular(&g, &vec![[K; 3]; n], &vec![PHI; n]).unwrap()
}

fn lumped(p: &ThermalProps) -> (f64, f64) {
    let cap = PHI * p.rho_fluid * p.cp_fluid + (1.0 - PHI) * p.rho_rock * p.cp_rock;
    let lam = PHI * p.cond_fluid + (1.0 - PHI) * p.cond_rock;
    (cap, lam)
}

/// Max abs error of the 1D advection–conduction front against Ogata–Banks.
pub fn ogata_banks_error() -> f64 {
    let (n, dx) = (1000, 0.05);
    let m = column(n, dx);
    let props = ThermalProps::default();
    let (cap, lam) = lumped(&props);
    let u = 1e-6; // Darcy flux (m/s), 1 m² cross-section
    let wells = [FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(u), vec![0], 0.1, 0.0)];
    let fixed = [FixedPressure { cell: n - 1, trans: m.boundary_trans(n - 1, 0), pressure: 1e7 }];
    let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &wells, &fixed).unwrap();
    let v = u * props.rho_fluid * props.cp_fluid / cap;
    let d = lam / cap;
    let t_end = 20.0 / v;
    let inp = HeatInputs {
        well_temp: vec![1.0],
        fixed_temp: vec![0.0],
        fixed_t: vec![FixedTemperature { cell: 0, conductance: lam * m.boundary_geom(0, 0), temperature: 1.0 }],
        heat_flux: vec![],
        t0: vec![0.0; n],
        times: vec![t_end],
        control: None,
        dt_max: 1e4,
    };
    let r = run_heat(&m, &sol, &wells, &fixed, &props, &inp).unwrap();
    (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            (r.temperature[0][i] - ogata_banks(x, t_end, v, d)).abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn ogata_banks_front() {
    let err = ogata_banks_error();
    println!("OGATA_BANKS max abs error = {err:.5}");
    assert!(err < 0.01, "Ogata–Banks max abs error {err}");
}

#[test]
fn pure_conduction_matches_erfc() {
    let (n, dx) = (400, 0.1);
    let m = column(n, dx);
    let props = ThermalProps::default();
    let (cap, lam) = lumped(&props);
    let alpha = lam / cap;
    let t_end = 4.0 / alpha; // sqrt(alpha t) = 2 m
    let sol = PressureSolution {
        p: vec![0.0; n],
        flux: vec![0.0; m.connections.len()],
        boundary_flux: vec![],
        wells: vec![],
        upwind_iterations: 0,
        warnings: vec![],
    };
    let inp = HeatInputs {
        well_temp: vec![],
        fixed_temp: vec![],
        fixed_t: vec![FixedTemperature { cell: 0, conductance: lam * m.boundary_geom(0, 0), temperature: 1.0 }],
        heat_flux: vec![],
        t0: vec![0.0; n],
        times: vec![t_end],
        control: None,
        dt_max: t_end / 400.0,
    };
    let r = run_heat(&m, &sol, &[], &[], &props, &inp).unwrap();
    let err = (0..n)
        .map(|i| (r.temperature[0][i] - conduction_step((i as f64 + 0.5) * dx, t_end, alpha)).abs())
        .fold(0.0, f64::max);
    println!("CONDUCTION max abs error = {err:.5}");
    assert!(err < 0.005, "erfc conduction max abs error {err}");
}

#[test]
fn energy_balance_closes_in_2d() {
    let (nx, ny) = (12, 9);
    let n = nx * ny;
    let g = RegularGrid::new([0.0; 3], [20.0, 20.0, 10.0], [nx, ny, 1]).unwrap();
    let perm: Vec<[f64; 3]> = (0..n).map(|c| [K * (1.0 + (c % 5) as f64); 3]).collect();
    let m = FlowModel::from_regular(&g, &perm, &vec![PHI; n]).unwrap();
    let wells = [
        FlowWell::vertical("I", WellKind::Injector, WellControl::Rate(2e-3), vec![0], 0.1, 0.0),
        FlowWell::vertical("P", WellKind::Producer, WellControl::Bhp(1e7), vec![n - 1], 0.1, 0.0),
    ];
    let fixed = [FixedPressure { cell: nx - 1, trans: m.boundary_trans(nx - 1, 0), pressure: 1.05e7 }];
    let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &wells, &fixed).unwrap();
    let props = ThermalProps::default();
    let inp = HeatInputs {
        well_temp: vec![40.0, 0.0],
        fixed_temp: vec![90.0],
        fixed_t: vec![FixedTemperature { cell: n / 2, conductance: 50.0, temperature: 120.0 }],
        heat_flux: vec![],
        t0: vec![80.0; n],
        times: vec![3e6, 3e7, 1e8],
        control: None,
        dt_max: 2e6,
    };
    let r = run_heat(&m, &sol, &wells, &fixed, &props, &inp).unwrap();
    for (de, ein) in r.energy_change.iter().zip(&r.energy_in) {
        let rel = (de - ein).abs() / ein.abs().max(1.0);
        assert!(rel < 1e-9, "energy balance {de} vs {ein} ({rel})");
    }
    // Cooling around the injector, bounded by the extreme input temperatures.
    let last = r.temperature.last().unwrap();
    assert!(last[0] < 80.0 && last[0] >= 40.0 - 1e-9);
    assert!(last.iter().all(|t| *t >= 40.0 - 1e-6 && *t <= 120.0 + 1e-6));
}

#[test]
fn rejects_bad_heat_input() {
    let m = column(5, 1.0);
    let sol = PressureSolution {
        p: vec![0.0; 5],
        flux: vec![0.0; 4],
        boundary_flux: vec![],
        wells: vec![],
        upwind_iterations: 0,
        warnings: vec![],
    };
    let ok = HeatInputs {
        well_temp: vec![],
        fixed_temp: vec![],
        fixed_t: vec![],
        heat_flux: vec![],
        t0: vec![0.0; 5],
        times: vec![1.0],
        control: None,
        dt_max: 1.0,
    };
    let p = ThermalProps::default();
    assert!(run_heat(&m, &sol, &[], &[], &p, &ok).is_ok());
    let bad = HeatInputs { t0: vec![0.0; 4], ..ok.clone() };
    assert!(run_heat(&m, &sol, &[], &[], &p, &bad).is_err());
    let bad = HeatInputs { times: vec![2.0, 1.0], ..ok.clone() };
    assert!(run_heat(&m, &sol, &[], &[], &p, &bad).is_err());
    let bad_p = ThermalProps { cp_fluid: -1.0, ..p };
    assert!(run_heat(&m, &sol, &[], &[], &bad_p, &ok).is_err());
}

#[test]
fn adaptive_steps_grow_reject_and_stay_accurate() {
    let (n, dx) = (400, 0.1);
    let m = column(n, dx);
    let props = ThermalProps::default();
    let (cap, lam) = lumped(&props);
    let alpha = lam / cap;
    let t_end = 4.0 / alpha;
    let sol = PressureSolution {
        p: vec![0.0; n],
        flux: vec![0.0; m.connections.len()],
        boundary_flux: vec![],
        wells: vec![],
        upwind_iterations: 0,
        warnings: vec![],
    };
    let base = HeatInputs {
        well_temp: vec![],
        fixed_temp: vec![],
        fixed_t: vec![FixedTemperature { cell: 0, conductance: lam * m.boundary_geom(0, 0), temperature: 1.0 }],
        heat_flux: vec![],
        t0: vec![0.0; n],
        times: vec![t_end / 4.0, t_end],
        dt_max: t_end,
        control: Some(StepControl::new(t_end / 1e5, 0.01)),
    };
    let r = run_heat(&m, &sol, &[], &[], &props, &base).unwrap();
    let err = (0..n)
        .map(|i| (r.temperature[1][i] - conduction_step((i as f64 + 0.5) * dx, t_end, alpha)).abs())
        .fold(0.0, f64::max);
    println!("ADAPTIVE steps {} rejected {} err {err:.5}", r.n_steps, r.n_rejected);
    assert!(err < 0.005, "adaptive conduction error {err}");
    assert!(r.n_steps < 400, "steps grew: {}", r.n_steps);
    for (de, ein) in r.energy_change.iter().zip(&r.energy_in) {
        assert!((de - ein).abs() < 1e-9 * ein.abs().max(1.0));
    }
    // A first step far too long is rejected and repeated.
    let big = HeatInputs { control: Some(StepControl::new(t_end, 0.01)), ..base.clone() };
    let r = run_heat(&m, &sol, &[], &[], &props, &big).unwrap();
    assert!(r.n_rejected > 0);
    // A floor that cannot be met is an error, not a silent large step.
    let floor = HeatInputs {
        control: Some(StepControl { dt_min: t_end, ..StepControl::new(t_end, 0.01) }),
        ..base.clone()
    };
    assert!(run_heat(&m, &sol, &[], &[], &props, &floor).is_err());
    let bad = HeatInputs { control: Some(StepControl::new(1.0, -1.0)), ..base };
    assert!(run_heat(&m, &sol, &[], &[], &props, &bad).is_err());
}

#[test]
fn thermal_props_from_iapws_water() {
    let p = ThermalProps::default().with_water(10e6, 353.15).unwrap();
    let w = sx_core::flow::water::liquid(10e6, 353.15).unwrap();
    assert_eq!((p.rho_fluid, p.cp_fluid, p.cond_fluid), (w.density, w.cp, w.conductivity));
    assert!(p.rho_fluid > 970.0 && p.rho_fluid < 980.0 && p.cp_fluid > 4150.0 && p.cp_fluid < 4220.0);
    assert_eq!(p.rho_rock, ThermalProps::default().rho_rock);
    assert!(ThermalProps::default().with_water(1e5, 400.0).is_err());
}

/// Fixed top temperature plus a basal heat flux: the steady state is the
/// linear conductive geotherm T = T0 + q x / λ, and energy balances on the way.
#[test]
fn basal_heat_flux_reaches_linear_geotherm() {
    let (n, dx) = (40, 25.0);
    let m = column(n, dx);
    let props = ThermalProps::default();
    let (_, lam) = lumped(&props);
    let fixed = [FixedPressure { cell: 0, trans: m.boundary_trans(0, 0), pressure: 1e7 }];
    let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &[], &fixed).unwrap();
    assert!(sol.flux.iter().all(|q| q.abs() < 1e-12));
    let (t0, q) = (15.0, 0.06); // °C, W/m² over the 1 m² face
    let base = HeatInputs {
        well_temp: vec![],
        fixed_temp: vec![t0],
        fixed_t: vec![FixedTemperature { cell: 0, conductance: lam * m.boundary_geom(0, 0), temperature: t0 }],
        heat_flux: vec![HeatFlux { cell: n - 1, power: q }],
        t0: vec![t0; n],
        times: vec![3.15e10, 3.15e13],
        dt_max: 3.15e11,
        control: None,
    };
    let r = run_heat(&m, &sol, &[], &fixed, &props, &base).unwrap();
    for (k, e) in r.energy_change.iter().enumerate() {
        assert!((e - r.energy_in[k]).abs() <= 1e-9 * e.abs().max(1.0));
    }
    let last = r.temperature.last().unwrap();
    for (i, t) in last.iter().enumerate() {
        let want = t0 + q * (i as f64 + 0.5) * dx / lam;
        assert!((t - want).abs() < 1e-6, "cell {i}: {t} vs {want}");
    }
    assert!(r.temperature[0][n - 1] < last[n - 1]);
    let bad = HeatInputs { heat_flux: vec![HeatFlux { cell: n, power: 1.0 }], ..base.clone() };
    assert!(run_heat(&m, &sol, &[], &fixed, &props, &bad).is_err());
    let bad = HeatInputs { heat_flux: vec![HeatFlux { cell: 0, power: f64::NAN }], ..base };
    assert!(run_heat(&m, &sol, &[], &fixed, &props, &bad).is_err());
}
