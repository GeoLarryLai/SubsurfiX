// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Experimental FIM: energy equation (thermal water, CO2–brine) and the
//! CO2–brine OBL adapter. Conservation and analytic checks on real runs.

use sx_core::flow::co2_brine::{mutual_solubility, SolubilityModel, M_CO2, M_H2O};
use sx_core::flow::water::liquid;
use sx_core::flow::{
    run_fim, Co2Brine, FimOptions, FlowModel, FlowWell, OblPhysics, OblTable, RockThermal, ThermalWater, WellControl,
    WellKind,
};
use sx_core::grid::RegularGrid;

const K: f64 = 1e-13;

fn well(name: &str, kind: WellKind, control: WellControl, cell: usize) -> FlowWell {
    FlowWell {
        name: name.into(),
        kind,
        control,
        cells: vec![cell],
        radius_m: 0.1,
        skin: 0.0,
        wi: None,
        ref_depth_m: None,
        bhp_limit: None,
        crossflow: false,
    }
}

fn line(n: usize, k: f64) -> FlowModel {
    let g = RegularGrid::new([0.0; 3], [1.0, 1.0, 1.0], [n, 1, 1]).unwrap();
    FlowModel::from_regular(&g, &vec![[k; 3]; n], &vec![0.2; n]).unwrap()
}

fn check_balance(r: &sx_core::flow::FimResult, init: &[f64], names: &[String], rel: f64) {
    for k in 0..r.times.len() {
        for (c, name) in names.iter().enumerate() {
            let d = r.mass_in_place[k][c] - init[c];
            let q = r.mass_net_in[k][c];
            let scale = init[c].abs().max(q.abs()).max(1.0);
            eprintln!("  t{k} {name}: change {d:.6e}, net in {q:.6e}");
            assert!((d - q).abs() < rel * scale, "{name}: change {d} vs net in {q}");
        }
    }
}

#[test]
fn fim_thermal_water_front_and_energy_balance() {
    let n = 100;
    let m = line(n, K);
    let rock = RockThermal::default();
    let ph = ThermalWater::new(rock, 0.0, 1e5);
    let (t0, t_inj, p0, q) = (350.0, 300.0, 1e7, 1e-5);
    let state0: Vec<f64> = (0..n).flat_map(|_| [p0, t0]).collect();
    let wells = [
        well("I", WellKind::Injector, WellControl::Rate(q), 0),
        well("P", WellKind::Producer, WellControl::Bhp(p0), n - 1),
    ];
    // Thermal front speed (advection): u ρc_w / (φ ρc_w + (1 − φ) ρc_r), ρc_w at the mean T.
    let w = liquid(p0, 0.5 * (t0 + t_inj)).unwrap();
    let rcw = w.density * w.cp;
    let v = q * rcw / (0.2 * rcw + 0.8 * rock.rho * rock.cp);
    let times = [20.0 / v, 50.0 / v];
    let opts = FimOptions {
        table_lo: vec![5e6, 290.0],
        table_hi: vec![3e7, 360.0],
        table_points: vec![26, 141],
        injection: vec![vec![t_inj], vec![t0]],
        dt_init: 3600.0,
        dt_max: times[0] / 100.0,
        ..FimOptions::default()
    };
    let r = run_fim(&m, &wells, &ph, &state0, &times, &opts).unwrap();
    let init = run_fim(&m, &wells, &ph, &state0, &[0.0], &opts).unwrap().mass_in_place[0].clone();
    eprintln!("thermal water: steps {}, cuts {}, newton {}, table pts {}, fallbacks {}", r.n_steps, r.n_cuts, r.n_newton, r.n_table_points, r.n_fallbacks);
    check_balance(&r, &init, &ph.component_names(), 1e-6);
    assert_eq!(r.n_fallbacks, 0);
    for (k, x_want) in [20.0, 50.0].iter().enumerate() {
        let tm = 0.5 * (t0 + t_inj);
        let i = (0..n).find(|&i| r.state[k][i * 2 + 1] > tm).unwrap();
        // Interpolate the crossing between cell centres i−1 and i.
        let (ta, tb) = (r.state[k][(i - 1) * 2 + 1], r.state[k][i * 2 + 1]);
        let x = (i as f64 - 0.5) + (tm - ta) / (tb - ta);
        eprintln!("thermal front t{k}: solver {x:.2} m vs advective {x_want:.2} m");
        assert!((x - x_want).abs() < 2.0, "front {x} vs {x_want}");
    }
}

fn co2_case(ph: &Co2Brine, n: usize, thermal: Option<(f64, f64)>) -> (sx_core::flow::FimResult, Vec<f64>) {
    let m = line(n, 1e-13);
    let p0 = 1.2e7;
    let q = 2e-6; // reservoir m³/s of the injected stream
    let state0: Vec<f64> = (0..n)
        .flat_map(|_| match thermal {
            Some((t0, _)) => vec![p0, 0.0, t0],
            None => vec![p0, 0.0],
        })
        .collect();
    let wells = [
        well("I", WellKind::Injector, WellControl::Rate(q), 0),
        well("P", WellKind::Producer, WellControl::Bhp(p0), n - 1),
    ];
    let (lo, hi, pts, inj) = match thermal {
        Some((t0, ti)) => (vec![8e6, 0.0, 300.0], vec![2e7, 1.0, 350.0], vec![13, 101, 26], vec![vec![1.0, ti], vec![0.0, t0]]),
        None => (vec![8e6, 0.0], vec![2e7, 1.0], vec![25, 201], vec![vec![1.0], vec![0.0]]),
    };
    let opts = FimOptions {
        table_lo: lo,
        table_hi: hi,
        table_points: pts,
        injection: inj,
        dt_init: 600.0,
        dt_max: 2.0 * 86400.0,
        tol: 1e-8,
        ..FimOptions::default()
    };
    let pv: f64 = m.pore_volume.iter().sum();
    let times = [0.15 * pv / q, 0.3 * pv / q];
    let r = run_fim(&m, &wells, ph, &state0, &times, &opts).unwrap();
    let init = run_fim(&m, &wells, ph, &state0, &[0.0], &opts).unwrap().mass_in_place[0].clone();
    (r, init)
}

#[test]
fn fim_co2_brine_injection_mass_balance_and_dissolved_co2() {
    let n = 50;
    for model in [SolubilityModel::SpycherPruess2005, SolubilityModel::ZiabakhshGanjiKooi2012] {
        let ph = Co2Brine::new(1.0, model, false, 323.15);
        let (r, init) = co2_case(&ph, n, None);
        let sg_max = (0..n).map(|i| r.saturation[1][i * 2 + 1]).fold(0.0, f64::max);
        eprintln!("CO2-brine {model:?}: steps {}, cuts {}, newton {}, table pts {}, fallbacks {}, max Sg {sg_max:.3}", r.n_steps, r.n_cuts, r.n_newton, r.n_table_points, r.n_fallbacks);
        check_balance(&r, &init, &ph.component_names(), 1e-6);
        assert!(sg_max > 0.2, "a CO2 plume should form");
        assert_eq!(r.n_fallbacks, 0);
        // Dissolved CO2 in two-phase cells, from the solver's own interpolated
        // aqueous flux operators, vs the property module's solubility.
        let table = OblTable::new(&ph, &[8e6, 0.0], &[2e7, 1.0], &[25, 201]).unwrap();
        let lay = ph.layout();
        let mut n_checked = 0;
        for i in 0..n {
            let s = &r.state[1][i * 2..i * 2 + 2];
            let sg = r.saturation[1][i * 2 + 1];
            if !(sg > 0.05 && sg < 0.7) {
                continue;
            }
            let mut v = vec![0.0; lay.n_ops()];
            let mut gr = vec![0.0; lay.n_ops() * 2];
            table.interpolate(s, &mut v, &mut gr);
            let x_solver = v[lay.beta(0, 0)] / (v[lay.beta(0, 0)] + v[lay.beta(1, 0)]);
            let sol = mutual_solubility(s[0], 323.15, 1.0, model).unwrap();
            let x_mod = sol.x_co2 * M_CO2 / (sol.x_co2 * M_CO2 + (1.0 - sol.x_co2) * M_H2O);
            if n_checked == 0 {
                eprintln!("  cell {i}: p {:.4e}, Sg {sg:.3}: aqueous CO2 mass fraction solver {x_solver:.5} vs module {x_mod:.5}", s[0]);
            }
            assert!((x_solver / x_mod - 1.0).abs() < 0.01, "cell {i}: {x_solver} vs {x_mod}");
            n_checked += 1;
        }
        assert!(n_checked > 3, "need two-phase cells to check ({n_checked})");
        // Ahead of the plume: dissolved-only cells never exceed the solubility.
        for i in 0..n {
            let s = &r.state[1][i * 2..i * 2 + 2];
            if r.saturation[1][i * 2 + 1] == 0.0 && s[1] > 1e-6 {
                let sol = mutual_solubility(s[0], 323.15, 1.0, model).unwrap();
                let x_mod = sol.x_co2 * M_CO2 / (sol.x_co2 * M_CO2 + (1.0 - sol.x_co2) * M_H2O);
                assert!(s[1] <= x_mod * 1.0001, "cell {i}: z {} above solubility {x_mod}", s[1]);
            }
        }
    }
}

#[test]
fn fim_co2_brine_cold_injection_energy_balance() {
    let n = 30;
    let ph = Co2Brine::new(1.0, SolubilityModel::SpycherPruess2005, true, 0.0);
    let (r, init) = co2_case(&ph, n, Some((340.0, 305.0)));
    let tmin = (0..n).map(|i| r.state[1][i * 3 + 2]).fold(f64::INFINITY, f64::min);
    eprintln!("CO2-brine thermal: steps {}, cuts {}, newton {}, table pts {}, fallbacks {}, min T {tmin:.2} K", r.n_steps, r.n_cuts, r.n_newton, r.n_table_points, r.n_fallbacks);
    check_balance(&r, &init, &ph.component_names(), 1e-6);
    assert!(tmin < 330.0, "cold front should cool the near-well cells");
    assert!(r.state[1][(n - 1) * 3 + 2] > 339.0, "far cells stay warm");
}
