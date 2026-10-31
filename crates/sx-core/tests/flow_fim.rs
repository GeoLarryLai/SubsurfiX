// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Experimental fully implicit OBL engine: analytic and conservation checks.

use sx_core::flow::analytic::buckley_leverett;
use sx_core::flow::{
    run_fim, BlackOil, Corey, DeadOil, FimOptions, FlowModel, FlowWell, OblPhysics, OblTable, OpLayout, PvtRow,
    WellControl, WellKind,
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

#[test]
fn obl_table_interpolates_multilinear_exactly() {
    // DeadOil α_w with incompressible equal densities is ρ z: linear in z.
    let ph = DeadOil { rho_o: 1000.0, ..DeadOil::default() };
    let t = OblTable::new(&ph, &[1e5, 0.0], &[1e7, 1.0], &[5, 11]).unwrap();
    let lay = OpLayout { ne: 2, np: 2, thermal: false };
    let mut v = vec![0.0; lay.n_ops()];
    let mut g = vec![0.0; lay.n_ops() * 2];
    t.interpolate(&[3.3e6, 0.437], &mut v, &mut g);
    assert!((v[lay.alpha(0)] - 437.0).abs() < 1e-9);
    assert!((g[lay.alpha(0) * 2 + 1] - 1000.0).abs() < 1e-9);
    assert!(t.n_evaluated() == 4, "lazy: only the 4 corners of one hypercube");
}

fn bl_case(n: usize) -> (FlowModel, f64, Corey, DeadOil) {
    let g = RegularGrid::new([0.0; 3], [1.0, 1.0, 1.0], [n, 1, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n], &vec![0.2; n]).unwrap();
    let corey = Corey { mu_o: 2e-3, ..Corey::default() };
    let ph = DeadOil { corey, rho_w: 1000.0, rho_o: 1000.0, ..DeadOil::default() };
    (m, 1e-5, corey, ph)
}

#[test]
fn fim_buckley_leverett_front_and_mass_balance() {
    let n = 200;
    let (m, q, corey, ph) = bl_case(n);
    let pv: f64 = m.pore_volume.iter().sum();
    let tds = [0.2, 0.4];
    let times: Vec<f64> = tds.iter().map(|td| td * pv / q).collect();
    let wells = [
        well("I", WellKind::Injector, WellControl::Rate(q), 0),
        well("P", WellKind::Producer, WellControl::Bhp(1e7), n - 1),
    ];
    let z0 = ph.z_from_sw(1e7, corey.swc);
    let state0: Vec<f64> = (0..n).flat_map(|_| [1e7, z0]).collect();
    let opts = FimOptions {
        table_lo: vec![5e6, 0.0],
        table_hi: vec![1e8, 1.0],
        table_points: vec![16, 501],
        injection: vec![vec![1.0], vec![0.0]],
        dt_init: times[0] / 200.0,
        dt_max: times[0] / 100.0,
        ..FimOptions::default()
    };
    let r = run_fim(&m, &wells, &ph, &state0, &times, &opts).unwrap();
    for (k, &td) in tds.iter().enumerate() {
        let sw: Vec<f64> = (0..n).map(|i| r.saturation[k][i * 2]).collect();
        let l1: f64 = (0..n)
            .map(|i| (sw[i] - buckley_leverett(&corey, (i as f64 + 0.5) / n as f64, td)).abs())
            .sum::<f64>()
            / n as f64;
        let analytic: Vec<f64> = (0..2000).map(|i| buckley_leverett(&corey, i as f64 / 2000.0, td)).collect();
        let xa = analytic.iter().position(|&s| s <= corey.swc + 1e-12).unwrap() as f64 / 2000.0;
        let s_shock = analytic[((xa * 2000.0) as usize).saturating_sub(1)];
        let half = 0.5 * (corey.swc + s_shock);
        let xn = sw.iter().position(|&s| s < half).unwrap() as f64 / n as f64;
        eprintln!("FIM BL td={td}: L1 {l1:.4}, front numeric {xn:.4} vs analytic {xa:.4} (shock Sw {s_shock:.4}); steps {}, cuts {}, newton {}", r.n_steps, r.n_cuts, r.n_newton);
        assert!(l1 < 0.03, "td {td}: L1 {l1}");
        assert!((xn - xa).abs() < 0.03, "front {xn} vs {xa}");
        // Mass balance: water gained = water injected (no breakthrough), from solver output.
        let gained = r.mass_in_place[k][0] - pv * 1000.0 * corey.swc;
        let inj = r.mass_net_in[k][0];
        assert!((gained - inj).abs() < 1e-6 * inj, "gained {gained} vs net in {inj}");
        assert!((inj - 1000.0 * q * times[k]).abs() < 1e-4 * inj, "injected {inj}");
    }
}

#[test]
fn fim_gravity_column_stays_hydrostatic() {
    let nz = 10;
    let g = RegularGrid::new([0.0; 3], [10.0, 10.0, 5.0], [1, 1, nz]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; nz], &vec![0.2; nz]).unwrap();
    let ph = DeadOil { c_w: 4e-10, c_o: 1e-9, ..DeadOil::default() };
    // Segregated: oil on top, water below; start from a non-equilibrium uniform pressure.
    let state0: Vec<f64> = (0..nz)
        .flat_map(|k| {
            let sw = if k < nz / 2 { 0.2 } else { 0.8 };
            [2e7, ph.z_from_sw(2e7, sw)]
        })
        .collect();
    let opts = FimOptions {
        gravity: 9.80665,
        table_lo: vec![1e7, 0.0],
        table_hi: vec![3e7, 1.0],
        table_points: vec![64, 201],
        injection: vec![],
        dt_init: 3600.0,
        dt_max: 10.0 * 86400.0,
        ..FimOptions::default()
    };
    let r = run_fim(&m, &[], &ph, &state0, &[100.0 * 86400.0], &opts).unwrap();
    let st = &r.state[0];
    // Pressure gradient inside each fluid ≈ ρ g (within table interpolation error).
    let dp_oil = st[2] - st[0];
    let dp_w = st[2 * (nz - 1)] - st[2 * (nz - 2)];
    // Discrete equilibrium: Δp = ρ̄ g Δz with ρ̄ the mean of the two cells'
    // phase densities at their own pressures (compressible fluids, so not
    // the reference densities).
    let rho = |p: f64, w: bool| if w { 1000.0 * (1.0 + 4e-10 * (p - 1e5)) } else { 800.0 * (1.0 + 1e-9 * (p - 1e5)) };
    let want_o = 0.5 * (rho(st[0], false) + rho(st[2], false)) * 9.80665 * 5.0;
    let want_w = 0.5 * (rho(st[2 * (nz - 2)], true) + rho(st[2 * (nz - 1)], true)) * 9.80665 * 5.0;
    eprintln!(
        "FIM gravity: dp oil {dp_oil:.3} Pa (ρ̄gΔz {want_o:.3}; ref-density {:.1}), dp water {dp_w:.3} Pa (ρ̄gΔz {want_w:.3}; ref-density {:.1})",
        800.0 * 9.80665 * 5.0,
        1000.0 * 9.80665 * 5.0
    );
    assert!((dp_oil / want_o - 1.0).abs() < 1e-5, "oil {dp_oil} vs {want_o}");
    assert!((dp_w / want_w - 1.0).abs() < 1e-5, "water {dp_w} vs {want_w}");
    // Closed system: mass conserved.
    let m0: f64 = r.mass_in_place[0].iter().sum();
    let ev = run_fim(&m, &[], &ph, &state0, &[0.0], &opts).unwrap();
    let mi: f64 = ev.mass_in_place[0].iter().sum();
    assert!((m0 - mi).abs() < 1e-8 * mi, "{m0} vs {mi}");
}

fn pvt() -> Vec<PvtRow> {
    // Synthetic, smooth black-oil table (bubble point region up to 30 MPa).
    (0..=10)
        .map(|i| {
            let p = 1e5 + 3e6 * i as f64;
            let f = (p - 1e5) / 3e7;
            PvtRow { p, rs: 1.0 + 150.0 * f, bo: 1.05 + 0.35 * f, mu_o: 2.0e-3 - 1.2e-3 * f, bg: 1.0 / (1.0 + 300.0 * f), mu_g: 1.2e-5 + 1.3e-5 * f }
        })
        .collect()
}

fn black_oil() -> BlackOil {
    BlackOil {
        pvt: pvt(),
        rho_sc: [0.9, 850.0, 1020.0],
        bw: 1.01,
        c_w: 4e-10,
        mu_w: 5e-4,
        s_res: [0.05, 0.15, 0.2],
        n_exp: [2.0, 2.0, 2.0],
        kr_max: [0.8, 0.9, 0.4],
        c_o_undersat: 1.5e-9,
        c_rock: 5e-10,
        p_ref: 1e5,
    }
}

#[test]
fn fim_black_oil_depletion_liberates_gas_and_conserves_mass() {
    let n = 20;
    let g = RegularGrid::new([0.0; 3], [10.0, 10.0, 10.0], [n, 1, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n], &vec![0.2; n]).unwrap();
    let ph = black_oil();
    // Saturated oil at 25 MPa (Rs from table), Sw = 0.2: compute mass fractions.
    let p0 = 1e5 + 2.4e7; // a PVT node
    let r0 = ph.pvt.iter().find(|r| r.p >= p0).copied().unwrap();
    let rs_mass = r0.rs * 0.9 / 850.0;
    let so_m = 0.8 * (850.0 + r0.rs * 0.9) / r0.bo; // oil phase mass per unit PV
    let w_m = 0.2 * 1020.0 / 1.01 * (1.0 + 4e-10 * (p0 - 1e5));
    let zo = so_m / (1.0 + rs_mass) / (so_m + w_m) * 0.999;
    let zg = zo * rs_mass * 0.95; // slightly undersaturated
    let state0: Vec<f64> = (0..n).flat_map(|_| [p0, zg, zo]).collect();
    let s_init = ph.saturations(&state0[0..3]);
    assert!(s_init[0] < 1e-12 && (s_init[2] - 0.2).abs() < 0.01, "{s_init:?}");
    let wells = [well("P", WellKind::Producer, WellControl::Bhp(5e6), n - 1)];
    let opts = FimOptions {
        table_lo: vec![1e6, 0.0, 0.0],
        table_hi: vec![3e7, 0.1, 1.0],
        table_points: vec![59, 101, 101],
        injection: vec![vec![0.0, 0.0]],
        dt_init: 3600.0,
        dt_max: 5.0 * 86400.0,
        tol: 1e-9,
        ..FimOptions::default()
    };
    let t = [30.0 * 86400.0, 365.0 * 86400.0];
    let r = run_fim(&m, &wells, &ph, &state0, &t, &opts).unwrap();
    let init = run_fim(&m, &wells, &ph, &state0, &[0.0], &opts).unwrap().mass_in_place[0].clone();
    for k in 0..t.len() {
        for c in 0..3 {
            let lhs = r.mass_in_place[k][c] - init[c];
            let rhs = r.mass_net_in[k][c];
            // Discrete balance holds to the Newton tolerance (1e-9 per cell per step).
            let total: f64 = init.iter().sum();
            eprintln!("FIM black-oil t{k} comp {c}: Δmass {lhs:.3} kg vs net well {rhs:.3} kg");
            assert!((lhs - rhs).abs() < 1e-7 * total, "comp {c}: Δmass {lhs} vs net {rhs}");
        }
    }
    let sg_max = (0..n).map(|i| r.saturation[1][i * 3]).fold(0.0, f64::max);
    eprintln!("FIM black-oil: max Sg {sg_max:.3}, p near well {:.3e}, steps {}, cuts {}, table pts {}", r.state[1][(n - 1) * 3], r.n_steps, r.n_cuts, r.n_table_points);
    assert!(sg_max > 0.01, "free gas should appear below the bubble point");
    assert!(r.well_mass_rate[1][1] < 0.0);
}
