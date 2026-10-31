// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Flow validation against analytic solutions.

use sx_core::flow::analytic::{buckley_leverett, thiem_drop};
use sx_core::flow::benchmarks::compare_spe10;
use sx_core::flow::implicit::run_fully_implicit;
use sx_core::flow::{
    peaceman_wi, run_impes, run_tracer, solve_pressure, Corey, FixedPressure, FlowModel, FlowWell,
    ImpesOptions, WellControl, WellKind,
};
use sx_core::grid::{CornerPointGrid, RegularGrid};

const K: f64 = 1e-13;
const MU: f64 = 1e-3;

fn line(n: usize, dx: f64, a: f64) -> (RegularGrid, Vec<[f64; 3]>, Vec<f64>) {
    let side = a.sqrt();
    let g = RegularGrid::new([0.0; 3], [dx, side, side], [n, 1, 1]).unwrap();
    (g, vec![[K; 3]; n], vec![0.2; n])
}

#[test]
fn linear_1d_matches_darcy() {
    let (g, perm, poro) = line(100, 10.0, 100.0);
    for m in [
        FlowModel::from_regular(&g, &perm, &poro).unwrap(),
        FlowModel::from_corner_point(&CornerPointGrid::from_regular(&g), &perm, &poro).unwrap(),
    ] {
        let (p0, p1) = (2e7, 1e7);
        let fixed = [
            FixedPressure {
                cell: 0,
                trans: m.boundary_trans(0, 0),
                pressure: p0,
            },
            FixedPressure {
                cell: 99,
                trans: m.boundary_trans(99, 0),
                pressure: p1,
            },
        ];
        let sol = solve_pressure(&m, &vec![1.0 / MU; 100], 1.0 / MU, &[], &fixed).unwrap();
        let q = K * 100.0 * (p0 - p1) / (MU * 1000.0);
        for (i, &p) in sol.p.iter().enumerate() {
            let x = (i as f64 + 0.5) * 10.0;
            assert!(
                (p - (p0 - (p0 - p1) * x / 1000.0)).abs() < 1e-6 * p0,
                "cell {i}"
            );
        }
        for f in &sol.flux {
            assert!((f - q).abs() < 1e-9 * q);
        }
        assert!(
            (sol.boundary_flux[0] - q).abs() < 1e-9 * q
                && (sol.boundary_flux[1] + q).abs() < 1e-9 * q
        );
    }
}

#[test]
fn radial_thiem_and_peaceman() {
    let n = 61;
    let (dx, h, rw, q) = (10.0, 10.0, 0.1, 1e-3);
    let g = RegularGrid::new([0.0; 3], [dx, dx, h], [n, n, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n * n], &vec![0.2; n * n]).unwrap();
    let c = n / 2;
    let ctr = |i: usize, j: usize| -> f64 {
        (((i as f64 - c as f64) * dx).powi(2) + ((j as f64 - c as f64) * dx).powi(2)).sqrt()
    };
    let p_ref = 1e7;
    let mut fixed = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if i == 0 || j == 0 || i == n - 1 || j == n - 1 {
                fixed.push(FixedPressure {
                    cell: i + n * j,
                    trans: 1e-3,
                    pressure: p_ref + thiem_drop(q, MU, K, h, ctr(i, j), rw),
                });
            }
        }
    }
    let well = FlowWell {
        name: "P1".into(),
        kind: WellKind::Producer,
        control: WellControl::Rate(q),
        cells: vec![c + n * c],
        radius_m: rw,
        skin: 0.0,
        wi: None,
        ref_depth_m: None,
        bhp_limit: None,
        crossflow: true,
    };
    let sol = solve_pressure(&m, &vec![1.0 / MU; n * n], 1.0 / MU, &[well], &fixed).unwrap();
    let w = &sol.wells[0];
    let span = thiem_drop(q, MU, K, h, ctr(0, c), rw);
    // Peaceman BHP vs the analytic wellbore pressure.
    let bhp_err = (w.bhp - p_ref).abs() / span;
    assert!(bhp_err < 0.01, "BHP error {bhp_err:.4} of the drawdown");
    assert!((w.rate + q).abs() < 1e-9 * q);
    // Cell pressures away from the well vs Thiem.
    let mut worst: f64 = 0.0;
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let r = ctr(i, j);
            if r >= 3.0 * dx {
                let want = p_ref + thiem_drop(q, MU, K, h, r, rw);
                worst = worst.max((sol.p[i + n * j] - want).abs() / span);
            }
        }
    }
    assert!(worst < 0.01, "worst Thiem error {worst:.4} of the drawdown");
    eprintln!(
        "FLOW radial: Peaceman BHP error {:.3}% , Thiem max error {:.3}% of drawdown",
        100.0 * bhp_err,
        100.0 * worst
    );
    // Isotropic WI reduces to 2πkh / ln(0.198 dx / rw).
    let wi = peaceman_wi(K, K, dx, dx, h, rw, 0.0);
    let want = 2.0 * std::f64::consts::PI * K * h
        / ((0.28 / 2f64.sqrt() * (2.0f64).sqrt() * dx / 2f64.sqrt() / rw).ln());
    assert!((wi - want).abs() < 1e-12 * want);
}

fn line_wells(n: usize, q: f64) -> Vec<FlowWell> {
    vec![
        FlowWell {
            name: "I".into(),
            kind: WellKind::Injector,
            control: WellControl::Rate(q),
            cells: vec![0],
            radius_m: 0.1,
            skin: 0.0,
            wi: None,
            ref_depth_m: None,
            bhp_limit: None,
            crossflow: true,
        },
        FlowWell {
            name: "P".into(),
            kind: WellKind::Producer,
            control: WellControl::Bhp(1e7),
            cells: vec![n - 1],
            radius_m: 0.1,
            skin: 0.0,
            wi: None,
            ref_depth_m: None,
            bhp_limit: None,
            crossflow: true,
        },
    ]
}

#[test]
fn tracer_front_and_mass_balance() {
    let n = 400;
    let (g, perm, poro) = line(n, 1.0, 1.0);
    let m = FlowModel::from_regular(&g, &perm, &poro).unwrap();
    let q = 1e-5;
    let wells = line_wells(n, q);
    let sol = solve_pressure(&m, &vec![1.0 / MU; n], 1.0 / MU, &wells, &[]).unwrap();
    let pv: f64 = m.pore_volume.iter().sum();
    let t = 0.5 * pv / q;
    let r = run_tracer(
        &m,
        &sol,
        &wells,
        &[1.0, 0.0],
        &[],
        &[],
        &vec![0.0; n],
        &[t],
        0.9,
    )
    .unwrap();
    let c = &r.conc[0];
    let mass: f64 = c.iter().zip(&m.pore_volume).map(|(a, b)| a * b).sum();
    assert!(
        (mass - q * t).abs() < 1e-9 * q * t,
        "mass {mass} vs {}",
        q * t
    );
    let front = c.iter().position(|&v| v < 0.5).unwrap() as f64 / n as f64;
    assert!((front - 0.5).abs() < 0.02, "front at {front}");
}

#[test]
fn buckley_leverett_profile() {
    let n = 400;
    let (g, perm, poro) = line(n, 1.0, 1.0);
    let m = FlowModel::from_regular(&g, &perm, &poro).unwrap();
    let q = 1e-5;
    let corey = Corey {
        mu_o: 2e-3,
        ..Corey::default()
    };
    let pv: f64 = m.pore_volume.iter().sum();
    let tds = [0.2, 0.4];
    let times: Vec<f64> = tds.iter().map(|td| td * pv / q).collect();
    let r = run_impes(
        &m,
        &line_wells(n, q),
        &[],
        &corey,
        &vec![corey.swc; n],
        &times,
        &ImpesOptions {
            cfl: 0.5,
            pressure_dt: times[0] / 10.0,
            boundary_fw: 1.0,
        },
    )
    .unwrap();
    for (k, &td) in tds.iter().enumerate() {
        let l1: f64 = r.sw[k]
            .iter()
            .enumerate()
            .map(|(i, &s)| (s - buckley_leverett(&corey, (i as f64 + 0.5) / n as f64, td)).abs())
            .sum::<f64>()
            / n as f64;
        eprintln!("FLOW BL td={td}: mean |Sw - analytic| = {l1:.4}");
        assert!(l1 < 0.01, "td {td}: L1 {l1}");
        // Water injected = water gained (no breakthrough yet).
        let gained: f64 = r.sw[k]
            .iter()
            .zip(&m.pore_volume)
            .map(|(s, v)| (s - corey.swc) * v)
            .sum();
        assert!((gained - q * times[k]).abs() < 1e-6 * q * times[k]);
        assert!(r.water_rate[k][0] > 0.0 && r.oil_rate[k][1] < 0.0);
    }
}

#[test]
fn unimplemented_stubs_say_so() {
    assert!(run_fully_implicit()
        .unwrap_err()
        .to_string()
        .contains("not implemented (stub)"));
    assert!(compare_spe10()
        .unwrap_err()
        .to_string()
        .contains("not implemented (stub)"));
}

#[test]
fn needs_a_pressure_reference() {
    let (g, perm, poro) = line(5, 1.0, 1.0);
    let m = FlowModel::from_regular(&g, &perm, &poro).unwrap();
    let w = FlowWell {
        name: "I".into(),
        kind: WellKind::Injector,
        control: WellControl::Rate(1.0),
        cells: vec![0],
        radius_m: 0.1,
        skin: 0.0,
        wi: None,
        ref_depth_m: None,
        bhp_limit: None,
        crossflow: true,
    };
    assert!(solve_pressure(&m, &[1.0; 5], 1.0, &[w], &[]).is_err());
}

#[test]
fn corner_point_varying_thickness_conserves_flux() {
    // One row of 20 cells, top flat, thickness 5 m .. 15 m along x.
    let nx = 20;
    let xs: Vec<f64> = (0..=nx).map(|i| 10.0 * i as f64).collect();
    let ys = [0.0, 10.0];
    let top = vec![1000.0; 2 * (nx + 1)];
    let base: Vec<f64> = (0..2)
        .flat_map(|_| (0..=nx).map(|i| 1005.0 + 10.0 * i as f64 / nx as f64))
        .collect();
    let g = CornerPointGrid::from_surfaces(&xs, &ys, &[top, base]).unwrap();
    let m = FlowModel::from_corner_point(&g, &vec![[K; 3]; nx], &vec![0.2; nx]).unwrap();
    assert_eq!(m.connections.len(), nx - 1);
    let fixed = [
        FixedPressure {
            cell: 0,
            trans: m.boundary_trans(0, 0),
            pressure: 2e7,
        },
        FixedPressure {
            cell: nx - 1,
            trans: m.boundary_trans(nx - 1, 0),
            pressure: 1e7,
        },
    ];
    let sol = solve_pressure(&m, &vec![1.0 / MU; nx], 1.0 / MU, &[], &fixed).unwrap();
    let f0 = sol.flux[0];
    assert!(sol.flux.iter().all(|f| (f - f0).abs() < 1e-9 * f0));
    assert!((sol.boundary_flux[0] + sol.boundary_flux[1]).abs() < 1e-9 * f0);
    // Thicker cells need a smaller gradient: pressure steps shrink along x.
    let d: Vec<f64> = sol.p.windows(2).map(|w| w[0] - w[1]).collect();
    assert!(d.windows(2).all(|w| w[1] < w[0]));
}
