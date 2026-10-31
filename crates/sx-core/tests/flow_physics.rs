// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Validation of gravity, capillary pressure, compressibility (transient
//! pressure) and deviated wells against analytic or limiting cases.

use sx_core::flow::analytic::{buckley_leverett, theis_drawdown, thiem_drop};
use sx_core::flow::{
    peaceman_wi, run_two_phase, solve_pressure, Capillary, Corey, FixedPressure, FlowModel,
    FlowWell, Physics, TwoPhaseOptions, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

const K: f64 = 1e-13;
const MU: f64 = 1e-3;
const G: f64 = 9.80665;

fn column(nz: usize, dz: f64, k: f64) -> FlowModel {
    let g = RegularGrid::new([0.0, 0.0, 1000.0], [10.0, 10.0, dz], [1, 1, nz]).unwrap();
    FlowModel::from_regular(&g, &vec![[k; 3]; nz], &vec![0.2; nz]).unwrap()
}

#[test]
fn hydrostatic_equilibrium_water_and_oil() {
    let m = column(50, 2.0, K);
    let ph = Physics {
        gravity: G,
        rho_w: 1000.0,
        rho_o: 750.0,
        ..Physics::default()
    };
    let p_top = 1e7;
    for (sw, rho) in [(1.0, 1000.0), (0.0, 750.0)] {
        let corey = Corey {
            swc: 0.0,
            sor: 0.0,
            ..Corey::default()
        };
        let fixed = [FixedPressure {
            cell: 0,
            trans: m.boundary_trans(0, 2),
            pressure: p_top,
        }];
        let r = run_two_phase(
            &m,
            &[],
            &fixed,
            &corey,
            &ph,
            &vec![sw; 50],
            &vec![p_top; 50],
            &[1e5],
            &TwoPhaseOptions::default(),
        )
        .unwrap();
        let z0 = m.center[0][2];
        let worst = r.p[0]
            .iter()
            .zip(&m.center)
            .map(|(p, c)| (p - (p_top + rho * G * (c[2] - z0))).abs())
            .fold(0.0, f64::max);
        eprintln!("FLOW hydrostatic sw={sw}: max |p - p_hydro| = {worst:.2e} Pa");
        assert!(worst < 1e-6 * p_top, "{worst}");
        assert!(r.sw[0].iter().all(|&s| (s - sw).abs() < 1e-12));
    }
}

#[test]
fn gravity_segregation_reaches_segregated_state() {
    let nz = 20;
    let m = column(nz, 2.0, 1e-12);
    let corey = Corey {
        swc: 0.1,
        sor: 0.1,
        ..Corey::default()
    };
    let ph = Physics {
        gravity: G,
        rho_w: 1000.0,
        rho_o: 700.0,
        c_rock: 1e-9,
        ..Physics::default()
    };
    // Water on top of oil.
    let sw0: Vec<f64> = (0..nz)
        .map(|k| if k < nz / 2 { 0.9 } else { 0.1 })
        .collect();
    let p0: Vec<f64> = (0..nz).map(|k| 1e7 + 850.0 * G * 2.0 * k as f64).collect();
    let times = [3e6, 1e8];
    let r = run_two_phase(
        &m,
        &[],
        &[],
        &corey,
        &ph,
        &sw0,
        &p0,
        &times,
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    let w0: f64 = sw0.iter().zip(&m.pore_volume).map(|(s, v)| s * v).sum();
    for k in 0..times.len() {
        let rel = (r.water_in_place[k] - w0).abs() / w0;
        assert!(rel < 1e-3, "water balance {rel}");
    }
    let s = &r.sw[1];
    let top: f64 = s[..nz / 2].iter().sum::<f64>() / (nz / 2) as f64;
    let bot: f64 = s[nz / 2..].iter().sum::<f64>() / (nz / 2) as f64;
    eprintln!(
        "FLOW segregation: mean Sw top {top:.3}, bottom {bot:.3} after {} steps",
        r.n_steps
    );
    assert!(bot > 0.85 && top < 0.15, "top {top} bottom {bot}");
    // The early state is part way (water moving down).
    let mid_top: f64 = r.sw[0][..nz / 2].iter().sum::<f64>() / (nz / 2) as f64;
    assert!(mid_top < 0.9 - 1e-3);
    // At the end each phase is hydrostatic where mobile: oil pressure gradient
    // in the oil zone ≈ ρo g.
    let dp = (r.p[1][5] - r.p[1][2]) / 6.0;
    assert!(
        (dp / (700.0 * G) - 1.0).abs() < 0.02,
        "oil-zone gradient {dp}"
    );
}

#[test]
fn capillary_gravity_equilibrium_is_preserved() {
    let nz = 25;
    let m = column(nz, 2.0, 1e-13);
    let corey = Corey {
        swc: 0.15,
        sor: 0.15,
        ..Corey::default()
    };
    let cap = Capillary::VanGenuchten {
        alpha: 1e-4,
        n: 2.0,
    };
    let ph = Physics {
        gravity: G,
        rho_w: 1050.0,
        rho_o: 750.0,
        capillary: cap,
        c_rock: 1e-9,
        ..Physics::default()
    };
    let z_fwl = 1040.0;
    let sw0: Vec<f64> = m
        .center
        .iter()
        .map(|c| cap.sw_at(&corey, (ph.rho_w - ph.rho_o) * G * (z_fwl - c[2])))
        .collect();
    let p0: Vec<f64> = m
        .center
        .iter()
        .map(|c| 1e7 + ph.rho_o * G * (c[2] - 1000.0))
        .collect();
    let r = run_two_phase(
        &m,
        &[],
        &[],
        &corey,
        &ph,
        &sw0,
        &p0,
        &[1e7],
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    let drift = r.sw[0]
        .iter()
        .zip(&sw0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("FLOW capillary-gravity equilibrium: max |dSw| = {drift:.2e} over 1e7 s ({} steps); Sw top {:.3} bottom {:.3}", r.n_steps, sw0[0], sw0[nz - 1]);
    assert!(drift < 2e-3, "{drift}");
    assert!(sw0[0] < 0.5 && sw0[nz - 1] > 0.84);
    // Pc really matters here: the same start without Pc segregates.
    let ph0 = Physics {
        capillary: Capillary::None,
        ..ph
    };
    let r0 = run_two_phase(
        &m,
        &[],
        &[],
        &corey,
        &ph0,
        &sw0,
        &p0,
        &[1e7],
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    let drift0 = r0.sw[0]
        .iter()
        .zip(&sw0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(drift0 > 50.0 * drift, "no-Pc drift {drift0}");
}

#[test]
fn capillary_redistribution_to_uniform_saturation() {
    let n = 10;
    let g = RegularGrid::new([0.0; 3], [2.0, 1.0, 1.0], [n, 1, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[1e-12; 3]; n], &vec![0.2; n]).unwrap();
    let corey = Corey {
        swc: 0.1,
        sor: 0.1,
        ..Corey::default()
    };
    let cap = Capillary::BrooksCorey {
        pd: 1e4,
        lambda: 2.0,
    };
    let ph = Physics {
        capillary: cap,
        c_rock: 1e-9,
        ..Physics::default()
    };
    let sw0: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.6 } else { 0.3 }).collect();
    let r = run_two_phase(
        &m,
        &[],
        &[],
        &corey,
        &ph,
        &sw0,
        &vec![1e7; n],
        &[1e5, 1e8],
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    let mean0: f64 = sw0.iter().sum::<f64>() / n as f64;
    let mean: f64 = r.sw[1].iter().sum::<f64>() / n as f64;
    let spread = r.sw[1].iter().map(|s| (s - 0.45).abs()).fold(0.0, f64::max);
    eprintln!("FLOW capillary redistribution: final spread {spread:.2e}, mean {mean:.6} (start {mean0}) , {} steps", r.n_steps);
    assert!((mean - mean0).abs() < 1e-6);
    assert!(spread < 5e-3, "{spread}");
    // Imbibition goes from wet to dry: early on the left loses, the right gains.
    assert!(r.sw[0][n / 2 - 1] < 0.6 && r.sw[0][n / 2] > 0.3);
}

#[test]
fn transient_drawdown_matches_theis() {
    let n = 61;
    let (dx, h, phi, ct, q, rw) = (10.0, 10.0, 0.2, 1e-9, 1e-3, 0.1);
    let g = RegularGrid::new([0.0; 3], [dx, dx, h], [n, n, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n * n], &vec![phi; n * n]).unwrap();
    let corey = Corey {
        swc: 0.0,
        sor: 0.0,
        ..Corey::default()
    };
    let ph = Physics {
        c_rock: ct,
        ..Physics::default()
    };
    let c = n / 2;
    let well = FlowWell::vertical(
        "P",
        WellKind::Producer,
        WellControl::Rate(q),
        vec![c + n * c],
        rw,
        0.0,
    );
    let p_i = 2e7;
    let times = [3e3, 1e4];
    let opts = TwoPhaseOptions {
        max_dt: 100.0,
        ..TwoPhaseOptions::default()
    };
    let r = run_two_phase(
        &m,
        &[well],
        &[],
        &corey,
        &ph,
        &vec![1.0; n * n],
        &vec![p_i; n * n],
        &times,
        &opts,
    )
    .unwrap();
    for (k, &t) in times.iter().enumerate() {
        let mut worst: f64 = 0.0;
        for d in [5usize, 10, 15] {
            let want = theis_drawdown(q, MU, K, h, phi, ct, d as f64 * dx, t);
            let got = p_i - r.p[k][c + d + n * c];
            let scale = theis_drawdown(q, MU, K, h, phi, ct, 5.0 * dx, t);
            worst = worst.max((got - want).abs() / scale);
        }
        let bhp_want = p_i - theis_drawdown(q, MU, K, h, phi, ct, rw, t);
        let bhp_err = (r.bhp[k][0] - bhp_want).abs() / (p_i - bhp_want);
        eprintln!("FLOW Theis t={t}: max error at r=50,100,150 m {:.2}% of the r=50 m drawdown ; BHP error {:.2}%", 100.0 * worst, 100.0 * bhp_err);
        assert!(worst < 0.01, "t {t}: {worst}");
        assert!(bhp_err < 0.03, "t {t}: BHP {bhp_err}");
        // Closed system: expansion supplies exactly the production.
        let expansion: f64 = r.p[k]
            .iter()
            .zip(&m.pore_volume)
            .map(|(p, v)| v * ct * (p_i - p))
            .sum();
        assert!(
            (expansion - q * t).abs() < 1e-6 * q * t,
            "expansion {expansion} vs {}",
            q * t
        );
        assert!((r.water_rate[k][0] + q).abs() < 1e-9 * q);
    }
}

#[test]
fn two_phase_without_physics_matches_buckley_leverett() {
    let n = 200;
    let g = RegularGrid::new([0.0; 3], [1.0, 1.0, 1.0], [n, 1, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n], &vec![0.2; n]).unwrap();
    let corey = Corey {
        mu_o: 2e-3,
        ..Corey::default()
    };
    let q = 1e-5;
    let wells = [
        FlowWell::vertical(
            "I",
            WellKind::Injector,
            WellControl::Rate(q),
            vec![0],
            0.1,
            0.0,
        ),
        FlowWell::vertical(
            "P",
            WellKind::Producer,
            WellControl::Bhp(1e7),
            vec![n - 1],
            0.1,
            0.0,
        ),
    ];
    let pv: f64 = m.pore_volume.iter().sum();
    let td = 0.3;
    let r = run_two_phase(
        &m,
        &wells,
        &[],
        &corey,
        &Physics::default(),
        &vec![corey.swc; n],
        &vec![1e7; n],
        &[td * pv / q],
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    let l1: f64 = r.sw[0]
        .iter()
        .enumerate()
        .map(|(i, &s)| (s - buckley_leverett(&corey, (i as f64 + 0.5) / n as f64, td)).abs())
        .sum::<f64>()
        / n as f64;
    eprintln!("FLOW two-phase (physics off) BL td=0.3: L1 {l1:.4}");
    assert!(l1 < 0.015, "{l1}");
    assert!((r.water_net_in[0] - q * td * pv / q).abs() < 1e-6 * q * td * pv / q);
}

#[test]
fn vertical_trajectory_reproduces_vertical_peaceman() {
    let g = RegularGrid::new([0.0, 0.0, 1000.0], [20.0, 30.0, 4.0], [3, 3, 5]).unwrap();
    let perm: Vec<[f64; 3]> = (0..45)
        .map(|i| [K * (1.0 + i as f64 * 0.1), 2.0 * K, 0.1 * K])
        .collect();
    let m = FlowModel::from_regular(&g, &perm, &vec![0.2; 45]).unwrap();
    let comp = m
        .trajectory_completions(&[[30.0, 45.0, 1000.0], [30.0, 45.0, 1020.0]], 0.1, 1.0)
        .unwrap();
    assert_eq!(comp.len(), 5);
    for (k, cp) in comp.iter().enumerate() {
        let c = 4 + 9 * k;
        assert_eq!(cp.cell, c);
        let want = peaceman_wi(perm[c][0], perm[c][1], 20.0, 30.0, 4.0, 0.1, 1.0);
        assert!((cp.wi - want).abs() < 1e-9 * want, "{} vs {want}", cp.wi);
    }
    // A slanted path: projected lengths add up to the path's Δx and Δz.
    let comp = m
        .trajectory_completions(&[[1.0, 45.0, 1001.0], [59.0, 45.0, 1019.0]], 0.1, 0.0)
        .unwrap();
    let lx: f64 = comp.iter().map(|c| c.length[0]).sum();
    let lz: f64 = comp.iter().map(|c| c.length[2]).sum();
    assert!(
        (lx - 58.0).abs() < 1e-9 && (lz - 18.0).abs() < 1e-9,
        "{lx} {lz}"
    );
    assert!(comp.len() >= 5);
    assert!(m
        .trajectory_completions(&[[500.0, 0.0, 0.0], [600.0, 0.0, 0.0]], 0.1, 0.0)
        .is_err());
}

#[test]
fn horizontal_well_matches_rotated_radial_solution() {
    let (nx, n) = (10usize, 41usize);
    let (d, rw, q) = (10.0, 0.1, 1e-3);
    let g = RegularGrid::new([0.0; 3], [d, d, d], [nx, n, n]).unwrap();
    let ncell = nx * n * n;
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; ncell], &vec![0.2; ncell]).unwrap();
    let c = n / 2;
    let yc = (c as f64 + 0.5) * d;
    let comp = m
        .trajectory_completions(&[[0.0, yc, yc], [nx as f64 * d, yc, yc]], rw, 0.0)
        .unwrap();
    assert_eq!(comp.len(), nx);
    let len = nx as f64 * d;
    let ryz = |j: usize, k: usize| {
        (((j as f64 - c as f64) * d).powi(2) + ((k as f64 - c as f64) * d).powi(2)).sqrt()
    };
    let p_ref = 1e7;
    let mut fixed = Vec::new();
    for k in 0..n {
        for j in 0..n {
            if j == 0 || k == 0 || j == n - 1 || k == n - 1 {
                for i in 0..nx {
                    fixed.push(FixedPressure {
                        cell: i + nx * (j + n * k),
                        trans: 1e-3,
                        pressure: p_ref + thiem_drop(q, MU, K, len, ryz(j, k), rw),
                    });
                }
            }
        }
    }
    let well = FlowWell {
        wi: Some(comp.iter().map(|c| c.wi).collect()),
        ..FlowWell::vertical(
            "H",
            WellKind::Producer,
            WellControl::Rate(q),
            comp.iter().map(|c| c.cell).collect(),
            rw,
            0.0,
        )
    };
    let sol = solve_pressure(&m, &vec![1.0 / MU; ncell], 1.0 / MU, &[well], &fixed).unwrap();
    let span = thiem_drop(q, MU, K, len, c as f64 * d, rw);
    let err = (sol.wells[0].bhp - p_ref).abs() / span;
    eprintln!(
        "FLOW horizontal well: BHP error {:.3}% of drawdown",
        100.0 * err
    );
    assert!(err < 0.01, "{err}");
}

#[test]
fn deviated_well_with_gravity_column_head() {
    // A shut-in style check: one BHP producer in a water column with gravity;
    // flow per completion follows p_wb = BHP + ρ g (z − z_ref).
    let m = column(10, 5.0, K);
    let corey = Corey {
        swc: 0.0,
        sor: 0.0,
        ..Corey::default()
    };
    let ph = Physics {
        gravity: G,
        ..Physics::default()
    };
    let p0: Vec<f64> = m
        .center
        .iter()
        .map(|c| 1e7 + 1000.0 * G * (c[2] - m.center[0][2]))
        .collect();
    let fixed = [FixedPressure {
        cell: 0,
        trans: m.boundary_trans(0, 2),
        pressure: 1e7,
    }];
    // BHP at the top completion equal to the column pressure there: no flow anywhere.
    let comp = m
        .trajectory_completions(&[[5.0, 5.0, 1000.0], [5.0, 5.0, 1050.0]], 0.1, 0.0)
        .unwrap();
    let w = FlowWell {
        wi: Some(comp.iter().map(|c| c.wi).collect()),
        ref_depth_m: Some(m.center[0][2]),
        bhp_limit: None,
        crossflow: true,
        ..FlowWell::vertical(
            "W",
            WellKind::Producer,
            WellControl::Bhp(1e7),
            comp.iter().map(|c| c.cell).collect(),
            0.1,
            0.0,
        )
    };
    let r = run_two_phase(
        &m,
        &[w],
        &fixed,
        &corey,
        &ph,
        &[1.0; 10],
        &p0,
        &[1e4],
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    assert!(r.water_rate[0][0].abs() < 1e-12, "{}", r.water_rate[0][0]);
    let worst = r.p[0]
        .iter()
        .zip(&p0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-3, "{worst}");
}

#[test]
fn bad_physics_input_is_rejected() {
    let m = column(3, 1.0, K);
    let c = Corey::default();
    let bad = Physics {
        capillary: Capillary::VanGenuchten {
            alpha: 1e-4,
            n: 0.5,
        },
        c_rock: 1e-9,
        ..Physics::default()
    };
    assert!(run_two_phase(
        &m,
        &[],
        &[],
        &c,
        &bad,
        &[0.5; 3],
        &[1e7; 3],
        &[1.0],
        &TwoPhaseOptions::default()
    )
    .is_err());
    // Incompressible and closed: no pressure reference.
    assert!(run_two_phase(
        &m,
        &[],
        &[],
        &c,
        &Physics::default(),
        &[0.5; 3],
        &[1e7; 3],
        &[1.0],
        &TwoPhaseOptions::default()
    )
    .is_err());
    assert!(run_two_phase(
        &m,
        &[],
        &[],
        &c,
        &Physics {
            c_rock: 1e-9,
            ..Physics::default()
        },
        &[0.5; 2],
        &[1e7; 3],
        &[1.0],
        &TwoPhaseOptions::default()
    )
    .is_err());
}

#[test]
fn compressible_two_phase_conserves_water_and_oil_mass() {
    // Injector + BHP producer with gravity, Brooks-Corey Pc and rock/water/oil compressibility.
    let (nx, ny, nz) = (8usize, 8usize, 4usize);
    let n = nx * ny * nz;
    let g = RegularGrid::new([0.0, 0.0, 1000.0], [100.0, 100.0, 5.0], [nx, ny, nz]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K, K, 0.1 * K]; n], &vec![0.2; n]).unwrap();
    let corey = Corey {
        swc: 0.2,
        sor: 0.2,
        ..Corey::default()
    };
    let ph = Physics {
        gravity: G,
        rho_w: 1050.0,
        rho_o: 800.0,
        capillary: Capillary::BrooksCorey {
            pd: 5e3,
            lambda: 2.0,
        },
        c_rock: 5e-10,
        c_w: 4e-10,
        c_o: 1e-9,
        p_ref: 1e7,
    };
    let col = |i: usize, j: usize| (0..nz).map(|k| i + nx * (j + ny * k)).collect::<Vec<_>>();
    let wells = [
        FlowWell::vertical(
            "I",
            WellKind::Injector,
            WellControl::Rate(2e-3),
            col(1, 1),
            0.1,
            0.0,
        ),
        FlowWell::vertical(
            "P",
            WellKind::Producer,
            WellControl::Bhp(1.5e7),
            col(6, 6),
            0.1,
            0.0,
        ),
    ];
    let z: Vec<f64> = m.center.iter().map(|c| c[2]).collect();
    let p0: Vec<f64> = z.iter().map(|z| 2e7 + 800.0 * G * (z - 1000.0)).collect();
    let sw0 = vec![0.2; n];
    let yr = 3.156e7;
    let times = [0.5 * yr, 2.0 * yr, 5.0 * yr];
    let r = run_two_phase(
        &m,
        &wells,
        &[],
        &corey,
        &ph,
        &sw0,
        &p0,
        &times,
        &TwoPhaseOptions::default(),
    )
    .unwrap();
    let w0: f64 = (0..n)
        .map(|c| {
            let dp = p0[c] - ph.p_ref;
            m.pore_volume[c] * (1.0 + ph.c_rock * dp) * sw0[c] * (1.0 + ph.c_w * dp)
        })
        .sum();
    for (k, &t) in times.iter().enumerate() {
        let err = (r.water_in_place[k] - w0 - r.water_net_in[k]).abs() / r.water_net_in[k].abs();
        eprintln!(
            "FLOW mass balance t={:.1} yr: relative water error {err:.2e}, clamp loss {:.2e}",
            t / yr,
            r.water_clamp_loss[k]
        );
        assert!(err < 1e-9, "t {k}: {err}");
        assert_eq!(r.water_clamp_loss[k], 0.0);
        let o0: f64 = (0..n)
            .map(|c| {
                let dp = p0[c] - ph.p_ref;
                m.pore_volume[c] * (1.0 + ph.c_rock * dp) * (1.0 - sw0[c]) * (1.0 + ph.c_o * dp)
            })
            .sum();
        let oerr = (r.oil_in_place[k] - o0 - r.oil_net_in[k]).abs() / r.oil_net_in[k].abs();
        eprintln!(
            "FLOW oil balance t={:.1} yr: relative error {oerr:.2e}",
            times[k] / yr
        );
        assert!(oerr < 1e-9, "oil {k}: {oerr}");
        // Saturations stay consistent with the tracked masses: Σ PV(p)(1 − Sw) b_o(p) vs Σ Mo.
        let o_sat: f64 = (0..n)
            .map(|c| {
                let dp = r.p[k][c] - ph.p_ref;
                m.pore_volume[c] * (1.0 + ph.c_rock * dp) * (1.0 - r.sw[k][c]) * (1.0 + ph.c_o * dp)
            })
            .sum();
        let vmis = (o_sat - r.oil_in_place[k]).abs() / r.oil_in_place[k];
        eprintln!("FLOW volume misfit t={:.1} yr: {vmis:.2e}", times[k] / yr);
        assert!(vmis < 1e-4, "misfit {k}: {vmis}");
    }
    // The field depletes (producer outruns the injector), so storage terms are exercised.
    let dp = (0..n).map(|c| p0[c] - r.p[2][c]).sum::<f64>() / n as f64;
    eprintln!("FLOW mass balance: mean pressure drop {dp:.3e} Pa");
    assert!(dp > 1e5, "mean drop {dp}");
}

#[test]
fn pressure_change_control_adapts_theis_steps() {
    let n = 61;
    let (dx, h, phi, ct, q, rw) = (10.0, 10.0, 0.2, 1e-9, 1e-3, 0.1);
    let g = RegularGrid::new([0.0; 3], [dx, dx, h], [n, n, 1]).unwrap();
    let m = FlowModel::from_regular(&g, &vec![[K; 3]; n * n], &vec![phi; n * n]).unwrap();
    let corey = Corey { swc: 0.0, sor: 0.0, ..Corey::default() };
    let ph = Physics { c_rock: ct, ..Physics::default() };
    let c = n / 2;
    let well = FlowWell::vertical("P", WellKind::Producer, WellControl::Rate(q), vec![c + n * c], rw, 0.0);
    let p_i = 2e7;
    let times = [1e4];
    let run = |opts: TwoPhaseOptions| {
        run_two_phase(&m, std::slice::from_ref(&well), &[], &corey, &ph, &vec![1.0; n * n], &vec![p_i; n * n], &times, &opts)
            .unwrap()
    };
    let r = run(TwoPhaseOptions { max_dt: 1e4, dp_target: 1e4, ..TwoPhaseOptions::default() });
    let want = theis_drawdown(q, MU, K, h, phi, ct, 10.0 * dx, 1e4);
    let got = p_i - r.p[0][c + 10 + n * c];
    eprintln!("FLOW dp-control Theis: {} steps, error {:.2}%", r.n_steps, 100.0 * (got / want - 1.0));
    assert!((got / want - 1.0).abs() < 0.02);
    assert!(r.n_steps > 3 && r.n_steps < 100, "{}", r.n_steps);
    let expansion: f64 = r.p[0].iter().zip(&m.pore_volume).map(|(p, v)| v * ct * (p_i - p)).sum();
    assert!((expansion - q * 1e4).abs() < 1e-6 * q * 1e4);
    // Without control the single step is far less accurate.
    let r1 = run(TwoPhaseOptions { max_dt: 1e4, ..TwoPhaseOptions::default() });
    let got1 = p_i - r1.p[0][c + 10 + n * c];
    assert!(r1.n_steps < r.n_steps && (got1 / want - 1.0).abs() > (got / want - 1.0).abs());
    let bad = TwoPhaseOptions { ds_target: 0.0, ..TwoPhaseOptions::default() };
    assert!(run_two_phase(&m, &[], &[], &corey, &ph, &vec![1.0; n * n], &vec![p_i; n * n], &times, &bad).is_err());
}
