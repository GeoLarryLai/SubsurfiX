// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Checks of the finite-element water + heat solver (`flow::fem`).
//!
//! Fast: steady 1D flow (linear pressure, Darcy slope), 1D conduction against
//! the erfc solution, 1D advection–conduction against Ogata–Banks, mass and
//! energy balance.
//! Heavy (`#[ignore]`, `cargo test -p sx-core --test fem_ht -- --ignored --nocapture`):
//! FEM against FVM on a 10 × 10 × 4 block doublet.

use sx_core::flow::analytic::{conduction_step, ogata_banks};
use sx_core::flow::ogs::regular_corners;
use sx_core::flow::{
    run_fem_ht, run_fim, water, FemOptions, FemResult, FimOptions, FlowModel, FlowWell, RockThermal, ThermalWater, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

const YR: f64 = 3.15576e7;
const ROCK: RockThermal = RockThermal { rho: 2650.0, cp: 900.0, conductivity: 2.5 };

/// A line of `nx` cells of `dx` × 1 × 1 m.
fn line(nx: usize, dx: f64, perm: f64, poro: f64) -> (RegularGrid, FlowModel) {
    let g = RegularGrid { origin: [0.0, 0.0, 1000.0], cell: [dx, 1.0, 1.0], dims: [nx, 1, 1] };
    let m = FlowModel::from_regular(&g, &vec![[perm; 3]; nx], &vec![poro; nx]).unwrap();
    (g, m)
}

/// Relative water and energy balance errors between the first and last report.
fn balance(m: &[Vec<f64>], net: &[Vec<f64>]) -> [f64; 2] {
    let (a, b) = (m.len() - 1, net.len() - 1);
    std::array::from_fn(|c| ((m[a][c] - m[0][c]) - (net[b][c] - net[0][c])).abs() / (net[b][c] - net[0][c]).abs().max(1e-300))
}

/// Volumetric heat capacity (J/(m³ K)) and conductivity (W/(m K)) of rock
/// with water at (p, T).
fn bulk_thermal(p: f64, t: f64, poro: f64) -> (f64, f64, f64) {
    let w = water::liquid(p, t).unwrap();
    let rho_c = poro * w.density * w.cp + (1.0 - poro) * ROCK.rho * ROCK.cp;
    let kappa = poro * w.conductivity + (1.0 - poro) * ROCK.conductivity;
    (rho_c, kappa, w.density * w.cp)
}

#[test]
fn steady_flow_is_linear_with_darcy_slope() {
    // Isothermal steady flow between a rate injector (cell 0) and a BHP producer (last cell):
    // between the well cells the pressure is linear with slope −q μ / (k A).
    let (nx, perm, p, t, q) = (50, 1e-12, 1e7, 320.0, 1e-6);
    let (g, m) = line(nx, 1.0, perm, 0.2);
    let wells = [
        FlowWell::vertical("INJ", WellKind::Injector, WellControl::Rate(q), vec![0], 0.1, 0.0),
        FlowWell::vertical("PROD", WellKind::Producer, WellControl::Bhp(p), vec![nx - 1], 0.1, 0.0),
    ];
    let opts = FemOptions { injection_temperature: vec![t, t], ..FemOptions::default() };
    let phys = ThermalWater::new(ROCK, 0.0, 1e5);
    let r = run_fem_ht(&regular_corners(&g), &m, &wells, &phys, &vec![p; nx], &vec![t; nx], &[2e5], &opts).unwrap();
    let pts: Vec<(f64, f64)> = r.nodes.iter().zip(&r.p_nodes[0]).filter(|(x, _)| x[0] >= 5.0 && x[0] <= 45.0).map(|(x, &v)| (x[0], v)).collect();
    let k = pts.len() as f64;
    let (mx, mp) = (pts.iter().map(|v| v.0).sum::<f64>() / k, pts.iter().map(|v| v.1).sum::<f64>() / k);
    let slope = pts.iter().map(|v| (v.0 - mx) * (v.1 - mp)).sum::<f64>() / pts.iter().map(|v| (v.0 - mx).powi(2)).sum::<f64>();
    let off = pts.iter().map(|v| (v.1 - mp - slope * (v.0 - mx)).abs()).fold(0.0, f64::max);
    let darcy = -q * water::liquid(p, t).unwrap().viscosity / perm;
    assert!(off < 1e-4 * (slope * 40.0).abs(), "pressure departs from a line by {off} Pa");
    assert!((slope / darcy - 1.0).abs() < 1e-3, "slope {slope} Pa/m vs Darcy {darcy} Pa/m");
}

#[test]
fn conduction_matches_erfc() {
    // Hot half against cold half of a closed bar: T = T_c + ΔT/2 erfc(x / 2√(αt)) about the middle.
    let (nx, poro, p) = (60, 0.01, 1e7);
    let (tc, th) = (340.0, 350.0);
    let (g, m) = line(nx, 1.0, 1e-18, poro);
    let t0: Vec<f64> = (0..nx).map(|i| if i < nx / 2 { th } else { tc }).collect();
    let t_end = 2.5e7;
    let opts = FemOptions { dt_init: 1e4, dt_max: 2.5e5, ..FemOptions::default() };
    let phys = ThermalWater::new(ROCK, 0.0, 1e5);
    let r = run_fem_ht(&regular_corners(&g), &m, &[], &phys, &vec![p; nx], &t0, &[t_end], &opts).unwrap();
    let (rho_c, kappa, _) = bulk_thermal(p, 0.5 * (tc + th), poro);
    let alpha = kappa / rho_c;
    let mut worst = 0.0f64;
    for (a, x) in r.nodes.iter().enumerate() {
        let s = x[0] - 30.0;
        let exact = tc + (th - tc) * if s >= 0.0 { 0.5 * conduction_step(s, t_end, alpha) } else { 1.0 - 0.5 * conduction_step(-s, t_end, alpha) };
        worst = worst.max((r.temperature_nodes[0][a] - exact).abs());
    }
    assert!(worst < 0.02 * (th - tc), "max |T - erfc| = {worst} K");
}

/// 1D injection of cold water at a fixed rate into a warm line, BHP producer at the far end.
fn injection_line(stabilization: f64) -> (FemResult, f64, f64, f64) {
    let (nx, dx, poro, p) = (100, 1.0, 0.2, 1e7);
    let (t_inj, t_res, u) = (300.0, 350.0, 1e-6);
    let (g, m) = line(nx, dx, 1e-12, poro);
    let wells = [
        FlowWell::vertical("INJ", WellKind::Injector, WellControl::Rate(u), vec![0], 0.1, 0.0),
        FlowWell::vertical("PROD", WellKind::Producer, WellControl::Bhp(p), vec![nx - 1], 0.1, 0.0),
    ];
    let t_end = 2.6e7;
    let opts = FemOptions {
        injection_temperature: vec![t_inj, t_res],
        dt_init: 1e4,
        dt_max: 2e5,
        stabilization,
        ..FemOptions::default()
    };
    let phys = ThermalWater::new(ROCK, 0.0, 1e5);
    let r = run_fem_ht(&regular_corners(&g), &m, &wells, &phys, &vec![p; nx], &vec![t_res; nx], &[0.0, t_end], &opts).unwrap();
    let (rho_c, kappa, rho_c_w) = bulk_thermal(p, 0.5 * (t_inj + t_res), poro);
    let v = u * rho_c_w / rho_c;
    (r, v, kappa / rho_c, t_end)
}

#[test]
fn injection_front_matches_ogata_banks_with_upwind_dispersion() {
    let (r, v, d, t_end) = injection_line(1.0);
    let (t_inj, t_res) = (300.0, 350.0);
    // s = 1 adds the first-order upwind dispersion v·dx/2; the source sits at the first cell centre.
    let d_eff = d + 0.5 * v * 1.0;
    let mut worst = 0.0f64;
    for (a, x) in r.nodes.iter().enumerate().filter(|(_, x)| x[0] >= 10.0 && x[0] <= 90.0) {
        let num = (r.temperature_nodes[1][a] - t_res) / (t_inj - t_res);
        worst = worst.max((num - ogata_banks(x[0] - 0.5, t_end, v, d_eff)).abs());
    }
    assert!(worst < 0.08, "max |T_norm - Ogata-Banks| = {worst}");
    let [bw, be] = balance(&r.mass_in_place, &r.mass_net_in);
    assert!(bw < 1e-6 && be < 1e-6, "balance error water {bw:.2e}, energy {be:.2e}");
}

#[test]
fn stabilized_front_has_no_undershoot() {
    let (r, _, _, _) = injection_line(1.0);
    let lo = r.temperature_nodes[1].iter().copied().fold(f64::INFINITY, f64::min);
    assert!(lo > 300.0 - 0.05, "min T = {lo} K with s = 1");
}

#[test]
#[ignore = "heavy: FEM vs FVM block doublet; run with --ignored"]
fn block_doublet_matches_fim() {
    let (nx, ny, nz) = (10, 10, 4);
    let n = nx * ny * nz;
    let g = RegularGrid { origin: [0.0, 0.0, 2000.0], cell: [500.0, 500.0, 5.0], dims: [nx, ny, nz] };
    let md = 9.869_233e-16;
    let k_layer = [100.0, 300.0, 50.0, 200.0];
    let perm: Vec<[f64; 3]> = (0..n).map(|c| { let k = k_layer[c / (nx * ny)] * md; [k, k, 0.1 * k] }).collect();
    let model = FlowModel::from_regular(&g, &perm, &vec![0.2; n]).unwrap();
    let col = |i: usize, j: usize| (0..nz).map(|k| i + nx * (j + ny * k)).collect::<Vec<_>>();
    let (bhp, t_res, t_inj) = (2e7, 353.15, 303.15);
    let wells = [
        FlowWell::vertical("INJ", WellKind::Injector, WellControl::Rate(0.02), col(5, 5), 0.1, 0.0),
        FlowWell::vertical("PROD", WellKind::Producer, WellControl::Bhp(bhp), col(2, 2), 0.1, 0.0),
    ];
    let times: Vec<f64> = (1..=30).map(|y| y as f64 * YR).collect();
    let phys = ThermalWater::new(ROCK, 0.0, 1e5);

    let fem_opts = FemOptions { injection_temperature: vec![t_inj, t_res], dt_init: 86_400.0, dt_max: 0.05 * YR, ..FemOptions::default() };
    let t = std::time::Instant::now();
    let fem = run_fem_ht(&regular_corners(&g), &model, &wells, &phys, &vec![bhp; n], &vec![t_res; n], &times, &fem_opts).unwrap();
    let t_fem = t.elapsed().as_secs_f64();

    let fim_opts = FimOptions {
        table_lo: vec![5e6, 290.0],
        table_hi: vec![5e7, 360.0],
        table_points: vec![41, 71],
        injection: vec![vec![t_inj], vec![t_res]],
        dt_init: 86_400.0,
        dt_max: 0.05 * YR,
        ..FimOptions::default()
    };
    let state0: Vec<f64> = (0..n).flat_map(|_| [bhp, t_res]).collect();
    let t = std::time::Instant::now();
    let fim = run_fim(&model, &wells, &ThermalWater::new(ROCK, 0.0, 1e5), &state0, &times, &fim_opts).unwrap();
    let t_fim = t.elapsed().as_secs_f64();

    let prod = col(2, 2);
    let mean = |v: &dyn Fn(usize) -> f64| prod.iter().map(|&c| v(c)).sum::<f64>() / prod.len() as f64;
    let tp_fem: Vec<f64> = (0..times.len()).map(|k| mean(&|c| fem.temperature[k][c])).collect();
    let tp_fim: Vec<f64> = (0..times.len()).map(|k| mean(&|c| fim.state[k][2 * c + 1])).collect();
    let worst = tp_fem.iter().zip(&tp_fim).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    for k in (4..times.len()).step_by(5) {
        println!("year {:2}: producer T FEM {:.2} K, FVM {:.2} K", k + 1, tp_fem[k], tp_fim[k]);
    }
    // Thermal breakthrough: first report year with the producer 1 K below the reservoir.
    let breakthrough = |tp: &[f64]| tp.iter().position(|&v| v < t_res - 1.0).map(|k| k + 1);
    let (bt_fem, bt_fim) = (breakthrough(&tp_fem), breakthrough(&tp_fim));
    let [fw, fe] = balance(&fem.mass_in_place, &fem.mass_net_in);
    println!("run time FEM {t_fem:.1} s, FVM {t_fim:.1} s; FEM steps {} cuts {} Picard {}; max producer |dT| {worst:.3} K; breakthrough year FEM {bt_fem:?}, FVM {bt_fim:?}; FEM balance {fw:.1e} / {fe:.1e}", fem.n_steps, fem.n_cuts, fem.n_iterations);
    assert!(fw < 1e-6 && fe < 1e-6, "FEM balance error water {fw:.2e}, energy {fe:.2e}");
    assert!(worst < 5.0, "producer temperature FEM vs FVM differs by {worst:.2} K");
    if let (Some(a), Some(b)) = (bt_fem, bt_fim) {
        assert!(a.abs_diff(b) <= 3, "breakthrough year FEM {a} vs FVM {b}");
    } else {
        assert_eq!(bt_fem, bt_fim, "breakthrough in one solver only");
    }
}
