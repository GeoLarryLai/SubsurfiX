// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Element/interface assembly tests, NOT validation of a complete multiphase FEM solver.
use sx_core::flow::fem::{interface::ResistiveInterface, multiphase::ComponentElement};
use sx_core::flow::{
    BlackOil, Capillary, Co2Brine, DeadOil, OblPhysics, PvtRow, RockThermal, ThermalWater,
};

fn box_corners() -> [[f64; 3]; 8] {
    std::array::from_fn(|a| {
        [
            (a & 1) as f64 * 10.0,
            ((a >> 1) & 1) as f64 * 20.0,
            1000.0 + ((a >> 2) & 1) as f64 * 5.0,
        ]
    })
}
fn element() -> ComponentElement {
    ComponentElement::new(&box_corners(), [1e-12, 2e-12, 0.5e-12], 0.2).unwrap()
}
fn black_oil() -> BlackOil {
    BlackOil {
        pvt: vec![
            PvtRow {
                p: 1e5,
                rs: 1.0,
                bo: 1.05,
                mu_o: 2e-3,
                bg: 1.0,
                mu_g: 1.2e-5,
            },
            PvtRow {
                p: 3e7,
                rs: 151.0,
                bo: 1.4,
                mu_o: 0.8e-3,
                bg: 1.0 / 301.0,
                mu_g: 2.5e-5,
            },
        ],
        rho_sc: [0.9, 850.0, 1020.0],
        bw: 1.01,
        c_w: 4e-10,
        mu_w: 5e-4,
        s_res: [0.05, 0.15, 0.2],
        n_exp: [2.0; 3],
        kr_max: [0.8, 0.9, 0.4],
        c_o_undersat: 1.5e-9,
        c_rock: 5e-10,
        p_ref: 1e5,
    }
}
fn check_constant(physics: &dyn OblPhysics, state: &[f64]) {
    let s = state.repeat(8);
    let c = physics.n_vars();
    let b = element()
        .residual(physics, &s, &s, 100.0, 0.0, &vec![0.0; c])
        .unwrap();
    assert!(
        b.residual.iter().all(|r| r.abs() < 1e-12),
        "{:?}",
        b.residual
    );
    assert_eq!(b.mass_before, b.mass_after);
    let mut ops = vec![0.0; physics.layout().n_ops()];
    physics.evaluate(state, &mut ops);
    for k in 0..c {
        assert!((b.mass_after[k] - 200.0 * ops[k]).abs() < 1e-10 * b.mass_after[k].abs().max(1.0));
    }
}
fn check_component_sum(physics: &dyn OblPhysics, state: &[f64]) {
    let c = physics.n_vars();
    let old = state.repeat(8);
    let mut new = old.clone();
    for a in 0..8 {
        new[a * c] += 20000.0 * (a as f64 - 3.0);
        new[a * c + 1] += 0.0002 * a as f64;
    }
    let source: Vec<f64> = (0..c).map(|k| 0.02 * (k + 1) as f64).collect();
    let dt = 123.0;
    let b = element()
        .residual(physics, &old, &new, dt, 9.81, &source)
        .unwrap();
    for k in 0..c {
        let r: f64 = (0..8).map(|a| b.residual[a * c + k]).sum();
        let expected = b.mass_after[k] - b.mass_before[k] - dt * 1000.0 * source[k];
        assert!(
            (r - expected).abs() < 1e-9 * b.mass_after[k].abs().max(1.0),
            "component {k}: {r} versus {expected}"
        );
    }
}
#[test]
fn water_oil_constant_and_component_sum() {
    let p = DeadOil::default();
    check_constant(&p, &[1e7, 0.4]);
    check_component_sum(&p, &[1e7, 0.4]);
}
#[test]
fn black_oil_constant_and_component_sum() {
    let p = black_oil();
    check_constant(&p, &[1e7, 0.12, 0.65]);
    check_component_sum(&p, &[1e7, 0.12, 0.65]);
}
#[test]
fn co2_brine_constant_and_component_sum() {
    let p = Co2Brine::default();
    check_constant(&p, &[1e7, 0.2]);
    check_component_sum(&p, &[1e7, 0.2]);
    assert_eq!(p.n_fallbacks(), 0);
}
#[test]
fn capillary_gradient_is_not_silently_dropped() {
    let plain = DeadOil::default();
    let mut cap = plain.clone();
    cap.capillary = Capillary::BrooksCorey {
        pd: 2e4,
        lambda: 2.0,
    };
    let s: Vec<f64> = (0..8)
        .flat_map(|a| [1e7, 0.4 + 0.02 * (a & 1) as f64])
        .collect();
    let a = element()
        .residual(&plain, &s, &s, 100.0, 0.0, &[0.0; 2])
        .unwrap();
    let b = element()
        .residual(&cap, &s, &s, 100.0, 0.0, &[0.0; 2])
        .unwrap();
    assert!(a.residual.iter().all(|v| v.abs() < 1e-12));
    assert!(b.residual.iter().any(|v| v.abs() > 1e-6));
    for c in 0..2 {
        assert!((0..8).map(|a| b.residual[2 * a + c]).sum::<f64>().abs() < 1e-8);
    }
}
#[test]
fn hydrostatic_field_has_zero_residual_with_depth_positive_down() {
    let p = DeadOil {
        rho_o: 1000.0,
        ..DeadOil::default()
    };
    let s: Vec<f64> = box_corners()
        .iter()
        .flat_map(|x| [1e7 + 1000.0 * 9.81 * x[2], 0.5])
        .collect();
    let b = element()
        .residual(&p, &s, &s, 100.0, 9.81, &[0.0; 2])
        .unwrap();
    assert!(
        b.residual.iter().all(|r| r.abs() < 1e-8),
        "{:?}",
        b.residual
    );
}
#[test]
fn anisotropic_darcy_patch_has_correct_boundary_flux() {
    // Constant mobilities with a linear pressure field. Sum over the four
    // x=10 nodes equals beta*Kx*grad(p)*face_area*dt for each component.
    let p = DeadOil::default();
    let state = [1e7, 0.4];
    let mut op = vec![0.0; p.layout().n_ops()];
    p.evaluate(&state, &mut op);
    let s: Vec<f64> = box_corners()
        .iter()
        .flat_map(|x| [1e7 + 2000.0 * x[0], 0.4])
        .collect();
    let dt = 100.0;
    let b = element().residual(&p, &s, &s, dt, 0.0, &[0.0; 2]).unwrap();
    for c in 0..2 {
        let rate: f64 = (0..8)
            .filter(|a| a & 1 == 1)
            .map(|a| b.residual[2 * a + c])
            .sum();
        let expected =
            dt * 100.0 * 1e-12 * 2000.0 * (0..2).map(|ph| op[p.layout().beta(c, ph)]).sum::<f64>();
        assert!((rate - expected).abs() < 1e-10 * expected.abs());
    }
}
#[test]
fn rejects_bad_shapes_states_materials_and_thermal_physics() {
    let p = DeadOil::default();
    let s = [1e7, 0.4].repeat(8);
    assert!(element()
        .residual(&p, &s[..15], &s, 1.0, 0.0, &[0.0; 2])
        .is_err());
    for dt in [0.0, -1.0, f64::NAN] {
        assert!(element().residual(&p, &s, &s, dt, 0.0, &[0.0; 2]).is_err());
    }
    for z in [-0.1, 1.1, f64::NAN] {
        let mut bad = s.clone();
        bad[1] = z;
        assert!(element()
            .residual(&p, &s, &bad, 1.0, 0.0, &[0.0; 2])
            .is_err());
    }
    assert!(ComponentElement::new(&box_corners(), [-1.0; 3], 0.2).is_err());
    assert!(ComponentElement::new(&box_corners(), [1e-12; 3], 1.1).is_err());
    let t = ThermalWater::new(RockThermal::default(), 0.0, 1e5);
    assert!(element()
        .residual(
            &t,
            &[1e7, 323.15].repeat(8),
            &[1e7, 323.15].repeat(8),
            1.0,
            0.0,
            &[0.0; 2]
        )
        .is_err());
}
fn face(u: [f64; 3], v: [f64; 3]) -> [[f64; 3]; 4] {
    std::array::from_fn(|a| {
        std::array::from_fn(|d| 100.0 + (a & 1) as f64 * u[d] + ((a >> 1) & 1) as f64 * v[d])
    })
}
fn check_interface(physics: &dyn OblPhysics, ls: &[f64], rs: &[f64]) {
    // Two planes with distinct normals/strike/dip, already cut and paired.
    for (u, v) in [
        ([10.0, 2.0, 1.0], [-3.0, 8.0, 4.0]),
        ([6.0, -4.0, 3.0], [2.0, 7.0, -5.0]),
    ] {
        let f = ResistiveInterface::new(&face(u, v), 1e-13).unwrap();
        let c = physics.n_vars();
        let left = ls.repeat(4);
        let right = rs.repeat(4);
        let b = f.residual(physics, &left, &right, 100.0).unwrap();
        assert!(b.transfer.iter().any(|v| v.abs() > 1e-9));
        for k in 0..c {
            let l: f64 = (0..4).map(|a| b.left[a * c + k]).sum();
            let r: f64 = (0..4).map(|a| b.right[a * c + k]).sum();
            assert_eq!(l, -r);
            assert!((l - b.transfer[k]).abs() < 1e-12 * l.abs().max(1.0));
        }
        // Reverse pressures AND fluid composition: equal/opposite transfer,
        // with the same physical upstream side retained.
        let reversed = f.residual(physics, &right, &left, 100.0).unwrap();
        for k in 0..c {
            assert!(
                (b.transfer[k] + reversed.transfer[k]).abs() < 1e-12 * b.transfer[k].abs().max(1.0)
            );
        }
        let same = f.residual(physics, &left, &left, 100.0).unwrap();
        assert!(same.transfer.iter().all(|v| *v == 0.0));
        let sealed = ResistiveInterface::new(&face(u, v), 0.0)
            .unwrap()
            .residual(physics, &left, &right, 100.0)
            .unwrap();
        assert!(sealed.transfer.iter().all(|v| *v == 0.0));
    }
}
#[test]
fn tilted_interfaces_conserve_water_oil_components() {
    check_interface(&DeadOil::default(), &[1.1e7, 0.8], &[1e7, 0.3]);
}
#[test]
fn tilted_interfaces_conserve_black_oil_components() {
    check_interface(&black_oil(), &[1.1e7, 0.2, 0.6], &[1e7, 0.1, 0.7]);
}
#[test]
fn tilted_interfaces_conserve_co2_brine_components() {
    let p = Co2Brine::default();
    check_interface(&p, &[1.1e7, 0.4], &[1e7, 0.1]);
    assert_eq!(p.n_fallbacks(), 0);
}
#[test]
fn interface_area_upstream_mobility_and_reversal_match_analytic_rate() {
    let p = DeadOil::default();
    let x = face([10.0, 0.0, 0.0], [0.0, 3.0, 4.0]);
    let f = ResistiveInterface::new(&x, 2e-13).unwrap();
    assert!((f.area() - 50.0).abs() < 1e-12);
    let left = [1.1e7, 0.8];
    let right = [1e7, 0.3];
    let l = p.layout();
    let mut op = vec![0.0; l.n_ops()];
    p.evaluate(&left, &mut op);
    let b = f
        .residual(&p, &left.repeat(4), &right.repeat(4), 100.0)
        .unwrap();
    for c in 0..2 {
        let expected = 100.0 * 50.0 * 2e-13 * 1e6 * (0..2).map(|ph| op[l.beta(c, ph)]).sum::<f64>();
        assert!((b.transfer[c] - expected).abs() < 1e-12 * expected);
    }
}
#[test]
fn rejects_warped_folded_and_invalid_interfaces() {
    let x = face([10.0, 0.0, 0.0], [0.0, 10.0, 0.0]);
    let mut warped = x;
    warped[3][2] += 1.0;
    assert!(ResistiveInterface::new(&warped, 1e-13).is_err());
    let mut folded = x;
    folded.swap(2, 3);
    assert!(ResistiveInterface::new(&folded, 1e-13).is_err());
    assert!(ResistiveInterface::new(&x, -1.0).is_err());
    assert!(ResistiveInterface::new(&[[0.0; 3]; 4], 1e-13).is_err());
    let p = DeadOil::default();
    let f = ResistiveInterface::new(&x, 1e-13).unwrap();
    assert!(f
        .residual(&p, &[1e7, 0.4].repeat(3), &[1e7, 0.4].repeat(4), 1.0)
        .is_err());
}

#[test]
fn cached_element_batches_match_bitwise_with_one_and_four_threads() {
    use sx_core::flow::fem::multiphase::{assemble_components, ComponentTask};
    use sx_core::flow::with_threads;
    let e = element();
    let p = DeadOil::default();
    let source = [0.01, -0.02];
    let old = [1e7, 0.4].repeat(8);
    let current: Vec<Vec<f64>> = (0..1024)
        .map(|i| {
            (0..8)
                .flat_map(|a| {
                    [
                        1e7 + 100.0 * i as f64 + 200.0 * a as f64,
                        0.4 + 0.001 * (i % 10) as f64,
                    ]
                })
                .collect()
        })
        .collect();
    let tasks: Vec<_> = current
        .iter()
        .map(|s| ComponentTask {
            element: &e,
            old: &old,
            current: s,
            source: &source,
        })
        .collect();
    let serial = with_threads(Some(1), || assemble_components(&tasks, &p, 100.0, 9.81)).unwrap();
    let parallel = with_threads(Some(4), || assemble_components(&tasks, &p, 100.0, 9.81)).unwrap();
    assert_eq!(serial.len(), 1024);
    for (a, b) in serial.iter().zip(&parallel) {
        assert_eq!(a.residual, b.residual);
        assert_eq!(a.mass_before, b.mass_before);
        assert_eq!(a.mass_after, b.mass_after);
    }
    assert_ne!(parallel[0].residual, parallel[1].residual);
    assert!(assemble_components(&[], &p, 100.0, 0.0).is_err());
}
#[test]
fn assembled_component_patch_cancels_at_a_skew_mesh_interior_node() {
    let p = DeadOil::default();
    let mut sum = [0.0_f64; 2];
    let mut scale = [0.0_f64; 2];
    let node = |i: usize, j: usize, k: usize| {
        let mut x = [
            10.0 * i as f64 + 2.0 * j as f64,
            20.0 * j as f64 + 3.0 * k as f64,
            1000.0 + 5.0 * k as f64,
        ];
        if (i, j, k) == (1, 1, 1) {
            x[0] += 0.6;
            x[1] -= 0.4;
            x[2] += 0.2;
        }
        x
    };
    for cell in 0..8 {
        let (i, j, k) = (cell & 1, (cell >> 1) & 1, (cell >> 2) & 1);
        let x = std::array::from_fn(|a| node(i + (a & 1), j + ((a >> 1) & 1), k + ((a >> 2) & 1)));
        let state: Vec<f64> = x
            .iter()
            .flat_map(|x| [1e7 + 2000.0 * x[0] - 1000.0 * x[1] + 3000.0 * x[2], 0.4])
            .collect();
        let b = ComponentElement::new(&x, [1e-12, 2e-12, 0.5e-12], 0.2)
            .unwrap()
            .residual(&p, &state, &state, 100.0, 0.0, &[0.0; 2])
            .unwrap();
        let a = (1 - i) + 2 * (1 - j) + 4 * (1 - k);
        for c in 0..2 {
            sum[c] += b.residual[2 * a + c];
            scale[c] += b.residual[2 * a + c].abs();
        }
    }
    for c in 0..2 {
        assert!(sum[c].abs() < 1e-10 * scale[c], "{sum:?} / {scale:?}");
    }
}
