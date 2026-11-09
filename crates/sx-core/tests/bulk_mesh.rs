// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Bulk mesh FVM connections, FEM use, and explicit unsupported-interface guards.
use sx_core::flow::{
    run_fem_ht, with_threads, BulkMesh, FemOptions, FlowModel, FlowWell, RockThermal, ThermalWater,
    WellControl, WellKind,
};
fn mesh() -> BulkMesh {
    let nodes = (0..2)
        .flat_map(|k| {
            (0..2).flat_map(move |j| (0..3).map(move |i| [i as f64, j as f64, k as f64 + 1000.0]))
        })
        .collect();
    let cell = |i| std::array::from_fn(|c| i + (c & 1) + 3 * (((c >> 1) & 1) + 2 * ((c >> 2) & 1)));
    BulkMesh::new(nodes, vec![cell(0), cell(1)]).unwrap()
}
#[test]
fn bulk_matches_regular_connections_and_conserves_pore_volume() {
    let mesh = mesh();
    let m = FlowModel::from_bulk_mesh(&mesh, &[[1e-12; 3]; 2], &[0.2; 2], None).unwrap();
    assert_eq!(m.connections.len(), 1);
    let c = m.connections[0];
    assert!((c.trans / 1e-12 - 1.0).abs() < 1e-12);
    assert!((m.pore_volume.iter().sum::<f64>() - 0.4).abs() < 1e-12);
    assert!((m.bulk_volume.iter().sum::<f64>() - 2.0).abs() < 1e-12);
}
#[test]
fn arbitrary_cell_order_uses_topology_not_logical_neighbours() {
    let mut mesh = mesh();
    mesh.cells.reverse();
    let m = FlowModel::from_bulk_mesh(&mesh, &[[1e-12; 3]; 2], &[0.2; 2], None).unwrap();
    assert_eq!(m.connections.len(), 1);
    assert!(m.center[0][0] > m.center[1][0]);
}
#[test]
fn inactive_cells_do_not_connect() {
    let m = FlowModel::from_bulk_mesh(&mesh(), &[[1e-12; 3]; 2], &[0.2; 2], Some(&[true, false]))
        .unwrap();
    assert!(m.connections.is_empty());
    assert_eq!(m.pore_volume[1], 0.0);
}
#[test]
fn skew_two_point_flux_is_rejected_instead_of_claiming_consistency() {
    let mut mesh = mesh();
    for p in &mut mesh.nodes {
        p[0] += 0.2 * p[1];
    }
    assert!(
        FlowModel::from_bulk_mesh(&mesh, &[[1e-12; 3]; 2], &[0.2; 2], None)
            .unwrap_err()
            .to_string()
            .contains("non-orthogonal")
    );
}
#[test]
fn duplicate_node_and_cell_are_rejected() {
    let mut m = mesh();
    m.nodes.push(m.nodes[0]);
    assert!(BulkMesh::new(m.nodes, m.cells).is_err());
    let mut m = mesh();
    m.cells.push(m.cells[0]);
    assert!(BulkMesh::new(m.nodes, m.cells).is_err());
}
#[test]
fn fem_runs_on_bulk_mesh_and_serial_parallel_agree() {
    let mesh = mesh();
    let m = FlowModel::from_bulk_mesh(&mesh, &[[1e-12; 3]; 2], &[0.2; 2], None).unwrap();
    let wells = [
        FlowWell::vertical(
            "INJ",
            WellKind::Injector,
            WellControl::Rate(1e-8),
            vec![0],
            0.1,
            0.0,
        ),
        FlowWell::vertical(
            "PROD",
            WellKind::Producer,
            WellControl::Bhp(1e7),
            vec![1],
            0.1,
            0.0,
        ),
    ];
    let opts = FemOptions {
        injection_temperature: vec![310.0, 320.0],
        dt_init: 10.0,
        dt_max: 100.0,
        ..FemOptions::default()
    };
    let solve = || {
        run_fem_ht(
            &mesh.corners(),
            &m,
            &wells,
            &ThermalWater::new(
                RockThermal {
                    rho: 2650.0,
                    cp: 900.0,
                    conductivity: 2.5,
                },
                0.0,
                1e5,
            ),
            &[1e7; 2],
            &[320.0; 2],
            &[0.0, 100.0],
            &opts,
        )
    };
    let one = with_threads(Some(1), solve).unwrap();
    let two = with_threads(Some(2), solve).unwrap();
    assert_eq!(one.p, two.p);
    assert_eq!(one.temperature, two.temperature);
    for c in 0..2 {
        let d = one.mass_in_place[1][c] - one.mass_in_place[0][c];
        let q = one.mass_net_in[1][c] - one.mass_net_in[0][c];
        assert!((d - q).abs() < 1e-5 * q.abs().max(1.0));
    }
}
#[test]
fn fem_rejects_pinch_interface_explicitly() {
    let mesh = mesh();
    let mut m = FlowModel::from_bulk_mesh(&mesh, &[[1e-12; 3]; 2], &[0.2; 2], None).unwrap();
    m.n_pinch_connections = 1;
    let r = run_fem_ht(
        &mesh.corners(),
        &m,
        &[],
        &ThermalWater::new(
            RockThermal {
                rho: 2650.0,
                cp: 900.0,
                conductivity: 2.5,
            },
            0.0,
            1e5,
        ),
        &[1e7; 2],
        &[320.0; 2],
        &[0.0],
        &FemOptions::default(),
    );
    assert!(r.unwrap_err().to_string().contains("pinched"));
}

#[test]
fn element_parallel_path_matches_one_thread_on_1024_cells() {
    use sx_core::flow::ogs::regular_corners;
    use sx_core::grid::RegularGrid;
    let g = RegularGrid {
        origin: [0.0, 0.0, 1000.0],
        cell: [1.0; 3],
        dims: [16, 8, 8],
    };
    let n = 1024;
    let m = FlowModel::from_regular(&g, &vec![[1e-12; 3]; n], &vec![0.2; n]).unwrap();
    let cs = regular_corners(&g);
    let initial: Vec<f64> = (0..n)
        .map(|i| 300.0 + 20.0 * (i % 16) as f64 / 15.0)
        .collect();
    let solve = || {
        run_fem_ht(
            &cs,
            &m,
            &[],
            &ThermalWater::new(
                RockThermal {
                    rho: 2650.0,
                    cp: 900.0,
                    conductivity: 2.5,
                },
                0.0,
                1e5,
            ),
            &vec![1e7; n],
            &initial,
            &[0.0, 1e4],
            &FemOptions {
                dt_init: 1e4,
                ..FemOptions::default()
            },
        )
    };
    let one = with_threads(Some(1), solve).unwrap();
    let two = with_threads(Some(2), solve).unwrap();
    assert!(one.n_steps > 0);
    assert!(one.temperature[0]
        .iter()
        .zip(&one.temperature[1])
        .any(|(a, b)| (a - b).abs() > 1e-5));
    assert_eq!(one.p, two.p);
    assert_eq!(one.temperature, two.temperature);
    assert_eq!(one.mass_in_place, two.mass_in_place);
}

#[test]
fn fvm_multiphase_runs_on_bulk_mesh_with_component_mass_balance() {
    use sx_core::flow::{run_fim, DeadOil, FimOptions};
    let m = FlowModel::from_bulk_mesh(&mesh(), &[[1e-12; 3]; 2], &[0.2; 2], None).unwrap();
    let ph = DeadOil::default();
    let z0 = ph.z_from_sw(1e7, ph.corey.swc);
    let wells = [
        FlowWell::vertical(
            "INJ",
            WellKind::Injector,
            WellControl::Rate(1e-5),
            vec![0],
            0.1,
            0.0,
        ),
        FlowWell::vertical(
            "PROD",
            WellKind::Producer,
            WellControl::Bhp(1e7),
            vec![1],
            0.1,
            0.0,
        ),
    ];
    let opts = FimOptions {
        table_lo: vec![5e6, 0.0],
        table_hi: vec![1e8, 1.0],
        table_points: vec![16, 101],
        injection: vec![vec![1.0], vec![0.0]],
        dt_init: 1.0,
        dt_max: 10.0,
        ..FimOptions::default()
    };
    let result = run_fim(&m, &wells, &ph, &[1e7, z0, 1e7, z0], &[0.0, 100.0], &opts).unwrap();
    assert!(result.n_steps > 0);
    for c in 0..2 {
        let d = result.mass_in_place[1][c] - result.mass_in_place[0][c];
        let q = result.mass_net_in[1][c] - result.mass_net_in[0][c];
        assert!((d - q).abs() < 1e-6 * q.abs().max(1.0));
    }
}
