// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Fault and pinch-out connections, net-to-gross and transmissibility multipliers.

use std::collections::BTreeMap;

use sx_core::flow::{FlowModel, FlowOptions};
use sx_core::grid::{CornerPointGrid, RegularGrid};

const K: f64 = 1e-13;

fn faulted(throw: f64) -> CornerPointGrid {
    // 2 x 1 x 2, 10 m cells; the i = 1 column is shifted down by `throw`.
    let r = RegularGrid::new([0.0, 0.0, 1000.0], [10.0, 10.0, 10.0], [2, 1, 2]).unwrap();
    let g = CornerPointGrid::from_regular(&r);
    let mut z = g.zcorn.clone();
    for (idx, v) in z.iter_mut().enumerate() {
        if idx % 4 >= 2 {
            *v += throw;
        }
    }
    CornerPointGrid::new(g.dims, g.coord.clone(), z, None).unwrap()
}

fn trans(m: &FlowModel, a: usize, b: usize) -> Option<f64> {
    m.connections.iter().find(|c| c.a == a && c.b == b).map(|c| c.trans)
}

#[test]
fn half_cell_throw_gives_overlap_connections() {
    // Cells: left (0, 2) and right (1, 3) for k = 0, 1. Throw 5 m on 10 m cells.
    let (perm, poro) = (vec![[K; 3]; 4], vec![0.2; 4]);
    let m = FlowModel::from_corner_point(&faulted(5.0), &perm, &poro).unwrap();
    assert_eq!(m.n_fault_connections, 3);
    // Overlap 5 m x 10 m; centroid offset (5, 0, 2.5) from both centres:
    // half T = 50 K 5 / 31.25 = 8 K, T = 4 K.
    for (a, b) in [(0, 1), (2, 1), (2, 3)] {
        let t = trans(&m, a, b).unwrap_or_else(|| panic!("no {a}-{b}"));
        assert!((t / (4.0 * K) - 1.0).abs() < 1e-12, "{a}-{b}: {t}");
    }
    assert!(trans(&m, 0, 3).is_none());
    // Vertical connections untouched.
    assert!(trans(&m, 0, 2).is_some() && trans(&m, 1, 3).is_some());
}

#[test]
fn full_cell_throw_connects_offset_layers_like_neighbours() {
    let (perm, poro) = (vec![[K; 3]; 4], vec![0.2; 4]);
    let flat = FlowModel::from_corner_point(&faulted(0.0), &perm, &poro).unwrap();
    let m = FlowModel::from_corner_point(&faulted(10.0), &perm, &poro).unwrap();
    assert_eq!(m.n_fault_connections, 1);
    assert!((trans(&m, 2, 1).unwrap() / trans(&flat, 0, 1).unwrap() - 1.0).abs() < 1e-12);
    // Below tolerance: plain logical connections.
    let m = FlowModel::from_corner_point(&faulted(1e-4), &perm, &poro).unwrap();
    assert_eq!(m.n_fault_connections, 0);
    assert_eq!(m.connections.len(), flat.connections.len());
}

#[test]
fn flow_across_a_fault_balances() {
    use sx_core::flow::{solve_pressure, FixedPressure};
    let (perm, poro) = (vec![[K; 3]; 4], vec![0.2; 4]);
    let m = FlowModel::from_corner_point(&faulted(5.0), &perm, &poro).unwrap();
    let fixed = [
        FixedPressure { cell: 0, trans: m.boundary_trans(0, 0), pressure: 2e7 },
        FixedPressure { cell: 3, trans: m.boundary_trans(3, 0), pressure: 1e7 },
    ];
    let sol = solve_pressure(&m, &[1e3; 4], 1e3, &[], &fixed).unwrap();
    let (qi, qo) = (sol.boundary_flux[0], sol.boundary_flux[1]);
    assert!(qi > 0.0 && (qi + qo).abs() < 1e-12 * qi);
}

#[test]
fn pinch_out_connects_across_thin_cells() {
    let xs = [0.0, 10.0];
    let surf = |z: f64| vec![z; 4];
    let g = CornerPointGrid::from_surfaces(&xs, &xs, &[surf(0.0), surf(5.0), surf(5.0), surf(5.0), surf(9.0)]).unwrap();
    let (perm, poro) = (vec![[K; 3]; 4], vec![0.2; 4]);
    let m = FlowModel::from_corner_point(&g, &perm, &poro).unwrap();
    assert_eq!(m.n_pinch_connections, 1);
    // Half T: 100 m2 K / 2.5 m and 100 m2 K / 2 m.
    let expect = 1.0 / (2.5 / (100.0 * K) + 2.0 / (100.0 * K));
    assert!((trans(&m, 0, 3).unwrap() / expect - 1.0).abs() < 1e-12);
    // A thick inactive cell blocks it.
    let g = CornerPointGrid::new(g.dims, g.coord.clone(), g.zcorn.clone(), Some(vec![true, false, true, true])).unwrap();
    let m = FlowModel::from_corner_point(&g, &perm, &poro).unwrap();
    assert_eq!(m.n_pinch_connections, 1);
    let g2 = CornerPointGrid::from_surfaces(&xs, &xs, &[surf(0.0), surf(5.0), surf(7.0), surf(9.0)]).unwrap();
    let g2 = CornerPointGrid::new(g2.dims, g2.coord.clone(), g2.zcorn.clone(), Some(vec![true, false, true])).unwrap();
    let m = FlowModel::from_corner_point(&g2, &[[K; 3]; 3], &[0.2; 3]).unwrap();
    assert_eq!(m.n_pinch_connections, 0);
    assert!(m.connections.is_empty());
}

#[test]
fn ntg_and_multipliers_scale_transmissibility() {
    let r = RegularGrid::new([0.0; 3], [10.0; 3], [3, 1, 2]).unwrap();
    let g = CornerPointGrid::from_regular(&r);
    let n = 6;
    let (perm, poro) = (vec![[K; 3]; n], vec![0.2; n]);
    let base = FlowModel::from_corner_point(&g, &perm, &poro).unwrap();
    let t = |m: &FlowModel, a, b| m.connections.iter().find(|c| c.a == a && c.b == b).unwrap().trans;
    let mut props = BTreeMap::new();
    props.insert("NTG".to_string(), vec![0.5; n]);
    let mut mx = vec![1.0; n];
    mx[0] = 0.1; // face 0 -> 1
    props.insert("MULTX".to_string(), mx);
    let mut mxm = vec![1.0; n];
    mxm[2] = 0.3; // face 2 -> 1
    props.insert("MULTX-".to_string(), mxm);
    let mut mz = vec![1.0; n];
    mz[1] = 0.0; // seal 1 -> 4
    props.insert("MULTZ".to_string(), mz);
    let m = FlowModel::from_corner_point_with(&g, &perm, &poro, &FlowOptions::from_props(&props)).unwrap();
    assert!((t(&m, 0, 1) / t(&base, 0, 1) - 0.05).abs() < 1e-12);
    assert!((t(&m, 1, 2) / t(&base, 1, 2) - 0.15).abs() < 1e-12);
    // Vertical: NTG does not touch kz.
    assert!((t(&m, 0, 3) / t(&base, 0, 3) - 1.0).abs() < 1e-12);
    assert!(!m.connections.iter().any(|c| c.a == 1 && c.b == 4));
    assert!((m.pore_volume[0] / base.pore_volume[0] - 0.5).abs() < 1e-12);
    assert!((m.perm[0][0] - 0.5 * K).abs() < 1e-25 && m.perm[0][2] == K);
    // Bad input.
    let mut bad = BTreeMap::new();
    bad.insert("MULTQ".to_string(), vec![1.0; n]);
    let o = FlowOptions { mult: bad, ..FlowOptions::default() };
    assert!(FlowModel::from_corner_point_with(&g, &perm, &poro, &o).is_err());
    let o = FlowOptions { ntg: Some(vec![1.5; n]), ..FlowOptions::default() };
    assert!(FlowModel::from_corner_point_with(&g, &perm, &poro, &o).is_err());
}

#[test]
fn grdecl_ntg_and_mult_reach_flow_options() {
    let r = RegularGrid::new([0.0; 3], [10.0; 3], [2, 1, 1]).unwrap();
    let g = CornerPointGrid::from_regular(&r);
    let mut props = BTreeMap::new();
    props.insert("NTG".to_string(), vec![0.8, 0.6]);
    props.insert("MULTX".to_string(), vec![0.5, 1.0]);
    props.insert("MULTY-".to_string(), vec![1.0, 1.0]);
    let text = g.to_grdecl(&props).unwrap();
    let (_, back) = CornerPointGrid::from_grdecl(&text).unwrap();
    let o = FlowOptions::from_props(&back);
    assert_eq!(o.ntg, Some(vec![0.8, 0.6]));
    assert_eq!(o.mult.len(), 2);
    assert_eq!(o.mult["MULTX"], vec![0.5, 1.0]);
}
