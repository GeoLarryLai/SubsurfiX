// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! OpenGeoSys (OGS) mesh export: node sharing, orientation, volumes and arrays.

use sx_core::flow::ogs::{ogs_vtu, regular_corners, CellArray};
use sx_core::grid::{CornerPointGrid, RegularGrid};

/// Parse points and connectivity back out of the ASCII `.vtu`.
fn parse(s: &str) -> (Vec<[f64; 3]>, Vec<[usize; 8]>) {
    let block = |name: &str| -> Vec<f64> {
        let start = s.find(name).unwrap();
        let body = &s[start..];
        let body = &body[body.find('>').unwrap() + 1..];
        body[..body.find("</DataArray>").unwrap()].split_whitespace().map(|t| t.parse().unwrap()).collect()
    };
    let p = block("NumberOfComponents=\"3\" format=\"ascii\"");
    let c = block("Name=\"connectivity\"");
    (
        p.chunks(3).map(|x| [x[0], x[1], x[2]]).collect(),
        c.chunks(8).map(|x| std::array::from_fn(|i| x[i] as usize)).collect(),
    )
}

/// Signed hexahedron volume, VTK node order, six tetrahedra around the 0–6 diagonal.
fn hex_volume(p: &[[f64; 3]], h: &[usize; 8]) -> f64 {
    let tet = |a: usize, b: usize, c: usize, d: usize| {
        let (a, b, c, d) = (p[h[a]], p[h[b]], p[h[c]], p[h[d]]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])) / 6.0
    };
    tet(0, 1, 2, 6) + tet(0, 2, 3, 6) + tet(0, 3, 7, 6) + tet(0, 7, 4, 6) + tet(0, 4, 5, 6) + tet(0, 5, 1, 6)
}

#[test]
fn regular_grid_shares_nodes_and_keeps_volume() {
    let g = RegularGrid::new([100.0, 200.0, 1000.0], [10.0, 20.0, 5.0], [3, 2, 2]).unwrap();
    let corners = regular_corners(&g);
    let poro = vec![0.25; 12];
    let arrays = [CellArray { name: "porosity", values: &poro, components: 1 }];
    let (s, info) = ogs_vtu(&corners, &[true; 12], None, &arrays, 1e-6).unwrap();
    assert_eq!((info.n_nodes, info.n_cells), (4 * 3 * 3, 12));
    let (p, h) = parse(&s);
    assert_eq!((p.len(), h.len()), (36, 12));
    for cell in &h {
        assert!((hex_volume(&p, cell) - 1000.0).abs() < 1e-9);
    }
    assert!(p.iter().all(|x| (-1010.0..=-1000.0).contains(&x[2])));
    assert!(s.contains("Name=\"MaterialIDs\"") && s.contains("Name=\"porosity\""));
    // Points, three cell arrays (connectivity, offsets, types), MaterialIDs, bulk_cell_id, porosity.
    assert_eq!(s.matches("<DataArray").count(), 1 + 3 + 3);
}

#[test]
fn inactive_cells_dropped_and_ids_kept() {
    let g = RegularGrid::new([0.0; 3], [1.0; 3], [2, 1, 1]).unwrap();
    let (s, info) = ogs_vtu(&regular_corners(&g), &[false, true], Some(&[3, 7]), &[], 1e-6).unwrap();
    assert_eq!((info.n_nodes, info.n_cells), (8, 1));
    let after = |name: &str| s[s.find(name).unwrap()..].split('>').nth(1).unwrap().split('<').next().unwrap().trim().to_string();
    assert_eq!(after("Name=\"MaterialIDs\""), "7");
    assert_eq!(after("Name=\"bulk_cell_id\""), "1");
}

#[test]
fn faulted_corner_point_grid_is_positive_and_non_conforming() {
    let r = RegularGrid::new([0.0, 0.0, 1000.0], [10.0, 10.0, 10.0], [2, 1, 2]).unwrap();
    let g = CornerPointGrid::from_regular(&r);
    let mut z = g.zcorn.clone();
    for (idx, v) in z.iter_mut().enumerate() {
        if idx % 4 >= 2 {
            *v += 5.0;
        }
    }
    let g = CornerPointGrid::new(g.dims, g.coord.clone(), z, None).unwrap();
    let corners: Vec<_> = (0..4).map(|c| g.corners(c)).collect();
    let (s, info) = ogs_vtu(&corners, &[true; 4], None, &[], 1e-6).unwrap();
    // Unfaulted: 3 pillars x 2 y-lines x 3 depths = 18. The thrown column has its
    // own 3 depths on the shared pillar (non-conforming fault face): 6 more.
    assert_eq!(info.n_nodes, 18 + 6, "{info:?}");
    let (p, h) = parse(&s);
    for cell in &h {
        assert!((hex_volume(&p, cell) - 1000.0).abs() < 1e-9);
    }
}

#[test]
fn rejects_bad_input() {
    let g = RegularGrid::new([0.0; 3], [1.0; 3], [2, 1, 1]).unwrap();
    let c = regular_corners(&g);
    let v = vec![1.0; 2];
    let ok = |a: &[CellArray<'_>]| ogs_vtu(&c, &[true; 2], None, a, 1e-6).is_ok();
    assert!(ok(&[CellArray { name: "k", values: &v, components: 1 }]));
    assert!(!ok(&[CellArray { name: "k", values: &v, components: 2 }]));
    assert!(!ok(&[CellArray { name: "a<b", values: &v, components: 1 }]));
    assert!(!ok(&[CellArray { name: "MaterialIDs", values: &v, components: 1 }]));
    assert!(!ok(&[CellArray { name: "k", values: &v, components: 1 }, CellArray { name: "k", values: &v, components: 1 }]));
    assert!(ogs_vtu(&c, &[true], None, &[], 1e-6).is_err());
    assert!(ogs_vtu(&c, &[true; 2], Some(&[1]), &[], 1e-6).is_err());
    assert!(ogs_vtu(&c, &[true; 2], None, &[], 0.0).is_err());
}
