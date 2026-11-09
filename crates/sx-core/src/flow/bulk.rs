// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Conforming hexahedral bulk meshes, with explicit shared-node topology.
//!
//! FVM uses a two-point flux approximation and therefore requires planar faces
//! and permeability-orthogonal centre-to-face directions. Skew/non-orthogonal
//! input is rejected, not silently treated as a consistent discretization.
//! Hanging faces, fault overlaps and general polyhedra require other interfaces.

use super::fem::hex8::gauss_points;
use super::{Connection, FlowModel};
use crate::error::{SxError, SxResult};
use std::collections::{BTreeMap, BTreeSet};

/// Explicit bulk mesh. Cell nodes use `di + 2*dj + 4*dk` corner order;
/// coordinates are `(x, y, depth)` in metres, with depth positive down.
/// Shared faces must share the same four node IDs. No geometric remeshing,
/// hanging-node constraints or fault-interface treatment is inferred.
#[derive(Debug, Clone)]
pub struct BulkMesh {
    pub nodes: Vec<[f64; 3]>,
    pub cells: Vec<[usize; 8]>,
}
const FACES: [[usize; 4]; 6] = [
    [0, 2, 6, 4],
    [1, 3, 7, 5],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 1, 3, 2],
    [4, 5, 7, 6],
];
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|d| a[d] - b[d])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|d| a[d] * b[d]).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn mean(p: &[[f64; 3]]) -> [f64; 3] {
    std::array::from_fn(|d| p.iter().map(|x| x[d]).sum::<f64>() / p.len() as f64)
}
fn invalid(s: &str) -> SxError {
    SxError::InvalidInput(format!("bulk mesh: {s}"))
}
impl BulkMesh {
    /// Validate finite coordinates, cell indices, unique vertices/cells, positive
    /// element volumes and manifold face ownership. Duplicate coordinates with
    /// different node IDs are rejected to avoid silently sealing an interface.
    pub fn new(nodes: Vec<[f64; 3]>, cells: Vec<[usize; 8]>) -> SxResult<Self> {
        if nodes.is_empty() || cells.is_empty() {
            return Err(invalid("nodes and cells must be nonempty"));
        }
        let mut coords = BTreeSet::new();
        for p in &nodes {
            if p.iter().any(|v| !v.is_finite()) {
                return Err(invalid("coordinates must be finite"));
            }
            let key = p.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
            if !coords.insert(key) {
                return Err(invalid("coincident vertices must use one shared node ID"));
            }
        }
        let mut unique = BTreeSet::new();
        let mut faces = BTreeMap::<[usize; 4], usize>::new();
        for cell in &cells {
            if cell.iter().any(|&a| a >= nodes.len()) {
                return Err(invalid("cell node index out of range"));
            }
            let mut key = *cell;
            key.sort_unstable();
            if key.windows(2).any(|w| w[0] == w[1]) || !unique.insert(key) {
                return Err(invalid("repeated node or duplicate cell"));
            }
            let x = cell.map(|a| nodes[a]);
            gauss_points(&x)?;
            for f in FACES {
                let mut key = f.map(|v| cell[v]);
                key.sort_unstable();
                let count = faces.entry(key).or_default();
                *count += 1;
                if *count > 2 {
                    return Err(invalid("non-manifold face has more than two cells"));
                }
            }
        }
        Ok(Self { nodes, cells })
    }
    /// Corners for the continuous Galerkin FEM solver or mesh export.
    pub fn corners(&self) -> Vec<[[f64; 3]; 8]> {
        self.cells
            .iter()
            .map(|c| c.map(|a| self.nodes[a]))
            .collect()
    }
}
impl FlowModel {
    /// Build FVM connections on a conforming hexahedral bulk mesh.
    ///
    /// Permeability is diagonal in world coordinates (m²), porosity is in [0,1].
    /// Non-orthogonal faces are rejected: this entry point is deliberately not a
    /// general skew-mesh flux scheme. Bulk cell IDs are explicit; `dims=[n,1,1]`
    /// is a flat storage shape, not physical Cartesian dimensions. Well completions
    /// must use explicit cell IDs; the Peaceman well approximation uses cell extents.
    /// Mesh conformance (no hanging faces/overlap) is the caller's responsibility.
    pub fn from_bulk_mesh(
        mesh: &BulkMesh,
        perm: &[[f64; 3]],
        poro: &[f64],
        mask: Option<&[bool]>,
    ) -> SxResult<Self> {
        // Revalidate since fields are public and callers may have mutated them.
        let checked = BulkMesh::new(mesh.nodes.clone(), mesh.cells.clone())?;
        let cs = checked.corners();
        let n = cs.len();
        if perm.len() != n || poro.len() != n || mask.is_some_and(|m| m.len() != n) {
            return Err(SxError::Shape(
                "bulk mesh: perm, poro and active need one entry per cell".into(),
            ));
        }
        if perm.iter().flatten().any(|v| !v.is_finite() || *v < 0.0)
            || poro
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(invalid(
                "permeability must be finite/nonnegative and porosity in [0,1]",
            ));
        }
        let mut m = Self {
            dims: [n, 1, 1],
            active: vec![false; n],
            pore_volume: vec![0.0; n],
            bulk_volume: vec![0.0; n],
            center: vec![[0.0; 3]; n],
            extent: vec![[0.0; 3]; n],
            perm: perm.to_vec(),
            connections: vec![],
            warnings: vec![
                "bulk mesh dims are flat storage; use explicit cell IDs for wells".into(),
            ],
            n_fault_connections: 0,
            n_pinch_connections: 0,
        };
        // face owner -> (cell, half permeability conductance, half geometric conductance, normal, centre).
        let mut faces = BTreeMap::<[usize; 4], Vec<(usize, f64, f64, [f64; 3], [f64; 3])>>::new();
        for (c, x) in cs.iter().enumerate() {
            let gps = gauss_points(x)?;
            let vol: f64 = gps.iter().map(|q| q.w).sum();
            m.bulk_volume[c] = vol;
            m.center[c] = std::array::from_fn(|d| {
                gps.iter()
                    .map(|q| q.w * (0..8).map(|v| q.n[v] * x[v][d]).sum::<f64>())
                    .sum::<f64>()
                    / vol
            });
            m.active[c] =
                mask.is_none_or(|a| a[c]) && poro[c] > 0.0 && perm[c].iter().any(|&k| k > 0.0);
            m.pore_volume[c] = if m.active[c] { vol * poro[c] } else { 0.0 };
            for axis in 0..3 {
                m.extent[c][axis] = dot(
                    sub(
                        mean(&FACES[2 * axis + 1].map(|v| x[v])),
                        mean(&FACES[2 * axis].map(|v| x[v])),
                    ),
                    sub(
                        mean(&FACES[2 * axis + 1].map(|v| x[v])),
                        mean(&FACES[2 * axis].map(|v| x[v])),
                    ),
                )
                .sqrt();
            }
            for f in FACES {
                let p = f.map(|v| x[v]);
                let fc = mean(&p);
                let area = cross(sub(p[2], p[0]), sub(p[3], p[1])).map(|v| v * 0.5);
                let an = dot(area, area).sqrt();
                let d = sub(fc, m.center[c]);
                let dd = dot(d, d);
                if !(an > 0.0 && dd > 0.0) {
                    return Err(invalid("degenerate face"));
                }
                let plane = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                if dot(plane, sub(p[3], p[0])).abs() > 1e-9 * dot(plane, plane).sqrt() * dd.sqrt() {
                    return Err(invalid("warped faces need a different flux approximation"));
                }
                let ka = std::array::from_fn(|d| perm[c][d] * area[d]);
                let kd = dot(ka, ka).sqrt();
                let cos = if kd > 0.0 {
                    dot(ka, d).abs() / (kd * dd.sqrt())
                } else {
                    1.0
                };
                if m.active[c] && cos < 1.0 - 1e-8 {
                    return Err(invalid("non-orthogonal permeability/face geometry needs a multi-point or corrected flux approximation"));
                }
                let ht = dot(ka, d).abs() / dd;
                let hg = dot(area, d).abs() / dd;
                let mut key = f.map(|v| checked.cells[c][v]);
                key.sort_unstable();
                faces.entry(key).or_default().push((c, ht, hg, area, fc));
            }
        }
        for owners in faces.values() {
            if let [a, b] = owners.as_slice() {
                if dot(a.3, sub(a.4, m.center[a.0])) * dot(a.3, sub(a.4, m.center[b.0])) >= 0.0 {
                    return Err(invalid("shared-face cells must lie on opposite sides"));
                }
                if m.active[a.0] && m.active[b.0] && a.1 > 0.0 && b.1 > 0.0 {
                    m.connections.push(Connection {
                        a: a.0,
                        b: b.0,
                        trans: 1.0 / (1.0 / a.1 + 1.0 / b.1),
                        geom: [a.2, b.2],
                    });
                }
            }
        }
        m.connections.sort_by_key(|c| (c.a.min(c.b), c.a.max(c.b)));
        Ok(m)
    }
}
