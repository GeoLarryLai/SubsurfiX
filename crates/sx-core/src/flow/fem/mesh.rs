// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Node-based hexahedral mesh built from the 8 corners of every active cell.
//!
//! Corners closer than `merge_tol` become one node (the rule of
//! [`ogs_vtu`](super::super::ogs::ogs_vtu)), so neighbours that share a face
//! share its 4 nodes. Coordinates stay in the grid frame (x, y, depth
//! positive down); element nodes keep the corner-point order
//! `c = di + 2·dj + 4·dk` of [`super::hex8`].

use std::collections::HashMap;

use crate::error::{SxError, SxResult};

/// Hexahedral mesh of the active cells of a grid.
#[derive(Debug, Clone, PartialEq)]
pub struct HexMesh {
    /// Node coordinates (x, y, depth), m.
    pub nodes: Vec<[f64; 3]>,
    /// Node ids of each element, corner-point order.
    pub elems: Vec<[usize; 8]>,
    /// Grid cell of each element.
    pub cell_of_elem: Vec<usize>,
    /// Element of each grid cell (`None` for inactive cells).
    pub elem_of_cell: Vec<Option<usize>>,
}

impl HexMesh {
    /// Mesh of the cells with `active[c]`, from their corners (one entry per
    /// grid cell, corner-point order).
    ///
    /// # Errors
    /// `Shape` when `corners` and `active` differ in length; `InvalidInput`
    /// for a bad `merge_tol`, a non-finite corner or no active cell.
    pub fn from_corners(corners: &[[[f64; 3]; 8]], active: &[bool], merge_tol: f64) -> SxResult<Self> {
        let n = corners.len();
        if active.len() != n {
            return Err(SxError::Shape("fem mesh: corners and active need one entry per cell".into()));
        }
        if !(merge_tol > 0.0 && merge_tol.is_finite()) {
            return Err(SxError::InvalidInput("fem mesh: merge_tol must be > 0".into()));
        }
        let mut key_to_node: HashMap<[i64; 3], usize> = HashMap::new();
        let mut mesh = HexMesh { nodes: Vec::new(), elems: Vec::new(), cell_of_elem: Vec::new(), elem_of_cell: vec![None; n] };
        for c in (0..n).filter(|&c| active[c]) {
            let mut ids = [0usize; 8];
            for (v, id) in ids.iter_mut().enumerate() {
                let p = corners[c][v];
                if p.iter().any(|x| !x.is_finite()) {
                    return Err(SxError::InvalidInput(format!("fem mesh: cell {c} has a non-finite corner")));
                }
                let key = [0, 1, 2].map(|a| (p[a] / merge_tol).round() as i64);
                *id = *key_to_node.entry(key).or_insert_with(|| {
                    mesh.nodes.push(p);
                    mesh.nodes.len() - 1
                });
            }
            mesh.elem_of_cell[c] = Some(mesh.elems.len());
            mesh.elems.push(ids);
            mesh.cell_of_elem.push(c);
        }
        if mesh.elems.is_empty() {
            return Err(SxError::InvalidInput("fem mesh: no active cell".into()));
        }
        Ok(mesh)
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Nodal values from cell values: the mean over the active cells that
    /// share each node.
    pub fn cell_to_nodes(&self, cell: &[f64]) -> Vec<f64> {
        let mut sum = vec![0.0; self.n_nodes()];
        let mut cnt = vec![0usize; self.n_nodes()];
        for (e, ids) in self.elems.iter().enumerate() {
            for &a in ids {
                sum[a] += cell[self.cell_of_elem[e]];
                cnt[a] += 1;
            }
        }
        sum.iter().zip(&cnt).map(|(s, &k)| s / k as f64).collect()
    }

    /// Cell values from nodal values: the mean of each element's 8 nodes (the
    /// trilinear value at the cell centre); inactive cells take `inactive[c]`.
    pub fn nodes_to_cells(&self, nodal: &[f64], inactive: &[f64]) -> Vec<f64> {
        (0..self.elem_of_cell.len())
            .map(|c| match self.elem_of_cell[c] {
                Some(e) => self.elems[e].iter().map(|&a| nodal[a]).sum::<f64>() / 8.0,
                None => inactive[c],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::ogs::regular_corners;
    use crate::grid::RegularGrid;

    fn grid() -> RegularGrid {
        RegularGrid { origin: [0.0, 0.0, 1000.0], cell: [10.0, 20.0, 5.0], dims: [3, 2, 2] }
    }

    #[test]
    fn regular_grid_shares_nodes() {
        let g = grid();
        let m = HexMesh::from_corners(&regular_corners(&g), &[true; 12], 1e-6).unwrap();
        assert_eq!(m.elems.len(), 12);
        assert_eq!(m.n_nodes(), 4 * 3 * 3);
    }

    #[test]
    fn inactive_cells_drop_their_own_nodes() {
        let g = grid();
        let mut active = [true; 12];
        active[11] = false; // corner cell (2, 1, 1): owns one node no other cell touches
        let m = HexMesh::from_corners(&regular_corners(&g), &active, 1e-6).unwrap();
        assert_eq!(m.elems.len(), 11);
        assert_eq!(m.n_nodes(), 36 - 1);
        assert_eq!(m.elem_of_cell[11], None);
    }

    #[test]
    fn cell_node_round_trip_keeps_uniform_values() {
        let g = grid();
        let m = HexMesh::from_corners(&regular_corners(&g), &[true; 12], 1e-6).unwrap();
        let nodal = m.cell_to_nodes(&[3.5; 12]);
        assert!(nodal.iter().all(|&v| (v - 3.5).abs() < 1e-14));
        let cells = m.nodes_to_cells(&nodal, &[0.0; 12]);
        assert!(cells.iter().all(|&v| (v - 3.5).abs() < 1e-14));
    }
}
