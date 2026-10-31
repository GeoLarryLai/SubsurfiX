// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Export of a flow grid to OpenGeoSys (OGS): a VTK unstructured-grid file
//! (`.vtu`, ASCII XML) with hexahedral cells, shared (merged) nodes and the
//! `MaterialIDs` cell array that OGS uses to pick media.
//!
//! Depth (positive down) becomes `z = -depth` (OGS is z up). Only active
//! cells are written; `bulk_cell_id` maps them back to grid indices.
//! Nodes closer than `merge_tol` are merged, so neighbours that share a face
//! share nodes; displaced fault faces stay non-conforming (no node sharing).

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::error::{SxError, SxResult};
use crate::grid::RegularGrid;

/// One cell array to write (one value per grid cell, active or not).
#[derive(Debug, Clone, PartialEq)]
pub struct CellArray<'a> {
    /// Array name in the file.
    pub name: &'a str,
    /// Values per grid cell.
    pub values: &'a [f64],
    /// Components per cell (1 for scalars, 3 for e.g. permeability).
    pub components: usize,
}

/// Summary of a written mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OgsMeshInfo {
    /// Nodes after merging.
    pub n_nodes: usize,
    /// Hexahedra written (active cells).
    pub n_cells: usize,
}

/// Corners of every cell of a regular grid, in the corner-point order
/// `c = di + 2·dj + 4·dk` (z as depth).
pub fn regular_corners(g: &RegularGrid) -> Vec<[[f64; 3]; 8]> {
    (0..g.n_cells())
        .map(|idx| {
            let (i, j, k) = g.ijk(idx);
            let mut out = [[0.0; 3]; 8];
            for (c, o) in out.iter_mut().enumerate() {
                let d = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
                let ijk = [i, j, k];
                for a in 0..3 {
                    o[a] = g.origin[a] + (ijk[a] + d[a]) as f64 * g.cell[a];
                }
            }
            out
        })
        .collect()
}

/// VTK hexahedron (type 12) node order from corner-point corners once z is
/// flipped up: the deeper face (dk = 1) is the base, counterclockwise seen
/// from above, then the shallower face.
const VTK_ORDER: [usize; 8] = [4, 5, 7, 6, 0, 1, 3, 2];

/// Build the `.vtu` text for the active cells.
///
/// # Errors
/// `Shape` when `corners`, `active`, `material_ids` or an array do not have
/// one entry per cell; `InvalidInput` for non-finite corners, a bad
/// `merge_tol` or an array name that is empty or not XML-safe.
pub fn ogs_vtu(
    corners: &[[[f64; 3]; 8]],
    active: &[bool],
    material_ids: Option<&[i32]>,
    arrays: &[CellArray<'_>],
    merge_tol: f64,
) -> SxResult<(String, OgsMeshInfo)> {
    let n = corners.len();
    if active.len() != n || material_ids.is_some_and(|m| m.len() != n) {
        return Err(SxError::Shape("ogs: active / material_ids need one value per cell".into()));
    }
    if !(merge_tol > 0.0 && merge_tol.is_finite()) {
        return Err(SxError::InvalidInput("ogs: merge_tol must be > 0".into()));
    }
    for (i, a) in arrays.iter().enumerate() {
        if arrays[..i].iter().any(|b| b.name == a.name) {
            return Err(SxError::InvalidInput(format!("ogs: array {:?} given twice", a.name)));
        }
        let ok_name = !a.name.is_empty()
            && a.name.chars().all(|ch| ch.is_ascii_alphanumeric() || "_-.".contains(ch))
            && a.name != "MaterialIDs"
            && a.name != "bulk_cell_id";
        if !ok_name || a.components == 0 {
            return Err(SxError::InvalidInput(format!("ogs: bad array {:?}", a.name)));
        }
        if a.values.len() != n * a.components {
            return Err(SxError::Shape(format!(
                "ogs: array {} has {} values for {n} cells x {} components",
                a.name,
                a.values.len(),
                a.components
            )));
        }
    }
    let cells: Vec<usize> = (0..n).filter(|&c| active[c]).collect();
    let mut key_to_node: HashMap<[i64; 3], usize> = HashMap::new();
    let mut nodes: Vec<[f64; 3]> = Vec::new();
    let mut conn: Vec<usize> = Vec::with_capacity(8 * cells.len());
    for &c in &cells {
        for &v in &VTK_ORDER {
            let p = corners[c][v];
            if p.iter().any(|x| !x.is_finite()) {
                return Err(SxError::InvalidInput(format!("ogs: cell {c} has a non-finite corner")));
            }
            let p = [p[0], p[1], -p[2]];
            let key = [0, 1, 2].map(|a| (p[a] / merge_tol).round() as i64);
            let id = *key_to_node.entry(key).or_insert_with(|| {
                nodes.push(p);
                nodes.len() - 1
            });
            conn.push(id);
        }
    }
    let info = OgsMeshInfo { n_nodes: nodes.len(), n_cells: cells.len() };
    let mut s = String::new();
    let w = |s: &mut String, t: &str| s.push_str(t);
    w(&mut s, "<?xml version=\"1.0\"?>\n");
    w(&mut s, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    w(&mut s, "<UnstructuredGrid>\n");
    let _ = writeln!(s, "<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">", info.n_nodes, info.n_cells);
    w(&mut s, "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for p in &nodes {
        let _ = writeln!(s, "{:e} {:e} {:e}", p[0], p[1], p[2]);
    }
    w(&mut s, "</DataArray>\n</Points>\n<Cells>\n");
    w(&mut s, "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    for ch in conn.chunks(8) {
        let line: Vec<String> = ch.iter().map(|x| x.to_string()).collect();
        let _ = writeln!(s, "{}", line.join(" "));
    }
    w(&mut s, "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    for i in 0..cells.len() {
        let _ = writeln!(s, "{}", 8 * (i + 1));
    }
    w(&mut s, "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for _ in &cells {
        w(&mut s, "12\n");
    }
    w(&mut s, "</DataArray>\n</Cells>\n<CellData>\n");
    w(&mut s, "<DataArray type=\"Int32\" Name=\"MaterialIDs\" format=\"ascii\">\n");
    for &c in &cells {
        let _ = writeln!(s, "{}", material_ids.map_or(0, |m| m[c]));
    }
    w(&mut s, "</DataArray>\n<DataArray type=\"UInt64\" Name=\"bulk_cell_id\" format=\"ascii\">\n");
    for &c in &cells {
        let _ = writeln!(s, "{c}");
    }
    w(&mut s, "</DataArray>\n");
    for a in arrays {
        let _ = writeln!(
            s,
            "<DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"{}\" format=\"ascii\">",
            a.name, a.components
        );
        for &c in &cells {
            let v: Vec<String> = a.values[c * a.components..(c + 1) * a.components]
                .iter()
                .map(|x| format!("{x:e}"))
                .collect();
            let _ = writeln!(s, "{}", v.join(" "));
        }
        w(&mut s, "</DataArray>\n");
    }
    w(&mut s, "</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n");
    Ok((s, info))
}
