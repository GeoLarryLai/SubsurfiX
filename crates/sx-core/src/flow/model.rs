// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Cells and TPFA transmissibilities.

use std::collections::BTreeMap;

use crate::error::{SxError, SxResult};
use crate::grid::{CornerPointGrid, RegularGrid};

/// Options for [`FlowModel::from_corner_point_with`].
///
/// * `ntg`: net-to-gross per cell (0..=1). Pore volume and the horizontal
///   permeability (x, y) are multiplied by it, as in ECLIPSE-style simulators.
/// * `mult`: transmissibility multipliers per cell keyed by GRDECL name:
///   `MULTX`, `MULTY`, `MULTZ` act on the face towards +i/+j/+k, `MULTX-`,
///   `MULTY-`, `MULTZ-` on the face towards −i/−j/−k. A connection gets the
///   product of the multipliers of both cells on its shared face.
/// * `fault_tol_m`: depth mismatch (m) above which two shared face corners
///   count as displaced (a fault), and thickness at or below which a cell
///   counts as pinched out. Default 1e-3 m.
///
/// Faults: where neighbour columns have displaced faces, each active cell is
/// connected to every active cell of the neighbour column whose face
/// overlaps it on the shared pillars (non-neighbour connections); the
/// transmissibility uses the overlap area and centroid. Pinch-outs: an
/// active cell is connected vertically to the next active cell below when
/// only inactive zero-thickness cells lie between them.
#[derive(Debug, Clone)]
pub struct FlowOptions {
    /// Net-to-gross per cell.
    pub ntg: Option<Vec<f64>>,
    /// Transmissibility multipliers (MULTX, MULTY, MULTZ, MULTX-, MULTY-, MULTZ-).
    pub mult: BTreeMap<String, Vec<f64>>,
    /// Corner depth mismatch tolerance (m).
    pub fault_tol_m: f64,
}

impl Default for FlowOptions {
    fn default() -> Self {
        Self {
            ntg: None,
            mult: BTreeMap::new(),
            fault_tol_m: 1e-3,
        }
    }
}

/// GRDECL multiplier keywords understood by [`FlowOptions`].
pub const MULT_KEYWORDS: [&str; 6] = ["MULTX", "MULTY", "MULTZ", "MULTX-", "MULTY-", "MULTZ-"];

impl FlowOptions {
    /// Take `NTG` and the `MULT*` keywords from GRDECL properties
    /// (as returned by [`CornerPointGrid::from_grdecl`]).
    pub fn from_props(props: &BTreeMap<String, Vec<f64>>) -> Self {
        let mut o = Self {
            ntg: props.get("NTG").cloned(),
            ..Self::default()
        };
        for k in MULT_KEYWORDS {
            if let Some(v) = props.get(k) {
                o.mult.insert(k.to_string(), v.clone());
            }
        }
        o
    }
}

/// One cell-to-cell connection with its geometric transmissibility
/// `T = 1 / (1/T_a + 1/T_b)` (m³), mobility not included.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Connection {
    /// First cell.
    pub a: usize,
    /// Second cell.
    pub b: usize,
    /// Geometric transmissibility (m³).
    pub trans: f64,
    /// Half-cell geometric factors `|A·d|/|d|²` (m) of `a` and `b` (unit
    /// conductivity), used for heat conduction.
    pub geom: [f64; 2],
}

/// A flow model: active cells, pore volumes, connections.
#[derive(Debug, Clone)]
pub struct FlowModel {
    /// Grid dimensions (nx, ny, nz); cell index = i + nx (j + ny k).
    pub dims: [usize; 3],
    /// Active flag per cell (inactive cells carry no flow).
    pub active: Vec<bool>,
    /// Pore volume per cell (m³), net-to-gross applied.
    pub pore_volume: Vec<f64>,
    /// Bulk volume per cell (m³).
    pub bulk_volume: Vec<f64>,
    /// Cell centre (m).
    pub center: Vec<[f64; 3]>,
    /// Cell extent dx, dy, dz (m), used by Peaceman wells.
    pub extent: Vec<[f64; 3]>,
    /// Diagonal permeability kx, ky, kz (m²).
    pub perm: Vec<[f64; 3]>,
    /// Connections between active neighbours.
    pub connections: Vec<Connection>,
    /// Warnings raised while building.
    pub warnings: Vec<String>,
    /// Connections across fault throws (face overlap on shared pillars).
    pub n_fault_connections: usize,
    /// Connections across pinched-out (zero-thickness, inactive) cells.
    pub n_pinch_connections: usize,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm2(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
fn mean(ps: &[[f64; 3]]) -> [f64; 3] {
    let mut m = [0.0; 3];
    for p in ps {
        for d in 0..3 {
            m[d] += p[d] / ps.len() as f64;
        }
    }
    m
}

/// Corner ids of a cell face, in cyclic order: axis 0..3, side 0 (low) or 1 (high).
fn face_corners(axis: usize, side: usize) -> [usize; 4] {
    // corner c = di + 2 dj + 4 dk
    let bit = 1 << axis;
    let (o1, o2) = match axis {
        0 => (2, 4),
        1 => (1, 4),
        _ => (1, 2),
    };
    let base = if side == 1 { bit } else { 0 };
    [base, base + o1, base + o1 + o2, base + o2]
}

/// Half transmissibility of a cell towards a face: `|A·K·d| / |d|²`.
fn half_trans(
    corners: &[[f64; 8]; 3],
    axis: usize,
    side: usize,
    center: [f64; 3],
    k: [f64; 3],
) -> (f64, [f64; 3]) {
    let ids = face_corners(axis, side);
    let p: Vec<[f64; 3]> = ids
        .iter()
        .map(|&c| [corners[0][c], corners[1][c], corners[2][c]])
        .collect();
    let area = cross(sub(p[2], p[0]), sub(p[3], p[1]));
    let area = [0.5 * area[0], 0.5 * area[1], 0.5 * area[2]];
    let fc = mean(&p);
    let d = sub(fc, center);
    let akd = area[0] * k[0] * d[0] + area[1] * k[1] * d[1] + area[2] * k[2] * d[2];
    (akd.abs() / norm2(d).max(1e-300), fc)
}


/// Overlap of two lateral faces given as depths `[top0, top1, bot0, bot1]`
/// at the two shared pillars (s = 0, 1; depth positive down). Returns
/// `(∫ h ds, s̄, z̄)`: the overlap height integrated over s, and the
/// centroid in (s, z); `None` when they do not overlap.
fn face_overlap(a: [f64; 4], b: [f64; 4]) -> Option<(f64, f64, f64)> {
    let top = |s: f64| (a[0] + s * (a[1] - a[0])).max(b[0] + s * (b[1] - b[0]));
    let bot = |s: f64| (a[2] + s * (a[3] - a[2])).min(b[2] + s * (b[3] - b[2]));
    // Breakpoints: where the top lines cross, the bottom lines cross, or
    // any top line meets any bottom line.
    let mut br = vec![0.0, 1.0];
    let lines = |v: [f64; 4]| [(v[0], v[1] - v[0]), (v[2], v[3] - v[2])];
    let all: Vec<(f64, f64)> = lines(a).into_iter().chain(lines(b)).collect();
    for (i, p) in all.iter().enumerate() {
        for q in &all[i + 1..] {
            let dm = p.1 - q.1;
            if dm.abs() > 1e-300 {
                let s = (q.0 - p.0) / dm;
                if s > 0.0 && s < 1.0 {
                    br.push(s);
                }
            }
        }
    }
    br.sort_by(f64::total_cmp);
    let (mut area, mut ms, mut mz) = (0.0, 0.0, 0.0);
    for w in br.windows(2) {
        let (s0, s1) = (w[0], w[1]);
        let ds = s1 - s0;
        if ds <= 0.0 {
            continue;
        }
        // On each piece top, bot are linear; h = bot - top is linear.
        let (t0, t1, b0, b1) = (top(s0), top(s1), bot(s0), bot(s1));
        let (h0, h1) = (b0 - t0, b1 - t1);
        if h0 <= 0.0 && h1 <= 0.0 {
            continue;
        }
        let (m0, m1) = (0.5 * (t0 + b0), 0.5 * (t1 + b1));
        // Integrals of linear products over the piece.
        let int2 = |f0: f64, f1: f64, g0: f64, g1: f64| ds * (2.0 * f0 * g0 + 2.0 * f1 * g1 + f0 * g1 + f1 * g0) / 6.0;
        area += ds * 0.5 * (h0 + h1);
        ms += int2(s0, s1, h0, h1);
        mz += int2(m0, m1, h0, h1);
    }
    (area > 1e-12).then(|| (area, ms / area, mz / area))
}

impl FlowModel {
    /// Build from a regular grid. `perm` and `poro` have one entry per cell.
    ///
    /// # Errors
    /// `SxError::Shape` for wrong lengths; `InvalidInput` for negative values.
    pub fn from_regular(g: &RegularGrid, perm: &[[f64; 3]], poro: &[f64]) -> SxResult<Self> {
        Self::from_corner_point(&CornerPointGrid::from_regular(g), perm, poro)
    }

    /// Build from a corner-point grid (inactive cells per `actnum`, or zero
    /// porosity / permeability, carry no flow).
    ///
    /// # Errors
    /// `SxError::Shape` for wrong lengths; `InvalidInput` for negative or
    /// non-finite values.
    pub fn from_corner_point(
        g: &CornerPointGrid,
        perm: &[[f64; 3]],
        poro: &[f64],
    ) -> SxResult<Self> {
        Self::from_corner_point_with(g, perm, poro, &FlowOptions::default())
    }

    /// Build from a corner-point grid with net-to-gross, transmissibility
    /// multipliers and the fault / pinch-out guard (see [`FlowOptions`]).
    ///
    /// # Errors
    /// As [`Self::from_corner_point`]; `Shape` / `InvalidInput` for bad NTG or
    /// multipliers (wrong length, negative, NTG > 1, unknown keyword);
    /// (Faulted and pinched grids are connected, see [`FlowOptions`].)
    #[allow(clippy::needless_range_loop)] // axis indexes several per-axis tables
    pub fn from_corner_point_with(
        g: &CornerPointGrid,
        perm: &[[f64; 3]],
        poro: &[f64],
        opts: &FlowOptions,
    ) -> SxResult<Self> {
        let n = g.n_cells();
        if perm.len() != n || poro.len() != n {
            return Err(SxError::Shape(format!(
                "flow model: {n} cells, perm {}, poro {}",
                perm.len(),
                poro.len()
            )));
        }
        if perm
            .iter()
            .flatten()
            .chain(poro)
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(SxError::InvalidInput(
                "flow model: perm and poro must be finite and >= 0".into(),
            ));
        }
        let bad_vec = |what: &str, v: &[f64], max: f64| -> SxResult<()> {
            if v.len() != n {
                return Err(SxError::Shape(format!("flow model: {what} has {} values for {n} cells", v.len())));
            }
            if v.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > max) {
                return Err(SxError::InvalidInput(format!(
                    "flow model: {what} must be finite, >= 0{}",
                    if max.is_finite() { " and <= 1" } else { "" }
                )));
            }
            Ok(())
        };
        if let Some(v) = &opts.ntg {
            bad_vec("NTG", v, 1.0)?;
        }
        for (k, v) in &opts.mult {
            if !MULT_KEYWORDS.contains(&k.as_str()) {
                return Err(SxError::InvalidInput(format!(
                    "flow model: unknown multiplier {k:?} (use one of {MULT_KEYWORDS:?})"
                )));
            }
            bad_vec(k, v, f64::INFINITY)?;
        }
        if !(opts.fault_tol_m.is_finite() && opts.fault_tol_m >= 0.0) {
            return Err(SxError::InvalidInput("flow model: fault_tol_m must be finite and >= 0".into()));
        }
        let ntg = |c: usize| opts.ntg.as_ref().map_or(1.0, |v| v[c]);
        let perm: Vec<[f64; 3]> = perm
            .iter()
            .enumerate()
            .map(|(c, k)| [k[0] * ntg(c), k[1] * ntg(c), k[2]])
            .collect();
        let poro: Vec<f64> = poro.iter().enumerate().map(|(c, p)| p * ntg(c)).collect();
        let mult = |key: &str, c: usize| opts.mult.get(key).map_or(1.0, |v| v[c]);
        let [nx, ny, nz] = g.dims;
        let mut corners = Vec::with_capacity(n);
        let mut center = Vec::with_capacity(n);
        let mut extent = Vec::with_capacity(n);
        let mut pv = Vec::with_capacity(n);
        let mut bulk = Vec::with_capacity(n);
        let mut active = Vec::with_capacity(n);
        for c in 0..n {
            let cs = g.corners(c);
            let mut t = [[0.0; 8]; 3];
            for (q, p) in cs.iter().enumerate() {
                for d in 0..3 {
                    t[d][q] = p[d];
                }
            }
            let ctr = mean(&cs);
            let mut ext = [0.0; 3];
            for (axis, e) in ext.iter_mut().enumerate() {
                let lo: Vec<[f64; 3]> = face_corners(axis, 0).iter().map(|&q| cs[q]).collect();
                let hi: Vec<[f64; 3]> = face_corners(axis, 1).iter().map(|&q| cs[q]).collect();
                *e = norm2(sub(mean(&hi), mean(&lo))).sqrt();
            }
            let vol = g.volume(c);
            let act = g.actnum[c] && poro[c] > 0.0 && perm[c].iter().any(|&k| k > 0.0) && vol > 0.0;
            pv.push(if act { vol * poro[c] } else { 0.0 });
            bulk.push(vol.max(0.0));
            active.push(act);
            corners.push(t);
            center.push(ctr);
            extent.push(ext);
        }
        let mut connections = Vec::new();
        let cell = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
        // Lateral face (axis 0 or 1, side) of a cell as (top0, top1, bot0, bot1)
        // depths at the two pillars, s = 0 and s = 1.
        let face_z = |c: usize, axis: usize, side: usize| -> [f64; 4] {
            let sbit = if axis == 0 { 2 } else { 1 };
            let base = if side == 1 { 1 << axis } else { 0 };
            let z = &corners[c][2];
            [z[base], z[base + sbit], z[base + 4], z[base + sbit + 4]]
        };
        // Column pairs with a throw anywhere get overlap (fault) connections.
        let mut faulted = vec![[false; 2]; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                for axis in 0..2 {
                    let (ni, nj) = if axis == 0 { (i + 1, j) } else { (i, j + 1) };
                    if ni >= nx || nj >= ny {
                        continue;
                    }
                    faulted[i + nx * j][axis] = (0..nz).any(|k| {
                        let za = face_z(cell(i, j, k), axis, 1);
                        let zb = face_z(cell(ni, nj, k), axis, 0);
                        za.iter().zip(zb).any(|(p, q)| (p - q).abs() > opts.fault_tol_m)
                    });
                }
            }
        }
        let push = |connections: &mut Vec<Connection>, a: usize, b: usize, axis: usize, h: [(f64, f64); 2]| {
            let m = mult(MULT_KEYWORDS[axis], a) * mult(MULT_KEYWORDS[axis + 3], b);
            let ((ta, ga), (tb, gb)) = (h[0], h[1]);
            if ta > 0.0 && tb > 0.0 && m > 0.0 {
                connections.push(Connection {
                    a,
                    b,
                    trans: m / (1.0 / ta + 1.0 / tb),
                    geom: [ga, gb],
                });
            }
        };
        let logical = |a: usize, b: usize, axis: usize| -> [(f64, f64); 2] {
            [
                (
                    half_trans(&corners[a], axis, 1, center[a], perm[a]).0,
                    half_trans(&corners[a], axis, 1, center[a], [1.0; 3]).0,
                ),
                (
                    half_trans(&corners[b], axis, 0, center[b], perm[b]).0,
                    half_trans(&corners[b], axis, 0, center[b], [1.0; 3]).0,
                ),
            ]
        };
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let a = cell(i, j, k);
                    if !active[a] {
                        continue;
                    }
                    for axis in 0..3 {
                        let (ni, nj, nk) = match axis {
                            0 => (i + 1, j, k),
                            1 => (i, j + 1, k),
                            _ => (i, j, k + 1),
                        };
                        if ni >= nx || nj >= ny || nk >= nz || (axis < 2 && faulted[i + nx * j][axis]) {
                            continue;
                        }
                        let b = cell(ni, nj, nk);
                        if active[b] {
                            push(&mut connections, a, b, axis, logical(a, b, axis));
                        }
                    }
                }
            }
        }
        // Fault (non-neighbour) connections: every active pair across a
        // faulted column pair whose faces overlap on the shared pillars.
        let mut n_fault = 0;
        for j in 0..ny {
            for i in 0..nx {
                for axis in 0..2 {
                    if !faulted[i + nx * j][axis] {
                        continue;
                    }
                    let (ni, nj) = if axis == 0 { (i + 1, j) } else { (i, j + 1) };
                    for ka in 0..nz {
                        let a = cell(i, j, ka);
                        if !active[a] {
                            continue;
                        }
                        let za = face_z(a, axis, 1);
                        for kb in 0..nz {
                            let b = cell(ni, nj, kb);
                            if !active[b] {
                                continue;
                            }
                            let Some((hint, s_bar, z_bar)) = face_overlap(za, face_z(b, axis, 0)) else {
                                continue;
                            };
                            // Physical overlap: pillar positions from cell a's face.
                            let ids = face_corners(axis, 1);
                            let sbit = if axis == 0 { 2 } else { 1 };
                            let base = 1 << axis;
                            let xy = |q: usize| {
                                let c = &corners[a];
                                [0.5 * (c[0][q] + c[0][q + 4]), 0.5 * (c[1][q] + c[1][q + 4])]
                            };
                            let (p0, p1) = (xy(base), xy(base + sbit));
                            let len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
                            let pts: Vec<[f64; 3]> =
                                ids.iter().map(|&q| [corners[a][0][q], corners[a][1][q], corners[a][2][q]]).collect();
                            let nrm = cross(sub(pts[2], pts[0]), sub(pts[3], pts[1]));
                            let nl = norm2(nrm).sqrt();
                            if nl <= 0.0 || len <= 0.0 {
                                continue;
                            }
                            let area = len * hint;
                            let av = [nrm[0] / nl * area, nrm[1] / nl * area, nrm[2] / nl * area];
                            let fc = [p0[0] + s_bar * (p1[0] - p0[0]), p0[1] + s_bar * (p1[1] - p0[1]), z_bar];
                            let ht = |c: usize, k: [f64; 3]| {
                                let d = sub(fc, center[c]);
                                let akd = av[0] * k[0] * d[0] + av[1] * k[1] * d[1] + av[2] * k[2] * d[2];
                                akd.abs() / norm2(d).max(1e-300)
                            };
                            let before = connections.len();
                            push(
                                &mut connections,
                                a,
                                b,
                                axis,
                                [(ht(a, perm[a]), ht(a, [1.0; 3])), (ht(b, perm[b]), ht(b, [1.0; 3]))],
                            );
                            n_fault += connections.len() - before;
                        }
                    }
                }
            }
        }
        // Pinch-out connections: active cells separated only by zero-thickness cells.
        let thin = |c: usize| {
            let t = &corners[c][2];
            (0..4).all(|q| (t[q + 4] - t[q]).abs() <= opts.fault_tol_m)
        };
        let mut n_pinch = 0;
        for j in 0..ny {
            for i in 0..nx {
                let mut k = 0;
                while k < nz {
                    let a = cell(i, j, k);
                    if !active[a] {
                        k += 1;
                        continue;
                    }
                    let mut k2 = k + 1;
                    while k2 < nz && !active[cell(i, j, k2)] && thin(cell(i, j, k2)) {
                        k2 += 1;
                    }
                    if k2 > k + 1 && k2 < nz && active[cell(i, j, k2)] {
                        let b = cell(i, j, k2);
                        let before = connections.len();
                        push(&mut connections, a, b, 2, logical(a, b, 2));
                        n_pinch += connections.len() - before;
                    }
                    k = k2.max(k + 1);
                }
            }
        }
        let warnings = Vec::new();
        Ok(Self {
            dims: g.dims,
            active,
            pore_volume: pv,
            bulk_volume: bulk,
            center,
            extent,
            perm,
            connections,
            warnings,
            n_fault_connections: n_fault,
            n_pinch_connections: n_pinch,
        })
    }

    /// Number of cells (active or not).
    pub fn n_cells(&self) -> usize {
        self.active.len()
    }

    /// Half-cell geometric factor (m) from a cell centre to its boundary face
    /// on `axis`, assuming a box cell: `A / (d/2)`. Multiply by a
    /// conductivity (W/(m·K)) for a boundary conductance (W/K).
    pub fn boundary_geom(&self, cell: usize, axis: usize) -> f64 {
        let e = self.extent[cell];
        let a = axis.min(2);
        let area = match a {
            0 => e[1] * e[2],
            1 => e[0] * e[2],
            _ => e[0] * e[1],
        };
        2.0 * area / e[a]
    }

    /// Half transmissibility (m³) from a cell centre to its boundary face on
    /// `axis` (0 x, 1 y, 2 z), assuming a box cell: `k A / (d/2)`.
    pub fn boundary_trans(&self, cell: usize, axis: usize) -> f64 {
        let e = self.extent[cell];
        let area = match axis {
            0 => e[1] * e[2],
            1 => e[0] * e[2],
            _ => e[0] * e[1],
        };
        2.0 * self.perm[cell][axis.min(2)] * area / e[axis.min(2)]
    }
}

/// One completed cell of a deviated well: cell, projected path lengths
/// `[lx, ly, lz]` (m) inside it, Peaceman well index (m³).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Completion {
    /// Cell index.
    pub cell: usize,
    /// |Δx|, |Δy|, |Δz| of the path inside the cell (m).
    pub length: [f64; 3],
    /// Well index (m³) from [`super::pressure::peaceman_wi_segment`].
    pub wi: f64,
}

impl FlowModel {
    /// Cell containing point `q` (box test on centre ± extent/2), if any.
    fn locate(&self, q: [f64; 3], cand: &[usize]) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        for &c in cand {
            let (ctr, e) = (self.center[c], self.extent[c]);
            let mut worst = 0.0f64;
            for d in 0..3 {
                worst = worst.max((q[d] - ctr[d]).abs() / (0.5 * e[d]).max(1e-12));
            }
            if worst <= 1.0 + 1e-9 && best.is_none_or(|(w, _)| worst < w) {
                best = Some((worst, c));
            }
        }
        best.map(|(_, c)| c)
    }

    /// Completions of a deviated well along a polyline `path` (x, y, depth in
    /// m): the path is walked in short steps (1/50 of the smallest cell size),
    /// each step's |Δx|, |Δy|, |Δz| is added to the active cell it falls in,
    /// and each cell gets a Peaceman index from those projected lengths.
    /// Cells are assumed close to boxes (centre ± extent/2); path parts
    /// outside the active grid are skipped.
    ///
    /// # Errors
    /// `InvalidInput` for fewer than 2 points, a non-positive radius or a path
    /// that misses every active cell.
    pub fn trajectory_completions(
        &self,
        path: &[[f64; 3]],
        rw: f64,
        skin: f64,
    ) -> SxResult<Vec<Completion>> {
        if path.len() < 2 || !(rw > 0.0) || path.iter().flatten().any(|v| !v.is_finite()) {
            return Err(SxError::InvalidInput(
                "trajectory: need >= 2 finite points and a radius > 0".into(),
            ));
        }
        // xy buckets of active cells.
        let act: Vec<usize> = (0..self.n_cells()).filter(|&c| self.active[c]).collect();
        let hmin = act
            .iter()
            .flat_map(|&c| self.extent[c])
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        let bx = act
            .iter()
            .map(|&c| self.extent[c][0].max(self.extent[c][1]))
            .fold(0.0, f64::max)
            .max(1e-9);
        let key = |x: f64, y: f64| ((x / bx).floor() as i64, (y / bx).floor() as i64);
        let mut buckets: std::collections::HashMap<(i64, i64), Vec<usize>> =
            std::collections::HashMap::new();
        for &c in &act {
            let (x, y) = (self.center[c][0], self.center[c][1]);
            buckets.entry(key(x, y)).or_default().push(c);
        }
        let near = |q: [f64; 3]| -> Vec<usize> {
            let (i, j) = key(q[0], q[1]);
            let mut v = Vec::new();
            for di in -1..=1 {
                for dj in -1..=1 {
                    if let Some(b) = buckets.get(&(i + di, j + dj)) {
                        v.extend_from_slice(b);
                    }
                }
            }
            v
        };
        let step = hmin / 50.0;
        let mut acc: Vec<(usize, [f64; 3])> = Vec::new();
        for seg in path.windows(2) {
            let d = sub(seg[1], seg[0]);
            let len = norm2(d).sqrt();
            let ns = (len / step).ceil().max(1.0) as usize;
            for s in 0..ns {
                let t = (s as f64 + 0.5) / ns as f64;
                let q = [
                    seg[0][0] + t * d[0],
                    seg[0][1] + t * d[1],
                    seg[0][2] + t * d[2],
                ];
                let Some(c) = self.locate(q, &near(q)) else {
                    continue;
                };
                let inc = [
                    d[0].abs() / ns as f64,
                    d[1].abs() / ns as f64,
                    d[2].abs() / ns as f64,
                ];
                match acc.iter_mut().find(|(cc, _)| *cc == c) {
                    Some((_, l)) => (0..3).for_each(|k| l[k] += inc[k]),
                    None => acc.push((c, inc)),
                }
            }
        }
        if acc.is_empty() {
            return Err(SxError::InvalidInput(
                "trajectory: path misses every active cell".into(),
            ));
        }
        Ok(acc
            .into_iter()
            .map(|(cell, length)| Completion {
                cell,
                length,
                wi: super::pressure::peaceman_wi_segment(
                    self.perm[cell],
                    self.extent[cell],
                    length,
                    rw,
                    skin,
                ),
            })
            .collect())
    }
}
