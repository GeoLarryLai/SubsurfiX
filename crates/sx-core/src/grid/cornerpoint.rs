// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Corner-point grid (Eclipse / GRDECL layout): pillars (`COORD`), corner
//! depths (`ZCORN`) and active flags (`ACTNUM`).
//!
//! - `COORD`: `(nx+1)·(ny+1)` pillars, each `x1 y1 z1 x2 y2 z2` (top, bottom),
//!   pillar index `pi + (nx+1)·pj`.
//! - `ZCORN`: `8·nx·ny·nz` depths; z grows downward. Index
//!   `(2i+di) + 2nx·((2j+dj) + 2ny·(2k+dk))` for corner bits `di, dj, dk`.
//! - Cell index = `i + nx·(j + ny·k)`; corner `c = di + 2·dj + 4·dk`.
//!
//! Faults are whatever `ZCORN` encodes (corners of neighbouring cells may
//! differ). Not supported: `MAPAXES`, `COORDSYS`, local grid
//! refinement (LGR), `DXV`/`TOPS` block-centred input, and pinch-out
//! handling (`MINPV`/`PINCH`) are not supported; [`CornerPointGrid::locate`]
//! assumes cells are not self-intersecting.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::error::{SxError, SxResult};
use crate::grid::regular::RegularGrid;
use crate::grid::upscale::CellLocator;

/// Corner-point grid.
#[derive(Debug, Clone, PartialEq)]
pub struct CornerPointGrid {
    /// Cells along x, y, z.
    pub dims: [usize; 3],
    /// Pillars, `6·(nx+1)·(ny+1)` values.
    pub coord: Vec<f64>,
    /// Corner depths, `8·nx·ny·nz` values.
    pub zcorn: Vec<f64>,
    /// Active flags, `nx·ny·nz` values.
    pub actnum: Vec<bool>,
    /// Per column (`i + nx·j`): x/y bounding box `[xmin, xmax, ymin, ymax]`.
    col_box: Vec<[f64; 4]>,
}

/// Tetrahedra of the hexahedron (Kuhn split along the 0–7 diagonal).
const TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

fn tet_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]))
        / 6.0
}

/// Barycentric containment with a relative tolerance.
fn in_tet(p: [f64; 3], t: [[f64; 3]; 4]) -> bool {
    let v = tet_volume(t[0], t[1], t[2], t[3]);
    if v.abs() < 1e-14 {
        return false;
    }
    let tol = -1e-9 * v.abs();
    let s = v.signum();
    [
        tet_volume(p, t[1], t[2], t[3]),
        tet_volume(t[0], p, t[2], t[3]),
        tet_volume(t[0], t[1], p, t[3]),
        tet_volume(t[0], t[1], t[2], p),
    ]
    .iter()
    .all(|x| x * s >= tol)
}

impl CornerPointGrid {
    /// Build from raw arrays (checks lengths; `actnum` `None` = all active).
    ///
    /// # Errors
    /// [`SxError::Shape`] on wrong lengths or zero dims; [`SxError::InvalidInput`]
    /// on non-finite `COORD`/`ZCORN` or a horizontal pillar with distinct depths.
    pub fn new(
        dims: [usize; 3],
        coord: Vec<f64>,
        zcorn: Vec<f64>,
        actnum: Option<Vec<bool>>,
    ) -> SxResult<Self> {
        let [nx, ny, nz] = dims;
        if nx * ny * nz == 0 {
            return Err(SxError::Shape(format!("corner-point dims {dims:?}")));
        }
        if coord.len() != 6 * (nx + 1) * (ny + 1) || zcorn.len() != 8 * nx * ny * nz {
            return Err(SxError::Shape(format!(
                "COORD has {} (want {}), ZCORN has {} (want {})",
                coord.len(),
                6 * (nx + 1) * (ny + 1),
                zcorn.len(),
                8 * nx * ny * nz
            )));
        }
        if coord.iter().chain(&zcorn).any(|v| !v.is_finite()) {
            return Err(SxError::InvalidInput("COORD/ZCORN must be finite".into()));
        }
        let actnum = actnum.unwrap_or_else(|| vec![true; nx * ny * nz]);
        if actnum.len() != nx * ny * nz {
            return Err(SxError::Shape(format!(
                "ACTNUM has {} values, want {}",
                actnum.len(),
                nx * ny * nz
            )));
        }
        let mut g = Self {
            dims,
            coord,
            zcorn,
            actnum,
            col_box: vec![],
        };
        g.col_box = (0..nx * ny)
            .map(|c| {
                let (i, j) = (c % nx, c / nx);
                let mut b = [
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ];
                for k in 0..nz {
                    for p in g.corners(g.index(i, j, k)) {
                        b = [
                            b[0].min(p[0]),
                            b[1].max(p[0]),
                            b[2].min(p[1]),
                            b[3].max(p[1]),
                        ];
                    }
                }
                b
            })
            .collect();
        Ok(g)
    }

    /// Corner-point copy of a regular grid (vertical pillars).
    pub fn from_regular(r: &RegularGrid) -> Self {
        let [nx, ny, _] = r.dims;
        let xs: Vec<f64> = (0..=nx)
            .map(|i| r.origin[0] + i as f64 * r.cell[0])
            .collect();
        let ys: Vec<f64> = (0..=ny)
            .map(|j| r.origin[1] + j as f64 * r.cell[1])
            .collect();
        let zs: Vec<f64> = (0..=r.dims[2])
            .map(|k| r.origin[2] + k as f64 * r.cell[2])
            .collect();
        let surfaces: Vec<Vec<f64>> = zs.iter().map(|&z| vec![z; (nx + 1) * (ny + 1)]).collect();
        Self::from_surfaces(&xs, &ys, &surfaces)
            .expect("a valid regular grid gives a valid corner-point grid")
    }

    /// Unfaulted grid with vertical pillars at node coordinates `xs` × `ys`
    /// and `nz + 1` layer surfaces, each holding depths at the
    /// `(nx+1)·(ny+1)` nodes (index `pi + (nx+1)·pj`), top first. Layers
    /// with negative thickness are clamped to zero (pinched, set inactive).
    ///
    /// # Errors
    /// [`SxError::Shape`] on wrong sizes; [`SxError::InvalidInput`] if `xs` or
    /// `ys` is not strictly increasing.
    pub fn from_surfaces(xs: &[f64], ys: &[f64], surfaces: &[Vec<f64>]) -> SxResult<Self> {
        if xs.len() < 2 || ys.len() < 2 || surfaces.len() < 2 {
            return Err(SxError::Shape(
                "need ≥ 2 node columns, rows and surfaces".into(),
            ));
        }
        if xs.windows(2).any(|w| !(w[1] > w[0])) || ys.windows(2).any(|w| !(w[1] > w[0])) {
            return Err(SxError::InvalidInput(
                "node coordinates must increase".into(),
            ));
        }
        let (nx, ny, nz) = (xs.len() - 1, ys.len() - 1, surfaces.len() - 1);
        let nn = (nx + 1) * (ny + 1);
        if surfaces.iter().any(|s| s.len() != nn) {
            return Err(SxError::Shape(format!(
                "each surface needs {nn} node depths"
            )));
        }
        // Monotone surfaces: clamp each below the one above.
        let mut surf: Vec<Vec<f64>> = surfaces.to_vec();
        for k in 1..=nz {
            let (above, below) = surf.split_at_mut(k);
            for (z, &up) in below[0].iter_mut().zip(&above[k - 1]).take(nn) {
                *z = z.max(up);
            }
        }
        let (ztop, zbot) = (
            surf[0].iter().cloned().fold(f64::INFINITY, f64::min),
            surf[nz].iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let mut coord = Vec::with_capacity(6 * nn);
        for &y in ys {
            for &x in xs {
                coord.extend([x, y, ztop, x, y, zbot.max(ztop + 1.0)]);
            }
        }
        let mut zcorn = vec![0.0; 8 * nx * ny * nz];
        let mut act = vec![true; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut thick = 0.0f64;
                    for c in 0..8 {
                        let (di, dj, dk) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                        let z = surf[k + dk][(i + di) + (nx + 1) * (j + dj)];
                        zcorn[Self::zidx(nx, ny, i, j, k, c)] = z;
                        if dk == 1 {
                            thick = thick.max(z - surf[k][(i + di) + (nx + 1) * (j + dj)]);
                        }
                    }
                    act[i + nx * (j + ny * k)] = thick > 0.0;
                }
            }
        }
        Self::new([nx, ny, nz], coord, zcorn, Some(act))
    }

    fn zidx(nx: usize, ny: usize, i: usize, j: usize, k: usize, c: usize) -> usize {
        let (di, dj, dk) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
        (2 * i + di) + 2 * nx * ((2 * j + dj) + 2 * ny * (2 * k + dk))
    }

    /// Number of cells.
    pub fn n_cells(&self) -> usize {
        self.dims.iter().product()
    }

    /// Linear index of `(i, j, k)`.
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// `(i, j, k)` of a linear index.
    pub fn ijk(&self, idx: usize) -> (usize, usize, usize) {
        let [nx, ny, _] = self.dims;
        (idx % nx, (idx / nx) % ny, idx / (nx * ny))
    }

    /// The 8 corners of a cell, `c = di + 2·dj + 4·dk`. x, y lie on the
    /// pillar at the corner depth (linear along the pillar).
    pub fn corners(&self, idx: usize) -> [[f64; 3]; 8] {
        let (i, j, k) = self.ijk(idx);
        let [nx, ny, _] = self.dims;
        let mut out = [[0.0; 3]; 8];
        for (c, o) in out.iter_mut().enumerate() {
            let (di, dj) = (c & 1, (c >> 1) & 1);
            let z = self.zcorn[Self::zidx(nx, ny, i, j, k, c)];
            let p = 6 * ((i + di) + (nx + 1) * (j + dj));
            let pl = &self.coord[p..p + 6];
            let dz = pl[5] - pl[2];
            let t = if dz.abs() > 1e-12 {
                (z - pl[2]) / dz
            } else {
                0.0
            };
            *o = [pl[0] + t * (pl[3] - pl[0]), pl[1] + t * (pl[4] - pl[1]), z];
        }
        out
    }

    /// Cell bulk volume (six-tetrahedron split; exact for planar faces).
    pub fn volume(&self, idx: usize) -> f64 {
        let c = self.corners(idx);
        TETS.iter()
            .map(|t| tet_volume(c[t[0]], c[t[1]], c[t[2]], c[t[3]]).abs())
            .sum()
    }

    /// Cell centre: mean of the 8 corners.
    pub fn center(&self, idx: usize) -> [f64; 3] {
        let c = self.corners(idx);
        let mut m = [0.0; 3];
        for p in c {
            for a in 0..3 {
                m[a] += p[a] / 8.0;
            }
        }
        m
    }

    /// Active cell containing `p`; `None` outside every active cell or for NaN.
    /// Column search by bounding box, then exact tetrahedron tests.
    pub fn locate(&self, p: [f64; 3]) -> Option<usize> {
        if p.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let [nx, ny, nz] = self.dims;
        for col in 0..nx * ny {
            let b = self.col_box[col];
            if p[0] < b[0] || p[0] > b[1] || p[1] < b[2] || p[1] > b[3] {
                continue;
            }
            for k in 0..nz {
                let idx = col + nx * ny * k;
                if !self.actnum[idx] {
                    continue;
                }
                let c = self.corners(idx);
                let (zmin, zmax) = c
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), q| {
                        (a.min(q[2]), b.max(q[2]))
                    });
                if p[2] < zmin || p[2] > zmax {
                    continue;
                }
                if TETS
                    .iter()
                    .any(|t| in_tet(p, [c[t[0]], c[t[1]], c[t[2]], c[t[3]]]))
                {
                    return Some(idx);
                }
            }
        }
        None
    }

    /// Write GRDECL text: `SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`, then each
    /// property (cell order, one value per cell) under its keyword.
    ///
    /// # Errors
    /// [`SxError::Shape`] if a property has the wrong length.
    pub fn to_grdecl(&self, props: &BTreeMap<String, Vec<f64>>) -> SxResult<String> {
        let [nx, ny, nz] = self.dims;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "-- Written by SubsurfiX\nSPECGRID\n {nx} {ny} {nz} 1 F /\n"
        );
        let block = |s: &mut String, kw: &str, vals: &mut dyn Iterator<Item = String>| {
            let _ = writeln!(s, "{kw}");
            for (n, v) in vals.enumerate() {
                s.push_str(&v);
                s.push(if n % 6 == 5 { '\n' } else { ' ' });
            }
            s.push_str("\n/\n\n");
        };
        block(
            &mut s,
            "COORD",
            &mut self.coord.iter().map(|v| format!("{v}")),
        );
        block(
            &mut s,
            "ZCORN",
            &mut self.zcorn.iter().map(|v| format!("{v}")),
        );
        block(
            &mut s,
            "ACTNUM",
            &mut self
                .actnum
                .iter()
                .map(|&a| if a { "1".into() } else { "0".into() }),
        );
        for (k, v) in props {
            if v.len() != self.n_cells() {
                return Err(SxError::Shape(format!(
                    "property {k} has {} values, want {}",
                    v.len(),
                    self.n_cells()
                )));
            }
            block(
                &mut s,
                &k.to_ascii_uppercase(),
                &mut v.iter().map(|x| format!("{x}")),
            );
        }
        Ok(s)
    }

    /// Parse GRDECL text (`SPECGRID` or `DIMENS`, `COORD`, `ZCORN`, optional
    /// `ACTNUM`, and any other numeric keyword with one value per cell as a
    /// property). Handles `--` comments and `n*value` repeats. Unknown
    /// keywords with other lengths are skipped.
    ///
    /// # Errors
    /// [`SxError::Format`] on missing keywords or bad numbers; errors of [`Self::new`].
    pub fn from_grdecl(text: &str) -> SxResult<(Self, BTreeMap<String, Vec<f64>>)> {
        let mut kws: Vec<(String, Vec<String>)> = vec![];
        let mut open = false;
        for line in text.lines() {
            let line = line.split("--").next().unwrap_or("");
            for tok in line.split_whitespace() {
                if !open {
                    if tok.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                        kws.push((tok.to_ascii_uppercase(), vec![]));
                        open = !matches!(tok.to_ascii_uppercase().as_str(), "ECHO" | "NOECHO");
                    }
                    continue;
                }
                let (body, end) = match tok.strip_suffix('/') {
                    Some(b) => (b, true),
                    None => (tok, false),
                };
                if !body.is_empty() {
                    if let Some(k) = kws.last_mut() {
                        k.1.push(body.to_string());
                    }
                }
                if end {
                    open = false;
                }
            }
        }
        let expand = |toks: &[String]| -> SxResult<Vec<f64>> {
            let mut out = vec![];
            for t in toks {
                let bad = || SxError::Format(format!("GRDECL: bad number {t:?}"));
                if let Some((n, v)) = t.split_once('*') {
                    let n: usize = n.parse().map_err(|_| bad())?;
                    let v: f64 = v.parse().map_err(|_| bad())?;
                    out.extend(std::iter::repeat_n(v, n));
                } else {
                    out.push(t.parse().map_err(|_| bad())?);
                }
            }
            Ok(out)
        };
        let get = |k: &str| kws.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_slice());
        let dims_t = get("SPECGRID")
            .or_else(|| get("DIMENS"))
            .ok_or_else(|| SxError::Format("GRDECL: no SPECGRID/DIMENS".into()))?;
        let d: Vec<usize> = dims_t
            .iter()
            .take(3)
            .map(|t| {
                t.parse()
                    .map_err(|_| SxError::Format(format!("GRDECL: bad dims {t:?}")))
            })
            .collect::<SxResult<_>>()?;
        if d.len() != 3 {
            return Err(SxError::Format("GRDECL: SPECGRID needs nx ny nz".into()));
        }
        let coord =
            expand(get("COORD").ok_or_else(|| SxError::Format("GRDECL: no COORD".into()))?)?;
        let zcorn =
            expand(get("ZCORN").ok_or_else(|| SxError::Format("GRDECL: no ZCORN".into()))?)?;
        let act = get("ACTNUM")
            .map(expand)
            .transpose()?
            .map(|v| v.into_iter().map(|x| x != 0.0).collect());
        let g = Self::new([d[0], d[1], d[2]], coord, zcorn, act)?;
        let mut props = BTreeMap::new();
        for (k, v) in &kws {
            if matches!(
                k.as_str(),
                "SPECGRID" | "DIMENS" | "COORD" | "ZCORN" | "ACTNUM"
            ) {
                continue;
            }
            if let Ok(vals) = expand(v) {
                if vals.len() == g.n_cells() {
                    props.insert(k.clone(), vals);
                }
            }
        }
        Ok((g, props))
    }
}

impl CellLocator for CornerPointGrid {
    fn locate_cell(&self, p: [f64; 3]) -> Option<usize> {
        self.locate(p)
    }
}
