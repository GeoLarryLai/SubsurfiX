// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//
// Partly derived from MRST, The MATLAB Reservoir Simulation Toolbox
// (co2lab-ve `addVERelpermSharpInterface.m`, "simple" sharp-interface
// relative permeabilities). Copyright 2009-2026 SINTEF Digital,
// Mathematics & Cybernetics. MRST is GPL-3.0-or-later; this derived code
// is distributed here under AGPL-3.0-or-later (GPLv3 section 13).
//! **Experimental** CO2 vertical-equilibrium (VE) sharp-interface model on a
//! 2D top-surface grid, with structural (spill-point) and residual trapping.
//!
//! Formulation (Nilsen et al. 2011, 2016; Nordbotten & Celia 2012):
//! CO2 sits under the caprock as a layer of thickness `h` with saturation
//! `1 − s_wr`; below the current plume and above the historical maximum
//! `h_max` CO2 is residual at `s_gr`; brine fills the rest of the column
//! `H`. Upscaled relative permeabilities (MRST co2lab "simple" model):
//! `kr_g = (1 − s_wr) h / H`, `kr_w = (H − h_max + (1 − s_gr)(h_max − h)) / H`.
//! Pressure `p` is the CO2 pressure at the caprock (depth `z_t`, positive
//! down); phase potentials `Φ_g = p − ρ_g g z_t`,
//! `Φ_w = p − ρ_w g z_t − Δρ g h` (no capillary pressure, hydrostatic in
//! each column).
//!
//! Time stepping: IMPES. Each step solves the incompressible total-flux
//! pressure equation (TPFA, upwinded mobilities from the previous step,
//! Jacobi-preconditioned conjugate gradients), then updates the CO2 volume
//! explicitly with upwind CO2 fluxes under a CFL limit. The CO2 volume per
//! column is the conserved variable; `h`, `h_max` follow from it, so the
//! CO2 balance (in place + out through open edges = injected) closes to
//! round-off. Densities and viscosities are constant per run.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{SxError, SxResult};
use crate::grid::CornerPointGrid;

/// Gravity (m/s²).
pub const G: f64 = 9.80665;

/// Top-surface (2D) grid for VE runs.
#[derive(Debug, Clone)]
pub struct VeGrid {
    /// Caprock depth per column (m, positive down).
    pub top: Vec<f64>,
    /// Aquifer thickness per column (m).
    pub height: Vec<f64>,
    /// Porosity per column (pore-volume weighted).
    pub poro: Vec<f64>,
    /// Horizontal permeability per column (m², thickness weighted).
    pub perm: Vec<f64>,
    /// Plan-view area per column (m²).
    pub area: Vec<f64>,
    /// Column centre x, y (m).
    pub center: Vec<[f64; 2]>,
    /// Interior connections `(a, b, T)`, T = harmonic of `k H L / d` (m³).
    pub conns: Vec<(usize, usize, f64)>,
    /// Outer faces `(cell, T_half)` (m³).
    pub outer: Vec<(usize, f64)>,
    /// Column index (i + nx j) of each VE cell in the source grid.
    pub column: Vec<usize>,
    /// Source grid columns (nx, ny).
    pub dims: [usize; 2],
}

fn half(k: f64, h: f64, len: f64, d: f64) -> f64 {
    k * h * len / d.max(1e-12)
}

impl VeGrid {
    /// Cartesian top surface (a direction with a single cell is a
    /// cross-section: its side faces are closed): node coordinates `xs` (nx+1), `ys` (ny+1);
    /// per cell (i + nx j) caprock depth, thickness, porosity and
    /// horizontal permeability. Cells with thickness or porosity 0 are dropped.
    ///
    /// # Errors
    /// `Shape` / `InvalidInput` on bad input.
    pub fn from_cartesian(xs: &[f64], ys: &[f64], top: &[f64], height: &[f64], poro: &[f64], perm: &[f64]) -> SxResult<Self> {
        if xs.len() < 2 || ys.len() < 2 || xs.windows(2).any(|w| !(w[1] > w[0])) || ys.windows(2).any(|w| !(w[1] > w[0])) {
            return Err(SxError::InvalidInput("ve: xs, ys need >= 2 increasing values".into()));
        }
        let (nx, ny) = (xs.len() - 1, ys.len() - 1);
        let n = nx * ny;
        if [top.len(), height.len(), poro.len(), perm.len()].iter().any(|&l| l != n) {
            return Err(SxError::Shape(format!("ve: need {n} values per array")));
        }
        let cols: Vec<Col> = (0..n)
            .map(|c| {
                let (i, j) = (c % nx, c / nx);
                Col {
                    top: top[c],
                    h: height[c],
                    poro: poro[c],
                    perm: perm[c],
                    xy: [[xs[i], ys[j]], [xs[i + 1], ys[j]], [xs[i], ys[j + 1]], [xs[i + 1], ys[j + 1]]],
                }
            })
            .collect();
        Self::build(nx, ny, cols)
    }

    /// Top surface of a corner-point grid: per column, caprock = top of the
    /// first active cell, base = bottom of the last active cell; porosity
    /// pore-volume weighted, permeability (kx) thickness weighted. Lateral
    /// connections follow logical (i, j) neighbours; fault throws are not
    /// represented in 2D beyond the caprock depth difference.
    ///
    /// # Errors
    /// `Shape` on wrong lengths.
    pub fn from_corner_point(g: &CornerPointGrid, perm: &[[f64; 3]], poro: &[f64]) -> SxResult<Self> {
        let [nx, ny, nz] = g.dims;
        let n = g.n_cells();
        if perm.len() != n || poro.len() != n {
            return Err(SxError::Shape(format!("ve: {n} cells, perm {}, poro {}", perm.len(), poro.len())));
        }
        let mut cols = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let act: Vec<usize> = (0..nz).map(|k| g.index(i, j, k)).filter(|&c| g.actnum[c] && poro[c] > 0.0 && g.volume(c) > 0.0).collect();
                let c0 = g.corners(g.index(i, j, 0));
                let xy = [[c0[0][0], c0[0][1]], [c0[1][0], c0[1][1]], [c0[2][0], c0[2][1]], [c0[3][0], c0[3][1]]];
                if act.is_empty() {
                    cols.push(Col { top: 0.0, h: 0.0, poro: 0.0, perm: 0.0, xy });
                    continue;
                }
                let ct = g.corners(act[0]);
                let cb = g.corners(*act.last().unwrap_or(&act[0]));
                let top = (0..4).map(|q| ct[q][2]).sum::<f64>() / 4.0;
                let bot = (4..8).map(|q| cb[q][2]).sum::<f64>() / 4.0;
                let (mut v, mut pv, mut kh, mut hh) = (0.0, 0.0, 0.0, 0.0);
                for &c in &act {
                    let cs = g.corners(c);
                    let dz = (4..8).map(|q| cs[q][2]).sum::<f64>() / 4.0 - (0..4).map(|q| cs[q][2]).sum::<f64>() / 4.0;
                    v += g.volume(c);
                    pv += g.volume(c) * poro[c];
                    kh += perm[c][0] * dz;
                    hh += dz;
                }
                cols.push(Col {
                    top,
                    h: (bot - top).max(0.0),
                    poro: pv / v,
                    perm: if hh > 0.0 { kh / hh } else { 0.0 },
                    xy: [[ct[0][0], ct[0][1]], [ct[1][0], ct[1][1]], [ct[2][0], ct[2][1]], [ct[3][0], ct[3][1]]],
                });
            }
        }
        Self::build(nx, ny, cols)
    }

    fn build(nx: usize, ny: usize, cols: Vec<Col>) -> SxResult<Self> {
        if cols.iter().any(|c| !(c.top.is_finite() && c.h >= 0.0 && c.poro >= 0.0 && c.perm >= 0.0)) {
            return Err(SxError::InvalidInput("ve: depths finite, thickness/poro/perm >= 0".into()));
        }
        let mut id = vec![usize::MAX; nx * ny];
        let mut g = VeGrid { top: vec![], height: vec![], poro: vec![], perm: vec![], area: vec![], center: vec![], conns: vec![], outer: vec![], column: vec![], dims: [nx, ny] };
        for (c, col) in cols.iter().enumerate() {
            if col.h > 0.0 && col.poro > 0.0 && col.perm > 0.0 {
                id[c] = g.top.len();
                let p = &col.xy; // corners 0:(i,j) 1:(i+1,j) 2:(i,j+1) 3:(i+1,j+1)
                let quad = [p[0], p[1], p[3], p[2]];
                let mut a = 0.0;
                for q in 0..4 {
                    let (u, v) = (quad[q], quad[(q + 1) % 4]);
                    a += u[0] * v[1] - v[0] * u[1];
                }
                g.top.push(col.top);
                g.height.push(col.h);
                g.poro.push(col.poro);
                g.perm.push(col.perm);
                g.area.push(0.5 * a.abs());
                g.center.push([(p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0, (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0]);
                g.column.push(c);
            }
        }
        // Faces: east (corners 1,3) and north (corners 2,3) of each column.
        let face = |col: &Col, east: bool| if east { (col.xy[1], col.xy[3]) } else { (col.xy[2], col.xy[3]) };
        let west_south = |col: &Col, west: bool| if west { (col.xy[0], col.xy[2]) } else { (col.xy[0], col.xy[1]) };
        let geom = |f: ([f64; 2], [f64; 2]), ctr: [f64; 2]| {
            let len = ((f.1[0] - f.0[0]).powi(2) + (f.1[1] - f.0[1]).powi(2)).sqrt();
            let m = [0.5 * (f.0[0] + f.1[0]), 0.5 * (f.0[1] + f.1[1])];
            (len, ((m[0] - ctr[0]).powi(2) + (m[1] - ctr[1]).powi(2)).sqrt())
        };
        for j in 0..ny {
            for i in 0..nx {
                let c = i + nx * j;
                let a = id[c];
                if a == usize::MAX {
                    continue;
                }
                let ca = &cols[c];
                for east in [true, false] {
                    let (ni, nj) = if east { (i + 1, j) } else { (i, j + 1) };
                    let (l, d) = geom(face(ca, east), g.center[a]);
                    let ta = half(ca.perm, ca.h, l, d);
                    if ni < nx && nj < ny && id[ni + nx * nj] != usize::MAX {
                        let b = id[ni + nx * nj];
                        let cb = &cols[ni + nx * nj];
                        let (lb, db) = geom(west_south(cb, east), g.center[b]);
                        let tb = half(cb.perm, cb.h, lb, db);
                        g.conns.push((a, b, 1.0 / (1.0 / ta + 1.0 / tb)));
                    } else if (east && nx > 1) || (!east && ny > 1) {
                        g.outer.push((a, ta));
                    }
                    let (pi, pj) = if east { (i.wrapping_sub(1), j) } else { (i, j.wrapping_sub(1)) };
                    if ((east && nx > 1) || (!east && ny > 1)) && (pi >= nx || pj >= ny || id[pi + nx * pj] == usize::MAX) {
                        let (l, d) = geom(west_south(ca, east), g.center[a]);
                        g.outer.push((a, half(ca.perm, ca.h, l, d)));
                    }
                }
            }
        }
        if g.top.is_empty() {
            return Err(SxError::EmptyInput("ve: no active columns".into()));
        }
        Ok(g)
    }

    /// Number of VE cells.
    pub fn n_cells(&self) -> usize {
        self.top.len()
    }

    /// Neighbour lists.
    fn adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![vec![]; self.n_cells()];
        for &(a, b, _) in &self.conns {
            adj[a].push(b);
            adj[b].push(a);
        }
        adj
    }
}

struct Col {
    top: f64,
    h: f64,
    poro: f64,
    perm: f64,
    xy: [[f64; 2]; 4],
}

/// Structural traps on the caprock (spill-point analysis).
#[derive(Debug, Clone, PartialEq)]
pub struct Traps {
    /// Trap id per cell (`None` = not in a trap).
    pub trap: Vec<Option<usize>>,
    /// Spill depth per trap (m).
    pub spill_depth: Vec<f64>,
    /// Filled depth per cell: the depth CO2 can reach under the caprock
    /// before it spills (= caprock depth outside traps).
    pub fill_depth: Vec<f64>,
}

#[derive(PartialEq)]
struct Item(f64, usize);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Spill-point analysis by priority flood (Barnes et al. 2014) on the
/// caprock depth, flooding inward from the cells with an outer face: CO2
/// rising under the caprock fills each local depth minimum down to the
/// shallowest depth at which it can escape. Cells whose filled depth
/// exceeds the caprock depth by more than `tol` (m) form traps
/// (connected components, one spill depth each). A cell-based method of
/// our own; MRST co2lab offers cell- and node-based variants (Nilsen et
/// al. 2015) whose trap volumes can differ at the grid scale.
pub fn trap_analysis(g: &VeGrid, tol: f64) -> Traps {
    let n = g.n_cells();
    let adj = g.adjacency();
    let mut fill = vec![f64::NAN; n];
    let mut heap = BinaryHeap::new();
    let mut seen = vec![false; n];
    for &(c, _) in &g.outer {
        if !seen[c] {
            seen[c] = true;
            fill[c] = g.top[c];
            heap.push(Item(g.top[c], c));
        }
    }
    if heap.is_empty() {
        // Closed island: the deepest cell is the only exit.
        let c = (0..n).max_by(|&a, &b| g.top[a].total_cmp(&g.top[b])).unwrap_or(0);
        seen[c] = true;
        fill[c] = g.top[c];
        heap.push(Item(g.top[c], c));
    }
    while let Some(Item(f, c)) = heap.pop() {
        for &b in &adj[c] {
            if !seen[b] {
                seen[b] = true;
                fill[b] = g.top[b].max(f);
                heap.push(Item(fill[b], b));
            }
        }
    }
    for (c, v) in fill.iter_mut().enumerate() {
        if v.is_nan() {
            *v = g.top[c];
        }
    }
    let mut trap = vec![None; n];
    let mut spill = vec![];
    for s in 0..n {
        if trap[s].is_some() || fill[s] - g.top[s] <= tol {
            continue;
        }
        let t = spill.len();
        spill.push(fill[s]);
        let mut stack = vec![s];
        trap[s] = Some(t);
        while let Some(c) = stack.pop() {
            for &b in &adj[c] {
                if trap[b].is_none() && fill[b] - g.top[b] > tol && (fill[b] - fill[s]).abs() <= tol.max(1e-9) {
                    trap[b] = Some(t);
                    stack.push(b);
                }
            }
        }
    }
    Traps { trap, spill_depth: spill, fill_depth: fill }
}

/// Fluid and rock-fluid settings of a VE run.
#[derive(Debug, Clone, PartialEq)]
pub struct VeFluid {
    /// CO2 density (kg/m³).
    pub rho_co2: f64,
    /// Brine density (kg/m³).
    pub rho_brine: f64,
    /// CO2 viscosity (Pa·s).
    pub mu_co2: f64,
    /// Brine viscosity (Pa·s).
    pub mu_brine: f64,
    /// Residual brine saturation in the plume.
    pub s_wr: f64,
    /// Residual CO2 saturation left behind a receding plume.
    pub s_gr: f64,
}

impl VeFluid {
    /// Densities and viscosities from the CO2–brine property module at
    /// pressure `p_pa`, temperature `t_k` and NaCl molality `m_nacl`
    /// (pure CO2: Peng-Robinson + Fenghour; brine without dissolved CO2).
    ///
    /// # Errors
    /// From the brine correlations outside their range.
    pub fn from_co2_brine(p_pa: f64, t_k: f64, m_nacl: f64, s_wr: f64, s_gr: f64) -> SxResult<Self> {
        use super::co2_brine as cb;
        let rho_co2 = cb::co2_density(p_pa, t_k);
        Ok(Self {
            rho_co2,
            mu_co2: cb::co2_viscosity(rho_co2, t_k),
            rho_brine: cb::brine_density(p_pa, t_k, m_nacl, 0.0)?,
            mu_brine: cb::brine_viscosity(p_pa, t_k, m_nacl, 0.0)?,
            s_wr,
            s_gr,
        })
    }
}

/// A CO2 injector: cell, reservoir-volume rate (m³/s) between `t_start` and `t_end` (s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VeWell {
    /// VE cell.
    pub cell: usize,
    /// CO2 rate at reservoir conditions (m³/s).
    pub rate: f64,
    /// Start (s).
    pub t_start: f64,
    /// End (s).
    pub t_end: f64,
}

/// Run settings.
#[derive(Debug, Clone, PartialEq)]
pub struct VeOptions {
    /// Gravity (m/s²).
    pub gravity: f64,
    /// Outer faces open to hydrostatic brine (true) or closed (false;
    /// pressure then pinned at the first cell).
    pub open_boundary: bool,
    /// Brine pressure at depth `z_ref` (Pa).
    pub p_ref: f64,
    /// Reference depth (m).
    pub z_ref: f64,
    /// CFL number of the explicit CO2 update.
    pub cfl: f64,
    /// Largest step (s).
    pub dt_max: f64,
    /// Trap tolerance for the inventory (m).
    pub trap_tol: f64,
}

impl Default for VeOptions {
    fn default() -> Self {
        Self { gravity: G, open_boundary: true, p_ref: 1e7, z_ref: 1000.0, cfl: 0.5, dt_max: 365.25 * 86_400.0, trap_tol: 1e-3 }
    }
}

/// Output of [`run_co2_ve`] at the requested times.
#[derive(Debug, Clone, PartialEq)]
pub struct VeResult {
    /// Output times (s).
    pub times: Vec<f64>,
    /// Plume thickness per time per cell (m).
    pub h: Vec<Vec<f64>>,
    /// Historical maximum thickness per time per cell (m).
    pub h_max: Vec<Vec<f64>>,
    /// Caprock pressure per time per cell (Pa).
    pub p: Vec<Vec<f64>>,
    /// CO2 volume (reservoir m³) per time per cell.
    pub volume: Vec<Vec<f64>>,
    /// Inventory per time (CO2 mass, kg): injected, in place, residual,
    /// structural (mobile CO2 above the spill depth in traps), free plume,
    /// left through open edges.
    pub inventory: Vec<[f64; 6]>,
    /// Explicit steps taken.
    pub n_steps: usize,
    /// Trap analysis used for the inventory.
    pub traps: Traps,
}

/// Inventory column names of [`VeResult::inventory`].
pub const INVENTORY: [&str; 6] = ["injected", "in_place", "residual", "structural", "free_plume", "exited"];

/// Jacobi-preconditioned CG on a sym. positive-definite TPFA matrix.
fn pcg(n: usize, diag: &[f64], off: &[(usize, usize, f64)], b: &[f64], x: &mut [f64]) -> SxResult<()> {
    let mul = |v: &[f64], out: &mut [f64]| {
        for i in 0..n {
            out[i] = diag[i] * v[i];
        }
        for &(a, c, t) in off {
            out[a] -= t * v[c];
            out[c] -= t * v[a];
        }
    };
    let mut r = vec![0.0; n];
    mul(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
    let mut z: Vec<f64> = (0..n).map(|i| r[i] / diag[i]).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut ap = vec![0.0; n];
    for _ in 0..(20 * n + 100) {
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1e-13 * bn {
            return Ok(());
        }
        mul(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = r[i] / diag[i];
        }
        let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    Err(SxError::Numerical("ve: pressure CG did not converge".into()))
}

/// Run the VE model from an empty aquifer. See the module docs.
///
/// # Errors
/// `InvalidInput` for bad settings, `Numerical` if a pressure solve fails.
pub fn run_co2_ve(g: &VeGrid, fluid: &VeFluid, wells: &[VeWell], times: &[f64], opts: &VeOptions) -> SxResult<VeResult> {
    let n = g.n_cells();
    if wells.iter().any(|w| w.cell >= n || !(w.rate >= 0.0) || !(w.t_end >= w.t_start)) {
        return Err(SxError::InvalidInput("ve: well cell out of range or bad rate/times".into()));
    }
    if times.windows(2).any(|w| !(w[1] > w[0])) || times.first().is_some_and(|&t| !(t >= 0.0)) {
        return Err(SxError::InvalidInput("ve: times must be >= 0 and increasing".into()));
    }
    let f = fluid;
    if !(f.rho_brine > f.rho_co2 && f.rho_co2 > 0.0 && f.mu_co2 > 0.0 && f.mu_brine > 0.0 && (0.0..1.0).contains(&f.s_wr) && (0.0..1.0).contains(&f.s_gr) && f.s_wr + f.s_gr < 1.0) {
        return Err(SxError::InvalidInput("ve: need rho_brine > rho_co2 > 0, viscosities > 0, s_wr + s_gr < 1".into()));
    }
    if !(opts.cfl > 0.0 && opts.cfl <= 1.0 && opts.dt_max > 0.0 && opts.gravity >= 0.0) {
        return Err(SxError::InvalidInput("ve: need 0 < cfl <= 1, dt_max > 0, gravity >= 0".into()));
    }
    let gr = opts.gravity;
    let drho = f.rho_brine - f.rho_co2;
    let (krg_end, krw_end) = (1.0 - f.s_wr, 1.0 - f.s_gr);
    let pva: Vec<f64> = (0..n).map(|c| g.poro[c] * g.area[c]).collect(); // pore volume per metre
    let traps = trap_analysis(g, opts.trap_tol);
    let hydro = |z: f64| opts.p_ref + f.rho_brine * gr * (z - opts.z_ref);
    let mut vol = vec![0.0; n];
    let mut h = vec![0.0; n];
    let mut hmax = vec![0.0; n];
    let mut p: Vec<f64> = (0..n).map(|c| hydro(g.top[c])).collect();
    let (mut t, mut injected, mut exited, mut n_steps) = (0.0, 0.0, 0.0, 0usize);
    let mut out = VeResult { times: vec![], h: vec![], h_max: vec![], p: vec![], volume: vec![], inventory: vec![], n_steps: 0, traps: traps.clone() };
    let lam_g = |c: usize, h: &[f64]| krg_end * h[c] / g.height[c] / f.mu_co2;
    let lam_w = |c: usize, h: &[f64], hm: &[f64]| (g.height[c] - hm[c] + krw_end * (hm[c] - h[c])) / g.height[c] / f.mu_brine;
    let phi_g = |c: usize, p: &[f64]| p[c] - f.rho_co2 * gr * g.top[c];
    let phi_w = |c: usize, p: &[f64], h: &[f64]| p[c] - f.rho_brine * gr * g.top[c] - drho * gr * h[c];
    let record = |out: &mut VeResult, t: f64, h: &[f64], hm: &[f64], p: &[f64], vol: &[f64], injected: f64, exited: f64| {
        let (mut res, mut st, mut free) = (0.0, 0.0, 0.0);
        for c in 0..n {
            let r = pva[c] * f.s_gr * (hm[c] - h[c]);
            let mob = vol[c] - r;
            let trapped = if traps.trap[c].is_some() { pva[c] * krg_end * h[c].min((traps.fill_depth[c] - g.top[c]).max(0.0)) } else { 0.0 };
            let trapped = trapped.min(mob.max(0.0));
            res += r;
            st += trapped;
            free += mob - trapped;
        }
        let tot: f64 = vol.iter().sum();
        let rc = f.rho_co2;
        out.times.push(t);
        out.h.push(h.to_vec());
        out.h_max.push(hm.to_vec());
        out.p.push(p.to_vec());
        out.volume.push(vol.to_vec());
        out.inventory.push([injected * rc, tot * rc, res * rc, st * rc, free * rc, exited * rc]);
    };
    let pin = if opts.open_boundary { None } else { Some(0) };
    for &t_out in times {
        while t < t_out - 1e-9 * t_out.max(1.0) {
            // Well sources over this step (constant within the step).
            let mut q = vec![0.0; n];
            for w in wells {
                if t >= w.t_start && t < w.t_end {
                    q[w.cell] += w.rate;
                }
            }
            // Pressure: two passes so upwinding follows the new potentials.
            for _ in 0..2 {
                let mut diag = vec![0.0; n];
                let mut rhs = q.clone();
                let mut off = Vec::with_capacity(g.conns.len());
                for &(a, b, tr) in &g.conns {
                    let ug = if phi_g(a, &p) >= phi_g(b, &p) { a } else { b };
                    let uw = if phi_w(a, &p, &h) >= phi_w(b, &p, &h) { a } else { b };
                    let (lg, lw) = (lam_g(ug, &h), lam_w(uw, &h, &hmax));
                    let c = tr * (lg + lw);
                    diag[a] += c;
                    diag[b] += c;
                    off.push((a, b, c));
                    let gab = tr * (lg * f.rho_co2 * gr * (g.top[a] - g.top[b]) + lw * (f.rho_brine * gr * (g.top[a] - g.top[b]) + drho * gr * (h[a] - h[b])));
                    rhs[a] += gab;
                    rhs[b] -= gab;
                }
                if opts.open_boundary {
                    for &(a, tr) in &g.outer {
                        let pb = hydro(g.top[a]);
                        let out_g = phi_g(a, &p) >= pb - f.rho_co2 * gr * g.top[a];
                        let lg = if out_g { lam_g(a, &h) } else { 0.0 };
                        let out_w = phi_w(a, &p, &h) >= pb - f.rho_brine * gr * g.top[a];
                        let lw = if out_w { lam_w(a, &h, &hmax) } else { 1.0 / f.mu_brine };
                        let c = tr * (lg + lw);
                        diag[a] += c;
                        rhs[a] += c * pb + tr * lw * drho * gr * h[a];
                    }
                }
                if let Some(c0) = pin {
                    let big = diag[c0].max(1e-300) * 1e6;
                    diag[c0] += big;
                    rhs[c0] += big * hydro(g.top[c0]);
                }
                pcg(n, &diag, &off, &rhs, &mut p)?;
            }
            // CO2 fluxes and stable step.
            let mut flux = vec![0.0; g.conns.len()];
            let mut rate = vec![0.0; n];
            for (e, &(a, b, tr)) in g.conns.iter().enumerate() {
                let d = phi_g(a, &p) - phi_g(b, &p);
                let u = if d >= 0.0 { a } else { b };
                flux[e] = tr * lam_g(u, &h) * d;
                let hs = h[a].max(h[b]) / g.height[u];
                let s = tr * krg_end / f.mu_co2 * (d.abs() / g.height[u] + drho * gr * hs.max(1e-3));
                rate[a] += s;
                rate[b] += s;
            }
            let mut bflux = vec![0.0; g.outer.len()];
            if opts.open_boundary {
                for (e, &(a, tr)) in g.outer.iter().enumerate() {
                    let d = phi_g(a, &p) - (hydro(g.top[a]) - f.rho_co2 * gr * g.top[a]);
                    if d > 0.0 {
                        bflux[e] = tr * lam_g(a, &h) * d;
                        rate[a] += tr * krg_end / f.mu_co2 * d / g.height[a];
                    }
                }
            }
            let mut dt = (t_out - t).min(opts.dt_max);
            for c in 0..n {
                if rate[c] > 0.0 {
                    dt = dt.min(opts.cfl * pva[c] * krg_end / rate[c]);
                }
                if q[c] > 0.0 {
                    // Do not overfill a column within one step.
                    let room = pva[c] * (krg_end * g.height[c]) - vol[c];
                    dt = dt.min((opts.cfl * pva[c] * krg_end * g.height[c]).max(room) / q[c]);
                }
            }
            for w in wells {
                if w.t_start > t && w.t_start < t + dt {
                    dt = w.t_start - t;
                }
                if w.t_end > t && w.t_end < t + dt {
                    dt = w.t_end - t;
                }
            }
            // Outflow limiter: never take more than the mobile CO2 present.
            let mut outv = vec![0.0; n];
            for (e, &(a, b, _)) in g.conns.iter().enumerate() {
                if flux[e] > 0.0 { outv[a] += flux[e] } else { outv[b] -= flux[e] }
            }
            for (e, &(a, _)) in g.outer.iter().enumerate() {
                outv[a] += bflux[e];
            }
            for c in 0..n {
                let mobile = vol[c] - pva[c] * f.s_gr * (hmax[c] - h[c]);
                if outv[c] * dt > mobile && outv[c] > 0.0 {
                    dt = dt.min(mobile.max(0.0) / outv[c]).max(1e-6 * dt);
                }
            }
            for (e, &(a, b, _)) in g.conns.iter().enumerate() {
                vol[a] -= dt * flux[e];
                vol[b] += dt * flux[e];
            }
            for (e, &(a, _)) in g.outer.iter().enumerate() {
                vol[a] -= dt * bflux[e];
                exited += dt * bflux[e];
            }
            for c in 0..n {
                vol[c] += dt * q[c];
                injected += dt * q[c];
            }
            // Thickness from volume.
            for c in 0..n {
                let v = vol[c].max(0.0) / pva[c];
                if v >= krg_end * hmax[c] {
                    h[c] = (v / krg_end).min(g.height[c]);
                    hmax[c] = h[c];
                } else if v >= f.s_gr * hmax[c] {
                    h[c] = (v - f.s_gr * hmax[c]) / (krg_end - f.s_gr);
                } else {
                    h[c] = 0.0;
                    hmax[c] = if f.s_gr > 0.0 { v / f.s_gr } else { 0.0 };
                }
            }
            t += dt;
            n_steps += 1;
            if n_steps > 50_000_000 {
                return Err(SxError::Numerical("ve: too many steps".into()));
            }
        }
        record(&mut out, t, &h, &hmax, &p, &vol, injected, exited);
    }
    out.n_steps = n_steps;
    Ok(out)
}

/// Nordbotten & Celia (2006) similarity solution for the CO2–brine
/// interface around an injector in a horizontal confined aquifer, gravity
/// neglected: plume thickness `h(r)` for injected reservoir volume `v`,
/// thickness `big_h`, effective pore fraction `phi_eff` (porosity times
/// the CO2 saturation in the plume) and mobility ratio `lambda = λ_c / λ_b`.
pub fn nordbotten_celia_h(r: f64, v: f64, big_h: f64, phi_eff: f64, lambda: f64) -> f64 {
    let x = (lambda * v / (std::f64::consts::PI * phi_eff * big_h * r * r)).sqrt();
    let h = big_h * (x - 1.0) / (lambda - 1.0);
    h.clamp(0.0, big_h)
}
