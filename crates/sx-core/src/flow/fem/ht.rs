// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! **Experimental** finite-element (FEM) single-phase water flow with heat
//! transport, after the OpenGeoSys HT (hydro-thermal) process (Kolditz et
//! al. 2012). Written from the published equations; no OpenGeoSys code.
//!
//! Unknowns are pressure and temperature at the nodes of a [`HexMesh`]
//! (trilinear hexahedra on the grid corners, [`super::hex8`]). Equations, with
//! a closed (no-flow, insulated) outer boundary:
//!
//! ```text
//! ∂(φρ)/∂t + ∇·(ρ q) = Q_w
//! ∂(φρU + (1 − φ) ρ_r c_r (T − T_ref))/∂t + ∇·(ρ h q) − ∇·(κ ∇T) = Q_E
//! q = −(k/μ)(∇p − ρ g ∇z),   κ = φ κ_f + (1 − φ) κ_r
//! ```
//!
//! with the water, rock and porosity model of [`ThermalWater`] (IAPWS-IF97,
//! `φ(p) = φ₀ (1 + c_r (p − p_ref))`, `U = h − p/ρ`), so a run compares
//! directly with [`run_fim`](crate::flow::run_fim) on the same [`FlowModel`].
//! Depth `z` is positive down.
//!
//! * Continuous Galerkin weak form, 2 × 2 × 2 Gauss points, water properties at the
//!   Gauss points; permeability, porosity and conductivity per element (cell).
//! * Storage lumped to the nodes (the OpenGeoSys `mass_lumping` option): each
//!   element spreads its pore volume and rock volume (bulk − pore) to its
//!   nodes in proportion to `∫ N_a dV`, so the total pore volume equals the
//!   [`FlowModel`]'s.
//! * Heat advection is stabilized by streamline-upwind artificial conduction
//!   along the Darcy flux (Brooks & Hughes 1982),
//!   `K_su = s (ℓ/2) ρ c_p q qᵀ / |q|` with `ℓ = 2|q| / Σ_a |q·∇N_a|` the
//!   element length along the flow; `s = 1` is first-order upwinding in 1D
//!   (as in the FVM solvers), `s = 0` plain Galerkin, which
//!   oscillates when the cell Péclet number is large.
//! * Wells (Peaceman 1978; the model of [`run_fim`](crate::flow::run_fim)): a
//!   completed cell exchanges `WI β (p_wb − p_c)` with its element, `p_c` the
//!   value at the cell centre (mean of its 8 nodes), split 1/8 per node. One
//!   BHP unknown per well, rate (reservoir volume) or BHP control, rate wells
//!   with `bhp_limit` switch to BHP at the limit, no crossflow, wellbore
//!   density frozen per step.
//! * Backward Euler with the step control of [`run_fim`](crate::flow::run_fim).
//!   Each step is solved by Picard iteration: the residual is exact; the
//!   Jacobian keeps the flux coefficients (ρk/μ, h, κ, `K_su`) at the last
//!   iterate and includes the storage derivatives (forward differences) and
//!   the advective `c_p ∂T` term.
//!
//! Unknowns per node `[p, T]`, then one BHP per well (the layout of
//! [`run_fim`](crate::flow::run_fim)), solved with the same [`FimLinear`].
//!
//! References:
//! * Kolditz, O., et al. (2012). OpenGeoSys: an open-source initiative for
//!   numerical simulation of thermo-hydro-mechanical/chemical (THM/C)
//!   processes in porous media. Environ. Earth Sci. 67, 589–599.
//!   doi:10.1007/s12665-012-1546-x
//! * Brooks, A. N., Hughes, T. J. R. (1982). Streamline upwind/Petrov–Galerkin
//!   formulations for convection dominated flows with particular emphasis on
//!   the incompressible Navier–Stokes equations. Comput. Methods Appl. Mech.
//!   Eng. 32, 199–259.
//! * Peaceman, D. W. (1978). Interpretation of well-block pressures in
//!   numerical reservoir simulation. SPE J. 18(3), 183–194.
//!
//! Status: experimental; faulted grids are rejected (nodes do not match
//! across fault throws). See `docs/fem_ht.md`.

#![allow(clippy::needless_range_loop)]

use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat, Triplet};
use faer::Mat;

use super::hex8::{gauss_points, GaussPoint, N_NODES};
use super::mesh::HexMesh;
use crate::error::{SxError, SxResult};
use crate::flow::fim_physics::T_REF_ROCK;
use crate::flow::linsolve::{FimLinear, LinearSolver, LinearStats};
use crate::flow::pressure::well_indices;
use crate::flow::solver::{par_for, par_loops, with_threads};
use rayon::prelude::*;
use crate::flow::{FlowModel, FlowWell, OblPhysics, ThermalWater, WellControl, WellKind};

/// Pressure step (Pa) of the forward-difference storage derivatives.
const DP_FD: f64 = 100.0;
/// Temperature step (K) of the forward-difference storage derivatives.
const DT_FD: f64 = 1e-3;

/// Settings of [`run_fem_ht`].
#[derive(Debug, Clone, PartialEq)]
pub struct FemOptions {
    /// Gravity (m/s²); 0 = off.
    pub gravity: f64,
    /// Injection temperature (K) per well (ignored for producers).
    pub injection_temperature: Vec<f64>,
    /// First time step (s).
    pub dt_init: f64,
    /// Largest time step (s).
    pub dt_max: f64,
    /// Smallest time step (s); a failure below it is an error.
    pub dt_min: f64,
    /// Growth factor after a converged step.
    pub dt_growth: f64,
    /// Cut factor after a failed step.
    pub dt_cut: f64,
    /// Picard iterations per step.
    pub max_iter: usize,
    /// Tolerance on the normalized residual (water residual / nodal water
    /// mass; energy residual / nodal heat capacity, i.e. in K).
    pub tol: f64,
    /// Largest temperature change per iteration (K).
    pub max_dtemp: f64,
    /// Streamline-upwind tuning `s` (1 = first-order upwind in 1D; 0 = plain Galerkin).
    pub stabilization: f64,
    /// Corners closer than this (m) share a node.
    pub merge_tol: f64,
    /// Linear solver (as [`run_fim`](crate::flow::run_fim)).
    pub linear_solver: LinearSolver,
}

impl Default for FemOptions {
    fn default() -> Self {
        Self {
            gravity: 0.0,
            injection_temperature: Vec::new(),
            dt_init: 86_400.0,
            dt_max: 30.0 * 86_400.0,
            dt_min: 1e-3,
            dt_growth: 1.5,
            dt_cut: 0.5,
            max_iter: 30,
            tol: 1e-6,
            max_dtemp: 20.0,
            stabilization: 1.0,
            merge_tol: 1e-6,
            linear_solver: LinearSolver::default(),
        }
    }
}

/// Output of [`run_fem_ht`], at the requested times.
#[derive(Debug, Clone, PartialEq)]
pub struct FemResult {
    /// Output times (s).
    pub times: Vec<f64>,
    /// Pressure per time per grid cell (Pa): value at the cell centre (mean of
    /// its 8 nodes); inactive cells keep `p0`.
    pub p: Vec<Vec<f64>>,
    /// Temperature per time per grid cell (K), as `p`.
    pub temperature: Vec<Vec<f64>>,
    /// Pressure per time per node (Pa).
    pub p_nodes: Vec<Vec<f64>>,
    /// Temperature per time per node (K).
    pub temperature_nodes: Vec<Vec<f64>>,
    /// Node coordinates (x, y, depth), m.
    pub nodes: Vec<[f64; 3]>,
    /// BHP per time per well (Pa).
    pub bhp: Vec<Vec<f64>>,
    /// Rates per time per well, positive = into the reservoir, well-major
    /// `[w·2 + c]`: water (kg/s), energy (W).
    pub well_mass_rate: Vec<Vec<f64>>,
    /// Water (kg) and energy (J, fluid + rock) in place per time.
    pub mass_in_place: Vec<Vec<f64>>,
    /// Cumulative water (kg) and energy (J) injected minus produced up to each time.
    pub mass_net_in: Vec<Vec<f64>>,
    /// Accepted steps.
    pub n_steps: usize,
    /// Repeated (cut) steps.
    pub n_cuts: usize,
    /// Total Picard iterations.
    pub n_iterations: usize,
    /// Water-property evaluations that failed and used a fallback (see [`ThermalWater`]).
    pub n_fallbacks: usize,
    /// Linear solves, GMRES iterations and direct-LU fallbacks.
    pub linear: LinearStats,
}

/// Fixed data of a run: mesh, integration points, lumped volumes, wells.
struct Problem<'a> {
    mesh: HexMesh,
    gps: Vec<[GaussPoint; 8]>,
    /// Nodal pore volume (m³).
    pv: Vec<f64>,
    /// Nodal rock volume (m³).
    vr: Vec<f64>,
    /// Porosity per element.
    poro: Vec<f64>,
    model: &'a FlowModel,
    wells: &'a [FlowWell],
    wi: Vec<Vec<f64>>,
    depth_ref: Vec<f64>,
    phys: &'a ThermalWater,
    opts: &'a FemOptions,
}

/// Residual, Jacobian triplets, well rates and the convergence measure at one iterate.
struct Assembly {
    r: Vec<f64>,
    jac: Vec<Triplet<usize, usize, f64>>,
    rates: Vec<f64>,
    err: f64,
}

impl Problem<'_> {
    /// Water and fluid internal energy per unit reference pore volume: `[φρ, φρU]`.
    fn storage(&self, p: f64, t: f64) -> [f64; 2] {
        let w = self.phys.props(p, t);
        let phi = 1.0 + self.phys.c_rock * (p - self.phys.p_ref);
        [phi * w.density, phi * (w.density * w.enthalpy - p)]
    }

    /// Value of variable `d` (0 = p, 1 = T) at the centre of element `e`.
    fn centre(&self, x: &[f64], e: usize, d: usize) -> f64 {
        self.mesh.elems[e].iter().map(|&a| x[2 * a + d]).sum::<f64>() / N_NODES as f64
    }

    /// Water (kg) and energy (J, fluid + rock) in place.
    fn in_place(&self, x: &[f64]) -> Vec<f64> {
        let rock = self.phys.rock;
        let mut m = vec![0.0; 2];
        for a in 0..self.mesh.n_nodes() {
            let s = self.storage(x[2 * a], x[2 * a + 1]);
            m[0] += self.pv[a] * s[0];
            m[1] += self.pv[a] * s[1] + self.vr[a] * rock.rho * rock.cp * (x[2 * a + 1] - T_REF_ROCK);
        }
        m
    }

    /// Residual and Picard Jacobian of one step of length `step` from the
    /// nodal state whose storage `[φρ, φρU, T]` is `old`.
    fn assemble(&self, x: &[f64], old: &[[f64; 3]], bhp: &[f64], rho_wb: &[f64], ctrl: &[WellControl], step: f64) -> Assembly {
        let (nn, nw) = (self.mesh.n_nodes(), self.wells.len());
        let g = self.opts.gravity;
        let rock = self.phys.rock;
        let c_rock_vol = rock.rho * rock.cp;
        let mut r = vec![0.0; 2 * nn + nw];
        let mut jac: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(4 * nn + 3 * 64 * self.mesh.elems.len());
        // Lumped storage.
        let mut mass = vec![0.0; nn];
        let mut cap = vec![0.0; nn];
        for a in 0..nn {
            let (p, t) = (x[2 * a], x[2 * a + 1]);
            let s = self.storage(p, t);
            let sp = self.storage(p + DP_FD, t);
            let st = self.storage(p, t + DT_FD);
            r[2 * a] = self.pv[a] * (s[0] - old[a][0]);
            r[2 * a + 1] = self.pv[a] * (s[1] - old[a][1]) + self.vr[a] * c_rock_vol * (t - old[a][2]);
            mass[a] = self.pv[a] * s[0];
            cap[a] = self.pv[a] * (st[1] - s[1]) / DT_FD + self.vr[a] * c_rock_vol;
            jac.push(Triplet::new(2 * a, 2 * a, self.pv[a] * (sp[0] - s[0]) / DP_FD));
            jac.push(Triplet::new(2 * a, 2 * a + 1, self.pv[a] * (st[0] - s[0]) / DT_FD));
            jac.push(Triplet::new(2 * a + 1, 2 * a, self.pv[a] * (sp[1] - s[1]) / DP_FD));
            jac.push(Triplet::new(2 * a + 1, 2 * a + 1, cap[a]));
        }
        // Element fluxes: Darcy, enthalpy advection, conduction.
        let element = |(e, ids): (usize, &[usize; N_NODES])| {
            let mut re = [[0.0f64; 2]; N_NODES];
            let k = self.model.perm[self.mesh.cell_of_elem[e]];
            let phi = self.poro[e];
            let pe: [f64; N_NODES] = std::array::from_fn(|v| x[2 * ids[v]]);
            let te: [f64; N_NODES] = std::array::from_fn(|v| x[2 * ids[v] + 1]);
            // Per node pair (a, b): ∂R_w,a/∂p_b, ∂R_E,a/∂p_b, ∂R_E,a/∂T_b.
            let mut ke = [[[0.0f64; 3]; N_NODES]; N_NODES];
            for gp in &self.gps[e] {
                let p: f64 = (0..N_NODES).map(|v| gp.n[v] * pe[v]).sum();
                let t: f64 = (0..N_NODES).map(|v| gp.n[v] * te[v]).sum();
                let grad_p: [f64; 3] = std::array::from_fn(|d| (0..N_NODES).map(|v| gp.dn[v][d] * pe[v]).sum());
                let grad_t: [f64; 3] = std::array::from_fn(|d| (0..N_NODES).map(|v| gp.dn[v][d] * te[v]).sum());
                let w = self.phys.props(p, t);
                let mob: [f64; 3] = std::array::from_fn(|d| w.density * k[d] / w.viscosity);
                // Darcy mass flux ρq (kg/(m² s)).
                let m: [f64; 3] = std::array::from_fn(|d| -mob[d] * (grad_p[d] - if d == 2 { w.density * g } else { 0.0 }));
                let kappa = phi * w.conductivity + (1.0 - phi) * rock.conductivity;
                let mut kt = [[0.0f64; 3]; 3];
                for (d, row) in kt.iter_mut().enumerate() {
                    row[d] = kappa;
                }
                let q = m.map(|v| v / w.density);
                let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                if self.opts.stabilization > 0.0 && qn > 0.0 {
                    let sum: f64 = gp.dn.iter().map(|dn| (q[0] * dn[0] + q[1] * dn[1] + q[2] * dn[2]).abs()).sum();
                    if sum > 0.0 {
                        let ell = 2.0 * qn / sum;
                        let coef = self.opts.stabilization * 0.5 * ell * w.density * w.cp / qn;
                        for i in 0..3 {
                            for j in 0..3 {
                                kt[i][j] += coef * q[i] * q[j];
                            }
                        }
                    }
                }
                let kgt: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| kt[i][j] * grad_t[j]).sum());
                let wq = step * gp.w;
                for va in 0..N_NODES {
                    let da = gp.dn[va];
                    let dm = da[0] * m[0] + da[1] * m[1] + da[2] * m[2];
                    let dk = da[0] * kgt[0] + da[1] * kgt[1] + da[2] * kgt[2];
                    re[va][0] -= wq * dm;
                    re[va][1] += wq * (dk - w.enthalpy * dm);
                    for vb in 0..N_NODES {
                        let db = gp.dn[vb];
                        let s: f64 = (0..3).map(|d| da[d] * mob[d] * db[d]).sum();
                        let kk: f64 = (0..3).map(|i| (0..3).map(|j| da[i] * kt[i][j] * db[j]).sum::<f64>()).sum();
                        ke[va][vb][0] += wq * s;
                        ke[va][vb][1] += wq * w.enthalpy * s;
                        ke[va][vb][2] += wq * (kk - w.cp * gp.n[vb] * dm);
                    }
                }
            }
            (re, ke)
        };
        let contributions: Vec<_> = if par_loops(self.mesh.elems.len().saturating_mul(8)) {
            self.mesh.elems.par_iter().enumerate().map(element).collect()
        } else {
            self.mesh.elems.iter().enumerate().map(element).collect()
        };
        for (ids, (re, ke)) in self.mesh.elems.iter().zip(contributions) {
            for va in 0..N_NODES {
                r[2 * ids[va]] += re[va][0];
                r[2 * ids[va] + 1] += re[va][1];
            }
            for va in 0..N_NODES {
                for vb in 0..N_NODES {
                    let (ra, cb) = (2 * ids[va], 2 * ids[vb]);
                    jac.push(Triplet::new(ra, cb, ke[va][vb][0]));
                    jac.push(Triplet::new(ra + 1, cb, ke[va][vb][1]));
                    jac.push(Triplet::new(ra + 1, cb + 1, ke[va][vb][2]));
                }
            }
        }
        // Wells: each completion exchanges with the 8 nodes of its cell.
        let mut rates = vec![0.0; 2 * nw];
        let mut wres = vec![0.0; nw];
        for w in 0..nw {
            let col = 2 * nn + w;
            let well = &self.wells[w];
            let inj = well.kind == WellKind::Injector;
            // ∂(p_wb − p_c or p_c − p_wb)/∂p_c, and +1 for flow out of the reservoir.
            let (sgn, out) = if inj { (-1.0, -1.0) } else { (1.0, 1.0) };
            let (mut vol, mut dvol_dbhp, mut dvol_open) = (0.0, 0.0, 0.0);
            let mut dvol_dx: Vec<(usize, f64)> = Vec::new();
            for (j, &i) in well.cells.iter().enumerate() {
                let e = self.mesh.elem_of_cell[i].expect("well cells are active (checked by well_indices)");
                let wi = self.wi[w][j];
                let pc = self.centre(x, e, 0);
                let pwb = bhp[w] + rho_wb[w] * g * (self.model.center[i][2] - self.depth_ref[w]);
                let wp = self.phys.props(pc, if inj { self.opts.injection_temperature[w] } else { self.centre(x, e, 1) });
                let lam = 1.0 / wp.viscosity;
                let (bw, be) = (wp.density * lam, wp.density * wp.enthalpy * lam);
                let dp = if inj { pwb - pc } else { pc - pwb };
                dvol_open += wi * lam;
                if dp < 0.0 {
                    continue;
                }
                vol += wi * lam * dp;
                dvol_dbhp -= sgn * wi * lam;
                let (qw, qe) = (wi * bw * dp, wi * be * dp);
                rates[2 * w] -= out * qw;
                rates[2 * w + 1] -= out * qe;
                let eighth = 1.0 / N_NODES as f64;
                for &a in &self.mesh.elems[e] {
                    r[2 * a] += out * step * qw * eighth;
                    r[2 * a + 1] += out * step * qe * eighth;
                    jac.push(Triplet::new(2 * a, col, -step * wi * bw * eighth));
                    jac.push(Triplet::new(2 * a + 1, col, -step * wi * be * eighth));
                    for &b in &self.mesh.elems[e] {
                        jac.push(Triplet::new(2 * a, 2 * b, step * wi * bw * eighth * eighth));
                        jac.push(Triplet::new(2 * a + 1, 2 * b, step * wi * be * eighth * eighth));
                        if !inj {
                            // Produced enthalpy follows the cell temperature: ∂(ρh/μ)/∂T ≈ ρ c_p/μ.
                            jac.push(Triplet::new(2 * a + 1, 2 * b + 1, step * wi * dp * wp.density * wp.cp * lam * eighth * eighth));
                        }
                    }
                    dvol_dx.push((2 * a, sgn * wi * lam * eighth));
                }
            }
            match ctrl[w] {
                WellControl::Bhp(b) => {
                    wres[w] = (bhp[w] - b) / 1e5;
                    jac.push(Triplet::new(col, col, 1e-5));
                }
                WellControl::Rate(q) => {
                    let s = 1.0 / q.abs().max(1e-30);
                    wres[w] = (vol - q.abs()) * s;
                    // Closed completions: use the open-well slope so the BHP row is never empty.
                    let d = if dvol_dbhp != 0.0 { dvol_dbhp } else { -sgn * dvol_open };
                    jac.push(Triplet::new(col, col, d * s));
                    for (k, v) in dvol_dx {
                        jac.push(Triplet::new(col, k, v * s));
                    }
                }
            }
            r[col] = wres[w];
        }
        // Water residuals normalized by the nodal water mass, energy residuals
        // by the nodal heat capacity (i.e. in K).
        let mut err = 0.0f64;
        for a in 0..nn {
            err = err.max(r[2 * a].abs() / mass[a].max(1e-12));
            err = err.max(r[2 * a + 1].abs() / cap[a].abs().max(1e-12));
        }
        for v in &wres {
            err = err.max(v.abs());
        }
        Assembly { r, jac, rates, err }
    }
}

/// Cached Jacobian sparsity: triplet indices, symbolic structure and argsort.
type JacPattern = (Vec<Pair<usize, usize>>, SymbolicSparseColMat<usize>, Argsort<usize>);

/// Compressed-column matrix of the triplets, reusing the symbolic structure
/// while the triplet index sequence repeats (as in [`run_fim`](crate::flow::run_fim)).
fn matrix(nu: usize, jac: &[Triplet<usize, usize, f64>], pat: &mut Option<JacPattern>) -> Option<SparseColMat<usize, f64>> {
    let same = pat.as_ref().is_some_and(|(ix, _, _)| ix.len() == jac.len() && ix.iter().zip(jac).all(|(p, t)| p.row == t.row && p.col == t.col));
    if !same {
        let ix: Vec<Pair<usize, usize>> = jac.iter().map(|t| Pair { row: t.row, col: t.col }).collect();
        let (sym, ord) = SymbolicSparseColMat::<usize>::try_new_from_indices(nu, nu, &ix).ok()?;
        *pat = Some((ix, sym, ord));
    }
    let (_, sym, ord) = pat.as_ref()?;
    let vals: Vec<f64> = jac.iter().map(|t| t.val).collect();
    SparseColMat::<usize, f64>::new_from_argsort(sym.clone(), ord, &vals).ok()
}

/// Finite-element water flow and heat transport run (experimental).
///
/// `corners` holds the 8 corners of every grid cell (corner-point order,
/// depth positive down; e.g. [`CornerPointGrid::corners`](crate::grid::CornerPointGrid::corners)
/// or [`ogs::regular_corners`](crate::flow::ogs::regular_corners)) of the grid
/// `model` was built from. `p0` (Pa) and `t0` (K) are per cell; a node starts
/// at the mean over the active cells around it. Wells use [`FlowWell`] as
/// [`run_fim`](crate::flow::run_fim) does (`crossflow` is ignored: always off).
///
/// # Errors
/// `Shape`/`InvalidInput` for bad input, a faulted grid or a degenerate
/// cell; `Numerical` when a step fails below `dt_min`.
#[allow(clippy::too_many_arguments)]
pub fn run_fem_ht(
    corners: &[[[f64; 3]; 8]],
    model: &FlowModel,
    wells: &[FlowWell],
    phys: &ThermalWater,
    p0: &[f64],
    t0: &[f64],
    times: &[f64],
    opts: &FemOptions,
) -> SxResult<FemResult> {
    with_threads(None, || run_fem_ht_inner(corners, model, wells, phys, p0, t0, times, opts))
}

#[allow(clippy::too_many_arguments)]
fn run_fem_ht_inner(
    corners: &[[[f64; 3]; 8]],
    model: &FlowModel,
    wells: &[FlowWell],
    phys: &ThermalWater,
    p0: &[f64],
    t0: &[f64],
    times: &[f64],
    opts: &FemOptions,
) -> SxResult<FemResult> {
    let n = model.n_cells();
    if corners.len() != n || p0.len() != n || t0.len() != n {
        return Err(SxError::Shape("fem: corners, p0 and t0 need one entry per cell".into()));
    }
    if model.n_fault_connections > 0 || model.n_pinch_connections > 0 {
        return Err(SxError::InvalidInput("fem: faulted or pinched corner-point grids need explicit interface treatment; use run_fim".into()));
    }
    if times.windows(2).any(|w| !(w[1] > w[0])) || times.first().is_some_and(|&t| !(t >= 0.0)) {
        return Err(SxError::InvalidInput("fem: times must be >= 0 and increasing".into()));
    }
    if !(opts.dt_init > 0.0
        && opts.dt_max >= opts.dt_min
        && opts.dt_min > 0.0
        && opts.dt_growth >= 1.0
        && opts.dt_cut > 0.0
        && opts.dt_cut < 1.0
        && opts.tol > 0.0
        && opts.max_iter > 0
        && opts.max_dtemp > 0.0
        && opts.stabilization >= 0.0)
    {
        return Err(SxError::InvalidInput("fem: bad time-step, Picard or stabilization settings".into()));
    }
    if opts.injection_temperature.len() != wells.len() || opts.injection_temperature.iter().any(|t| !t.is_finite()) {
        return Err(SxError::InvalidInput("fem: injection_temperature needs one finite value (K) per well".into()));
    }
    if wells.iter().any(|w| w.bhp_limit.is_some() && !matches!(w.control, WellControl::Rate(_))) {
        return Err(SxError::InvalidInput("fem: bhp_limit needs a rate-controlled well".into()));
    }
    let mesh = HexMesh::from_corners(corners, &model.active, opts.merge_tol)?;
    // Continuous Galerkin continuity cannot represent a nonmatching FVM link.
    for link in &model.connections {
        let a=mesh.elem_of_cell[link.a];let b=mesh.elem_of_cell[link.b];
        if let (Some(a),Some(b))=(a,b) {
            if mesh.elems[a].iter().filter(|v|mesh.elems[b].contains(v)).count()!=4 {
                return Err(SxError::InvalidInput("fem: a connection lacks a matching four-node face; explicit interface constraints are required".into()));
            }
        }
    }
    let (nn, nw) = (mesh.n_nodes(), wells.len());
    let nu = 2 * nn + nw;
    // Integration points and lumped nodal pore / rock volumes.
    let mut gps = Vec::with_capacity(mesh.elems.len());
    let (mut pv, mut vr) = (vec![0.0; nn], vec![0.0; nn]);
    let mut poro = Vec::with_capacity(mesh.elems.len());
    for (e, ids) in mesh.elems.iter().enumerate() {
        let c = mesh.cell_of_elem[e];
        let xe: [[f64; 3]; N_NODES] = std::array::from_fn(|v| mesh.nodes[ids[v]]);
        let g = gauss_points(&xe).map_err(|_| SxError::InvalidInput(format!("fem: cell {c} is degenerate or folded")))?;
        let vol: f64 = g.iter().map(|q| q.w).sum();
        let (bulk, pore) = (model.bulk_volume[c], model.pore_volume[c]);
        poro.push(if bulk > 0.0 { (pore / bulk).clamp(0.0, 1.0) } else { 0.0 });
        for (v, &a) in ids.iter().enumerate() {
            let share = g.iter().map(|q| q.w * q.n[v]).sum::<f64>() / vol;
            pv[a] += pore * share;
            vr[a] += (bulk - pore).max(0.0) * share;
        }
        gps.push(g);
    }
    let (mut wi, mut depth_ref) = (Vec::new(), Vec::new());
    for w in wells {
        wi.push(well_indices(model, w)?);
        depth_ref.push(w.ref_depth_m.unwrap_or(model.center[w.cells[0]][2]));
    }
    let pb = Problem { mesh, gps, pv, vr, poro, model, wells, wi, depth_ref, phys, opts };

    let (pn0, tn0) = (pb.mesh.cell_to_nodes(p0), pb.mesh.cell_to_nodes(t0));
    let mut x: Vec<f64> = (0..nn).flat_map(|a| [pn0[a], tn0[a]]).collect();
    // Initial BHP guesses.
    let mut bhp: Vec<f64> = wells
        .iter()
        .map(|w| match w.control {
            WellControl::Bhp(b) => b,
            WellControl::Rate(_) => {
                let pc = w.cells.iter().map(|&c| p0[c]).sum::<f64>() / w.cells.len() as f64;
                if w.kind == WellKind::Injector { pc + 1e5 } else { pc - 1e5 }
            }
        })
        .collect();
    let mut res = FemResult {
        times: Vec::new(),
        p: Vec::new(),
        temperature: Vec::new(),
        p_nodes: Vec::new(),
        temperature_nodes: Vec::new(),
        nodes: pb.mesh.nodes.clone(),
        bhp: Vec::new(),
        well_mass_rate: Vec::new(),
        mass_in_place: Vec::new(),
        mass_net_in: Vec::new(),
        n_steps: 0,
        n_cuts: 0,
        n_iterations: 0,
        n_fallbacks: 0,
        linear: LinearStats::default(),
    };
    let record = |res: &mut FemResult, t: f64, x: &[f64], bhp: &[f64], rates: &[f64], net: &[f64]| {
        let pn: Vec<f64> = (0..nn).map(|a| x[2 * a]).collect();
        let tn: Vec<f64> = (0..nn).map(|a| x[2 * a + 1]).collect();
        res.times.push(t);
        res.p.push(pb.mesh.nodes_to_cells(&pn, p0));
        res.temperature.push(pb.mesh.nodes_to_cells(&tn, t0));
        res.p_nodes.push(pn);
        res.temperature_nodes.push(tn);
        res.bhp.push(bhp.to_vec());
        res.well_mass_rate.push(rates.to_vec());
        res.mass_in_place.push(pb.in_place(x));
        res.mass_net_in.push(net.to_vec());
    };
    let mut net_in = vec![0.0; 2];
    let mut last_rates = vec![0.0; 2 * nw];
    let mut t = 0.0;
    let mut dt = opts.dt_init.min(opts.dt_max);
    let mut lin = FimLinear::new(opts.linear_solver, 2, nn);
    let mut pat: Option<JacPattern> = None;
    let mut ctrl: Vec<WellControl> = wells.iter().map(|w| w.control).collect();
    for &t_out in times {
        while t < t_out - 1e-9 * t_out.max(1.0) {
            let step = dt.min(t_out - t);
            let xn = x.clone();
            let old: Vec<[f64; 3]> = (0..nn)
                .map(|a| {
                    let s = pb.storage(xn[2 * a], xn[2 * a + 1]);
                    [s[0], s[1], xn[2 * a + 1]]
                })
                .collect();
            // Wellbore densities frozen over the step.
            let rho_wb: Vec<f64> = wells
                .iter()
                .enumerate()
                .map(|(w, well)| {
                    let e = pb.mesh.elem_of_cell[well.cells[0]].expect("well cells are active (checked by well_indices)");
                    let tw = if well.kind == WellKind::Injector { opts.injection_temperature[w] } else { pb.centre(&xn, e, 1) };
                    phys.props(pb.centre(&xn, e, 0), tw).density
                })
                .collect();
            let mut bhp_try = bhp.clone();
            let mut ok = false;
            let mut rates = vec![0.0; 2 * nw];
            for _it in 0..opts.max_iter {
                res.n_iterations += 1;
                let asm = pb.assemble(&x, &old, &bhp_try, &rho_wb, &ctrl, step);
                rates = asm.rates;
                if !asm.err.is_finite() {
                    break;
                }
                if asm.err < opts.tol {
                    ok = true;
                    break;
                }
                if asm.jac.iter().any(|t| !t.val.is_finite()) {
                    break;
                }
                let Some(a) = matrix(nu, &asm.jac, &mut pat) else { break };
                let mut dx = Mat::<f64>::from_fn(nu, 1, |i, _| -asm.r[i]);
                if lin.solve(&a, &mut dx, par_for(nu)).is_err() || (0..nu).any(|i| !dx[(i, 0)].is_finite()) {
                    break;
                }
                // Limit the temperature change per iteration.
                let maxdt = (0..nn).map(|a| dx[(2 * a + 1, 0)].abs()).fold(0.0, f64::max);
                let scale = if maxdt > opts.max_dtemp { opts.max_dtemp / maxdt } else { 1.0 };
                for (i, v) in x.iter_mut().enumerate() {
                    *v += scale * dx[(i, 0)];
                }
                for w in 0..nw {
                    bhp_try[w] += scale * dx[(2 * nn + w, 0)];
                }
            }
            if ok {
                // BHP-limit switching: redo the step under BHP control.
                let mut switched = false;
                for w in 0..nw {
                    if let (WellControl::Rate(_), Some(lim)) = (ctrl[w], wells[w].bhp_limit) {
                        let over = if wells[w].kind == WellKind::Injector { bhp_try[w] > lim } else { bhp_try[w] < lim };
                        if over {
                            ctrl[w] = WellControl::Bhp(lim);
                            switched = true;
                        }
                    }
                }
                if switched {
                    x = xn;
                    continue;
                }
                for (c, v) in net_in.iter_mut().enumerate() {
                    *v += step * (0..nw).map(|w| rates[w * 2 + c]).sum::<f64>();
                }
                last_rates = rates;
                bhp = bhp_try;
                t += step;
                res.n_steps += 1;
                dt = (step * opts.dt_growth).min(opts.dt_max).max(dt);
            } else {
                x = xn;
                res.n_cuts += 1;
                dt = step * opts.dt_cut;
                if dt < opts.dt_min {
                    return Err(SxError::Numerical(format!("fem: Picard iteration failed at t = {t:.6e} s with dt below dt_min")));
                }
            }
        }
        record(&mut res, t_out, &x, &bhp, &last_rates, &net_in);
    }
    res.linear = lin.stats;
    res.n_fallbacks = phys.n_fallbacks();
    Ok(res)
}
