// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! **Experimental** fully implicit (FIM) multiphase, multicomponent flow with
//! operator-based linearization (OBL).
//!
//! Discretization: cell-centred finite volumes on the [`FlowModel`] grid,
//! two-point flux approximation (TPFA), backward Euler in time, a fully
//! coupled Newton solve, phase-potential upwinding and Peaceman wells.
//! The nonlinear terms are written in the operator form of Voskov (2017) and
//! Khait & Voskov (2017): for every equation `c` (component mass)
//!
//! ```text
//! PV₀ [α_c(ω) − α_c(ωⁿ)] + Δt Σ_faces T Σ_p β_{c,p}(ω_up(p)) ΔΦ_p + Δt q_c = 0
//! α_c = φ(p)/φ₀ Σ_p ρ_p x_{c,p} S_p      (accumulation operator)
//! β_{c,p} = ρ_p x_{c,p} k_{r,p} / μ_p     (flux operator of phase p)
//! ΔΦ_p = (p_a − Pc_{p,a}) − (p_b − Pc_{p,b}) − ρ̄_p g (z_a − z_b)
//! ```
//!
//! with the state `ω = [p, z_1, …, z_{nc−1}]` (pressure and overall mass
//! fractions; the last fraction is the complement). Operators are not
//! evaluated from the physics inside the Newton loop: they are read from a
//! multilinear-interpolation table in state space whose nodes are evaluated
//! on demand (lazily) the first time an interpolation hypercube is visited,
//! and the Jacobian uses the derivatives of that piecewise-multilinear
//! interpolant, as in OBL. Physics enter only through [`OblPhysics`].
//!
//! Depth `z` is positive down, so `Φ = p − ρ g z`. Mass rates, kg/s.
//!
//! Wells (Peaceman, Peaceman 1978): one bottom-hole pressure (BHP) unknown per
//! well, wellbore pressure `p_wb = BHP + ρ_wb g (z − z_ref)` with `ρ_wb` frozen
//! per step (injection stream for injectors, completed-cell mixture for
//! producers). Rate control is on reservoir volume (m³/s): the injection
//! stream volume for injectors, the total phase volume for producers. An
//! injector injects the stream `z_inj` with the flux operators evaluated at
//! `[p_cell, z_inj]`. No crossflow: a completion only flows in the well
//! direction. A rate well with `bhp_limit` switches to BHP at the limit.
//!
//! Time steps: a converged step grows the next by `dt_growth`; a failed
//! Newton solve (no convergence in `max_newton` iterations, a singular or
//! non-finite system) repeats the step with `dt_cut · Δt`.
//!
//! References:
//! * Voskov, D. V. (2017). Operator-based linearization approach for modeling
//!   of multiphase multi-component flow in porous media. J. Comput. Phys. 337,
//!   275–288. doi:10.1016/j.jcp.2017.02.041
//! * Khait, M., Voskov, D. V. (2017). Operator-based linearization for
//!   general purpose reservoir simulation. J. Pet. Sci. Eng. 157, 990–998.
//!   doi:10.1016/j.petrol.2017.08.009
//! * Peaceman, D. W. (1978). Interpretation of well-block pressures in
//!   numerical reservoir simulation. SPE J. 18(3), 183–194.
//! * Aziz, K., Settari, A. (1979). Petroleum Reservoir Simulation.
//!
//! Status: experimental prototype; energy (thermal) assembly is new and only
//! lightly tested. Not yet benchmarked against open-DARTS or
//! OpenGeoSys; see `docs/multiphase_fim.md`.

use std::cell::RefCell;
use std::collections::HashMap;

use faer::prelude::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::Mat;

use super::model::FlowModel;
use super::pressure::{well_indices, FlowWell, WellControl, WellKind};
use crate::error::{SxError, SxResult};

/// Layout of the operator vector returned by [`OblPhysics::evaluate`] for
/// `ne` equations and `np` phases:
/// `[α (ne) | β (ne·np, equation-major) | ρ (np) | Pc (np) | λ = kr/μ (np)]`,
/// followed for a thermal physics by `[U_rock | κ_fluid]`.
///
/// For a thermal physics the last equation is energy: `α_E` is the fluid
/// internal energy per unit reference pore volume (J/m³), `β_{E,p} = h_p ρ_p
/// k_{r,p}/μ_p`, `U_rock` the rock internal energy per unit rock volume
/// (J/m³) and `κ_fluid` the saturation-weighted fluid conductivity (W/(m K)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpLayout {
    /// Equations (= state variables).
    pub ne: usize,
    /// Phases.
    pub np: usize,
    /// Energy equation present (last equation; temperature is the last state variable).
    pub thermal: bool,
}

impl OpLayout {
    /// Accumulation operator of equation `c`.
    pub fn alpha(&self, c: usize) -> usize {
        c
    }
    /// Flux operator of equation `c`, phase `p`.
    pub fn beta(&self, c: usize, p: usize) -> usize {
        self.ne + c * self.np + p
    }
    /// Phase density (kg/m³), used for gravity.
    pub fn rho(&self, p: usize) -> usize {
        self.ne + self.ne * self.np + p
    }
    /// Phase capillary pressure: phase pressure is `p − Pc_p` (Pa).
    pub fn pc(&self, p: usize) -> usize {
        self.ne + self.ne * self.np + self.np + p
    }
    /// Phase mobility `kr/μ` (1/(Pa·s)), used for well volume rates.
    pub fn lambda(&self, p: usize) -> usize {
        self.ne + self.ne * self.np + 2 * self.np + p
    }
    /// Rock internal energy per unit rock volume (thermal only).
    pub fn rock_energy(&self) -> usize {
        self.ne + self.ne * self.np + 3 * self.np
    }
    /// Saturation-weighted fluid conductivity (thermal only).
    pub fn fluid_cond(&self) -> usize {
        self.ne + self.ne * self.np + 3 * self.np + 1
    }
    /// Total operator count.
    pub fn n_ops(&self) -> usize {
        self.ne + self.ne * self.np + 3 * self.np + if self.thermal { 2 } else { 0 }
    }
}

/// Physics plugged into the OBL engine. State `ω = [p, z_1 … z_{ne−1}]`.
///
/// A CO₂–brine compositional physics implements this trait with components
/// (H₂O, CO₂), phases (aqueous, CO₂-rich) and its flash in `evaluate`
/// (see `fim_physics::Co2Brine`). A thermal physics returns `true` from
/// [`OblPhysics::thermal`]: temperature (K) is then the last state variable
/// and energy the last equation (see [`OpLayout`]).
pub trait OblPhysics {
    /// Number of state variables = number of conservation equations.
    fn n_vars(&self) -> usize;
    /// Number of phases.
    fn n_phases(&self) -> usize;
    /// Phase names (for reporting).
    fn phase_names(&self) -> Vec<String>;
    /// Component (equation) names.
    fn component_names(&self) -> Vec<String>;
    /// Fill `out` (length [`OpLayout::n_ops`]) with the operators at `state`.
    /// Must be finite for every state inside the table bounds.
    fn evaluate(&self, state: &[f64], out: &mut [f64]);
    /// Phase saturations at `state` (length `n_phases`).
    fn saturations(&self, state: &[f64]) -> Vec<f64>;
    /// Energy equation solved (temperature is the last state variable).
    /// Default: isothermal.
    fn thermal(&self) -> bool {
        false
    }
    /// Rock thermal conductivity (W/(m K)); the cell conductivity is
    /// `φ κ_fluid + (1 − φ) κ_rock` with `φ` = pore / bulk volume.
    fn rock_conductivity(&self) -> f64 {
        0.0
    }
    /// Number of property evaluations that failed and used a fallback
    /// (reported in [`FimResult::n_fallbacks`]). Default: none.
    fn n_fallbacks(&self) -> usize {
        0
    }
    /// Operator layout.
    fn layout(&self) -> OpLayout {
        OpLayout { ne: self.n_vars(), np: self.n_phases(), thermal: self.thermal() }
    }
}

/// Multilinear interpolation table of the operators over a box in state
/// space with uniform axes; nodes are evaluated on first use.
pub struct OblTable<'a> {
    phys: &'a dyn OblPhysics,
    lo: Vec<f64>,
    h: Vec<f64>,
    n: Vec<usize>,
    n_ops: usize,
    cache: RefCell<HashMap<usize, Box<[f64]>>>,
}

impl<'a> OblTable<'a> {
    /// Table with `n_points[d] ≥ 2` nodes on `[lo[d], hi[d]]`.
    ///
    /// # Errors
    /// `InvalidInput` for inconsistent bounds or sizes.
    pub fn new(phys: &'a dyn OblPhysics, lo: &[f64], hi: &[f64], n_points: &[usize]) -> SxResult<Self> {
        let nv = phys.n_vars();
        if lo.len() != nv || hi.len() != nv || n_points.len() != nv {
            return Err(SxError::InvalidInput("OBL table: bounds and n_points need one value per state variable".into()));
        }
        if (0..nv).any(|d| !(hi[d] > lo[d]) || n_points[d] < 2 || !lo[d].is_finite() || !hi[d].is_finite()) {
            return Err(SxError::InvalidInput("OBL table: need hi > lo and n_points >= 2".into()));
        }
        Ok(Self {
            phys,
            lo: lo.to_vec(),
            h: (0..nv).map(|d| (hi[d] - lo[d]) / (n_points[d] - 1) as f64).collect(),
            n: n_points.to_vec(),
            n_ops: phys.layout().n_ops(),
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// Number of nodes evaluated so far.
    pub fn n_evaluated(&self) -> usize {
        self.cache.borrow().len()
    }

    fn node(&self, idx: &[usize]) -> Box<[f64]> {
        let mut key = 0usize;
        for d in (0..idx.len()).rev() {
            key = key * self.n[d] + idx[d];
        }
        if let Some(v) = self.cache.borrow().get(&key) {
            return v.clone();
        }
        let x: Vec<f64> = idx.iter().enumerate().map(|(d, &i)| self.lo[d] + self.h[d] * i as f64).collect();
        let mut out = vec![0.0; self.n_ops].into_boxed_slice();
        self.phys.evaluate(&x, &mut out);
        self.cache.borrow_mut().insert(key, out.clone());
        out
    }

    /// Interpolated operators `val` (n_ops) and their derivatives `grad`
    /// (n_ops × n_vars, row-major) at `x` (clamped into the box).
    pub fn interpolate(&self, x: &[f64], val: &mut [f64], grad: &mut [f64]) {
        let nv = x.len();
        let mut base = vec![0usize; nv];
        let mut f = vec![0.0; nv];
        for d in 0..nv {
            let t = ((x[d] - self.lo[d]) / self.h[d]).clamp(0.0, (self.n[d] - 1) as f64);
            let i = (t.floor() as usize).min(self.n[d] - 2);
            base[d] = i;
            f[d] = t - i as f64;
        }
        val.iter_mut().for_each(|v| *v = 0.0);
        grad.iter_mut().for_each(|v| *v = 0.0);
        let mut idx = vec![0usize; nv];
        for corner in 0..(1usize << nv) {
            let mut w = 1.0;
            for d in 0..nv {
                let bit = (corner >> d) & 1;
                idx[d] = base[d] + bit;
                w *= if bit == 1 { f[d] } else { 1.0 - f[d] };
            }
            let v = self.node(&idx);
            #[allow(clippy::needless_range_loop)]
            for d in 0..nv {
                // ∂w/∂x_d: replace factor d by ±1/h.
                let mut dw = if (corner >> d) & 1 == 1 { 1.0 } else { -1.0 } / self.h[d];
                for e in 0..nv {
                    if e != d {
                        dw *= if (corner >> e) & 1 == 1 { f[e] } else { 1.0 - f[e] };
                    }
                }
                for o in 0..self.n_ops {
                    grad[o * nv + d] += dw * v[o];
                }
            }
            for o in 0..self.n_ops {
                val[o] += w * v[o];
            }
        }
    }
}

/// Settings of [`run_fim`].
#[derive(Debug, Clone, PartialEq)]
pub struct FimOptions {
    /// Gravity (m/s²); 0 = off.
    pub gravity: f64,
    /// Table lower bounds per state variable.
    pub table_lo: Vec<f64>,
    /// Table upper bounds per state variable.
    pub table_hi: Vec<f64>,
    /// Table nodes per state variable.
    pub table_points: Vec<usize>,
    /// Injection stream `z_1 … z_{ne−1}` per well (ignored for producers).
    pub injection: Vec<Vec<f64>>,
    /// First time step (s).
    pub dt_init: f64,
    /// Largest time step (s).
    pub dt_max: f64,
    /// Smallest time step (s); a failure below it is an error.
    pub dt_min: f64,
    /// Growth factor after a converged step.
    pub dt_growth: f64,
    /// Cut factor after a failed Newton solve.
    pub dt_cut: f64,
    /// Newton iterations per step.
    pub max_newton: usize,
    /// Tolerance on the normalized residual (mass residual / (PV · cell mass density)).
    pub tol: f64,
    /// Largest change of any mass fraction per Newton iteration (update chopping).
    pub max_dz: f64,
    /// Largest temperature change per Newton iteration (K, thermal only).
    pub max_dtemp: f64,
}

impl Default for FimOptions {
    fn default() -> Self {
        Self {
            gravity: 0.0,
            table_lo: Vec::new(),
            table_hi: Vec::new(),
            table_points: Vec::new(),
            injection: Vec::new(),
            dt_init: 86_400.0,
            dt_max: 30.0 * 86_400.0,
            dt_min: 1e-3,
            dt_growth: 1.5,
            dt_cut: 0.5,
            max_newton: 20,
            tol: 1e-6,
            max_dz: 0.2,
            max_dtemp: 20.0,
        }
    }
}

/// Output of [`run_fim`], at the requested times.
#[derive(Debug, Clone, PartialEq)]
pub struct FimResult {
    /// Output times (s).
    pub times: Vec<f64>,
    /// State `[p, z…]` per time, per cell (n_times × n_cells·n_vars).
    pub state: Vec<Vec<f64>>,
    /// Saturations per time (n_times × n_cells·n_phases).
    pub saturation: Vec<Vec<f64>>,
    /// BHP per time per well (Pa).
    pub bhp: Vec<Vec<f64>>,
    /// Component mass rate per time per well (kg/s, positive = into the
    /// reservoir), well-major: `[w·ne + c]`.
    pub well_mass_rate: Vec<Vec<f64>>,
    /// Component mass in place per time (kg), from the solver's own
    /// accumulation operators.
    pub mass_in_place: Vec<Vec<f64>>,
    /// Cumulative component mass injected minus produced up to each time (kg).
    pub mass_net_in: Vec<Vec<f64>>,
    /// Accepted steps.
    pub n_steps: usize,
    /// Repeated (cut) steps.
    pub n_cuts: usize,
    /// Total Newton iterations.
    pub n_newton: usize,
    /// OBL table nodes evaluated.
    pub n_table_points: usize,
    /// Physics evaluations that failed and used a fallback value
    /// ([`OblPhysics::n_fallbacks`]); nonzero means some table nodes are
    /// not the physics' own values.
    pub n_fallbacks: usize,
}

struct WellData {
    wi: Vec<f64>,
    depth_ref: f64,
    control: WellControl,
    inj: Vec<f64>,
}

struct Eval {
    val: Vec<f64>,
    grad: Vec<f64>,
}

fn eval_all(table: &OblTable, x: &[f64], n: usize, nv: usize, n_ops: usize) -> Vec<Eval> {
    (0..n)
        .map(|i| {
            let mut e = Eval { val: vec![0.0; n_ops], grad: vec![0.0; n_ops * nv] };
            table.interpolate(&x[i * nv..(i + 1) * nv], &mut e.val, &mut e.grad);
            e
        })
        .collect()
}

/// Fully implicit run with OBL operators (experimental).
///
/// `state0` holds `[p, z_1 … z_{ne−1}]` per cell (row-major). Wells use
/// [`FlowWell`] (vertical Peaceman or given `wi`, rate or BHP control,
/// optional `bhp_limit`); `crossflow` is not supported (always off).
///
/// # Errors
/// `InvalidInput`/`Shape` for bad input, `Numerical` when a step fails below
/// `dt_min`.
///
/// Thermal physics: the last state variable is temperature (K) and the last
/// equation energy,
/// `PV₀ (α_E − α_Eⁿ) + V_r (U_r − U_rⁿ) + Δt Σ T β_{E,p} ΔΦ_p + Δt Σ G (T_a − T_b) + Δt q_E = 0`,
/// with `V_r` = bulk − pore volume and the conductance `G` the series
/// combination of the half-cell conductances `κ_cell · geom`. An injector stream then lists `z…` and the injection
/// temperature. Energy rates are W, energy in place J (fluid + rock).
pub fn run_fim(
    model: &FlowModel,
    wells: &[FlowWell],
    phys: &dyn OblPhysics,
    state0: &[f64],
    times: &[f64],
    opts: &FimOptions,
) -> SxResult<FimResult> {
    let lay = phys.layout();
    let th = lay.thermal;
    let ie = lay.ne - 1; // energy equation / temperature variable when thermal
    if th && lay.ne < 2 {
        return Err(SxError::InvalidInput("fim: a thermal physics needs pressure and temperature".into()));
    }
    let (nv, np, n_ops) = (lay.ne, lay.np, lay.n_ops());
    let n = model.n_cells();
    if state0.len() != n * nv {
        return Err(SxError::Shape(format!("fim: state0 needs {} values (n_cells·n_vars)", n * nv)));
    }
    if times.windows(2).any(|w| !(w[1] > w[0])) || times.first().is_some_and(|&t| !(t >= 0.0)) {
        return Err(SxError::InvalidInput("fim: times must be >= 0 and increasing".into()));
    }
    if !(opts.dt_init > 0.0 && opts.dt_max >= opts.dt_min && opts.dt_min > 0.0 && opts.dt_growth >= 1.0 && opts.dt_cut > 0.0 && opts.dt_cut < 1.0 && opts.tol > 0.0 && opts.max_newton > 0 && opts.max_dz > 0.0 && opts.max_dtemp > 0.0) {
        return Err(SxError::InvalidInput("fim: bad time-step or Newton settings".into()));
    }
    if opts.injection.len() != wells.len() || opts.injection.iter().any(|z| z.len() != nv - 1) {
        return Err(SxError::InvalidInput("fim: injection needs one stream (n_vars − 1 values: fractions, then temperature if thermal) per well".into()));
    }
    if wells.iter().any(|w| w.bhp_limit.is_some() && !matches!(w.control, WellControl::Rate(_))) {
        return Err(SxError::InvalidInput("fim: bhp_limit needs a rate-controlled well".into()));
    }
    let table = OblTable::new(phys, &opts.table_lo, &opts.table_hi, &opts.table_points)?;
    let (zlo, zhi): (Vec<f64>, Vec<f64>) = (opts.table_lo.clone(), opts.table_hi.clone());
    let mut wd = Vec::new();
    for (w, inj) in wells.iter().zip(&opts.injection) {
        let wi = well_indices(model, w)?;
        let depth_ref = w.ref_depth_m.unwrap_or(model.center[w.cells[0]][2]);
        wd.push(WellData { wi, depth_ref, control: w.control, inj: inj.clone() });
    }
    let nw = wells.len();
    let nu = n * nv + nw;
    let mut x = state0.to_vec();
    for i in 0..n {
        for d in 0..nv {
            x[i * nv + d] = x[i * nv + d].clamp(zlo[d], zhi[d]);
        }
    }
    // Initial BHP guesses.
    let mut bhp: Vec<f64> = wells
        .iter()
        .map(|w| match w.control {
            WellControl::Bhp(b) => b,
            WellControl::Rate(_) => {
                let pc = w.cells.iter().map(|&c| x[c * nv]).sum::<f64>() / w.cells.len() as f64;
                if w.kind == WellKind::Injector { pc + 1e5 } else { pc - 1e5 }
            }
        })
        .collect();

    let g = opts.gravity;
    let pv: Vec<f64> = model.pore_volume.clone();
    let vr: Vec<f64> = (0..n).map(|i| if th { (model.bulk_volume[i] - pv[i]).max(0.0) } else { 0.0 }).collect();
    let poro: Vec<f64> = (0..n).map(|i| if model.bulk_volume[i] > 0.0 { (pv[i] / model.bulk_volume[i]).clamp(0.0, 1.0) } else { 0.0 }).collect();
    let k_rock = phys.rock_conductivity();
    let mut res = FimResult {
        times: Vec::new(),
        state: Vec::new(),
        saturation: Vec::new(),
        bhp: Vec::new(),
        well_mass_rate: Vec::new(),
        mass_in_place: Vec::new(),
        mass_net_in: Vec::new(),
        n_steps: 0,
        n_cuts: 0,
        n_newton: 0,
        n_table_points: 0,
        n_fallbacks: 0,
    };
    let mut net_in = vec![0.0; nv];
    let mut last_rates = vec![0.0; nw * nv];
    let mut t = 0.0;
    let mut dt = opts.dt_init.min(opts.dt_max);
    let mass_of = |ev: &[Eval]| -> Vec<f64> {
        (0..nv)
            .map(|c| {
                (0..n)
                    .filter(|&i| model.active[i])
                    .map(|i| pv[i] * ev[i].val[lay.alpha(c)] + if th && c == ie { vr[i] * ev[i].val[lay.rock_energy()] } else { 0.0 })
                    .sum()
            })
            .collect()
    };
    let record = |res: &mut FimResult, t: f64, x: &[f64], bhp: &[f64], rates: &[f64], net: &[f64]| {
        let ev = eval_all(&table, x, n, nv, n_ops);
        res.times.push(t);
        res.state.push(x.to_vec());
        res.saturation.push((0..n).flat_map(|i| phys.saturations(&x[i * nv..(i + 1) * nv])).collect());
        res.bhp.push(bhp.to_vec());
        res.well_mass_rate.push(rates.to_vec());
        res.mass_in_place.push(mass_of(&ev));
        res.mass_net_in.push(net.to_vec());
    };
    let mut ctrl: Vec<WellControl> = wd.iter().map(|w| w.control).collect();
    for &t_out in times {
        while t < t_out - 1e-9 * t_out.max(1.0) {
            let step = dt.min(t_out - t);
            let xn = x.clone();
            let ev_n = eval_all(&table, &xn, n, nv, n_ops);
            // Wellbore densities frozen over the step.
            let rho_wb: Vec<f64> = (0..nw)
                .map(|w| {
                    let st: Vec<f64> = if wells[w].kind == WellKind::Injector {
                        std::iter::once(xn[wells[w].cells[0] * nv]).chain(wd[w].inj.iter().copied()).collect()
                    } else {
                        xn[wells[w].cells[0] * nv..(wells[w].cells[0] + 1) * nv].to_vec()
                    };
                    let mut v = vec![0.0; n_ops];
                    let mut gr = vec![0.0; n_ops * nv];
                    table.interpolate(&st, &mut v, &mut gr);
                    let s = phys.saturations(&st);
                    (0..np).map(|p| s[p] * v[lay.rho(p)]).sum()
                })
                .collect();
            let mut bhp_try = bhp.clone();
            let mut ok = false;
            let mut rates = vec![0.0; nw * nv];
            let (mut prev_err, mut relax) = (f64::INFINITY, 1.0f64);
            for _it in 0..opts.max_newton {
                res.n_newton += 1;
                let ev = eval_all(&table, &x, n, nv, n_ops);
                let mut r = vec![0.0; nu];
                let mut jac: Vec<Triplet<usize, usize, f64>> = Vec::new();
                // Accumulation.
                for i in 0..n {
                    if !model.active[i] {
                        for c in 0..nv {
                            jac.push(Triplet::new(i * nv + c, i * nv + c, 1.0));
                        }
                        continue;
                    }
                    for c in 0..nv {
                        let a = lay.alpha(c);
                        r[i * nv + c] = pv[i] * (ev[i].val[a] - ev_n[i].val[a]);
                        for k in 0..nv {
                            jac.push(Triplet::new(i * nv + c, i * nv + k, pv[i] * ev[i].grad[a * nv + k]));
                        }
                    }
                    if th {
                        let u = lay.rock_energy();
                        r[i * nv + ie] += vr[i] * (ev[i].val[u] - ev_n[i].val[u]);
                        for k in 0..nv {
                            jac.push(Triplet::new(i * nv + ie, i * nv + k, vr[i] * ev[i].grad[u * nv + k]));
                        }
                    }
                }
                // TPFA fluxes with phase-potential upwinding.
                for con in &model.connections {
                    let (a, b) = (con.a, con.b);
                    if !(model.active[a] && model.active[b]) {
                        continue;
                    }
                    let tt = step * con.trans;
                    let dz = model.center[a][2] - model.center[b][2];
                    for p in 0..np {
                        let (ea, eb) = (&ev[a], &ev[b]);
                        let rho_bar = 0.5 * (ea.val[lay.rho(p)] + eb.val[lay.rho(p)]);
                        let dphi = (x[a * nv] - ea.val[lay.pc(p)]) - (x[b * nv] - eb.val[lay.pc(p)]) - rho_bar * g * dz;
                        // dΦ/dω_a, dΦ/dω_b.
                        let dphi_da: Vec<f64> = (0..nv)
                            .map(|k| f64::from(u8::from(k == 0)) - ea.grad[lay.pc(p) * nv + k] - 0.5 * g * dz * ea.grad[lay.rho(p) * nv + k])
                            .collect();
                        let dphi_db: Vec<f64> = (0..nv)
                            .map(|k| -f64::from(u8::from(k == 0)) + eb.grad[lay.pc(p) * nv + k] - 0.5 * g * dz * eb.grad[lay.rho(p) * nv + k])
                            .collect();
                        let up = if dphi >= 0.0 { a } else { b };
                        let eu = &ev[up];
                        for c in 0..nv {
                            let bo = lay.beta(c, p);
                            let beta = eu.val[bo];
                            if beta == 0.0 && eu.grad[bo * nv..(bo + 1) * nv].iter().all(|&v| v == 0.0) {
                                continue;
                            }
                            let f = tt * beta * dphi;
                            r[a * nv + c] += f;
                            r[b * nv + c] -= f;
                            for k in 0..nv {
                                let mut da = tt * beta * dphi_da[k];
                                let mut db = tt * beta * dphi_db[k];
                                let du = tt * eu.grad[bo * nv + k] * dphi;
                                if up == a { da += du } else { db += du }
                                jac.push(Triplet::new(a * nv + c, a * nv + k, da));
                                jac.push(Triplet::new(a * nv + c, b * nv + k, db));
                                jac.push(Triplet::new(b * nv + c, a * nv + k, -da));
                                jac.push(Triplet::new(b * nv + c, b * nv + k, -db));
                            }
                        }
                    }
                }
                // Heat conduction: G = ga gb / (ga + gb), g = κ_cell · geom.
                if th {
                    let kf = lay.fluid_cond();
                    for con in &model.connections {
                        let (a, b) = (con.a, con.b);
                        if !(model.active[a] && model.active[b]) {
                            continue;
                        }
                        let ka = poro[a] * ev[a].val[kf] + (1.0 - poro[a]) * k_rock;
                        let kb = poro[b] * ev[b].val[kf] + (1.0 - poro[b]) * k_rock;
                        let (ga, gb) = (ka * con.geom[0], kb * con.geom[1]);
                        if !(ga > 0.0 && gb > 0.0) {
                            continue;
                        }
                        let gab = ga * gb / (ga + gb);
                        let dtemp = x[a * nv + ie] - x[b * nv + ie];
                        let f = step * gab * dtemp;
                        r[a * nv + ie] += f;
                        r[b * nv + ie] -= f;
                        // ∂G/∂ga = gb²/(ga+gb)², ga = (φ κ_f + …) geom.
                        let dga = gb * gb / ((ga + gb) * (ga + gb)) * con.geom[0] * poro[a];
                        let dgb = ga * ga / ((ga + gb) * (ga + gb)) * con.geom[1] * poro[b];
                        for k in 0..nv {
                            let mut da = step * dtemp * dga * ev[a].grad[kf * nv + k];
                            let mut db = step * dtemp * dgb * ev[b].grad[kf * nv + k];
                            if k == ie {
                                da += step * gab;
                                db -= step * gab;
                            }
                            jac.push(Triplet::new(a * nv + ie, a * nv + k, da));
                            jac.push(Triplet::new(a * nv + ie, b * nv + k, db));
                            jac.push(Triplet::new(b * nv + ie, a * nv + k, -da));
                            jac.push(Triplet::new(b * nv + ie, b * nv + k, -db));
                        }
                    }
                }
                // Wells.
                rates.iter_mut().for_each(|v| *v = 0.0);
                let mut wres = vec![0.0; nw];
                for w in 0..nw {
                    let col = n * nv + w;
                    let well = &wells[w];
                    let mut vol = 0.0;
                    let mut dvol_dbhp = 0.0;
                    let mut dvol_open = 0.0;
                    let mut dvol_dx: Vec<(usize, f64)> = Vec::new();
                    for (j, &i) in well.cells.iter().enumerate() {
                        let wi = wd[w].wi[j];
                        let pwb = bhp_try[w] + rho_wb[w] * g * (model.center[i][2] - wd[w].depth_ref);
                        if well.kind == WellKind::Injector {
                            let st: Vec<f64> = std::iter::once(x[i * nv]).chain(wd[w].inj.iter().copied()).collect();
                            let mut v = vec![0.0; n_ops];
                            let mut gr = vec![0.0; n_ops * nv];
                            table.interpolate(&st, &mut v, &mut gr);
                            let dp = pwb - x[i * nv];
                            let lt: f64 = (0..np).map(|p| v[lay.lambda(p)]).sum();
                            dvol_open += wi * lt;
                            if dp < 0.0 {
                                continue;
                            }
                            vol += wi * lt * dp;
                            dvol_dbhp += wi * lt;
                            let dlt: f64 = (0..np).map(|p| gr[lay.lambda(p) * nv]).sum();
                            dvol_dx.push((i * nv, wi * (dlt * dp - lt)));
                            for c in 0..nv {
                                let bt: f64 = (0..np).map(|p| v[lay.beta(c, p)]).sum();
                                let dbt: f64 = (0..np).map(|p| gr[lay.beta(c, p) * nv]).sum();
                                let q = wi * bt * dp; // into the cell
                                rates[w * nv + c] += q;
                                r[i * nv + c] -= step * q;
                                jac.push(Triplet::new(i * nv + c, i * nv, -step * wi * (dbt * dp - bt)));
                                jac.push(Triplet::new(i * nv + c, col, -step * wi * bt));
                            }
                        } else {
                            let e = &ev[i];
                            for p in 0..np {
                                let dp = x[i * nv] - e.val[lay.pc(p)] - pwb;
                                let lam = e.val[lay.lambda(p)];
                                dvol_open += wi * lam;
                                if dp < 0.0 {
                                    continue;
                                }
                                vol += wi * lam * dp;
                                dvol_dbhp -= wi * lam;
                                for k in 0..nv {
                                    let ddp = f64::from(u8::from(k == 0)) - e.grad[lay.pc(p) * nv + k];
                                    dvol_dx.push((i * nv + k, wi * (e.grad[lay.lambda(p) * nv + k] * dp + lam * ddp)));
                                }
                                for c in 0..nv {
                                    let bo = lay.beta(c, p);
                                    let beta = e.val[bo];
                                    let q = wi * beta * dp; // out of the cell
                                    rates[w * nv + c] -= q;
                                    r[i * nv + c] += step * q;
                                    for k in 0..nv {
                                        let ddp = f64::from(u8::from(k == 0)) - e.grad[lay.pc(p) * nv + k];
                                        jac.push(Triplet::new(i * nv + c, i * nv + k, step * wi * (e.grad[bo * nv + k] * dp + beta * ddp)));
                                    }
                                    jac.push(Triplet::new(i * nv + c, col, -step * wi * beta));
                                }
                            }
                        }
                    }
                    match ctrl[w] {
                        WellControl::Bhp(b) => {
                            wres[w] = (bhp_try[w] - b) / 1e5;
                            jac.push(Triplet::new(col, col, 1e-5));
                        }
                        WellControl::Rate(q) => {
                            let s = 1.0 / q.abs().max(1e-30);
                            wres[w] = (vol - q.abs()) * s;
                            // Closed completions: use the open-well slope so
                            // the BHP row is never empty.
                            let d = if dvol_dbhp != 0.0 { dvol_dbhp } else if well.kind == WellKind::Injector { dvol_open } else { -dvol_open };
                            jac.push(Triplet::new(col, col, d * s));
                            for (k, v) in dvol_dx {
                                jac.push(Triplet::new(col, k, v * s));
                            }
                        }
                    }
                    r[col] = wres[w];
                }
                // Convergence check: mass residuals normalized by the cell mass,
                // the energy residual by the cell heat capacity (i.e. in K).
                let mut err = 0.0f64;
                let nm = if th { nv - 1 } else { nv };
                for i in (0..n).filter(|&i| model.active[i]) {
                    let m: f64 = (0..nm).map(|c| ev[i].val[lay.alpha(c)]).sum::<f64>().max(1e-12);
                    for c in 0..nm {
                        err = err.max(r[i * nv + c].abs() / (pv[i] * m));
                    }
                    if th {
                        let cap = (pv[i] * ev[i].grad[lay.alpha(ie) * nv + ie] + vr[i] * ev[i].grad[lay.rock_energy() * nv + ie]).abs().max(1e-12);
                        err = err.max(r[i * nv + ie].abs() / cap);
                    }
                }
                for r in &wres {
                    err = err.max(r.abs());
                }
                if !err.is_finite() {
                    break;
                }
                if err < opts.tol {
                    ok = true;
                    break;
                }
                // Damping against Newton cycling on the piecewise-multilinear
                // operators: halve the update while the residual stalls.
                relax = if err > 0.9 * prev_err { (relax * 0.5).max(1.0 / 16.0) } else { (relax * 2.0).min(1.0) };
                prev_err = err;
                if jac.iter().any(|t| !t.val.is_finite()) {
                    break;
                }
                let a = match SparseColMat::<usize, f64>::try_new_from_triplets(nu, nu, &jac) { Ok(a) => a, Err(e) => { dbg_msg(&format!("matrix: {e:?}")); break } };
                let lu = match a.sp_lu() { Ok(l) => l, Err(e) => { dbg_msg(&format!("lu: {e:?}")); break } };
                let rhs = Mat::<f64>::from_fn(nu, 1, |i, _| -r[i]);
                let dx = lu.solve(&rhs);
                if (0..nu).any(|i| !dx[(i, 0)].is_finite()) {
                    dbg_msg("non-finite update");
                    break;
                }
                // Update chopping on the fractions.
                let mut maxdz = 0.0f64;
                let mut maxdt = 0.0f64;
                for i in 0..n {
                    for d in 1..nm {
                        maxdz = maxdz.max(dx[(i * nv + d, 0)].abs());
                    }
                    if th {
                        maxdt = maxdt.max(dx[(i * nv + ie, 0)].abs());
                    }
                }
                let mut scale = relax * if maxdz > opts.max_dz { opts.max_dz / maxdz } else { 1.0 };
                if maxdt * scale > opts.max_dtemp {
                    scale = opts.max_dtemp / maxdt;
                }
                for i in 0..n {
                    for d in 0..nv {
                        let v = x[i * nv + d] + scale * dx[(i * nv + d, 0)];
                        x[i * nv + d] = v.clamp(zlo[d], zhi[d]);
                    }
                }
                dbg_msg(&format!("dp max {:.3e} dbhp {:?} scale {scale}", (0..n).map(|i| dx[(i * nv, 0)].abs()).fold(0.0, f64::max), (0..nw).map(|w| dx[(n * nv + w, 0)]).collect::<Vec<_>>()));
                for w in 0..nw {
                    bhp_try[w] += scale * dx[(n * nv + w, 0)];
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
                for c in 0..nv {
                    net_in[c] += step * (0..nw).map(|w| rates[w * nv + c]).sum::<f64>();
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
                    return Err(SxError::Numerical(format!("fim: Newton failed at t = {t:.6e} s with dt below dt_min")));
                }
            }
        }
        record(&mut res, t_out, &x, &bhp, &last_rates, &net_in);
    }
    res.n_table_points = table.n_evaluated();
    res.n_fallbacks = phys.n_fallbacks();
    Ok(res)
}

fn dbg_msg(m: &str) {
    if std::env::var_os("SX_FIM_DEBUG").is_some() {
        eprintln!("fim: {m}");
    }
}
