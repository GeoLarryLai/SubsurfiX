// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Two-phase water/oil IMPES with gravity, capillary pressure and slightly
//! compressible rock and fluids (transient pressure).
//!
//! Pressure unknown: oil pressure `p` per cell; water pressure `p − Pc(Sw)`.
//! Depth `z` is positive down, so the phase potential is `Φ = p − ρ g z`.
//! Each step (backward Euler in pressure, explicit in saturation):
//!
//! ```text
//! PV c_t (p − pⁿ)/Δt + Σ_f T [λw (Δp − ΔPc − ρw g Δz) + λo (Δp − ρo g Δz)] = q
//! c_t = c_r + Sw c_w + So c_o
//! ```
//!
//! with phase mobilities upwinded per face by the sign of the phase
//! potential difference (re-checked after the solve, up to 4 passes). The
//! water mass `PV(p) Sw b_w(p)` is then updated explicitly with the same face
//! fluxes; `PV = PV₀ (1 + c_r (p − p_ref))`, `b = 1 + c (p − p_ref)`.
//! With compressibility on, water and oil masses `Mw`, `Mo` are both carried
//! per cell and updated with their upwind face fluxes, `Sw = Mw / (PV bw)`, and
//! the pressure step adds the volume-balance correction
//! `−(PV(pⁿ) − Mw/bw(pⁿ) − Mo/bo(pⁿ))/Δt` to the right-hand side, so both phases
//! are conserved to round-off and the saturation misfit is driven back to zero
//! each step. Sw is only bounded to the physical [0, 1]; it may leave
//! [Swc, 1 − Sor] (immobile phases); any mass removed by that bound is reported
//! in `water_clamp_loss`.
//! Time steps follow a Courant (CFL) limit from the mobility derivatives and
//! the capillary diffusion `λw |dPc/dSw|`.
//!
//! Wells: wellbore pressure at a completion `p_wb = BHP + ρ_wb g (z − z_ref)`,
//! `ρ_wb` = water density for injectors and the mobility-weighted mixture of
//! the completed cells for producers. Capillary pressure is not applied at
//! wells or fixed-pressure cells.

use std::collections::BTreeMap;

use super::model::FlowModel;
use super::pressure::{completion_direction, solve_spd, switch_controls, well_indices, FixedPressure, FlowWell, WellControl, WellKind};
use super::transport::Corey;
use crate::error::{SxError, SxResult};

/// Normalized saturation floor used by the capillary curves (Brooks-Corey and
/// van Genuchten diverge at Se = 0).
pub const SE_MIN: f64 = 0.02;

/// Capillary pressure `Pc = p_o − p_w` (Pa) as a function of the normalized
/// water saturation `Se = (Sw − Swc)/(1 − Swc − Sor)` (clamped to
/// [`SE_MIN`], 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Capillary {
    /// No capillary pressure.
    None,
    /// Brooks-Corey: `Pc = pd Se^(−1/λ)`; entry pressure `pd` (Pa), pore-size index `λ`.
    BrooksCorey {
        /// Entry pressure (Pa).
        pd: f64,
        /// Pore-size distribution index.
        lambda: f64,
    },
    /// van Genuchten: `Pc = (Se^(−1/m) − 1)^(1/n) / α`, `m = 1 − 1/n`; `α` in 1/Pa.
    VanGenuchten {
        /// α (1/Pa).
        alpha: f64,
        /// n (> 1).
        n: f64,
    },
}

impl Capillary {
    /// Pc (Pa) at normalized saturation `se`.
    pub fn pc_se(&self, se: f64) -> f64 {
        let se = se.clamp(SE_MIN, 1.0);
        match *self {
            Self::None => 0.0,
            Self::BrooksCorey { pd, lambda } => pd * se.powf(-1.0 / lambda),
            Self::VanGenuchten { alpha, n } => {
                let m = 1.0 - 1.0 / n;
                (se.powf(-1.0 / m) - 1.0).max(0.0).powf(1.0 / n) / alpha
            }
        }
    }

    /// Pc (Pa) at water saturation `sw` with the end points of `corey`.
    pub fn pc(&self, corey: &Corey, sw: f64) -> f64 {
        self.pc_se((sw - corey.swc) / (1.0 - corey.swc - corey.sor))
    }

    /// Water saturation where `Pc = pc` (inverse, by bisection; Pc is
    /// non-increasing in Sw). Below the curve's minimum the result is `1 − Sor`.
    pub fn sw_at(&self, corey: &Corey, pc: f64) -> f64 {
        let (a, b) = (corey.swc, 1.0 - corey.sor);
        if matches!(self, Self::None) || pc <= self.pc(corey, b) {
            return b;
        }
        let (mut lo, mut hi) = (a, b);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if self.pc(corey, mid) > pc {
                lo = mid
            } else {
                hi = mid
            }
        }
        0.5 * (lo + hi)
    }

    fn validate(&self) -> SxResult<()> {
        let ok = match *self {
            Self::None => true,
            Self::BrooksCorey { pd, lambda } => {
                pd.is_finite() && pd >= 0.0 && lambda.is_finite() && lambda > 0.0
            }
            Self::VanGenuchten { alpha, n } => {
                alpha.is_finite() && alpha > 0.0 && n.is_finite() && n > 1.0
            }
        };
        if ok {
            Ok(())
        } else {
            Err(SxError::InvalidInput("capillary: need pd >= 0, lambda > 0 (Brooks-Corey) or alpha > 0, n > 1 (van Genuchten)".into()))
        }
    }
}

/// Physics switches of [`run_two_phase`]. The default has everything off
/// (no gravity, no capillary pressure, incompressible).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Physics {
    /// Gravity (m/s²); 0 = off, 9.80665 = Earth.
    pub gravity: f64,
    /// Water density at `p_ref` (kg/m³).
    pub rho_w: f64,
    /// Oil density at `p_ref` (kg/m³).
    pub rho_o: f64,
    /// Capillary pressure curve.
    pub capillary: Capillary,
    /// Rock (pore volume) compressibility (1/Pa).
    pub c_rock: f64,
    /// Water compressibility (1/Pa).
    pub c_w: f64,
    /// Oil compressibility (1/Pa).
    pub c_o: f64,
    /// Reference pressure of pore volumes and densities (Pa).
    pub p_ref: f64,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            gravity: 0.0,
            rho_w: 1000.0,
            rho_o: 800.0,
            capillary: Capillary::None,
            c_rock: 0.0,
            c_w: 0.0,
            c_o: 0.0,
            p_ref: 1e7,
        }
    }
}

impl Physics {
    fn compressible(&self) -> bool {
        self.c_rock > 0.0 || self.c_w > 0.0 || self.c_o > 0.0
    }
    fn validate(&self) -> SxResult<()> {
        self.capillary.validate()?;
        let ok = [
            self.gravity,
            self.rho_w,
            self.rho_o,
            self.c_rock,
            self.c_w,
            self.c_o,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
            && self.p_ref.is_finite();
        if ok {
            Ok(())
        } else {
            Err(SxError::InvalidInput(
                "physics: gravity, densities and compressibilities must be finite and >= 0".into(),
            ))
        }
    }
}

/// Options of [`run_two_phase`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPhaseOptions {
    /// Courant number (≤ 1).
    pub cfl: f64,
    /// Longest step (s).
    pub max_dt: f64,
    /// Water fraction of fluid entering through fixed-pressure cells.
    pub boundary_fw: f64,
    /// Target largest saturation change per step: the next step is scaled
    /// by `0.9 · ds_target / max|ΔSw|` (change-based control on top of the
    /// CFL limit; infinite = off).
    pub ds_target: f64,
    /// Target largest pressure change per step (Pa), as `ds_target`; a step
    /// changing pressure by more than twice the target is repeated shorter.
    pub dp_target: f64,
}

impl Default for TwoPhaseOptions {
    fn default() -> Self {
        Self {
            cfl: 0.5,
            max_dt: f64::INFINITY,
            boundary_fw: 1.0,
            ds_target: f64::INFINITY,
            dp_target: f64::INFINITY,
        }
    }
}

/// Result of [`run_two_phase`] at each report time.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPhaseResult {
    /// Report times (s).
    pub times: Vec<f64>,
    /// Water saturation per report, per cell.
    pub sw: Vec<Vec<f64>>,
    /// Oil pressure per report, per cell (Pa).
    pub p: Vec<Vec<f64>>,
    /// Bottom-hole pressure per report, per well (Pa).
    pub bhp: Vec<Vec<f64>>,
    /// Rate well at its BHP limit, per report, per well.
    pub bhp_limited: Vec<Vec<bool>>,
    /// Water rate per report, per well (m³/s reservoir, + into the reservoir).
    pub water_rate: Vec<Vec<f64>>,
    /// Oil rate per report, per well (m³/s reservoir, + into the reservoir).
    pub oil_rate: Vec<Vec<f64>>,
    /// Water volume at `p_ref` in place per report (m³).
    pub water_in_place: Vec<f64>,
    /// Cumulative water mass removed by the physical bound 0 ≤ Sw ≤ 1 (reference volume, m³); zero in normal runs.
    pub water_clamp_loss: Vec<f64>,
    /// Oil in place `Σ PV(p)(1 − Sw) b_o(p)` (reference volume, m³). Oil is the complement of water, so its
    /// balance closes only to the linearization error of the pressure step (diagnostic).
    pub oil_in_place: Vec<f64>,
    /// Cumulative net oil inflow through wells and fixed-pressure cells (reference volume, m³).
    pub oil_net_in: Vec<f64>,
    /// Cumulative water injected minus produced at `p_ref`, per report (m³).
    pub water_net_in: Vec<f64>,
    /// Steps (pressure solves that were accepted).
    pub n_steps: usize,
    /// Steps repeated by the pressure-change control.
    pub n_rejected: usize,
}

struct Step {
    p: Vec<f64>,
    /// Water / oil flux per connection a→b (m³/s).
    fw: Vec<f64>,
    fo: Vec<f64>,
    /// Water / oil rate per well per completion (+ in).
    well_w: Vec<Vec<f64>>,
    well_o: Vec<Vec<f64>>,
    bhp: Vec<f64>,
    /// Rate well running at its BHP limit.
    limited: Vec<bool>,
    /// Water flux in per fixed-pressure entry.
    fix_w: Vec<f64>,
    fix_o: Vec<f64>,
}

struct Ctx<'a> {
    m: &'a FlowModel,
    wells: &'a [FlowWell],
    fixed: &'a [FixedPressure],
    c: &'a Corey,
    ph: &'a Physics,
    wis: Vec<Vec<f64>>,
    z_ref: Vec<f64>,
    bw_fixed: f64,
}

impl Ctx<'_> {
    fn mob(&self, s: f64) -> (f64, f64) {
        self.c.mobility(s)
    }
    fn pc(&self, s: f64) -> f64 {
        self.ph.capillary.pc(self.c, s)
    }
    fn z(&self, c: usize) -> f64 {
        self.m.center[c][2]
    }

    /// One pressure solve with rate ⇄ BHP-limit switching.
    fn solve(
        &self,
        pn: &[f64],
        s: &[f64],
        p_guess: &[f64],
        dt: f64,
        vol_err: Option<&[f64]>,
    ) -> SxResult<Step> {
        let mut ctrl: Vec<WellControl> = self.wells.iter().map(|w| w.control).collect();
        let max_switch = 2 * self.wells.len() + 2;
        let mut round = 0;
        loop {
            let mut st = self.solve_ctrl(pn, s, p_guess, dt, vol_err, &ctrl)?;
            let res: Vec<(f64, f64)> = st
                .bhp
                .iter()
                .zip(st.well_w.iter().zip(&st.well_o))
                .map(|(&b, (w, o))| (b, w.iter().sum::<f64>() + o.iter().sum::<f64>()))
                .collect();
            if round == max_switch || !switch_controls(self.wells, &mut ctrl, &res) {
                st.limited = self
                    .wells
                    .iter()
                    .zip(&ctrl)
                    .map(|(w, c)| matches!(w.control, WellControl::Rate(_)) && matches!(c, WellControl::Bhp(_)))
                    .collect();
                return Ok(st);
            }
            round += 1;
        }
    }

    /// One pressure solve over `dt` from (pn, s) with well controls `ctrl`;
    /// `p_guess` sets the upwind directions.
    #[allow(clippy::too_many_arguments)]
    fn solve_ctrl(
        &self,
        pn: &[f64],
        s: &[f64],
        p_guess: &[f64],
        dt: f64,
        vol_err: Option<&[f64]>,
        ctrl: &[WellControl],
    ) -> SxResult<Step> {
        let m = self.m;
        let n = m.n_cells();
        let g = self.ph.gravity;
        let (rw, ro) = (self.ph.rho_w, self.ph.rho_o);
        let mob: Vec<(f64, f64)> = s.iter().map(|&v| self.mob(v)).collect();
        let pc: Vec<f64> = s.iter().map(|&v| self.pc(v)).collect();
        let inj = self.c.krw_max / self.c.mu_w;
        // Wellbore densities.
        let rho_wb: Vec<f64> = self
            .wells
            .iter()
            .map(|w| match w.kind {
                WellKind::Injector => rw,
                WellKind::Producer => {
                    let (mut num, mut den) = (0.0, 0.0);
                    for &c in &w.cells {
                        let (a, b) = mob[c];
                        num += a * rw + b * ro;
                        den += a + b;
                    }
                    if den > 0.0 {
                        num / den
                    } else {
                        ro
                    }
                }
            })
            .collect();
        let mut guess = p_guess.to_vec();
        // Per completion: (total mobility, water fraction if flowing into the reservoir); None = along the well.
        let mut comp: Vec<Vec<Option<(f64, f64)>>> = self.wells.iter().map(|w| vec![None; w.cells.len()]).collect();
        for _pass in 0..4 {
            // Face mobilities upwinded from `guess`.
            let faces: Vec<(f64, f64)> = m
                .connections
                .iter()
                .map(|cn| {
                    let dz = self.z(cn.a) - self.z(cn.b);
                    let dpw = guess[cn.a] - pc[cn.a] - guess[cn.b] + pc[cn.b] - rw * g * dz;
                    let dpo = guess[cn.a] - guess[cn.b] - ro * g * dz;
                    (
                        if dpw >= 0.0 { mob[cn.a].0 } else { mob[cn.b].0 },
                        if dpo >= 0.0 { mob[cn.a].1 } else { mob[cn.b].1 },
                    )
                })
                .collect();
            let mut n_unk = n;
            let unk: Vec<Option<usize>> = self
                .wells
                .iter()
                .zip(ctrl)
                .map(|(_, c)| {
                    if matches!(c, WellControl::Rate(_)) {
                        n_unk += 1;
                        Some(n_unk - 1)
                    } else {
                        None
                    }
                })
                .collect();
            let mut lower: BTreeMap<(usize, usize), f64> = BTreeMap::new();
            let mut rhs = vec![0.0; n_unk];
            let mut add = |r: usize, c: usize, v: f64| {
                let key = if r >= c { (r, c) } else { (c, r) };
                *lower.entry(key).or_insert(0.0) += v;
            };
            for c in 0..n {
                if !m.active[c] {
                    add(c, c, 1.0);
                    continue;
                }
                if self.ph.compressible() && dt.is_finite() {
                    let ct = self.ph.c_rock + s[c] * self.ph.c_w + (1.0 - s[c]) * self.ph.c_o;
                    let a = m.pore_volume[c] * ct / dt;
                    add(c, c, a);
                    rhs[c] += a * pn[c];
                    // Volume-balance correction: drive PV(p) − Vw − Vo back to zero this step.
                    if let Some(v) = vol_err {
                        rhs[c] -= v[c] / dt;
                    }
                }
            }
            for (cn, &(lw, lo)) in m.connections.iter().zip(&faces) {
                let t = cn.trans;
                add(cn.a, cn.a, t * (lw + lo));
                add(cn.b, cn.b, t * (lw + lo));
                add(cn.a, cn.b, -t * (lw + lo));
                let dz = self.z(cn.a) - self.z(cn.b);
                // Outflow from a has + (lw+lo) Δp − lw ΔPc − (lw ρw + lo ρo) g Δz.
                let extra = t * (lw * (pc[cn.a] - pc[cn.b]) + (lw * rw + lo * ro) * g * dz);
                rhs[cn.a] += extra;
                rhs[cn.b] -= extra;
            }
            for f in self.fixed {
                let (a, b) = mob[f.cell];
                let t = f.trans * (a + b);
                add(f.cell, f.cell, t);
                rhs[f.cell] += t * f.pressure;
            }
            for (wi_, ((w, wi), u)) in self.wells.iter().zip(&self.wis).zip(&unk).enumerate() {
                for (k, (&c, &wic)) in w.cells.iter().zip(wi).enumerate() {
                    let lam = match (comp[wi_][k], w.kind) {
                        (Some((l, _)), _) => l,
                        (None, WellKind::Injector) => inj,
                        (None, WellKind::Producer) => mob[c].0 + mob[c].1,
                    };
                    let t = wic * lam;
                    let head = rho_wb[wi_] * g * (self.z(c) - self.z_ref[wi_]);
                    add(c, c, t);
                    rhs[c] += t * head;
                    match (ctrl[wi_], u) {
                        (WellControl::Bhp(pw), _) => rhs[c] += t * pw,
                        (WellControl::Rate(_), Some(u)) => {
                            add(*u, *u, t);
                            add(*u, c, -t);
                            rhs[*u] -= t * head;
                        }
                        _ => {}
                    }
                }
                if let (WellControl::Rate(q), Some(u)) = (ctrl[wi_], u) {
                    rhs[*u] += match w.kind {
                        WellKind::Injector => q,
                        WellKind::Producer => -q,
                    };
                }
            }
            let x = solve_spd(n_unk, &lower, rhs)?;
            let p: Vec<f64> = (0..n)
                .map(|c| if m.active[c] { x[c] } else { f64::NAN })
                .collect();
            // Upwind consistency.
            let consistent = m.connections.iter().zip(&faces).all(|(cn, &(lw, lo))| {
                let dz = self.z(cn.a) - self.z(cn.b);
                let dpw = p[cn.a] - pc[cn.a] - p[cn.b] + pc[cn.b] - rw * g * dz;
                let dpo = p[cn.a] - p[cn.b] - ro * g * dz;
                let uw = if dpw >= 0.0 { mob[cn.a].0 } else { mob[cn.b].0 };
                let uo = if dpo >= 0.0 { mob[cn.a].1 } else { mob[cn.b].1 };
                uw == lw && uo == lo
            });
            // Completion directions and crossflow mobilities from this solution.
            let mut bhp = Vec::new();
            let mut next_comp = Vec::with_capacity(self.wells.len());
            for (wi_, ((w, wi), u)) in self.wells.iter().zip(&self.wis).zip(&unk).enumerate() {
                let b = match (ctrl[wi_], u) {
                    (WellControl::Bhp(pw), _) => pw,
                    (_, Some(u)) => x[*u],
                    _ => f64::NAN,
                };
                let pwb = |c: usize| b + rho_wb[wi_] * g * (self.z(c) - self.z_ref[wi_]);
                // Producer wellbore mixture: total mobility and water fraction of the inflowing completions.
                let (mut ql, mut qsum, mut qwat) = (0.0, 0.0, 0.0);
                for (&c, &wic) in w.cells.iter().zip(wi) {
                    if completion_direction(w, pwb(c), p[c]) {
                        let (a, o) = mob[c];
                        let q = wic * (a + o) * (p[c] - pwb(c)).abs();
                        ql += q * (a + o);
                        qsum += q;
                        qwat += wic * a * (p[c] - pwb(c)).abs();
                    }
                }
                let cells_mob = || {
                    let (a, o) = w.cells.iter().fold((0.0, 0.0), |acc, &c| (acc.0 + mob[c].0, acc.1 + mob[c].1));
                    ((a + o) / w.cells.len() as f64, a / (a + o).max(1e-300))
                };
                let mix = if qsum > 0.0 { (ql / qsum, qwat / qsum) } else { cells_mob() };
                next_comp.push(
                    w.cells
                        .iter()
                        .map(|&c| {
                            if completion_direction(w, pwb(c), p[c]) {
                                None
                            } else if !w.crossflow {
                                Some((0.0, 0.0))
                            } else {
                                Some(match w.kind {
                                    WellKind::Injector => (mob[c].0 + mob[c].1, mob[c].0 / (mob[c].0 + mob[c].1).max(1e-300)),
                                    WellKind::Producer => mix,
                                })
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                bhp.push(b);
            }
            let comp_settled = next_comp.iter().zip(&comp).all(|(a, b)| {
                a.iter().zip(b).all(|(x, y)| match (x, y) {
                    (None, None) => true,
                    (Some(x), Some(y)) => (x.0 - y.0).abs() <= 1e-10 * x.0.abs().max(y.0.abs()) && (x.1 - y.1).abs() <= 1e-10,
                    _ => false,
                })
            });
            if (!consistent || !comp_settled) && _pass < 3 {
                guess = p;
                comp = next_comp;
                continue;
            }
            let mut fw = Vec::with_capacity(faces.len());
            let mut fo = Vec::with_capacity(faces.len());
            for (cn, &(lw, lo)) in m.connections.iter().zip(&faces) {
                let dz = self.z(cn.a) - self.z(cn.b);
                fw.push(cn.trans * lw * (p[cn.a] - pc[cn.a] - p[cn.b] + pc[cn.b] - rw * g * dz));
                fo.push(cn.trans * lo * (p[cn.a] - p[cn.b] - ro * g * dz));
            }
            let mut well_w = Vec::new();
            let mut well_o = Vec::new();
            for (wi_, (w, wi)) in self.wells.iter().zip(&self.wis).enumerate() {
                let b = bhp[wi_];
                let (mut qw, mut qo) = (Vec::new(), Vec::new());
                for (k, (&c, &wic)) in w.cells.iter().zip(wi).enumerate() {
                    let pwb = b + rho_wb[wi_] * g * (self.z(c) - self.z_ref[wi_]);
                    let (a, o) = mob[c];
                    // Same mobility as the matrix this pass, so the well rates balance exactly.
                    let (lam, f) = match (comp[wi_][k], w.kind) {
                        (Some(lf), _) => lf,
                        (None, WellKind::Injector) => (inj, 1.0),
                        (None, WellKind::Producer) => (a + o, a / (a + o).max(1e-300)),
                    };
                    let q = wic * lam * (pwb - p[c]);
                    qw.push(q * f);
                    qo.push(q * (1.0 - f));
                }
                well_w.push(qw);
                well_o.push(qo);
            }
            let mut fix_w = Vec::new();
            let mut fix_o = Vec::new();
            for f in self.fixed {
                let (a, o) = mob[f.cell];
                let q = f.trans * (a + o) * (f.pressure - p[f.cell]);
                let fr = if q > 0.0 {
                    self.bw_fixed
                } else {
                    a / (a + o).max(1e-300)
                };
                fix_w.push(q * fr);
                fix_o.push(q * (1.0 - fr));
            }
            return Ok(Step {
                p,
                fw,
                fo,
                well_w,
                well_o,
                bhp,
                limited: vec![false; self.wells.len()],
                fix_w,
                fix_o,
            });
        }
        unreachable!("loop returns on the last pass")
    }
}

fn max_deriv(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let n = 400;
    let h = (b - a) / n as f64;
    (0..n)
        .map(|i| ((f(a + h * (i + 1) as f64) - f(a + h * i as f64)) / h).abs())
        .fold(0.0, f64::max)
}

/// Two-phase water/oil IMPES with gravity, capillary pressure and slight
/// compressibility; see the module notes. `p0`: initial oil pressure per
/// cell (Pa). Injectors inject water.
///
/// # Errors
/// `Shape` for wrong lengths; `InvalidInput` for bad parameters or times, or
/// an incompressible model with no BHP well or fixed pressure; `Numerical`
/// when a solve fails.
#[allow(clippy::too_many_arguments)]
pub fn run_two_phase(
    model: &FlowModel,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    corey: &Corey,
    physics: &Physics,
    sw0: &[f64],
    p0: &[f64],
    times: &[f64],
    opts: &TwoPhaseOptions,
) -> SxResult<TwoPhaseResult> {
    corey.validate()?;
    super::pressure::check_controls(wells, fixed)?;
    physics.validate()?;
    let n = model.n_cells();
    if sw0.len() != n || p0.len() != n {
        return Err(SxError::Shape(
            "two-phase: sw0 and p0 need one value per cell".into(),
        ));
    }
    if !(opts.cfl > 0.0 && opts.cfl <= 1.0)
        || !(opts.ds_target > 0.0)
        || !(opts.dp_target > 0.0)
        || !(opts.max_dt > 0.0)
        || !(0.0..=1.0).contains(&opts.boundary_fw)
    {
        return Err(SxError::InvalidInput(
            "two-phase: cfl in (0, 1], max_dt > 0, boundary_fw in [0, 1], ds_target > 0, dp_target > 0".into(),
        ));
    }
    if times.windows(2).any(|w| !(w[1] > w[0])) || times.first().is_some_and(|&t| !(t >= 0.0)) {
        return Err(SxError::InvalidInput(
            "report times must be >= 0 and strictly increasing".into(),
        ));
    }
    if !physics.compressible()
        && fixed.is_empty()
        && !wells
            .iter()
            .any(|w| matches!(w.control, WellControl::Bhp(_)))
    {
        return Err(SxError::InvalidInput(
            "incompressible two-phase run needs a BHP well or a fixed-pressure cell".into(),
        ));
    }
    if p0
        .iter()
        .zip(&model.active)
        .any(|(p, &a)| a && !p.is_finite())
    {
        return Err(SxError::InvalidInput(
            "two-phase: p0 must be finite on active cells".into(),
        ));
    }
    let wis = wells
        .iter()
        .map(|w| well_indices(model, w))
        .collect::<SxResult<Vec<_>>>()?;
    for f in fixed {
        if f.cell >= n || !model.active[f.cell] {
            return Err(SxError::InvalidInput(format!(
                "fixed pressure: cell {} is outside the grid or inactive",
                f.cell
            )));
        }
    }
    let z_ref = wells
        .iter()
        .map(|w| w.ref_depth_m.unwrap_or(model.center[w.cells[0]][2]))
        .collect();
    let ctx = Ctx {
        m: model,
        wells,
        fixed,
        c: corey,
        ph: physics,
        wis,
        z_ref,
        bw_fixed: opts.boundary_fw,
    };
    let (smin, smax) = (corey.swc, 1.0 - corey.sor);
    let cap_diff = {
        let k = 400;
        let h = (smax - smin) / k as f64;
        (0..k)
            .map(|i| {
                let s = smin + h * (i as f64 + 0.5);
                corey.mobility(s).0 * ((ctx.pc(s + 0.5 * h) - ctx.pc(s - 0.5 * h)) / h).abs()
            })
            .fold(0.0, f64::max)
    };
    let pv0 = &model.pore_volume;
    let pv_at = |c: usize, p: f64| pv0[c] * (1.0 + physics.c_rock * (p - physics.p_ref));
    let bw = |p: f64| 1.0 + physics.c_w * (p - physics.p_ref);
    let mut s: Vec<f64> = sw0.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let mut p = p0.to_vec();
    let mut res = TwoPhaseResult {
        times: times.to_vec(),
        sw: vec![],
        p: vec![],
        bhp: vec![],
        bhp_limited: vec![],
        water_rate: vec![],
        oil_rate: vec![],
        water_in_place: vec![],
        water_clamp_loss: vec![],
        oil_in_place: vec![],
        oil_net_in: vec![],
        water_net_in: vec![],
        n_steps: 0,
        n_rejected: 0,
    };
    let wip = |s: &[f64], p: &[f64]| -> f64 {
        (0..n)
            .filter(|&c| model.active[c])
            .map(|c| pv_at(c, p[c]) * s[c] * bw(p[c]))
            .sum()
    };
    let mut net_in = 0.0;
    let mut clamp_loss = 0.0;
    let mut oil_in = 0.0;
    let bo = |p: f64| 1.0 + physics.c_o * (p - physics.p_ref);
    let oip = |s: &[f64], p: &[f64]| -> f64 {
        (0..n)
            .filter(|&c| model.active[c])
            .map(|c| pv_at(c, p[c]) * (1.0 - s[c]) * bo(p[c]))
            .sum()
    };
    let mut t = 0.0;
    let mut dt_next = opts.max_dt;
    let mut last: Option<Step> = None;
    let dfw = corey.max_dfds();
    let dgrav = max_deriv(
        &|s| {
            let (a, b) = corey.mobility(s);
            if a + b > 0.0 {
                a * b / (a + b)
            } else {
                0.0
            }
        },
        smin,
        smax,
    );
    let drho = (physics.rho_w - physics.rho_o).abs() * physics.gravity;
    let dt_cfl = |st: &Step| -> f64 {
        let mut rate = vec![0.0; n];
        for (cn, (&fw, &fo)) in model.connections.iter().zip(st.fw.iter().zip(&st.fo)) {
            // Total-velocity speed, gravity segregation speed, capillary diffusion.
            let dz = (model.center[cn.a][2] - model.center[cn.b][2]).abs();
            let v =
                (fw + fo).abs() * dfw + cn.trans * drho * dz * dgrav + 2.0 * cn.trans * cap_diff;
            rate[cn.a] += v;
            rate[cn.b] += v;
        }
        for (w, (qw, qo)) in wells.iter().zip(st.well_w.iter().zip(&st.well_o)) {
            for ((&c, &a), &b) in w.cells.iter().zip(qw).zip(qo) {
                rate[c] += (a + b).abs() * dfw;
            }
        }
        let mut dt = f64::INFINITY;
        for c in 0..n {
            if rate[c] > 0.0 && pv0[c] > 0.0 {
                dt = dt.min(opts.cfl * pv0[c] * (smax - smin) / rate[c]);
            }
        }
        dt
    };
    // Compressible runs carry water and oil mass per cell; Sw = Mw/(PV bw), and the pressure step
    // corrects the volume misfit PV − Mw/bw − Mo/bo, so both phases are conserved exactly.
    let comp = physics.compressible();
    let mut mw: Vec<f64> = (0..n).map(|c| pv_at(c, p[c]) * s[c] * bw(p[c])).collect();
    let mut mo: Vec<f64> = (0..n)
        .map(|c| pv_at(c, p[c]) * (1.0 - s[c]) * bo(p[c]))
        .collect();
    for &tr in times {
        while t < tr {
            let verr: Option<Vec<f64>> = comp.then(|| {
                (0..n)
                    .map(|c| {
                        if model.active[c] && pv0[c] > 0.0 {
                            pv_at(c, p[c]) - mw[c] / bw(p[c]) - mo[c] / bo(p[c])
                        } else {
                            0.0
                        }
                    })
                    .collect()
            });
            let mut dt = dt_next.min(opts.max_dt).min(tr - t);
            let mut st = ctx.solve(
                &p,
                &s,
                last.as_ref().map_or(&p, |l| &l.p),
                dt,
                verr.as_deref(),
            )?;
            let lim = dt_cfl(&st);
            if dt > lim * 1.0001 {
                dt = lim.min(tr - t);
                st = ctx.solve(&p, &s, &st.p, dt, verr.as_deref())?;
            }
            // Pressure-change control repeats a step that overshoots 2× the target.
            if opts.dp_target.is_finite() {
                loop {
                    let dp = (0..n)
                        .filter(|&c| model.active[c])
                        .map(|c| (st.p[c] - p[c]).abs())
                        .fold(0.0, f64::max);
                    if dp <= 2.0 * opts.dp_target {
                        break;
                    }
                    let shorter = dt * (0.9 * opts.dp_target / dp).max(0.25);
                    if shorter < 1e-9 * tr.max(1.0) {
                        return Err(SxError::Numerical(format!(
                            "two-phase: step below {shorter:.3e} s for dp_target {}",
                            opts.dp_target
                        )));
                    }
                    dt = shorter;
                    res.n_rejected += 1;
                    st = ctx.solve(&p, &s, &st.p, dt, verr.as_deref())?;
                }
            }
            // Water update.
            let mut dmw = vec![0.0; n];
            for (cn, &f) in model.connections.iter().zip(&st.fw) {
                let up = if f >= 0.0 { cn.a } else { cn.b };
                let m = f * bw(st.p[up]);
                dmw[cn.a] -= m;
                dmw[cn.b] += m;
            }
            for (w, qw) in wells.iter().zip(&st.well_w) {
                for (&c, &q) in w.cells.iter().zip(qw) {
                    let m = q * bw(st.p[c]);
                    dmw[c] += m;
                    net_in += dt * m;
                }
            }
            let mut dmo = vec![0.0; n];
            for (cn, &f) in model.connections.iter().zip(&st.fo) {
                let up = if f >= 0.0 { cn.a } else { cn.b };
                let m = f * bo(st.p[up]);
                dmo[cn.a] -= m;
                dmo[cn.b] += m;
            }
            for (w, qo) in wells.iter().zip(&st.well_o) {
                for (&c, &q) in w.cells.iter().zip(qo) {
                    let m = q * bo(st.p[c]);
                    dmo[c] += m;
                    oil_in += dt * m;
                }
            }
            for (f, &q) in fixed.iter().zip(&st.fix_o) {
                let m = q * bo(st.p[f.cell]);
                dmo[f.cell] += m;
                oil_in += dt * m;
            }
            for (f, &q) in fixed.iter().zip(&st.fix_w) {
                let m = q * bw(st.p[f.cell]);
                dmw[f.cell] += m;
                net_in += dt * m;
            }
            let s_old = s.clone();
            for c in 0..n {
                if !model.active[c] || pv0[c] <= 0.0 {
                    continue;
                }
                // Conservative accumulation: water mass PV(p)·Sw·b_w(p) at old and new pressure.
                // Sw may leave [Swc, 1 − Sor] (immobile water or oil); only the physical bound [0, 1] is applied.
                let store = pv_at(c, st.p[c]) * bw(st.p[c]);
                if comp {
                    mw[c] += dt * dmw[c];
                    mo[c] += dt * dmo[c];
                    if mw[c] < 0.0 {
                        clamp_loss += mw[c];
                        mw[c] = 0.0;
                    }
                    mo[c] = mo[c].max(0.0);
                    s[c] = (mw[c] / store).clamp(0.0, 1.0);
                } else {
                    let mass = pv_at(c, p[c]) * s[c] * bw(p[c]) + dt * dmw[c];
                    let raw = mass / store;
                    s[c] = raw.clamp(0.0, 1.0);
                    clamp_loss += (raw - s[c]) * store;
                }
            }
            let (mut ds, mut dp) = (0.0_f64, 0.0_f64);
            for c in (0..n).filter(|&c| model.active[c]) {
                ds = ds.max((s[c] - s_old[c]).abs());
                dp = dp.max((st.p[c] - p[c]).abs());
            }
            p.clone_from(&st.p);
            t = if tr - t <= dt * (1.0 + 1e-12) {
                tr
            } else {
                t + dt
            };
            res.n_steps += 1;
            let scale = |target: f64, change: f64| {
                if target.is_finite() && change > 0.0 {
                    (0.9 * target / change).clamp(0.25, 2.0) * dt
                } else {
                    f64::INFINITY
                }
            };
            dt_next = (1.5 * dt)
                .min(dt_cfl(&st))
                .min(scale(opts.ds_target, ds))
                .min(scale(opts.dp_target, dp));
            last = Some(st);
        }
        let st = match &last {
            Some(l) => l,
            None => {
                last = Some(ctx.solve(&p, &s, &p, opts.max_dt, None)?);
                last.as_ref().unwrap_or_else(|| unreachable!())
            }
        };
        res.bhp.push(st.bhp.clone());
        res.bhp_limited.push(st.limited.clone());
        res.water_rate
            .push(st.well_w.iter().map(|v| v.iter().sum()).collect());
        res.oil_rate
            .push(st.well_o.iter().map(|v| v.iter().sum()).collect());
        res.p.push(p.clone());
        res.sw.push(s.clone());
        res.water_in_place.push(if comp {
            (0..n).filter(|&c| model.active[c]).map(|c| mw[c]).sum()
        } else {
            wip(&s, &p)
        });
        res.water_net_in.push(net_in);
        res.water_clamp_loss.push(clamp_loss);
        res.oil_in_place.push(if comp {
            (0..n).filter(|&c| model.active[c]).map(|c| mo[c]).sum()
        } else {
            oip(&s, &p)
        });
        res.oil_net_in.push(oil_in);
    }
    Ok(res)
}
