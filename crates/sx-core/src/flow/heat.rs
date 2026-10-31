// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Heat transport on a solved single-phase flux field.
//!
//! Solved after the pressure solve (sequential coupling): energy per cell
//! `C_i dT_i/dt = −Σ ρ_f c_f q T_up − Σ G_ij (T_i − T_j) + sources`, with
//!
//! * lumped heat capacity `C_i = V_i (φ ρ_f c_f + (1 − φ) ρ_r c_r)` (J/K),
//! * upwind advection of `ρ_f c_f T` with the face fluxes of the pressure
//!   solution,
//! * conduction with cell conductivity `λ_i = φ λ_f + (1 − φ) λ_r` and the
//!   harmonic mean over each face, `G = 1 / (1/(λ_a g_a) + 1/(λ_b g_b))`,
//! * injection temperature at injecting well cells and at inflowing
//!   fixed-pressure boundaries; outflow carries the cell temperature,
//! * fixed-temperature boundaries through a conductance (W/K).
//!
//! Backward Euler in time, implicit in `T`; the non-symmetric system is
//! solved with a sparse LU factorization (faer). Properties are constant
//! (no temperature dependence), the fluid is single-phase liquid, and there
//! is no steam or phase change. Inactive cells keep no heat (NaN output).

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky, SymmetricOrdering,
};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuRef, NumericLu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, MatMut, Side};
use rayon::prelude::*;

use super::solver::{par_for, par_loops, with_threads, ReportOptions, Snapshot, PAR_MIN_LEN};

use super::stepping::StepControl;
use super::model::FlowModel;
use super::pressure::{FixedPressure, FlowWell, PressureSolution};
use crate::error::{SxError, SxResult};

/// Constant thermal properties (SI).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalProps {
    /// Fluid density (kg/m³).
    pub rho_fluid: f64,
    /// Fluid heat capacity (J/(kg·K)).
    pub cp_fluid: f64,
    /// Rock grain density (kg/m³).
    pub rho_rock: f64,
    /// Rock heat capacity (J/(kg·K)).
    pub cp_rock: f64,
    /// Fluid thermal conductivity (W/(m·K)).
    pub cond_fluid: f64,
    /// Rock thermal conductivity (W/(m·K)).
    pub cond_rock: f64,
}

impl Default for ThermalProps {
    /// Water and a sandstone: 1000 kg/m³, 4180 J/(kg·K), 2650 kg/m³,
    /// 900 J/(kg·K), 0.6 and 2.5 W/(m·K).
    fn default() -> Self {
        Self {
            rho_fluid: 1000.0,
            cp_fluid: 4180.0,
            rho_rock: 2650.0,
            cp_rock: 900.0,
            cond_fluid: 0.6,
            cond_rock: 2.5,
        }
    }
}

impl ThermalProps {
    /// These properties with the fluid density, heat capacity and
    /// conductivity of liquid water at `p_pa` (Pa) and `t_k` (K) from
    /// IAPWS-IF97 and IAPWS 2011 (see [`super::water::liquid`]).
    ///
    /// # Errors
    /// `InvalidInput` outside the liquid range of IF97 region 1.
    pub fn with_water(self, p_pa: f64, t_k: f64) -> SxResult<Self> {
        let w = super::water::liquid(p_pa, t_k)?;
        Ok(Self {
            rho_fluid: w.density,
            cp_fluid: w.cp,
            cond_fluid: w.conductivity,
            ..self
        })
    }

    fn validate(&self) -> SxResult<()> {
        let v = [
            self.rho_fluid,
            self.cp_fluid,
            self.rho_rock,
            self.cp_rock,
            self.cond_fluid,
            self.cond_rock,
        ];
        if v.iter().any(|x| !x.is_finite() || *x < 0.0) || self.rho_fluid * self.cp_fluid <= 0.0 {
            return Err(SxError::InvalidInput(
                "heat: properties must be finite and >= 0, fluid rho*cp > 0".into(),
            ));
        }
        Ok(())
    }
}

/// A cell tied to a fixed temperature through a conductance (W/K), e.g.
/// `λ · FlowModel::boundary_geom(cell, axis)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedTemperature {
    /// Cell.
    pub cell: usize,
    /// Conductance (W/K).
    pub conductance: f64,
    /// Boundary temperature (any consistent unit, e.g. °C).
    pub temperature: f64,
}

/// A prescribed heat flow into a cell (a Neumann boundary), e.g. a basal
/// geothermal flux times the face area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatFlux {
    /// Cell.
    pub cell: usize,
    /// Heat flow into the cell (W); negative takes heat out.
    pub power: f64,
}

/// Heat run inputs besides the model and the pressure solution.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatInputs {
    /// Injection temperature per well (used only where the well injects).
    pub well_temp: Vec<f64>,
    /// Inflow temperature per fixed-pressure boundary (used only on inflow).
    pub fixed_temp: Vec<f64>,
    /// Fixed-temperature (conductive) boundaries.
    pub fixed_t: Vec<FixedTemperature>,
    /// Prescribed heat-flow (Neumann) boundaries.
    pub heat_flux: Vec<HeatFlux>,
    /// Initial temperature per cell.
    pub t0: Vec<f64>,
    /// Output times (s), increasing, > 0.
    pub times: Vec<f64>,
    /// Largest time step (s).
    pub dt_max: f64,
    /// Adaptive stepping on the largest temperature change per step; `None`
    /// keeps the fixed step `dt_max`.
    pub control: Option<StepControl>,
}

/// Heat run output.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatResult {
    /// Output times (s).
    pub times: Vec<f64>,
    /// Temperature per output time and cell.
    pub temperature: Vec<Vec<f64>>,
    /// Stored energy change since t = 0 per output time (J, relative to T = 0).
    pub energy_change: Vec<f64>,
    /// Net energy in through wells and boundaries per output time (J, cumulative).
    pub energy_in: Vec<f64>,
    /// Time steps taken (accepted).
    pub n_steps: usize,
    /// Steps repeated by the adaptive controller.
    pub n_rejected: usize,
}

fn numerical(msg: &str) -> SxError {
    SxError::Numerical(format!("heat solve: {msg}"))
}

/// Run heat transport on a fixed flux field `sol` (from
/// [`super::solve_pressure`] with the same `wells` and `fixed`).
///
/// # Errors
/// `InvalidInput` / `Shape` for bad properties, lengths, cells or times;
/// `Numerical` when the LU factorization fails.
pub fn run_heat(
    model: &FlowModel,
    sol: &PressureSolution,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    props: &ThermalProps,
    inp: &HeatInputs,
) -> SxResult<HeatResult> {
    run_heat_with(model, sol, wells, fixed, props, inp, &mut ReportOptions::default())
}

/// [`run_heat`] with report-snapshot control: keep every N-th temperature
/// field, stream them to a sink, or both (see [`ReportOptions`]). The energy
/// series keep every report time.
///
/// # Errors
/// As [`run_heat`], plus errors returned by the sink.
pub fn run_heat_with(
    model: &FlowModel,
    sol: &PressureSolution,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    props: &ThermalProps,
    inp: &HeatInputs,
    report: &mut ReportOptions<'_>,
) -> SxResult<HeatResult> {
    with_threads(None, move || heat_impl(model, sol, wells, fixed, props, inp, report))
}

/// Factorization of the heat matrix: Cholesky when it is symmetric (no
/// advection), LU otherwise. The symbolic part is computed once per run.
enum HeatFactor {
    Llt { symbolic: SymbolicCholesky<usize>, l: Vec<f64> },
    Lu { symbolic: SymbolicLu<usize>, numeric: NumericLu<usize, f64> },
}

fn heat_impl(
    model: &FlowModel,
    sol: &PressureSolution,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    props: &ThermalProps,
    inp: &HeatInputs,
    report: &mut ReportOptions<'_>,
) -> SxResult<HeatResult> {
    props.validate()?;
    let n = model.n_cells();
    let bad = |m: String| Err(SxError::Shape(format!("heat: {m}")));
    if inp.t0.len() != n {
        return bad(format!("t0 has {} values for {n} cells", inp.t0.len()));
    }
    if sol.flux.len() != model.connections.len()
        || sol.wells.len() != wells.len()
        || sol.boundary_flux.len() != fixed.len()
    {
        return bad("pressure solution does not match the model, wells or boundaries".into());
    }
    if inp.well_temp.len() != wells.len() || inp.fixed_temp.len() != fixed.len() {
        return bad("need one temperature per well and per fixed-pressure boundary".into());
    }
    let finite = inp
        .t0
        .iter()
        .chain(&inp.well_temp)
        .chain(&inp.fixed_temp)
        .all(|v| v.is_finite());
    if !finite || !(inp.dt_max > 0.0 && inp.dt_max.is_finite()) {
        return Err(SxError::InvalidInput(
            "heat: temperatures must be finite and dt_max > 0".into(),
        ));
    }
    if let Some(c) = &inp.control {
        c.validate()?;
    }
    let mut prev = 0.0;
    for &t in &inp.times {
        if !(t.is_finite() && t > prev) {
            return Err(SxError::InvalidInput("heat: times must be increasing and > 0".into()));
        }
        prev = t;
    }
    for f in &inp.fixed_t {
        if f.cell >= n || !model.active[f.cell] || !(f.conductance >= 0.0) || !f.temperature.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "heat: fixed temperature at cell {} needs an active cell, conductance >= 0 and a finite temperature",
                f.cell
            )));
        }
    }
    for f in &inp.heat_flux {
        if f.cell >= n || !model.active[f.cell] || !f.power.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "heat: heat flux at cell {} needs an active cell and a finite power",
                f.cell
            )));
        }
    }
    let rc = props.rho_fluid * props.cp_fluid;
    let phi = |c: usize| {
        if model.bulk_volume[c] > 0.0 {
            (model.pore_volume[c] / model.bulk_volume[c]).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let cap: Vec<f64> = (0..n)
        .map(|c| {
            let p = phi(c);
            model.bulk_volume[c] * (p * rc + (1.0 - p) * props.rho_rock * props.cp_rock)
        })
        .collect();
    let lam: Vec<f64> = (0..n)
        .map(|c| {
            let p = phi(c);
            p * props.cond_fluid + (1.0 - p) * props.cond_rock
        })
        .collect();
    // Time-independent operator (W/K) and source (W).
    let mut trip: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut diag = vec![0.0; n];
    let mut src = vec![0.0; n];
    for (c, &q) in model.connections.iter().zip(&sol.flux) {
        let (up, dn) = if q >= 0.0 { (c.a, c.b) } else { (c.b, c.a) };
        let adv = rc * q.abs();
        diag[up] += adv;
        trip.push(Triplet::new(dn, up, -adv));
        let ga = lam[c.a] * c.geom[0];
        let gb = lam[c.b] * c.geom[1];
        if ga > 0.0 && gb > 0.0 {
            let g = 1.0 / (1.0 / ga + 1.0 / gb);
            diag[c.a] += g;
            diag[c.b] += g;
            trip.push(Triplet::new(c.a, c.b, -g));
            trip.push(Triplet::new(c.b, c.a, -g));
        }
    }
    let mut add_rate = |cell: usize, q: f64, t_in: f64, diag: &mut [f64]| {
        if q > 0.0 {
            src[cell] += rc * q * t_in;
        } else {
            diag[cell] += rc * (-q);
        }
    };
    for ((w, r), &tw) in wells.iter().zip(&sol.wells).zip(&inp.well_temp) {
        for (&c, &q) in w.cells.iter().zip(&r.cell_rates) {
            add_rate(c, q, tw, &mut diag);
        }
    }
    for ((f, &q), &tf) in fixed.iter().zip(&sol.boundary_flux).zip(&inp.fixed_temp) {
        add_rate(f.cell, q, tf, &mut diag);
    }
    for f in &inp.fixed_t {
        diag[f.cell] += f.conductance;
        src[f.cell] += f.conductance * f.temperature;
    }
    for f in &inp.heat_flux {
        src[f.cell] += f.power;
    }
    // Net boundary energy rate (W) for temperature field `t`.
    let inflow = |t: &[f64]| -> f64 {
        let mut e = 0.0;
        for ((w, r), &tw) in wells.iter().zip(&sol.wells).zip(&inp.well_temp) {
            for (&c, &q) in w.cells.iter().zip(&r.cell_rates) {
                e += rc * q * if q > 0.0 { tw } else { t[c] };
            }
        }
        for ((f, &q), &tf) in fixed.iter().zip(&sol.boundary_flux).zip(&inp.fixed_temp) {
            e += rc * q * if q > 0.0 { tf } else { t[f.cell] };
        }
        for f in &inp.fixed_t {
            e += f.conductance * (f.temperature - t[f.cell]);
        }
        e += inp.heat_flux.iter().map(|f| f.power).sum::<f64>();
        e
    };
    let energy = |t: &[f64]| -> f64 { (0..n).filter(|&c| model.active[c]).map(|c| cap[c] * t[c]).sum() };
    let mut t: Vec<f64> = inp.t0.clone();
    let e0 = energy(&t);
    let mut cum_in = 0.0;
    let mut res = HeatResult {
        times: inp.times.clone(),
        temperature: vec![],
        energy_change: vec![],
        energy_in: vec![],
        n_steps: 0,
        n_rejected: 0,
    };
    let mut now = 0.0;
    // Fixed pattern: the operator plus one diagonal entry per cell, built once;
    // a new dt only rewrites the diagonal values and refactors numerically.
    let symmetric = sol.flux.iter().all(|&q| q == 0.0);
    let mut tp = trip;
    tp.extend((0..n).map(|c| Triplet::new(c, c, 0.0)));
    let mut a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &tp)
        .map_err(|e| numerical(&format!("cannot build matrix: {e:?}")))?;
    drop(tp);
    let diag_slot: Vec<usize> = (0..n)
        .map(|c| {
            let r = a.symbolic().row_idx_of_col_raw(c);
            let k = r.binary_search(&c).expect("diagonal entry present");
            a.symbolic().col_ptr()[c] + k
        })
        .collect();
    let diag_base: Vec<f64> = diag_slot.iter().map(|&k| a.val()[k]).collect();
    let par = par_for(n);
    let (mut factor, mut mem) = if symmetric {
        let symbolic = factorize_symbolic_cholesky(a.symbolic(), Side::Lower, SymmetricOrdering::Amd, CholeskySymbolicParams::default())
            .map_err(|e| numerical(&format!("symbolic Cholesky failed: {e:?}")))?;
        let mem = MemBuffer::try_new(StackReq::any_of(&[
            symbolic.factorize_numeric_llt_scratch::<f64>(par, Default::default()),
            symbolic.solve_in_place_scratch::<f64>(1, par),
        ]))
        .map_err(|e| numerical(&format!("cannot allocate workspace: {e:?}")))?;
        let l = vec![0.0; symbolic.len_val()];
        (HeatFactor::Llt { symbolic, l }, mem)
    } else {
        let symbolic = factorize_symbolic_lu(a.symbolic(), Default::default())
            .map_err(|e| numerical(&format!("symbolic LU failed: {e:?}")))?;
        let mem = MemBuffer::try_new(StackReq::any_of(&[
            symbolic.factorize_numeric_lu_scratch::<f64>(par, Default::default()),
            symbolic.solve_in_place_scratch::<f64>(1, par),
        ]))
        .map_err(|e| numerical(&format!("cannot allocate workspace: {e:?}")))?;
        (HeatFactor::Lu { symbolic, numeric: NumericLu::new() }, mem)
    };
    let mut cached: Option<f64> = None;
    let mut next = vec![0.0; n];
    let pl = par_loops(n);
    let n_rep = inp.times.len();
    let mut i_rep = 0usize;
    let mut dt_next = inp.control.map_or(inp.dt_max, |c| c.dt_init.min(inp.dt_max));
    for &tr in &inp.times {
        while now < tr {
            let dt = dt_next.min(inp.dt_max).min(tr - now);
            let reuse = cached.is_some_and(|d| (d - dt).abs() <= 1e-12 * dt);
            if !reuse {
                {
                    let v = a.val_mut();
                    for c in 0..n {
                        let d = if model.active[c] { cap[c] / dt + diag[c] } else { 1.0 };
                        v[diag_slot[c]] = diag_base[c] + d;
                    }
                }
                if a.val().iter().any(|x| !x.is_finite()) {
                    return Err(numerical("matrix has non-finite entries"));
                }
                let stack = MemStack::new(&mut mem);
                match &mut factor {
                    HeatFactor::Llt { symbolic, l } => {
                        symbolic
                            .factorize_numeric_llt::<f64>(l, a.as_ref(), Side::Lower, LltRegularization::default(), par, stack, Default::default())
                            .map_err(|e| numerical(&format!("Cholesky failed: {e:?}")))?;
                    }
                    HeatFactor::Lu { symbolic, numeric } => {
                        symbolic
                            .factorize_numeric_lu::<f64>(numeric, a.as_ref(), par, stack, Default::default())
                            .map_err(|e| numerical(&format!("LU failed: {e:?}")))?;
                    }
                }
                cached = Some(dt);
            }
            let fill = |(c, x): (usize, &mut f64)| {
                *x = if model.active[c] { cap[c] / dt * t[c] + src[c] } else { t[c] };
            };
            if pl {
                next.par_iter_mut().with_min_len(PAR_MIN_LEN).enumerate().for_each(fill);
            } else {
                next.iter_mut().enumerate().for_each(fill);
            }
            {
                let stack = MemStack::new(&mut mem);
                let rhs = MatMut::from_column_major_slice_mut(&mut next, n, 1);
                match &factor {
                    HeatFactor::Llt { symbolic, l } => LltRef::new(symbolic, l).solve_in_place_with_conj(Conj::No, rhs, par, stack),
                    HeatFactor::Lu { symbolic, numeric } => LuRef::new_unchecked(symbolic, numeric).solve_in_place_with_conj(Conj::No, rhs, par, stack),
                }
            }
            if next.iter().any(|v| !v.is_finite()) {
                return Err(numerical("solution is not finite"));
            }
            if let Some(c) = &inp.control {
                let change = (0..n)
                    .filter(|&i| model.active[i])
                    .map(|i| (next[i] - t[i]).abs())
                    .fold(0.0, f64::max);
                let (ok, d) = c.judge(dt, change)?;
                // A step shortened to hit a report time does not limit growth.
                dt_next = if dt < dt_next && ok { dt_next.max(d) } else { d };
                if !ok {
                    res.n_rejected += 1;
                    continue;
                }
            }
            cum_in += inflow(&next) * dt;
            std::mem::swap(&mut t, &mut next);
            now = if tr - now <= dt { tr } else { now + dt };
            res.n_steps += 1;
        }
        res.energy_change.push(energy(&t) - e0);
        res.energy_in.push(cum_in);
        if report.keeps(i_rep, n_rep) {
            let field: Vec<f64> = t
                .iter()
                .enumerate()
                .map(|(c, &v)| if model.active[c] { v } else { f64::NAN })
                .collect();
            report.emit(&Snapshot { index: i_rep, time: tr, fields: &[("temperature", &field)] })?;
            if report.in_memory {
                res.temperature.push(field);
            }
        }
        i_rep += 1;
    }
    Ok(res)
}
