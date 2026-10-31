// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Explicit upwind transport: tracer and two-phase IMPES.

use super::model::FlowModel;
use super::pressure::{solve_pressure, FixedPressure, FlowWell, PressureSolution};
use crate::error::{SxError, SxResult};

/// Corey relative permeabilities and fluid viscosities (water / oil).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corey {
    /// Connate water saturation.
    pub swc: f64,
    /// Residual oil saturation.
    pub sor: f64,
    /// Water exponent.
    pub nw: f64,
    /// Oil exponent.
    pub no: f64,
    /// Water end point.
    pub krw_max: f64,
    /// Oil end point.
    pub kro_max: f64,
    /// Water viscosity (Pa·s).
    pub mu_w: f64,
    /// Oil viscosity (Pa·s).
    pub mu_o: f64,
}

impl Default for Corey {
    fn default() -> Self {
        Self {
            swc: 0.2,
            sor: 0.2,
            nw: 2.0,
            no: 2.0,
            krw_max: 1.0,
            kro_max: 1.0,
            mu_w: 1e-3,
            mu_o: 1e-3,
        }
    }
}

impl Corey {
    fn se(&self, s: f64) -> f64 {
        ((s - self.swc) / (1.0 - self.swc - self.sor)).clamp(0.0, 1.0)
    }
    /// Water and oil relative permeability at water saturation `s`.
    pub fn kr(&self, s: f64) -> (f64, f64) {
        let se = self.se(s);
        (
            self.krw_max * se.powf(self.nw),
            self.kro_max * (1.0 - se).powf(self.no),
        )
    }
    /// Water and oil mobility (1/(Pa·s)).
    pub fn mobility(&self, s: f64) -> (f64, f64) {
        let (krw, kro) = self.kr(s);
        (krw / self.mu_w, kro / self.mu_o)
    }
    /// Water fractional flow.
    pub fn fw(&self, s: f64) -> f64 {
        let (lw, lo) = self.mobility(s);
        if lw + lo > 0.0 {
            lw / (lw + lo)
        } else {
            0.0
        }
    }
    /// Largest |dfw/ds| on a fine sample (for the CFL limit).
    pub fn max_dfds(&self) -> f64 {
        let n = 2000;
        let (a, b) = (self.swc, 1.0 - self.sor);
        let h = (b - a) / n as f64;
        (0..n)
            .map(|i| ((self.fw(a + h * (i + 1) as f64) - self.fw(a + h * i as f64)) / h).abs())
            .fold(0.0, f64::max)
    }
    pub(super) fn validate(&self) -> SxResult<()> {
        let ok = self.swc >= 0.0
            && self.sor >= 0.0
            && self.swc + self.sor < 1.0
            && [
                self.nw,
                self.no,
                self.krw_max,
                self.kro_max,
                self.mu_w,
                self.mu_o,
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0);
        if ok {
            Ok(())
        } else {
            Err(SxError::InvalidInput("corey: need swc, sor >= 0, swc + sor < 1, positive exponents, end points and viscosities".into()))
        }
    }
}

/// Per-cell sources from one pressure solution.
struct Sources {
    /// Σ inflow rate × injected value, per cell.
    in_amount: Vec<f64>,
    /// Outflow rate through wells / boundaries, per cell.
    out_rate: Vec<f64>,
    /// Total outflow (connections + sinks), per cell (CFL).
    out_total: Vec<f64>,
}

fn sources(
    model: &FlowModel,
    sol: &PressureSolution,
    wells: &[FlowWell],
    well_val: &[f64],
    fixed: &[FixedPressure],
    fixed_val: &[f64],
) -> Sources {
    let n = model.n_cells();
    let mut in_amount = vec![0.0; n];
    let mut out_rate = vec![0.0; n];
    for ((w, r), &v) in wells.iter().zip(&sol.wells).zip(well_val) {
        for (&c, &q) in w.cells.iter().zip(&r.cell_rates) {
            if q > 0.0 {
                in_amount[c] += q * v
            } else {
                out_rate[c] -= q
            }
        }
    }
    for ((f, &q), &v) in fixed.iter().zip(&sol.boundary_flux).zip(fixed_val) {
        if q > 0.0 {
            in_amount[f.cell] += q * v
        } else {
            out_rate[f.cell] -= q
        }
    }
    let mut out_total = out_rate.clone();
    for (c, &f) in model.connections.iter().zip(&sol.flux) {
        if f > 0.0 {
            out_total[c.a] += f
        } else {
            out_total[c.b] -= f
        }
    }
    Sources {
        in_amount,
        out_rate,
        out_total,
    }
}

/// One explicit upwind step of `pv dx/dt = in − out` with transported
/// fraction `frac(x)`.
fn step(
    model: &FlowModel,
    flux: &[f64],
    src: &Sources,
    x: &mut [f64],
    dt: f64,
    frac: &dyn Fn(f64) -> f64,
) {
    let f: Vec<f64> = x.iter().map(|&v| frac(v)).collect();
    let mut dm: Vec<f64> = (0..x.len())
        .map(|c| src.in_amount[c] - src.out_rate[c] * f[c])
        .collect();
    for (c, &q) in model.connections.iter().zip(flux) {
        let up = if q > 0.0 { f[c.a] } else { f[c.b] };
        dm[c.a] -= q * up;
        dm[c.b] += q * up;
    }
    for (i, v) in x.iter_mut().enumerate() {
        let pv = model.pore_volume[i];
        if pv > 0.0 {
            *v += dt * dm[i] / pv;
        }
    }
}

fn dt_limit(model: &FlowModel, src: &Sources, cfl: f64, speed: f64) -> f64 {
    let mut dt = f64::INFINITY;
    for (c, &o) in src.out_total.iter().enumerate() {
        if o > 0.0 && model.pore_volume[c] > 0.0 {
            dt = dt.min(cfl * model.pore_volume[c] / (o * speed.max(1e-12)));
        }
    }
    dt
}

fn check_times(times: &[f64]) -> SxResult<()> {
    if times.windows(2).any(|w| !(w[1] > w[0])) || times.first().is_some_and(|&t| !(t >= 0.0)) {
        return Err(SxError::InvalidInput(
            "report times must be >= 0 and strictly increasing".into(),
        ));
    }
    Ok(())
}

/// Tracer result.
#[derive(Debug, Clone, PartialEq)]
pub struct TracerResult {
    /// Report times (s).
    pub times: Vec<f64>,
    /// Concentration per report time, per cell.
    pub conc: Vec<Vec<f64>>,
    /// Number of explicit steps taken.
    pub n_steps: usize,
}

/// Tracer transport on a fixed flux field (from [`solve_pressure`]).
/// `well_conc` / `fixed_conc`: concentration of fluid entering through each
/// well / fixed-pressure entry. Explicit upwind, `cfl` ≤ 1.
///
/// # Errors
/// `Shape` / `InvalidInput` for inconsistent lengths, times or `cfl`.
#[allow(clippy::too_many_arguments)]
pub fn run_tracer(
    model: &FlowModel,
    sol: &PressureSolution,
    wells: &[FlowWell],
    well_conc: &[f64],
    fixed: &[FixedPressure],
    fixed_conc: &[f64],
    c0: &[f64],
    times: &[f64],
    cfl: f64,
) -> SxResult<TracerResult> {
    super::pressure::check_controls(wells, fixed)?;
    if c0.len() != model.n_cells()
        || well_conc.len() != wells.len()
        || fixed_conc.len() != fixed.len()
        || sol.flux.len() != model.connections.len()
        || sol.wells.len() != wells.len()
    {
        return Err(SxError::Shape("tracer: inconsistent lengths".into()));
    }
    if !(cfl > 0.0 && cfl <= 1.0) {
        return Err(SxError::InvalidInput(
            "tracer: cfl must be in (0, 1]".into(),
        ));
    }
    check_times(times)?;
    let src = sources(model, sol, wells, well_conc, fixed, fixed_conc);
    let dt_max = dt_limit(model, &src, cfl, 1.0);
    let mut c = c0.to_vec();
    let (mut t, mut n_steps, mut out) = (0.0, 0usize, Vec::with_capacity(times.len()));
    for &tr in times {
        while t < tr {
            let dt = dt_max.min(tr - t);
            step(model, &sol.flux, &src, &mut c, dt, &|v| v);
            t = if tr - t <= dt { tr } else { t + dt };
            n_steps += 1;
        }
        out.push(c.clone());
    }
    Ok(TracerResult {
        times: times.to_vec(),
        conc: out,
        n_steps,
    })
}

/// IMPES options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpesOptions {
    /// Courant number of the saturation steps (≤ 1).
    pub cfl: f64,
    /// Longest time between pressure solves (s).
    pub pressure_dt: f64,
    /// Water fractional flow of fluid entering through fixed-pressure cells.
    pub boundary_fw: f64,
}

impl Default for ImpesOptions {
    fn default() -> Self {
        Self {
            cfl: 0.5,
            pressure_dt: f64::INFINITY,
            boundary_fw: 1.0,
        }
    }
}

/// IMPES result at each report time.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpesResult {
    /// Report times (s).
    pub times: Vec<f64>,
    /// Water saturation per report, per cell.
    pub sw: Vec<Vec<f64>>,
    /// Pressure per report, per cell (Pa).
    pub p: Vec<Vec<f64>>,
    /// Water rate per report, per well (m³/s, + into the reservoir).
    pub water_rate: Vec<Vec<f64>>,
    /// Oil rate per report, per well (m³/s, + into the reservoir).
    pub oil_rate: Vec<Vec<f64>>,
    /// Pressure solves.
    pub n_pressure: usize,
    /// Saturation steps.
    pub n_steps: usize,
}

/// Two-phase water/oil IMPES. Injectors inject water; mobility of the
/// injected fluid is `krw_max / mu_w`.
///
/// # Errors
/// As [`solve_pressure`], plus `InvalidInput` for bad Corey parameters,
/// options or times, `Shape` for a wrong `sw0` length.
pub fn run_impes(
    model: &FlowModel,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    corey: &Corey,
    sw0: &[f64],
    times: &[f64],
    opts: &ImpesOptions,
) -> SxResult<ImpesResult> {
    corey.validate()?;
    super::pressure::check_controls(wells, fixed)?;
    if sw0.len() != model.n_cells() {
        return Err(SxError::Shape(
            "impes: sw0 length differs from the cell count".into(),
        ));
    }
    if !(opts.cfl > 0.0 && opts.cfl <= 1.0)
        || !(opts.pressure_dt > 0.0)
        || !(0.0..=1.0).contains(&opts.boundary_fw)
    {
        return Err(SxError::InvalidInput(
            "impes: cfl in (0, 1], pressure_dt > 0, boundary_fw in [0, 1]".into(),
        ));
    }
    check_times(times)?;
    let inj_mob = corey.krw_max / corey.mu_w;
    let speed = corey.max_dfds();
    let well_val = vec![1.0; wells.len()];
    let fixed_val = vec![opts.boundary_fw; fixed.len()];
    let mut s = sw0.to_vec();
    let mut res = ImpesResult {
        times: times.to_vec(),
        sw: vec![],
        p: vec![],
        water_rate: vec![],
        oil_rate: vec![],
        n_pressure: 0,
        n_steps: 0,
    };
    let mut t = 0.0;
    let solve = |s: &[f64]| -> SxResult<PressureSolution> {
        let mob: Vec<f64> = s
            .iter()
            .map(|&v| {
                let (a, b) = corey.mobility(v);
                a + b
            })
            .collect();
        solve_pressure(model, &mob, inj_mob, wells, fixed)
    };
    for &tr in times {
        let mut sol = solve(&s)?;
        res.n_pressure += 1;
        while t < tr {
            let t_p = (t + opts.pressure_dt).min(tr);
            let src = sources(model, &sol, wells, &well_val, fixed, &fixed_val);
            let dt_max = dt_limit(model, &src, opts.cfl, speed);
            while t < t_p {
                let dt = dt_max.min(t_p - t);
                step(model, &sol.flux, &src, &mut s, dt, &|v| corey.fw(v));
                for v in s.iter_mut() {
                    *v = v.clamp(corey.swc, 1.0 - corey.sor);
                }
                t = if t_p - t <= dt { t_p } else { t + dt };
                res.n_steps += 1;
            }
            if t < tr {
                sol = solve(&s)?;
                res.n_pressure += 1;
            }
        }
        let sol = solve(&s)?;
        let mut wr = Vec::with_capacity(wells.len());
        let mut or = Vec::with_capacity(wells.len());
        for (w, r) in wells.iter().zip(&sol.wells) {
            let (mut qw, mut qo) = (0.0, 0.0);
            for (&c, &q) in w.cells.iter().zip(&r.cell_rates) {
                let f = if q > 0.0 { 1.0 } else { corey.fw(s[c]) };
                qw += q * f;
                qo += q * (1.0 - f);
            }
            wr.push(qw);
            or.push(qo);
        }
        res.water_rate.push(wr);
        res.oil_rate.push(or);
        res.p.push(sol.p);
        res.sw.push(s.clone());
    }
    Ok(res)
}
