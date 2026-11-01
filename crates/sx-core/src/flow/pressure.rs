// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Incompressible TPFA pressure solve with Peaceman wells.

use std::cell::RefCell;

use rayon::prelude::*;

use super::model::FlowModel;
use super::solver::{par_loops, with_threads, SpdCache, SpdEntries, PAR_MIN_LEN};
use crate::error::{SxError, SxResult};

/// Injector or producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WellKind {
    /// Injects (uses the injected-fluid mobility).
    Injector,
    /// Produces (uses the cell mobility).
    Producer,
}

/// Well control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WellControl {
    /// Total reservoir-volume rate magnitude (m³/s); the sign follows the kind.
    Rate(f64),
    /// Bottom-hole pressure (Pa).
    Bhp(f64),
}

/// A well completed in one or more cells. Vertical by default (Peaceman with
/// the cell dx, dy, dz); a deviated well carries its own well indices from
/// [`FlowModel::trajectory_completions`](super::FlowModel::trajectory_completions).
#[derive(Debug, Clone, PartialEq)]
pub struct FlowWell {
    /// Name.
    pub name: String,
    /// Kind.
    pub kind: WellKind,
    /// Control.
    pub control: WellControl,
    /// Completed cells.
    pub cells: Vec<usize>,
    /// Wellbore radius (m).
    pub radius_m: f64,
    /// Skin (dimensionless).
    pub skin: f64,
    /// Well index per completed cell (m³); `None` = vertical Peaceman.
    pub wi: Option<Vec<f64>>,
    /// Depth (m) at which `Bhp` / the reported BHP applies; `None` = depth of
    /// the first completed cell. Used only when gravity is on.
    pub ref_depth_m: Option<f64>,
    /// BHP limit (Pa) of a rate-controlled well: the largest BHP of an
    /// injector, the smallest of a producer. When the rate would need a BHP
    /// beyond it the well switches to BHP control at the limit, and back to
    /// rate control when that BHP would exceed the target rate.
    pub bhp_limit: Option<f64>,
    /// Allow crossflow: a completion may flow against the well direction
    /// (into the reservoir at a producer, out of it at an injector), with the
    /// upstream mobility (the wellbore mixture or the cell). When `false`
    /// such completions are shut.
    pub crossflow: bool,
}

impl FlowWell {
    /// Vertical well with no WI override and the default reference depth.
    pub fn vertical(
        name: &str,
        kind: WellKind,
        control: WellControl,
        cells: Vec<usize>,
        radius_m: f64,
        skin: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            kind,
            control,
            cells,
            radius_m,
            skin,
            wi: None,
            ref_depth_m: None,
            bhp_limit: None,
            crossflow: true,
        }
    }
}

/// Check well controls and fixed-pressure boundaries before a run.
///
/// Sign convention: a rate is a magnitude (m³/s of reservoir volume) and must
/// be finite and >= 0; the well kind sets the direction (an injector adds
/// fluid, a producer removes it). Pressures (BHP and boundary) must be finite,
/// in Pa; an incompressible run only uses pressure differences, so a relative
/// datum is fine there. Boundary transmissibility must be finite and >= 0.
///
/// # Errors
/// `InvalidInput` naming the well or boundary and the bad value.
pub fn check_controls(wells: &[FlowWell], fixed: &[FixedPressure]) -> SxResult<()> {
    for w in wells {
        match w.control {
            WellControl::Rate(q) if !q.is_finite() || q < 0.0 => {
                return Err(SxError::InvalidInput(format!(
                    "well {}: rate {q} m³/s; rates are magnitudes >= 0 and the kind ({:?}) sets the direction",
                    w.name, w.kind
                )))
            }
            WellControl::Bhp(p) if !p.is_finite() => {
                return Err(SxError::InvalidInput(format!(
                    "well {}: bhp {p} Pa is not finite",
                    w.name
                )))
            }
            _ => {}
        }
        if let Some(l) = w.bhp_limit {
            if !l.is_finite() {
                return Err(SxError::InvalidInput(format!("well {}: bhp limit {l} Pa is not finite", w.name)));
            }
        }
        if !w.skin.is_finite() || w.skin <= -10.0 {
            return Err(SxError::InvalidInput(format!(
                "well {}: skin {} must be finite and > -10",
                w.name, w.skin
            )));
        }
    }
    for (i, f) in fixed.iter().enumerate() {
        if !f.trans.is_finite() || f.trans < 0.0 || !f.pressure.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "fixed boundary #{i} (cell {}): trans {} m³ must be finite and >= 0 and pressure {} Pa finite",
                f.cell, f.trans, f.pressure
            )));
        }
    }
    Ok(())
}

/// A cell held at a fixed pressure through a boundary transmissibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPressure {
    /// Cell.
    pub cell: usize,
    /// Geometric transmissibility to the boundary (m³), e.g.
    /// [`FlowModel::boundary_trans`].
    pub trans: f64,
    /// Boundary pressure (Pa).
    pub pressure: f64,
}

/// Solved well.
#[derive(Debug, Clone, PartialEq)]
pub struct WellResult {
    /// Bottom-hole pressure (Pa).
    pub bhp: f64,
    /// Total rate into the reservoir (m³/s; negative = production).
    pub rate: f64,
    /// Rate per completed cell, same sign rule.
    pub cell_rates: Vec<f64>,
    /// Peaceman well index per completed cell (m³).
    pub wi: Vec<f64>,
    /// The rate well ran at its BHP limit.
    pub bhp_limited: bool,
}

/// Pressure solution.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureSolution {
    /// Cell pressure (Pa); NaN for inactive cells.
    pub p: Vec<f64>,
    /// Volumetric flux per connection, from `a` to `b` (m³/s).
    pub flux: Vec<f64>,
    /// Flux into the reservoir per fixed-pressure entry (m³/s).
    pub boundary_flux: Vec<f64>,
    /// Wells, input order.
    pub wells: Vec<WellResult>,
    /// Picard iterations used to settle the upwind directions.
    pub upwind_iterations: usize,
    /// Warnings (e.g. upwind directions did not settle; arithmetic-mean fallback used).
    pub warnings: Vec<String>,
}

/// Cap on Picard iterations for upwind face mobility in [`solve_pressure`].
pub const UPWIND_MAX_ITER: usize = 30;

/// Face mobility from the sign of the pressure drop: the upstream cell's
/// mobility; the arithmetic mean on a (near) zero drop, so that stagnant
/// faces do not flip back and forth between iterations.
pub(super) fn upwind_face_mobility(model: &FlowModel, mobility: &[f64], p: Option<&[f64]>) -> Vec<f64> {
    let mean = |a: usize, b: usize| 0.5 * (mobility[a] + mobility[b]);
    let conns = &model.connections;
    let par = par_loops(conns.len());
    let Some(p) = p else {
        return if par {
            conns.par_iter().with_min_len(PAR_MIN_LEN).map(|c| mean(c.a, c.b)).collect()
        } else {
            conns.iter().map(|c| mean(c.a, c.b)).collect()
        };
    };
    let dpabs = |c: &super::model::Connection| (p[c.a] - p[c.b]).abs();
    // Max is exact in any order.
    let scale = if par {
        conns.par_iter().with_min_len(PAR_MIN_LEN).map(dpabs).reduce(|| 0.0, f64::max)
    } else {
        conns.iter().map(dpabs).fold(0.0, f64::max)
    };
    let f = |c: &super::model::Connection| {
        let dp = p[c.a] - p[c.b];
        if dp.abs() <= 1e-9 * scale {
            mean(c.a, c.b)
        } else if dp > 0.0 {
            mobility[c.a]
        } else {
            mobility[c.b]
        }
    };
    if par {
        conns.par_iter().with_min_len(PAR_MIN_LEN).map(f).collect()
    } else {
        conns.iter().map(f).collect()
    }
}

/// Peaceman well index (m³) of a well segment with projected lengths
/// `l = [lx, ly, lz]` (m) in a cell of size `ext` with permeability `k`:
/// each axis uses the Peaceman formula with the two permeabilities and cell
/// sizes normal to it; the three are combined as `√(WIx² + WIy² + WIz²)`
/// (the projection rule used by ECLIPSE-style simulators and MRST).
pub fn peaceman_wi_segment(k: [f64; 3], ext: [f64; 3], l: [f64; 3], rw: f64, skin: f64) -> f64 {
    let wx = if l[0] > 0.0 {
        peaceman_wi(k[1], k[2], ext[1], ext[2], l[0], rw, skin)
    } else {
        0.0
    };
    let wy = if l[1] > 0.0 {
        peaceman_wi(k[0], k[2], ext[0], ext[2], l[1], rw, skin)
    } else {
        0.0
    };
    let wz = if l[2] > 0.0 {
        peaceman_wi(k[0], k[1], ext[0], ext[1], l[2], rw, skin)
    } else {
        0.0
    };
    (wx * wx + wy * wy + wz * wz).sqrt()
}

/// Peaceman well index (m³) of a vertical well in a cell of size dx, dy, dz
/// with permeabilities kx, ky: `2π √(kx ky) dz / (ln(re/rw) + s)` with
/// `re = 0.28 √(√(ky/kx) dx² + √(kx/ky) dy²) / ((ky/kx)^¼ + (kx/ky)^¼)`.
pub fn peaceman_wi(kx: f64, ky: f64, dx: f64, dy: f64, dz: f64, rw: f64, skin: f64) -> f64 {
    let r = ky / kx;
    let re = 0.28 * (r.sqrt() * dx * dx + (1.0 / r).sqrt() * dy * dy).sqrt()
        / (r.powf(0.25) + (1.0 / r).powf(0.25));
    2.0 * std::f64::consts::PI * (kx * ky).sqrt() * dz / ((re / rw).ln() + skin)
}

fn numerical(msg: &str) -> SxError {
    SxError::Numerical(format!("pressure solve: {msg}"))
}

/// Solve the incompressible pressure equation
/// `Σ T λ (p_i − p_j) + Σ WI λ_w (p_i − p_w) = 0` with cell mobility `mobility`
/// (1/(Pa·s)); injectors use `inj_mobility`. Rate wells add their BHP as an
/// unknown. Connection mobility is taken from the upstream cell (phase-
/// potential upwinding, as in [`super::run_two_phase`]): the directions come
/// from Picard iteration starting from the arithmetic-mean solution and stop
/// when no face changes direction. If they have not settled after
/// [`UPWIND_MAX_ITER`] iterations the arithmetic-mean solution is returned
/// with a warning. Fluxes always use the same face mobility as the matrix,
/// so the result is mass conservative either way. Fixed-pressure entries use
/// the cell mobility.
///
/// # Errors
/// `Shape` / `InvalidInput` for bad input (unknown or inactive cell, no BHP
/// well and no fixed pressure, non-positive radius); `Numerical` when the
/// factorization fails.
pub fn solve_pressure(
    model: &FlowModel,
    mobility: &[f64],
    inj_mobility: f64,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
) -> SxResult<PressureSolution> {
    solve_pressure_with(model, mobility, inj_mobility, wells, fixed, &PressureOptions::default())
}

/// Options for [`solve_pressure_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureOptions {
    /// Cap on Picard iterations for the upwind directions (default
    /// [`UPWIND_MAX_ITER`]); 0 forces the arithmetic-mean fallback.
    pub max_upwind_iter: usize,
    /// Mobility (1/(Pa·s)) of the fluid entering through fixed-pressure
    /// boundaries; `None` uses the cell mobility. Outflow always uses the
    /// cell mobility.
    pub boundary_inflow_mobility: Option<f64>,
}

impl Default for PressureOptions {
    fn default() -> Self {
        Self {
            max_upwind_iter: UPWIND_MAX_ITER,
            boundary_inflow_mobility: None,
        }
    }
}

/// [`solve_pressure`] with options.
///
/// # Errors
/// As [`solve_pressure`]; `InvalidInput` for a non-finite or negative
/// boundary inflow mobility.
pub fn solve_pressure_with(
    model: &FlowModel,
    mobility: &[f64],
    inj_mobility: f64,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    opts: &PressureOptions,
) -> SxResult<PressureSolution> {
    check_controls(wells, fixed)?;
    if opts.boundary_inflow_mobility.is_some_and(|m| !(m.is_finite() && m >= 0.0)) {
        return Err(SxError::InvalidInput(
            "pressure solve: boundary inflow mobility must be finite and >= 0".into(),
        ));
    }
    with_threads(None, || solve_pressure_impl(model, mobility, inj_mobility, wells, fixed, opts))
}

fn solve_pressure_impl(
    model: &FlowModel,
    mobility: &[f64],
    inj_mobility: f64,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    opts: &PressureOptions,
) -> SxResult<PressureSolution> {
    let mut ctrl: Vec<WellControl> = wells.iter().map(|w| w.control).collect();
    let max_switch = 2 * wells.len() + 2;
    // One factorization cache for every upwind and control-switch solve.
    let cache = RefCell::new(SpdCache::default());
    let entries = RefCell::new(SpdEntries::default());
    for round in 0.. {
        let mut sol = solve_controls(model, mobility, inj_mobility, wells, &ctrl, fixed, opts, &cache, &entries)?;
        let changed = switch_controls(wells, &mut ctrl, &sol.wells.iter().map(|r| (r.bhp, r.rate)).collect::<Vec<_>>());
        if !changed || round == max_switch {
            if changed {
                sol.warnings.push(format!(
                    "pressure solve: well controls did not settle in {max_switch} switches"
                ));
            }
            for (r, (w, c)) in sol.wells.iter_mut().zip(wells.iter().zip(&ctrl)) {
                r.bhp_limited = matches!(w.control, WellControl::Rate(_)) && matches!(c, WellControl::Bhp(_));
            }
            return Ok(sol);
        }
    }
    unreachable!("the loop returns")
}

/// Rate ⇄ BHP-limit switching from solved (bhp, rate) per well; true when a
/// control changed.
pub(super) fn switch_controls(wells: &[FlowWell], ctrl: &mut [WellControl], res: &[(f64, f64)]) -> bool {
    let mut changed = false;
    for ((w, c), &(bhp, rate)) in wells.iter().zip(ctrl.iter_mut()).zip(res) {
        let (WellControl::Rate(q), Some(lim)) = (w.control, w.bhp_limit) else {
            continue;
        };
        let inj = w.kind == WellKind::Injector;
        match *c {
            WellControl::Rate(_) => {
                if (inj && bhp > lim) || (!inj && bhp < lim) {
                    *c = WellControl::Bhp(lim);
                    changed = true;
                }
            }
            WellControl::Bhp(_) => {
                let along = if inj { rate } else { -rate };
                if along > q * (1.0 + 1e-9) {
                    *c = WellControl::Rate(q);
                    changed = true;
                }
            }
        }
    }
    changed
}

/// True when a completion flows along the well direction (into the
/// reservoir at an injector, out of it at a producer).
pub(super) fn completion_direction(w: &FlowWell, bhp: f64, p: f64) -> bool {
    match w.kind {
        WellKind::Injector => bhp >= p,
        WellKind::Producer => bhp <= p,
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_controls(
    model: &FlowModel,
    mobility: &[f64],
    inj_mobility: f64,
    wells: &[FlowWell],
    ctrl: &[WellControl],
    fixed: &[FixedPressure],
    opts: &PressureOptions,
    cache: &RefCell<SpdCache>,
    entries: &RefCell<SpdEntries>,
) -> SxResult<PressureSolution> {
    let max_iter = opts.max_upwind_iter;
    let nconn = model.connections.len();
    let n = model.n_cells();
    if mobility.len() != n {
        return Err(SxError::Shape(format!(
            "mobility has {} values for {n} cells",
            mobility.len()
        )));
    }
    if !ctrl.iter().any(|c| matches!(c, WellControl::Bhp(_))) && fixed.is_empty() {
        return Err(SxError::InvalidInput(
            "pressure solve needs a BHP-controlled well or a fixed-pressure cell".into(),
        ));
    }
    let check_cell = |c: usize, what: &str| -> SxResult<()> {
        if c >= n || !model.active[c] {
            return Err(SxError::InvalidInput(format!(
                "{what}: cell {c} is outside the grid or inactive"
            )));
        }
        Ok(())
    };
    // Well indices.
    let mut wis: Vec<Vec<f64>> = Vec::with_capacity(wells.len());
    for w in wells {
        if !(w.radius_m > 0.0) || w.cells.is_empty() {
            return Err(SxError::InvalidInput(format!(
                "well {}: needs cells and a radius > 0",
                w.name
            )));
        }
        wis.push(well_indices(model, w)?);
    }
    let mut n_unk = n;
    let mut bhp_unknown: Vec<Option<usize>> = Vec::with_capacity(wells.len());
    for c in ctrl {
        if matches!(c, WellControl::Rate(_)) {
            bhp_unknown.push(Some(n_unk));
            n_unk += 1;
        } else {
            bhp_unknown.push(None);
        }
    }
    let along_mob = |w: &FlowWell, c: usize| match w.kind {
        WellKind::Injector => inj_mobility,
        WellKind::Producer => mobility[c],
    };
    // Face mobilities: connections, fixed-pressure entries, then completions.
    let all_mob = |x: Option<&[f64]>| -> Vec<f64> {
        let p = x.map(|x| &x[..n]);
        let mut v = upwind_face_mobility(model, mobility, p);
        for f in fixed {
            let inflow = p.is_some_and(|p| f.pressure > p[f.cell]);
            v.push(match (inflow, opts.boundary_inflow_mobility) {
                (true, Some(m)) => m,
                _ => mobility[f.cell],
            });
        }
        for ((w, wi), (c, u)) in wells.iter().zip(&wis).zip(ctrl.iter().zip(&bhp_unknown)) {
            let bhp = match (c, u, x) {
                (WellControl::Bhp(b), _, _) => Some(*b),
                (_, Some(u), Some(x)) => Some(x[*u]),
                _ => None,
            };
            let Some((bhp, p)) = bhp.zip(p) else {
                v.extend(w.cells.iter().map(|&c| along_mob(w, c)));
                continue;
            };
            // Wellbore mixture of a producer: mobility of the inflowing completions, rate weighted.
            let (mut num, mut den) = (0.0, 0.0);
            for (&c, &wic) in w.cells.iter().zip(wi) {
                if completion_direction(w, bhp, p[c]) {
                    let q = wic * mobility[c] * (p[c] - bhp).abs();
                    num += q * mobility[c];
                    den += q;
                }
            }
            let mix = if den > 0.0 {
                num / den
            } else {
                w.cells.iter().map(|&c| mobility[c]).sum::<f64>() / w.cells.len() as f64
            };
            for &c in &w.cells {
                v.push(if completion_direction(w, bhp, p[c]) {
                    along_mob(w, c)
                } else if !w.crossflow {
                    0.0
                } else {
                    match w.kind {
                        WellKind::Injector => mobility[c],
                        WellKind::Producer => mix,
                    }
                });
            }
        }
        v
    };
    let ncomp_off = nconn + fixed.len();
    let assemble_solve = |face_mob: &[f64]| -> SxResult<Vec<f64>> {
        let mut lower = entries.borrow_mut();
        lower.clear();
        let mut rhs = vec![0.0; n_unk];
        let add = |r: usize, c: usize, v: f64, m: &mut SpdEntries| m.add(r, c, v);
        for c in 0..n {
            if !model.active[c] {
                add(c, c, 1.0, &mut lower);
            }
        }
        for (conn, fm) in model.connections.iter().zip(face_mob) {
            let t = conn.trans * fm;
            add(conn.a, conn.a, t, &mut lower);
            add(conn.b, conn.b, t, &mut lower);
            add(conn.a, conn.b, -t, &mut lower);
        }
        for (f, fm) in fixed.iter().zip(&face_mob[nconn..]) {
            check_cell(f.cell, "fixed pressure")?;
            let t = f.trans * fm;
            add(f.cell, f.cell, t, &mut lower);
            rhs[f.cell] += t * f.pressure;
        }
        let mut k = ncomp_off;
        for (((w, wi), unk), ctl) in wells.iter().zip(&wis).zip(&bhp_unknown).zip(ctrl) {
            let mut tsum = 0.0;
            for (&c, &wic) in w.cells.iter().zip(wi) {
                let t = wic * face_mob[k];
                k += 1;
                tsum += t;
                add(c, c, t, &mut lower);
                match (ctl, unk) {
                    (WellControl::Bhp(pw), _) => rhs[c] += t * pw,
                    (WellControl::Rate(_), Some(u)) => {
                        add(*u, *u, t, &mut lower);
                        add(*u, c, -t, &mut lower);
                    }
                    _ => {}
                }
            }
            if let (WellControl::Rate(q), Some(u)) = (ctl, unk) {
                if tsum <= 0.0 {
                    return Err(SxError::Numerical(format!(
                        "pressure solve: rate well {} has every completion shut",
                        w.name
                    )));
                }
                rhs[*u] += match w.kind {
                    WellKind::Injector => *q,
                    WellKind::Producer => -q,
                };
            }
        }
        if !lower.all_finite() {
            return Err(numerical("matrix has non-finite entries (check mobility)"));
        }
        cache.borrow_mut().solve(n_unk, &lower, &mut rhs)?;
        Ok(rhs)
    };
    let mean_mob = all_mob(None);
    let x_mean = assemble_solve(&mean_mob)?;
    let (mut x, mut face_mob) = (x_mean.clone(), mean_mob.clone());
    let mut iterations = 0;
    let mut warnings = Vec::new();
    loop {
        let next = all_mob(Some(&x));
        // Directions settle exactly; the wellbore-mixture mobility of a crossflowing producer to 1e-10.
        let settled = next.iter().zip(&face_mob).all(|(a, b)| (a - b).abs() <= 1e-10 * a.abs().max(b.abs()));
        if iterations > 0 && settled {
            break;
        }
        if iterations == max_iter {
            warnings.push(format!(
                "pressure solve: upwind directions did not settle in {max_iter} iterations; \
                 used the arithmetic-mean face mobility"
            ));
            x = x_mean;
            face_mob = mean_mob;
            break;
        }
        x = assemble_solve(&next)?;
        face_mob = next;
        iterations += 1;
    }
    let mut p: Vec<f64> = x[..n].to_vec();
    for (c, v) in p.iter_mut().enumerate() {
        if !model.active[c] {
            *v = f64::NAN;
        }
    }
    let fl = |(c, fm): (&super::model::Connection, &f64)| c.trans * fm * (p[c.a] - p[c.b]);
    let flux = if par_loops(nconn) {
        model.connections.par_iter().zip(&face_mob[..nconn]).with_min_len(PAR_MIN_LEN).map(fl).collect()
    } else {
        model.connections.iter().zip(&face_mob).map(fl).collect()
    };
    let boundary_flux = fixed
        .iter()
        .zip(&face_mob[nconn..])
        .map(|(f, fm)| f.trans * fm * (f.pressure - p[f.cell]))
        .collect();
    let mut out_wells = Vec::with_capacity(wells.len());
    let mut k = ncomp_off;
    for (((w, wi), unk), ctl) in wells.iter().zip(wis).zip(&bhp_unknown).zip(ctrl) {
        let bhp = match (ctl, unk) {
            (WellControl::Bhp(pw), _) => *pw,
            (_, Some(u)) => x[*u],
            _ => f64::NAN,
        };
        let cell_rates: Vec<f64> = w
            .cells
            .iter()
            .zip(&wi)
            .map(|(&c, &wic)| {
                k += 1;
                wic * face_mob[k - 1] * (bhp - p[c])
            })
            .collect();
        out_wells.push(WellResult {
            bhp,
            rate: cell_rates.iter().sum(),
            cell_rates,
            wi,
            bhp_limited: false,
        });
    }
    Ok(PressureSolution {
        p,
        flux,
        boundary_flux,
        wells: out_wells,
        upwind_iterations: iterations,
        warnings,
    })
}

/// Well indices of a well: the override if given, else vertical Peaceman.
pub(super) fn well_indices(model: &FlowModel, w: &FlowWell) -> SxResult<Vec<f64>> {
    if !(w.radius_m > 0.0) || w.cells.is_empty() {
        return Err(SxError::InvalidInput(format!(
            "well {}: needs cells and a radius > 0",
            w.name
        )));
    }
    for &c in &w.cells {
        if c >= model.n_cells() || !model.active[c] {
            return Err(SxError::InvalidInput(format!(
                "well {}: cell {c} is outside the grid or inactive",
                w.name
            )));
        }
    }
    if let Some(wi) = &w.wi {
        if wi.len() != w.cells.len() || wi.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(SxError::InvalidInput(format!(
                "well {}: wi needs one finite value >= 0 per cell",
                w.name
            )));
        }
        return Ok(wi.clone());
    }
    Ok(w.cells
        .iter()
        .map(|&c| {
            let [dx, dy, dz] = model.extent[c];
            let [kx, ky, _] = model.perm[c];
            peaceman_wi(kx, ky, dx, dy, dz, w.radius_m, w.skin)
        })
        .collect())
}

/// Solve a symmetric positive definite system given by its lower triangle
/// (one-off; repeated solves should keep a [`SpdCache`]).
#[allow(dead_code)]
pub(super) fn solve_spd(
    n_unk: usize,
    lower: &std::collections::BTreeMap<(usize, usize), f64>,
    mut rhs: Vec<f64>,
) -> SxResult<Vec<f64>> {
    let mut e = SpdEntries::default();
    for (&(r, c), &v) in lower {
        e.add(r, c, v);
    }
    SpdCache::default().solve(n_unk, &e, &mut rhs)?;
    Ok(rhs)
}
