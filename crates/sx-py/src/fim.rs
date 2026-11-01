// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for the experimental fully implicit OBL engine
//! (`sx_core::flow::implicit`): dead-oil (water–oil), black-oil, CO2–brine
//! and thermal-water runs.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sx_core::flow::{self, BlackOil, Co2Brine, DeadOil, FimOptions, FimResult, OblPhysics, PvtRow, RockThermal, ThermalWater, WellKind};

use crate::errors::to_py_err;
use crate::flow::{corey_of, physics_of, rows2, wells_of, PyFlowModel};

fn arr3<'py>(py: Python<'py>, rows: &[Vec<f64>], b: usize, c: usize) -> PyResult<Bound<'py, PyAny>> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Ok(PyArray1::from_vec(py, flat).reshape([rows.len(), b, c])?.into_any())
}

fn result_dict<'py>(py: Python<'py>, r: FimResult, ph: &dyn OblPhysics, n: usize, nw: usize) -> PyResult<Bound<'py, PyDict>> {
    let (nv, np) = (ph.n_vars(), ph.n_phases());
    let nz = if ph.thermal() { nv - 2 } else { nv - 1 };
    if r.n_fallbacks > 0 {
        let msg = std::ffi::CString::new(format!(
            "fim: {} property evaluations failed and used fallback values (state outside the property models' range); results near those states are not physical",
            r.n_fallbacks
        ))
        .unwrap_or_default();
        PyErr::warn(py, &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(), &msg, 1)?;
    }
    let d = PyDict::new(py);
    d.set_item("times", r.times.clone())?;
    let p: Vec<Vec<f64>> = r.state.iter().map(|s| (0..n).map(|i| s[i * nv]).collect()).collect();
    d.set_item("p", rows2(py, &p)?)?;
    let z: Vec<Vec<f64>> = r.state.iter().map(|s| (0..n).flat_map(|i| s[i * nv + 1..i * nv + 1 + nz].to_vec()).collect()).collect();
    d.set_item("z", arr3(py, &z, n, nz)?)?;
    if ph.thermal() {
        let t: Vec<Vec<f64>> = r.state.iter().map(|s| (0..n).map(|i| s[i * nv + nv - 1]).collect()).collect();
        d.set_item("temperature", rows2(py, &t)?)?;
    }
    d.set_item("saturation", arr3(py, &r.saturation, n, np)?)?;
    d.set_item("bhp", rows2(py, &r.bhp)?)?;
    d.set_item("well_mass_rate", arr3(py, &r.well_mass_rate, nw, nv)?)?;
    d.set_item("mass_in_place", rows2(py, &r.mass_in_place)?)?;
    d.set_item("mass_net_in", rows2(py, &r.mass_net_in)?)?;
    d.set_item("phase_names", ph.phase_names())?;
    d.set_item("component_names", ph.component_names())?;
    d.set_item("n_steps", r.n_steps)?;
    d.set_item("n_cuts", r.n_cuts)?;
    d.set_item("n_newton", r.n_newton)?;
    d.set_item("n_table_points", r.n_table_points)?;
    d.set_item("n_fallbacks", r.n_fallbacks)?;
    d.set_item("n_linear_solves", r.linear.n_solves)?;
    d.set_item("n_linear_iterations", r.linear.n_iterations)?;
    d.set_item("n_linear_fallbacks", r.linear.n_fallbacks)?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn options(
    nv: usize,
    p_range: (f64, f64),
    table_points: Vec<usize>,
    z_hi: Vec<f64>,
    injection: Vec<Vec<f64>>,
    gravity: f64,
    dt_init: f64,
    dt_max: f64,
    dt_min: f64,
    max_newton: usize,
    tol: f64,
    linear_solver: &str,
) -> PyResult<FimOptions> {
    let linear_solver = flow::LinearSolver::from_name(linear_solver).ok_or_else(|| PyValueError::new_err(format!("linear_solver must be 'iterative' or 'direct', got {linear_solver:?}")))?;
    if table_points.len() != nv {
        return Err(PyValueError::new_err(format!("table_points needs {nv} values")));
    }
    let mut lo = vec![p_range.0];
    let mut hi = vec![p_range.1];
    lo.extend(std::iter::repeat_n(0.0, nv - 1));
    hi.extend(z_hi);
    Ok(FimOptions {
        gravity,
        table_lo: lo,
        table_hi: hi,
        table_points,
        injection,
        dt_init,
        dt_max,
        dt_min,
        max_newton,
        tol,
        linear_solver,
        ..FimOptions::default()
    })
}

/// **Experimental.** Fully implicit water–oil (dead-oil) run with
/// operator-based linearization (Voskov 2017; Khait & Voskov 2017).
///
/// `corey` and `physics` are the same dicts as `FlowModel.run_two_phase`
/// (physics: gravity, rho_w, rho_o, c_rock, c_w, c_o, p_ref, capillary…).
/// Wells: as `run_two_phase` (crossflow is ignored: always off). Injectors
/// inject water unless `injection` gives the water mass fraction per well.
/// `p_range` (Pa) bounds the OBL table; pressures are clamped into it, so
/// make it wide enough. Returns a dict with times, p (n_t, n), z (n_t, n, 1),
/// saturation (n_t, n, 2: water, oil), bhp, well_mass_rate (n_t, n_wells, 2;
/// kg/s, + = into the reservoir), mass_in_place, mass_net_in (n_t, 2; kg),
/// n_steps, n_cuts, n_newton, n_table_points.
#[pyfunction]
#[pyo3(signature = (model, wells, sw0, p0, times, corey=None, physics=None, injection=None, p_range=(1e5, 1e8), table_points=(32, 256), dt_init=86400.0, dt_max=30.0*86400.0, dt_min=1e-3, max_newton=20, tol=1e-6, linear_solver="iterative"))]
#[allow(clippy::too_many_arguments)]
pub fn run_fim_water_oil<'py>(
    py: Python<'py>,
    model: &PyFlowModel,
    wells: &Bound<'py, PyList>,
    sw0: PyReadonlyArray1<'_, f64>,
    p0: PyReadonlyArray1<'_, f64>,
    times: Vec<f64>,
    corey: Option<&Bound<'py, PyDict>>,
    physics: Option<&Bound<'py, PyDict>>,
    injection: Option<Vec<f64>>,
    p_range: (f64, f64),
    table_points: (usize, usize),
    dt_init: f64,
    dt_max: f64,
    dt_min: f64,
    max_newton: usize,
    tol: f64,
    linear_solver: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ws = wells_of(wells)?;
    let c = corey_of(corey)?;
    let pp = physics_of(physics)?;
    let ph = DeadOil { corey: c, capillary: pp.capillary, rho_w: pp.rho_w, rho_o: pp.rho_o, c_w: pp.c_w, c_o: pp.c_o, c_rock: pp.c_rock, p_ref: pp.p_ref };
    let (sw0, p0) = (sw0.as_slice()?, p0.as_slice()?);
    let n = model.m.n_cells();
    if sw0.len() != n || p0.len() != n {
        return Err(PyValueError::new_err("sw0 and p0 need one value per cell"));
    }
    let state0: Vec<f64> = (0..n).flat_map(|i| [p0[i], ph.z_from_sw(p0[i], sw0[i])]).collect();
    let inj: Vec<Vec<f64>> = match injection {
        Some(v) if v.len() == ws.len() => v.into_iter().map(|z| vec![z]).collect(),
        Some(_) => return Err(PyValueError::new_err("injection needs one water mass fraction per well")),
        None => ws.iter().map(|w| vec![if w.kind == WellKind::Injector { 1.0 } else { 0.0 }]).collect(),
    };
    let opts = options(2, p_range, vec![table_points.0, table_points.1], vec![1.0], inj, pp.gravity, dt_init, dt_max, dt_min, max_newton, tol, linear_solver)?;
    let m = &model.m;
    let r = py.detach(|| flow::run_fim(m, &ws, &ph, &state0, &times, &opts)).map_err(to_py_err)?;
    result_dict(py, r, &ph, n, ws.len())
}

/// **Experimental.** Fully implicit black-oil run (gas dissolved in oil;
/// components and phases gas, oil, water) with OBL.
///
/// `pvt`: (n_rows, 6) array of p (Pa), Rs (sm³/sm³), Bo, μo (Pa·s), Bg,
/// μg (Pa·s), increasing p. `zg0`, `zo0`: initial gas and oil mass
/// fractions per cell. `injection`: [zg, zo] per well (default pure water
/// for injectors). Corey per phase (gas, oil, water) from `s_res`, `n_exp`,
/// `kr_max`. No capillary pressure. Returns the same dict as
/// `run_fim_water_oil` with 3 phases/components.
#[pyfunction]
#[pyo3(signature = (model, wells, p0, zg0, zo0, times, pvt, rho_sc=(0.9, 850.0, 1020.0), bw=1.0, c_w=4e-10, mu_w=5e-4, s_res=(0.05, 0.15, 0.2), n_exp=(2.0, 2.0, 2.0), kr_max=(0.8, 0.9, 0.4), c_o_undersat=1.5e-9, c_rock=0.0, p_ref=1e5, gravity=0.0, injection=None, p_range=(1e5, 5e7), table_points=(64, 101, 101), z_gas_max=1.0, dt_init=86400.0, dt_max=30.0*86400.0, dt_min=1e-3, max_newton=20, tol=1e-6, linear_solver="iterative"))]
#[allow(clippy::too_many_arguments)]
pub fn run_fim_black_oil<'py>(
    py: Python<'py>,
    model: &PyFlowModel,
    wells: &Bound<'py, PyList>,
    p0: PyReadonlyArray1<'_, f64>,
    zg0: PyReadonlyArray1<'_, f64>,
    zo0: PyReadonlyArray1<'_, f64>,
    times: Vec<f64>,
    pvt: PyReadonlyArray2<'_, f64>,
    rho_sc: (f64, f64, f64),
    bw: f64,
    c_w: f64,
    mu_w: f64,
    s_res: (f64, f64, f64),
    n_exp: (f64, f64, f64),
    kr_max: (f64, f64, f64),
    c_o_undersat: f64,
    c_rock: f64,
    p_ref: f64,
    gravity: f64,
    injection: Option<Vec<(f64, f64)>>,
    p_range: (f64, f64),
    table_points: (usize, usize, usize),
    z_gas_max: f64,
    dt_init: f64,
    dt_max: f64,
    dt_min: f64,
    max_newton: usize,
    tol: f64,
    linear_solver: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ws = wells_of(wells)?;
    let t = pvt.as_array();
    if t.ncols() != 6 || t.nrows() < 2 {
        return Err(PyValueError::new_err("pvt must be (n_rows >= 2, 6): p, Rs, Bo, mu_o, Bg, mu_g"));
    }
    let rows: Vec<PvtRow> = (0..t.nrows())
        .map(|i| PvtRow { p: t[[i, 0]], rs: t[[i, 1]], bo: t[[i, 2]], mu_o: t[[i, 3]], bg: t[[i, 4]], mu_g: t[[i, 5]] })
        .collect();
    if rows.windows(2).any(|w| w[1].p.partial_cmp(&w[0].p) != Some(std::cmp::Ordering::Greater) || w[1].rs < w[0].rs) || !rows.iter().all(|r| [r.bo, r.bg, r.mu_o, r.mu_g].iter().all(|v| v.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater))) {
        return Err(PyValueError::new_err("pvt: p must increase, Rs must not decrease, Bo, Bg, mu > 0"));
    }
    let ph = BlackOil {
        pvt: rows,
        rho_sc: [rho_sc.0, rho_sc.1, rho_sc.2],
        bw,
        c_w,
        mu_w,
        s_res: [s_res.0, s_res.1, s_res.2],
        n_exp: [n_exp.0, n_exp.1, n_exp.2],
        kr_max: [kr_max.0, kr_max.1, kr_max.2],
        c_o_undersat,
        c_rock,
        p_ref,
    };
    let (p0, zg0, zo0) = (p0.as_slice()?, zg0.as_slice()?, zo0.as_slice()?);
    let n = model.m.n_cells();
    if p0.len() != n || zg0.len() != n || zo0.len() != n {
        return Err(PyValueError::new_err("p0, zg0, zo0 need one value per cell"));
    }
    let state0: Vec<f64> = (0..n).flat_map(|i| [p0[i], zg0[i], zo0[i]]).collect();
    let inj: Vec<Vec<f64>> = match injection {
        Some(v) if v.len() == ws.len() => v.into_iter().map(|(a, b)| vec![a, b]).collect(),
        Some(_) => return Err(PyValueError::new_err("injection needs one (zg, zo) per well")),
        None => ws.iter().map(|_| vec![0.0, 0.0]).collect(),
    };
    let opts = options(3, p_range, vec![table_points.0, table_points.1, table_points.2], vec![z_gas_max, 1.0], inj, gravity, dt_init, dt_max, dt_min, max_newton, tol, linear_solver)?;
    let m = &model.m;
    let r = py.detach(|| flow::run_fim(m, &ws, &ph, &state0, &times, &opts)).map_err(to_py_err)?;
    result_dict(py, r, &ph, n, ws.len())
}

/// Water mass fraction `z_w` of the dead-oil state at pressure `p` and water
/// saturation `sw` (densities from the same `physics` dict).
#[pyfunction]
#[pyo3(signature = (p, sw, physics=None))]
pub fn fim_water_mass_fraction(p: f64, sw: f64, physics: Option<&Bound<'_, PyDict>>) -> PyResult<f64> {
    let pp = physics_of(physics)?;
    let ph = DeadOil { rho_w: pp.rho_w, rho_o: pp.rho_o, c_w: pp.c_w, c_o: pp.c_o, p_ref: pp.p_ref, ..DeadOil::default() };
    Ok(ph.z_from_sw(p, sw))
}

fn rock_of(rock: Option<&Bound<'_, PyDict>>) -> PyResult<RockThermal> {
    let mut r = RockThermal::default();
    if let Some(d) = rock {
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            let v: f64 = v.extract()?;
            match key.as_str() {
                "rho" => r.rho = v,
                "cp" => r.cp = v,
                "conductivity" => r.conductivity = v,
                _ => return Err(PyValueError::new_err(format!("rock: unknown key {key:?} (rho, cp, conductivity)"))),
            }
        }
    }
    Ok(r)
}

fn thermal_injection(ws: &[flow::FlowWell], inj_t: Option<Vec<f64>>, t0: &[f64]) -> PyResult<Vec<f64>> {
    match inj_t {
        Some(v) if v.len() == ws.len() => Ok(v),
        Some(_) => Err(PyValueError::new_err("injection_temperature needs one value (K) per well")),
        None if ws.iter().all(|w| w.kind != WellKind::Injector) => Ok(ws.iter().map(|w| t0[w.cells[0]]).collect()),
        None => Err(PyValueError::new_err("injection_temperature (K, one per well) is required with injectors")),
    }
}

/// **Experimental.** Fully implicit CO2 injection into brine with OBL
/// (CO2–brine properties of `co2_brine_flash`: Spycher & Pruess 2005 by
/// default, or `solubility_model="ziabakhsh_ganji_kooi_2012"`).
///
/// State: pressure `p0` (Pa) and overall CO2 mass fraction `z0` (salt-free
/// basis; 0 = brine) per cell; NaCl molality `m_nacl` is fixed. Isothermal at
/// `temperature` (K) unless `thermal=True`, then `t0` (K per cell) is the
/// initial temperature, the energy equation is solved and
/// `injection_temperature` (K per well) is required for injectors.
/// Injectors inject pure CO2 unless `injection` gives the CO2 mass fraction
/// per well; rate control is on reservoir volume (m³/s). Corey relative
/// permeability (`s_res`, `n_exp`, `kr_max`: aqueous, gas), no capillary
/// pressure. `p_range`/`t_range` bound the OBL table (states are clamped
/// into it). Returns the dict of `run_fim_water_oil` with phases
/// (aqueous, gas), components (CO2, H2O[, energy]), `z` (n_t, n, 1: CO2
/// mass fraction), `temperature` (n_t, n) if thermal, and `n_fallbacks`
/// (failed flashes that used fallback values; a RuntimeWarning is issued
/// when nonzero).
#[pyfunction]
#[pyo3(signature = (model, wells, p0, z0, times, temperature=323.15, m_nacl=0.0, solubility_model="spycher_pruess_2005", thermal=false, t0=None, injection_temperature=None, injection=None, s_res=(0.2, 0.05), n_exp=(2.0, 2.0), kr_max=(1.0, 1.0), c_rock=0.0, p_ref=1e5, rock=None, gravity=0.0, p_range=(1e5, 4e7), t_range=(285.15, 373.15), table_points=None, dt_init=3600.0, dt_max=30.0*86400.0, dt_min=1e-3, max_newton=20, tol=1e-6, linear_solver="iterative"))]
#[allow(clippy::too_many_arguments)]
pub fn run_fim_co2_brine<'py>(
    py: Python<'py>,
    model: &PyFlowModel,
    wells: &Bound<'py, PyList>,
    p0: PyReadonlyArray1<'_, f64>,
    z0: PyReadonlyArray1<'_, f64>,
    times: Vec<f64>,
    temperature: f64,
    m_nacl: f64,
    solubility_model: &str,
    thermal: bool,
    t0: Option<PyReadonlyArray1<'_, f64>>,
    injection_temperature: Option<Vec<f64>>,
    injection: Option<Vec<f64>>,
    s_res: (f64, f64),
    n_exp: (f64, f64),
    kr_max: (f64, f64),
    c_rock: f64,
    p_ref: f64,
    rock: Option<&Bound<'py, PyDict>>,
    gravity: f64,
    p_range: (f64, f64),
    t_range: (f64, f64),
    table_points: Option<Vec<usize>>,
    dt_init: f64,
    dt_max: f64,
    dt_min: f64,
    max_newton: usize,
    tol: f64,
    linear_solver: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ws = wells_of(wells)?;
    let mut ph = Co2Brine::new(m_nacl, crate::co2::model_of(solubility_model)?, thermal, temperature);
    ph.s_res = [s_res.0, s_res.1];
    ph.n_exp = [n_exp.0, n_exp.1];
    ph.kr_max = [kr_max.0, kr_max.1];
    ph.c_rock = c_rock;
    ph.p_ref = p_ref;
    ph.rock = rock_of(rock)?;
    let (p0, z0) = (p0.as_slice()?, z0.as_slice()?);
    let n = model.m.n_cells();
    if p0.len() != n || z0.len() != n {
        return Err(PyValueError::new_err("p0 and z0 need one value per cell"));
    }
    let t0v: Vec<f64> = match (&t0, thermal) {
        (Some(t), true) => t.as_slice()?.to_vec(),
        (None, true) => return Err(PyValueError::new_err("thermal=True needs t0 (K per cell)")),
        (Some(_), false) => return Err(PyValueError::new_err("t0 is only used with thermal=True")),
        (None, false) => Vec::new(),
    };
    if thermal && t0v.len() != n {
        return Err(PyValueError::new_err("t0 needs one value per cell"));
    }
    let zi: Vec<f64> = match injection {
        Some(v) if v.len() == ws.len() => v,
        Some(_) => return Err(PyValueError::new_err("injection needs one CO2 mass fraction per well")),
        None => ws.iter().map(|w| if w.kind == WellKind::Injector { 1.0 } else { 0.0 }).collect(),
    };
    let nv = ph.n_vars();
    let tp = table_points.unwrap_or(if thermal { vec![32, 101, 36] } else { vec![64, 201] });
    let (inj, state0): (Vec<Vec<f64>>, Vec<f64>) = if thermal {
        let ti = thermal_injection(&ws, injection_temperature, &t0v)?;
        (zi.iter().zip(&ti).map(|(&z, &t)| vec![z, t]).collect(), (0..n).flat_map(|i| [p0[i], z0[i], t0v[i]]).collect())
    } else {
        if injection_temperature.is_some() {
            return Err(PyValueError::new_err("injection_temperature is only used with thermal=True"));
        }
        (zi.iter().map(|&z| vec![z]).collect(), (0..n).flat_map(|i| [p0[i], z0[i]]).collect())
    };
    let mut opts = options(nv, p_range, tp, vec![1.0; nv - 1], inj, gravity, dt_init, dt_max, dt_min, max_newton, tol, linear_solver)?;
    if thermal {
        opts.table_lo[2] = t_range.0;
        opts.table_hi[2] = t_range.1;
    }
    let m = &model.m;
    let r = py.detach(|| flow::run_fim(m, &ws, &ph, &state0, &times, &opts)).map_err(to_py_err)?;
    result_dict(py, r, &ph, n, ws.len())
}

/// **Experimental.** Fully implicit single-phase liquid water with an energy
/// equation (IAPWS-IF97 water, rock heat capacity and conduction).
///
/// `p0` (Pa) and `t0` (K) per cell; `injection_temperature` (K per well) is
/// required with injectors. `rock`: dict with rho, cp, conductivity (default
/// 2650 kg/m³, 900 J/(kg K), 2.5 W/(m K)). Returns the dict of
/// `run_fim_water_oil` with `temperature` (n_t, n), components (water,
/// energy; energy rates W, energy in place J) and `n_fallbacks`.
#[pyfunction]
#[pyo3(signature = (model, wells, p0, t0, times, injection_temperature=None, rock=None, c_rock=0.0, p_ref=1e5, gravity=0.0, p_range=(1e5, 5e7), t_range=(275.0, 420.0), table_points=(32, 146), dt_init=3600.0, dt_max=30.0*86400.0, dt_min=1e-3, max_newton=20, tol=1e-6, linear_solver="iterative"))]
#[allow(clippy::too_many_arguments)]
pub fn run_fim_thermal_water<'py>(
    py: Python<'py>,
    model: &PyFlowModel,
    wells: &Bound<'py, PyList>,
    p0: PyReadonlyArray1<'_, f64>,
    t0: PyReadonlyArray1<'_, f64>,
    times: Vec<f64>,
    injection_temperature: Option<Vec<f64>>,
    rock: Option<&Bound<'py, PyDict>>,
    c_rock: f64,
    p_ref: f64,
    gravity: f64,
    p_range: (f64, f64),
    t_range: (f64, f64),
    table_points: (usize, usize),
    dt_init: f64,
    dt_max: f64,
    dt_min: f64,
    max_newton: usize,
    tol: f64,
    linear_solver: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ws = wells_of(wells)?;
    let ph = ThermalWater::new(rock_of(rock)?, c_rock, p_ref);
    let (p0, t0) = (p0.as_slice()?, t0.as_slice()?);
    let n = model.m.n_cells();
    if p0.len() != n || t0.len() != n {
        return Err(PyValueError::new_err("p0 and t0 need one value per cell"));
    }
    let ti = thermal_injection(&ws, injection_temperature, t0)?;
    let inj: Vec<Vec<f64>> = ti.iter().map(|&t| vec![t]).collect();
    let state0: Vec<f64> = (0..n).flat_map(|i| [p0[i], t0[i]]).collect();
    let mut opts = options(2, p_range, vec![table_points.0, table_points.1], vec![1.0], inj, gravity, dt_init, dt_max, dt_min, max_newton, tol, linear_solver)?;
    opts.table_lo[1] = t_range.0;
    opts.table_hi[1] = t_range.1;
    let m = &model.m;
    let r = py.detach(|| flow::run_fim(m, &ws, &ph, &state0, &times, &opts)).map_err(to_py_err)?;
    result_dict(py, r, &ph, n, ws.len())
}
