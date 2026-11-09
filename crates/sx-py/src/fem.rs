// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for the experimental continuous Galerkin finite-element solver (FEM)
//! (`sx_core::flow::fem`): water flow and heat transport in the style of the
//! OpenGeoSys HT process.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sx_core::flow::{self, FemOptions, ThermalWater};

use crate::errors::to_py_err;
use crate::fim::{arr3, rock_of, thermal_injection};
use crate::flow::{rows2, wells_of, PyFlowModel};

/// **Experimental.** Finite-element single-phase liquid water with heat
/// transport (OpenGeoSys HT process): pressure and temperature at the cell
/// corners (trilinear hexahedra), with the water (IAPWS-IF97), rock and
/// Peaceman wells of `run_fim_thermal_water`.
///
/// Arguments as `run_fim_thermal_water` without the OBL table (`p_range`,
/// `t_range`, `table_points`); `max_iter` Picard iterations per step;
/// `stabilization` the streamline-upwind factor (1 = first-order upwind,
/// 0 = plain Galerkin). Faulted grids are rejected. Returns a dict with the
/// keys of `run_fim_thermal_water` that apply: times, p (Pa) and
/// temperature (K) (n_t, n) at the cell centres, bhp, well_mass_rate
/// (n_t, n_wells, 2; water kg/s, energy W, + = into the reservoir),
/// mass_in_place and mass_net_in (n_t, 2; kg, J), phase_names,
/// component_names, n_steps, n_cuts, n_fallbacks, n_linear_solves,
/// n_linear_iterations, n_linear_fallbacks; plus p_nodes and
/// temperature_nodes (n_t, n_nodes), nodes (n_nodes, 3; x, y, depth in m)
/// and n_iterations (Picard).
#[pyfunction]
#[pyo3(signature = (model, wells, p0, t0, times, injection_temperature=None, rock=None, c_rock=0.0, p_ref=1e5, gravity=0.0, dt_init=3600.0, dt_max=30.0*86400.0, dt_min=1e-3, max_iter=30, tol=1e-6, stabilization=1.0, linear_solver="iterative"))]
#[allow(clippy::too_many_arguments)]
pub fn run_fem_thermal_water<'py>(
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
    dt_init: f64,
    dt_max: f64,
    dt_min: f64,
    max_iter: usize,
    tol: f64,
    stabilization: f64,
    linear_solver: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ws = wells_of(wells)?;
    let ph = ThermalWater::new(rock_of(rock)?, c_rock, p_ref);
    let (p0, t0) = (p0.as_slice()?.to_vec(), t0.as_slice()?.to_vec());
    let n = model.m.n_cells();
    if p0.len() != n || t0.len() != n {
        return Err(PyValueError::new_err("p0 and t0 need one value per cell"));
    }
    let linear_solver = flow::LinearSolver::from_name(linear_solver).ok_or_else(|| PyValueError::new_err(format!("linear_solver must be 'iterative' or 'direct', got {linear_solver:?}")))?;
    let opts = FemOptions {
        gravity,
        injection_temperature: thermal_injection(&ws, injection_temperature, &t0)?,
        dt_init,
        dt_max,
        dt_min,
        max_iter,
        tol,
        stabilization,
        linear_solver,
        ..FemOptions::default()
    };
    let (m, corners) = (&model.m, &model.corners);
    let r = py.detach(|| flow::run_fem_ht(corners, m, &ws, &ph, &p0, &t0, &times, &opts)).map_err(to_py_err)?;
    if r.n_fallbacks > 0 {
        let msg = std::ffi::CString::new(format!(
            "fem: {} water-property evaluations failed and used fallback values (state outside the property models' range); results near those states are not physical",
            r.n_fallbacks
        ))
        .unwrap_or_default();
        PyErr::warn(py, &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(), &msg, 1)?;
    }
    let nodes: Vec<Vec<f64>> = r.nodes.iter().map(|x| x.to_vec()).collect();
    let d = PyDict::new(py);
    d.set_item("times", r.times.clone())?;
    d.set_item("p", rows2(py, &r.p)?)?;
    d.set_item("temperature", rows2(py, &r.temperature)?)?;
    d.set_item("p_nodes", rows2(py, &r.p_nodes)?)?;
    d.set_item("temperature_nodes", rows2(py, &r.temperature_nodes)?)?;
    d.set_item("nodes", rows2(py, &nodes)?)?;
    d.set_item("bhp", rows2(py, &r.bhp)?)?;
    d.set_item("well_mass_rate", arr3(py, &r.well_mass_rate, ws.len(), 2)?)?;
    d.set_item("mass_in_place", rows2(py, &r.mass_in_place)?)?;
    d.set_item("mass_net_in", rows2(py, &r.mass_net_in)?)?;
    d.set_item("phase_names", vec!["water"])?;
    d.set_item("component_names", vec!["water", "energy"])?;
    d.set_item("n_steps", r.n_steps)?;
    d.set_item("n_cuts", r.n_cuts)?;
    d.set_item("n_iterations", r.n_iterations)?;
    d.set_item("n_fallbacks", r.n_fallbacks)?;
    d.set_item("n_linear_solves", r.linear.n_solves)?;
    d.set_item("n_linear_iterations", r.linear.n_iterations)?;
    d.set_item("n_linear_fallbacks", r.linear.n_fallbacks)?;
    Ok(d)
}
