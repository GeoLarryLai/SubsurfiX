// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for the experimental CO2 vertical-equilibrium model
//! (`sx_core::flow::ve`).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sx_core::flow::ve::{self, VeFluid, VeGrid, VeOptions, VeWell};

use crate::errors::to_py_err;
use crate::flow::rows2;
use crate::grid::PyCornerPointGrid;

/// Top-surface (2D) grid for CO2 vertical-equilibrium runs.
#[pyclass(name = "VeGrid", module = "subsurfix._native")]
pub struct PyVeGrid {
    pub(crate) g: VeGrid,
}

#[pymethods]
impl PyVeGrid {
    /// Cartesian top surface: node coordinates `xs` (nx+1), `ys` (ny+1);
    /// per column (i + nx j) caprock depth (m, down), thickness (m),
    /// porosity and horizontal permeability (m²).
    #[staticmethod]
    fn from_cartesian(
        xs: Vec<f64>,
        ys: Vec<f64>,
        top: Vec<f64>,
        thickness: Vec<f64>,
        poro: Vec<f64>,
        perm: Vec<f64>,
    ) -> PyResult<Self> {
        Ok(Self { g: VeGrid::from_cartesian(&xs, &ys, &top, &thickness, &poro, &perm).map_err(to_py_err)? })
    }

    /// Top surface of a corner-point grid; `perm` (n_cells, 3) m², `poro` (n_cells,).
    #[staticmethod]
    fn from_corner_point(grid: PyRef<'_, PyCornerPointGrid>, perm: PyReadonlyArray2<'_, f64>, poro: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let a = perm.as_array();
        if a.ncols() != 3 {
            return Err(PyValueError::new_err("perm must be (n_cells, 3)"));
        }
        let p: Vec<[f64; 3]> = a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        Ok(Self { g: VeGrid::from_corner_point(&grid.g, &p, poro.as_slice()?).map_err(to_py_err)? })
    }

    /// Number of VE cells (active columns).
    #[getter]
    fn n_cells(&self) -> usize {
        self.g.n_cells()
    }
    /// Caprock depth per cell (m).
    #[getter]
    fn top<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.g.top)
    }
    /// Thickness per cell (m).
    #[getter]
    fn thickness<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.g.height)
    }
    /// Porosity per cell.
    #[getter]
    fn poro<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.g.poro)
    }
    /// Plan-view area per cell (m²).
    #[getter]
    fn area<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.g.area)
    }
    /// Source-grid column index (i + nx j) per cell.
    #[getter]
    fn column<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_iter(py, self.g.column.iter().map(|&c| c as i64))
    }
    /// Cell centres (n, 2).
    fn centers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        rows2(py, &self.g.center.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
    }

    /// Spill-point analysis: dict with `trap` (id per cell, -1 = none),
    /// `spill_depth` per trap and `fill_depth` per cell.
    #[pyo3(signature = (tol=1e-3))]
    fn trap_analysis<'py>(&self, py: Python<'py>, tol: f64) -> PyResult<Bound<'py, PyDict>> {
        let t = ve::trap_analysis(&self.g, tol);
        let d = PyDict::new(py);
        d.set_item("trap", PyArray1::from_iter(py, t.trap.iter().map(|x| x.map_or(-1, |v| v as i64))))?;
        d.set_item("spill_depth", PyArray1::from_vec(py, t.spill_depth))?;
        d.set_item("fill_depth", PyArray1::from_vec(py, t.fill_depth))?;
        Ok(d)
    }
}

/// CO2 and brine densities (kg/m³) and viscosities (Pa·s) used by the VE
/// model, from the CO2–brine property module at `p` (Pa), `t` (K), NaCl
/// molality `m_nacl`.
#[pyfunction]
#[pyo3(signature = (p, t, m_nacl=0.0))]
pub fn ve_fluid_properties(p: f64, t: f64, m_nacl: f64) -> PyResult<std::collections::BTreeMap<String, f64>> {
    let f = VeFluid::from_co2_brine(p, t, m_nacl, 0.0, 0.0).map_err(to_py_err)?;
    Ok([("rho_co2", f.rho_co2), ("rho_brine", f.rho_brine), ("mu_co2", f.mu_co2), ("mu_brine", f.mu_brine)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect())
}

/// Nordbotten & Celia (2006) similarity solution h(r) (gravity neglected).
#[pyfunction]
pub fn nordbotten_celia_h(r: f64, volume: f64, thickness: f64, phi_eff: f64, mobility_ratio: f64) -> f64 {
    ve::nordbotten_celia_h(r, volume, thickness, phi_eff, mobility_ratio)
}

/// **Experimental.** CO2 vertical-equilibrium sharp-interface run (IMPES).
///
/// `wells`: list of dicts (cell, rate m³/s CO2 at reservoir conditions,
/// t_start=0, t_end=inf). Fluid: `p_fluid`/`t_fluid`/`m_nacl` take
/// densities and viscosities from the CO2–brine module, unless `fluid`
/// (dict rho_co2, rho_brine, mu_co2, mu_brine) overrides them.
/// Returns dict: times, h, h_max, p, volume (n_times, n_cells), inventory
/// (n_times, 6) kg with `inventory_names`, trap, spill_depth, n_steps, and
/// the fluid used.
#[pyfunction]
#[pyo3(signature = (grid, wells, times, s_wr=0.1, s_gr=0.2, p_fluid=1e7, t_fluid=313.15, m_nacl=0.0, fluid=None, gravity=9.80665, open_boundary=true, p_ref=None, z_ref=None, cfl=0.5, dt_max=365.25*86400.0, trap_tol=1e-3))]
#[allow(clippy::too_many_arguments)]
pub fn run_co2_ve<'py>(
    py: Python<'py>,
    grid: PyRef<'_, PyVeGrid>,
    wells: &Bound<'_, PyList>,
    times: Vec<f64>,
    s_wr: f64,
    s_gr: f64,
    p_fluid: f64,
    t_fluid: f64,
    m_nacl: f64,
    fluid: Option<std::collections::BTreeMap<String, f64>>,
    gravity: f64,
    open_boundary: bool,
    p_ref: Option<f64>,
    z_ref: Option<f64>,
    cfl: f64,
    dt_max: f64,
    trap_tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let mut f = VeFluid::from_co2_brine(p_fluid, t_fluid, m_nacl, s_wr, s_gr).map_err(to_py_err)?;
    if let Some(o) = fluid {
        for (k, v) in o {
            match k.as_str() {
                "rho_co2" => f.rho_co2 = v,
                "rho_brine" => f.rho_brine = v,
                "mu_co2" => f.mu_co2 = v,
                "mu_brine" => f.mu_brine = v,
                _ => return Err(PyValueError::new_err(format!("fluid: unknown key {k:?}"))),
            }
        }
    }
    let mut ws = Vec::new();
    for w in wells.iter() {
        let d = w.cast::<PyDict>().map_err(|_| PyValueError::new_err("each well must be a dict"))?;
        let get = |k: &str| -> PyResult<Option<f64>> { d.get_item(k)?.map(|v| v.extract::<f64>()).transpose() };
        let cell: usize = d.get_item("cell")?.ok_or_else(|| PyValueError::new_err("well needs 'cell'"))?.extract()?;
        ws.push(VeWell {
            cell,
            rate: get("rate")?.ok_or_else(|| PyValueError::new_err("well needs 'rate'"))?,
            t_start: get("t_start")?.unwrap_or(0.0),
            t_end: get("t_end")?.unwrap_or(f64::INFINITY),
        });
    }
    let g = &grid.g;
    let z_ref = z_ref.unwrap_or_else(|| g.top.iter().sum::<f64>() / g.top.len() as f64);
    let opts = VeOptions { gravity, open_boundary, p_ref: p_ref.unwrap_or(p_fluid), z_ref, cfl, dt_max, trap_tol };
    let r = py.detach(|| ve::run_co2_ve(g, &f, &ws, &times, &opts)).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("times", r.times)?;
    d.set_item("h", rows2(py, &r.h)?)?;
    d.set_item("h_max", rows2(py, &r.h_max)?)?;
    d.set_item("p", rows2(py, &r.p)?)?;
    d.set_item("volume", rows2(py, &r.volume)?)?;
    d.set_item("inventory", rows2(py, &r.inventory.iter().map(|x| x.to_vec()).collect::<Vec<_>>())?)?;
    d.set_item("inventory_names", ve::INVENTORY.to_vec())?;
    d.set_item("trap", PyArray1::from_iter(py, r.traps.trap.iter().map(|x| x.map_or(-1, |v| v as i64))))?;
    d.set_item("spill_depth", r.traps.spill_depth)?;
    d.set_item("n_steps", r.n_steps)?;
    let fd = PyDict::new(py);
    for (k, v) in [("rho_co2", f.rho_co2), ("rho_brine", f.rho_brine), ("mu_co2", f.mu_co2), ("mu_brine", f.mu_brine), ("s_wr", f.s_wr), ("s_gr", f.s_gr)] {
        fd.set_item(k, v)?;
    }
    d.set_item("fluid", fd)?;
    Ok(d)
}
