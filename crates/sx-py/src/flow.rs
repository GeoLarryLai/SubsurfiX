// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for `sx_core::flow`: two-point flux approximation (TPFA)
//! pressure, Peaceman wells, tracer and two-phase IMPES. SI units.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use sx_core::flow::{
    self, Capillary, Corey, FixedPressure, FlowModel, FlowWell, ImpesOptions, Physics,
    TwoPhaseOptions, WellControl, WellKind,
};
use sx_core::grid::RegularGrid;

use crate::errors::to_py_err;
use crate::grid::PyCornerPointGrid;

fn perm_of(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(PyValueError::new_err(
            "perm must have shape (n_cells, 3): kx, ky, kz in m²",
        ));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// Factor to m³/s for a rate unit.
fn rate_factor(unit: &str) -> PyResult<f64> {
    Ok(match unit.trim().to_ascii_lowercase().replace('³', "3").as_str() {
        "m3/s" => 1.0,
        "m3/d" | "m3/day" => 1.0 / 86_400.0,
        "l/s" => 1e-3,
        "bbl/d" | "bbl/day" => 0.158_987_294_928 / 86_400.0,
        o => {
            return Err(PyValueError::new_err(format!(
                "rate unit {o:?} not one of m3/s, m3/d, L/s, bbl/d"
            )))
        }
    })
}

/// Factor to Pa for a pressure unit.
fn pressure_factor(unit: &str) -> PyResult<f64> {
    Ok(match unit.trim().to_ascii_lowercase().as_str() {
        "pa" => 1.0,
        "kpa" => 1e3,
        "mpa" => 1e6,
        "bar" => 1e5,
        "psi" | "psia" => 6_894.757_293_168_361,
        o => {
            return Err(PyValueError::new_err(format!(
                "pressure unit {o:?} not one of Pa, kPa, MPa, bar, psi (gauge pressures: convert first)"
            )))
        }
    })
}

/// A fixed-pressure boundary: `cell` held at `pressure` through the boundary
/// transmissibility `trans` (m³, e.g. `FlowModel.boundary_trans`).
/// `pressure_unit` is Pa (default), kPa, MPa, bar or psi; stored in Pa.
#[pyclass(name = "FixedPressure", module = "subsurfix._native", frozen)]
#[derive(Clone, Copy)]
pub struct PyFixedPressure {
    inner: FixedPressure,
}

#[pymethods]
impl PyFixedPressure {
    #[new]
    #[pyo3(signature = (cell, trans, pressure, pressure_unit="Pa"))]
    fn new(cell: usize, trans: f64, pressure: f64, pressure_unit: &str) -> PyResult<Self> {
        let inner = FixedPressure {
            cell,
            trans,
            pressure: pressure * pressure_factor(pressure_unit)?,
        };
        flow::check_controls(&[], &[inner]).map_err(to_py_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn cell(&self) -> usize {
        self.inner.cell
    }
    #[getter]
    fn trans(&self) -> f64 {
        self.inner.trans
    }
    /// Pressure in Pa.
    #[getter]
    fn pressure(&self) -> f64 {
        self.inner.pressure
    }
    fn __repr__(&self) -> String {
        format!(
            "FixedPressure(cell={}, trans={:e}, pressure={:e})",
            self.inner.cell, self.inner.trans, self.inner.pressure
        )
    }
}

/// `(a, b, trans)` arrays returned by `FlowModel.connections`.
type Connections<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

fn wells_of(list: &Bound<'_, PyList>) -> PyResult<Vec<FlowWell>> {
    let mut out = Vec::new();
    for item in list.iter() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each well must be a dict"))?;
        let get = |k: &str| d.get_item(k).ok().flatten();
        let name: String = get("name")
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| format!("W{}", out.len() + 1));
        let kind = match get("kind")
            .map(|v| v.extract::<String>())
            .transpose()?
            .as_deref()
        {
            Some("injector") => WellKind::Injector,
            Some("producer") => WellKind::Producer,
            other => {
                return Err(PyValueError::new_err(format!(
                    "well {name}: kind must be 'injector' or 'producer', got {other:?}"
                )))
            }
        };
        for k in d.keys() {
            let k: String = k.extract()?;
            if !matches!(
                k.as_str(),
                "name" | "kind" | "rate" | "rate_unit" | "bhp" | "bhp_unit" | "cells" | "radius"
                    | "skin" | "wi" | "ref_depth" | "bhp_limit" | "crossflow"
            ) {
                return Err(PyValueError::new_err(format!("well {name}: unknown key {k:?}")));
            }
        }
        let unit = |k: &str, default: &str| -> PyResult<String> {
            Ok(get(k).map(|v| v.extract()).transpose()?.unwrap_or_else(|| default.to_string()))
        };
        let control = match (get("rate"), get("bhp")) {
            (Some(r), None) => {
                WellControl::Rate(r.extract::<f64>()? * rate_factor(&unit("rate_unit", "m3/s")?)?)
            }
            (None, Some(p)) => {
                WellControl::Bhp(p.extract::<f64>()? * pressure_factor(&unit("bhp_unit", "Pa")?)?)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "well {name}: give exactly one of 'rate' (m³/s) or 'bhp' (Pa)"
                )))
            }
        };
        let cells: Vec<usize> = get("cells")
            .ok_or_else(|| PyValueError::new_err(format!("well {name}: 'cells' is required")))?
            .extract()?;
        let radius_m: f64 = get("radius")
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.1);
        let skin: f64 = get("skin").map(|v| v.extract()).transpose()?.unwrap_or(0.0);
        let wi: Option<Vec<f64>> = get("wi").map(|v| v.extract()).transpose()?;
        let ref_depth_m: Option<f64> = get("ref_depth").map(|v| v.extract()).transpose()?;
        let bhp_limit: Option<f64> = get("bhp_limit")
            .map(|v| v.extract::<f64>())
            .transpose()?
            .map(|v| -> PyResult<f64> { Ok(v * pressure_factor(&unit("bhp_unit", "Pa")?)?) })
            .transpose()?;
        if bhp_limit.is_some() && !matches!(control, WellControl::Rate(_)) {
            return Err(PyValueError::new_err(format!("well {name}: bhp_limit needs a rate-controlled well")));
        }
        let crossflow: bool = get("crossflow").map(|v| v.extract()).transpose()?.unwrap_or(true);
        out.push(FlowWell {
            name,
            kind,
            control,
            cells,
            radius_m,
            skin,
            wi,
            ref_depth_m,
            bhp_limit,
            crossflow,
        });
    }
    Ok(out)
}

/// Boundaries from a list of `FixedPressure` or dicts with `cell`, `trans`,
/// `pressure` and optional `pressure_unit`.
fn fixed_of(v: Option<&Bound<'_, PyList>>) -> PyResult<Vec<FixedPressure>> {
    let Some(list) = v else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(list.len());
    for (i, item) in list.iter().enumerate() {
        if let Ok(f) = item.cast::<PyFixedPressure>() {
            out.push(f.get().inner);
            continue;
        }
        let Ok(d) = item.cast::<PyDict>() else {
            let hint = match item.cast::<PyTuple>() {
                Ok(t) if t.len() == 3 => {
                    let r = |j: usize| t.get_item(j).and_then(|x| x.repr()).map(|x| x.to_string()).unwrap_or_default();
                    format!(
                        "tuples are no longer accepted; write FixedPressure({}, {}, {}, pressure_unit='Pa')",
                        r(0),
                        r(1),
                        r(2)
                    )
                }
                _ => "use FixedPressure(cell, trans, pressure, pressure_unit='Pa')".to_string(),
            };
            return Err(PyValueError::new_err(format!(
                "fixed[{i}]: {hint} (or a dict with keys cell, trans, pressure, pressure_unit)"
            )));
        };
        let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
            d.get_item(k)?
                .ok_or_else(|| PyValueError::new_err(format!("fixed[{i}]: missing {k:?}")))
        };
        for k in d.keys() {
            let k: String = k.extract()?;
            if !matches!(k.as_str(), "cell" | "trans" | "pressure" | "pressure_unit") {
                return Err(PyValueError::new_err(format!("fixed[{i}]: unknown key {k:?}")));
            }
        }
        let unit: String = match d.get_item("pressure_unit")? {
            Some(u) => u.extract()?,
            None => "Pa".into(),
        };
        let f = PyFixedPressure::new(
            get("cell")?.extract()?,
            get("trans")?.extract()?,
            get("pressure")?.extract()?,
            &unit,
        )?;
        out.push(f.inner);
    }
    Ok(out)
}

fn corey_of(d: Option<&Bound<'_, PyDict>>) -> PyResult<Corey> {
    let mut c = Corey::default();
    if let Some(d) = d {
        for (k, v) in d.iter() {
            let k: String = k.extract()?;
            let v: f64 = v.extract()?;
            match k.as_str() {
                "swc" => c.swc = v,
                "sor" => c.sor = v,
                "nw" => c.nw = v,
                "no" => c.no = v,
                "krw_max" => c.krw_max = v,
                "kro_max" => c.kro_max = v,
                "mu_w" => c.mu_w = v,
                "mu_o" => c.mu_o = v,
                _ => return Err(PyValueError::new_err(format!("unknown Corey key {k:?}"))),
            }
        }
    }
    Ok(c)
}

fn physics_of(d: Option<&Bound<'_, PyDict>>) -> PyResult<Physics> {
    let mut ph = Physics::default();
    let (mut kind, mut pd, mut lambda, mut alpha, mut n) = (None::<String>, 0.0, 2.0, 1e-4, 2.0);
    if let Some(d) = d {
        for (k, v) in d.iter() {
            let k: String = k.extract()?;
            if k == "capillary" {
                kind = v.extract::<Option<String>>()?;
                continue;
            }
            let v: f64 = v.extract()?;
            match k.as_str() {
                "gravity" => ph.gravity = v,
                "rho_w" => ph.rho_w = v,
                "rho_o" => ph.rho_o = v,
                "c_rock" => ph.c_rock = v,
                "c_w" => ph.c_w = v,
                "c_o" => ph.c_o = v,
                "p_ref" => ph.p_ref = v,
                "pd" => pd = v,
                "lambda" => lambda = v,
                "alpha" => alpha = v,
                "n" => n = v,
                _ => return Err(PyValueError::new_err(format!("unknown physics key {k:?}"))),
            }
        }
    }
    ph.capillary = match kind.as_deref() {
        None | Some("none") => Capillary::None,
        Some("brooks_corey") => Capillary::BrooksCorey { pd, lambda },
        Some("van_genuchten") => Capillary::VanGenuchten { alpha, n },
        Some(o) => {
            return Err(PyValueError::new_err(format!(
                "capillary must be 'brooks_corey', 'van_genuchten' or None, got {o:?}"
            )))
        }
    };
    Ok(ph)
}

fn rows2<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray2::from_vec2(py, rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Flow model: cells, pore volumes and TPFA transmissibilities.
#[pyclass(name = "FlowModel", module = "subsurfix._native")]
pub struct PyFlowModel {
    m: FlowModel,
    corners: Vec<[[f64; 3]; 8]>,
}

#[pymethods]
impl PyFlowModel {
    /// Build on a regular grid; `perm` (n_cells, 3) in m², `poro` (n_cells,).
    #[staticmethod]
    fn from_regular(
        origin: [f64; 3],
        cell: [f64; 3],
        dims: [usize; 3],
        perm: PyReadonlyArray2<'_, f64>,
        poro: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let g = RegularGrid::new(origin, cell, dims).map_err(to_py_err)?;
        let m =
            FlowModel::from_regular(&g, &perm_of(&perm)?, poro.as_slice()?).map_err(to_py_err)?;
        Ok(Self { m, corners: flow::ogs::regular_corners(&g) })
    }

    /// Build on a corner-point grid. `ntg` (n_cells,) net-to-gross; `mult` a
    /// dict of GRDECL multipliers (MULTX, MULTY, MULTZ, MULTX-, MULTY-,
    /// MULTZ-) to (n_cells,) arrays. Faults get overlap (non-neighbour)
    /// connections and pinch-outs vertical connections; `fault_tol_m` is the
    /// throw / thickness tolerance.
    #[staticmethod]
    #[pyo3(signature = (grid, perm, poro, ntg=None, mult=None, fault_tol_m=1e-3))]
    fn from_corner_point(
        grid: PyRef<'_, PyCornerPointGrid>,
        perm: PyReadonlyArray2<'_, f64>,
        poro: PyReadonlyArray1<'_, f64>,
        ntg: Option<PyReadonlyArray1<'_, f64>>,
        mult: Option<std::collections::BTreeMap<String, Vec<f64>>>,
        fault_tol_m: f64,
    ) -> PyResult<Self> {
        let opts = sx_core::flow::FlowOptions {
            ntg: ntg.map(|a| a.as_slice().map(<[f64]>::to_vec)).transpose()?,
            mult: mult.unwrap_or_default().into_iter().map(|(k, v)| (k.to_ascii_uppercase(), v)).collect(),
            fault_tol_m,
        };
        let m = FlowModel::from_corner_point_with(&grid.g, &perm_of(&perm)?, poro.as_slice()?, &opts)
            .map_err(to_py_err)?;
        let corners = (0..grid.g.n_cells()).map(|c| grid.g.corners(c)).collect();
        Ok(Self { m, corners })
    }

    /// Write the active cells to an OpenGeoSys (OGS) mesh (`.vtu`, ASCII):
    /// hexahedra with merged shared nodes, z up (`z = -depth`), cell arrays
    /// `MaterialIDs` (`material_ids`, default 0), `bulk_cell_id`,
    /// `permeability` (m², 3 components), `porosity`, plus `cell_data`
    /// (name to an (n_cells,) or (n_cells, k) array). Nodes closer than
    /// `merge_tol` (m) merge. Returns {"n_nodes", "n_cells"}.
    #[pyo3(signature = (path, cell_data=None, material_ids=None, merge_tol=1e-6))]
    fn export_ogs(
        &self,
        path: std::path::PathBuf,
        cell_data: Option<std::collections::BTreeMap<String, PyReadonlyArrayDyn<'_, f64>>>,
        material_ids: Option<Vec<i32>>,
        merge_tol: f64,
    ) -> PyResult<std::collections::BTreeMap<String, usize>> {
        let n = self.m.n_cells();
        let perm: Vec<f64> = self.m.perm.iter().flatten().copied().collect();
        let poro: Vec<f64> = (0..n)
            .map(|c| if self.m.bulk_volume[c] > 0.0 { self.m.pore_volume[c] / self.m.bulk_volume[c] } else { 0.0 })
            .collect();
        let mut owned: Vec<(String, Vec<f64>, usize)> = Vec::new();
        for (k, a) in cell_data.unwrap_or_default() {
            let shape = a.shape().to_vec();
            let comps = match shape.as_slice() {
                [m] if *m == n => 1,
                [m, k] if *m == n && *k > 0 => *k,
                _ => return Err(PyValueError::new_err(format!("cell_data[{k:?}]: need shape (n_cells,) or (n_cells, k)"))),
            };
            owned.push((k, a.as_array().iter().copied().collect(), comps));
        }
        let mut arrays = vec![
            flow::ogs::CellArray { name: "permeability", values: &perm, components: 3 },
            flow::ogs::CellArray { name: "porosity", values: &poro, components: 1 },
        ];
        arrays.extend(owned.iter().map(|(k, v, c)| flow::ogs::CellArray { name: k, values: v, components: *c }));
        let (text, info) =
            flow::ogs::ogs_vtu(&self.corners, &self.m.active, material_ids.as_deref(), &arrays, merge_tol)
                .map_err(to_py_err)?;
        std::fs::write(&path, text).map_err(|e| PyValueError::new_err(format!("{}: {e}", path.display())))?;
        Ok([("n_nodes".to_string(), info.n_nodes), ("n_cells".to_string(), info.n_cells)].into())
    }

    /// Warnings raised while building.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.m.warnings.clone()
    }

    /// Number of connections across fault throws.
    #[getter]
    fn n_fault_connections(&self) -> usize {
        self.m.n_fault_connections
    }

    /// Number of connections across pinched-out cells.
    #[getter]
    fn n_pinch_connections(&self) -> usize {
        self.m.n_pinch_connections
    }

    /// Number of cells.
    #[getter]
    fn n_cells(&self) -> usize {
        self.m.n_cells()
    }

    /// Pore volume per cell (m³).
    fn pore_volume<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.m.pore_volume)
    }

    /// Connections as (a, b, trans) arrays; trans in m³.
    fn connections<'py>(&self, py: Python<'py>) -> Connections<'py> {
        let c = &self.m.connections;
        (
            PyArray1::from_iter(py, c.iter().map(|x| x.a as i64)),
            PyArray1::from_iter(py, c.iter().map(|x| x.b as i64)),
            PyArray1::from_iter(py, c.iter().map(|x| x.trans)),
        )
    }

    /// Half transmissibility from a cell to its boundary face (axis 0, 1, 2).
    fn boundary_trans(&self, cell: usize, axis: usize) -> PyResult<f64> {
        if cell >= self.m.n_cells() || axis > 2 {
            return Err(PyValueError::new_err("cell or axis out of range"));
        }
        Ok(self.m.boundary_trans(cell, axis))
    }

    /// Incompressible pressure solve. `wells`: list of dicts (name, kind
    /// 'injector'|'producer', rate (m³/s) or bhp (Pa), cells, radius=0.1,
    /// skin=0, optional rate_unit m3/s|m3/d|L/s|bbl/d and bhp_unit
    /// Pa|kPa|MPa|bar|psi). Rates are magnitudes >= 0; the kind sets the
    /// direction. `fixed`: list of `FixedPressure` (or dicts with its fields). Returns a dict with
    /// p, flux, boundary_flux, upwind_iterations, warnings and wells (bhp, rate, cell_rates, wi).
    /// `boundary_mobility`: mobility of fluid entering through `fixed` boundaries
    /// (default: the cell mobility).
    #[pyo3(signature = (mobility, wells, fixed=None, inj_mobility=None, boundary_mobility=None))]
    fn solve_pressure<'py>(
        &self,
        py: Python<'py>,
        mobility: PyReadonlyArray1<'_, f64>,
        wells: &Bound<'py, PyList>,
        fixed: Option<&Bound<'_, PyList>>,
        inj_mobility: Option<f64>,
        boundary_mobility: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mob = mobility.as_slice()?;
        let inj = inj_mobility.unwrap_or_else(|| mob.first().copied().unwrap_or(1.0));
        let ws = wells_of(wells)?;
        let o = flow::PressureOptions { boundary_inflow_mobility: boundary_mobility, ..Default::default() };
        let sol = flow::solve_pressure_with(&self.m, mob, inj, &ws, &fixed_of(fixed)?, &o).map_err(to_py_err)?;
        let d = PyDict::new(py);
        d.set_item("p", PyArray1::from_slice(py, &sol.p))?;
        d.set_item("flux", PyArray1::from_slice(py, &sol.flux))?;
        d.set_item(
            "boundary_flux",
            PyArray1::from_slice(py, &sol.boundary_flux),
        )?;
        d.set_item("upwind_iterations", sol.upwind_iterations)?;
        d.set_item("warnings", sol.warnings.clone())?;
        let wl = PyList::empty(py);
        for (w, r) in ws.iter().zip(&sol.wells) {
            let x = PyDict::new(py);
            x.set_item("name", &w.name)?;
            x.set_item("bhp", r.bhp)?;
            x.set_item("rate", r.rate)?;
            x.set_item("cell_rates", r.cell_rates.clone())?;
            x.set_item("wi", r.wi.clone())?;
            x.set_item("bhp_limited", r.bhp_limited)?;
            wl.append(x)?;
        }
        d.set_item("wells", wl)?;
        Ok(d)
    }

    /// Half-cell geometric factor (m) to the boundary face on `axis`;
    /// times a conductivity gives a boundary conductance (W/K).
    fn boundary_geom(&self, cell: usize, axis: usize) -> PyResult<f64> {
        if cell >= self.m.n_cells() || axis > 2 {
            return Err(PyValueError::new_err("cell out of range or axis not 0, 1, 2"));
        }
        Ok(self.m.boundary_geom(cell, axis))
    }

    /// Heat transport after a single-phase pressure solve with `mobility`
    /// (constant properties, liquid only, implicit upwind advection plus
    /// harmonic-mean conduction). `well_temp`: injection temperature per
    /// well; `fixed_temp`: inflow temperature per fixed-pressure boundary;
    /// `fixed_t`: list of dicts {cell, conductance (W/K), temperature};
    /// `heat_flux`: list of dicts {cell, power} (W into the cell, a
    /// prescribed-flux boundary such as a basal geothermal flux times area);
    /// `props`: dict overriding rho_fluid, cp_fluid, rho_rock, cp_rock,
    /// cond_fluid, cond_rock (SI); `water_p` (Pa) with `water_t_k` (K) sets
    /// the fluid properties from IAPWS-IF97 first. Returns {"times", "temperature"
    /// (n_times, n_cells), "energy_change", "energy_in" (J), "n_steps",
    /// "n_rejected", "p", "flux"}. `dt_change` turns on adaptive stepping:
    /// the step (starting at `dt_init`, default `dt_max`) is scaled so the
    /// largest temperature change per step is about `dt_change`.
    #[pyo3(signature = (mobility, wells, well_temp, t0, times, fixed=None, fixed_temp=None, fixed_t=None, props=None, dt_max=86400.0, inj_mobility=None, dt_init=None, dt_change=None, heat_flux=None))]
    #[allow(clippy::too_many_arguments)]
    fn run_heat<'py>(
        &self,
        py: Python<'py>,
        mobility: PyReadonlyArray1<'_, f64>,
        wells: &Bound<'py, PyList>,
        well_temp: Vec<f64>,
        t0: PyReadonlyArray1<'_, f64>,
        times: Vec<f64>,
        fixed: Option<&Bound<'_, PyList>>,
        fixed_temp: Option<Vec<f64>>,
        fixed_t: Option<Vec<std::collections::BTreeMap<String, f64>>>,
        props: Option<std::collections::BTreeMap<String, f64>>,
        dt_max: f64,
        inj_mobility: Option<f64>,
        dt_init: Option<f64>,
        dt_change: Option<f64>,
        heat_flux: Option<Vec<std::collections::BTreeMap<String, f64>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mob = mobility.as_slice()?;
        let inj = inj_mobility.unwrap_or_else(|| mob.first().copied().unwrap_or(1.0));
        let ws = wells_of(wells)?;
        let fx = fixed_of(fixed)?;
        let sol = flow::solve_pressure(&self.m, mob, inj, &ws, &fx).map_err(to_py_err)?;
        let mut tp = flow::ThermalProps::default();
        let mut props = props.unwrap_or_default();
        match (props.remove("water_p"), props.remove("water_t_k")) {
            (Some(p), Some(t)) => tp = tp.with_water(p, t).map_err(to_py_err)?,
            (None, None) => {}
            _ => return Err(PyValueError::new_err("props: water_p and water_t_k go together")),
        }
        for (k, v) in props {
            let slot = match k.as_str() {
                "rho_fluid" => &mut tp.rho_fluid,
                "cp_fluid" => &mut tp.cp_fluid,
                "rho_rock" => &mut tp.rho_rock,
                "cp_rock" => &mut tp.cp_rock,
                "cond_fluid" => &mut tp.cond_fluid,
                "cond_rock" => &mut tp.cond_rock,
                _ => return Err(PyValueError::new_err(format!("props: unknown key {k:?}"))),
            };
            *slot = v;
        }
        let mut ft = Vec::new();
        for (i, d) in fixed_t.unwrap_or_default().into_iter().enumerate() {
            let get = |k: &str| {
                d.get(k).copied().ok_or_else(|| {
                    PyValueError::new_err(format!("fixed_t[{i}]: needs keys cell, conductance, temperature"))
                })
            };
            let cell = get("cell")?;
            if !(cell >= 0.0 && cell.fract() == 0.0) {
                return Err(PyValueError::new_err(format!("fixed_t[{i}]: cell must be a non-negative integer")));
            }
            ft.push(flow::FixedTemperature {
                cell: cell as usize,
                conductance: get("conductance")?,
                temperature: get("temperature")?,
            });
        }
        let mut hf = Vec::new();
        for (i, d) in heat_flux.unwrap_or_default().into_iter().enumerate() {
            let (Some(&cell), Some(&power)) = (d.get("cell"), d.get("power")) else {
                return Err(PyValueError::new_err(format!("heat_flux[{i}]: needs keys cell, power")));
            };
            if !(cell >= 0.0 && cell.fract() == 0.0) {
                return Err(PyValueError::new_err(format!("heat_flux[{i}]: cell must be a non-negative integer")));
            }
            hf.push(flow::HeatFlux { cell: cell as usize, power });
        }
        let inp = flow::HeatInputs {
            well_temp,
            fixed_temp: fixed_temp.unwrap_or_else(|| vec![0.0; fx.len()]),
            fixed_t: ft,
            heat_flux: hf,
            t0: t0.as_slice()?.to_vec(),
            times,
            dt_max,
            control: dt_change.map(|ch| flow::StepControl::new(dt_init.unwrap_or(dt_max), ch)),
        };
        let r = flow::run_heat(&self.m, &sol, &ws, &fx, &tp, &inp).map_err(to_py_err)?;
        let d = PyDict::new(py);
        d.set_item("times", r.times.clone())?;
        d.set_item("temperature", rows2(py, &r.temperature)?)?;
        d.set_item("energy_change", r.energy_change.clone())?;
        d.set_item("energy_in", r.energy_in.clone())?;
        d.set_item("n_steps", r.n_steps)?;
        d.set_item("n_rejected", r.n_rejected)?;
        d.set_item("p", PyArray1::from_slice(py, &sol.p))?;
        d.set_item("flux", PyArray1::from_slice(py, &sol.flux))?;
        Ok(d)
    }

    /// Single-phase tracer: pressure solve with `mobility`, then explicit
    /// upwind transport. Returns {"times", "conc" (n_times, n_cells)}.
    #[pyo3(signature = (mobility, wells, well_conc, c0, times, fixed=None, fixed_conc=None, cfl=0.9))]
    #[allow(clippy::too_many_arguments)]
    fn run_tracer<'py>(
        &self,
        py: Python<'py>,
        mobility: PyReadonlyArray1<'_, f64>,
        wells: &Bound<'py, PyList>,
        well_conc: Vec<f64>,
        c0: PyReadonlyArray1<'_, f64>,
        times: Vec<f64>,
        fixed: Option<&Bound<'_, PyList>>,
        fixed_conc: Option<Vec<f64>>,
        cfl: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mob = mobility.as_slice()?;
        let ws = wells_of(wells)?;
        let fx = fixed_of(fixed)?;
        let fc = fixed_conc.unwrap_or_else(|| vec![0.0; fx.len()]);
        let sol = flow::solve_pressure(&self.m, mob, mob.first().copied().unwrap_or(1.0), &ws, &fx)
            .map_err(to_py_err)?;
        let r = flow::run_tracer(
            &self.m,
            &sol,
            &ws,
            &well_conc,
            &fx,
            &fc,
            c0.as_slice()?,
            &times,
            cfl,
        )
        .map_err(to_py_err)?;
        let d = PyDict::new(py);
        d.set_item("times", r.times)?;
        d.set_item("conc", rows2(py, &r.conc)?)?;
        d.set_item("n_steps", r.n_steps)?;
        Ok(d)
    }

    /// Two-phase water/oil IMPES with Corey relative permeabilities
    /// (`corey` dict: swc, sor, nw, no, krw_max, kro_max, mu_w, mu_o).
    /// Returns {"times", "sw", "p" (n_times, n_cells), "water_rate",
    /// "oil_rate" (n_times, n_wells), "n_pressure", "n_steps"}.
    #[pyo3(signature = (wells, sw0, times, fixed=None, corey=None, cfl=0.5, pressure_dt=f64::INFINITY, boundary_fw=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn run_impes<'py>(
        &self,
        py: Python<'py>,
        wells: &Bound<'py, PyList>,
        sw0: PyReadonlyArray1<'_, f64>,
        times: Vec<f64>,
        fixed: Option<&Bound<'_, PyList>>,
        corey: Option<&Bound<'py, PyDict>>,
        cfl: f64,
        pressure_dt: f64,
        boundary_fw: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ws = wells_of(wells)?;
        let c = corey_of(corey)?;
        let r = flow::run_impes(
            &self.m,
            &ws,
            &fixed_of(fixed)?,
            &c,
            sw0.as_slice()?,
            &times,
            &ImpesOptions {
                cfl,
                pressure_dt,
                boundary_fw,
            },
        )
        .map_err(to_py_err)?;
        let d = PyDict::new(py);
        d.set_item("times", r.times)?;
        d.set_item("sw", rows2(py, &r.sw)?)?;
        d.set_item("p", rows2(py, &r.p)?)?;
        d.set_item("water_rate", rows2(py, &r.water_rate)?)?;
        d.set_item("oil_rate", rows2(py, &r.oil_rate)?)?;
        d.set_item("n_pressure", r.n_pressure)?;
        d.set_item("n_steps", r.n_steps)?;
        Ok(d)
    }

    /// Cell centres (n_cells, 3): x, y, depth (m).
    fn centers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows: Vec<Vec<f64>> = self.m.center.iter().map(|c| c.to_vec()).collect();
        rows2(py, &rows)
    }

    /// Active flag per cell.
    fn active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.m.active)
    }

    /// Deviated-well completions along a polyline `path` (n_points, 3) of
    /// x, y, depth (m): {"cells", "wi" (m³), "length" (n, 3) projected
    /// lengths}. Pass "cells" and "wi" in a well dict.
    #[pyo3(signature = (path, radius=0.1, skin=0.0))]
    fn trajectory_completions<'py>(
        &self,
        py: Python<'py>,
        path: PyReadonlyArray2<'_, f64>,
        radius: f64,
        skin: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pts = perm_of(&path)?;
        let comp = self
            .m
            .trajectory_completions(&pts, radius, skin)
            .map_err(to_py_err)?;
        let d = PyDict::new(py);
        d.set_item("cells", comp.iter().map(|c| c.cell).collect::<Vec<_>>())?;
        d.set_item("wi", comp.iter().map(|c| c.wi).collect::<Vec<_>>())?;
        d.set_item(
            "length",
            rows2(
                py,
                &comp.iter().map(|c| c.length.to_vec()).collect::<Vec<_>>(),
            )?,
        )?;
        Ok(d)
    }

    /// Two-phase IMPES with gravity, capillary pressure and compressibility.
    /// `physics` dict: gravity (m/s², 0 = off), rho_w, rho_o (kg/m³), c_rock,
    /// c_w, c_o (1/Pa), p_ref (Pa), capillary ('brooks_corey' with pd, lambda
    /// | 'van_genuchten' with alpha (1/Pa), n | None). Wells may carry "wi"
    /// and "ref_depth" (m). Returns {"times", "sw", "p" (n_times, n_cells),
    /// "bhp", "water_rate", "oil_rate" (n_times, n_wells), "water_in_place",
    /// "water_net_in", "water_clamp_loss", "oil_in_place", "oil_net_in", "n_steps"}.
    #[pyo3(signature = (wells, sw0, p0, times, fixed=None, corey=None, physics=None, cfl=0.5, max_dt=f64::INFINITY, boundary_fw=1.0, ds_target=f64::INFINITY, dp_target=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_two_phase<'py>(
        &self,
        py: Python<'py>,
        wells: &Bound<'py, PyList>,
        sw0: PyReadonlyArray1<'_, f64>,
        p0: PyReadonlyArray1<'_, f64>,
        times: Vec<f64>,
        fixed: Option<&Bound<'_, PyList>>,
        corey: Option<&Bound<'py, PyDict>>,
        physics: Option<&Bound<'py, PyDict>>,
        cfl: f64,
        max_dt: f64,
        boundary_fw: f64,
        ds_target: f64,
        dp_target: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ws = wells_of(wells)?;
        let c = corey_of(corey)?;
        let ph = physics_of(physics)?;
        let (sw0, p0) = (sw0.as_slice()?.to_vec(), p0.as_slice()?.to_vec());
        let fx = fixed_of(fixed)?;
        let m = &self.m;
        let r = py
            .detach(|| {
                flow::run_two_phase(
                    m,
                    &ws,
                    &fx,
                    &c,
                    &ph,
                    &sw0,
                    &p0,
                    &times,
                    &TwoPhaseOptions {
                        cfl,
                        max_dt,
                        boundary_fw,
                        ds_target,
                        dp_target,
                    },
                )
            })
            .map_err(to_py_err)?;
        let d = PyDict::new(py);
        d.set_item("times", r.times)?;
        d.set_item("sw", rows2(py, &r.sw)?)?;
        d.set_item("p", rows2(py, &r.p)?)?;
        d.set_item("bhp", rows2(py, &r.bhp)?)?;
        d.set_item("bhp_limited", r.bhp_limited)?;
        d.set_item("water_rate", rows2(py, &r.water_rate)?)?;
        d.set_item("oil_rate", rows2(py, &r.oil_rate)?)?;
        d.set_item("water_in_place", r.water_in_place)?;
        d.set_item("water_net_in", r.water_net_in)?;
        d.set_item("water_clamp_loss", r.water_clamp_loss)?;
        d.set_item("oil_in_place", r.oil_in_place)?;
        d.set_item("oil_net_in", r.oil_net_in)?;
        d.set_item("n_steps", r.n_steps)?;
        d.set_item("n_rejected", r.n_rejected)?;
        Ok(d)
    }
}

/// Liquid water properties from IAPWS-IF97 region 1 (density kg/m³,
/// enthalpy J/kg, cp J/(kg K), compressibility 1/Pa), IAPWS 2008 viscosity
/// (Pa s) and IAPWS 2011 conductivity (W/(m K)) at pressures `p_pa` (Pa)
/// and temperatures `t_k` (K) of equal length. Returns a dict of arrays.
#[pyfunction]
pub fn water_properties<'py>(
    py: Python<'py>,
    p_pa: PyReadonlyArray1<'_, f64>,
    t_k: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (p, t) = (p_pa.as_slice()?, t_k.as_slice()?);
    if p.len() != t.len() {
        return Err(PyValueError::new_err("water_properties: p_pa and t_k need the same length"));
    }
    let ws: Vec<flow::water::LiquidWater> = p
        .iter()
        .zip(t)
        .map(|(&p, &t)| flow::water::liquid(p, t))
        .collect::<Result<_, _>>()
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    let col = |f: fn(&flow::water::LiquidWater) -> f64| ws.iter().map(f).collect::<Vec<f64>>();
    d.set_item("density", PyArray1::from_vec(py, col(|w| w.density)))?;
    d.set_item("enthalpy", PyArray1::from_vec(py, col(|w| w.enthalpy)))?;
    d.set_item("cp", PyArray1::from_vec(py, col(|w| w.cp)))?;
    d.set_item("compressibility", PyArray1::from_vec(py, col(|w| w.compressibility)))?;
    d.set_item("viscosity", PyArray1::from_vec(py, col(|w| w.viscosity)))?;
    d.set_item("conductivity", PyArray1::from_vec(py, col(|w| w.conductivity)))?;
    Ok(d)
}

/// Capillary pressure (Pa) at water saturation `sw` (Corey end points and
/// `physics` capillary keys as in `FlowModel.run_two_phase`).
#[pyfunction]
#[pyo3(signature = (sw, corey=None, physics=None))]
pub fn capillary_pressure(
    sw: f64,
    corey: Option<&Bound<'_, PyDict>>,
    physics: Option<&Bound<'_, PyDict>>,
) -> PyResult<f64> {
    Ok(physics_of(physics)?.capillary.pc(&corey_of(corey)?, sw))
}

/// Theis / line-source drawdown (Pa) at radius `r` (m), time `t` (s).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn theis_drawdown(q: f64, mu: f64, k: f64, h: f64, phi: f64, ct: f64, r: f64, t: f64) -> f64 {
    flow::analytic::theis_drawdown(q, mu, k, h, phi, ct, r, t)
}

/// Peaceman well index (m³) of a vertical well.
#[pyfunction]
#[pyo3(signature = (kx, ky, dx, dy, dz, rw, skin=0.0))]
pub fn peaceman_wi(kx: f64, ky: f64, dx: f64, dy: f64, dz: f64, rw: f64, skin: f64) -> f64 {
    flow::peaceman_wi(kx, ky, dx, dy, dz, rw, skin)
}

/// Analytic Buckley-Leverett water saturation at x/L = `xd`, injected pore
/// volumes `td` (Corey dict as in `FlowModel.run_impes`).
#[pyfunction]
#[pyo3(signature = (xd, td, corey=None))]
pub fn buckley_leverett(xd: f64, td: f64, corey: Option<&Bound<'_, PyDict>>) -> PyResult<f64> {
    Ok(flow::analytic::buckley_leverett(&corey_of(corey)?, xd, td))
}
