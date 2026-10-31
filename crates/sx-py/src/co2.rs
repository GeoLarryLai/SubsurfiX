// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for the experimental CO2-brine fluid properties.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sx_core::flow::co2_brine::{self, SolubilityModel};

use crate::errors::to_py_err;

pub(crate) fn model_of(name: &str) -> PyResult<SolubilityModel> {
    match name {
        "spycher_pruess_2005" => Ok(SolubilityModel::SpycherPruess2005),
        "ziabakhsh_ganji_kooi_2012" => Ok(SolubilityModel::ZiabakhshGanjiKooi2012),
        _ => Err(PyValueError::new_err(format!(
            "unknown solubility model {name:?}"
        ))),
    }
}

/// Mutual CO2-H2O solubility in NaCl brine (EXPERIMENTAL; Spycher et al.
/// 2003, Spycher & Pruess 2005). Arrays of equal length; returns `x_co2`,
/// `m_co2`, `y_h2o`.
#[pyfunction]
#[pyo3(signature = (p_pa, t_k, m_nacl, model="spycher_pruess_2005"))]
pub fn co2_solubility<'py>(
    py: Python<'py>,
    p_pa: PyReadonlyArray1<'_, f64>,
    t_k: PyReadonlyArray1<'_, f64>,
    m_nacl: PyReadonlyArray1<'_, f64>,
    model: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (p, t, m) = (p_pa.as_slice()?, t_k.as_slice()?, m_nacl.as_slice()?);
    if p.len() != t.len() || p.len() != m.len() {
        return Err(PyValueError::new_err(
            "co2_solubility: p_pa, t_k and m_nacl need the same length",
        ));
    }
    let md = model_of(model)?;
    let s: Vec<_> = (0..p.len())
        .map(|i| co2_brine::mutual_solubility(p[i], t[i], m[i], md))
        .collect::<Result<_, _>>()
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item(
        "x_co2",
        PyArray1::from_vec(py, s.iter().map(|v| v.x_co2).collect()),
    )?;
    d.set_item(
        "m_co2",
        PyArray1::from_vec(py, s.iter().map(|v| v.m_co2).collect()),
    )?;
    d.set_item(
        "y_h2o",
        PyArray1::from_vec(py, s.iter().map(|v| v.y_h2o).collect()),
    )?;
    Ok(d)
}

/// Two-phase CO2-brine split (EXPERIMENTAL) at overall CO2 mole fraction
/// `z_co2` (salt-free basis). Returns phase fractions, compositions,
/// densities (kg/m³), viscosities (Pa s), enthalpies (J/kg), gas saturation.
#[pyfunction]
#[pyo3(signature = (p_pa, t_k, z_co2, m_nacl, model="spycher_pruess_2005"))]
pub fn co2_brine_flash<'py>(
    py: Python<'py>,
    p_pa: f64,
    t_k: f64,
    z_co2: f64,
    m_nacl: f64,
    model: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let f = co2_brine::flash(p_pa, t_k, z_co2, m_nacl, model_of(model)?).map_err(to_py_err)?;
    let d = PyDict::new(py);
    for (k, v) in [
        ("nu_gas_raw", f.nu_gas_raw),
        ("nu_gas", f.nu_gas),
        ("x_co2", f.x_co2),
        ("y_h2o", f.y_h2o),
        ("rho_aq", f.rho_aq),
        ("rho_gas", f.rho_gas),
        ("mu_aq", f.mu_aq),
        ("mu_gas", f.mu_gas),
        ("h_aq", f.h_aq),
        ("h_gas", f.h_gas),
        ("s_gas", f.s_gas),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}
