// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for `sx_core::sim`: sequential Gaussian simulation (SGS),
//! sequential indicator simulation (SIS), normal scores and the
//! `ml_propagation`.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sx_core::sim::{self, PropagationModel, PropagationParams, QrfParams, SgsParams, SisParams};

use crate::errors::to_py_err;
use crate::grid::{model_of, points_of};

fn to2<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    rows: &[Vec<T>],
) -> PyResult<Bound<'py, PyArray2<T>>> {
    PyArray2::from_vec2(py, rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn rows_of(a: &PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    a.as_array()
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect()
}

/// SGS (normal-score, ordinary-kriging based). `kind/nugget/psill/range`:
/// variogram of the normal scores. Returns shape (n_realizations, n_targets).
#[pyfunction]
#[pyo3(signature = (points, values, targets, kind, nugget, psill, range, n_realizations=10, max_neighbors=16, seed=0, max_distance=f64::INFINITY))]
#[allow(clippy::too_many_arguments)]
pub fn sgs<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    kind: &str,
    nugget: f64,
    psill: f64,
    range: f64,
    n_realizations: usize,
    max_neighbors: usize,
    seed: u64,
    max_distance: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let model = model_of(kind, nugget, psill, range)?;
    let (p, v, t) = (
        points_of(&points)?,
        values.as_array().to_vec(),
        points_of(&targets)?,
    );
    let prm = SgsParams {
        model,
        max_neighbors,
        max_distance,
        n_realizations,
        seed,
    };
    let r = py
        .detach(|| sim::sgs(&p, &v, &t, &prm))
        .map_err(to_py_err)?;
    to2(py, &r)
}

/// SIS of facies codes with one shared indicator variogram. Returns int32
/// shape (n_realizations, n_targets).
#[pyfunction]
#[pyo3(signature = (points, codes, targets, kind, nugget, psill, range, n_realizations=10, max_neighbors=16, seed=0, max_distance=f64::INFINITY))]
#[allow(clippy::too_many_arguments)]
pub fn sis<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    codes: PyReadonlyArray1<'py, i32>,
    targets: PyReadonlyArray2<'py, f64>,
    kind: &str,
    nugget: f64,
    psill: f64,
    range: f64,
    n_realizations: usize,
    max_neighbors: usize,
    seed: u64,
    max_distance: f64,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let model = model_of(kind, nugget, psill, range)?;
    let (p, c, t) = (
        points_of(&points)?,
        codes.as_array().to_vec(),
        points_of(&targets)?,
    );
    let prm = SisParams {
        models: vec![],
        default_model: model,
        max_neighbors,
        max_distance,
        n_realizations,
        seed,
    };
    let r = py
        .detach(|| sim::sis(&p, &c, &t, &prm))
        .map_err(to_py_err)?;
    to2(py, &r)
}

/// Normal scores of `values` (rank-based; NaN stays NaN).
#[pyfunction]
pub fn normal_score<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v = values.as_array().to_vec();
    let ns = sim::NormalScore::fit(&v).map_err(to_py_err)?;
    Ok(PyArray1::from_iter(py, v.iter().map(|x| ns.forward(*x))))
}

/// `ml_propagation`: quantile predictions,
/// shape (n_pred, n_quantiles). `model`: "qrf" (default, quantile regression forest) or "lightgbm".
#[pyfunction]
#[pyo3(signature = (train_x, train_y, pred_x, model="qrf", quantiles=vec![0.1, 0.5, 0.9], n_trees=200, min_leaf=10, seed=0))]
#[allow(clippy::too_many_arguments)]
pub fn ml_propagation<'py>(
    py: Python<'py>,
    train_x: PyReadonlyArray2<'py, f64>,
    train_y: PyReadonlyArray1<'py, f64>,
    pred_x: PyReadonlyArray2<'py, f64>,
    model: &str,
    quantiles: Vec<f64>,
    n_trees: usize,
    min_leaf: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let prm = PropagationParams {
        model: PropagationModel::parse(model).map_err(to_py_err)?,
        quantiles,
        qrf: QrfParams {
            n_trees,
            min_leaf,
            ..QrfParams::default()
        },
        seed,
        ..PropagationParams::default()
    };
    let (tx, ty, px) = (
        rows_of(&train_x),
        train_y.as_array().to_vec(),
        rows_of(&pred_x),
    );
    if tx.is_empty() || px.is_empty() {
        return Err(PyValueError::new_err(
            "train_x and pred_x need at least one row",
        ));
    }
    let r = py
        .detach(|| sim::ml_propagation(&tx, &ty, &px, &prm))
        .map_err(to_py_err)?;
    to2(py, &r)
}
