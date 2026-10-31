// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for `sx_core::facies`.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sx_core::facies::{self, Classifier, SeparabilityInput, SeparabilityParams};

use crate::errors::to_py_err;

/// How many top-level classes the logs can tell apart, and which logs matter.
///
/// `logs`: (n_rows, n_logs), NaN = missing. `labels`: class code per row
/// (negative = unlabelled) or None for the cluster mode. `groups`: group per
/// row (usually the well); groups are held out together. Returns a dict.
#[pyfunction]
#[pyo3(signature = (logs, names, groups, labels=None, classifier="gaussian", n_folds=5, seed=0, min_rows_class=200, max_rows=5000, max_logs=0, min_coverage=0.3, target_accuracy=0.7, min_recall=0.5, tol=0.02, k_max=8, ps_threshold=0.8, ps_margin=0.05, valley_max=0.7, ps_splits=10))]
#[allow(clippy::too_many_arguments)]
pub fn separability<'py>(
    py: Python<'py>,
    logs: PyReadonlyArray2<'py, f64>,
    names: Vec<String>,
    groups: PyReadonlyArray1<'py, i32>,
    labels: Option<PyReadonlyArray1<'py, i32>>,
    classifier: &str,
    n_folds: usize,
    seed: u64,
    min_rows_class: usize,
    max_rows: usize,
    max_logs: usize,
    min_coverage: f64,
    target_accuracy: f64,
    min_recall: f64,
    tol: f64,
    k_max: usize,
    ps_threshold: f64,
    ps_margin: f64,
    valley_max: f64,
    ps_splits: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rows: Vec<Vec<f64>> = logs
        .as_array()
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    let g = groups.as_array().to_vec();
    let l = labels.map(|a| a.as_array().to_vec());
    let prm = SeparabilityParams {
        classifier: Classifier::parse(classifier).map_err(to_py_err)?,
        n_folds,
        seed,
        min_rows_class,
        max_rows,
        max_logs,
        min_coverage,
        target_accuracy,
        min_recall,
        tol,
        k_max,
        ps_threshold,
        ps_margin,
        valley_max,
        ps_splits,
        ..SeparabilityParams::default()
    };
    if names.is_empty() {
        return Err(PyValueError::new_err("names must not be empty"));
    }
    let r = py
        .detach(|| {
            facies::separability(
                &SeparabilityInput {
                    logs: &rows,
                    names: &names,
                    labels: l.as_deref(),
                    groups: &g,
                },
                &prm,
            )
        })
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("mode", &r.mode)?;
    d.set_item("names", &r.names)?;
    let levels = PyList::empty(py);
    for lv in &r.levels {
        let e = PyDict::new(py);
        e.set_item("k", lv.k)?;
        e.set_item("classes", &lv.classes)?;
        levels.append(e)?;
    }
    d.set_item("levels", levels)?;
    d.set_item("dropped_classes", &r.dropped_classes)?;
    d.set_item("reference_logs", &r.reference_logs)?;
    let scores = PyList::empty(py);
    for s in &r.scores {
        let e = PyDict::new(py);
        e.set_item("k", s.k)?;
        e.set_item("logs", &s.logs)?;
        e.set_item("n_rows", s.n_rows)?;
        e.set_item("coverage", s.coverage)?;
        e.set_item("balanced_accuracy", s.balanced_accuracy)?;
        e.set_item("min_recall", s.min_recall)?;
        e.set_item("kappa", s.kappa)?;
        e.set_item("effective_classes", s.effective_classes)?;
        e.set_item("prediction_strength", s.prediction_strength)?;
        e.set_item("prediction_strength_null", s.prediction_strength_null)?;
        e.set_item("valley_ratio", s.valley_ratio)?;
        scores.append(e)?;
    }
    d.set_item("scores", scores)?;
    d.set_item("recommended_k", r.recommended_k)?;
    d.set_item("best_logs", &r.best_logs)?;
    let imp = PyDict::new(py);
    for i in &r.importance {
        imp.set_item(&r.names[i.log], i.drop)?;
    }
    d.set_item("importance", imp)?;
    d.set_item("confusion", &r.confusion)?;
    Ok(d)
}

/// Five-facies interpretation and roll-up from GR, RHOB, NPHI (limestone
/// units) and PEF; NaN = missing.
///
/// `method`: "cutoffs" or "mixing". `cutoffs_csv`: rules table text
/// (header `order,facies,variable,min,max,coarse`); None = defaults.
/// `endpoints`: dict of mineral -> (rhob, nphi, pef) overrides.
/// `nphi_shift`: subtracted from NPHI first (v/v).
#[pyfunction]
#[pyo3(signature = (gr, rhob, nphi, pef, method="cutoffs", cutoffs_csv=None, endpoints=None, shale_clay=0.5, chert_quartz=0.2, nphi_shift=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn facies_interpret<'py>(
    py: Python<'py>,
    gr: PyReadonlyArray1<'py, f64>,
    rhob: PyReadonlyArray1<'py, f64>,
    nphi: PyReadonlyArray1<'py, f64>,
    pef: PyReadonlyArray1<'py, f64>,
    method: &str,
    cutoffs_csv: Option<&str>,
    endpoints: Option<std::collections::HashMap<String, (f64, f64, f64)>>,
    shale_clay: f64,
    chert_quartz: f64,
    nphi_shift: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray1;
    let mut p = facies::FaciesParams {
        method: facies::Method::parse(method).map_err(to_py_err)?,
        shale_clay,
        chert_quartz,
        nphi_shift,
        ..facies::FaciesParams::default()
    };
    if let Some(c) = cutoffs_csv {
        p.cutoffs = facies::Cutoffs::from_csv(c).map_err(to_py_err)?;
    }
    for (k, v) in endpoints.unwrap_or_default() {
        let e = [v.0, v.1, v.2];
        match k.as_str() {
            "quartz" => p.endpoints.quartz = e,
            "calcite" => p.endpoints.calcite = e,
            "dolomite" => p.endpoints.dolomite = e,
            "clay" => p.endpoints.clay = e,
            "fluid" => p.endpoints.fluid = e,
            _ => return Err(PyValueError::new_err(format!("unknown endpoint {k:?}"))),
        }
    }
    let (g, r, n, f) = (
        gr.as_array().to_vec(),
        rhob.as_array().to_vec(),
        nphi.as_array().to_vec(),
        pef.as_array().to_vec(),
    );
    let out = facies::interpret(
        &facies::FaciesInput {
            gr: &g,
            rhob: &r,
            nphi: &n,
            pef: &f,
        },
        &p,
    )
    .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("facies", PyArray1::from_vec(py, out.facies))?;
    d.set_item("rollup", PyArray1::from_vec(py, out.rollup))?;
    d.set_item("coarse", PyArray1::from_vec(py, out.coarse))?;
    d.set_item("rhomaa", PyArray1::from_vec(py, out.rhomaa))?;
    d.set_item("umaa", PyArray1::from_vec(py, out.umaa))?;
    if !out.volumes.is_empty() {
        let names = ["quartz", "calcite", "dolomite", "clay", "porosity"];
        for (j, nm) in names.iter().enumerate() {
            d.set_item(
                format!("v_{nm}"),
                PyArray1::from_vec(py, out.volumes.iter().map(|v| v[j]).collect()),
            )?;
        }
        d.set_item("misfit", PyArray1::from_vec(py, out.misfit))?;
    }
    d.set_item(
        "facies_names",
        facies::Facies::ALL
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("rollup_names", vec!["sandstone", "mudstone", "carbonate"])?;
    Ok(d)
}

/// Default cutoffs table as CSV text.
#[pyfunction]
pub fn facies_default_cutoffs() -> String {
    facies::Cutoffs::default().to_csv()
}

/// Fit a Gaussian roll-up classifier on labelled rows (labels 0..2,
/// others ignored) and predict rows of `x_pred`; -1 where a log is missing.
#[pyfunction]
#[pyo3(signature = (x_train, y_train, x_pred, shrink=0.1))]
pub fn facies_rollup_classify<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f64>,
    y_train: PyReadonlyArray1<'py, i32>,
    x_pred: PyReadonlyArray2<'py, f64>,
    shrink: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<i32>>> {
    let rows = |a: &PyReadonlyArray2<'py, f64>| -> Vec<Vec<f64>> {
        a.as_array()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect()
    };
    let m = facies::RollupClassifier::fit(&rows(&x_train), &y_train.as_array().to_vec(), shrink)
        .map_err(to_py_err)?;
    Ok(numpy::PyArray1::from_vec(py, m.predict(&rows(&x_pred))))
}
