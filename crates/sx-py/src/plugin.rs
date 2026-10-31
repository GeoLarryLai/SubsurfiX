// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python model plug-in slot: any object with scikit-learn style
//! `fit(X, y, sample_weight=None)` and `predict(X)` replaces the built-in fill
//! model for one log.
//!
//! Every Python touch runs inside `Python::attach`. A `_native` entry that can
//! reach a plug-in releases the GIL first (`py.detach`), else a worker thread
//! calling `Python::attach` waits forever.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sx_core::fill::sequential::{FillModel, RegressorFactory};
use sx_core::pipeline::ModelOverrides;
use sx_core::types::{FeatureMatrix, LogKind, Regressor};
use sx_core::{SxError, SxResult};

use crate::errors::to_py_err;

/// Longest Python error text (message + traceback) kept in an error, in characters.
const MAX_ERR_CHARS: usize = 4000;

/// Registered prototype per log. Locked only with the GIL held and never held
/// across a call into Python (dropping a `Py` can run `__del__`).
static REGISTRY: Mutex<BTreeMap<LogKind, Py<PyAny>>> = Mutex::new(BTreeMap::new());

/// One plug-in model instance: its own deep copy of the registered object.
pub(crate) struct PyRegressor {
    log: LogKind,
    /// Descriptor, e.g. `"py:mypkg.models.MeanModel"`.
    name: String,
    model: Py<PyAny>,
    fitted_names: Option<Vec<String>>,
}

impl PyRegressor {
    fn py_error(&self, py: Python<'_>, method: &str, err: &PyErr) -> SxError {
        model_error(py, &self.name, self.log, method, err)
    }

    /// Call `set_feature_names(names)` when the object has a callable one.
    fn send_names(&self, model: &Bound<'_, PyAny>, names: &[String]) -> SxResult<()> {
        let py = model.py();
        let call = || -> PyResult<()> {
            if let Some(f) = model.getattr_opt("set_feature_names")? {
                if f.is_callable() {
                    f.call1((PyList::new(py, names)?,))?;
                }
            }
            Ok(())
        };
        call().map_err(|e| self.py_error(py, "set_feature_names", &e))
    }
}

impl Regressor for PyRegressor {
    fn name(&self) -> &str {
        &self.name
    }

    /// `model.fit(X, y)`, or `model.fit(X, y, sample_weight=w)` when `weight` is
    /// given; `X` is a C-contiguous `float32` copy of `x`, shape `(n_rows, n_cols)`.
    /// NaN rule: NaN features pass through as missing; non-finite `y` or weight → error.
    /// Errors: `EmptyInput` (no rows), `InvalidInput` (no columns, bad `y` or
    /// weight), `Shape` (lengths), `Model` (any Python exception).
    fn fit(&mut self, x: &FeatureMatrix, y: &[f32], weight: Option<&[f32]>) -> SxResult<()> {
        let log = self.log.name();
        let n = x.n_rows;
        if n == 0 {
            return Err(SxError::EmptyInput(format!("plug-in {log} fit: no rows")));
        }
        if x.n_cols == 0 {
            return Err(SxError::InvalidInput(format!(
                "plug-in {log} fit: no feature columns"
            )));
        }
        check_matrix(self.log, "fit", x)?;
        if y.len() != n {
            return Err(SxError::Shape(format!(
                "plug-in {log} fit: y has {} values for {n} rows",
                y.len()
            )));
        }
        if let Some(w) = weight {
            if w.len() != n {
                return Err(SxError::Shape(format!(
                    "plug-in {log} fit: weight has {} values for {n} rows",
                    w.len()
                )));
            }
        }
        let bad_y = y.iter().filter(|v| !v.is_finite()).count();
        if bad_y > 0 {
            return Err(SxError::InvalidInput(format!(
                "plug-in {log} fit: {bad_y} non-finite y values of {n}"
            )));
        }
        if let Some(w) = weight {
            let bad_w = w.iter().filter(|v| !v.is_finite() || **v < 0.0).count();
            if bad_w > 0 {
                return Err(SxError::InvalidInput(format!(
                    "plug-in {log} fit: {bad_w} weights of {n} are non-finite or negative"
                )));
            }
        }
        Python::attach(|py| {
            let model = self.model.bind(py);
            let xa = to_numpy(py, x).map_err(|e| self.py_error(py, "fit", &e))?;
            let ya = PyArray1::from_slice(py, y);
            self.send_names(model, &x.names)?;
            let done = match weight {
                None => model.call_method1("fit", (xa, ya)),
                Some(w) => {
                    let kwargs = PyDict::new(py);
                    kwargs
                        .set_item("sample_weight", PyArray1::from_slice(py, w))
                        .and_then(|()| model.call_method("fit", (xa, ya), Some(&kwargs)))
                }
            };
            done.map(drop).map_err(|e| self.py_error(py, "fit", &e))
        })?;
        self.fitted_names = Some(x.names.clone());
        Ok(())
    }

    /// `model.predict(X)` cast with `numpy.ascontiguousarray(r, dtype=float32)`;
    /// shape `(n_rows,)` or `(n_rows, 1)` accepted. Zero rows: empty output without
    /// calling Python. NaN rule: NaN features pass through; non-finite predictions
    /// → error. Errors: `Model` (before fit, non-finite output, Python exception),
    /// `Shape` (column names differ from fit, wrong output shape).
    fn predict(&self, x: &FeatureMatrix) -> SxResult<Vec<f32>> {
        let log = self.log.name();
        let Some(fitted) = &self.fitted_names else {
            return Err(SxError::Model(format!(
                "plug-in {log}: predict called before fit"
            )));
        };
        let n_names = fitted.len().max(x.names.len());
        if let Some(i) = (0..n_names).find(|&i| fitted.get(i) != x.names.get(i)) {
            return Err(SxError::Shape(format!(
                "plug-in {log} predict: column {i} is {}, fitted with {}",
                show_name(x.names.get(i)),
                show_name(fitted.get(i))
            )));
        }
        let n = x.n_rows;
        if n == 0 {
            return Ok(Vec::new());
        }
        check_matrix(self.log, "predict", x)?;
        Python::attach(|py| {
            let model = self.model.bind(py);
            let xa = to_numpy(py, x).map_err(|e| self.py_error(py, "predict", &e))?;
            self.send_names(model, &x.names)?;
            let out = model
                .call_method1("predict", (xa,))
                .and_then(|r| as_f32(py, &r))
                .map_err(|e| self.py_error(py, "predict", &e))?;
            let shape = out.shape();
            if !matches!(shape, [m] | [m, 1] if *m == n) {
                return Err(SxError::Shape(format!(
                    "plug-in {log} predict returned shape {}, want (n_rows,) with n_rows = {n}",
                    shape_text(shape)
                )));
            }
            let values = out
                .as_slice()
                .map_err(|e| SxError::Model(format!("plug-in {log} predict: {e}")))?;
            let bad = values.iter().filter(|v| !v.is_finite()).count();
            if bad > 0 {
                return Err(SxError::Model(format!(
                    "plug-in {log} predict returned {bad} non-finite values of {n}"
                )));
            }
            Ok(values.to_vec())
        })
    }
}

/// Plug-ins are never saved: `to_text` is `None`.
impl FillModel for PyRegressor {}

/// Factory over a registered prototype object.
pub(crate) struct PyModelFactory {
    log: LogKind,
    descriptor: String,
    prototype: Py<PyAny>,
}

impl PyModelFactory {
    /// Factory of `log` over `model` (kept by reference, not copied).
    /// Errors: `InvalidInput` without callable `fit` and `predict`; `Model` if
    /// the type name cannot be read.
    pub(crate) fn new(py: Python<'_>, log: LogKind, model: &Bound<'_, PyAny>) -> SxResult<Self> {
        check_callables(log, model)?;
        let descriptor = describe(model).map_err(|e| {
            SxError::Model(format!(
                "plug-in for {} type name: {}",
                log.name(),
                err_text(py, &e)
            ))
        })?;
        Ok(Self {
            log,
            descriptor,
            prototype: model.clone().unbind(),
        })
    }

    /// `"py:" + type(model).__module__ + "." + type(model).__qualname__`.
    pub(crate) fn descriptor(&self) -> &str {
        &self.descriptor
    }
}

impl RegressorFactory for PyModelFactory {
    /// Fresh `copy.deepcopy` of the prototype, unfitted; the registered object
    /// itself is never fitted. Errors: `InvalidInput` for another log; `Model`
    /// if the copy fails.
    fn make(&self, log: LogKind) -> SxResult<Box<dyn FillModel>> {
        if log != self.log {
            return Err(SxError::InvalidInput(format!(
                "plug-in {} is for {}, not {}",
                self.descriptor,
                self.log.name(),
                log.name()
            )));
        }
        let model = Python::attach(|py| {
            deep_copy(self.prototype.bind(py))
                .map(Bound::unbind)
                .map_err(|e| model_error(py, &self.descriptor, log, "deepcopy", &e))
        })?;
        Ok(Box::new(PyRegressor {
            log,
            name: self.descriptor.clone(),
            model,
            fitted_names: None,
        }))
    }
}

/// Registry snapshot for one run: (descriptor, factory) per registered log, in
/// suite order. Later registry changes do not affect it.
/// Errors: `RuntimeError` if the registry is poisoned; `InvalidInputError` if a
/// registered object lost its callable `fit` / `predict`.
pub(crate) fn snapshot_overrides(py: Python<'_>) -> PyResult<ModelOverrides> {
    let entries: Vec<(LogKind, Py<PyAny>)> = registry()?
        .iter()
        .map(|(&log, m)| (log, m.clone_ref(py)))
        .collect();
    let mut out = ModelOverrides::new();
    for (log, model) in entries {
        let factory = PyModelFactory::new(py, log, model.bind(py)).map_err(to_py_err)?;
        let descriptor = factory.descriptor().to_owned();
        let factory: Arc<dyn RegressorFactory> = Arc::new(factory);
        out.insert(log, (descriptor, factory));
    }
    Ok(out)
}

/// Register `model` as the fill model of `log` (name, case-insensitive);
/// replaces an earlier entry. The original object is stored.
/// Errors: `NotFoundError` for an unknown log; `InvalidInputError` without
/// callable `fit` and `predict`, or if `copy.deepcopy(model)` fails.
#[pyfunction]
pub(crate) fn register_model(py: Python<'_>, log: &str, model: Bound<'_, PyAny>) -> PyResult<()> {
    let log = LogKind::from_name(log).map_err(to_py_err)?;
    check_callables(log, &model).map_err(to_py_err)?;
    deep_copy(&model).map_err(|e| {
        to_py_err(SxError::InvalidInput(format!(
            "plug-in for {} cannot be deep-copied: {}",
            log.name(),
            err_text(py, &e)
        )))
    })?;
    let previous = registry()?.insert(log, model.unbind());
    drop(previous);
    Ok(())
}

/// Remove every plug-in (`log=None`) or the one of `log`; no entry is a no-op.
/// Errors: `NotFoundError` for an unknown log name.
#[pyfunction]
#[pyo3(signature = (log=None))]
pub(crate) fn clear_models(_py: Python<'_>, log: Option<&str>) -> PyResult<()> {
    let removed: Vec<Py<PyAny>> = match log {
        None => std::mem::take(&mut *registry()?).into_values().collect(),
        Some(name) => {
            let log = LogKind::from_name(name).map_err(to_py_err)?;
            registry()?.remove(&log).into_iter().collect()
        }
    };
    drop(removed);
    Ok(())
}

/// `{LOG name: descriptor}` of every registered plug-in, in suite order.
#[pyfunction]
pub(crate) fn registered_models(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (log, (descriptor, _)) in snapshot_overrides(py)? {
        out.set_item(log.name(), descriptor)?;
    }
    Ok(out)
}

/// Test hook: factory over `model`, `make(log)`, then fit on (`x_fit`, `y`,
/// `weight`) and predict `x_pred` on a spawned thread with the GIL released.
/// Errors: as `register_model` and the model's `fit` / `predict`; `ShapeError`
/// for non-C-contiguous input or a column count that differs from `names`.
#[pyfunction]
#[pyo3(signature = (log, model, x_fit, names, y, x_pred, weight=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _plugin_roundtrip<'py>(
    py: Python<'py>,
    log: &str,
    model: Bound<'py, PyAny>,
    x_fit: PyReadonlyArray2<'py, f32>,
    names: Vec<String>,
    y: PyReadonlyArray1<'py, f32>,
    x_pred: PyReadonlyArray2<'py, f32>,
    weight: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let log = LogKind::from_name(log).map_err(to_py_err)?;
    let mut regressor = PyModelFactory::new(py, log, &model)
        .and_then(|f| f.make(log))
        .map_err(to_py_err)?;
    let fit_x = matrix(&x_fit, &names).map_err(to_py_err)?;
    let pred_x = matrix(&x_pred, &names).map_err(to_py_err)?;
    let y = vector(&y, "y").map_err(to_py_err)?;
    let weight = weight
        .as_ref()
        .map(|w| vector(w, "weight"))
        .transpose()
        .map_err(to_py_err)?;
    let out = py.detach(|| {
        std::thread::scope(|s| {
            s.spawn(|| {
                regressor.fit(&fit_x, &y, weight.as_deref())?;
                regressor.predict(&pred_x)
            })
            .join()
            .unwrap_or_else(|_| Err(SxError::Model("plug-in thread panicked".into())))
        })
    });
    out.map(|v| PyArray1::from_vec(py, v)).map_err(to_py_err)
}

fn registry() -> PyResult<MutexGuard<'static, BTreeMap<LogKind, Py<PyAny>>>> {
    REGISTRY
        .lock()
        .map_err(|_| PyRuntimeError::new_err("plug-in registry poisoned"))
}

/// `model.fit` and `model.predict` exist and are callable.
fn check_callables(log: LogKind, model: &Bound<'_, PyAny>) -> SxResult<()> {
    for method in ["fit", "predict"] {
        let callable = match model.getattr_opt(method) {
            Ok(f) => f.is_some_and(|f| f.is_callable()),
            Err(e) => {
                return Err(SxError::InvalidInput(format!(
                    "plug-in for {} {method}: {}",
                    log.name(),
                    err_text(model.py(), &e)
                )))
            }
        };
        if !callable {
            return Err(SxError::InvalidInput(format!(
                "plug-in for {} must have callable fit and predict",
                log.name()
            )));
        }
    }
    Ok(())
}

fn deep_copy<'py>(model: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    model
        .py()
        .import("copy")?
        .call_method1("deepcopy", (model,))
}

fn describe(model: &Bound<'_, PyAny>) -> PyResult<String> {
    let ty = model.get_type();
    Ok(format!(
        "py:{}.{}",
        ty.getattr("__module__")?.str()?,
        ty.getattr("__qualname__")?.str()?
    ))
}

/// `plug-in {name} for {LOG} {method}: {err}`.
fn model_error(py: Python<'_>, name: &str, log: LogKind, method: &str, err: &PyErr) -> SxError {
    SxError::Model(format!(
        "plug-in {name} for {} {method}: {}",
        log.name(),
        err_text(py, err)
    ))
}

/// `PyErr` display text (`"ValueError: boom"`), then `"\n"` + formatted
/// traceback when present, cut to [`MAX_ERR_CHARS`] characters.
fn err_text(py: Python<'_>, err: &PyErr) -> String {
    let mut text = err.to_string();
    if let Some(tb) = err.traceback(py).and_then(|tb| tb.format().ok()) {
        text.push('\n');
        text.push_str(tb.trim_end());
    }
    text.chars().take(MAX_ERR_CHARS).collect()
}

/// `data` and `names` agree with `n_rows` x `n_cols`.
fn check_matrix(log: LogKind, method: &str, x: &FeatureMatrix) -> SxResult<()> {
    if x.n_rows.checked_mul(x.n_cols) != Some(x.data.len()) || x.names.len() != x.n_cols {
        return Err(SxError::Shape(format!(
            "plug-in {} {method}: {} values and {} names for {} x {}",
            log.name(),
            x.data.len(),
            x.names.len(),
            x.n_rows,
            x.n_cols
        )));
    }
    Ok(())
}

/// `x` as a C-contiguous `float32` array of shape `(n_rows, n_cols)` (one copy).
fn to_numpy<'py>(py: Python<'py>, x: &FeatureMatrix) -> PyResult<Bound<'py, PyArray2<f32>>> {
    PyArray1::from_slice(py, &x.data).reshape([x.n_rows, x.n_cols])
}

/// `numpy.ascontiguousarray(r, dtype=numpy.float32)`.
fn as_f32<'py>(py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, f32>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float32")?)?;
    let a = np.call_method("ascontiguousarray", (r,), Some(&kwargs))?;
    a.extract()
}

fn shape_text(shape: &[usize]) -> String {
    match shape {
        [one] => format!("({one},)"),
        _ => {
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            format!("({})", dims.join(", "))
        }
    }
}

fn show_name(name: Option<&String>) -> String {
    name.map_or_else(|| "missing".to_owned(), |s| format!("{s:?}"))
}

/// Copy a C-contiguous 2-D array into a `FeatureMatrix` named `names`.
fn matrix(a: &PyReadonlyArray2<'_, f32>, names: &[String]) -> SxResult<FeatureMatrix> {
    let &[n_rows, n_cols] = a.shape() else {
        return Err(SxError::Shape(
            "plug-in roundtrip: matrix is not 2-D".into(),
        ));
    };
    if n_cols != names.len() {
        return Err(SxError::Shape(format!(
            "plug-in roundtrip: {n_cols} columns, {} names",
            names.len()
        )));
    }
    if !a.is_c_contiguous() {
        return Err(SxError::Shape(
            "plug-in roundtrip: matrix is not C-contiguous".into(),
        ));
    }
    let data = a
        .as_slice()
        .map_err(|e| SxError::Shape(format!("plug-in roundtrip: {e}")))?;
    FeatureMatrix::new(n_rows, names.to_vec(), data.to_vec())
}

fn vector(a: &PyReadonlyArray1<'_, f32>, what: &str) -> SxResult<Vec<f32>> {
    a.as_slice()
        .map(<[f32]>::to_vec)
        .map_err(|e| SxError::Shape(format!("plug-in roundtrip {what}: {e}")))
}
