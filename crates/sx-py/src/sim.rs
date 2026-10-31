// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for `sx_core::sim`: sequential Gaussian simulation (SGS),
//! sequential indicator simulation (SIS), normal scores and the
//! `ml_propagation`.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sx_core::sim::{
    self, EmbedHoldout, EmberInput, EmberParams, PropagationModel, PropagationParams, QrfParams,
    SgsParams, SisParams, Spread,
};

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

/// Quantiles `[row][q]` of a property at `pred_x` learned from
/// `(train_x, train_y)`, shape (n_pred, n_quantiles). `model`: "qrf"
/// (default, quantile regression forest) or "lightgbm".
///
/// `method="ml"`: plain quantile model on the features. `method="ember"`:
/// EMBER envelope quantiles; needs `train_xyz` and `pred_xyz` (n, 3), uses
/// `train_x`/`pred_x` as secondary inputs and `train_groups` (default: one
/// group per row) for the embedded-model search; `embed`, `embed_holdout`,
/// `random_split` and `calibrate` as in `ember`.
#[pyfunction]
#[pyo3(signature = (train_x, train_y, pred_x, model="qrf", quantiles=vec![0.1, 0.5, 0.9], n_trees=200, min_leaf=10, seed=0, method="ml", train_xyz=None, pred_xyz=None, train_groups=None, anisotropy=(1.0, 1.0, 1.0), max_neighbors=16, embed=true, embed_holdout="group", random_split=true, calibrate=false, conformal=None))]
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
    method: &str,
    train_xyz: Option<PyReadonlyArray2<'py, f64>>,
    pred_xyz: Option<PyReadonlyArray2<'py, f64>>,
    train_groups: Option<PyReadonlyArray1<'py, i32>>,
    anisotropy: (f64, f64, f64),
    max_neighbors: usize,
    embed: bool,
    embed_holdout: &str,
    random_split: bool,
    calibrate: bool,
    conformal: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut prm = PropagationParams {
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
    if let Some(c) = conformal {
        prm.conformal_lgbm = c;
        prm.conformal_qrf = c;
    }
    match method {
        "ml" => {
            let g = train_groups.as_ref().map(|a| a.as_array().to_vec());
            let r = py
                .detach(|| sim::ml_propagation_grouped(&tx, &ty, g.as_deref(), &px, &prm))
                .map_err(to_py_err)?;
            to2(py, &r)
        }
        "ember" => {
            let (Some(txyz), Some(pxyz)) = (train_xyz, pred_xyz) else {
                return Err(PyValueError::new_err(
                    "method='ember' needs train_xyz and pred_xyz",
                ));
            };
            let (tp, pp) = (points_of(&txyz)?, points_of(&pxyz)?);
            let g = groups_or_rows(train_groups, ty.len());
            let ep = EmberParams {
                propagation: prm,
                anisotropy: [anisotropy.0, anisotropy.1, anisotropy.2],
                max_neighbors,
                embed,
                embed_holdout: parse_holdout(embed_holdout)?,
                random_split,
                calibrate,
                ..EmberParams::default()
            };
            let r = py
                .detach(|| {
                    sim::ember(
                        &EmberInput {
                            train_xyz: &tp,
                            train_y: &ty,
                            train_groups: &g,
                            train_secondary: &tx,
                            pred_xyz: &pp,
                            pred_secondary: &px,
                        },
                        &ep,
                    )
                })
                .map_err(to_py_err)?;
            to2(py, &r.quantiles)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown method {method:?} (use 'ml' or 'ember')"
        ))),
    }
}

/// `None` -> s = 1, "inflate" -> fitted on training folds, never below 1,
/// "auto" -> fitted, may shrink, number -> fixed.
/// `spread` argument: None, a mode name or a factor.
#[derive(FromPyObject)]
pub enum SpreadArg {
    Factor(f64),
    Mode(String),
}

fn parse_spread(v: Option<SpreadArg>) -> PyResult<Spread> {
    match v {
        None => Ok(Spread::None),
        Some(SpreadArg::Mode(s)) => match s.as_str() {
            "auto" => Ok(Spread::Auto),
            "inflate" => Ok(Spread::Inflate),
            "none" => Ok(Spread::None),
            _ => Err(PyValueError::new_err(format!(
                "spread {s:?} (use None, 'inflate', 'auto' or a number)"
            ))),
        },
        Some(SpreadArg::Factor(f)) if f > 0.0 => Ok(Spread::Fixed(f)),
        Some(SpreadArg::Factor(_)) => Err(PyValueError::new_err("spread must be > 0")),
    }
}

fn parse_holdout(s: &str) -> PyResult<EmbedHoldout> {
    match s {
        "sample" => Ok(EmbedHoldout::Sample),
        "group" => Ok(EmbedHoldout::Group),
        o => Err(PyValueError::new_err(format!(
            "embed_holdout {o:?} (use 'sample' or 'group')"
        ))),
    }
}

fn groups_or_rows(g: Option<PyReadonlyArray1<'_, i32>>, n: usize) -> Vec<i32> {
    g.map(|a| a.as_array().to_vec())
        .unwrap_or_else(|| (0..n as i32).collect())
}

fn secondary_or_empty(a: Option<PyReadonlyArray2<'_, f64>>, n: usize) -> Vec<Vec<f64>> {
    a.map(|a| rows_of(&a)).unwrap_or_else(|| vec![vec![]; n])
}

#[allow(clippy::too_many_arguments)]
fn ember_params(
    model: &str,
    quantiles: Vec<f64>,
    n_trees: usize,
    min_leaf: usize,
    seed: u64,
    anisotropy: (f64, f64, f64),
    use_coordinates: bool,
    embed_ranges: Option<Vec<f64>>,
    max_neighbors: usize,
    max_per_group: usize,
    n_realizations: usize,
    sampling_variogram: Option<(String, f64, f64, f64)>,
    sim_neighbors: usize,
    opts: (bool, &str, bool, usize, bool),
    spread: Spread,
) -> PyResult<EmberParams> {
    let embed_holdout = parse_holdout(opts.1)?;
    Ok(EmberParams {
        embed: opts.0,
        embed_holdout,
        random_split: opts.2,
        gibbs_sweeps: opts.3,
        calibrate: opts.4,
        spread,
        calib_block: 0.0,
        propagation: PropagationParams {
            model: PropagationModel::parse(model).map_err(to_py_err)?,
            quantiles,
            qrf: QrfParams {
                n_trees,
                min_leaf,
                ..QrfParams::default()
            },
            seed,
            ..PropagationParams::default()
        },
        anisotropy: [anisotropy.0, anisotropy.1, anisotropy.2],
        use_coordinates,
        embed_ranges: embed_ranges.unwrap_or_default(),
        max_neighbors,
        max_per_group,
        n_realizations,
        sampling_variogram: match sampling_variogram {
            Some((k, n, p, r)) => Some(model_of(&k, n, p, r)?),
            None => None,
        },
        sim_neighbors,
    })
}

/// EMBER: embedded kriging inputs, quantile-forest envelope, conditional
/// realizations. Returns a dict with `quantiles` (n_targets, n_q), `mean`,
/// `std`, `realizations` (n_realizations, n_targets), `embedded`
/// (n_targets, n_models), `sampling_variogram` and `embed_ranges`.
#[pyfunction]
#[pyo3(signature = (train_xyz, train_y, pred_xyz, train_groups=None, train_secondary=None, pred_secondary=None, model="qrf", quantiles=vec![0.1, 0.5, 0.9], n_trees=200, min_leaf=10, seed=0, anisotropy=(1.0, 1.0, 1.0), use_coordinates=true, embed_ranges=None, max_neighbors=16, max_per_group=4, n_realizations=0, sampling_variogram=None, sim_neighbors=16, embed=true, embed_holdout="group", random_split=true, gibbs_sweeps=100, calibrate=false, spread=Some(SpreadArg::Mode("inflate".into()))))]
#[allow(clippy::too_many_arguments)]
pub fn ember<'py>(
    py: Python<'py>,
    train_xyz: PyReadonlyArray2<'py, f64>,
    train_y: PyReadonlyArray1<'py, f64>,
    pred_xyz: PyReadonlyArray2<'py, f64>,
    train_groups: Option<PyReadonlyArray1<'py, i32>>,
    train_secondary: Option<PyReadonlyArray2<'py, f64>>,
    pred_secondary: Option<PyReadonlyArray2<'py, f64>>,
    model: &str,
    quantiles: Vec<f64>,
    n_trees: usize,
    min_leaf: usize,
    seed: u64,
    anisotropy: (f64, f64, f64),
    use_coordinates: bool,
    embed_ranges: Option<Vec<f64>>,
    max_neighbors: usize,
    max_per_group: usize,
    n_realizations: usize,
    sampling_variogram: Option<(String, f64, f64, f64)>,
    sim_neighbors: usize,
    embed: bool,
    embed_holdout: &str,
    random_split: bool,
    gibbs_sweeps: usize,
    calibrate: bool,
    spread: Option<SpreadArg>,
) -> PyResult<Bound<'py, PyDict>> {
    let ep = ember_params(
        model,
        quantiles,
        n_trees,
        min_leaf,
        seed,
        anisotropy,
        use_coordinates,
        embed_ranges,
        max_neighbors,
        max_per_group,
        n_realizations,
        sampling_variogram,
        sim_neighbors,
        (embed, embed_holdout, random_split, gibbs_sweeps, calibrate),
        parse_spread(spread)?,
    )?;
    let (tp, pp, ty) = (
        points_of(&train_xyz)?,
        points_of(&pred_xyz)?,
        train_y.as_array().to_vec(),
    );
    let g = groups_or_rows(train_groups, ty.len());
    let (ts, ps) = (
        secondary_or_empty(train_secondary, ty.len()),
        secondary_or_empty(pred_secondary, pp.len()),
    );
    let r = py
        .detach(|| {
            sim::ember(
                &EmberInput {
                    train_xyz: &tp,
                    train_y: &ty,
                    train_groups: &g,
                    train_secondary: &ts,
                    pred_xyz: &pp,
                    pred_secondary: &ps,
                },
                &ep,
            )
        })
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("quantiles", to2(py, &r.quantiles)?)?;
    d.set_item("mean", PyArray1::from_vec(py, r.mean))?;
    d.set_item("std", PyArray1::from_vec(py, r.std))?;
    let real = if r.realizations.is_empty() {
        PyArray2::zeros(py, [0, pp.len()], false)
    } else {
        to2(py, &r.realizations)?
    };
    d.set_item("realizations", real)?;
    let emb = if r.embedded.is_empty() {
        PyArray2::zeros(py, [0, r.embed_ranges.len()], false)
    } else {
        to2(py, &r.embedded)?
    };
    d.set_item("embedded", emb)?;
    let v = r.sampling_variogram;
    let vd = PyDict::new(py);
    vd.set_item("kind", format!("{:?}", v.kind).to_lowercase())?;
    vd.set_item("nugget", v.nugget)?;
    vd.set_item("psill", v.psill)?;
    vd.set_item("range", v.range)?;
    d.set_item("sampling_variogram", vd)?;
    d.set_item("embed_ranges", r.embed_ranges)?;
    Ok(d)
}

/// Blocked spatial cross-validation of EMBER against plain
/// `ml_propagation`. Groups are held out by map blocks of side `block`
/// (scaled units; 0 = one block per group). Returns one dict per method:
/// `method`, `mae`, `rmse`, `coverage_p10_p90`, `width_p10_p90`, `n`.
#[pyfunction]
#[pyo3(signature = (xyz, y, groups, secondary=None, n_folds=5, block=0.0, model="qrf", n_trees=200, min_leaf=10, seed=0, anisotropy=(1.0, 1.0, 1.0), use_coordinates=true, embed_ranges=None, max_neighbors=16, max_per_group=4, n_realizations=20, sampling_variogram=None, sim_neighbors=16, embed=true, embed_holdout="group", random_split=true, gibbs_sweeps=100, calibrate=false, spread=Some(SpreadArg::Mode("inflate".into())), conformal=None))]
#[allow(clippy::too_many_arguments)]
pub fn ember_cv<'py>(
    py: Python<'py>,
    xyz: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    groups: PyReadonlyArray1<'py, i32>,
    secondary: Option<PyReadonlyArray2<'py, f64>>,
    n_folds: usize,
    block: f64,
    model: &str,
    n_trees: usize,
    min_leaf: usize,
    seed: u64,
    anisotropy: (f64, f64, f64),
    use_coordinates: bool,
    embed_ranges: Option<Vec<f64>>,
    max_neighbors: usize,
    max_per_group: usize,
    n_realizations: usize,
    sampling_variogram: Option<(String, f64, f64, f64)>,
    sim_neighbors: usize,
    embed: bool,
    embed_holdout: &str,
    random_split: bool,
    gibbs_sweeps: usize,
    calibrate: bool,
    spread: Option<SpreadArg>,
    conformal: Option<bool>,
) -> PyResult<Bound<'py, PyList>> {
    let mut ep = ember_params(
        model,
        vec![0.1, 0.5, 0.9],
        n_trees,
        min_leaf,
        seed,
        anisotropy,
        use_coordinates,
        embed_ranges,
        max_neighbors,
        max_per_group,
        n_realizations,
        sampling_variogram,
        sim_neighbors,
        (embed, embed_holdout, random_split, gibbs_sweeps, calibrate),
        parse_spread(spread)?,
    )?;
    if let Some(c) = conformal {
        ep.propagation.conformal_lgbm = c;
        ep.propagation.conformal_qrf = c;
    }
    let (p, v, g) = (
        points_of(&xyz)?,
        y.as_array().to_vec(),
        groups.as_array().to_vec(),
    );
    let s = secondary_or_empty(secondary, v.len());
    let r = py
        .detach(|| sim::ember_cv(&p, &v, &g, &s, &ep, n_folds, block))
        .map_err(to_py_err)?;
    let out = PyList::empty(py);
    for c in r {
        let d = PyDict::new(py);
        d.set_item("method", c.method)?;
        d.set_item("mae", c.mae)?;
        d.set_item("rmse", c.rmse)?;
        d.set_item("coverage_p10_p90", c.coverage_p10_p90)?;
        d.set_item("width_p10_p90", c.width_p10_p90)?;
        d.set_item("n", c.n)?;
        out.append(d)?;
    }
    Ok(out)
}
