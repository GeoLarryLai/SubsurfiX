// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Python bindings for `sx_core::grid`: regular-grid upscaling, variogram
//! fitting, inverse distance weighting (IDW) and ordinary kriging (OK).
//! Points are float64 arrays of shape (n, 3): x, y, z.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sx_core::grid::{
    self, Averaging, IdwParams, KrigingParams, ModelKind, RegularGrid, VariogramModel,
};

use crate::errors::to_py_err;

type Arr<'py, T> = Bound<'py, PyArray1<T>>;

fn pts(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "points must have shape (n, 3), got (n, {})",
            v.ncols()
        )));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn mk_grid(origin: [f64; 3], cell: [f64; 3], dims: [usize; 3]) -> PyResult<RegularGrid> {
    RegularGrid::new(origin, cell, dims).map_err(to_py_err)
}

/// Upscale samples to cells. Returns (cell_index, value, count) arrays; cell
/// index = i + nx·(j + ny·k). `method`: arithmetic | harmonic | geometric.
#[pyfunction]
#[pyo3(signature = (origin, cell, dims, points, values, method="arithmetic", min_count=1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn upscale_continuous<'py>(
    py: Python<'py>,
    origin: [f64; 3],
    cell: [f64; 3],
    dims: [usize; 3],
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    method: &str,
    min_count: usize,
) -> PyResult<(Arr<'py, u64>, Arr<'py, f64>, Arr<'py, u64>)> {
    let g = mk_grid(origin, cell, dims)?;
    let m = Averaging::parse(method).map_err(to_py_err)?;
    let r = grid::upscale_continuous(
        &g,
        &pts(&points)?,
        &values.as_array().to_vec(),
        m,
        min_count,
    )
    .map_err(to_py_err)?;
    Ok((
        PyArray1::from_iter(py, r.iter().map(|c| c.0 as u64)),
        PyArray1::from_iter(py, r.iter().map(|c| c.1)),
        PyArray1::from_iter(py, r.iter().map(|c| c.2 as u64)),
    ))
}

/// Facies majority per cell (ties → smallest code; negative codes skipped).
/// Returns (cell_index, code, count).
#[pyfunction]
#[pyo3(signature = (origin, cell, dims, points, codes, min_count=1))]
#[allow(clippy::type_complexity)]
pub fn upscale_facies<'py>(
    py: Python<'py>,
    origin: [f64; 3],
    cell: [f64; 3],
    dims: [usize; 3],
    points: PyReadonlyArray2<'py, f64>,
    codes: PyReadonlyArray1<'py, i32>,
    min_count: usize,
) -> PyResult<(Arr<'py, u64>, Arr<'py, i32>, Arr<'py, u64>)> {
    let g = mk_grid(origin, cell, dims)?;
    let r = grid::upscale_facies(&g, &pts(&points)?, &codes.as_array().to_vec(), min_count)
        .map_err(to_py_err)?;
    Ok((
        PyArray1::from_iter(py, r.iter().map(|c| c.0 as u64)),
        PyArray1::from_iter(py, r.iter().map(|c| c.1)),
        PyArray1::from_iter(py, r.iter().map(|c| c.2 as u64)),
    ))
}

/// Cell centres of a regular grid, shape (nx·ny·nz, 3), in cell-index order.
#[pyfunction]
pub fn grid_centers<'py>(
    py: Python<'py>,
    origin: [f64; 3],
    cell: [f64; 3],
    dims: [usize; 3],
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let g = mk_grid(origin, cell, dims)?;
    let rows: Vec<Vec<f64>> = (0..g.n_cells()).map(|i| g.center(i).to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Experimental variogram. Returns (lag, gamma, pairs).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn experimental_variogram<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    lag_width: f64,
    n_lags: usize,
) -> PyResult<(Arr<'py, f64>, Arr<'py, f64>, Arr<'py, u64>)> {
    let b = grid::experimental_variogram(
        &pts(&points)?,
        &values.as_array().to_vec(),
        lag_width,
        n_lags,
    )
    .map_err(to_py_err)?;
    Ok((
        PyArray1::from_iter(py, b.iter().map(|x| x.lag)),
        PyArray1::from_iter(py, b.iter().map(|x| x.gamma)),
        PyArray1::from_iter(py, b.iter().map(|x| x.pairs as u64)),
    ))
}

/// Fit a model to bins (weights = pairs). Returns dict kind/nugget/psill/range.
#[pyfunction]
#[pyo3(signature = (lag, gamma, pairs, kind="spherical"))]
pub fn fit_variogram<'py>(
    py: Python<'py>,
    lag: PyReadonlyArray1<'py, f64>,
    gamma: PyReadonlyArray1<'py, f64>,
    pairs: PyReadonlyArray1<'py, u64>,
    kind: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (l, g, p) = (lag.as_array(), gamma.as_array(), pairs.as_array());
    if l.len() != g.len() || l.len() != p.len() {
        return Err(PyValueError::new_err(
            "lag, gamma and pairs must have equal length",
        ));
    }
    let bins: Vec<grid::VariogramBin> = (0..l.len())
        .map(|i| grid::VariogramBin {
            lag: l[i],
            gamma: g[i],
            pairs: p[i] as usize,
        })
        .collect();
    let m = grid::fit_variogram(&bins, ModelKind::parse(kind).map_err(to_py_err)?)
        .map_err(to_py_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("kind", kind.to_ascii_lowercase())?;
    d.set_item("nugget", m.nugget)?;
    d.set_item("psill", m.psill)?;
    d.set_item("range", m.range)?;
    Ok(d)
}

/// IDW estimates at targets (NaN where no datum is in reach).
#[pyfunction]
#[pyo3(signature = (points, values, targets, power=2.0, max_neighbors=12, max_distance=f64::INFINITY))]
pub fn idw<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    power: f64,
    max_neighbors: usize,
    max_distance: f64,
) -> PyResult<Arr<'py, f64>> {
    let (p, v, t) = (pts(&points)?, values.as_array().to_vec(), pts(&targets)?);
    let prm = IdwParams {
        power,
        max_neighbors,
        max_distance,
    };
    let r = py
        .detach(|| grid::idw(&p, &v, &t, &prm))
        .map_err(to_py_err)?;
    Ok(PyArray1::from_vec(py, r))
}

/// Ordinary kriging. Returns (estimate, variance).
#[pyfunction]
#[pyo3(signature = (points, values, targets, kind, nugget, psill, range, max_neighbors=12, max_distance=f64::INFINITY))]
#[allow(clippy::too_many_arguments)]
pub fn ordinary_kriging<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    kind: &str,
    nugget: f64,
    psill: f64,
    range: f64,
    max_neighbors: usize,
    max_distance: f64,
) -> PyResult<(Arr<'py, f64>, Arr<'py, f64>)> {
    if !(range > 0.0 && nugget >= 0.0 && psill >= 0.0) {
        return Err(PyValueError::new_err(
            "need range > 0, nugget ≥ 0, psill ≥ 0",
        ));
    }
    let model = VariogramModel {
        kind: ModelKind::parse(kind).map_err(to_py_err)?,
        nugget,
        psill,
        range,
    };
    let (p, v, t) = (pts(&points)?, values.as_array().to_vec(), pts(&targets)?);
    let prm = KrigingParams {
        model,
        max_neighbors,
        max_distance,
    };
    let r = py
        .detach(|| grid::ordinary_kriging(&p, &v, &t, &prm))
        .map_err(to_py_err)?;
    Ok((
        PyArray1::from_vec(py, r.estimate),
        PyArray1::from_vec(py, r.variance),
    ))
}

/// Universal kriging with a polynomial drift in x, y (`drift`: none |
/// linear | quadratic; give the residual variogram). Returns (estimate, variance).
#[pyfunction]
#[pyo3(signature = (points, values, targets, kind, nugget, psill, range, drift="linear", max_neighbors=16, max_distance=f64::INFINITY))]
#[allow(clippy::too_many_arguments)]
pub fn universal_kriging<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    kind: &str,
    nugget: f64,
    psill: f64,
    range: f64,
    drift: &str,
    max_neighbors: usize,
    max_distance: f64,
) -> PyResult<(Arr<'py, f64>, Arr<'py, f64>)> {
    let model = model_of(kind, nugget, psill, range)?;
    let d = grid::Drift::parse(drift).map_err(to_py_err)?;
    let (p, v, t) = (pts(&points)?, values.as_array().to_vec(), pts(&targets)?);
    let prm = KrigingParams {
        model,
        max_neighbors,
        max_distance,
    };
    let r = py
        .detach(|| grid::universal_kriging(&p, &v, &t, &prm, d))
        .map_err(to_py_err)?;
    Ok((
        PyArray1::from_vec(py, r.estimate),
        PyArray1::from_vec(py, r.variance),
    ))
}

/// Least-squares polynomial trend (x, y) fitted on the data, evaluated at
/// `targets` (for detrend-then-krige).
#[pyfunction]
#[pyo3(signature = (points, values, targets, drift="linear"))]
pub fn trend_surface<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    drift: &str,
) -> PyResult<Arr<'py, f64>> {
    let d = grid::Drift::parse(drift).map_err(to_py_err)?;
    let tr = grid::fit_trend(&pts(&points)?, &values.as_array().to_vec(), d).map_err(to_py_err)?;
    Ok(PyArray1::from_iter(
        py,
        pts(&targets)?.iter().map(|q| tr.eval(q)),
    ))
}

pub(crate) fn model_of(
    kind: &str,
    nugget: f64,
    psill: f64,
    range: f64,
) -> PyResult<VariogramModel> {
    if !(range > 0.0 && nugget >= 0.0 && psill >= 0.0) {
        return Err(PyValueError::new_err(
            "need range > 0, nugget ≥ 0, psill ≥ 0",
        ));
    }
    Ok(VariogramModel {
        kind: ModelKind::parse(kind).map_err(to_py_err)?,
        nugget,
        psill,
        range,
    })
}

pub(crate) fn points_of(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    pts(a)
}

/// Corner-point grid (Eclipse layout: COORD pillars, ZCORN, ACTNUM).
#[pyclass(name = "CornerPointGrid", module = "subsurfix._native")]
pub struct PyCornerPointGrid {
    pub(crate) g: grid::CornerPointGrid,
}

#[pymethods]
impl PyCornerPointGrid {
    /// Build from raw arrays (ACTNUM optional: all active).
    #[new]
    #[pyo3(signature = (dims, coord, zcorn, actnum=None))]
    fn new(
        dims: [usize; 3],
        coord: PyReadonlyArray1<'_, f64>,
        zcorn: PyReadonlyArray1<'_, f64>,
        actnum: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<Self> {
        let act = actnum.map(|a| a.as_array().iter().map(|v| *v != 0).collect());
        let g = grid::CornerPointGrid::new(
            dims,
            coord.as_array().to_vec(),
            zcorn.as_array().to_vec(),
            act,
        )
        .map_err(to_py_err)?;
        Ok(Self { g })
    }

    /// Unfaulted grid from node coordinates and layer surfaces, shape
    /// (nz+1, (nx+1)·(ny+1)), top first (depth grows downward).
    #[staticmethod]
    fn from_surfaces(
        xs: PyReadonlyArray1<'_, f64>,
        ys: PyReadonlyArray1<'_, f64>,
        surfaces: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let s: Vec<Vec<f64>> = surfaces
            .as_array()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        let g = grid::CornerPointGrid::from_surfaces(
            &xs.as_array().to_vec(),
            &ys.as_array().to_vec(),
            &s,
        )
        .map_err(to_py_err)?;
        Ok(Self { g })
    }

    /// Corner-point copy of a regular grid.
    #[staticmethod]
    fn from_regular(origin: [f64; 3], cell: [f64; 3], dims: [usize; 3]) -> PyResult<Self> {
        Ok(Self {
            g: grid::CornerPointGrid::from_regular(&mk_grid(origin, cell, dims)?),
        })
    }

    /// Parse GRDECL text. Returns (grid, {keyword: per-cell array}).
    #[staticmethod]
    fn from_grdecl<'py>(
        py: Python<'py>,
        text: &str,
    ) -> PyResult<(Self, Bound<'py, pyo3::types::PyDict>)> {
        let (g, props) = grid::CornerPointGrid::from_grdecl(text).map_err(to_py_err)?;
        let d = pyo3::types::PyDict::new(py);
        for (k, v) in props {
            d.set_item(k, PyArray1::from_vec(py, v))?;
        }
        Ok((Self { g }, d))
    }

    /// GRDECL text with optional per-cell properties.
    #[pyo3(signature = (props=None))]
    fn to_grdecl(
        &self,
        props: Option<std::collections::BTreeMap<String, Vec<f64>>>,
    ) -> PyResult<String> {
        self.g
            .to_grdecl(&props.unwrap_or_default())
            .map_err(to_py_err)
    }

    /// (nx, ny, nz).
    #[getter]
    fn dims(&self) -> (usize, usize, usize) {
        (self.g.dims[0], self.g.dims[1], self.g.dims[2])
    }

    /// COORD array.
    #[getter]
    fn coord<'py>(&self, py: Python<'py>) -> Arr<'py, f64> {
        PyArray1::from_slice(py, &self.g.coord)
    }

    /// ZCORN array.
    #[getter]
    fn zcorn<'py>(&self, py: Python<'py>) -> Arr<'py, f64> {
        PyArray1::from_slice(py, &self.g.zcorn)
    }

    /// ACTNUM as 0/1.
    #[getter]
    fn actnum<'py>(&self, py: Python<'py>) -> Arr<'py, i32> {
        PyArray1::from_iter(py, self.g.actnum.iter().map(|a| i32::from(*a)))
    }

    /// Cell centres, shape (n_cells, 3).
    fn centers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let rows: Vec<Vec<f64>> = (0..self.g.n_cells())
            .map(|i| self.g.center(i).to_vec())
            .collect();
        numpy::PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Cell bulk volumes.
    fn volumes<'py>(&self, py: Python<'py>) -> Arr<'py, f64> {
        PyArray1::from_iter(py, (0..self.g.n_cells()).map(|i| self.g.volume(i)))
    }

    /// Active cell index per point (−1 outside).
    fn locate<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Arr<'py, i64>> {
        let p = pts(&points)?;
        let r: Vec<i64> = py.detach(|| {
            p.iter()
                .map(|q| self.g.locate(*q).map_or(-1, |c| c as i64))
                .collect()
        });
        Ok(PyArray1::from_vec(py, r))
    }

    /// Upscale samples into cells; returns (cell_index, value, count).
    #[pyo3(signature = (points, values, method="arithmetic", min_count=1))]
    #[allow(clippy::type_complexity)]
    fn upscale_continuous<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        values: PyReadonlyArray1<'py, f64>,
        method: &str,
        min_count: usize,
    ) -> PyResult<(Arr<'py, u64>, Arr<'py, f64>, Arr<'py, u64>)> {
        let m = Averaging::parse(method).map_err(to_py_err)?;
        let r = grid::upscale_continuous(
            &self.g,
            &pts(&points)?,
            &values.as_array().to_vec(),
            m,
            min_count,
        )
        .map_err(to_py_err)?;
        Ok((
            PyArray1::from_iter(py, r.iter().map(|c| c.0 as u64)),
            PyArray1::from_iter(py, r.iter().map(|c| c.1)),
            PyArray1::from_iter(py, r.iter().map(|c| c.2 as u64)),
        ))
    }

    /// Facies majority per cell; returns (cell_index, code, count).
    #[pyo3(signature = (points, codes, min_count=1))]
    #[allow(clippy::type_complexity)]
    fn upscale_facies<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        codes: PyReadonlyArray1<'py, i32>,
        min_count: usize,
    ) -> PyResult<(Arr<'py, u64>, Arr<'py, i32>, Arr<'py, u64>)> {
        let r = grid::upscale_facies(
            &self.g,
            &pts(&points)?,
            &codes.as_array().to_vec(),
            min_count,
        )
        .map_err(to_py_err)?;
        Ok((
            PyArray1::from_iter(py, r.iter().map(|c| c.0 as u64)),
            PyArray1::from_iter(py, r.iter().map(|c| c.1)),
            PyArray1::from_iter(py, r.iter().map(|c| c.2 as u64)),
        ))
    }
}
