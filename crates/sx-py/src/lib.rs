// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
mod errors;
mod flow;
mod grid;
mod plugin;
mod project;
mod sim;
mod testing;

use pyo3::prelude::*;
use sx_core::SxError;

/// Commented project-file template; every key at its default (`crs` is `EPSG:32614`).
#[pyfunction]
fn template_toml() -> String {
    sx_core::config::template_toml()
}

/// Raise the Python exception for one engine error. Test helper.
#[pyfunction]
fn _raise_sx_error(variant: &str, message: &str) -> PyResult<()> {
    let err = match variant {
        "InvalidInput" => SxError::InvalidInput(message.to_owned()),
        "EmptyInput" => SxError::EmptyInput(message.to_owned()),
        "Shape" => SxError::Shape(message.to_owned()),
        "NotFound" => SxError::NotFound(message.to_owned()),
        "Config" => SxError::Config(message.to_owned()),
        "Io" => SxError::Io {
            path: "test-path".to_owned(),
            source: std::io::Error::other(message.to_owned()),
        },
        "Format" => SxError::Format(message.to_owned()),
        "Numerical" => SxError::Numerical(message.to_owned()),
        "Model" => SxError::Model(message.to_owned()),
        "Infeasible" => SxError::Infeasible(message.to_owned()),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown variant {variant:?}"
            )));
        }
    };
    Err(errors::to_py_err(err))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(template_toml, m)?)?;
    m.add_function(wrap_pyfunction!(_raise_sx_error, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::register_model, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::clear_models, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::registered_models, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::_plugin_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(project::version, m)?)?;
    m.add_function(wrap_pyfunction!(project::stage_names, m)?)?;
    m.add_function(wrap_pyfunction!(project::stage_products, m)?)?;
    m.add_function(wrap_pyfunction!(project::config_json, m)?)?;
    m.add_function(wrap_pyfunction!(project::run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(project::pipeline_status, m)?)?;
    m.add_function(wrap_pyfunction!(project::score_holdout, m)?)?;
    m.add_function(wrap_pyfunction!(project::init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(grid::upscale_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(grid::upscale_facies, m)?)?;
    m.add_function(wrap_pyfunction!(grid::grid_centers, m)?)?;
    m.add_function(wrap_pyfunction!(grid::experimental_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(grid::fit_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(grid::idw, m)?)?;
    m.add_function(wrap_pyfunction!(grid::ordinary_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(grid::universal_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(grid::trend_surface, m)?)?;
    m.add_function(wrap_pyfunction!(sim::sgs, m)?)?;
    m.add_function(wrap_pyfunction!(sim::sis, m)?)?;
    m.add_function(wrap_pyfunction!(sim::normal_score, m)?)?;
    m.add_function(wrap_pyfunction!(sim::ml_propagation, m)?)?;
    m.add_class::<grid::PyCornerPointGrid>()?;
    m.add_class::<flow::PyFlowModel>()?;
    m.add_class::<flow::PyFixedPressure>()?;
    m.add_function(wrap_pyfunction!(flow::peaceman_wi, m)?)?;
    m.add_function(wrap_pyfunction!(flow::buckley_leverett, m)?)?;
    m.add_function(wrap_pyfunction!(flow::capillary_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(flow::theis_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(flow::water_properties, m)?)?;
    #[cfg(feature = "testing")]
    m.add_function(wrap_pyfunction!(testing::_make_synthetic_project, m)?)?;
    Ok(())
}
