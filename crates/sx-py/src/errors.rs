// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use pyo3::prelude::*;
use sx_core::SxError;

/// Python class name for an engine error: variant name + "Error".
fn class_name(err: &SxError) -> &'static str {
    match err {
        SxError::InvalidInput(_) => "InvalidInputError",
        SxError::EmptyInput(_) => "EmptyInputError",
        SxError::Shape(_) => "ShapeError",
        SxError::NotFound(_) => "NotFoundError",
        SxError::Config(_) => "ConfigError",
        SxError::Io { .. } => "IoError",
        SxError::Format(_) => "FormatError",
        SxError::Numerical(_) => "NumericalError",
        SxError::Model(_) => "ModelError",
        SxError::Infeasible(_) => "InfeasibleError",
        _ => "SubsurfixError",
    }
}

/// Convert an engine error to the matching `subsurfix.errors` exception.
/// Message = the error's `Display` text. If `subsurfix.errors` cannot be
/// imported, the import error is returned instead (no silent substitute).
pub(crate) fn to_py_err(err: SxError) -> PyErr {
    let name = class_name(&err);
    let msg = err.to_string();
    Python::attach(|py| {
        let made = py
            .import("subsurfix.errors")
            .and_then(|m| m.getattr(name))
            .and_then(|cls| cls.call1((msg,)));
        match made {
            Ok(exc) => PyErr::from_value(exc),
            Err(e) => e,
        }
    })
}
