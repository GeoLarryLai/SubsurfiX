// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Test helpers. Compiled only with the `testing` Cargo feature, so release
//! wheels (built without it) do not carry them.
#![cfg(feature = "testing")]

use std::path::PathBuf;

use pyo3::prelude::*;
use sx_testutil::synth_project::{write_synthetic_project, SynthProjectSpec};

use crate::errors::to_py_err;

/// Write a synthetic project (`project.toml`, edge tables, one LAS per well)
/// into `dir`, GIL released; returns the `project.toml` path as text.
/// Errors: those of `write_synthetic_project` (for example `IoError`).
#[pyfunction]
#[pyo3(signature = (dir, n_wells=10, seed=0))]
pub(crate) fn _make_synthetic_project(
    py: Python<'_>,
    dir: PathBuf,
    n_wells: usize,
    seed: u64,
) -> PyResult<String> {
    let spec = SynthProjectSpec { n_wells, seed };
    py.detach(move || write_synthetic_project(&dir, &spec))
        .map(|path| path.display().to_string())
        .map_err(to_py_err)
}
