// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Shared test helpers: tolerance comparison and fixture paths.
pub mod close;
pub mod fixtures;
pub mod open_data;
pub mod synth;
pub mod synth_project;

pub use close::{assert_close, assert_close_scalar, check_close};
pub use fixtures::{
    data_fixture_path, data_fixtures_dir, fixture_path, load_json, read_json, workspace_root,
};
