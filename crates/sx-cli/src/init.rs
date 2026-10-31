// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `subsurfix init`.

use std::path::Path;

use sx_core::SxResult;

use crate::cli::io_error;

/// Project file name inside a project folder.
pub const PROJECT_FILE: &str = "project.toml";

/// Create `dir` and write a template [`PROJECT_FILE`] there.
///
/// Never overwrites. After writing, the template is loaded with
/// [`sx_core::config::load_project`]; that error is returned unchanged.
/// Prints `created {path}` using the path as given.
pub fn execute(dir: &Path) -> SxResult<()> {
    std::fs::create_dir_all(dir).map_err(|source| io_error(dir, source))?;
    let path = dir.join(PROJECT_FILE);
    if path.exists() {
        return Err(sx_core::SxError::InvalidInput(format!(
            "{} already exists; init never overwrites",
            path.display()
        )));
    }
    std::fs::write(&path, sx_core::config::template_toml())
        .map_err(|source| io_error(&path, source))?;
    sx_core::config::load_project(&path)?;
    println!("created {}", path.display());
    Ok(())
}
