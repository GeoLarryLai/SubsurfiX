// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Clap command line for `subsurfix`.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use sx_core::{SxError, SxResult};

/// Parsed command line.
#[derive(Debug, Parser)]
#[command(name = "subsurfix", version, about = "SubsurfiX well-log engine")]
pub struct Cli {
    /// More log output (debug level).
    #[arg(short, long, global = true, conflicts_with = "quiet")]
    pub verbose: bool,
    /// Less log output (warnings and errors only).
    #[arg(short, long, global = true)]
    pub quiet: bool,
    #[command(subcommand)]
    pub command: Command,
}

/// `subsurfix` subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Create a project folder with a template project.toml.
    Init {
        /// Directory to create.
        dir: PathBuf,
    },
    /// Run pipeline stages from --from to --to (inclusive).
    Run {
        /// Project file to load.
        #[arg(long, default_value = "project.toml")]
        config: PathBuf,
        /// First stage (default: ingest).
        #[arg(long)]
        from: Option<String>,
        /// Last stage (default: export).
        #[arg(long)]
        to: Option<String>,
        /// Rerun stages in the range even if they are up to date.
        #[arg(long)]
        force: bool,
        /// Worker threads (default: the config's setting).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// List stages and whether their manifest entries and outputs exist.
    Status {
        /// Project file whose folder is listed. The file is not parsed.
        #[arg(long, default_value = "project.toml")]
        config: PathBuf,
    },
    /// Print the version.
    Version,
}

/// Canonical project file and its parent directory.
///
/// A missing file is [`SxError::NotFound`]. Canonicalize failure is [`SxError::Io`].
/// A path with no parent is [`SxError::InvalidInput`].
pub(crate) fn resolve_config(config: &Path) -> SxResult<(PathBuf, PathBuf)> {
    if !config.is_file() {
        return Err(SxError::NotFound(format!(
            "config file {}",
            config.display()
        )));
    }
    let config_path = std::fs::canonicalize(config).map_err(|source| io_error(config, source))?;
    let project_dir = config_path.parent().map(Path::to_path_buf).ok_or_else(|| {
        SxError::InvalidInput(format!(
            "config path {} has no parent directory",
            config_path.display()
        ))
    })?;
    Ok((config_path, project_dir))
}

/// [`SxError::Io`] for `path`. `path` is stored as text; the engine's field is a `String`.
pub(crate) fn io_error(path: &Path, source: std::io::Error) -> SxError {
    SxError::Io {
        path: path.display().to_string(),
        source,
    }
}
