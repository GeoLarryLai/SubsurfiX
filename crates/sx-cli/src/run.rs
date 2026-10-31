// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `subsurfix run`.

use std::path::Path;

use sx_core::{SxError, SxResult};

use crate::cli::resolve_config;
use crate::pipeline_seam::{run_pipeline, RunRequest};
use crate::stages::Stage;

/// Check arguments and config, then stop at [`run_pipeline`].
///
/// On success, prints `ran:` and `skipped:` lines. Names are joined with `", "`,
/// or `"-"` when a list is empty.
pub fn execute(
    config: &Path,
    from: Option<&str>,
    to: Option<&str>,
    force: bool,
    threads: Option<usize>,
) -> SxResult<()> {
    let (config_path, project_dir) = resolve_config(config)?;
    let cfg = sx_core::config::load_project(&config_path)?;
    let from = named_stage(from, Stage::Ingest)?;
    let to = named_stage(to, Stage::Export)?;
    if from > to {
        return Err(SxError::InvalidInput(format!(
            "--from {} comes after --to {}",
            from.name(),
            to.name()
        )));
    }
    if threads == Some(0) {
        return Err(SxError::InvalidInput("--threads must be >= 1".to_owned()));
    }
    let summary = run_pipeline(&RunRequest {
        config_path,
        project_dir,
        config: cfg,
        from,
        to,
        force,
        threads,
    })?;
    println!("ran: {}", format_stage_list(&summary.ran));
    println!("skipped: {}", format_stage_list(&summary.skipped));
    Ok(())
}

fn named_stage(name: Option<&str>, default: Stage) -> SxResult<Stage> {
    match name {
        Some(s) => Stage::from_name(s),
        None => Ok(default),
    }
}

fn format_stage_list(stages: &[Stage]) -> String {
    if stages.is_empty() {
        return "-".to_owned();
    }
    stages
        .iter()
        .map(|stage| stage.name())
        .collect::<Vec<_>>()
        .join(", ")
}
