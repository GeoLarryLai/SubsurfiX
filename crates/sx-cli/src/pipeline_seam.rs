// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Interface between the command line and stage execution.

use std::path::PathBuf;

use sx_core::pipeline::StageAction;
use sx_core::SxResult;

use crate::stages::Stage;

/// Checked arguments for one `run`.
///
/// `threads == None` means the project config's thread setting.
#[allow(dead_code)] // Only `from` and `to` are read so far.
pub struct RunRequest {
    /// Absolute canonical path of the project file.
    pub config_path: PathBuf,
    /// Parent of [`Self::config_path`].
    pub project_dir: PathBuf,
    /// Loaded project file.
    pub config: sx_core::config::ProjectConfig,
    /// First stage to run, inclusive.
    pub from: Stage,
    /// Last stage to run, inclusive.
    pub to: Stage,
    /// Rerun stages in the range even if they are up to date.
    pub force: bool,
    /// Worker threads. `None` means the config's thread setting.
    pub threads: Option<usize>,
}

/// Stages reported after a pipeline run.
///
/// `ran` is stages that ran. `skipped` is stages
/// that were skipped or only checked.
#[allow(dead_code)] // Not returned by [`run_pipeline`].
pub struct RunSummary {
    /// Stages that ran.
    pub ran: Vec<Stage>,
    /// Stages that did not run.
    pub skipped: Vec<Stage>,
}

/// Run `req` from `from` through `to`.
///
/// Builds run options from `req` (`project_dir`, `from`, `to`, `force`,
/// `threads`, empty models), calls `sx_core::pipeline::run`, and maps outcomes
/// into [`RunSummary`]. No samples are touched here, so there is no NaN rule.
///
/// # Errors
/// Whatever `sx_core::pipeline::run` returns.
pub fn run_pipeline(req: &RunRequest) -> SxResult<RunSummary> {
    let mut o = sx_core::pipeline::RunOptions::new(&req.project_dir);
    o.from = Some(req.from);
    o.to = Some(req.to);
    o.force = req.force;
    o.threads = req.threads;
    let s = sx_core::pipeline::run(&req.config, &o)?;
    Ok(RunSummary {
        ran: s
            .stages
            .iter()
            .filter(|x| x.action == StageAction::Ran)
            .map(|x| x.stage)
            .collect(),
        skipped: s
            .stages
            .iter()
            .filter(|x| x.action == StageAction::Skipped)
            .map(|x| x.stage)
            .collect(),
    })
}
