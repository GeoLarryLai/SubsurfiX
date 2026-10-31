// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! A synthetic project folder for pipeline tests: `project.toml`, the edge
//! tables the ingest stage reads, and one LAS file per well.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sx_core::config::template_toml;
use sx_core::ingest::write_edges;
use sx_core::io::las::{well_las_string, LasExportOptions};
use sx_core::{SxError, SxResult};

use crate::synth::{generate, GapConfig, SynthConfig};

/// Size and seed of a synthetic project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthProjectSpec {
    /// Number of wells. At least 1.
    pub n_wells: usize,
    /// Seed of the synthetic data (the project `seed` stays 0).
    pub seed: u64,
}

impl Default for SynthProjectSpec {
    /// 10 wells, seed 0.
    fn default() -> Self {
        Self {
            n_wells: 10,
            seed: 0,
        }
    }
}

/// `project.toml`: the template with every key at its default, except the
/// lines replaced here (LAS folder, thread counts, the learning rate and the
/// refresh tolerance that tests edit by text).
pub fn project_toml() -> String {
    let mut out = String::new();
    let mut section = String::new();
    for line in template_toml().lines() {
        if line.starts_with('[') {
            section = line.trim().to_string();
        }
        let key = line.split_once(" = ").map(|(k, _)| k);
        let replaced = match (section.as_str(), key) {
            ("[ingest]", Some("las_dirs")) => Some("las_dirs = [\"las\"]"),
            ("[runtime]", Some("threads")) => Some("threads = 2"),
            ("[fill.lightgbm]", Some("num_threads")) => Some("num_threads = 1"),
            ("[fill.lightgbm]", Some("learning_rate")) => Some("learning_rate = 0.1"),
            ("[refresh]", Some("refresh_tol_ft")) => Some("refresh_tol_ft = 5.0"),
            _ => None,
        };
        out.push_str(replaced.unwrap_or(line));
        out.push('\n');
    }
    out
}

/// Write `project.toml`, `edges/*.parquet` and `las/<well name>.las` into
/// `dir`. Returns the `project.toml` path.
///
/// Data: [`generate`] with `spec.seed`, `spec.n_wells`, default gaps (six
/// tops, all six logs, vertical wells). LAS files hold RESD in linear ohm·m;
/// tops are written as MD picks.
///
/// # Errors
/// Errors of [`generate`], the LAS writer and the edge writer; `SxError::Io`.
pub fn write_synthetic_project(dir: &Path, spec: &SynthProjectSpec) -> SxResult<PathBuf> {
    let project = generate(&SynthConfig {
        seed: spec.seed,
        n_wells: spec.n_wells,
        gaps: Some(GapConfig::default()),
        ..SynthConfig::default()
    })?;
    let io = |path: &Path, e: std::io::Error| SxError::Io {
        path: path.display().to_string(),
        source: e,
    };
    let las_dir = dir.join("las");
    std::fs::create_dir_all(&las_dir).map_err(|e| io(&las_dir, e))?;
    let opts = LasExportOptions {
        resd_is_log10: false,
        ..LasExportOptions::default()
    };
    for w in &project.wells {
        let text = well_las_string(&w.header, &w.geometry, &w.logs, &opts)?;
        let path = las_dir.join(format!("{}.las", w.header.name));
        std::fs::write(&path, text).map_err(|e| io(&path, e))?;
    }
    let headers: Vec<_> = project.wells.iter().map(|w| w.header.clone()).collect();
    let tops: Vec<_> = project
        .tops
        .iter()
        .filter_map(|t| {
            let name = project.top_set.names.get(usize::from(t.top))?;
            Some((t.well, name.clone(), t.md_ft, 0u8))
        })
        .collect();
    write_edges(dir, &headers, &BTreeMap::new(), &tops)?;
    let toml = dir.join("project.toml");
    std::fs::write(&toml, project_toml()).map_err(|e| io(&toml, e))?;
    Ok(toml)
}
