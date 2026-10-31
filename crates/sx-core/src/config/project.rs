// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `ProjectConfig`: load `project.toml`, resolve paths, and pick a stage's hash section.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::align::AlignParams;
use super::correlate::CorrelateConfig;
use super::dtw::DtwConfig;
use super::edges::EdgesConfig;
use super::export::ExportConfig;
use super::fill::FillConfig;
use super::ingest::IngestConfig;
use super::neighbors::NeighborsConfig;
use super::normalize::NormalizeConfig;
use super::propagation::PropagationConfig;
use super::qc::QcConfig;
use super::refresh::RefreshConfig;
use super::subhorizons::SubhorizonsConfig;
use super::validate::validate;
use crate::{SxError, SxResult};

/// Whole project file. `project.crs` is required; every other section has a default.
///
/// There is no [`Default`]: the CRS has none. Use [`ProjectConfig::default_with_crs`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    /// global seed mixed into every random stream
    #[serde(default)]
    pub seed: u64,
    /// Required: a file without `[project] crs` fails with `missing field`.
    pub project: ProjectSection,
    /// rayon pool size; never part of a stage hash
    #[serde(default)]
    pub runtime: RuntimeConfig,
    /// header, tops, survey, and study-area sources
    #[serde(default)]
    pub edges: EdgesConfig,
    /// curve ingest
    #[serde(default)]
    pub ingest: IngestConfig,
    /// QC masks and jury
    #[serde(default)]
    pub qc: QcConfig,
    /// rough alignment
    #[serde(default)]
    pub align: AlignParams,
    /// interval and span DTW
    #[serde(default)]
    pub dtw: DtwConfig,
    /// normalization
    #[serde(default)]
    pub normalize: NormalizeConfig,
    /// neighbor curves
    #[serde(default)]
    pub neighbors: NeighborsConfig,
    /// gap fill
    #[serde(default)]
    pub fill: FillConfig,
    /// network correlation
    #[serde(default)]
    pub correlate: CorrelateConfig,
    /// sub-horizons
    #[serde(default)]
    pub subhorizons: SubhorizonsConfig,
    /// fill refresh
    #[serde(default)]
    pub refresh: RefreshConfig,
    /// export products
    #[serde(default)]
    pub export: ExportConfig,
    /// ml_propagation
    #[serde(default)]
    pub propagation: PropagationConfig,
}

/// `[project]`. `crs` has no default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSection {
    /// CRS of the project frame, any PROJ/EPSG text, e.g. "EPSG:32614"; read by Python, only checked non-empty here
    pub crs: String,
}

/// Global rayon pool size. `0` means all available cores. Never hashed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    /// size of the global rayon pool; 0 = all available cores; never hashed (threads)
    pub threads: usize,
}

/// Parse TOML text, then `validate`. Parse errors become `SxError::Config` (4.6).
///
/// Numeric fields keep the units of each key (ft, m, fractions, counts). Non-finite
/// floats fail validation. Relative paths are left as written; nothing is opened.
pub fn from_toml_str(text: &str) -> SxResult<ProjectConfig> {
    let cfg = match toml::from_str::<ProjectConfig>(text) {
        Ok(cfg) => cfg,
        Err(err) => {
            let message = err.message().to_owned();
            return Err(match err.span() {
                Some(span) => {
                    let (line, col) = line_col(text, span.start);
                    SxError::Config(format!(
                        "parse error at line {line}, column {col}: {message}"
                    ))
                }
                None => SxError::Config(format!("parse error: {message}")),
            });
        }
    };
    validate(&cfg)?;
    Ok(cfg)
}

/// Read the file (failure -> `SxError::Io`), then `from_toml_str`.
///
/// A missing file is `SxError::Io`, not `NotFound`. Paths inside the file are
/// not resolved.
pub fn load_project(path: &Path) -> SxResult<ProjectConfig> {
    let text = std::fs::read_to_string(path).map_err(|source| SxError::Io {
        path: path.display().to_string(),
        source,
    })?;
    from_toml_str(&text)
}

/// Absolute `p` stays as is; a relative `p` becomes `project_dir.join(p)`. Never touches the disk.
pub fn resolve_path(project_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_dir.join(path)
    }
}

/// Byte offset `offset` as 1-based line and column. Does not slice `text`.
fn line_col(text: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (index, ch) in text.char_indices() {
        if index >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

impl ProjectConfig {
    /// Every section at its default, `project.crs = crs`, `seed = 0`.
    ///
    /// Equal to `from_toml_str(&template_toml())` when `crs == "EPSG:32614"`.
    /// Does not validate, so an empty `crs` is returned as given.
    pub fn default_with_crs(crs: &str) -> ProjectConfig {
        ProjectConfig {
            seed: 0,
            project: ProjectSection {
                crs: crs.to_string(),
            },
            runtime: RuntimeConfig::default(),
            edges: EdgesConfig::default(),
            ingest: IngestConfig::default(),
            qc: QcConfig::default(),
            align: AlignParams::default(),
            dtw: DtwConfig::default(),
            normalize: NormalizeConfig::default(),
            neighbors: NeighborsConfig::default(),
            fill: FillConfig::default(),
            correlate: CorrelateConfig::default(),
            subhorizons: SubhorizonsConfig::default(),
            refresh: RefreshConfig::default(),
            export: ExportConfig::default(),
            propagation: PropagationConfig::default(),
        }
    }

    /// Config section(s) that stage `stage` hashes, as JSON (4.7).
    ///
    /// One listed section is returned as that object. Several are returned as
    /// one object keyed by section name. `project`, `runtime`, and `edges` are
    /// never included. An unknown stage is `SxError::InvalidInput`. A serialize
    /// failure or a missing section is `SxError::Config`.
    pub fn stage_section(&self, stage: &str) -> SxResult<Value> {
        let sections: &[&str] = match stage {
            "ingest" => &["ingest"],
            "qc" => &["qc"],
            "rough_align" => &["align", "dtw"],
            "normalize" => &["normalize", "qc"],
            "neighbors" => &["neighbors"],
            "fill" => &["fill"],
            "correlate" => &["correlate", "dtw"],
            "subhorizons" => &["subhorizons"],
            "refresh" => &["refresh"],
            "export" => &["export"],
            _ => {
                return Err(SxError::InvalidInput(format!(
                    "unknown stage {stage:?}; valid stages: ingest, qc, rough_align, normalize, neighbors, fill, correlate, subhorizons, refresh, export"
                )));
            }
        };
        let root = serde_json::to_value(self).map_err(|err| SxError::Config(err.to_string()))?;
        let Some(obj) = root.as_object() else {
            return Err(SxError::Config(
                "stage_section: config is not a table".to_string(),
            ));
        };
        let mut gathered = serde_json::Map::new();
        for section in sections {
            match obj.get(*section) {
                Some(value) if value.is_object() => {
                    gathered.insert((*section).to_string(), value.clone());
                }
                _ => {
                    return Err(SxError::Config(format!(
                        "stage_section: section {section:?} is missing or not a table"
                    )));
                }
            }
        }
        if let [section] = sections {
            return match gathered.remove(*section) {
                Some(value) => Ok(value),
                None => Err(SxError::Config(format!(
                    "stage_section: section {section:?} is missing or not a table"
                ))),
            };
        }
        Ok(Value::Object(gathered))
    }
}
