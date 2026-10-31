// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `manifest.json`: per-stage records and run history.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::atomic::write_atomic;
use crate::error::{SxError, SxResult};
use crate::io::atomic::io_error;

/// File name of the manifest inside the project folder.
pub(crate) const MANIFEST_FILE: &str = "manifest.json";

/// Value of the `schema` key.
pub(crate) const MANIFEST_SCHEMA: &str = "subsurfix.manifest/1";

/// What the last successful run of one stage used and wrote.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StageRecord {
    /// Stage format version ([`super::Stage::version`]).
    pub(crate) version: u32,
    /// `sx-core` package version that wrote the outputs; empty in manifests
    /// written before version stamps (such records count as stale).
    #[serde(default)]
    pub(crate) package_version: String,
    /// Stamp (stage key) of the stage before this one; empty for ingest and
    /// in manifests written before version stamps.
    #[serde(default)]
    pub(crate) upstream_key: String,
    pub(crate) stage_key: String,
    pub(crate) config_hash: String,
    /// Relative path (or absolute, outside the project) → SHA-256.
    pub(crate) inputs: BTreeMap<String, String>,
    /// Relative path → SHA-256.
    pub(crate) outputs: BTreeMap<String, String>,
    pub(crate) started_utc: String,
    pub(crate) finished_utc: String,
    pub(crate) n_wells: usize,
}

/// One stage that ran (or failed) in some run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct HistoryEntry {
    pub(crate) stage: String,
    /// `ran` or `failed`.
    pub(crate) action: String,
    pub(crate) stage_key: String,
    pub(crate) started_utc: String,
    pub(crate) finished_utc: String,
    pub(crate) n_wells: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

/// The whole manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Manifest {
    pub(crate) schema: String,
    pub(crate) stages: BTreeMap<String, StageRecord>,
    pub(crate) history: Vec<HistoryEntry>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            schema: MANIFEST_SCHEMA.to_string(),
            stages: BTreeMap::new(),
            history: Vec::new(),
        }
    }
}

impl Manifest {
    /// Read `<project>/manifest.json`; a missing file gives an empty manifest.
    ///
    /// # Errors
    /// `SxError::Format` (not parseable, wrong `schema`), `SxError::Io`.
    pub(crate) fn load(project_dir: &Path) -> SxResult<Self> {
        let path = project_dir.join(MANIFEST_FILE);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(io_error(&path, e)),
        };
        let m: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| SxError::Format(format!("{MANIFEST_FILE}: {e}")))?;
        if m.schema != MANIFEST_SCHEMA {
            return Err(SxError::Format(format!(
                "{MANIFEST_FILE}: schema is {:?}, expected {MANIFEST_SCHEMA:?}",
                m.schema
            )));
        }
        Ok(m)
    }

    /// Write `<project>/manifest.json` atomically: pretty JSON, 2-space indent,
    /// trailing newline.
    ///
    /// # Errors
    /// `SxError::Format` if it cannot be serialized; `SxError::Io`.
    pub(crate) fn save(&self, project_dir: &Path) -> SxResult<()> {
        let mut text = serde_json::to_string_pretty(self)
            .map_err(|e| SxError::Format(format!("{MANIFEST_FILE}: {e}")))?;
        text.push('\n');
        write_atomic(&project_dir.join(MANIFEST_FILE), |tmp| {
            std::fs::write(tmp, text.as_bytes()).map_err(|e| io_error(tmp, e))
        })?;
        Ok(())
    }
}
