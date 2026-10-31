// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[ingest]` settings.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Curve-ingest selection and alias settings.
///
/// Relative paths are relative to the project folder. This loader does not open them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestConfig {
    /// LAS folders in priority order; relative paths are relative to the project folder (paths)
    pub las_dirs: Vec<PathBuf>,
    /// only these wells; empty = every selected well (well names)
    pub wells: Vec<String>,
    /// wells to skip (well names)
    pub exclude_wells: Vec<String>,
    /// alias CSV replacing the built-in mnemonic lists; None = built-in lists (path)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias_dictionary: Option<PathBuf>,
    /// widest source gap bridged by interpolation (ft)
    pub max_gap_ft: f64,
    /// top order to use; empty = order from the data (top names)
    pub top_names: Vec<String>,
    /// how a LAS file finds its well: "header_first" (UWI/API/WELL in the LAS header, then the file name) or "file_name"
    pub las_match: String,
    /// extra mnemonics per log name (key = log name, case-insensitive)
    pub alias_extra: BTreeMap<String, Vec<String>>,
    /// LAS file name -> well key; always wins over header and file-name matching
    pub las_overrides: BTreeMap<String, String>,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            las_dirs: Vec::new(),
            wells: Vec::new(),
            exclude_wells: Vec::new(),
            alias_dictionary: None,
            max_gap_ft: 2.0,
            top_names: Vec::new(),
            las_match: "header_first".into(),
            alias_extra: BTreeMap::new(),
            las_overrides: BTreeMap::new(),
        }
    }
}
