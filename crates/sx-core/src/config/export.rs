// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[export]` settings.

use serde::{Deserialize, Serialize};

/// Which products the export stage writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExportConfig {
    /// write one LAS file per well
    pub las: bool,
    /// copy the final Parquet products into export/
    pub parquet: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            las: true,
            parquet: true,
        }
    }
}
