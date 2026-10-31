// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! The ten pipeline stages: names, order, versions and hashed config sections.

use crate::error::{SxError, SxResult};

/// The ten stages, in run order. `PartialOrd` follows this order.
///
/// `name()` is a unitless id. This type stores no samples, so there is no NaN rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// `ingest`
    Ingest,
    /// `qc`
    Qc,
    /// `rough_align`
    RoughAlign,
    /// `normalize`
    Normalize,
    /// `neighbors`
    Neighbors,
    /// `fill`
    Fill,
    /// `correlate`
    Correlate,
    /// `subhorizons`
    Subhorizons,
    /// `refresh`
    Refresh,
    /// `export`
    Export,
}

impl Stage {
    /// Stages in run order.
    pub const ALL: [Stage; 10] = [
        Stage::Ingest,
        Stage::Qc,
        Stage::RoughAlign,
        Stage::Normalize,
        Stage::Neighbors,
        Stage::Fill,
        Stage::Correlate,
        Stage::Subhorizons,
        Stage::Refresh,
        Stage::Export,
    ];

    /// Short id: `"ingest"`, `"qc"`, `"rough_align"`, `"normalize"`, `"neighbors"`,
    /// `"fill"`, `"correlate"`, `"subhorizons"`, `"refresh"`, `"export"`.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Ingest => "ingest",
            Stage::Qc => "qc",
            Stage::RoughAlign => "rough_align",
            Stage::Normalize => "normalize",
            Stage::Neighbors => "neighbors",
            Stage::Fill => "fill",
            Stage::Correlate => "correlate",
            Stage::Subhorizons => "subhorizons",
            Stage::Refresh => "refresh",
            Stage::Export => "export",
        }
    }

    /// Exact, case-sensitive match on [`Stage::name`].
    ///
    /// # Errors
    /// `SxError::InvalidInput("unknown stage \"x\"; valid stages: ingest, qc, ..., export")`.
    pub fn from_name(s: &str) -> SxResult<Stage> {
        if let Some(stage) = Stage::ALL.iter().copied().find(|st| st.name() == s) {
            return Ok(stage);
        }
        let valid = Stage::ALL
            .iter()
            .map(|st| st.name())
            .collect::<Vec<_>>()
            .join(", ");
        Err(SxError::InvalidInput(format!(
            "unknown stage {s:?}; valid stages: {valid}"
        )))
    }

    /// Position in [`Stage::ALL`], 0 (ingest) through 9 (export).
    pub fn index(self) -> usize {
        match self {
            Stage::Ingest => 0,
            Stage::Qc => 1,
            Stage::RoughAlign => 2,
            Stage::Normalize => 3,
            Stage::Neighbors => 4,
            Stage::Fill => 5,
            Stage::Correlate => 6,
            Stage::Subhorizons => 7,
            Stage::Refresh => 8,
            Stage::Export => 9,
        }
    }

    /// Stage version. Bump it when the stage's output for the same input changes.
    pub fn version(self) -> u32 {
        match self {
            // 2: range-map windows (neighbors: layer scope; fill: paired rows) and
            // tops check skips tops outside the logged range.
            // Ingest 2: aux_raw/3 (logging speed). Qc 2: closed-caliper
            // products and report columns.
            Stage::Neighbors | Stage::Fill | Stage::Correlate | Stage::Ingest | Stage::Qc => 2,
            Stage::RoughAlign
            | Stage::Normalize
            | Stage::Subhorizons
            | Stage::Refresh
            | Stage::Export => 1,
        }
    }

    /// Top-level config sections hashed into this stage's config hash.
    pub fn config_sections(self) -> &'static [&'static str] {
        match self {
            Stage::Ingest => &["ingest"],
            Stage::Qc => &["qc"],
            Stage::RoughAlign => &["align", "dtw"],
            Stage::Normalize => &["normalize", "qc"],
            Stage::Neighbors => &["neighbors"],
            Stage::Fill => &["fill"],
            Stage::Correlate => &["correlate", "dtw"],
            Stage::Subhorizons => &["subhorizons"],
            Stage::Refresh => &["refresh"],
            Stage::Export => &["export"],
        }
    }
}
