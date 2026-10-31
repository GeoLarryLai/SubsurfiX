// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Typed project configuration (`project.toml`).
//!
//! [`ProjectConfig`] gives every key a default, rejects unknown keys and bad
//! values with the key name in the message, and returns the section a stage
//! hashes. [`template_toml`] loads with no data files.

mod align;
mod correlate;
mod dtw;
mod edges;
mod export;
mod fill;
mod ingest;
mod neighbors;
mod normalize;
mod nphi_offset;
mod project;
mod propagation;
mod qc;
mod refresh;
mod subhorizons;
mod template;
mod validate;

pub use align::{AlignParams, ConfidenceParams, PairParams, TopsParams};
pub use correlate::{
    CorrelateConfig, CorrelateGraphConfig, CorrelateSolveConfig, CorrelateTopsConfig,
};
pub use dtw::{DtwConfig, IntervalParams, SpanParams};
pub use edges::{AreaSource, EdgesConfig, HeaderSource, PtsSource, SurveySource, TopsSource};
pub use export::ExportConfig;
pub use fill::{FillConfig, FillLadderConfig, FillLightgbmConfig};
pub use ingest::IngestConfig;
pub use neighbors::NeighborsConfig;
pub use normalize::NormalizeConfig;
pub use nphi_offset::NphiOffsetConfig;
pub use project::{
    from_toml_str, load_project, resolve_path, ProjectConfig, ProjectSection, RuntimeConfig,
};
pub use propagation::PropagationConfig;
pub use qc::{QcClosedCalConfig, QcConfig, QcJuryConfig};
pub use refresh::RefreshConfig;
pub use subhorizons::SubhorizonsConfig;
pub use template::template_toml;
pub use validate::validate;

pub(crate) mod log_names {
    use serde::de::Deserializer;
    use serde::ser::{SerializeSeq, Serializer};
    use serde::Deserialize;

    use crate::types::LogKind;

    /// Write log names as the short upper-case forms (`"GR"`, `"RHOB"`, …).
    pub fn serialize<S: Serializer>(values: &[LogKind], serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(values.len()))?;
        for kind in values {
            seq.serialize_element(kind.name())?;
        }
        seq.end()
    }

    /// Read log names through [`LogKind::from_name`] (trim, case-insensitive).
    ///
    /// An unknown name is a parse error that contains the name.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<LogKind>, D::Error> {
        let names = Vec::<String>::deserialize(deserializer)?;
        let mut out = Vec::with_capacity(names.len());
        for entry in names {
            match LogKind::from_name(&entry) {
                Ok(kind) => out.push(kind),
                Err(_) => {
                    return Err(serde::de::Error::custom(format!(
                        "unknown log name {entry:?}"
                    )));
                }
            }
        }
        Ok(out)
    }
}
