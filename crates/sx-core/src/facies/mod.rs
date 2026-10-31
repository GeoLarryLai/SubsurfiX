// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Facies tools.
//!
//! - [`separability`]: how many top-level lithology classes the available
//!   logs can tell apart, and which logs do it.
//! - [`scheme`]: roll-up classes and the five-facies interpretation layer.

pub mod separability;

pub use separability::{
    separability, Classifier, LogImportance, MergeLevel, ScoreRow, SeparabilityInput,
    SeparabilityParams, SeparabilityReport,
};

pub mod scheme;

pub use scheme::{
    interpret, nnls, Cutoffs, Endpoints, Facies, FaciesInput, FaciesParams, FaciesResult, Method,
    RollupClassifier, Rule, Var,
};
