// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! The single error type for the engine.

/// Every fallible engine function returns this error.
///
/// Python maps each variant to a subclass of `subsurfix.errors.SubsurfixError`
/// with the same name (for example `InvalidInput` → `InvalidInputError`).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SxError {
    /// A value is outside its allowed range or violates a stated rule.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A required input has no elements.
    #[error("empty input: {0}")]
    EmptyInput(String),

    /// Array lengths or matrix shapes do not agree.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// A well, log, top, or file that was asked for does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// The project configuration is invalid.
    #[error("config error: {0}")]
    Config(String),

    /// A file could not be read or written.
    #[error("io error at {path}: {source}")]
    Io {
        /// File path involved.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A file was read but its content does not follow the expected format.
    #[error("format error: {0}")]
    Format(String),

    /// A numerical routine failed (singular system, no convergence, overflow).
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// The gradient-boosted tree library or a plug-in model failed.
    #[error("model error: {0}")]
    Model(String),

    /// The problem has no solution under the given constraints
    /// (for example a DTW band that admits no path).
    #[error("infeasible: {0}")]
    Infeasible(String),
}

/// Result alias used across the engine.
pub type SxResult<T> = Result<T, SxError>;
