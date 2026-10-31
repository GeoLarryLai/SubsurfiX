// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stub (not implemented): fully implicit (Newton) two-phase solver.
//! Returns `SxError::Model` until built; use `run_impes` meanwhile.

use crate::error::{SxError, SxResult};

/// Stub, not implemented: fully implicit (Newton) two-phase solve.
///
/// # Errors
/// Always `SxError::Model` ("not implemented").
pub fn run_fully_implicit() -> SxResult<()> {
    Err(SxError::Model(
        "fully implicit solver: not implemented (stub); use run_impes".into(),
    ))
}
