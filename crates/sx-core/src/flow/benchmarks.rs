// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stub (not implemented): comparison against the SPE10 (Society of
//! Petroleum Engineers 10th Comparative Solution Project) model 2 and
//! OPM (Open Porous Media) Flow reference results.

use crate::error::{SxError, SxResult};

/// Stub, not implemented: compare against SPE10 model 2 / OPM Flow
/// reference results.
///
/// # Errors
/// Always `SxError::Model` ("not implemented").
pub fn compare_spe10() -> SxResult<()> {
    Err(SxError::Model(
        "SPE10/OPM comparison: not implemented (stub)".into(),
    ))
}
