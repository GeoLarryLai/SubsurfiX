// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Process exit codes for [`sx_core::SxError`].

use sx_core::SxError;

/// Map `e` to a process status.
///
/// `2` for [`SxError::Config`], [`SxError::InvalidInput`], and [`SxError::NotFound`].
/// `1` for every other variant (`SxError` is non-exhaustive). Success is not passed here.
pub fn exit_code(e: &SxError) -> u8 {
    match e {
        SxError::Config(_) | SxError::InvalidInput(_) | SxError::NotFound(_) => 2,
        _ => 1,
    }
}
