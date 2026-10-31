// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Plug-in model factory for the fill stage.

use super::ModelOverrides;
use crate::error::SxResult;
use crate::fill::sequential::{FillModel, LgbmFactory, RegressorFactory};
use crate::types::LogKind;

/// The plug-in model for an overridden log, else the built-in LightGBM model.
pub(crate) struct OverridingFactory<'a> {
    /// Built-in factory (`[fill.lightgbm]`, project seed).
    pub(crate) base: LgbmFactory,
    /// Per-log plug-ins.
    pub(crate) overrides: &'a ModelOverrides,
}

impl RegressorFactory for OverridingFactory<'_> {
    fn make(&self, log: LogKind) -> SxResult<Box<dyn FillModel>> {
        match self.overrides.get(&log) {
            Some((_, factory)) => factory.make(log),
            None => self.base.make(log),
        }
    }
}
