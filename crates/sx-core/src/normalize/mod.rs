// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stage 4: the consistency jury and the other checks
//! that need alignment, regional references, a two-point transform of the
//! logs in `normalize.logs` (GR, NPHI, PEF), and the post-normalization
//! checks.
//!
//! All work is in native space.
//! RHOB, DTC and RESD are never two-point normalized. QC only blanks or
//! flags; the transform is the only value change.

pub mod reference;
pub mod stage;
pub mod transform;

pub use reference::{
    fleet_pair, idw_mean, idw_pair, layer_bin, Anchor, DepthReason, FleetSource, RefPair,
    RefSource, Scope, Voter, VoterExclusion,
};
pub use stage::{
    logs_norm_batch, norm_anchors_batch, norm_report_batch, run_normalize_stage, AnchorRow,
    NormChannelAction, NormChannelRow, NormalizeInput, NormalizeOutput, PriorChannel,
    LOGS_NORM_TAG, NORM_ANCHORS_TAG, NORM_REPORT_TAG, QC_FLAGS_POST_TAG,
};
pub use transform::{gain_offset, transform_channel, GainOffset};
