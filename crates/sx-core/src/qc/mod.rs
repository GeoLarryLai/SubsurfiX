// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stage 2 (`qc`): per-channel QC rules as pure functions, the stage runner,
//! and the stage-4 checks that calls.
//!
//! QC never invents values. A rule may only set finite samples to NaN (per
//! sample or whole channel) or raise a flag. Value changes: unit rescue
//! multiplies a whole channel by one fixed factor; the
//! closed-caliper rescue ([`closed_cal`]) shifts kept closed-caliper RHOB by a
//! per-well bias and lists every shifted sample with its raw value in
//! `rhob_closed_caliper.parquet` (raw stays in `logs_raw`).
//!
//! Re-exports are explicit (no globs) so later modules such as the jury
//! can be added without name clashes.

use std::collections::BTreeMap;

use crate::types::{GridRange, LogKind, WellId};
use crate::units;

pub mod channel;
pub mod closed_cal;
pub mod divergence;
pub mod jury;
pub mod masks;
pub mod rescue;
pub mod ring;
pub mod stage;
pub mod stats;

/// Bit 0: linear value outside the physical bounds.
pub const QC_OUT_OF_BOUNDS: u8 = 1;
/// Bit 1: tail artifact (percentile limits plus pad).
pub const QC_TAIL: u8 = 2;
/// Bit 2: exact or near-exact repeat (flat run).
pub const QC_REPEAT: u8 = 4;
/// Bit 3: soft plateau.
pub const QC_PLATEAU: u8 = 8;
/// Bit 4: finite run shorter than the minimum run length.
pub const QC_SHORT_RUN: u8 = 16;
/// Bit 5: whole channel blanked (the reason is in the report).
pub const QC_CHANNEL: u8 = 32;
/// Bit 6: jury segment. Set only by the consistency jury.
pub const QC_JURY_SEGMENT: u8 = 64;
/// Bit 7: saturation / clipping pileup.
pub const QC_PILEUP: u8 = 128;

pub use channel::{gr_low_span, resd_decade_off};
pub use divergence::{jsd_scores, kl_knn_1d, DivRow};
pub use jury::{
    apply_jury, judge_channel, jury_reference, run_jury, Judgement, JuryCounts, JuryInput,
    JuryReference, JuryResult, JurySegment, JuryVerdict,
};
pub use masks::{
    apply_mask, badhole_mask, blank_channel, bounds_mask, closed_caliper_mask, constant_channel,
    pileup_mask, plateau_mask, range_mask, repeat_mask, short_run_mask, tail_mask, BadHoleParams,
    PlateauParams,
};
pub use rescue::{apply_rescue, plan_rescue, RescueReason, RescueRow};
pub use ring::{
    ring_test, Baseline, RingRow, SIGMA_CLUSTER, SIGMA_MEAN, SIGMA_P_HIGH, SIGMA_P_LOW, SIGMA_UNIT,
};
pub use stage::{
    closed_cal_runs_batch, closed_cal_samples_batch, flags_batch, logs_batch, low_density_batch,
    qc_report_batch, run_qc_stage, ChannelAction, ClosedCalRun, ClosedCalSample, QcChannelRow,
    QcInput, QcStageOutput, LOGS_QC_TAG, QC_FLAGS_TAG, QC_REPORT_TAG, RHOB_CLOSED_CALIPER_TAG,
};
pub use stats::{
    channel_stats, fleet_summary, low_variance_wells, post_flat_wells, ChannelStats, FleetSummary,
};

/// Per-sample QC bits of one well, one `Vec<u8>` per log, all of length
/// `grid.len`.
///
/// Every log in [`LogKind::ALL`] has an entry (all zeros if the log is
/// absent). Each sample carries at most one bit: the bit of the rule that
/// blanked it.
#[derive(Debug, Clone, PartialEq)]
pub struct QcFlags {
    /// Well id.
    pub id: WellId,
    /// Grid range covered.
    pub grid: GridRange,
    /// Bits per log.
    pub bits: BTreeMap<LogKind, Vec<u8>>,
}

/// Value in the log's native space: `log10(v)` for RESD (NaN if `v <= 0` or
/// not finite), else `v as f64`.
///
/// Unit: log10(ohm·m) for RESD, else the log's linear unit. NaN rule:
/// propagate (NaN in, NaN out). Errors: none.
pub fn native_value(log: LogKind, v: f32) -> f64 {
    if units::is_log10(log) {
        if v.is_finite() && v > 0.0 {
            f64::from(v).log10()
        } else {
            f64::NAN
        }
    } else {
        f64::from(v)
    }
}

/// Physical bounds in native space: the bounds, `log10` of both ends
/// for RESD (`-2.0` to `4.698970004336019`).
///
/// Unit: as [`native_value`]. Not a sample. Errors: none.
pub fn native_bounds(log: LogKind) -> (f64, f64) {
    let b = units::working_bounds(log);
    (b.lo, b.hi)
}

/// Native values of a linear `f32` channel (see [`native_value`]).
pub(crate) fn native_vec(log: LogKind, values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| native_value(log, v)).collect()
}
