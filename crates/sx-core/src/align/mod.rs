// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Rough alignment.
//!
//! From user tops and QC'd curves this module builds, per well: one ordered
//! top list ([`crate::types::TopSet`]) with an upper and a lower pad,
//! estimated depths for tops the user did not pick, an interval table with a
//! state and a confidence per interval, a layer coordinate per sample
//! (`interval + fraction`), and a match map from each well to its nearest
//! wells. [`rough_align`] runs the stage; the four batch builders turn its
//! result into the `intervals`, `tops_estimated`, `pairs_rough` and
//! `layer_coord` tables.
//!
//! Depths are feet (MD unless named `elev_ft`), distances meters, missing
//! values NaN. The canonical order and the estimation work in elevation; the
//! break table, intervals, pads and the coordinate work in MD. Interval-mode
//! matching calls [`crate::dtw::interval_path`]; depth mode uses the DTW
//! building blocks on the elevation axis.

#![warn(missing_docs)]

mod breaks;
mod depth;
mod estimate;
mod intervals;
mod pairs;
mod params;
mod picks;
mod run;
mod tables;

pub use breaks::{build_breaks, coord_at_md, interval_of_coord, layer_coord, Breaks};
pub use depth::{elev_at_md, md_at_elev};
pub use intervals::{interval_table, IntervalInfo};
pub use pairs::{PairMap, PairStatus};
pub use params::{AlignParams, ConfidenceParams, PairParams, TopsParams};
pub use picks::{canonical_order, UserPick};
pub use run::{
    rough_align, AlignInput, BoundaryRow, BoundaryStatus, CoordSource, RoughAlignment, WellCoord,
};
pub use tables::{intervals_batch, layer_coord_batch, pairs_rough_batch, tops_estimated_batch};
