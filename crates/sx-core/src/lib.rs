// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! SubsurfiX engine core.

// `!(x > 0.0)` is used on purpose: it is true for NaN, so one test rejects
// both bad values and NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod align;
pub mod config;
pub mod correlate;
pub mod dtw;
pub mod error;
pub mod fill;
pub mod flow;
pub mod geometry;
pub mod grid;
pub mod hash;
pub mod ingest;
pub mod io;
pub mod neighbors;
pub mod normalize;
pub mod pipeline;
pub mod qc;
pub mod refresh;
pub mod runtime;
pub mod sim;
pub mod types;
pub mod units;
pub mod util;

pub use error::{SxError, SxResult};
