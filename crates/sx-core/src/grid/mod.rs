// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! 3D property-modeling helpers.
//!
//! - [`RegularGrid`]: a regular (box-cell) 3D grid.
//! - [`CornerPointGrid`]: pillars + `ZCORN` (Eclipse layout), cell volumes and
//!   centres, point location, GRDECL read/write (limitations in the module docs).
//! - [`upscale`]: well-log samples to grid cells (arithmetic, harmonic,
//!   geometric mean; facies majority).
//! - [`variogram`]: experimental semivariogram and model fitting
//!   (spherical, exponential, Gaussian).
//! - [`interp`]: inverse distance weighting (IDW), ordinary kriging (OK),
//!   universal kriging (UK) with linear/quadratic drift, and trend fitting.
//!
//! Units: x, y, z in one length unit (the caller's; examples use metres or
//! feet consistently). z grows downward (depth). Missing values are NaN.

pub mod cornerpoint;
pub mod interp;
pub mod regular;
pub mod upscale;
pub mod variogram;

pub use cornerpoint::CornerPointGrid;
pub use interp::{
    calibrate_sill, fit_trend, idw, kriging_loo, ordinary_kriging, universal_kriging, Drift,
    IdwParams, KrigingParams, KrigingResult, Trend,
};
pub use regular::RegularGrid;
pub use upscale::{upscale_continuous, upscale_facies, Averaging, CellLocator, CellValues};
pub use variogram::{
    experimental_variogram, fit_variogram, ModelKind, VariogramBin, VariogramModel,
};
