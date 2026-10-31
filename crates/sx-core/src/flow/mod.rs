// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Flow and heat simulation on the regular and corner-point grids.
//!
//! * [`FlowModel`]: cells, pore volumes and two-point flux approximation
//!   (TPFA) transmissibilities built from a [`RegularGrid`] or a
//!   [`CornerPointGrid`] with a diagonal permeability tensor.
//! * [`solve_pressure`]: incompressible pressure solve with Peaceman wells
//!   (injector or producer, rate or bottom-hole pressure (BHP) control) and
//!   fixed-pressure cells; sparse Cholesky (faer).
//! * [`run_tracer`]: explicit upwind tracer transport on a flux field.
//! * [`run_impes`]: two-phase water/oil, implicit pressure explicit
//!   saturation (IMPES), Corey relative permeabilities.
//! * [`run_two_phase`]: IMPES with gravity, capillary pressure
//!   (Brooks-Corey / van Genuchten), slightly compressible rock and fluids
//!   (transient pressure) and deviated wells
//!   ([`FlowModel::trajectory_completions`], Peaceman along the path).
//! * [`run_heat`]: heat transport after the pressure solve (constant
//!   properties, single-phase liquid, implicit upwind advection and
//!   conduction, sparse LU).
//! * [`ogs`]: export of the grid to an OpenGeoSys (OGS) `.vtu` mesh.
//! * [`implicit`]: fully implicit solver (stub, returns an error).
//! * [`benchmarks`]: SPE10/OPM Flow comparison (stub, returns an error).
//!
//! Units are SI throughout: m, m², m³, Pa, Pa·s, s, m³/s (reservoir volume).
//! Depth is positive down. Faults get overlap (non-neighbour) connections and pinch-outs
//! vertical connections across zero-thickness cells. Limits: [`solve_pressure`] / [`run_impes`] are
//! incompressible without gravity or capillary pressure (use
//! [`run_two_phase`] for those); fluid compressibility enters the water
//! balance through `b_w` only (oil is the complement); no capillary pressure
//! at wells or boundaries.

pub mod analytic;
pub mod benchmarks;
mod heat;
pub mod implicit;
mod model;
mod multiphase;
pub mod ogs;
mod pressure;
mod stepping;
pub mod water;
mod transport;

pub use heat::{run_heat, FixedTemperature, HeatFlux, HeatInputs, HeatResult, ThermalProps};
pub use model::{Completion, Connection, FlowModel, FlowOptions, MULT_KEYWORDS};
pub use multiphase::{run_two_phase, Capillary, Physics, TwoPhaseOptions, TwoPhaseResult, SE_MIN};
pub use pressure::{
    check_controls, peaceman_wi, peaceman_wi_segment, solve_pressure, solve_pressure_with, FixedPressure, FlowWell, PressureOptions, PressureSolution, UPWIND_MAX_ITER,
    WellControl, WellKind, WellResult,
};
pub use stepping::StepControl;
pub use transport::{run_impes, run_tracer, Corey, ImpesOptions, ImpesResult, TracerResult};
