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
//! * [`implicit`]: **experimental** fully implicit multiphase engine with
//!   operator-based linearization (OBL) ([`run_fim`]); physics in
//!   [`fim_physics`] ([`DeadOil`], [`BlackOil`], [`Co2Brine`], [`ThermalWater`]).
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
pub mod co2_brine;
mod heat;
pub mod fim_physics;
pub mod implicit;
mod model;
mod multiphase;
pub mod ogs;
mod pressure;
mod solver;
mod stepping;
pub mod water;
mod transport;

pub use fim_physics::{BlackOil, Co2Brine, Co2BrineState, DeadOil, PvtRow, RockThermal, ThermalWater};
pub use implicit::{run_fim, FimOptions, FimResult, OblPhysics, OblTable, OpLayout};
pub use heat::{run_heat, run_heat_with, FixedTemperature, HeatFlux, HeatInputs, HeatResult, ThermalProps};
pub use model::{Completion, Connection, FlowModel, FlowOptions, MULT_KEYWORDS};
pub use multiphase::{run_two_phase, run_two_phase_with, Capillary, Physics, TwoPhaseOptions, TwoPhaseResult, SE_MIN};
pub use pressure::{
    check_controls, peaceman_wi, peaceman_wi_segment, solve_pressure, solve_pressure_with, FixedPressure, FlowWell, PressureOptions, PressureSolution, UPWIND_MAX_ITER,
    WellControl, WellKind, WellResult,
};
pub use solver::{num_threads, set_num_threads, with_threads, write_npy, NpyDirSink, ReportOptions, Snapshot, SnapshotSink};
pub use stepping::StepControl;
pub use transport::{run_impes, run_tracer, Corey, ImpesOptions, ImpesResult, TracerResult};
