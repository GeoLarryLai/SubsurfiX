// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! **Experimental** continuous Galerkin finite-element solver (FEM), OpenGeoSys-style
//! alongside FVM solvers.
//!
//! * [`HexMesh`]: node-based hexahedral mesh from the cell corners (shared
//!   nodes, the merge rule of the OpenGeoSys `.vtu` export).
//! * [`hex8`]: trilinear 8-node hexahedron, 2 × 2 × 2 Gauss points.
//! * [`run_fem_ht`]: water flow + heat transport (OpenGeoSys-style FEM) on that
//!   mesh, with the water, rock and well models of
//!   [`run_fim`](crate::flow::run_fim) / [`ThermalWater`](crate::flow::ThermalWater)
//!   so the two solvers compare on the same [`FlowModel`](crate::flow::FlowModel).

pub mod hex8;
/// Isothermal component-balance assembly only; not a multiphase time integrator.
pub mod multiphase;
/// Paired-face resistive interface assembly only; no fault mesh cutting.
pub mod interface;
mod ht;
mod mesh;

pub use ht::{run_fem_ht, FemOptions, FemResult};
pub use mesh::HexMesh;
