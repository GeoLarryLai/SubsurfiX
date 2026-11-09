// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Experimental zero-thickness resistive interface quadrature, not a mesh cutter.
//!
//! Two coincident planar faces have independent nodal primary variables, with
//! paired vertices in the same physical order (di + 2*dj). Phase-pressure jumps
//! drive conservative transfer using upstream component mobilities. This kernel
//! does not split elements, duplicate nodes or modify run_fem_ht; that solver's
//! rejection of fault/nonmatching interfaces remains in force.

use super::multiphase::{layout, state_ops};
use crate::error::{SxError, SxResult};
use crate::flow::implicit::OblPhysics;

fn invalid(s: &str) -> SxError {
    SxError::InvalidInput(format!("fem interface: {s}"))
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|d| a[d] - b[d])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|d| a[d] * b[d]).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[derive(Debug, Clone, Copy)]
struct FacePoint {
    n: [f64; 4],
    area: f64,
}

/// Contributions (kg) to the left/right residual; positive transfer is left to right.
#[derive(Debug, Clone)]
pub struct InterfaceBalance {
    pub left: Vec<f64>,
    pub right: Vec<f64>,
    pub transfer: Vec<f64>,
}

/// Shared planar quadrilateral geometry with hydraulic conductance k_normal/t
/// (m). Zero conductance seals the interface. Infinite conductance is not
/// supported; a perfectly bonded interface instead shares nodal unknowns.
#[derive(Debug, Clone)]
pub struct ResistiveInterface {
    points: [FacePoint; 4],
    conductance: f64,
}

impl ResistiveInterface {
    /// Arbitrary strike/dip is allowed, but faces must already be coincident,
    /// conforming, nondegenerate, planar and paired by physical vertex. Opposite
    /// local face order must be permuted before passing the nodal states here.
    /// The caller must check both meshes' geometry and supply separate DOFs;
    /// this constructor receives only the one agreed physical face.
    pub fn new(x: &[[f64; 3]; 4], conductance: f64) -> SxResult<Self> {
        if x.iter().flatten().any(|v| !v.is_finite())
            || !conductance.is_finite()
            || conductance < 0.0
        {
            return Err(invalid("finite coordinates and conductance >= 0 required"));
        }
        let scale = (1..4)
            .map(|a| dot(sub(x[a], x[0]), sub(x[a], x[0])).sqrt())
            .fold(0.0_f64, f64::max);
        let normal = cross(sub(x[1], x[0]), sub(x[2], x[0]));
        let norm = dot(normal, normal).sqrt();
        if !scale.is_finite()
            || !norm.is_finite()
            || norm <= 0.0
            || dot(normal, sub(x[3], x[0])).abs() > 1e-10 * norm * scale
        {
            return Err(invalid("degenerate or nonplanar face"));
        }
        let mut points = [FacePoint {
            n: [0.0; 4],
            area: 0.0,
        }; 4];
        let g = 1.0 / 3.0_f64.sqrt();
        for (i, q) in points.iter_mut().enumerate() {
            let u = if i & 1 == 0 { -g } else { g };
            let v = if i & 2 == 0 { -g } else { g };
            let mut du = [0.0; 3];
            let mut dv = [0.0; 3];
            for a in 0..4 {
                let su = if a & 1 == 0 { -1.0 } else { 1.0 };
                let sv = if a & 2 == 0 { -1.0 } else { 1.0 };
                q.n[a] = 0.25 * (1.0 + su * u) * (1.0 + sv * v);
                for d in 0..3 {
                    du[d] += 0.25 * su * (1.0 + sv * v) * (x[a][d] - x[0][d]);
                    dv[d] += 0.25 * sv * (1.0 + su * u) * (x[a][d] - x[0][d]);
                }
            }
            let area_vector = cross(du, dv);
            q.area = dot(area_vector, area_vector).sqrt();
            if !q.area.is_finite() || q.area <= 0.0 || dot(area_vector, normal) <= 0.0 {
                return Err(invalid("folded or degenerate face"));
            }
        }
        // Gauss points alone do not detect every concave/bow-tie face; require
        // the bilinear map to have a positive Jacobian at all four vertices.
        let cyclic = [0, 1, 3, 2];
        for i in 0..4 {
            let a = cyclic[i];
            let b = cyclic[(i + 1) % 4];
            let c = cyclic[(i + 2) % 4];
            if dot(cross(sub(x[b], x[a]), sub(x[c], x[b])), normal) <= 0.0 {
                return Err(invalid(
                    "face must be strictly convex and consistently ordered",
                ));
            }
        }
        Ok(Self {
            points,
            conductance,
        })
    }

    pub fn area(&self) -> f64 {
        self.points.iter().map(|q| q.area).sum()
    }

    /// Conservative isothermal component exchange on two matched four-node
    /// faces. Geometry is coincident, so the phase driving jump is p_left-Pc_left
    /// minus p_right-Pc_right; there is no depth difference across the interface.
    /// This neglects finite-thickness storage/tangential flow in the fault.
    pub fn residual(
        &self,
        physics: &dyn OblPhysics,
        left: &[f64],
        right: &[f64],
        dt: f64,
    ) -> SxResult<InterfaceBalance> {
        let l = layout(physics)?;
        if left.len() != 4 * l.ne || right.len() != 4 * l.ne {
            return Err(SxError::Shape(
                "fem interface: four nodal states required per side".into(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("finite dt > 0 required"));
        }
        let (mut lop, mut rop) = (vec![0.0; l.n_ops()], vec![0.0; l.n_ops()]);
        for a in 0..4 {
            state_ops(physics, l, &left[a * l.ne..(a + 1) * l.ne], &mut lop)?;
            state_ops(physics, l, &right[a * l.ne..(a + 1) * l.ne], &mut rop)?;
        }
        let (mut ls, mut rs) = (vec![0.0; l.ne], vec![0.0; l.ne]);
        let mut result = InterfaceBalance {
            left: vec![0.0; 4 * l.ne],
            right: vec![0.0; 4 * l.ne],
            transfer: vec![0.0; l.ne],
        };
        for q in &self.points {
            for c in 0..l.ne {
                ls[c] = left[c]
                    + (1..4)
                        .map(|a| q.n[a] * (left[a * l.ne + c] - left[c]))
                        .sum::<f64>();
                rs[c] = right[c]
                    + (1..4)
                        .map(|a| q.n[a] * (right[a * l.ne + c] - right[c]))
                        .sum::<f64>();
            }
            state_ops(physics, l, &ls, &mut lop)?;
            state_ops(physics, l, &rs, &mut rop)?;
            for p in 0..l.np {
                let jump = (ls[0] - rs[0]) - (lop[l.pc(p)] - rop[l.pc(p)]);
                let upstream = if jump >= 0.0 { &lop } else { &rop };
                for c in 0..l.ne {
                    let mass = dt * q.area * self.conductance * upstream[l.beta(c, p)] * jump;
                    result.transfer[c] += mass;
                    for a in 0..4 {
                        let term = q.n[a] * mass;
                        result.left[a * l.ne + c] += term;
                        result.right[a * l.ne + c] -= term;
                    }
                }
            }
        }
        if result
            .left
            .iter()
            .chain(&result.right)
            .chain(&result.transfer)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("interface residual overflow"));
        }
        Ok(result)
    }
}
