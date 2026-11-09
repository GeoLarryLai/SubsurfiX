// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Experimental isothermal multiphase FEM element residual, not a time integrator.
//!
//! Reuses the constitutive operators of the FIM-OBL solver, not its FVM fluxes.
//! Integrates the continuous Galerkin weak component balances on a trilinear
//! hexahedron, with consistent (not lumped) accumulation. Phase pressures are
//! interpolated from their nodal values; density and component mobilities are
//! evaluated at Gauss points. Depth is positive down.
//!
//! This is an unstabilized assembly building block. There is no global Newton
//! solve, bounded transport stabilization, well coupling or fault mesh cutting
//! here. It must NOT be advertised as a working multiphase FEM simulation path.
//! Thermal physics is rejected until its energy/conduction terms are assembled.

use super::hex8::{gauss_points, GaussPoint};
use crate::error::{SxError, SxResult};
use crate::flow::implicit::{OblPhysics, OpLayout};

fn invalid(s: &str) -> SxError {
    SxError::InvalidInput(format!("fem multiphase: {s}"))
}

/// Local backward-Euler residual (kg), node-major then component-major.
#[derive(Debug, Clone)]
pub struct ComponentBalance {
    pub residual: Vec<f64>,
    /// Integrated component masses (kg) before and after the step.
    pub mass_before: Vec<f64>,
    pub mass_after: Vec<f64>,
}

/// Cached geometry and elementwise material; no FVM transmissibilities used.
#[derive(Debug, Clone)]
pub struct ComponentElement {
    quadrature: [GaussPoint; 8],
    permeability: [f64; 3],
    porosity: f64,
}

/// Validate mass-fraction primary variables without silently clamping them.
/// This supports the existing isothermal operator convention [p, z_1, ...].
pub(super) fn layout(physics: &dyn OblPhysics) -> SxResult<OpLayout> {
    let l = physics.layout();
    if l.thermal || l.ne < 2 || l.np == 0 {
        return Err(invalid(
            "requires isothermal component physics with at least two equations",
        ));
    }
    Ok(l)
}

pub(super) fn state_ops(
    physics: &dyn OblPhysics,
    l: OpLayout,
    state: &[f64],
    out: &mut [f64],
) -> SxResult<()> {
    if state.len() != l.ne {
        return Err(SxError::Shape(
            "fem multiphase: state needs one entry per equation".into(),
        ));
    }
    if state.iter().any(|x| !x.is_finite())
        || state[0] <= 0.0
        || state[1..].iter().any(|&z| !(0.0..=1.0).contains(&z))
        || state[1..].iter().sum::<f64>() > 1.0
    {
        return Err(invalid(
            "pressure must be positive and component fractions must lie in the mass simplex",
        ));
    }
    // Poison the buffer so incomplete/invalid constitutive evaluations cannot
    // accidentally reuse values from another quadrature point.
    out.fill(f64::NAN);
    physics.evaluate(state, out);
    if out.iter().any(|x| !x.is_finite())
        || (0..l.ne).any(|c| out[l.alpha(c)] < 0.0)
        || (0..l.np).any(|p| out[l.rho(p)] <= 0.0 || out[l.lambda(p)] < 0.0)
        || (0..l.ne).any(|c| (0..l.np).any(|p| out[l.beta(c, p)] < 0.0))
    {
        return Err(invalid("non-finite or nonphysical constitutive operators"));
    }
    Ok(())
}

impl ComponentElement {
    /// Coordinates in (x, y, depth), diagonal world permeability (m²), and
    /// reference porosity. Eight-node order is di + 2*dj + 4*dk.
    pub fn new(corners: &[[f64; 3]; 8], permeability: [f64; 3], porosity: f64) -> SxResult<Self> {
        if corners.iter().flatten().any(|x| !x.is_finite())
            || permeability.iter().any(|&k| !k.is_finite() || k < 0.0)
            || !porosity.is_finite()
            || !(0.0..=1.0).contains(&porosity)
        {
            return Err(invalid("invalid coordinates, permeability or porosity"));
        }
        Ok(Self {
            quadrature: gauss_points(corners)?,
            permeability,
            porosity,
        })
    }

    /// Assemble ∫ N φ(α_new−α_old) + dt ∫ ∇N·K Σ β(∇p_phase−ρg∇depth)
    /// − dt ∫ N source. Source is component mass per bulk volume per second
    /// (kg/m³/s), constant within this element; positive means injection.
    ///
    /// State arrays are node-major, [p, independent component mass fractions]
    /// at each of eight nodes. No boundary terms are added: a caller must
    /// assemble internal/shared nodes and prescribe physical boundary fluxes.
    /// Summing nodal residuals gives mass_after − mass_before − dt*source*volume.
    pub fn residual(
        &self,
        physics: &dyn OblPhysics,
        old: &[f64],
        current: &[f64],
        dt: f64,
        gravity: f64,
        source: &[f64],
    ) -> SxResult<ComponentBalance> {
        let l = layout(physics)?;
        if old.len() != 8 * l.ne || current.len() != 8 * l.ne || source.len() != l.ne {
            return Err(SxError::Shape(
                "fem multiphase: eight nodal states and one source per component required".into(),
            ));
        }
        if !dt.is_finite()
            || dt <= 0.0
            || !gravity.is_finite()
            || gravity < 0.0
            || source.iter().any(|s| !s.is_finite())
        {
            return Err(invalid(
                "finite dt > 0, gravity >= 0 and component sources required",
            ));
        }
        let mut nodal = vec![vec![0.0; l.n_ops()]; 8];
        let mut scratch = vec![0.0; l.n_ops()];
        for a in 0..8 {
            state_ops(physics, l, &old[a * l.ne..(a + 1) * l.ne], &mut scratch)?;
            state_ops(
                physics,
                l,
                &current[a * l.ne..(a + 1) * l.ne],
                &mut nodal[a],
            )?;
        }
        let mut balance = ComponentBalance {
            residual: vec![0.0; 8 * l.ne],
            mass_before: vec![0.0; l.ne],
            mass_after: vec![0.0; l.ne],
        };
        let (mut old_q, mut new_q) = (vec![0.0; l.ne], vec![0.0; l.ne]);
        let (mut before, mut after) = (vec![0.0; l.n_ops()], vec![0.0; l.n_ops()]);
        let mut flux = vec![[0.0; 3]; l.ne];
        for q in &self.quadrature {
            // Preserve a uniform field exactly, and reduce cancellation from
            // absolute reservoir pressures in the phase-pressure gradients.
            for c in 0..l.ne {
                old_q[c] = old[c]
                    + (1..8)
                        .map(|a| q.n[a] * (old[a * l.ne + c] - old[c]))
                        .sum::<f64>();
                new_q[c] = current[c]
                    + (1..8)
                        .map(|a| q.n[a] * (current[a * l.ne + c] - current[c]))
                        .sum::<f64>();
            }
            state_ops(physics, l, &old_q, &mut before)?;
            state_ops(physics, l, &new_q, &mut after)?;
            flux.fill([0.0; 3]);
            for p in 0..l.np {
                let reference = current[0] - nodal[0][l.pc(p)];
                for d in 0..3 {
                    let grad = (0..8)
                        .map(|a| q.dn[a][d] * (current[a * l.ne] - nodal[a][l.pc(p)] - reference))
                        .sum::<f64>()
                        - if d == 2 {
                            after[l.rho(p)] * gravity
                        } else {
                            0.0
                        };
                    for c in 0..l.ne {
                        flux[c][d] += self.permeability[d] * after[l.beta(c, p)] * grad;
                    }
                }
            }
            for c in 0..l.ne {
                let b = self.porosity * before[l.alpha(c)];
                let n = self.porosity * after[l.alpha(c)];
                balance.mass_before[c] += q.w * b;
                balance.mass_after[c] += q.w * n;
                for a in 0..8 {
                    balance.residual[a * l.ne + c] += q.w
                        * (q.n[a] * (n - b - dt * source[c])
                            + dt * (0..3).map(|d| q.dn[a][d] * flux[c][d]).sum::<f64>());
                }
            }
        }
        if balance
            .residual
            .iter()
            .chain(&balance.mass_before)
            .chain(&balance.mass_after)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("residual overflow"));
        }
        Ok(balance)
    }
}

/// Borrowed work item for cached element assembly. Nodal states and sources use
/// the same ordering/units as [`ComponentElement::residual`].
pub struct ComponentTask<'a> {
    pub element: &'a ComponentElement,
    pub old: &'a [f64],
    pub current: &'a [f64],
    pub source: &'a [f64],
}

/// Evaluate independent element residuals in deterministic input order. Large
/// batches use the existing scoped Rayon pool and respect its process cap;
/// small batches stay serial. Global nodal scatter/boundary conditions remain
/// the caller's job. There is no nonlinear solve or time advancement here.
pub fn assemble_components(
    tasks: &[ComponentTask<'_>],
    physics: &(dyn OblPhysics + Sync),
    dt: f64,
    gravity: f64,
) -> SxResult<Vec<ComponentBalance>> {
    use crate::flow::solver::{par_loops, with_threads};
    use rayon::prelude::*;
    layout(physics)?;
    if tasks.is_empty() {
        return Err(invalid("element batch must be nonempty"));
    }
    let evaluate = |task: &ComponentTask<'_>| {
        task.element
            .residual(physics, task.old, task.current, dt, gravity, task.source)
    };
    with_threads(None, || {
        if par_loops(tasks.len().saturating_mul(8)) {
            tasks.par_iter().map(evaluate).collect()
        } else {
            tasks.iter().map(evaluate).collect()
        }
    })
}
