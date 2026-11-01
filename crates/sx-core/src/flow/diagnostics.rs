// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//
// Derived from MRST, The MATLAB Reservoir Simulation Toolbox (module
// `diagnostics`: computeTimeOfFlight.m, computeTOFandTracer.m,
// computeWellPairs.m, computeFandPhi.m, computeLorenz.m, computeSweep.m).
// Copyright 2009-2026 SINTEF Digital, Mathematics & Cybernetics.
// MRST is GPL-3.0-or-later; this derived code is distributed here under
// AGPL-3.0-or-later (GPLv3 section 13).
//! Flow diagnostics on a converged flux field (Shahvali et al. 2012;
//! Møyner, Krogstad & Lie 2015; MRST `diagnostics` module).
//!
//! * Forward time of flight (TOF) from the injectors and backward TOF to the
//!   producers: upwind finite-volume solve of `v·∇τ = φ` per cell,
//!   `(Σ inflow + q⁺) τ_i − Σ_j inflow_ji τ_j = PV_i` (one sparse LU).
//! * Injector and producer tracer partitions: the same upwind operator with
//!   unit concentration at each well's completions; each cell's tracers sum
//!   to one where all inflow comes from wells.
//! * Well-pair pore volumes and allocation (flux of each injector reaching
//!   each producer, and the reverse).
//! * F–Φ (flow capacity vs storage capacity), Lorenz coefficient and sweep
//!   efficiency vs dimensionless time, from the cell total travel time
//!   `τ_f + τ_b` under incompressible flow.
//!
//! The flux field is any per-connection flux with per-cell sources, e.g. a
//! [`PressureSolution`] (single phase, or total mobility frozen from a
//! multiphase state). TOF is capped at `max_tof` (default
//! `50 Σ PV / Σ q⁺`, as in MRST); cells that would exceed it (stagnant
//! regions) get `max_tof` and pass nothing downstream.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuRef, NumericLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, MatMut, Par};

use super::model::FlowModel;
use super::pressure::{FixedPressure, FlowWell, PressureSolution};
use crate::error::{SxError, SxResult};

/// Diagnostics options.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DiagnosticsOptions {
    /// TOF cap (s); `None` = `50 Σ PV / Σ q⁺`.
    pub max_tof: Option<f64>,
}

/// One injector–producer pair.
#[derive(Debug, Clone, PartialEq)]
pub struct WellPair {
    /// Index of the injector in the well list.
    pub injector: usize,
    /// Index of the producer in the well list.
    pub producer: usize,
    /// Pore volume swept by the pair, `Σ c_inj c_prod PV` (m³).
    pub pore_volume: f64,
    /// Injector flux that reaches the producer (m³/s, >= 0).
    pub injector_allocation: f64,
    /// Producer flux that comes from the injector (m³/s, >= 0).
    pub producer_allocation: f64,
}

/// Flow diagnostics result. Inactive cells hold NaN (TOF, tracers) and −1
/// (partitions).
#[derive(Debug, Clone, PartialEq)]
pub struct FlowDiagnostics {
    /// Forward TOF from the injectors (s).
    pub tof_forward: Vec<f64>,
    /// Backward TOF to the producers (s).
    pub tof_backward: Vec<f64>,
    /// Indices (in the well list) of the injectors (net rate > 0).
    pub injectors: Vec<usize>,
    /// Indices of the producers (net rate < 0).
    pub producers: Vec<usize>,
    /// Injector tracer per injector, per cell.
    pub injector_tracer: Vec<Vec<f64>>,
    /// Producer tracer per producer, per cell.
    pub producer_tracer: Vec<Vec<f64>>,
    /// Dominant injector per cell (position in `injectors`; −1 none).
    pub injector_partition: Vec<i64>,
    /// Dominant producer per cell (position in `producers`; −1 none).
    pub producer_partition: Vec<i64>,
    /// All injector–producer pairs.
    pub well_pairs: Vec<WellPair>,
    /// TOF cap used (s).
    pub max_tof: f64,
}

fn numerical(msg: &str) -> SxError {
    SxError::Numerical(format!("flow diagnostics: {msg}"))
}

/// Per-cell net source (m³/s, + into the reservoir) from wells and
/// fixed-pressure entries.
fn cell_sources(model: &FlowModel, sol: &PressureSolution, wells: &[FlowWell], fixed: &[FixedPressure]) -> Vec<f64> {
    let mut q = vec![0.0; model.n_cells()];
    for (w, r) in wells.iter().zip(&sol.wells) {
        for (&c, &v) in w.cells.iter().zip(&r.cell_rates) {
            q[c] += v;
        }
    }
    for (f, &v) in fixed.iter().zip(&sol.boundary_flux) {
        q[f.cell] += v;
    }
    q
}

/// Upwind TOF / tracer operator for one direction; solves `A [τ C] = [PV R]`.
struct Upwind {
    lu: (faer::sparse::linalg::lu::SymbolicLu<usize>, NumericLu<usize, f64>),
    pv: Vec<f64>,
    qp: Vec<f64>,
    n: usize,
}

impl Upwind {
    fn build(model: &FlowModel, flux: &[f64], q: &[f64], reverse: bool, max_tof: f64) -> SxResult<Self> {
        let n = model.n_cells();
        let s = if reverse { -1.0 } else { 1.0 };
        let qp: Vec<f64> = q.iter().map(|&v| (s * v).max(0.0)).collect();
        // in = flux a->b (enters b), out = flux b->a (enters a), as MRST.
        let mut inflow = qp.clone();
        for (c, &f) in model.connections.iter().zip(flux) {
            let f = s * f;
            if f > 0.0 {
                inflow[c.b] += f;
            } else {
                inflow[c.a] -= f;
            }
        }
        let mut pv = model.pore_volume.clone();
        let max_in = inflow.iter().copied().fold(0.0, f64::max);
        if max_in <= 0.0 {
            return Err(numerical("no flow (all fluxes and sources are zero)"));
        }
        let mut d = inflow;
        let mut above = vec![false; n];
        for i in 0..n {
            if !model.active[i] {
                d[i] = 1.0;
                pv[i] = 0.0;
                continue;
            }
            if pv[i] / d[i].max(f64::EPSILON * max_in) > max_tof {
                above[i] = true;
                d[i] = max_in;
                pv[i] = max_in * max_tof;
            }
        }
        let mut trip: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(n + model.connections.len());
        trip.extend((0..n).map(|i| Triplet::new(i, i, d[i])));
        for (c, &f) in model.connections.iter().zip(flux) {
            let f = s * f;
            if f > 0.0 && !above[c.b] {
                trip.push(Triplet::new(c.b, c.a, -f));
            } else if f < 0.0 && !above[c.a] {
                trip.push(Triplet::new(c.a, c.b, f));
            }
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trip)
            .map_err(|e| numerical(&format!("matrix assembly failed: {e:?}")))?;
        let symbolic = factorize_symbolic_lu(a.symbolic(), Default::default())
            .map_err(|e| numerical(&format!("symbolic LU failed: {e:?}")))?;
        let mut numeric = NumericLu::new();
        let mut mem = MemBuffer::try_new(symbolic.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default()))
            .map_err(|e| numerical(&format!("cannot allocate workspace: {e:?}")))?;
        symbolic
            .factorize_numeric_lu::<f64>(&mut numeric, a.as_ref(), Par::Seq, MemStack::new(&mut mem), Default::default())
            .map_err(|e| numerical(&format!("LU failed: {e:?}")))?;
        Ok(Self { lu: (symbolic, numeric), pv, qp, n })
    }

    fn solve(&self, rhs: &mut [f64]) -> SxResult<()> {
        let mut mem = MemBuffer::try_new(StackReq::any_of(&[self.lu.0.solve_in_place_scratch::<f64>(1, Par::Seq)]))
            .map_err(|e| numerical(&format!("cannot allocate workspace: {e:?}")))?;
        let m = MatMut::from_column_major_slice_mut(rhs, self.n, 1);
        LuRef::new_unchecked(&self.lu.0, &self.lu.1).solve_in_place_with_conj(Conj::No, m, Par::Seq, MemStack::new(&mut mem));
        if rhs.iter().any(|v| !v.is_finite()) {
            return Err(numerical("solution is not finite"));
        }
        Ok(())
    }

    fn tof(&self, model: &FlowModel, max_tof: f64) -> SxResult<Vec<f64>> {
        let mut t = self.pv.clone();
        self.solve(&mut t)?;
        Ok(t.iter().enumerate().map(|(i, &v)| if model.active[i] { v.min(max_tof) } else { f64::NAN }).collect())
    }

    fn tracer(&self, model: &FlowModel, cells: &[usize]) -> SxResult<Vec<f64>> {
        let mut r = vec![0.0; self.n];
        for &c in cells {
            r[c] = self.qp[c];
        }
        self.solve(&mut r)?;
        Ok(r.iter().enumerate().map(|(i, &v)| if model.active[i] { v } else { f64::NAN }).collect())
    }
}

fn partition(model: &FlowModel, tr: &[Vec<f64>]) -> Vec<i64> {
    (0..model.n_cells())
        .map(|c| {
            if !model.active[c] {
                return -1;
            }
            let mut best = (-1i64, 0.0);
            for (k, t) in tr.iter().enumerate() {
                if t[c] > best.1 {
                    best = (k as i64, t[c]);
                }
            }
            best.0
        })
        .collect()
}

/// Flow diagnostics of a solved flux field (`sol` from
/// [`solve_pressure`](super::solve_pressure) with the same `wells` and
/// `fixed`). Injectors and producers are the wells with net rate > 0 / < 0.
///
/// # Errors
/// `Shape` for inconsistent lengths, `InvalidInput` for a bad `max_tof`,
/// `Numerical` when there is no flow or the solve fails.
pub fn flow_diagnostics(
    model: &FlowModel,
    sol: &PressureSolution,
    wells: &[FlowWell],
    fixed: &[FixedPressure],
    opts: &DiagnosticsOptions,
) -> SxResult<FlowDiagnostics> {
    if sol.flux.len() != model.connections.len() || sol.wells.len() != wells.len() || sol.boundary_flux.len() != fixed.len() {
        return Err(SxError::Shape("flow diagnostics: solution does not match the model, wells or fixed entries".into()));
    }
    for (w, r) in wells.iter().zip(&sol.wells) {
        if w.cells.len() != r.cell_rates.len() {
            return Err(SxError::Shape(format!("flow diagnostics: well {} cell rates do not match its cells", w.name)));
        }
    }
    let q = cell_sources(model, sol, wells, fixed);
    let max_tof = match opts.max_tof {
        Some(t) if t.is_finite() && t > 0.0 => t,
        Some(_) => return Err(SxError::InvalidInput("flow diagnostics: max_tof must be finite and > 0".into())),
        None => {
            let pv: f64 = (0..model.n_cells()).filter(|&i| model.active[i]).map(|i| model.pore_volume[i]).sum();
            let qin: f64 = q.iter().map(|&v| v.max(0.0)).sum();
            if qin <= 0.0 {
                return Err(numerical("no inflow (no injector or inflow boundary)"));
            }
            50.0 * pv / qin
        }
    };
    let fwd = Upwind::build(model, &sol.flux, &q, false, max_tof)?;
    let bwd = Upwind::build(model, &sol.flux, &q, true, max_tof)?;
    let injectors: Vec<usize> = (0..wells.len()).filter(|&w| sol.wells[w].rate > 0.0).collect();
    let producers: Vec<usize> = (0..wells.len()).filter(|&w| sol.wells[w].rate < 0.0).collect();
    let injector_tracer = injectors.iter().map(|&w| fwd.tracer(model, &wells[w].cells)).collect::<SxResult<Vec<_>>>()?;
    let producer_tracer = producers.iter().map(|&w| bwd.tracer(model, &wells[w].cells)).collect::<SxResult<Vec<_>>>()?;
    let mut well_pairs = Vec::with_capacity(injectors.len() * producers.len());
    for (pi, &p) in producers.iter().enumerate() {
        for (ii, &i) in injectors.iter().enumerate() {
            let pore_volume = (0..model.n_cells())
                .filter(|&c| model.active[c])
                .map(|c| injector_tracer[ii][c] * producer_tracer[pi][c] * model.pore_volume[c])
                .sum();
            let alloc = |w: usize, tr: &[f64]| -> f64 {
                wells[w].cells.iter().zip(&sol.wells[w].cell_rates).map(|(&c, &r)| r.abs() * tr[c]).sum()
            };
            well_pairs.push(WellPair {
                injector: i,
                producer: p,
                pore_volume,
                injector_allocation: alloc(i, &producer_tracer[pi]),
                producer_allocation: alloc(p, &injector_tracer[ii]),
            });
        }
    }
    Ok(FlowDiagnostics {
        tof_forward: fwd.tof(model, max_tof)?,
        tof_backward: bwd.tof(model, max_tof)?,
        injector_partition: partition(model, &injector_tracer),
        producer_partition: partition(model, &producer_tracer),
        injectors,
        producers,
        injector_tracer,
        producer_tracer,
        well_pairs,
        max_tof,
    })
}

/// F–Φ curve from pore volumes and forward / backward TOF (cells with a
/// non-finite value are skipped). Returns `(F, Φ)`, both starting at 0 and
/// ending at 1.
///
/// # Errors
/// `Shape` for unequal lengths, `EmptyInput` when no cell is usable.
pub fn f_phi(pv: &[f64], tof_forward: &[f64], tof_backward: &[f64]) -> SxResult<(Vec<f64>, Vec<f64>)> {
    if pv.len() != tof_forward.len() || pv.len() != tof_backward.len() {
        return Err(SxError::Shape("f_phi: pv and TOF lengths differ".into()));
    }
    let mut cells: Vec<(f64, f64)> = (0..pv.len())
        .map(|i| (tof_forward[i] + tof_backward[i], pv[i]))
        .filter(|(t, v)| t.is_finite() && *t > 0.0 && v.is_finite() && *v > 0.0)
        .collect();
    if cells.is_empty() {
        return Err(SxError::EmptyInput("f_phi: no cell with finite TOF and pore volume".into()));
    }
    cells.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut phi, mut f) = (vec![0.0], vec![0.0]);
    let (mut sv, mut sf) = (0.0, 0.0);
    for &(t, v) in &cells {
        sv += v;
        sf += v / t;
        phi.push(sv);
        f.push(sf);
    }
    phi.iter_mut().for_each(|x| *x /= sv);
    f.iter_mut().for_each(|x| *x /= sf);
    Ok((f, phi))
}

/// Lorenz coefficient `2 (∫F dΦ − 1/2)` (trapezoidal rule): 0 for a
/// homogeneous displacement, towards 1 for strong heterogeneity.
pub fn lorenz_coefficient(f: &[f64], phi: &[f64]) -> f64 {
    let s: f64 = (1..f.len().min(phi.len())).map(|i| 0.5 * (f[i - 1] + f[i]) * (phi[i] - phi[i - 1])).sum();
    2.0 * (s - 0.5)
}

/// Sweep efficiency `E_v` vs dimensionless time `t_D` (pore volumes
/// injected) from an F–Φ curve. Returns `(E_v, t_D)`.
pub fn sweep_efficiency(f: &[f64], phi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = f.len().min(phi.len());
    let tol = f64::EPSILON.sqrt();
    let keep: Vec<usize> = (0..n).filter(|&i| i == 0 || f[i - 1] <= f[i] - tol).collect();
    let (ff, pp): (Vec<f64>, Vec<f64>) = keep.iter().map(|&i| (f[i], phi[i])).unzip();
    let mut td = vec![0.0];
    for i in 1..ff.len() {
        td.push((pp[i] - pp[i - 1]) / (ff[i] - ff[i - 1]));
    }
    let ev = (0..ff.len()).map(|i| pp[i] + (1.0 - ff[i]) * td[i]).collect();
    (ev, td)
}
