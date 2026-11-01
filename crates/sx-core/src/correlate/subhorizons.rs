// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Sub-horizon proposals from final layer time.
//!
//! Per non-pad interval: a regional median GR trace on a layer-time grid, a
//! zero-mean Ricker wavelet transform at several scales (FFT convolution),
//! zero crossings tracked across scales, and accepted crossings mapped to MD
//! in every taking-part well. Proposals are suggestions only: they never
//! change tops, boundaries or layer time.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::config::SubhorizonsConfig;
use crate::correlate::solve::{md_at_layer_time, LayerTimeWell};
use crate::correlate::tops::elev_at_md;
use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;
use crate::types::{
    IntervalRow, IntervalState, LogKind, TopSet, WellGeometry, WellHeader, WellId, WellLogs,
};
use crate::util::median;

const SUBHORIZONS_TAG: &str = "subsurfix.subhorizons/1";

/// One proposed sub-horizon (one layer-time value in one interval).
#[derive(Clone, Debug)]
pub struct SubhorizonProposal {
    /// 0-based id in (interval, layer time) order.
    pub id: u32,
    /// Interval index `i` (`1 ..= n − 1`).
    pub interval: u16,
    /// `"{upper top name}.sh{j}"`, `j` 1-based within the interval.
    pub name: String,
    /// Name of top `i − 1` (upper boundary of the interval).
    pub upper_top_name: String,
    /// Proposal layer time `u`.
    pub layer_time: f64,
    /// Step strength, API.
    pub strength_api: f64,
    /// +1 or −1.
    pub polarity: i8,
    /// Scales the crossing persists through.
    pub persistence: u8,
    /// Wells in the regional median at the proposal.
    pub n_wells: usize,
}

/// Depth of one proposal in one well.
#[derive(Clone, Debug)]
pub struct SubhorizonPick {
    /// Proposal id.
    pub id: u32,
    /// Well id.
    pub well: WellId,
    /// MD where the well's layer time equals the proposal, ft.
    pub md_ft: f64,
    /// Elevation at `md_ft`, ft (NaN outside the geometry).
    pub elev_ft: f64,
}

/// Sub-horizon proposals and their per-well depths.
#[derive(Clone, Debug)]
pub struct Subhorizons {
    /// Sorted by id.
    pub proposals: Vec<SubhorizonProposal>,
    /// Sorted by (id, well).
    pub picks: Vec<SubhorizonPick>,
}

/// Check the `[subhorizons]` keys ([`SxError::Config`]).
pub(crate) fn check_subhorizons_config(cfg: &SubhorizonsConfig) -> SxResult<()> {
    let bad =
        |key: &str, rule: &str| Err(SxError::Config(format!("subhorizons.{key} must be {rule}")));
    if cfg.n_grid < 2 {
        return bad("n_grid", ">= 2");
    }
    if cfg.min_wells < 1 {
        return bad("min_wells", ">= 1");
    }
    if cfg.scales.is_empty()
        || cfg.scales.iter().any(|&a| a < 1)
        || cfg.scales.windows(2).any(|w| matches!(w, [x, y] if x >= y))
    {
        return bad("scales", "non-empty, strictly ascending, each >= 1");
    }
    if cfg.min_persistence < 1 || cfg.min_persistence > cfg.scales.len() {
        return bad("min_persistence", "in 1 ..= len(scales)");
    }
    if cfg.link_tol_mult.is_nan() || cfg.link_tol_mult <= 0.0 {
        return bad("link_tol_mult", "> 0");
    }
    if cfg.min_strength_api.is_nan() || cfg.min_strength_api < 0.0 {
        return bad("min_strength_api", ">= 0");
    }
    Ok(())
}

/// Continuous Ricker wavelet `A (1 − t²/a²) exp(−t²/(2a²))`,
/// `A = 2 / (√(3a) π^(1/4))`; `t`, `a` in grid samples.
fn ricker(t: f64, a: f64) -> f64 {
    let amp = 2.0 / ((3.0 * a).sqrt() * std::f64::consts::PI.powf(0.25));
    let r = t * t / (a * a);
    amp * (1.0 - r) * (-0.5 * r).exp()
}

/// Zero-mean discrete Ricker kernel `ψ_d[t]`, `t = −h ..= h`,
/// `h = half_width_mult · a`. Center at index `h`.
///
/// Empty for `a = 0` or a half width that does not fit `i64`.
pub(crate) fn ricker_kernel(a: usize, half_width_mult: usize) -> Vec<f64> {
    let Some(h) = a
        .checked_mul(half_width_mult)
        .filter(|_| a > 0)
        .and_then(|h| i64::try_from(h).ok())
    else {
        return Vec::new();
    };
    let af = a as f64;
    let mut k: Vec<f64> = (-h..=h).map(|t| ricker(t as f64, af)).collect();
    let mean = k.iter().sum::<f64>() / k.len() as f64;
    for v in &mut k {
        *v -= mean;
    }
    k
}

/// Symmetric reflection without repeating the edge sample, `len ≥ 2`.
fn reflect(j: i64, len: usize) -> usize {
    let last = len as i64 - 1;
    let p = 2 * last;
    let mut r = j.checked_rem_euclid(p).unwrap_or(0);
    if r > last {
        r = p - r;
    }
    r as usize
}

/// FFT size for a segment of `len` samples and kernel half width `h`:
/// smallest power of two `≥ len + 4h`.
fn fft_size(len: usize, h: usize) -> SxResult<usize> {
    h.checked_mul(4)
        .and_then(|v| v.checked_add(len))
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| SxError::InvalidInput("wavelet transform: FFT size overflows".into()))
}

/// Forward and inverse plan of one FFT size.
type FftPair = (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>);

/// Forward and inverse FFT plans by size, all from one planner.
struct FftPlans {
    plans: BTreeMap<usize, FftPair>,
}

impl FftPlans {
    fn new(planner: &mut FftPlanner<f64>, sizes: &BTreeSet<usize>) -> Self {
        let plans = sizes
            .iter()
            .map(|&n| {
                (
                    n,
                    (planner.plan_fft_forward(n), planner.plan_fft_inverse(n)),
                )
            })
            .collect();
        Self { plans }
    }
}

/// `C[q] = Σ ψ_d[t] x_ext[q − t]` by FFT convolution.
fn cwt_planned(x: &[f64], kernel: &[f64], plans: &FftPlans) -> SxResult<Vec<f64>> {
    let len = x.len();
    if len < 2 {
        return Err(SxError::InvalidInput(
            "wavelet transform needs at least 2 samples".into(),
        ));
    }
    let h = kernel.len() / 2;
    let size = fft_size(len, h)?;
    let (fwd, inv) = plans.plans.get(&size).ok_or_else(|| {
        SxError::Numerical(format!("wavelet transform: no FFT plan of size {size}"))
    })?;
    let zero = Complex::new(0.0, 0.0);
    let mut sig = vec![zero; size];
    let h_i = h as i64;
    for (e, slot) in sig.iter_mut().take(len + 2 * h).enumerate() {
        let v = x
            .get(reflect(e as i64 - h_i, len))
            .copied()
            .unwrap_or(f64::NAN);
        *slot = Complex::new(v, 0.0);
    }
    let mut ker = vec![zero; size];
    for (slot, &v) in ker.iter_mut().zip(kernel) {
        *slot = Complex::new(v, 0.0);
    }
    fwd.process(&mut sig);
    fwd.process(&mut ker);
    for (s, k) in sig.iter_mut().zip(&ker) {
        *s *= *k;
    }
    inv.process(&mut sig);
    let n = size as f64;
    Ok((0..len)
        .map(|q| sig.get(q + 2 * h).map_or(f64::NAN, |c| c.re / n))
        .collect())
}

/// Kernels per scale and every FFT plan a segment of `2 ..= n_grid` samples needs.
struct Transform {
    kernels: Vec<Vec<f64>>,
    plans: FftPlans,
}

impl Transform {
    fn new(cfg: &SubhorizonsConfig) -> SxResult<Self> {
        let mut kernels = Vec::with_capacity(cfg.scales.len());
        let mut sizes = BTreeSet::new();
        for &a in &cfg.scales {
            let k = ricker_kernel(a, cfg.kernel_half_width_mult);
            if k.is_empty() {
                return Err(SxError::Config(format!(
                    "subhorizons: kernel of scale {a} is too large"
                )));
            }
            let h = k.len() / 2;
            let hi = fft_size(cfg.n_grid, h)?;
            let mut s = fft_size(2, h)?;
            while s <= hi {
                sizes.insert(s);
                s = s.checked_mul(2).ok_or_else(|| {
                    SxError::InvalidInput("wavelet transform: FFT size overflows".into())
                })?;
            }
            kernels.push(k);
        }
        let mut planner = FftPlanner::new();
        let plans = FftPlans::new(&mut planner, &sizes);
        Ok(Self { kernels, plans })
    }
}

/// One zero crossing of a transform.
#[derive(Clone, Copy, Debug)]
struct Crossing {
    /// Lower sample index.
    q: usize,
    /// Fractional position, samples.
    x: f64,
    polarity: i8,
}

/// Rising (`+1`) and falling (`−1`) zero crossings.
fn zero_crossings(c: &[f64]) -> Vec<Crossing> {
    c.windows(2)
        .enumerate()
        .filter_map(|(q, w)| {
            let &[c0, c1] = w else {
                return None;
            };
            let polarity = if c0 < 0.0 && c1 >= 0.0 {
                1
            } else if c0 > 0.0 && c1 <= 0.0 {
                -1
            } else {
                return None;
            };
            Some(Crossing {
                q,
                x: q as f64 + c0 / (c0 - c1),
                polarity,
            })
        })
        .collect()
}

/// Crossing index at each scale reached, finest first.
#[derive(Clone, Debug)]
struct Chain {
    polarity: i8,
    links: Vec<usize>,
}

/// Track chains from the finest scale upward.
fn track_chains(cross: &[Vec<Crossing>], scales: &[usize], link_tol_mult: f64) -> Vec<Chain> {
    let Some(first) = cross.first() else {
        return Vec::new();
    };
    let mut chains: Vec<Chain> = first
        .iter()
        .enumerate()
        .map(|(k, c)| Chain {
            polarity: c.polarity,
            links: vec![k],
        })
        .collect();
    for s in 0..cross.len().saturating_sub(1) {
        let (Some(here), Some(next), Some(&a_next)) =
            (cross.get(s), cross.get(s + 1), scales.get(s + 1))
        else {
            break;
        };
        let tol = link_tol_mult * a_next as f64;
        // (distance, chain position at s, crossing position at s + 1, chain, crossing)
        let mut cand: Vec<(f64, f64, f64, usize, usize)> = Vec::new();
        for (ci, ch) in chains.iter().enumerate() {
            if ch.links.len() != s + 1 {
                continue;
            }
            let Some(xc) = ch.links.last().and_then(|&k| here.get(k)).map(|c| c.x) else {
                continue;
            };
            for (ki, k) in next.iter().enumerate() {
                if k.polarity == ch.polarity {
                    let d = (k.x - xc).abs();
                    if d <= tol {
                        cand.push((d, xc, k.x, ci, ki));
                    }
                }
            }
        }
        cand.sort_by(|p, q| {
            p.0.total_cmp(&q.0)
                .then(p.1.total_cmp(&q.1))
                .then(p.2.total_cmp(&q.2))
                .then(p.3.cmp(&q.3))
                .then(p.4.cmp(&q.4))
        });
        let mut chain_used = vec![false; chains.len()];
        let mut cross_used = vec![false; next.len()];
        for &(_, _, _, ci, ki) in &cand {
            let (Some(cu), Some(ku)) = (chain_used.get(ci), cross_used.get(ki)) else {
                continue;
            };
            if *cu || *ku {
                continue;
            }
            if let Some(ch) = chains.get_mut(ci) {
                ch.links.push(ki);
            }
            if let Some(cu) = chain_used.get_mut(ci) {
                *cu = true;
            }
            if let Some(ku) = cross_used.get_mut(ki) {
                *ku = true;
            }
        }
    }
    chains
}

/// One accepted proposal of an interval before ids.
#[derive(Clone, Copy, Debug)]
struct Found {
    layer_time: f64,
    strength_api: f64,
    polarity: i8,
    persistence: usize,
    n_wells: usize,
}

/// Proposals of interval `i` from its regional trace (-10),
/// ordered by layer time. `r[g]` is NaN where the trace is invalid.
fn trace_proposals(
    i: usize,
    r: &[f64],
    n_wells: &[usize],
    tr: &Transform,
    cfg: &SubhorizonsConfig,
) -> SxResult<Vec<Found>> {
    let (Some(g0), Some(g1)) = (
        r.iter().position(|v| v.is_finite()),
        r.iter().rposition(|v| v.is_finite()),
    ) else {
        return Ok(Vec::new());
    };
    let len = g1 - g0 + 1;
    if len < cfg.min_segment_samples || len < 2 {
        return Ok(Vec::new());
    }
    let mut seg: Vec<f64> = r.get(g0..=g1).map_or_else(Vec::new, <[f64]>::to_vec);
    let mut prev: Option<usize> = None;
    for q in 0..len {
        if !seg.get(q).is_some_and(|v| v.is_finite()) {
            continue;
        }
        if let Some(p) = prev.filter(|&p| q > p + 1) {
            let (vp, vq) = (seg[p], seg[q]);
            for (m, v) in seg.iter_mut().enumerate().take(q).skip(p + 1) {
                *v = vp + (vq - vp) * (m - p) as f64 / (q - p) as f64;
            }
        }
        prev = Some(q);
    }
    let mean = seg.iter().sum::<f64>() / len as f64;
    let x: Vec<f64> = seg.iter().map(|v| v - mean).collect();

    let mut coeffs: Vec<Vec<f64>> = Vec::with_capacity(tr.kernels.len());
    let mut cross: Vec<Vec<Crossing>> = Vec::with_capacity(tr.kernels.len());
    for k in &tr.kernels {
        let c = cwt_planned(&x, k, &tr.plans)?;
        cross.push(zero_crossings(&c));
        coeffs.push(c);
    }
    let chains = track_chains(&cross, &cfg.scales, cfg.link_tol_mult);

    let s_ref = cfg.min_persistence.saturating_sub(1);
    let (Some(c_ref), Some(k_ref)) = (coeffs.get(s_ref), tr.kernels.get(s_ref)) else {
        return Ok(Vec::new());
    };
    let psi0 = k_ref.get(k_ref.len() / 2).copied().unwrap_or(f64::NAN);
    let lo = cfg.edge_gap_samples as f64;
    let hi = (len - 1) as f64 - cfg.edge_gap_samples as f64;
    // (finest position, polarity, persistence, strength)
    let mut cands: Vec<(f64, i8, usize, f64)> = Vec::new();
    for ch in &chains {
        let p = ch.links.len();
        if p < cfg.min_persistence {
            continue;
        }
        let (Some(fine), Some(at_ref)) = (
            ch.links
                .first()
                .and_then(|&k| cross.first().and_then(|c| c.get(k))),
            ch.links
                .get(s_ref)
                .and_then(|&k| cross.get(s_ref).and_then(|c| c.get(k))),
        ) else {
            continue;
        };
        let (Some(&c0), Some(&c1)) = (c_ref.get(at_ref.q), c_ref.get(at_ref.q + 1)) else {
            continue;
        };
        let strength = (c1 - c0).abs() / psi0;
        if strength >= cfg.min_strength_api && fine.x >= lo && fine.x <= hi {
            cands.push((fine.x, ch.polarity, p, strength));
        }
    }
    cands.sort_by(|a, b| {
        b.3.total_cmp(&a.3)
            .then(b.2.cmp(&a.2))
            .then(a.0.total_cmp(&b.0))
    });
    let spacing = cfg.min_spacing_samples as f64;
    let mut kept: Vec<(f64, i8, usize, f64)> = Vec::new();
    for c in cands {
        if kept.iter().all(|k| (c.0 - k.0).abs() >= spacing) {
            kept.push(c);
        }
    }
    let n_grid = cfg.n_grid as f64;
    let mut out: Vec<Found> = kept
        .into_iter()
        .map(|(x, polarity, persistence, strength_api)| {
            let g = g0 as f64 + x.round();
            Found {
                layer_time: i as f64 + (g0 as f64 + x + 0.5) / n_grid,
                strength_api,
                polarity,
                persistence,
                n_wells: n_wells.get(g as usize).copied().unwrap_or(0),
            }
        })
        .collect();
    out.sort_by(|a, b| a.layer_time.total_cmp(&b.layer_time));
    Ok(out)
}

/// Resampled GR at ascending layer times `us`; NaN where
/// not covered.
///
/// Relies on `tau` being non-decreasing over its finite samples (input
/// contract), so the first sample with `tau ≥ u` never moves up as `u` grows.
fn resample_gr(tau: &[f64], gr: &[f32], us: &[f64]) -> Vec<f64> {
    let n = tau.len().min(gr.len());
    let mut m = 0usize;
    us.iter()
        .map(|&u| {
            while m < n && !tau.get(m).is_some_and(|&t| t.is_finite() && t >= u) {
                m += 1;
            }
            gr_at(tau, gr, m, u)
        })
        .collect()
}

/// GR at layer time `u` given the first sample `m` with `tau[m] ≥ u`.
fn gr_at(tau: &[f64], gr: &[f32], m: usize, u: f64) -> f64 {
    let (Some(&tm), Some(&gm)) = (tau.get(m), gr.get(m)) else {
        return f64::NAN;
    };
    let gm = f64::from(gm);
    if tm == u {
        return gm;
    }
    let Some(prev) = m.checked_sub(1) else {
        return f64::NAN;
    };
    let (Some(&tp), Some(&gp)) = (tau.get(prev), gr.get(prev)) else {
        return f64::NAN;
    };
    if !tp.is_finite() {
        return f64::NAN;
    }
    let t = (u - tp) / (tm - tp);
    let gp = f64::from(gp);
    gp + t * (gm - gp)
}

/// Regional median trace: `(R, n_wells)`, `R[g]` NaN when
/// fewer than `min_wells` wells cover sample `g`.
fn regional_trace(cols: &[Vec<f64>], n_grid: usize, min_wells: usize) -> (Vec<f64>, Vec<usize>) {
    let mut r = Vec::with_capacity(n_grid);
    let mut counts = Vec::with_capacity(n_grid);
    for g in 0..n_grid {
        let vals: Vec<f64> = cols
            .iter()
            .filter_map(|c| c.get(g).copied().filter(|v| !v.is_nan()))
            .collect();
        counts.push(vals.len());
        r.push(if vals.len() >= min_wells {
            median(&vals)
        } else {
            f64::NAN
        });
    }
    (r, counts)
}

/// Inputs of one well.
struct WellIn<'a> {
    id: WellId,
    lt: &'a LayerTimeWell,
    gr: Option<&'a [f32]>,
    geo: &'a WellGeometry,
    states: &'a [IntervalState],
}

impl WellIn<'_> {
    /// Takes part in interval `i`: has GR and state `Observed` or `Estimated`.
    fn takes_part(&self, i: usize) -> bool {
        self.gr.is_some()
            && matches!(
                self.states.get(i),
                Some(IntervalState::Observed | IntervalState::Estimated)
            )
    }
}

type IntervalOut = Vec<(Found, Vec<(WellId, f64, f64)>)>;

/// Proposals of interval `i` with their covered wells `(well, md, elev)`;
/// proposals with no covered well are dropped.
fn interval_proposals(
    wells: &[WellIn<'_>],
    i: usize,
    tr: &Transform,
    cfg: &SubhorizonsConfig,
) -> SxResult<IntervalOut> {
    let n_grid = cfg.n_grid;
    let us: Vec<f64> = (0..n_grid)
        .map(|g| i as f64 + (g as f64 + 0.5) / n_grid as f64)
        .collect();
    let part: Vec<&WellIn<'_>> = wells.iter().filter(|w| w.takes_part(i)).collect();
    let cols: Vec<Vec<f64>> = part
        .iter()
        .map(|w| resample_gr(&w.lt.layer_time, w.gr.unwrap_or(&[]), &us))
        .collect();
    let (r, counts) = regional_trace(&cols, n_grid, cfg.min_wells);
    let found = trace_proposals(i, &r, &counts, tr, cfg)?;
    let mut out = Vec::with_capacity(found.len());
    for f in found {
        let picks: Vec<(WellId, f64, f64)> = part
            .iter()
            .filter_map(|w| {
                let md = md_at_layer_time(&w.lt.grid, &w.lt.layer_time, f.layer_time);
                (!md.is_nan()).then(|| (w.id, md, elev_at_md(w.geo, md)))
            })
            .collect();
        if !picks.is_empty() {
            out.push((f, picks));
        }
    }
    Ok(out)
}

/// Sub-horizon proposals from final layer time and complete normalized GR.
///
/// Sub-horizon picking with the constants from
/// `cfg`: per interval `1 ..= n − 1` (rayon, collected in interval order), a
/// median GR trace on `n_grid` layer-time samples over wells that take part
/// (state `Observed` / `Estimated` and a GR curve), the zero-mean Ricker
/// transform at each scale by FFT (one planner per call), zero crossings
/// tracked across scales, strength at scale `min_persistence − 1`, edge and
/// spacing rules, then the MD in each taking-part well by
/// [`md_at_layer_time`]. Ids follow (interval, layer time) after dropping
/// proposals with no covered well; `j` in the name counts the same kept
/// proposals. Inputs are only read.
///
/// **Choice:** A segment shorter than 2 samples gives no proposals
/// (it can hold no crossing).
///
/// Units: layer time, ft, API. NaN rule: skip (NaN GR or layer time means
/// "not covered"). Errors: [`SxError::Config`] (bad `[subhorizons]` keys),
/// [`SxError::EmptyInput`] (no wells), [`SxError::Shape`] (headers,
/// geometry, logs and layer time differ in wells or order; a layer-time grid
/// differs from the logs grid or its length; a GR curve of the wrong length;
/// a well without exactly the interval rows `0..=n`). `n ≤ 1` gives empty
/// output. Complexity: `O(intervals × wells × (samples + n_grid))` plus
/// `O(intervals × scales × F log F)`, `F` the FFT size.
pub fn propose_subhorizons(
    headers: &[WellHeader],
    geometry: &[WellGeometry],
    logs: &[WellLogs],
    layer_time: &[LayerTimeWell],
    intervals: &[IntervalRow],
    tops: &TopSet,
    cfg: &SubhorizonsConfig,
) -> SxResult<Subhorizons> {
    check_subhorizons_config(cfg)?;
    if headers.is_empty() {
        return Err(SxError::EmptyInput("subhorizons: no wells".into()));
    }
    let ids: Vec<WellId> = headers.iter().map(|h| h.id).collect();
    if geometry.iter().map(|g| g.id).ne(ids.iter().copied())
        || logs.iter().map(|l| l.id).ne(ids.iter().copied())
        || layer_time.iter().map(|t| t.well).ne(ids.iter().copied())
    {
        return Err(SxError::Shape(
            "subhorizons: headers, geometry, logs and layer time must have the same wells in the same order"
                .into(),
        ));
    }
    let n = tops.len();
    let mut states: BTreeMap<WellId, Vec<IntervalState>> = BTreeMap::new();
    for rows in intervals.chunk_by(|x, y| x.well == y.well) {
        let Some(first) = rows.first() else {
            continue;
        };
        if rows.len() != n + 1
            || rows
                .iter()
                .enumerate()
                .any(|(k, r)| usize::from(r.interval) != k)
        {
            return Err(SxError::Shape(format!(
                "subhorizons: well {} needs exactly the interval rows 0..={n}",
                first.well.0
            )));
        }
        if states
            .insert(first.well, rows.iter().map(|r| r.state).collect())
            .is_some()
        {
            return Err(SxError::Shape(format!(
                "subhorizons: interval rows of well {} are not contiguous",
                first.well.0
            )));
        }
    }
    let mut wells: Vec<WellIn<'_>> = Vec::with_capacity(headers.len());
    for ((g, l), lt) in geometry.iter().zip(logs).zip(layer_time) {
        if lt.grid != l.grid || lt.layer_time.len() != lt.grid.len {
            return Err(SxError::Shape(format!(
                "subhorizons: well {}: layer time grid differs from the logs grid",
                l.id.0
            )));
        }
        let gr = l.curves.get(&LogKind::Gr).map(Vec::as_slice);
        if gr.is_some_and(|v| v.len() != l.grid.len) {
            return Err(SxError::Shape(format!(
                "subhorizons: well {}: GR length differs from the grid",
                l.id.0
            )));
        }
        let st = states.get(&l.id).ok_or_else(|| {
            SxError::Shape(format!("subhorizons: no interval rows for well {}", l.id.0))
        })?;
        wells.push(WellIn {
            id: l.id,
            lt,
            gr,
            geo: g,
            states: st,
        });
    }
    if n < 2 {
        return Ok(Subhorizons {
            proposals: Vec::new(),
            picks: Vec::new(),
        });
    }

    let tr = Transform::new(cfg)?;
    let per_interval: Vec<SxResult<IntervalOut>> = (1..n)
        .into_par_iter()
        .map(|i| interval_proposals(&wells, i, &tr, cfg))
        .collect();

    let mut proposals = Vec::new();
    let mut picks = Vec::new();
    let mut next_id: u32 = 0;
    for (i, res) in (1..n).zip(per_interval) {
        let found = res?;
        let interval = u16::try_from(i)
            .map_err(|_| SxError::Shape(format!("subhorizons: interval {i} exceeds u16")))?;
        let upper = tops.names.get(i - 1).cloned().unwrap_or_default();
        for (j, (f, covered)) in found.into_iter().enumerate() {
            let id = next_id;
            next_id = next_id
                .checked_add(1)
                .ok_or_else(|| SxError::InvalidInput("subhorizons: too many proposals".into()))?;
            proposals.push(SubhorizonProposal {
                id,
                interval,
                name: format!("{upper}.sh{}", j + 1),
                upper_top_name: upper.clone(),
                layer_time: f.layer_time,
                strength_api: f.strength_api,
                polarity: f.polarity,
                persistence: u8::try_from(f.persistence).map_err(|_| {
                    SxError::InvalidInput(format!(
                        "subhorizons: persistence {} exceeds u8",
                        f.persistence
                    ))
                })?,
                n_wells: f.n_wells,
            });
            picks.extend(
                covered
                    .into_iter()
                    .map(|(well, md_ft, elev_ft)| SubhorizonPick {
                        id,
                        well,
                        md_ft,
                        elev_ft,
                    }),
            );
        }
    }
    picks.sort_by_key(|p| (p.id, p.well));
    tracing::info!(
        proposals = proposals.len(),
        picks = picks.len(),
        "sub-horizons proposed"
    );
    Ok(Subhorizons { proposals, picks })
}

/// `subhorizons` table (Parquet and CSV share it).
///
/// One row per pick, sorted by (`subhorizon_id`, `well_id`). Columns
/// `subhorizon_id` int32, `subhorizon_name` utf8, `interval` int32,
/// `upper_top_name` utf8, `layer_time`, `strength_api` float64, `polarity`,
/// `persistence` int32, `n_wells` int64, `well_id` int32, `well_name` utf8,
/// `md_ft`, `elev_ft` float64. Schema tag `subsurfix.subhorizons/1`.
///
/// NaN rule: NaN floats are stored as is. Errors: [`SxError::Shape`] (a pick
/// whose proposal id is missing), [`SxError::NotFound`] (a pick well missing
/// from `headers`), [`SxError::InvalidInput`] (an id or count above the
/// column type, batch build). No picks give a header-only batch.
/// Complexity: `O(picks log picks)`.
pub fn subhorizons_batch(
    sh: &Subhorizons,
    headers: &[WellHeader],
) -> SxResult<arrow::record_batch::RecordBatch> {
    let props: BTreeMap<u32, &SubhorizonProposal> =
        sh.proposals.iter().map(|p| (p.id, p)).collect();
    let names: BTreeMap<WellId, &str> = headers.iter().map(|h| (h.id, h.name.as_str())).collect();
    let mut picks: Vec<&SubhorizonPick> = sh.picks.iter().collect();
    picks.sort_by_key(|p| (p.id, p.well));
    let mut rows: Vec<(&SubhorizonPick, &SubhorizonProposal, &str)> =
        Vec::with_capacity(picks.len());
    for p in picks {
        let prop = props.get(&p.id).copied().ok_or_else(|| {
            SxError::Shape(format!("subhorizons table: no proposal with id {}", p.id))
        })?;
        let name = names.get(&p.well).copied().ok_or_else(|| {
            SxError::NotFound(format!("subhorizons table: header of well {}", p.well.0))
        })?;
        rows.push((p, prop, name));
    }
    let ids: Vec<i32> = rows
        .iter()
        .map(|r| {
            i32::try_from(r.0.id).map_err(|_| {
                SxError::InvalidInput(format!("subhorizon id {} exceeds int32", r.0.id))
            })
        })
        .collect::<SxResult<_>>()?;
    let n_wells: Vec<i64> = rows
        .iter()
        .map(|r| {
            i64::try_from(r.1.n_wells)
                .map_err(|_| SxError::InvalidInput(format!("count {} exceeds int64", r.1.n_wells)))
        })
        .collect::<SxResult<_>>()?;
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("subhorizon_id", Arc::new(Int32Array::from(ids))),
        (
            "subhorizon_name",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.1.name.as_str()),
            )),
        ),
        (
            "interval",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| i32::from(r.1.interval)),
            )),
        ),
        (
            "upper_top_name",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.1.upper_top_name.as_str()),
            )),
        ),
        (
            "layer_time",
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.1.layer_time),
            )),
        ),
        (
            "strength_api",
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.1.strength_api),
            )),
        ),
        (
            "polarity",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| i32::from(r.1.polarity)),
            )),
        ),
        (
            "persistence",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| i32::from(r.1.persistence)),
            )),
        ),
        ("n_wells", Arc::new(Int64Array::from(n_wells))),
        (
            "well_id",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.0.well.0),
            )),
        ),
        (
            "well_name",
            Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.2))),
        ),
        (
            "md_ft",
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.0.md_ft),
            )),
        ),
        (
            "elev_ft",
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.0.elev_ft),
            )),
        ),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(name, a)| Field::new(*name, a.data_type().clone(), false))
        .collect();
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, SUBHORIZONS_TAG)
}
