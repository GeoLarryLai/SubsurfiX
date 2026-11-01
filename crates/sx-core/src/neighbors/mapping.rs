// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Mapping a donor onto target rows: layer-coordinate and elevation
//! inverse lookups, the pinch rule, the donor value lookup with the 8 ft bracket,
//! and [`sample_donor`].
//!
//! Inverse lookups use binary search over the finite samples of
//! the coordinate, and over the elevation when it is finite and non-increasing
//! everywhere (otherwise a linear scan). Both give the reference definitions.

use std::ops::Range;

use crate::config::NeighborsConfig;
use crate::error::{SxError, SxResult};
use crate::neighbors::search::may_donate;
use crate::neighbors::weights::IntervalFacts;
use crate::neighbors::NeighborInputs;
use crate::types::{LogKind, Provenance, WellGeometry, WellId, WellLogs, GRID_STEP_FT};

/// `floor(u + eps)` as an interval id.
pub(crate) fn interval_of(u: f64, eps: f64) -> i64 {
    (u + eps).floor() as i64
}

/// Layer coordinate of one well with the indices of its finite samples.
pub(crate) struct CoordAxis<'a> {
    u: &'a [f64],
    finite: Vec<u32>,
}

impl<'a> CoordAxis<'a> {
    /// `None` when no sample is finite. Errors: `InvalidInput` if the finite
    /// samples decrease anywhere.
    fn new(id: WellId, u: &'a [f64]) -> SxResult<Option<Self>> {
        let mut finite = Vec::new();
        let mut prev = f64::NEG_INFINITY;
        for (j, &x) in u.iter().enumerate() {
            if !x.is_finite() {
                continue;
            }
            if x < prev {
                return Err(SxError::InvalidInput(format!(
                    "well {}: mapping coordinate decreases at sample {j}",
                    id.0
                )));
            }
            prev = x;
            let j = u32::try_from(j)
                .map_err(|_| SxError::InvalidInput(format!("well {}: grid too long", id.0)))?;
            finite.push(j);
        }
        Ok((!finite.is_empty()).then_some(Self { u, finite }))
    }

    /// Coordinate at local row `r` (NaN outside or undefined).
    pub(crate) fn at(&self, r: usize) -> f64 {
        self.u.get(r).copied().unwrap_or(f64::NAN)
    }

    /// Finite samples, in grid order.
    pub(crate) fn finite_values(&self) -> impl Iterator<Item = f64> + '_ {
        self.finite.iter().map(|&j| self.at(j as usize))
    }

    /// Fractional donor index where the coordinate equals `q`
    /// (pinch rule included); `None` if there is no valid bracket.
    pub(crate) fn position(&self, q: f64, eps: f64) -> Option<f64> {
        if !q.is_finite() {
            return None;
        }
        let m = self.finite.partition_point(|&j| self.at(j as usize) < q);
        if let Some(&j) = self.finite.get(m) {
            if self.at(j as usize) == q {
                return Some(f64::from(j));
            }
        }
        let lo = *self.finite.get(m.checked_sub(1)?)?;
        let hi = *self.finite.get(m)?;
        if hi != lo + 1 {
            return None;
        }
        let (ul, uh) = (self.at(lo as usize), self.at(hi as usize));
        if interval_of(uh, eps) - interval_of(ul, eps) > 1 {
            return None;
        }
        Some(f64::from(lo) + (q - ul) / (uh - ul))
    }
}

/// Elevation of one well (ft, positive up).
pub(crate) struct ElevAxis<'a> {
    e: &'a [f64],
    monotone: bool,
}

impl<'a> ElevAxis<'a> {
    fn new(e: &'a [f64]) -> Self {
        let monotone =
            e.iter().all(|x| x.is_finite()) && e.windows(2).all(|w| matches!(w, [a, b] if b <= a));
        Self { e, monotone }
    }

    /// Elevation at local row `r` (NaN outside).
    pub(crate) fn at(&self, r: usize) -> f64 {
        self.e.get(r).copied().unwrap_or(f64::NAN)
    }

    /// Smallest fractional index `p` with the first `j` such that
    /// `elev(j) >= t >= elev(j+1)`, linear between `j` and `j+1`.
    pub(crate) fn position(&self, t: f64) -> Option<f64> {
        if t.is_nan() {
            return None;
        }
        if self.monotone {
            let m = self.e.partition_point(|&x| x > t);
            if m == 0 {
                return (self.at(0) == t && self.e.len() >= 2).then_some(0.0);
            }
            return bracket(m - 1, self.at(m - 1), *self.e.get(m)?, t);
        }
        self.e.windows(2).enumerate().find_map(|(j, w)| match w {
            [a, b] if *a >= t && t >= *b => bracket(j, *a, *b, t),
            _ => None,
        })
    }
}

/// `p` inside the bracket `elev(j) = a >= t >= elev(j+1) = b`.
fn bracket(j: usize, a: f64, b: f64, t: f64) -> Option<f64> {
    if a == t {
        return Some(j as f64);
    }
    Some(j as f64 + (a - t) / (a - b))
}

/// Measured finite samples of one log of one well, in working units.
#[derive(Clone, Copy)]
pub(crate) struct Measured<'a> {
    values: &'a [f32],
    prov: &'a [u8],
    k_start: i32,
}

impl<'a> Measured<'a> {
    /// `None` when the well has no curve for `log`.
    pub(crate) fn new(logs: &'a WellLogs, log: LogKind) -> Option<Self> {
        Some(Self {
            values: logs.curves.get(&log)?,
            prov: logs.provenance.get(&log)?,
            k_start: logs.grid.k_start,
        })
    }

    /// Number of grid samples.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// Working value at sample `j` if it is `Measured` and finite.
    pub(crate) fn get(&self, j: usize) -> Option<f64> {
        if *self.prov.get(j)? != Provenance::Measured.code() {
            return None;
        }
        // Inputs are already in working units (RESD log10, from logs_norm).
        let v = f64::from(*self.values.get(j)?);
        v.is_finite().then_some(v)
    }

    /// Donor value at fractional index `p`: the sample at `p`, else
    /// linear in MD between the nearest measured samples at or above and at or
    /// below `p`, if both exist and are at most `bracket_ft` apart.
    pub(crate) fn value_at(&self, p: f64, bracket_ft: f64) -> Option<f64> {
        let len = self.len();
        if !p.is_finite() || p < 0.0 || len == 0 || p > (len - 1) as f64 {
            return None;
        }
        let floor = p.floor() as usize;
        let ceil = p.ceil() as usize;
        if floor == ceil {
            if let Some(v) = self.get(floor) {
                return Some(v);
            }
        }
        let reach = if bracket_ft.is_nan() || bracket_ft < 0.0 {
            0
        } else {
            let steps = (bracket_ft / GRID_STEP_FT).floor();
            if steps >= len as f64 {
                len
            } else {
                steps as usize
            }
        };
        let lo_end = ceil.saturating_sub(reach);
        let (jl, vl) = (lo_end..=floor)
            .rev()
            .find_map(|j| self.get(j).map(|v| (j, v)))?;
        let hi_end = floor.saturating_add(reach).min(len - 1);
        let (jh, vh) = (ceil..=hi_end).find_map(|j| self.get(j).map(|v| (j, v)))?;
        if jh <= jl {
            return None;
        }
        let k0 = f64::from(self.k_start);
        let md_lo = GRID_STEP_FT * (k0 + jl as f64);
        let md_hi = GRID_STEP_FT * (k0 + jh as f64);
        if md_hi - md_lo > bracket_ft {
            return None;
        }
        let md = GRID_STEP_FT * (k0 + p);
        Some(vl + (vh - vl) * (md - md_lo) / (md_hi - md_lo))
    }
}

/// One well as seen by the neighbor build.
pub(crate) struct WellView<'a> {
    pub id: WellId,
    /// Finite surface point, or `None`.
    pub xy: Option<[f64; 2]>,
    pub logs: &'a WellLogs,
    pub elev: Option<ElevAxis<'a>>,
    pub coord: Option<CoordAxis<'a>>,
    pub facts: &'a [IntervalFacts],
}

impl<'a> WellView<'a> {
    /// Errors: `NotFound` if `id` has no logs; `Shape` for length mismatches
    /// (logs, geometry, coordinate); `InvalidInput` for a decreasing coordinate.
    pub(crate) fn new(inputs: &NeighborInputs<'a>, id: WellId) -> SxResult<Self> {
        let logs = inputs
            .logs
            .get(&id)
            .ok_or_else(|| SxError::NotFound(format!("well {} has no logs", id.0)))?;
        logs.validate()?;
        let n = logs.grid.len;
        let geom: Option<&WellGeometry> = inputs.geometry.get(&id);
        if let Some(g) = geom {
            g.validate()?;
            if g.grid != logs.grid {
                return Err(SxError::Shape(format!(
                    "well {}: geometry grid differs from logs grid",
                    id.0
                )));
            }
        }
        let coord = match inputs.coord.get(&id) {
            Some(u) if u.len() != n => {
                return Err(SxError::Shape(format!(
                    "well {}: coordinate has {} samples, grid has {n}",
                    id.0,
                    u.len()
                )))
            }
            Some(u) => CoordAxis::new(id, u)?,
            None => None,
        };
        let xy = inputs
            .headers
            .get(&id)
            .map(|h| [h.x_m, h.y_m])
            .filter(|p| p.iter().all(|v| v.is_finite()));
        Ok(Self {
            id,
            xy,
            logs,
            elev: geom.map(|g| ElevAxis::new(&g.elev_ft)),
            coord,
            facts: inputs.facts.get(&id).map_or(&[], Vec::as_slice),
        })
    }

    /// Facts of interval `i` (`None` if negative or absent).
    pub(crate) fn fact(&self, i: i64) -> Option<&IntervalFacts> {
        self.facts.get(usize::try_from(i).ok()?)
    }
}

/// Donor value and interval key (`Some(i)` coordinate-mapped, `None`
/// elevation-matched) at target row `r`; `None` if there is no sample.
pub(crate) fn sample_row(
    target: &WellView<'_>,
    donor: &WellView<'_>,
    measured: &Measured<'_>,
    r: usize,
    cfg: &NeighborsConfig,
) -> Option<(f64, Option<i64>)> {
    let eps = cfg.interval_eps;
    let q = target.coord.as_ref().map_or(f64::NAN, |c| c.at(r));
    let (p, key) = match (&donor.coord, q.is_finite()) {
        (Some(dc), true) => {
            let i = interval_of(q, eps);
            let ok = |w: &WellView<'_>| w.fact(i).is_some_and(IntervalFacts::usable);
            if !ok(target) || !ok(donor) {
                return None;
            }
            (dc.position(q, eps)?, Some(i))
        }
        _ => {
            let e = target.elev.as_ref()?.at(r);
            (donor.elev.as_ref()?.position(e)?, None)
        }
    };
    let v = measured.value_at(p, cfg.donor_max_bracket_ft)?;
    Some((v, key))
}

/// Donor `log` values mapped onto the target's local rows `rows`.
/// Output length = `rows.len()`, NaN where no sample. Uses only measured donor samples.
/// Ignores weights. Rows that forbids are NaN.
///
/// Values are in working units (RESD in log10). A donor that may not donate
/// `log` to `target` (the target itself, an excluded channel, a holdout well)
/// or has no curve for `log` gives all NaN. NaN rule: skip. Errors: `NotFound`
/// if `target` is missing from `logs` or `geometry`, or `donor` from `logs`;
/// `Shape` if `rows` is not inside the target grid or for length mismatches;
/// `InvalidInput` for a decreasing coordinate. Complexity:
/// `O(rows × log donor grid)` (linear scan per row for a non-monotone elevation).
pub fn sample_donor(
    inputs: &NeighborInputs<'_>,
    target: WellId,
    donor: WellId,
    log: LogKind,
    rows: Range<usize>,
    cfg: &NeighborsConfig,
) -> SxResult<Vec<f32>> {
    if !inputs.geometry.contains_key(&target) {
        return Err(SxError::NotFound(format!(
            "sample donor: target {} has no geometry",
            target.0
        )));
    }
    let t = WellView::new(inputs, target)?;
    let d = WellView::new(inputs, donor)?;
    if rows.start > rows.end || rows.end > t.logs.grid.len {
        return Err(SxError::Shape(format!(
            "sample donor: rows {rows:?} outside target grid of {}",
            t.logs.grid.len
        )));
    }
    let mut out = vec![f32::NAN; rows.len()];
    if !may_donate(inputs, target, donor, log) {
        return Ok(out);
    }
    let Some(measured) = Measured::new(d.logs, log) else {
        return Ok(out);
    };
    for (slot, r) in out.iter_mut().zip(rows) {
        if let Some((v, _)) = sample_row(&t, &d, &measured, r, cfg) {
            *slot = v as f32;
        }
    }
    Ok(out)
}
