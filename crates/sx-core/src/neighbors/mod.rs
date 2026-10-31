// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Neighbor search, weighted median and neighbor curves.
//!
//! For every target well, every log and every grid row, [`build_neighbor_curves`]
//! gives one robust "what the neighbors say here" value with its spread,
//! contributor count, distance trust, quality, tier code and trusted flag.
//! Only `Measured` samples of other wells donate; a well's own reading never
//! enters its own curve. Values are in working units (RESD in `log10(ohm·m)`);
//! input logs are expected in those units already (logs_norm) and are not converted again.
//! The donor-mapping helpers ([`eligible_donors`], [`sample_donor`],
//! [`donor_score`]) are public for the gap-fill rules.
//!
//! Parallelism: rayon over targets; each target runs the
//! fixed order (logs in suite order, donors in list order) and results are
//! collected by `WellId`, so output is the same for any thread count.

#![warn(missing_docs)]

mod curves;
mod mapping;
mod median;
mod search;
mod table;
mod weights;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rayon::prelude::*;
use tracing::info;

use crate::config::NeighborsConfig;
use crate::error::{SxError, SxResult};
use crate::types::{GridRange, LogKind, WellGeometry, WellHeader, WellId, WellLogs};

pub use mapping::sample_donor;
pub use median::{weighted_median, WeightedMedian};
pub use search::eligible_donors;
pub use table::{neighbor_curves_from_batch, neighbor_curves_to_batch};
pub use weights::{donor_score, interval_facts, IntervalFacts, PairQuality};

/// Coverage window of one well: grid indices `k_first..=k_last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    /// First global grid index with a value.
    pub k_first: i32,
    /// Last global grid index with a value.
    pub k_last: i32,
}

/// First and last grid index where any log has a finite value (any provenance).
/// `None` if the well has no finite value. Pass the ingest-stage logs.
///
/// NaN rule: skip. Errors: none. Complexity: `O(grid × logs)`.
pub fn coverage_window(raw: &WellLogs) -> Option<Coverage> {
    let has_value = |i: usize| {
        raw.curves
            .values()
            .any(|c| c.get(i).is_some_and(|v| v.is_finite()))
    };
    let first = (0..raw.grid.len).find(|&i| has_value(i))?;
    let last = (first..raw.grid.len).rev().find(|&i| has_value(i))?;
    Some(Coverage {
        k_first: raw.grid.k(first),
        k_last: raw.grid.k(last),
    })
}

/// Everything the neighbor build reads. All maps keyed by `WellId`.
#[derive(Debug, Clone, Copy)]
pub struct NeighborInputs<'a> {
    /// Surface point (`x_m`, `y_m`) per well.
    pub headers: &'a BTreeMap<WellId, WellHeader>,
    /// Per-sample geometry (`elev_ft`) per well, on the well's logs grid.
    pub geometry: &'a BTreeMap<WellId, WellGeometry>,
    /// Only `Measured` (code 0) finite samples are read as donor values.
    pub logs: &'a BTreeMap<WellId, WellLogs>,
    /// Mapping coordinate per well on its grid (layer coordinate or layer time). Missing key = no coordinate.
    pub coord: &'a BTreeMap<WellId, Vec<f64>>,
    /// Interval facts per well (index = interval id).
    pub facts: &'a BTreeMap<WellId, Vec<IntervalFacts>>,
    /// Rough-alignment pair qualities.
    pub pairs: &'a PairQuality,
    /// Coverage window per well.
    pub coverage: &'a BTreeMap<WellId, Coverage>,
    /// Regional (ref_low, ref_high) for range-mapped logs. RESD in log10.
    pub ref_ranges: &'a BTreeMap<(WellId, LogKind), (f64, f64)>,
    /// Channels that may not donate.
    pub exclude: &'a BTreeSet<(WellId, LogKind)>,
    /// Wells that may not donate any log (holdout). Empty in production.
    pub donor_exclude_wells: &'a BTreeSet<WellId>,
}

/// One log's neighbor curve on a well's grid. Every vector has length `grid.len`.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborCurve {
    /// Neighbor value (after validation and range map), working units; NaN = none.
    pub value: Vec<f32>,
    /// Weighted median absolute deviation, same unit as `value`; NaN if `n < 2`.
    pub spread: Vec<f32>,
    /// Contributors at the accepting step.
    pub n_donors: Vec<u32>,
    /// Distance trust `Σ dist·w / Σ w`, m; NaN = none.
    pub dist_m: Vec<f64>,
    /// Aggregate quality `Σ w·q / Σ w`; NaN = none.
    pub quality: Vec<f32>,
    /// 1..=5, or `CODE_NONE`.
    pub tier: Vec<u8>,
    /// Trusted flag.
    pub trusted: Vec<bool>,
}

/// Neighbor curves of one well.
#[derive(Debug, Clone, PartialEq)]
pub struct WellNeighborCurves {
    /// Well id.
    pub id: WellId,
    /// The well's logs grid.
    pub grid: GridRange,
    /// All six logs, always present (all-NaN curve if nothing was found).
    pub curves: BTreeMap<LogKind, NeighborCurve>,
}

/// Build curves for `targets` (each must be a key of every per-well map except `coord`).
/// Donors are drawn from every well in `inputs.logs`.
/// NaN rule: skip. Errors: `NotFound` for a target missing from a map; `InvalidInput`
/// for a non-monotone coordinate; `Shape` for length mismatches.
/// Empty `targets`: returns an empty map.
///
/// Per target: tiered search, validation and trusted flag
///; then, after every target is built, the range map. Repeated
/// targets are built once. Complexity: about `O(targets × logs × rows × donors)`.
pub fn build_neighbor_curves(
    inputs: &NeighborInputs<'_>,
    targets: &[WellId],
    cfg: &NeighborsConfig,
) -> SxResult<BTreeMap<WellId, WellNeighborCurves>> {
    if targets.is_empty() {
        return Ok(BTreeMap::new());
    }
    let start = Instant::now();
    let ids: Vec<WellId> = targets
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for &id in &ids {
        let missing = [
            ("headers", inputs.headers.contains_key(&id)),
            ("geometry", inputs.geometry.contains_key(&id)),
            ("logs", inputs.logs.contains_key(&id)),
            ("facts", inputs.facts.contains_key(&id)),
            ("coverage", inputs.coverage.contains_key(&id)),
        ]
        .into_iter()
        .find(|(_, present)| !present);
        if let Some((map, _)) = missing {
            return Err(SxError::NotFound(format!(
                "neighbor curves: target {} missing from {map}",
                id.0
            )));
        }
    }
    let pool = search::Pool::new(inputs, cfg)?;
    let built: Vec<SxResult<WellNeighborCurves>> = ids
        .par_iter()
        .map(|&id| curves::build_target(&pool, id))
        .collect();
    let mut wells = Vec::with_capacity(built.len());
    for w in built {
        wells.push(w?);
    }
    let mapped: Vec<SxResult<WellNeighborCurves>> = wells
        .into_par_iter()
        .map(|mut w| {
            let coverage = inputs.coverage.get(&w.id);
            let coord = inputs.coord.get(&w.id).map(Vec::as_slice);
            curves::range_map_well(&mut w, coverage, coord, inputs.ref_ranges, cfg)?;
            Ok(w)
        })
        .collect();
    let mut out = BTreeMap::new();
    for w in mapped {
        let w = w?;
        out.insert(w.id, w);
    }
    info!(
        wells = out.len(),
        seconds = start.elapsed().as_secs_f64(),
        "neighbor curves built"
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IntervalState, Provenance};

    const N: usize = 12;

    fn header(id: i32, x_m: f64) -> WellHeader {
        WellHeader {
            id: WellId(id),
            name: format!("w{id}"),
            x_m,
            y_m: 0.0,
            kb_elev_ft: 1000.0,
        }
    }

    fn geometry(id: i32, grid: GridRange) -> WellGeometry {
        let tvd: Vec<f64> = (0..grid.len).map(|i| grid.md_ft(i)).collect();
        WellGeometry {
            id: WellId(id),
            grid,
            elev_ft: tvd.iter().map(|t| 1000.0 - t).collect(),
            tvd_ft: tvd,
            x_m: vec![0.0; grid.len],
            y_m: vec![0.0; grid.len],
        }
    }

    fn gr_logs(id: i32, grid: GridRange, values: Vec<f32>, prov: Vec<u8>) -> WellLogs {
        let mut logs = WellLogs::new(WellId(id), grid);
        logs.curves.insert(LogKind::Gr, values);
        logs.provenance.insert(LogKind::Gr, prov);
        logs
    }

    /// Vertical wells on one grid, no coordinate: rows are elevation-matched one to one.
    struct Project {
        headers: BTreeMap<WellId, WellHeader>,
        geometry: BTreeMap<WellId, WellGeometry>,
        logs: BTreeMap<WellId, WellLogs>,
        facts: BTreeMap<WellId, Vec<IntervalFacts>>,
        coverage: BTreeMap<WellId, Coverage>,
    }

    impl Project {
        fn new(wells: Vec<(i32, f64, Vec<f32>, Vec<u8>)>) -> Self {
            let grid = GridRange::new(2000, N).unwrap();
            let mut p = Project {
                headers: BTreeMap::new(),
                geometry: BTreeMap::new(),
                logs: BTreeMap::new(),
                facts: BTreeMap::new(),
                coverage: BTreeMap::new(),
            };
            for (id, x, values, prov) in wells {
                let w = WellId(id);
                p.headers.insert(w, header(id, x));
                p.geometry.insert(w, geometry(id, grid));
                p.logs.insert(w, gr_logs(id, grid, values, prov));
                p.facts.insert(
                    w,
                    vec![IntervalFacts {
                        state: IntervalState::Pad,
                        tvd_thickness_ft: 10.0,
                        confidence: 0.5,
                    }],
                );
                p.coverage.insert(
                    w,
                    Coverage {
                        k_first: 2000,
                        k_last: 2000 + N as i32 - 1,
                    },
                );
            }
            p
        }

        fn build(&self, target: i32) -> WellNeighborCurves {
            let (coord, refs, exclude, holdout) = (
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            );
            let pairs = PairQuality::default();
            let inputs = NeighborInputs {
                headers: &self.headers,
                geometry: &self.geometry,
                logs: &self.logs,
                coord: &coord,
                facts: &self.facts,
                pairs: &pairs,
                coverage: &self.coverage,
                ref_ranges: &refs,
                exclude: &exclude,
                donor_exclude_wells: &holdout,
            };
            let cfg = NeighborsConfig::default();
            let mut out = build_neighbor_curves(&inputs, &[WellId(target)], &cfg).unwrap();
            out.remove(&WellId(target)).unwrap()
        }
    }

    fn ramp(base: f32) -> Vec<f32> {
        (0..N).map(|i| base + i as f32).collect()
    }

    fn measured() -> Vec<u8> {
        vec![Provenance::Measured.code(); N]
    }

    #[test]
    fn neighbors_own_reading_never_enters_own_curve() {
        let donors = |p: &mut Vec<(i32, f64, Vec<f32>, Vec<u8>)>| {
            for (id, x, base) in [(2, 100.0, 60.0), (3, 200.0, 62.0), (4, 300.0, 64.0)] {
                p.push((id, x, ramp(base), measured()));
            }
        };
        let mut with_own = vec![(1, 0.0, ramp(400.0), measured())];
        donors(&mut with_own);
        let mut without_own = vec![(1, 0.0, vec![f32::NAN; N], vec![255; N])];
        donors(&mut without_own);
        let a = Project::new(with_own).build(1);
        let b = Project::new(without_own).build(1);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        let gr = &a.curves[&LogKind::Gr];
        assert_eq!(gr.tier, vec![1; N]);
        assert_eq!(gr.n_donors, vec![3; N]);
        for (i, v) in gr.value.iter().enumerate() {
            assert_eq!(*v, 60.0 + i as f32);
        }
    }

    #[test]
    fn neighbors_filled_samples_never_donate() {
        let mut filled = measured();
        for (i, code) in filled.iter_mut().enumerate().skip(6) {
            *code = if i % 2 == 0 {
                Provenance::Model.code()
            } else {
                Provenance::NeighborWeighted.code() + (i % 15) as u8
            };
        }
        let p = Project::new(vec![
            (1, 0.0, vec![f32::NAN; N], vec![255; N]),
            (2, 100.0, ramp(60.0), filled),
        ]);
        let gr = &p.build(1).curves[&LogKind::Gr];
        for i in 0..6 {
            assert_eq!(
                (gr.value[i], gr.n_donors[i], gr.tier[i]),
                (60.0 + i as f32, 1, 3)
            );
        }
        for i in 6..N {
            assert!(gr.value[i].is_nan(), "row {i}");
            assert_eq!((gr.n_donors[i], gr.tier[i]), (0, 255));
        }
    }
}
