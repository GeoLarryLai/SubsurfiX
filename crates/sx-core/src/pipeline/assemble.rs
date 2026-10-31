// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Owned per-well maps behind `NeighborInputs` / `FillInputs`, built from files.
//!
//! Sources: wells and geometry from ingest; `logs` = `logs_norm`
//! (working units, RESD log10); `coverage` from `logs_raw`; `facts` from the
//! interval table and the kept boundaries of `tops_estimated` (user picks plus
//! estimated tops); `pairs` from `pairs_rough`; `ref_ranges` from
//! `norm_anchors.csv`; `exclude` = channels without a measured sample in
//! `logs_norm` plus `jsd_outlier` channels of `norm_report.csv`;
//! `resd_family` from `wells.parquet` (unknown well → 0); `coord_source` and
//! the rough `u` from `layer_coord.parquet`. Only wells with logs enter.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::products::{
    GEOMETRY, INTERVALS, LAYER_COORD, LAYER_TIME, LOGS_NORM, LOGS_RAW, NORM_ANCHORS, NORM_REPORT,
    PAIRS_ROUGH, TOPS_ESTIMATED, WELLS,
};
use super::readers::{
    read_estimated_picks, read_geometry, read_intervals, read_jsd_outliers, read_layer_coord,
    read_layer_time, read_logs, read_logs_norm, read_pairs, read_ref_ranges, read_wells,
};
use crate::config::ProjectConfig;
use crate::error::{SxError, SxResult};
use crate::io::products::LOGS_RAW_TAG;
use crate::neighbors::{
    coverage_window, interval_facts, Coverage, IntervalFacts, NeighborInputs, PairQuality,
};
use crate::types::{
    IntervalRow, LogKind, Provenance, TopSet, WellGeometry, WellHeader, WellId, WellLogs,
};

/// Mapping coordinate of the neighbor inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoordChoice {
    /// Rough layer coordinate `u` (neighbors, fill).
    Rough,
    /// Final layer time `τ` (refresh).
    LayerTime,
}

/// Everything the neighbor build and the fill read, owned.
pub(crate) struct NeighborData {
    pub(crate) headers: BTreeMap<WellId, WellHeader>,
    pub(crate) geometry: BTreeMap<WellId, WellGeometry>,
    /// `logs_norm`.
    pub(crate) logs: BTreeMap<WellId, WellLogs>,
    /// `u` or `τ`, per [`CoordChoice`].
    pub(crate) coord: BTreeMap<WellId, Vec<f64>>,
    pub(crate) facts: BTreeMap<WellId, Vec<IntervalFacts>>,
    pub(crate) pairs: PairQuality,
    pub(crate) coverage: BTreeMap<WellId, Coverage>,
    pub(crate) ref_ranges: BTreeMap<(WellId, LogKind), (f64, f64)>,
    pub(crate) exclude: BTreeSet<(WellId, LogKind)>,
    pub(crate) donor_exclude_wells: BTreeSet<WellId>,
    pub(crate) resd_family: BTreeMap<WellId, u8>,
    pub(crate) coord_source: BTreeMap<WellId, u8>,
    /// Rough `u` (refresh scores movement against it).
    pub(crate) rough_u: BTreeMap<WellId, Vec<f64>>,
    /// Study top list of rough alignment.
    pub(crate) tops: TopSet,
}

impl NeighborData {
    /// Read the products listed in the module doc from `project`.
    ///
    /// # Errors
    /// Reader errors; `SxError::Shape` when a well with logs has no geometry
    /// or no interval rows, or from `interval_facts`.
    pub(crate) fn load(project: &Path, cfg: &ProjectConfig, choice: CoordChoice) -> SxResult<Self> {
        let p = |rel: &str| project.join(rel);
        let (headers_v, family) = read_wells(&p(WELLS))?;
        let geometry_v = read_geometry(&p(GEOMETRY))?;
        let raw = read_logs(&p(LOGS_RAW), LOGS_RAW_TAG)?;
        let norm = read_logs_norm(&p(LOGS_NORM))?;
        let ref_ranges = read_ref_ranges(&p(NORM_ANCHORS))?;
        let outliers = read_jsd_outliers(&p(NORM_REPORT))?;
        let (tops, rows) = read_intervals(&p(INTERVALS))?;
        let picks = read_estimated_picks(&p(TOPS_ESTIMATED), tops.len())?;
        let pairs = PairQuality::from_pairs(&read_pairs(&p(PAIRS_ROUGH))?)?;
        let (rough_u, sources) = read_layer_coord(&p(LAYER_COORD))?;

        let ids: BTreeSet<WellId> = norm.iter().map(|l| l.id).collect();
        let headers: BTreeMap<WellId, WellHeader> = headers_v
            .into_iter()
            .filter(|h| ids.contains(&h.id))
            .map(|h| (h.id, h))
            .collect();
        let geometry: BTreeMap<WellId, WellGeometry> = geometry_v
            .into_iter()
            .filter(|g| ids.contains(&g.id))
            .map(|g| (g.id, g))
            .collect();
        let coverage: BTreeMap<WellId, Coverage> = raw
            .iter()
            .filter(|l| ids.contains(&l.id))
            .filter_map(|l| coverage_window(l).map(|c| (l.id, c)))
            .collect();

        let mut by_well: BTreeMap<WellId, Vec<IntervalRow>> = BTreeMap::new();
        for r in rows {
            by_well.entry(r.well).or_default().push(r);
        }
        let mut facts = BTreeMap::new();
        for &id in &ids {
            let geom = geometry
                .get(&id)
                .ok_or_else(|| SxError::Shape(format!("well {} has logs but no geometry", id.0)))?;
            let rows = by_well.get(&id).ok_or_else(|| {
                SxError::Shape(format!("well {} has logs but no interval rows", id.0))
            })?;
            facts.insert(
                id,
                interval_facts(rows, &picks, &tops, geom, &cfg.neighbors)?,
            );
        }

        let mut exclude = outliers;
        for l in &norm {
            for log in LogKind::ALL {
                let measured = match (l.curves.get(&log), l.provenance.get(&log)) {
                    (Some(v), Some(p)) => v
                        .iter()
                        .zip(p)
                        .any(|(x, &c)| x.is_finite() && c == Provenance::Measured.code()),
                    _ => false,
                };
                if !measured {
                    exclude.insert((l.id, log));
                }
            }
        }

        let coord = match choice {
            CoordChoice::Rough => rough_u.clone(),
            CoordChoice::LayerTime => read_layer_time(&p(LAYER_TIME))?
                .into_iter()
                .map(|t| (t.well, t.layer_time))
                .collect(),
        };
        let resd_family = ids
            .iter()
            .map(|id| (*id, family.get(id).copied().unwrap_or(0)))
            .collect();
        let coord_source = ids
            .iter()
            .map(|id| (*id, sources.get(id).copied().unwrap_or(0)))
            .collect();
        Ok(Self {
            headers,
            geometry,
            logs: norm.into_iter().map(|l| (l.id, l)).collect(),
            coord,
            facts,
            pairs,
            coverage,
            ref_ranges,
            exclude,
            donor_exclude_wells: BTreeSet::new(),
            resd_family,
            coord_source,
            rough_u,
            tops,
        })
    }

    /// Borrowed view for `build_neighbor_curves` and the fill.
    pub(crate) fn inputs(&self) -> NeighborInputs<'_> {
        NeighborInputs {
            headers: &self.headers,
            geometry: &self.geometry,
            logs: &self.logs,
            coord: &self.coord,
            facts: &self.facts,
            pairs: &self.pairs,
            coverage: &self.coverage,
            ref_ranges: &self.ref_ranges,
            exclude: &self.exclude,
            donor_exclude_wells: &self.donor_exclude_wells,
        }
    }
}
