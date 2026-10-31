// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Fill refresh: score how far layers moved
//! between the rough layer coordinate `u` and the final layer time `τ`, then
//! re-fill gaps once, with the already trained models, only in wells whose
//! score exceeds `refresh_tol_ft`.
//!
//! Measured samples never change; wells that did not move are copied unchanged
//! even if a neighbor moved; there is no retraining and no second pass. Pure:
//! no file I/O (the pipeline writes `logs_final.parquet` and `refresh_report.csv`).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use tracing::{debug, info};

use crate::align::interval_of_coord;
use crate::config::{FillConfig, NeighborsConfig, RefreshConfig};
use crate::error::{SxError, SxResult};
use crate::fill::sequential::{refill_wells, TrainedModels};
use crate::fill::{is_fill_code, logs_filled_to_batch, FillInputs};
use crate::io::parquet::with_schema_tag;
use crate::neighbors::{build_neighbor_curves, NeighborInputs};
use crate::types::{GridRange, Provenance, WellId, WellLogs, CODE_NONE, GRID_STEP_FT};
use crate::util::percentile;

/// Schema tag of `logs_final.parquet`.
pub(crate) const LOGS_FINAL_TAG: &str = "subsurfix.logs_final/1";

/// Schema tag of the `refresh_report` batch.
pub(crate) const REFRESH_REPORT_TAG: &str = "subsurfix.refresh_report/1";

/// Inputs to the refresh step.
#[derive(Debug, Clone, Copy)]
pub struct RefreshInputs<'a> {
    /// Neighbor inputs with `coord` = final layer time τ, `logs` = logs_norm (what the first fill read).
    pub nbr_tau: NeighborInputs<'a>,
    /// Resistivity tool family per well, as in the first fill.
    pub resd_family: &'a BTreeMap<WellId, u8>,
    /// `layer_coord_source` per well, as in the first fill (from layer_coord.parquet); unchanged by refresh.
    pub coord_source: &'a BTreeMap<WellId, u8>,
    /// Number of tops `n`; non-pad intervals are ids `1..=n-1`, taken from `u`.
    pub n_tops: usize,
    /// Rough layer coordinate `u` per well on its grid (NaN outside coverage).
    pub layer_coord: &'a BTreeMap<WellId, Vec<f64>>,
    /// Fill stage output (logs_filled), keyed by well; defines the well list and grids.
    pub logs_filled: &'a BTreeMap<WellId, WellLogs>,
}

/// Movement score of one well.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WellScore {
    /// Well id.
    pub well: WellId,
    /// Percentile of `moved_ft` over scored samples, feet. NaN if none.
    pub score_ft: f64,
    /// `score_ft > refresh_tol_ft` (false when NaN).
    pub moved: bool,
    /// Samples that entered the percentile.
    pub n_scored: usize,
}

/// One row of `refresh_report.csv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefreshRow {
    /// Well id.
    pub well: WellId,
    /// Well score, feet; NaN when no sample was scored.
    pub score_ft: f64,
    /// True if the well was re-filled.
    pub moved: bool,
    /// `(log, sample)` pairs re-filled with a fill code (0 for wells not re-filled).
    pub n_samples_refilled: usize,
}

/// Result of [`run_refresh`].
#[derive(Debug, Clone, PartialEq)]
pub struct RefreshOutput {
    /// All wells; becomes logs_final.parquet (see [`logs_final_batch`]).
    pub logs: BTreeMap<WellId, WellLogs>,
    /// One row per well, ascending id; becomes refresh_report.csv.
    pub report: Vec<RefreshRow>,
}

/// Inverse of a layer coordinate on one well's grid. Build once, query many times.
#[derive(Debug, Clone)]
pub struct CoordInverse {
    grid: GridRange,
    coord: Vec<f64>,
    /// Running maximum of `coord`, NaN taken as `-inf`.
    prefix_max: Vec<f64>,
}

impl CoordInverse {
    /// Copy `coord` (one value per grid sample) and its running maximum.
    ///
    /// NaN rule: skip (a NaN sample never satisfies `coord ≥ t`).
    /// Errors: `Shape` if `coord.len() != grid.len`. Complexity: `O(n)`.
    pub fn new(grid: GridRange, coord: &[f64]) -> SxResult<Self> {
        if coord.len() != grid.len {
            return Err(SxError::Shape(format!(
                "refresh: coordinate has {} samples, grid has {}",
                coord.len(),
                grid.len
            )));
        }
        let mut run = f64::NEG_INFINITY;
        let prefix_max = coord
            .iter()
            .map(|&u| {
                if u > run {
                    run = u;
                }
                run
            })
            .collect();
        Ok(Self {
            grid,
            coord: coord.to_vec(),
            prefix_max,
        })
    }

    /// Smallest MD (ft) with `coord ≥ t`, linear between that sample and the one
    /// above it; that sample's MD when it is the first sample or the one above is NaN.
    ///
    /// NaN rule: NaN if no sample reaches `t`, or `t` is NaN or infinite.
    /// Errors: none. Complexity: `O(log n)`.
    pub fn md_at(&self, t: f64) -> f64 {
        if !t.is_finite() {
            return f64::NAN;
        }
        let j = self.prefix_max.partition_point(|&p| p < t);
        let Some(&uj) = self.coord.get(j) else {
            return f64::NAN;
        };
        let prev = j
            .checked_sub(1)
            .and_then(|i| self.coord.get(i).map(|&u| (i, u)))
            .filter(|(_, u)| !u.is_nan());
        match prev {
            Some((i, ui)) => self.grid.md_ft(i) + GRID_STEP_FT * (t - ui) / (uj - ui),
            None => self.grid.md_ft(j),
        }
    }
}

/// Score every well of `inputs.logs_filled`: P-`cfg.percentile` of `moved_ft`.
///
/// A sample is scored when `u` and `τ` are finite and its interval from `u`
/// (`interval_of_coord(u, n_tops)`) is a non-pad id `1..=n_tops-1`; then
/// `moved_ft = |md_at_u(τ) − md|`, and a NaN inverse drops the sample.
/// `moved` is `score_ft > refresh_tol_ft`, strictly.
///
/// Parallel over wells; result ascending id. NaN rule: skip. Errors: `Config`
/// for a bad `cfg`; `InvalidInput` if a well has no layer coordinate or layer
/// time; `Shape` if either length differs from the well's grid. Zero wells:
/// empty result. Complexity: `O(n log n)` per well.
pub fn score_wells(inputs: &RefreshInputs<'_>, cfg: &RefreshConfig) -> SxResult<Vec<WellScore>> {
    check_config(cfg)?;
    let wells: Vec<(WellId, GridRange, &[f64], &[f64])> = inputs
        .logs_filled
        .iter()
        .map(|(&w, logs)| {
            let (u, tau) = well_coords(inputs, w, logs.grid)?;
            Ok((w, logs.grid, u, tau))
        })
        .collect::<SxResult<_>>()?;
    let scored: Vec<SxResult<WellScore>> = wells
        .par_iter()
        .map(|&(w, grid, u, tau)| score_well(w, grid, u, tau, inputs.n_tops, cfg))
        .collect();
    scored.into_iter().collect()
}

/// Full refresh: score, then once re-fill the moved wells on τ and merge.
///
/// Moved wells get neighbor curves rebuilt on τ (each donor through its own τ)
/// and are re-filled by `refill_wells` with `models` (no training); its report
/// is discarded. Wells not moved are copied from `logs_filled`. No moved well:
/// neither neighbors nor fill is called. Takes `inputs` by value so `nbr_tau`
/// moves into the fill inputs.
///
/// NaN rule: skip (as [`score_wells`]). Errors: those of [`score_wells`] and
/// [`merge_refilled`]; errors from neighbors or fill unchanged. Zero wells:
/// empty output.
pub fn run_refresh(
    inputs: RefreshInputs<'_>,
    models: &TrainedModels,
    cfg: &RefreshConfig,
    neighbors_cfg: &NeighborsConfig,
    fill_cfg: &FillConfig,
) -> SxResult<RefreshOutput> {
    let scores = score_wells(&inputs, cfg)?;
    let moved: Vec<WellId> = scores.iter().filter(|s| s.moved).map(|s| s.well).collect();
    let (logs, refilled) = if moved.is_empty() {
        (inputs.logs_filled.clone(), BTreeMap::new())
    } else {
        let curves = build_neighbor_curves(&inputs.nbr_tau, &moved, neighbors_cfg)?;
        let fill_inputs = FillInputs {
            nbr: inputs.nbr_tau,
            neighbors: &curves,
            resd_family: inputs.resd_family,
            coord_source: inputs.coord_source,
            neighbors_cfg,
        };
        let out = refill_wells(&fill_inputs, fill_cfg, models, &moved)?;
        merge_refilled(inputs.logs_filled, out.logs)?
    };
    let report: Vec<RefreshRow> = scores
        .iter()
        .map(|s| RefreshRow {
            well: s.well,
            score_ft: s.score_ft,
            moved: s.moved,
            n_samples_refilled: refilled.get(&s.well).copied().unwrap_or(0),
        })
        .collect();
    info!(
        n_wells = report.len(),
        n_moved = moved.len(),
        n_samples_refilled = refilled.values().sum::<usize>(),
        "refresh done"
    );
    Ok(RefreshOutput { logs, report })
}

/// `logs_final` as an Arrow batch: the `fill::logs_filled_to_batch` schema
/// (rows by `well_id` then `k`, curves then provenance) with metadata
/// `subsurfix_schema = "subsurfix.logs_final/1"`.
///
/// NaN rule: stored as is. Errors: those of `logs_filled_to_batch`. Empty map: zero rows.
pub fn logs_final_batch(logs: &BTreeMap<WellId, WellLogs>) -> SxResult<RecordBatch> {
    with_schema_tag(logs_filled_to_batch(logs)?, LOGS_FINAL_TAG)
}

/// `refresh_report` as an Arrow batch: `well_id` Int32, `score_ft` Float64,
/// `moved` Boolean, `n_samples_refilled` Int64, rows as given, metadata
/// `subsurfix_schema = "subsurfix.refresh_report/1"`.
///
/// NaN rule: NaN `score_ft` stored as is. Errors: `InvalidInput` if a count
/// exceeds `i64`. Empty `rows`: zero rows.
pub fn report_batch(rows: &[RefreshRow]) -> SxResult<RecordBatch> {
    let counts = rows
        .iter()
        .map(|r| {
            i64::try_from(r.n_samples_refilled).map_err(|_| {
                SxError::InvalidInput(format!("refresh_report: well {} count too large", r.well.0))
            })
        })
        .collect::<SxResult<Vec<i64>>>()?;
    let fields = vec![
        Field::new("well_id", DataType::Int32, false),
        Field::new("score_ft", DataType::Float64, false),
        Field::new("moved", DataType::Boolean, false),
        Field::new("n_samples_refilled", DataType::Int64, false),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.well.0))),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.score_ft),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.moved).collect::<Vec<bool>>(),
        )),
        Arc::new(Int64Array::from(counts)),
    ];
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("cannot build refresh_report batch: {e}")))?;
    with_schema_tag(batch, REFRESH_REPORT_TAG)
}

/// Merge refilled wells into the fill output and check invariants.
/// Returns all wells and n_samples_refilled per refilled well.
///
/// Per refilled well: same grid and logs (`Shape`), `new.validate()`, measured
/// samples keep code 0 and bit-identical values and `CODE_NONE` stays
/// `CODE_NONE` (`Numerical`). A refilled well not in `filled`: `InvalidInput`.
pub(crate) fn merge_refilled(
    filled: &BTreeMap<WellId, WellLogs>,
    mut refilled: BTreeMap<WellId, WellLogs>,
) -> SxResult<(BTreeMap<WellId, WellLogs>, BTreeMap<WellId, usize>)> {
    if let Some(w) = refilled.keys().find(|w| !filled.contains_key(w)) {
        return Err(SxError::InvalidInput(format!(
            "refresh: refilled well {} is not in logs_filled",
            w.0
        )));
    }
    let mut logs = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for (&w, old) in filled {
        match refilled.remove(&w) {
            Some(new) => {
                counts.insert(w, check_refilled(old, &new)?);
                logs.insert(w, new);
            }
            None => {
                logs.insert(w, old.clone());
            }
        }
    }
    Ok((logs, counts))
}

fn check_config(cfg: &RefreshConfig) -> SxResult<()> {
    if !(cfg.refresh_tol_ft.is_finite() && cfg.refresh_tol_ft >= 0.0) {
        return Err(SxError::Config(format!(
            "refresh.refresh_tol_ft must be finite and >= 0, got {}",
            cfg.refresh_tol_ft
        )));
    }
    if !(cfg.percentile.is_finite() && (0.0..=100.0).contains(&cfg.percentile)) {
        return Err(SxError::Config(format!(
            "refresh.percentile must be in [0, 100], got {}",
            cfg.percentile
        )));
    }
    Ok(())
}

/// `u` and `τ` of well `w`, both of length `grid.len`.
fn well_coords<'a>(
    inputs: &RefreshInputs<'a>,
    w: WellId,
    grid: GridRange,
) -> SxResult<(&'a [f64], &'a [f64])> {
    let (Some(u), Some(tau)) = (inputs.layer_coord.get(&w), inputs.nbr_tau.coord.get(&w)) else {
        return Err(SxError::InvalidInput(format!(
            "refresh: well {} has no layer coordinate / layer time",
            w.0
        )));
    };
    for (name, len) in [("layer coordinate", u.len()), ("layer time", tau.len())] {
        if len != grid.len {
            return Err(SxError::Shape(format!(
                "refresh: well {}: {name} has {len} samples, grid has {}",
                w.0, grid.len
            )));
        }
    }
    Ok((u.as_slice(), tau.as_slice()))
}

fn score_well(
    well: WellId,
    grid: GridRange,
    u: &[f64],
    tau: &[f64],
    n_tops: usize,
    cfg: &RefreshConfig,
) -> SxResult<WellScore> {
    let inverse = CoordInverse::new(grid, u)?;
    let last_inner = i32::try_from(n_tops).unwrap_or(i32::MAX) - 1;
    let moved_ft: Vec<f64> = u
        .iter()
        .zip(tau)
        .enumerate()
        .filter(|&(_, (&ui, &ti))| {
            let id = interval_of_coord(ui, n_tops);
            ui.is_finite() && ti.is_finite() && (1..=last_inner).contains(&id)
        })
        .map(|(i, (_, &ti))| (inverse.md_at(ti) - grid.md_ft(i)).abs())
        .filter(|m| !m.is_nan())
        .collect();
    let score_ft = percentile(&moved_ft, cfg.percentile)?;
    let moved = score_ft.is_finite() && score_ft > cfg.refresh_tol_ft;
    debug!(
        well = well.0,
        score_ft,
        moved,
        n_scored = moved_ft.len(),
        "refresh score"
    );
    Ok(WellScore {
        well,
        score_ft,
        moved,
        n_scored: moved_ft.len(),
    })
}

/// Invariants of one refilled well; returns its count of re-filled samples.
fn check_refilled(old: &WellLogs, new: &WellLogs) -> SxResult<usize> {
    let w = old.id.0;
    if new.grid != old.grid || !old.curves.keys().eq(new.curves.keys()) {
        return Err(SxError::Shape(format!(
            "refresh: well {w}: refilled grid or log set differs from logs_filled"
        )));
    }
    new.validate()?;
    let mut count = 0;
    for (log, old_values) in &old.curves {
        let missing = || SxError::Shape(format!("refresh: well {w}: {} incomplete", log.name()));
        let old_prov = old.provenance.get(log).ok_or_else(missing)?;
        let new_values = new.curves.get(log).ok_or_else(missing)?;
        let new_prov = new.provenance.get(log).ok_or_else(missing)?;
        if old_values.len() != old.grid.len || old_prov.len() != old.grid.len {
            return Err(missing());
        }
        let samples = old_values
            .iter()
            .zip(old_prov)
            .zip(new_values.iter().zip(new_prov));
        for (i, ((v0, &p0), (v1, &p1))) in samples.enumerate() {
            let k = old.grid.k(i);
            if p0 == Provenance::Measured.code() {
                if p1 != p0 || v0.to_bits() != v1.to_bits() {
                    return Err(SxError::Numerical(format!(
                        "refresh changed measured sample: well {w}, {}, {k}",
                        log.name()
                    )));
                }
            } else if p0 == CODE_NONE {
                if p1 != CODE_NONE {
                    return Err(SxError::Numerical(format!(
                        "refresh filled a sample outside coverage: well {w}, {}, {k}",
                        log.name()
                    )));
                }
            } else if is_fill_code(p1) {
                count += 1;
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::fill::sequential::testkit::Project;
    use crate::neighbors::{Coverage, IntervalFacts, PairQuality};
    use crate::types::{LogKind, WellGeometry, WellHeader};

    fn no_models() -> TrainedModels {
        TrainedModels {
            models: BTreeMap::new(),
            n_train_rows: BTreeMap::new(),
        }
    }

    /// Empty neighbor maps for inputs that only score (never re-fill).
    #[derive(Default)]
    struct NoNeighbors {
        headers: BTreeMap<WellId, WellHeader>,
        geometry: BTreeMap<WellId, WellGeometry>,
        logs: BTreeMap<WellId, WellLogs>,
        facts: BTreeMap<WellId, Vec<IntervalFacts>>,
        pairs: PairQuality,
        coverage: BTreeMap<WellId, Coverage>,
        refs: BTreeMap<(WellId, LogKind), (f64, f64)>,
        exclude: BTreeSet<(WellId, LogKind)>,
        holdout: BTreeSet<WellId>,
        family: BTreeMap<WellId, u8>,
        source: BTreeMap<WellId, u8>,
    }

    impl NoNeighbors {
        fn inputs<'a>(
            &'a self,
            tau: &'a BTreeMap<WellId, Vec<f64>>,
            u: &'a BTreeMap<WellId, Vec<f64>>,
            filled: &'a BTreeMap<WellId, WellLogs>,
            n_tops: usize,
        ) -> RefreshInputs<'a> {
            RefreshInputs {
                nbr_tau: NeighborInputs {
                    headers: &self.headers,
                    geometry: &self.geometry,
                    logs: &self.logs,
                    coord: tau,
                    facts: &self.facts,
                    pairs: &self.pairs,
                    coverage: &self.coverage,
                    ref_ranges: &self.refs,
                    exclude: &self.exclude,
                    donor_exclude_wells: &self.holdout,
                },
                resd_family: &self.family,
                coord_source: &self.source,
                n_tops,
                layer_coord: u,
                logs_filled: filled,
            }
        }
    }

    fn gr_well(id: WellId, grid: GridRange) -> WellLogs {
        let mut logs = WellLogs::new(id, grid);
        let n = grid.len;
        logs.curves
            .insert(LogKind::Gr, (0..n).map(|i| 50.0 + i as f32).collect());
        logs.provenance.insert(LogKind::Gr, vec![0; n]);
        logs
    }

    #[test]
    fn refresh_sheet_worked_example() {
        // Rough interval 3 spans 5,924.5–6,124.5 ft: u = 3 + (md − 5,924.5)/200.
        let grid = GridRange::new(11849, 401).unwrap();
        let u: Vec<f64> = (0..grid.len)
            .map(|i| 3.0 + (grid.md_ft(i) - 5924.5) / 200.0)
            .collect();
        let inverse = CoordInverse::new(grid, &u).unwrap();
        let i = grid.index_of_k(12000).unwrap();
        assert_eq!(grid.md_ft(i), 6000.0);
        let z = inverse.md_at(3.40);
        assert!((z - 6004.5).abs() < 1e-9, "{z}");
        assert!(((z - grid.md_ft(i)).abs() - 4.5).abs() < 1e-9);

        // Layers sit 6.2 ft deeper everywhere except the 4.5 ft sample: P95 6.2 > 5 → refreshed.
        let mut tau: Vec<f64> = (0..grid.len)
            .map(|i| 3.0 + (grid.md_ft(i) + 6.2 - 5924.5) / 200.0)
            .collect();
        tau[i] = 3.40;
        let w = WellId(7);
        let nn = NoNeighbors::default();
        let (tau, u) = (BTreeMap::from([(w, tau)]), BTreeMap::from([(w, u)]));
        let filled = BTreeMap::from([(w, gr_well(w, grid))]);
        let inputs = nn.inputs(&tau, &u, &filled, 5);
        let scores = score_wells(&inputs, &RefreshConfig::default()).unwrap();
        assert_eq!(scores.len(), 1);
        assert!((scores[0].score_ft - 6.2).abs() < 1e-9, "{:?}", scores[0]);
        assert!(scores[0].moved);
    }

    #[test]
    fn refresh_below_tolerance_untouched() {
        // MD 49.0 … 52.0, intervals 0 and 2 are pads (2 tops).
        let grid = GridRange::new(98, 7).unwrap();
        let w = WellId(1);
        let u = vec![0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
        let tau = vec![f64::NAN, f64::NAN, 1.0, 1.3, 1.6, 1.75, 2.0];
        let nn = NoNeighbors::default();
        let (tau, u) = (BTreeMap::from([(w, tau)]), BTreeMap::from([(w, u)]));
        let filled = BTreeMap::from([(w, gr_well(w, grid))]);
        let inputs = nn.inputs(&tau, &u, &filled, 2);
        let out = run_refresh(
            inputs,
            &no_models(),
            &RefreshConfig::default(),
            &NeighborsConfig::default(),
            &FillConfig::default(),
        )
        .unwrap();
        assert_eq!(out.logs, filled);
        assert_eq!(out.report.len(), 1);
        let row = out.report[0];
        assert!((row.score_ft - 0.185).abs() < 1e-9, "{row:?}");
        assert!(!row.moved);
        assert_eq!(row.n_samples_refilled, 0);

        let batch = report_batch(&out.report).unwrap();
        let cols: Vec<(String, DataType)> = batch
            .schema_ref()
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect();
        assert_eq!(
            cols,
            vec![
                ("well_id".into(), DataType::Int32),
                ("score_ft".into(), DataType::Float64),
                ("moved".into(), DataType::Boolean),
                ("n_samples_refilled".into(), DataType::Int64),
            ]
        );
        assert_eq!(
            crate::io::parquet::schema_tag(&batch),
            Some(REFRESH_REPORT_TAG)
        );
    }

    #[test]
    fn refresh_measured_samples_never_change() {
        let p = Project::new();
        let first = p.inputs();
        let wells = [WellId(1), WellId(2), WellId(3)];
        let filled = refill_wells(&first, &FillConfig::default(), &no_models(), &wells)
            .unwrap()
            .logs;
        // u spans intervals 1..=3 of 4 tops (50 ft each); well 1's layers sit 10 ft deeper on τ.
        let grid = filled[&WellId(1)].grid;
        let u: Vec<f64> = (0..grid.len).map(|i| 1.0 + 0.01 * i as f64).collect();
        let mut tau = BTreeMap::new();
        for w in wells {
            let shift = if w == WellId(1) { 0.2 } else { 0.0 };
            tau.insert(w, u.iter().map(|x| x + shift).collect::<Vec<f64>>());
        }
        let u: BTreeMap<WellId, Vec<f64>> = wells.iter().map(|&w| (w, u.clone())).collect();
        let inputs = RefreshInputs {
            nbr_tau: NeighborInputs {
                coord: &tau,
                ..first.nbr
            },
            resd_family: first.resd_family,
            coord_source: first.coord_source,
            n_tops: 4,
            layer_coord: &u,
            logs_filled: &filled,
        };
        let out = run_refresh(
            inputs,
            &no_models(),
            &RefreshConfig::default(),
            &NeighborsConfig::default(),
            &FillConfig::default(),
        )
        .unwrap();

        let moved: Vec<(WellId, bool)> = out.report.iter().map(|r| (r.well, r.moved)).collect();
        assert_eq!(
            moved,
            vec![(WellId(1), true), (WellId(2), false), (WellId(3), false)]
        );
        assert!((out.report[0].score_ft - 10.0).abs() < 1e-9);
        for w in [WellId(2), WellId(3)] {
            assert_eq!(out.logs[&w], filled[&w]);
        }
        for (w, old) in &filled {
            let new = &out.logs[w];
            for (log, v0) in &old.curves {
                let (p0, v1, p1) = (&old.provenance[log], &new.curves[log], &new.provenance[log]);
                for i in 0..v0.len() {
                    if p0[i] == Provenance::Measured.code() {
                        assert_eq!((v1[i].to_bits(), p1[i]), (v0[i].to_bits(), 0));
                    }
                }
            }
        }
        let batch = logs_final_batch(&out.logs).unwrap();
        assert_eq!(batch.num_rows(), 3 * grid.len);
        assert_eq!(crate::io::parquet::schema_tag(&batch), Some(LOGS_FINAL_TAG));
    }
}
