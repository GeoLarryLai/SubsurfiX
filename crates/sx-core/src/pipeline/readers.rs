// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Read-back of stage products.
//!
//! Stages always load their inputs from files, so a skipped and a fresh
//! upstream stage give the same data. Every reader checks the
//! `subsurfix_schema` tag (`SxError::Format` on mismatch), returns wells
//! sorted by `WellId`, and checks that `k` rises by one per row within a well.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, AsArray, ListArray, PrimitiveArray, StringArray};
use arrow::datatypes::{
    ArrowPrimitiveType, Field, Float32Type, Float64Type, Int32Type, Schema, UInt8Type,
};
use arrow::record_batch::RecordBatch;

use crate::align::{PairMap, PairStatus, UserPick};
use crate::config::ProjectConfig;
use crate::correlate::solve::LayerTimeWell;
use crate::error::{SxError, SxResult};
use crate::fill::lgbm::LgbmParams;
use crate::fill::sequential::TrainedModels;
use crate::io::atomic::io_error;
use crate::io::csv::read_csv;
use crate::io::parquet::{read_batch, read_batch_expect};
use crate::io::products::{
    geometry_from_batch, logs_from_batch, tops_input_from_batch, wells_from_batch, GEOMETRY_TAG,
    TOPS_INPUT_TAG, WELLS_TAG,
};
use crate::neighbors::{neighbor_curves_from_batch, WellNeighborCurves};
use crate::normalize::{PriorChannel, LOGS_NORM_TAG};
use crate::qc::QcFlags;
use crate::types::{
    GridRange, IntervalRow, IntervalState, LogKind, PairPath, TopPick, TopSet, TopSource,
    WellGeometry, WellHeader, WellId, WellLogs,
};

pub(crate) const INTERVALS_TAG: &str = "subsurfix.intervals/1";
pub(crate) const TOPS_ESTIMATED_TAG: &str = "subsurfix.tops_estimated/1";
pub(crate) const PAIRS_ROUGH_TAG: &str = "subsurfix.pairs_rough/1";
pub(crate) const LAYER_COORD_TAG: &str = "subsurfix.layer_coord/1";
pub(crate) const LAYER_TIME_TAG: &str = "subsurfix.layer_time/1";

/// RESD value column of `logs_norm`, `logs_filled` and `logs_final`.
const RESD_LOG10_COLUMN: &str = "resd_log10ohmm";

fn format_at(path: &Path, msg: impl std::fmt::Display) -> SxError {
    SxError::Format(format!("{}: {msg}", path.display()))
}

/// Prefix a `Format` error with the file path.
fn at(path: &Path) -> impl Fn(SxError) -> SxError + '_ {
    move |e| match e {
        SxError::Format(m) => format_at(path, m),
        other => other,
    }
}

fn bad_column(path: &Path, name: &str) -> SxError {
    format_at(
        path,
        format!("column {name} is missing, of the wrong type, or has nulls"),
    )
}

fn prim<'a, T: ArrowPrimitiveType>(
    b: &'a RecordBatch,
    path: &Path,
    name: &str,
) -> SxResult<&'a [T::Native]> {
    b.column_by_name(name)
        .filter(|c| c.null_count() == 0)
        .and_then(|c| c.as_primitive_opt::<T>())
        .map(|a: &PrimitiveArray<T>| a.values().as_ref())
        .ok_or_else(|| bad_column(path, name))
}

fn strings<'a>(b: &'a RecordBatch, path: &Path, name: &str) -> SxResult<&'a StringArray> {
    b.column_by_name(name)
        .filter(|c| c.null_count() == 0)
        .and_then(|c| c.as_string_opt::<i32>())
        .ok_or_else(|| bad_column(path, name))
}

fn lists<'a>(b: &'a RecordBatch, path: &Path, name: &str) -> SxResult<&'a ListArray> {
    b.column_by_name(name)
        .filter(|c| c.null_count() == 0)
        .and_then(|c| c.as_list_opt::<i32>())
        .ok_or_else(|| bad_column(path, name))
}

/// Row `r` of a list column as a vector of `T`.
fn list_row<T: ArrowPrimitiveType>(
    col: &ListArray,
    path: &Path,
    name: &str,
    r: usize,
) -> SxResult<Vec<T::Native>> {
    let item = col.value(r);
    let values = item
        .as_primitive_opt::<T>()
        .filter(|a| a.null_count() == 0)
        .ok_or_else(|| bad_column(path, name))?;
    Ok(values.values().to_vec())
}

fn slice<T: Clone>(values: &[T], rows: &Range<usize>, path: &Path) -> SxResult<Vec<T>> {
    values
        .get(rows.clone())
        .map(<[T]>::to_vec)
        .ok_or_else(|| format_at(path, "column shorter than well_id"))
}

/// One well's contiguous rows of a per-sample table.
struct Run {
    id: WellId,
    grid: GridRange,
    rows: Range<usize>,
}

/// Split per-sample rows into wells: ids strictly ascending between wells,
/// `k` rising by one per row within a well.
fn runs(well: &[i32], k: &[i32], path: &Path) -> SxResult<Vec<Run>> {
    let mut out: Vec<Run> = Vec::new();
    for (r, (&w, &kk)) in well.iter().zip(k).enumerate() {
        match out.last_mut() {
            Some(run) if run.id.0 == w => {
                let want = i64::from(run.grid.k_start) + run.rows.len() as i64;
                if i64::from(kk) != want {
                    return Err(format_at(
                        path,
                        format!("row {r}: k {kk} breaks the grid run of well {w}"),
                    ));
                }
                run.rows.end = r + 1;
            }
            Some(run) if w < run.id.0 => {
                return Err(format_at(
                    path,
                    format!("row {r}: well_id {w} after {} (ids must ascend)", run.id.0),
                ));
            }
            _ => out.push(Run {
                id: WellId(w),
                grid: GridRange {
                    k_start: kk,
                    len: 0,
                },
                rows: r..r + 1,
            }),
        }
    }
    for run in &mut out {
        run.grid = GridRange::new(run.grid.k_start, run.rows.len()).map_err(at(path))?;
    }
    Ok(out)
}

/// Same batch with column `from` renamed to `to`.
fn rename_column(b: &RecordBatch, from: &str, to: &str) -> SxResult<RecordBatch> {
    let schema = b.schema();
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let f = f.as_ref().clone();
            if f.name() == from {
                f.with_name(to)
            } else {
                f
            }
        })
        .collect();
    let renamed = Schema::new_with_metadata(fields, schema.metadata().clone());
    RecordBatch::try_new(Arc::new(renamed), b.columns().to_vec())
        .map_err(|e| SxError::Format(format!("cannot rename column {from}: {e}")))
}

// ------------------------------------------------------------------ ingest

/// `wells.parquet`: headers sorted by id and `resd_family_code` per well.
pub(crate) fn read_wells(path: &Path) -> SxResult<(Vec<WellHeader>, BTreeMap<WellId, u8>)> {
    let rows = wells_from_batch(&read_batch_expect(path, WELLS_TAG)?).map_err(at(path))?;
    let mut headers: Vec<WellHeader> = rows.iter().map(|r| r.header.clone()).collect();
    headers.sort_by_key(|h| h.id);
    let family = rows
        .iter()
        .map(|r| (r.header.id, r.resd_family_code()))
        .collect();
    Ok((headers, family))
}

/// `geometry.parquet`, ascending id.
pub(crate) fn read_geometry(path: &Path) -> SxResult<Vec<WellGeometry>> {
    geometry_from_batch(&read_batch_expect(path, GEOMETRY_TAG)?).map_err(at(path))
}

/// A logs-layout file (`logs_raw`, `logs_qc`) tagged `tag`: provenance `0`
/// where finite, `CODE_NONE` elsewhere.
pub(crate) fn read_logs(path: &Path, tag: &str) -> SxResult<Vec<WellLogs>> {
    logs_from_batch(&read_batch_expect(path, tag)?).map_err(at(path))
}

/// `aux_raw.parquet` (caliper / density correction); empty when absent.
pub(crate) fn read_aux(path: &Path) -> SxResult<Vec<crate::types::WellAux>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    crate::io::products::aux_from_batch(&read_batch_expect(path, crate::io::products::AUX_RAW_TAG)?)
        .map_err(at(path))
}

/// `logs_norm.parquet` (RESD column `resd_log10ohmm`, values kept in log10).
pub(crate) fn read_logs_norm(path: &Path) -> SxResult<Vec<WellLogs>> {
    let b = read_batch_expect(path, LOGS_NORM_TAG)?;
    let b = rename_column(&b, RESD_LOG10_COLUMN, LogKind::Resd.column()).map_err(at(path))?;
    logs_from_batch(&b).map_err(at(path))
}

/// `tops_input.parquet`: the user top list and picks as [`UserPick`].
pub(crate) fn read_tops_input(path: &Path) -> SxResult<(TopSet, Vec<UserPick>)> {
    let (tops, rows) =
        tops_input_from_batch(&read_batch_expect(path, TOPS_INPUT_TAG)?).map_err(at(path))?;
    let picks = rows
        .into_iter()
        .map(|r| UserPick {
            well: r.well,
            name: r.top_name,
            md_ft: r.md_ft,
        })
        .collect();
    Ok((tops, picks))
}

// ---------------------------------------------------------------- qc flags

/// `qc_flags` / `qc_flags_post` (tag `tag`): bits per log per well.
pub(crate) fn read_flags(path: &Path, tag: &str) -> SxResult<Vec<QcFlags>> {
    let b = read_batch_expect(path, tag)?;
    let well = prim::<Int32Type>(&b, path, "well_id")?;
    let k = prim::<Int32Type>(&b, path, "k")?;
    let mut cols: Vec<(LogKind, &[u8])> = Vec::with_capacity(LogKind::ALL.len());
    for log in LogKind::ALL {
        let name = format!("{}_qc_bits", log.name().to_ascii_lowercase());
        cols.push((log, prim::<UInt8Type>(&b, path, &name)?));
    }
    runs(well, k, path)?
        .into_iter()
        .map(|run| {
            let mut bits = BTreeMap::new();
            for (log, v) in &cols {
                bits.insert(*log, slice(v, &run.rows, path)?);
            }
            Ok(QcFlags {
                id: run.id,
                grid: run.grid,
                bits,
            })
        })
        .collect()
}

// --------------------------------------------------------------------- csv

/// Header and rows of a CSV product, with a column lookup.
struct Csv {
    path: PathBuf,
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Csv {
    fn read(path: &Path) -> SxResult<Self> {
        let (header, rows) = read_csv(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            rows,
        })
    }

    fn col(&self, name: &str) -> SxResult<usize> {
        self.header
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format_at(&self.path, format!("missing column {name}")))
    }

    fn cell<'a>(&self, row: &'a [String], c: usize) -> SxResult<&'a str> {
        row.get(c)
            .map(String::as_str)
            .ok_or_else(|| format_at(&self.path, "short row"))
    }

    fn well_log(&self, row: &[String], cw: usize, cl: usize) -> SxResult<(WellId, LogKind)> {
        let w = self.cell(row, cw)?;
        let id = w
            .parse::<i32>()
            .map_err(|_| format_at(&self.path, format!("bad well_id {w:?}")))?;
        let log = LogKind::from_name(self.cell(row, cl)?).map_err(|e| format_at(&self.path, e))?;
        Ok((WellId(id), log))
    }

    /// Float cell; `""` is NaN.
    fn f64(&self, row: &[String], c: usize) -> SxResult<f64> {
        let s = self.cell(row, c)?;
        if s.is_empty() {
            return Ok(f64::NAN);
        }
        s.parse::<f64>()
            .map_err(|_| format_at(&self.path, format!("bad number {s:?}")))
    }
}

/// `qc_report.csv` → one [`PriorChannel`] per row: `sigma_bits`, and
/// `rescued = (rescue_reason == "rescued")`.
pub(crate) fn read_prior(path: &Path) -> SxResult<Vec<PriorChannel>> {
    let csv = Csv::read(path)?;
    let (cw, cl) = (csv.col("well_id")?, csv.col("log")?);
    let (cs, cr) = (csv.col("sigma_bits")?, csv.col("rescue_reason")?);
    let mut out = Vec::with_capacity(csv.rows.len());
    for row in &csv.rows {
        let (well, log) = csv.well_log(row, cw, cl)?;
        let s = csv.cell(row, cs)?;
        let sigma_bits = s
            .parse::<u8>()
            .map_err(|_| format_at(path, format!("bad sigma_bits {s:?}")))?;
        out.push(PriorChannel {
            well,
            log,
            sigma_bits,
            rescued: csv.cell(row, cr)? == "rescued",
        });
    }
    out.sort_by_key(|p| (p.well, p.log));
    Ok(out)
}

/// `norm_anchors.csv` → `(ref_low_native, ref_high_native)` per channel where
/// both are finite (RESD in log10).
pub(crate) fn read_ref_ranges(path: &Path) -> SxResult<BTreeMap<(WellId, LogKind), (f64, f64)>> {
    let csv = Csv::read(path)?;
    let (cw, cl) = (csv.col("well_id")?, csv.col("log")?);
    let (lo, hi) = (csv.col("ref_low_native")?, csv.col("ref_high_native")?);
    let mut out = BTreeMap::new();
    for row in &csv.rows {
        let key = csv.well_log(row, cw, cl)?;
        let (a, b) = (csv.f64(row, lo)?, csv.f64(row, hi)?);
        if a.is_finite() && b.is_finite() {
            out.insert(key, (a, b));
        }
    }
    Ok(out)
}

/// `norm_report.csv` → channels with `jsd_outlier = true`.
pub(crate) fn read_jsd_outliers(path: &Path) -> SxResult<BTreeSet<(WellId, LogKind)>> {
    let csv = Csv::read(path)?;
    let (cw, cl, co) = (
        csv.col("well_id")?,
        csv.col("log")?,
        csv.col("jsd_outlier")?,
    );
    let mut out = BTreeSet::new();
    for row in &csv.rows {
        if csv.cell(row, co)? == "true" {
            out.insert(csv.well_log(row, cw, cl)?);
        }
    }
    Ok(out)
}

// ------------------------------------------------------------ rough align

fn interval_state(code: u8) -> Option<IntervalState> {
    Some(match code {
        0 => IntervalState::Observed,
        1 => IntervalState::Pinched,
        2 => IntervalState::Estimated,
        3 => IntervalState::Unknown,
        4 => IntervalState::Pad,
        _ => return None,
    })
}

/// `intervals.parquet` → the study top list (top names of intervals `1..=n`
/// of the first well) and every row, sorted by `(well, interval)`.
pub(crate) fn read_intervals(path: &Path) -> SxResult<(TopSet, Vec<IntervalRow>)> {
    let b = read_batch_expect(path, INTERVALS_TAG)?;
    let well = prim::<Int32Type>(&b, path, "well_id")?;
    let interval = prim::<Int32Type>(&b, path, "interval")?;
    let top_name = strings(&b, path, "top_name")?;
    let top = prim::<Float64Type>(&b, path, "top_md_ft")?;
    let base = prim::<Float64Type>(&b, path, "base_md_ft")?;
    let state = prim::<UInt8Type>(&b, path, "state")?;
    let n_rows = b.num_rows();
    let mut rows = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let get = |v: &[f64]| v.get(r).copied().unwrap_or(f64::NAN);
        let (Some(&w), Some(&i), Some(&s)) = (well.get(r), interval.get(r), state.get(r)) else {
            return Err(format_at(path, "column shorter than well_id"));
        };
        rows.push(IntervalRow {
            well: WellId(w),
            interval: u16::try_from(i)
                .map_err(|_| format_at(path, format!("row {r}: bad interval {i}")))?,
            top_md_ft: get(top),
            base_md_ft: get(base),
            state: interval_state(s)
                .ok_or_else(|| format_at(path, format!("row {r}: bad state {s}")))?,
        });
    }
    if rows
        .windows(2)
        .any(|p| matches!(p, [x, y] if (x.well, x.interval) >= (y.well, y.interval)))
    {
        return Err(format_at(
            path,
            "rows are not sorted by (well_id, interval)",
        ));
    }
    let first = rows.first().map(|r| r.well);
    let names: Vec<String> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| Some(row.well) == first)
        .map(|(r, _)| top_name.value(r).to_string())
        .collect();
    let n = names.len().saturating_sub(1);
    let tops = TopSet::new(names.into_iter().skip(1).take(n).collect()).map_err(at(path))?;
    Ok((tops, rows))
}

fn top_source(code: u8) -> Option<TopSource> {
    match code {
        0 => Some(TopSource::User),
        1 => Some(TopSource::Estimated),
        2 => Some(TopSource::Pad),
        3 => Some(TopSource::Suggested),
        _ => None,
    }
}

/// `tops_estimated.parquet` → kept boundaries `1..=n_tops` as [`TopPick`]
/// (`top = boundary − 1`, source from the `source` column). These are the
/// user picks and estimated tops the interval table was built from.
pub(crate) fn read_estimated_picks(path: &Path, n_tops: usize) -> SxResult<Vec<TopPick>> {
    let b = read_batch_expect(path, TOPS_ESTIMATED_TAG)?;
    let well = prim::<Int32Type>(&b, path, "well_id")?;
    let boundary = prim::<Int32Type>(&b, path, "boundary")?;
    let md = prim::<Float64Type>(&b, path, "md_ft")?;
    let source = prim::<UInt8Type>(&b, path, "source")?;
    let status = prim::<UInt8Type>(&b, path, "status")?;
    let mut out = Vec::new();
    for r in 0..b.num_rows() {
        let (Some(&w), Some(&bd), Some(&m), Some(&src), Some(&st)) = (
            well.get(r),
            boundary.get(r),
            md.get(r),
            source.get(r),
            status.get(r),
        ) else {
            return Err(format_at(path, "column shorter than well_id"));
        };
        let Ok(bd) = usize::try_from(bd) else {
            continue;
        };
        if st != 0 || bd == 0 || bd > n_tops {
            continue;
        }
        let Some(src) = top_source(src) else {
            continue;
        };
        out.push(TopPick {
            well: WellId(w),
            top: u16::try_from(bd - 1)
                .map_err(|_| format_at(path, format!("row {r}: boundary too large")))?,
            md_ft: m,
            source: src,
        });
    }
    out.sort_by_key(|p| (p.well, p.top));
    Ok(out)
}

fn pair_status(code: u8) -> Option<PairStatus> {
    match code {
        0 => Some(PairStatus::IntervalMap),
        1 => Some(PairStatus::DepthMap),
        2 => Some(PairStatus::Failed),
        _ => None,
    }
}

/// `pairs_rough.parquet` → one [`PairMap`] per row (`a = target_id`,
/// `b = donor_id`, `interval = None`, `path` = zip of `k_target`, `k_donor`,
/// `cost = total_cost`).
pub(crate) fn read_pairs(path: &Path) -> SxResult<Vec<PairMap>> {
    let b = read_batch_expect(path, PAIRS_ROUGH_TAG)?;
    let target = prim::<Int32Type>(&b, path, "target_id")?;
    let donor = prim::<Int32Type>(&b, path, "donor_id")?;
    let rank = prim::<Int32Type>(&b, path, "rank")?;
    let dist = prim::<Float64Type>(&b, path, "distance_m")?;
    let status = prim::<UInt8Type>(&b, path, "status")?;
    let quality = prim::<Float32Type>(&b, path, "quality")?;
    let span = prim::<Float64Type>(&b, path, "span_ratio")?;
    let total = prim::<Float64Type>(&b, path, "total_cost")?;
    let mean = prim::<Float64Type>(&b, path, "mean_cost")?;
    let n_dtw = prim::<Int32Type>(&b, path, "n_dtw_intervals")?;
    let n_lin = prim::<Int32Type>(&b, path, "n_linear_intervals")?;
    let intervals = lists(&b, path, "intervals")?;
    let iq = lists(&b, path, "interval_quality")?;
    let im = lists(&b, path, "interval_method")?;
    let kt = lists(&b, path, "k_target")?;
    let kd = lists(&b, path, "k_donor")?;
    let short = || format_at(path, "column shorter than target_id");
    let u16_of = |v: i32, what: &str| {
        u16::try_from(v).map_err(|_| format_at(path, format!("bad {what} {v}")))
    };
    let mut out = Vec::with_capacity(b.num_rows());
    for r in 0..b.num_rows() {
        let k_t = list_row::<Int32Type>(kt, path, "k_target", r)?;
        let k_d = list_row::<Int32Type>(kd, path, "k_donor", r)?;
        if k_t.len() != k_d.len() {
            return Err(format_at(
                path,
                format!("row {r}: k_target and k_donor differ in length"),
            ));
        }
        let st = *status.get(r).ok_or_else(short)?;
        out.push(PairMap {
            path: PairPath {
                a: WellId(*target.get(r).ok_or_else(short)?),
                b: WellId(*donor.get(r).ok_or_else(short)?),
                interval: None,
                path: k_t.into_iter().zip(k_d).collect(),
                cost: *total.get(r).ok_or_else(short)?,
                quality: *quality.get(r).ok_or_else(short)?,
            },
            rank: u16_of(*rank.get(r).ok_or_else(short)?, "rank")?,
            distance_m: *dist.get(r).ok_or_else(short)?,
            status: pair_status(st)
                .ok_or_else(|| format_at(path, format!("row {r}: bad status {st}")))?,
            span_ratio: *span.get(r).ok_or_else(short)?,
            mean_cost: *mean.get(r).ok_or_else(short)?,
            n_dtw_intervals: u16_of(*n_dtw.get(r).ok_or_else(short)?, "n_dtw_intervals")?,
            n_linear_intervals: u16_of(*n_lin.get(r).ok_or_else(short)?, "n_linear_intervals")?,
            intervals: list_row::<Int32Type>(intervals, path, "intervals", r)?
                .into_iter()
                .map(|i| u16_of(i, "interval"))
                .collect::<SxResult<_>>()?,
            interval_quality: list_row::<Float32Type>(iq, path, "interval_quality", r)?,
            interval_method: list_row::<UInt8Type>(im, path, "interval_method", r)?,
        });
    }
    out.sort_by_key(|m| (m.path.a, m.rank));
    Ok(out)
}

/// `layer_coord.parquet` → rough layer coordinate `u` per well and its
/// `coord_source` (first row of the well).
pub(crate) type LayerCoord = (BTreeMap<WellId, Vec<f64>>, BTreeMap<WellId, u8>);

/// See [`LayerCoord`].
pub(crate) fn read_layer_coord(path: &Path) -> SxResult<LayerCoord> {
    let b = read_batch_expect(path, LAYER_COORD_TAG)?;
    let well = prim::<Int32Type>(&b, path, "well_id")?;
    let k = prim::<Int32Type>(&b, path, "k")?;
    let u = prim::<Float64Type>(&b, path, "layer_coord")?;
    let source = prim::<UInt8Type>(&b, path, "coord_source")?;
    let mut coord = BTreeMap::new();
    let mut src = BTreeMap::new();
    for run in runs(well, k, path)? {
        coord.insert(run.id, slice(u, &run.rows, path)?);
        src.insert(run.id, source.get(run.rows.start).copied().unwrap_or(0));
    }
    Ok((coord, src))
}

// --------------------------------------------------------- later products

/// `neighbor_curves.parquet`.
pub(crate) fn read_neighbor_curves(path: &Path) -> SxResult<BTreeMap<WellId, WellNeighborCurves>> {
    neighbor_curves_from_batch(&read_batch(path)?).map_err(at(path))
}

/// `logs_filled` / `logs_final` (tag `tag`): values (RESD in log10) and
/// provenance per well.
pub(crate) fn read_logs_with_prov(path: &Path, tag: &str) -> SxResult<BTreeMap<WellId, WellLogs>> {
    let b = read_batch_expect(path, tag)?;
    let well = prim::<Int32Type>(&b, path, "well_id")?;
    let k = prim::<Int32Type>(&b, path, "k")?;
    prim::<Float64Type>(&b, path, "md_ft")?;
    let mut cols: Vec<(LogKind, &[f32], &[u8])> = Vec::with_capacity(LogKind::ALL.len());
    for log in LogKind::ALL {
        let value = if log == LogKind::Resd {
            RESD_LOG10_COLUMN
        } else {
            log.column()
        };
        let prov = format!("{}_prov", log.name().to_ascii_lowercase());
        cols.push((
            log,
            prim::<Float32Type>(&b, path, value)?,
            prim::<UInt8Type>(&b, path, &prov)?,
        ));
    }
    let mut out = BTreeMap::new();
    for run in runs(well, k, path)? {
        let mut logs = WellLogs::new(run.id, run.grid);
        for (log, v, p) in &cols {
            logs.curves.insert(*log, slice(v, &run.rows, path)?);
            logs.provenance.insert(*log, slice(p, &run.rows, path)?);
        }
        out.insert(run.id, logs);
    }
    Ok(out)
}

/// `layer_time.parquet` → one [`LayerTimeWell`] per well, ascending id.
pub(crate) fn read_layer_time(path: &Path) -> SxResult<Vec<LayerTimeWell>> {
    let b = read_batch_expect(path, LAYER_TIME_TAG)?;
    let well = prim::<Int32Type>(&b, path, "well_id")?;
    let k = prim::<Int32Type>(&b, path, "k")?;
    let tau = prim::<Float64Type>(&b, path, "layer_time")?;
    runs(well, k, path)?
        .into_iter()
        .map(|run| {
            Ok(LayerTimeWell {
                well: run.id,
                grid: run.grid,
                layer_time: slice(tau, &run.rows, path)?,
            })
        })
        .collect()
}

/// Rebuild the built-in fill models from `fill/models/<LOG>.txt` files.
///
/// # Errors
/// `SxError::Io` (read), `SxError::Model` (bad text).
pub(crate) fn read_models(
    files: &[(LogKind, PathBuf)],
    cfg: &ProjectConfig,
) -> SxResult<TrainedModels> {
    let mut texts = BTreeMap::new();
    for (log, path) in files {
        let text = std::fs::read_to_string(path).map_err(|e| io_error(path, e))?;
        texts.insert(*log, text);
    }
    TrainedModels::from_lgbm_texts(
        &texts,
        &LgbmParams::from_config(&cfg.fill.lightgbm),
        cfg.seed,
    )
}
