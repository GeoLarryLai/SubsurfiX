// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Ingest products as Arrow batches: `wells`, `geometry`, the logs layout
//! (`logs_raw` at ingest) and `tops_input`, each with a builder and a reader.
//!
//! All fields are non-nullable. Builders set their tag with
//! [`with_schema_tag`]. Readers check the columns they need by name and type
//! (`Format` naming the column otherwise), ignore extra columns, and do not
//! check the tag (use `read_batch_expect` for that).

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    StringArray, UInt16Array, UInt8Array,
};
use arrow::datatypes::{
    DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type, Schema, UInt16Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;

use super::parquet::with_schema_tag;
use crate::error::{SxError, SxResult};
use crate::types::{
    GridRange, LogKind, TopSet, TopSource, WellAux, WellGeometry, WellHeader, WellId, WellLogs,
    CODE_NONE, GRID_STEP_FT,
};

/// Tag of the `wells` product.
pub const WELLS_TAG: &str = "subsurfix.wells/1";
/// Tag of the `geometry` product.
pub const GEOMETRY_TAG: &str = "subsurfix.geometry/1";
/// Tag of the ingest logs product (`logs_raw.parquet`).
pub const LOGS_RAW_TAG: &str = "subsurfix.logs_raw/1";
/// Tag of the `tops_input` product.
pub const TOPS_INPUT_TAG: &str = "subsurfix.tops_input/1";

/// Family text written when a curve family is absent.
const UNKNOWN_FAMILY: &str = "unknown";

/// The four logs that have a family column in `wells`, in column order.
const FAMILY_LOGS: [(LogKind, &str); 4] = [
    (LogKind::Gr, "gr_family"),
    (LogKind::Nphi, "nphi_family"),
    (LogKind::Dtc, "dtc_family"),
    (LogKind::Resd, "resd_family"),
];

/// One row of the `wells` product.
#[derive(Debug, Clone, PartialEq)]
pub struct WellRow {
    /// Well header (id, name, surface x/y in m, KB elevation in ft).
    pub header: WellHeader,
    /// Grid range of the well's logs; `None` = no logs.
    pub grid: Option<GridRange>,
    /// A deviation survey was used.
    pub has_survey: bool,
    /// Largest survey inclination, degrees; NaN without survey.
    pub max_inc_deg: f64,
    /// Number of LAS files used for the well.
    pub n_las_files: i64,
    /// Curve family per log: GR, NPHI, DTC, RESD are used, others ignored.
    pub families: BTreeMap<LogKind, String>,
}

impl WellRow {
    /// Code of the RESD family: `induction` 1, `laterolog` 2, `lateral` 3,
    /// anything else (or absent) 0.
    pub fn resd_family_code(&self) -> u8 {
        match self.families.get(&LogKind::Resd).map(String::as_str) {
            Some("induction") => 1,
            Some("laterolog") => 2,
            Some("lateral") => 3,
            _ => 0,
        }
    }
}

/// One user pick of one top in one well.
#[derive(Debug, Clone, PartialEq)]
pub struct TopInputRow {
    /// Well id.
    pub well: WellId,
    /// Index into the project top list (shallow to deep, 0-based).
    pub top_idx: u16,
    /// Top name.
    pub top_name: String,
    /// Pick measured depth, ft.
    pub md_ft: f64,
    /// TVD at the pick, ft.
    pub tvd_ft: f64,
    /// Elevation at the pick, ft; NaN when KB is unknown.
    pub elev_ft: f64,
    /// Origin of the pick.
    pub source: TopSource,
}

// ---------------------------------------------------------------- helpers

fn field(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, false)
}

/// Build a tagged batch from non-nullable fields and columns.
fn finish(fields: Vec<Field>, columns: Vec<ArrayRef>, tag: &str) -> SxResult<RecordBatch> {
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, tag)
}

fn str_array<'a>(values: impl Iterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values))
}

/// Column `name` of type `dt` with no nulls, else `Format`.
fn need<'a>(batch: &'a RecordBatch, name: &str, dt: &DataType) -> SxResult<&'a ArrayRef> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| SxError::Format(format!("missing column {name}")))?;
    if col.data_type() != dt {
        return Err(SxError::Format(format!(
            "column {name}: expected {dt}, found {}",
            col.data_type()
        )));
    }
    if col.null_count() > 0 {
        return Err(SxError::Format(format!("column {name}: contains nulls")));
    }
    Ok(col)
}

fn wrong(name: &str) -> SxError {
    SxError::Format(format!("column {name}: unexpected array type"))
}

fn i32_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a Int32Array> {
    need(b, n, &DataType::Int32)?
        .as_primitive_opt::<Int32Type>()
        .ok_or_else(|| wrong(n))
}
fn i64_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a Int64Array> {
    need(b, n, &DataType::Int64)?
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| wrong(n))
}
fn u8_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a UInt8Array> {
    need(b, n, &DataType::UInt8)?
        .as_primitive_opt::<UInt8Type>()
        .ok_or_else(|| wrong(n))
}
fn u16_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a UInt16Array> {
    need(b, n, &DataType::UInt16)?
        .as_primitive_opt::<UInt16Type>()
        .ok_or_else(|| wrong(n))
}
fn f32_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a Float32Array> {
    need(b, n, &DataType::Float32)?
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| wrong(n))
}
fn f64_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a Float64Array> {
    need(b, n, &DataType::Float64)?
        .as_primitive_opt::<Float64Type>()
        .ok_or_else(|| wrong(n))
}
fn bool_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a BooleanArray> {
    need(b, n, &DataType::Boolean)?
        .as_boolean_opt()
        .ok_or_else(|| wrong(n))
}
fn str_col<'a>(b: &'a RecordBatch, n: &str) -> SxResult<&'a StringArray> {
    need(b, n, &DataType::Utf8)?
        .as_string_opt::<i32>()
        .ok_or_else(|| wrong(n))
}

/// References to `items` sorted by id; duplicate ids are `InvalidInput`.
fn sorted_unique<'a, T>(
    items: &'a [T],
    id: impl Fn(&T) -> i32,
    what: &str,
) -> SxResult<Vec<&'a T>> {
    let mut v: Vec<&T> = items.iter().collect();
    v.sort_by_key(|t| id(t));
    if let Some(w) = v.windows(2).find(|w| match w {
        [a, b] => id(a) == id(b),
        _ => false,
    }) {
        if let Some(t) = w.first() {
            return Err(SxError::InvalidInput(format!(
                "duplicate well id {} in {what}",
                id(t)
            )));
        }
    }
    Ok(v)
}

/// One well's contiguous run of rows in a per-sample batch.
struct Run {
    id: WellId,
    grid: GridRange,
    rows: Range<usize>,
}

/// Split a per-sample batch into per-well runs. Ids must ascend and each
/// well's `k` must rise by 1 per row (`Format` otherwise).
fn runs(well: &Int32Array, k: &Int32Array) -> SxResult<Vec<Run>> {
    let mut out: Vec<Run> = Vec::new();
    for (r, (&w, &kk)) in well.values().iter().zip(k.values()).enumerate() {
        match out.last_mut() {
            Some(run) if run.id.0 == w => {
                let want = i64::from(run.grid.k_start) + run.rows.len() as i64;
                if i64::from(kk) != want {
                    return Err(SxError::Format(format!(
                        "row {r}: k {kk} breaks the grid run of well {w} (expected {want})"
                    )));
                }
                run.rows.end = r + 1;
            }
            Some(run) if w < run.id.0 => {
                return Err(SxError::Format(format!(
                    "row {r}: well_id {w} after {} (ids must ascend)",
                    run.id.0
                )));
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
        run.grid = GridRange::new(run.grid.k_start, run.rows.len())
            .map_err(|e| SxError::Format(format!("well {}: {e}", run.id.0)))?;
    }
    Ok(out)
}

/// The `well_id`, `k`, `md_ft` columns repeated over each well's grid.
fn grid_columns(
    grids: impl Iterator<Item = (WellId, GridRange)>,
    total: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut ids = Vec::with_capacity(total);
    let mut ks = Vec::with_capacity(total);
    let mut mds = Vec::with_capacity(total);
    for (id, grid) in grids {
        for i in 0..grid.len {
            ids.push(id.0);
            ks.push(grid.k(i));
            mds.push(grid.md_ft(i));
        }
    }
    (ids, ks, mds)
}

// ------------------------------------------------------------------ wells

/// Build the `wells` batch (`WELLS_TAG`, not per-sample), rows sorted by id.
///
/// Missing family keys are written as `unknown`. `md_top_ft` / `md_base_ft`
/// are NaN when the well has no logs. Units: ft, m, deg as in the column
/// names. NaN rule: stored as is.
///
/// # Errors
/// `SxError::InvalidInput` for a duplicate id or a grid with `len == 0`.
pub fn wells_batch(rows: &[WellRow]) -> SxResult<RecordBatch> {
    let sorted = sorted_unique(rows, |r| r.header.id.0, "wells")?;
    for r in &sorted {
        if r.grid.is_some_and(|g| g.len == 0) {
            return Err(SxError::InvalidInput(format!(
                "well {}: grid has zero length",
                r.header.id.0
            )));
        }
    }
    let n_samples: Vec<i64> = sorted
        .iter()
        .map(|r| r.grid.map_or(0, |g| g.len as i64))
        .collect();
    let family = |log: LogKind| -> ArrayRef {
        str_array(
            sorted
                .iter()
                .map(|r| r.families.get(&log).map_or(UNKNOWN_FAMILY, String::as_str)),
        )
    };
    let md_top: Vec<f64> = sorted
        .iter()
        .map(|r| {
            r.grid
                .map_or(f64::NAN, |g| GRID_STEP_FT * f64::from(g.k_start))
        })
        .collect();
    let md_base: Vec<f64> = sorted
        .iter()
        .map(|r| {
            r.grid.map_or(f64::NAN, |g| {
                GRID_STEP_FT * (f64::from(g.k_start) + g.len as f64 - 1.0)
            })
        })
        .collect();

    let fields = vec![
        field("well_id", DataType::Int32),
        field("well_name", DataType::Utf8),
        field("x_m", DataType::Float64),
        field("y_m", DataType::Float64),
        field("kb_elev_ft", DataType::Float64),
        field("has_logs", DataType::Boolean),
        field("k_start", DataType::Int32),
        field("n_samples", DataType::Int64),
        field("md_top_ft", DataType::Float64),
        field("md_base_ft", DataType::Float64),
        field("has_survey", DataType::Boolean),
        field("max_inc_deg", DataType::Float64),
        field("n_las_files", DataType::Int64),
        field("gr_family", DataType::Utf8),
        field("nphi_family", DataType::Utf8),
        field("dtc_family", DataType::Utf8),
        field("resd_family", DataType::Utf8),
        field("resd_family_code", DataType::UInt8),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            sorted.iter().map(|r| r.header.id.0),
        )),
        str_array(sorted.iter().map(|r| r.header.name.as_str())),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.header.x_m),
        )),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.header.y_m),
        )),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.header.kb_elev_ft),
        )),
        Arc::new(BooleanArray::from(
            sorted.iter().map(|r| r.grid.is_some()).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from_iter_values(
            sorted.iter().map(|r| r.grid.map_or(0, |g| g.k_start)),
        )),
        Arc::new(Int64Array::from(n_samples)),
        Arc::new(Float64Array::from(md_top)),
        Arc::new(Float64Array::from(md_base)),
        Arc::new(BooleanArray::from(
            sorted.iter().map(|r| r.has_survey).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.max_inc_deg),
        )),
        Arc::new(Int64Array::from_iter_values(
            sorted.iter().map(|r| r.n_las_files),
        )),
        family(LogKind::Gr),
        family(LogKind::Nphi),
        family(LogKind::Dtc),
        family(LogKind::Resd),
        Arc::new(UInt8Array::from_iter_values(
            sorted.iter().map(|r| r.resd_family_code()),
        )),
    ];
    finish(fields, columns, WELLS_TAG)
}

/// Read `wells` rows back in file order.
///
/// `grid` is `None` when `has_logs` is false; `families` gets all four keys
/// (GR, NPHI, DTC, RESD). Derived columns (`md_top_ft`, `md_base_ft`,
/// `resd_family_code`) are ignored.
///
/// # Errors
/// `SxError::Format` for a missing column, a wrong type, or a bad grid.
pub fn wells_from_batch(batch: &RecordBatch) -> SxResult<Vec<WellRow>> {
    let id = i32_col(batch, "well_id")?;
    let name = str_col(batch, "well_name")?;
    let x = f64_col(batch, "x_m")?;
    let y = f64_col(batch, "y_m")?;
    let kb = f64_col(batch, "kb_elev_ft")?;
    let has_logs = bool_col(batch, "has_logs")?;
    let k_start = i32_col(batch, "k_start")?;
    let n_samples = i64_col(batch, "n_samples")?;
    let has_survey = bool_col(batch, "has_survey")?;
    let inc = f64_col(batch, "max_inc_deg")?;
    let n_las = i64_col(batch, "n_las_files")?;
    let fam: Vec<(LogKind, &StringArray)> = FAMILY_LOGS
        .iter()
        .map(|&(log, col)| Ok((log, str_col(batch, col)?)))
        .collect::<SxResult<_>>()?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for r in 0..batch.num_rows() {
        let grid = if has_logs.value(r) {
            let len = usize::try_from(n_samples.value(r))
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| {
                    SxError::Format(format!(
                        "row {r}: has_logs is true but n_samples is {}",
                        n_samples.value(r)
                    ))
                })?;
            Some(
                GridRange::new(k_start.value(r), len)
                    .map_err(|e| SxError::Format(format!("row {r}: {e}")))?,
            )
        } else {
            None
        };
        out.push(WellRow {
            header: WellHeader {
                id: WellId(id.value(r)),
                name: name.value(r).to_string(),
                x_m: x.value(r),
                y_m: y.value(r),
                kb_elev_ft: kb.value(r),
            },
            grid,
            has_survey: has_survey.value(r),
            max_inc_deg: inc.value(r),
            n_las_files: n_las.value(r),
            families: fam
                .iter()
                .map(|(log, col)| (*log, col.value(r).to_string()))
                .collect(),
        });
    }
    Ok(out)
}

// --------------------------------------------------------------- geometry

/// Build the per-sample `geometry` batch (`GEOMETRY_TAG`), sorted by id then k.
///
/// Columns: `well_id`, `k`, `md_ft` (`0.5 * k`), `tvd_ft`, `elev_ft`, `x_m`,
/// `y_m`. Units: ft and m. NaN rule: stored as is.
///
/// # Errors
/// `SxError::InvalidInput` for a duplicate id; `SxError::Shape` when a vector
/// length differs from `grid.len`.
pub fn geometry_batch(wells: &[WellGeometry]) -> SxResult<RecordBatch> {
    let sorted = sorted_unique(wells, |w| w.id.0, "geometry")?;
    let mut total = 0usize;
    for w in &sorted {
        w.validate()?;
        total += w.grid.len;
    }
    let (ids, ks, mds) = grid_columns(sorted.iter().map(|w| (w.id, w.grid)), total);
    let concat = |get: fn(&WellGeometry) -> &Vec<f64>| -> Vec<f64> {
        let mut v = Vec::with_capacity(total);
        for w in &sorted {
            v.extend_from_slice(get(w));
        }
        v
    };
    let fields = vec![
        field("well_id", DataType::Int32),
        field("k", DataType::Int32),
        field("md_ft", DataType::Float64),
        field("tvd_ft", DataType::Float64),
        field("elev_ft", DataType::Float64),
        field("x_m", DataType::Float64),
        field("y_m", DataType::Float64),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(ids)),
        Arc::new(Int32Array::from(ks)),
        Arc::new(Float64Array::from(mds)),
        Arc::new(Float64Array::from(concat(|w| &w.tvd_ft))),
        Arc::new(Float64Array::from(concat(|w| &w.elev_ft))),
        Arc::new(Float64Array::from(concat(|w| &w.x_m))),
        Arc::new(Float64Array::from(concat(|w| &w.y_m))),
    ];
    finish(fields, columns, GEOMETRY_TAG)
}

/// Read a `geometry` batch into one `WellGeometry` per well, ascending id.
///
/// `md_ft` is not read (it is `0.5 * k` by contract).
///
/// # Errors
/// `SxError::Format` for a missing column, a wrong type, descending ids, or a
/// broken `k` run.
pub fn geometry_from_batch(batch: &RecordBatch) -> SxResult<Vec<WellGeometry>> {
    let well = i32_col(batch, "well_id")?;
    let k = i32_col(batch, "k")?;
    let tvd = f64_col(batch, "tvd_ft")?.values();
    let elev = f64_col(batch, "elev_ft")?.values();
    let x = f64_col(batch, "x_m")?.values();
    let y = f64_col(batch, "y_m")?.values();
    let slice = |v: &[f64], rows: &Range<usize>| -> SxResult<Vec<f64>> {
        v.get(rows.clone())
            .map(<[f64]>::to_vec)
            .ok_or_else(|| SxError::Format("column shorter than well_id".into()))
    };
    runs(well, k)?
        .into_iter()
        .map(|run| {
            Ok(WellGeometry {
                id: run.id,
                grid: run.grid,
                tvd_ft: slice(tvd, &run.rows)?,
                elev_ft: slice(elev, &run.rows)?,
                x_m: slice(x, &run.rows)?,
                y_m: slice(y, &run.rows)?,
            })
        })
        .collect()
}

// ------------------------------------------------------------------- logs

/// Build the per-sample logs batch tagged `tag` (`LOGS_RAW_TAG` for ingest).
///
/// Columns: `well_id`, `k`, `md_ft`, then the six `LogKind::column()` value
/// columns in suite order as `f32` (resistivity linear ohm·m). No provenance
/// columns. Rows are sorted by id then k. A log missing from `curves` is
/// written as NaN. NaN rule: stored as is.
///
/// # Errors
/// `SxError::InvalidInput` for a duplicate id or an empty `tag`;
/// `SxError::Shape` when a well fails `WellLogs::validate`.
pub fn logs_batch(wells: &[WellLogs], tag: &str) -> SxResult<RecordBatch> {
    if tag.is_empty() {
        return Err(SxError::InvalidInput("schema tag is empty".into()));
    }
    let sorted = sorted_unique(wells, |w| w.id.0, "logs")?;
    let mut total = 0usize;
    for w in &sorted {
        w.validate()?;
        total += w.grid.len;
    }
    let (ids, ks, mds) = grid_columns(sorted.iter().map(|w| (w.id, w.grid)), total);
    let mut fields = vec![
        field("well_id", DataType::Int32),
        field("k", DataType::Int32),
        field("md_ft", DataType::Float64),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(ids)),
        Arc::new(Int32Array::from(ks)),
        Arc::new(Float64Array::from(mds)),
    ];
    for log in LogKind::ALL {
        let mut values: Vec<f32> = Vec::with_capacity(total);
        for w in &sorted {
            match w.curves.get(&log) {
                Some(c) => values.extend_from_slice(c),
                None => values.resize(values.len() + w.grid.len, f32::NAN),
            }
        }
        fields.push(field(log.column(), DataType::Float32));
        columns.push(Arc::new(Float32Array::from(values)));
    }
    finish(fields, columns, tag)
}

/// Read any batch with the logs-layout columns (extra columns are ignored).
///
/// Returns all six logs for every well, ascending id. Provenance is `0`
/// (Measured) where the value is finite and `CODE_NONE` (255) where it is NaN
/// (or infinite).
///
/// # Errors
/// `SxError::Format` for a missing column, a wrong type, descending ids, or a
/// broken `k` run.
pub fn logs_from_batch(batch: &RecordBatch) -> SxResult<Vec<WellLogs>> {
    let well = i32_col(batch, "well_id")?;
    let k = i32_col(batch, "k")?;
    f64_col(batch, "md_ft")?;
    let values: Vec<(LogKind, &[f32])> = LogKind::ALL
        .iter()
        .map(|&log| Ok((log, f32_col(batch, log.column())?.values().as_ref())))
        .collect::<SxResult<_>>()?;
    let mut out = Vec::new();
    for run in runs(well, k)? {
        let mut logs = WellLogs::new(run.id, run.grid);
        for (log, all) in &values {
            let part = all
                .get(run.rows.clone())
                .ok_or_else(|| SxError::Format("column shorter than well_id".into()))?;
            logs.provenance.insert(
                *log,
                part.iter()
                    .map(|v| if v.is_finite() { 0 } else { CODE_NONE })
                    .collect(),
            );
            logs.curves.insert(*log, part.to_vec());
        }
        out.push(logs);
    }
    Ok(out)
}

// --------------------------------------------------------------- tops_input

/// Build the `tops_input` batch (`TOPS_INPUT_TAG`, not per-sample).
///
/// Rows are sorted by `(well_id, top_idx)`. Units: ft. NaN rule: stored as is
/// (`elev_ft` is NaN when KB is unknown).
///
/// # Errors
/// `SxError::InvalidInput` for a duplicate `(well, top_idx)`, a `top_idx` out
/// of range, a `top_name` that differs from `tops.names[top_idx]`, or a top of
/// `tops` without any row.
pub fn tops_input_batch(tops: &TopSet, rows: &[TopInputRow]) -> SxResult<RecordBatch> {
    let mut sorted: Vec<&TopInputRow> = rows.iter().collect();
    sorted.sort_by_key(|r| (r.well.0, r.top_idx));
    let mut seen = vec![false; tops.len()];
    for (i, r) in sorted.iter().enumerate() {
        let idx = usize::from(r.top_idx);
        let name = tops.names.get(idx).ok_or_else(|| {
            SxError::InvalidInput(format!(
                "well {}: top_idx {} out of range (0..{})",
                r.well.0,
                r.top_idx,
                tops.len()
            ))
        })?;
        if *name != r.top_name {
            return Err(SxError::InvalidInput(format!(
                "well {}: top_idx {} is {name:?}, row says {:?}",
                r.well.0, r.top_idx, r.top_name
            )));
        }
        if let Some(s) = seen.get_mut(idx) {
            *s = true;
        }
        if let Some(prev) = i.checked_sub(1).and_then(|p| sorted.get(p)) {
            if prev.well == r.well && prev.top_idx == r.top_idx {
                return Err(SxError::InvalidInput(format!(
                    "duplicate pick for well {} top {}",
                    r.well.0, r.top_idx
                )));
            }
        }
    }
    if let Some(i) = seen.iter().position(|s| !s) {
        return Err(SxError::InvalidInput(format!(
            "top {:?} has no pick",
            tops.names.get(i).map_or("", String::as_str)
        )));
    }
    let fields = vec![
        field("well_id", DataType::Int32),
        field("top_idx", DataType::UInt16),
        field("top_name", DataType::Utf8),
        field("md_ft", DataType::Float64),
        field("tvd_ft", DataType::Float64),
        field("elev_ft", DataType::Float64),
        field("source", DataType::UInt8),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            sorted.iter().map(|r| r.well.0),
        )),
        Arc::new(UInt16Array::from_iter_values(
            sorted.iter().map(|r| r.top_idx),
        )),
        str_array(sorted.iter().map(|r| r.top_name.as_str())),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.md_ft),
        )),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.tvd_ft),
        )),
        Arc::new(Float64Array::from_iter_values(
            sorted.iter().map(|r| r.elev_ft),
        )),
        Arc::new(UInt8Array::from_iter_values(
            sorted.iter().map(|r| r.source as u8),
        )),
    ];
    finish(fields, columns, TOPS_INPUT_TAG)
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

/// Read a `tops_input` batch: the rebuilt `TopSet` and the rows in file order.
///
/// The top list comes from the distinct `(top_idx, top_name)` pairs sorted by
/// `top_idx`; the indices must be exactly `0..n`, each with one name.
///
/// # Errors
/// `SxError::Format` for a missing column, a wrong type, inconsistent or gapped
/// top indices, an invalid top list, or an unknown `source` code.
pub fn tops_input_from_batch(batch: &RecordBatch) -> SxResult<(TopSet, Vec<TopInputRow>)> {
    let well = i32_col(batch, "well_id")?;
    let idx = u16_col(batch, "top_idx")?;
    let name = str_col(batch, "top_name")?;
    let md = f64_col(batch, "md_ft")?;
    let tvd = f64_col(batch, "tvd_ft")?;
    let elev = f64_col(batch, "elev_ft")?;
    let source = u8_col(batch, "source")?;

    let mut names: BTreeMap<u16, String> = BTreeMap::new();
    let mut rows = Vec::with_capacity(batch.num_rows());
    for r in 0..batch.num_rows() {
        let top_idx = idx.value(r);
        let top_name = name.value(r);
        match names.get(&top_idx) {
            Some(prev) if prev != top_name => {
                return Err(SxError::Format(format!(
                    "top_idx {top_idx} has two names: {prev:?} and {top_name:?}"
                )))
            }
            Some(_) => {}
            None => {
                names.insert(top_idx, top_name.to_string());
            }
        }
        let src = top_source(source.value(r)).ok_or_else(|| {
            SxError::Format(format!("row {r}: unknown source code {}", source.value(r)))
        })?;
        rows.push(TopInputRow {
            well: WellId(well.value(r)),
            top_idx,
            top_name: top_name.to_string(),
            md_ft: md.value(r),
            tvd_ft: tvd.value(r),
            elev_ft: elev.value(r),
            source: src,
        });
    }
    for (i, key) in names.keys().enumerate() {
        if usize::from(*key) != i {
            return Err(SxError::Format(format!(
                "top indices are not 0..{}: found {key} at position {i}",
                names.len()
            )));
        }
    }
    let tops = TopSet::new(names.into_values().collect())
        .map_err(|e| SxError::Format(format!("top list: {e}")))?;
    Ok((tops, rows))
}

// ---------------------------------------------------------------- aux_raw

/// Schema tag of `aux_raw.parquet` (caliper and density correction for QC).
pub const AUX_RAW_TAG: &str = "subsurfix.aux_raw/3";

/// Build the `aux_raw` batch: `well_id`, `k`, `md_ft`, `cal_in`, `drho_gcc`, `bs_in`, `lspd_ftmin`,
/// `temp_c` (f32, NaN = missing), rows sorted by id then `k`.
///
/// # Errors
/// `SxError::InvalidInput` for a duplicate id; `SxError::Shape` for a curve
/// whose length differs from its grid.
pub fn aux_batch(wells: &[WellAux], tag: &str) -> SxResult<RecordBatch> {
    let sorted = sorted_unique(wells, |w| w.id.0, "aux")?;
    let mut total = 0usize;
    for w in &sorted {
        if w.cal_in.len() != w.grid.len
            || w.drho.len() != w.grid.len
            || w.bs_in.len() != w.grid.len
            || w.lspd_ftmin.len() != w.grid.len
            || w.temp_c.len() != w.grid.len
        {
            return Err(SxError::Shape(format!(
                "well {}: aux length differs from grid",
                w.id.0
            )));
        }
        total += w.grid.len;
    }
    let (ids, ks, mds) = grid_columns(sorted.iter().map(|w| (w.id, w.grid)), total);
    let mut cal = Vec::with_capacity(total);
    let mut drho = Vec::with_capacity(total);
    let mut bs = Vec::with_capacity(total);
    let mut sp = Vec::with_capacity(total);
    let mut tc = Vec::with_capacity(total);
    for w in &sorted {
        cal.extend_from_slice(&w.cal_in);
        drho.extend_from_slice(&w.drho);
        bs.extend_from_slice(&w.bs_in);
        sp.extend_from_slice(&w.lspd_ftmin);
        tc.extend_from_slice(&w.temp_c);
    }
    finish(
        vec![
            field("well_id", DataType::Int32),
            field("k", DataType::Int32),
            field("md_ft", DataType::Float64),
            field("cal_in", DataType::Float32),
            field("drho_gcc", DataType::Float32),
            field("bs_in", DataType::Float32),
            field("lspd_ftmin", DataType::Float32),
            field("temp_c", DataType::Float32),
        ],
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int32Array::from(ks)),
            Arc::new(Float64Array::from(mds)),
            Arc::new(Float32Array::from(cal)),
            Arc::new(Float32Array::from(drho)),
            Arc::new(Float32Array::from(bs)),
            Arc::new(Float32Array::from(sp)),
            Arc::new(Float32Array::from(tc)),
        ],
        tag,
    )
}

/// Read an `aux_raw` batch back. A batch written before `temp_c` existed
/// reads as all-NaN temperature.
///
/// # Errors
/// `SxError::Format` for a missing column or a broken `k` run.
pub fn aux_from_batch(batch: &RecordBatch) -> SxResult<Vec<WellAux>> {
    let well = i32_col(batch, "well_id")?;
    let k = i32_col(batch, "k")?;
    let cal = f32_col(batch, "cal_in")?.values();
    let drho = f32_col(batch, "drho_gcc")?.values();
    let bs = f32_col(batch, "bs_in")?.values();
    let sp = f32_col(batch, "lspd_ftmin")?.values();
    let tc = batch
        .column_by_name("temp_c")
        .map(|_| f32_col(batch, "temp_c").map(|a| a.values().to_vec()))
        .transpose()?;
    let mut out = Vec::new();
    for run in runs(well, k)? {
        let c = cal
            .get(run.rows.clone())
            .ok_or_else(|| SxError::Format("short cal_in".into()))?;
        let d = drho
            .get(run.rows.clone())
            .ok_or_else(|| SxError::Format("short drho_gcc".into()))?;
        let b = bs
            .get(run.rows.clone())
            .ok_or_else(|| SxError::Format("short bs_in".into()))?;
        let s = sp
            .get(run.rows.clone())
            .ok_or_else(|| SxError::Format("short lspd_ftmin".into()))?;
        out.push(WellAux {
            id: run.id,
            grid: run.grid,
            cal_in: c.to_vec(),
            drho: d.to_vec(),
            bs_in: b.to_vec(),
            lspd_ftmin: s.to_vec(),
            temp_c: match &tc {
                Some(t) => t
                    .get(run.rows.clone())
                    .ok_or_else(|| SxError::Format("short temp_c".into()))?
                    .to_vec(),
                None => vec![f32::NAN; run.grid.len],
            },
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::parquet::{read_batch_expect, schema_tag, write_batch, SCHEMA_KEY};

    fn header(id: i32, name: &str) -> WellHeader {
        WellHeader {
            id: WellId(id),
            name: name.into(),
            x_m: 1000.5 + f64::from(id),
            y_m: 2000.25,
            kb_elev_ft: 3000.0,
        }
    }

    fn families(resd: &str) -> BTreeMap<LogKind, String> {
        BTreeMap::from([
            (LogKind::Gr, "gr_a".to_string()),
            (LogKind::Nphi, "nphi_a".to_string()),
            (LogKind::Dtc, "dtc_a".to_string()),
            (LogKind::Resd, resd.to_string()),
        ])
    }

    fn assert_rows_equal(a: &[WellRow], b: &[WellRow]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            // NaN != NaN under derived PartialEq: compare bits, then the rest.
            assert_eq!(x.max_inc_deg.to_bits(), y.max_inc_deg.to_bits());
            let (mut x, mut y) = (x.clone(), y.clone());
            x.max_inc_deg = 0.0;
            y.max_inc_deg = 0.0;
            assert_eq!(x, y);
        }
    }

    #[test]
    fn example_c_logs_round_trip() {
        let mut logs = WellLogs::new(WellId(7), GridRange::new(2000, 3).unwrap());
        logs.curves.insert(LogKind::Gr, vec![45.0, f32::NAN, 60.0]);
        logs.provenance.insert(LogKind::Gr, vec![0, CODE_NONE, 0]);
        let batch = logs_batch(&[logs], LOGS_RAW_TAG).unwrap();
        assert_eq!(schema_tag(&batch), Some(LOGS_RAW_TAG));
        let names: Vec<&str> = batch
            .schema_ref()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(
            names,
            [
                "well_id",
                "k",
                "md_ft",
                "gr_api",
                "rhob_gcc",
                "nphi_frac",
                "dtc_usft",
                "resd_ohmm",
                "pef_bfrac"
            ]
        );

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("logs_raw.parquet");
        write_batch(&p, &batch).unwrap();
        let back = read_batch_expect(&p, LOGS_RAW_TAG).unwrap();
        assert!(read_batch_expect(&p, "subsurfix.logs_qc/1").is_err());
        let wells = logs_from_batch(&back).unwrap();
        assert_eq!(wells.len(), 1);
        let w = &wells[0];
        assert_eq!(w.id, WellId(7));
        assert_eq!(w.grid, GridRange::new(2000, 3).unwrap());
        let k: Vec<i32> = back.column(1).as_primitive::<Int32Type>().values().to_vec();
        let md: Vec<f64> = back
            .column(2)
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        assert_eq!(k, [2000, 2001, 2002]);
        assert_eq!(md, [1000.0, 1000.5, 1001.0]);
        let gr = &w.curves[&LogKind::Gr];
        assert_eq!(gr[0], 45.0);
        assert!(gr[1].is_nan());
        assert_eq!(gr[2], 60.0);
        assert_eq!(w.provenance[&LogKind::Gr], vec![0, 255, 0]);
        for log in LogKind::ALL.into_iter().filter(|l| *l != LogKind::Gr) {
            assert!(w.curves[&log].iter().all(|v| v.is_nan()));
            assert_eq!(w.provenance[&log], vec![255, 255, 255]);
        }
        w.validate().unwrap();

        // Written twice: identical bytes. File metadata is only the tag.
        let p2 = dir.path().join("again.parquet");
        write_batch(&p2, &batch).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), std::fs::read(&p2).unwrap());
        assert_eq!(back.schema_ref().metadata().len(), 1);
        assert_eq!(
            back.schema_ref()
                .metadata()
                .get(SCHEMA_KEY)
                .map(String::as_str),
            Some(LOGS_RAW_TAG)
        );
    }

    #[test]
    fn logs_many_wells_sorted_and_errors() {
        let mk = |id: i32, k0: i32, n: usize, base: f32| {
            let mut l = WellLogs::new(WellId(id), GridRange::new(k0, n).unwrap());
            l.curves
                .insert(LogKind::Resd, (0..n).map(|i| base + i as f32).collect());
            l.provenance.insert(LogKind::Resd, vec![0; n]);
            l
        };
        let wells = [mk(9, 10, 2, 1.0), mk(3, 0, 3, 5.0)];
        let batch = logs_batch(&wells, "t/1").unwrap();
        let back = logs_from_batch(&batch).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].id, WellId(3));
        assert_eq!(back[0].grid.len, 3);
        assert_eq!(back[1].curves[&LogKind::Resd], vec![1.0, 2.0]);

        assert!(matches!(
            logs_batch(&[mk(1, 0, 2, 0.0), mk(1, 5, 2, 0.0)], "t/1"),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            logs_batch(&wells, ""),
            Err(SxError::InvalidInput(_))
        ));
        let mut bad = mk(1, 0, 2, 0.0);
        bad.provenance.remove(&LogKind::Resd);
        assert!(matches!(logs_batch(&[bad], "t/1"), Err(SxError::Shape(_))));
        // Empty input gives an empty batch that reads back as no wells.
        let empty = logs_batch(&[], "t/1").unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert!(logs_from_batch(&empty).unwrap().is_empty());
        // Missing column.
        let projected = batch.project(&[0, 1, 2]).unwrap();
        match logs_from_batch(&projected) {
            Err(SxError::Format(m)) => assert!(m.contains("gr_api")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn example_g_wells_round_trip() {
        let rows = vec![
            WellRow {
                header: header(5, "no-logs"),
                grid: None,
                has_survey: false,
                max_inc_deg: f64::NAN,
                n_las_files: 0,
                families: families("unknown"),
            },
            WellRow {
                header: header(2, "W-2"),
                grid: Some(GridRange::new(2000, 100).unwrap()),
                has_survey: true,
                max_inc_deg: 87.5,
                n_las_files: 3,
                families: families("laterolog"),
            },
        ];
        let batch = wells_batch(&rows).unwrap();
        assert_eq!(schema_tag(&batch), Some(WELLS_TAG));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wells.parquet");
        write_batch(&p, &batch).unwrap();
        let back = wells_from_batch(&read_batch_expect(&p, WELLS_TAG).unwrap()).unwrap();
        let sorted = vec![rows[1].clone(), rows[0].clone()];
        assert_rows_equal(&back, &sorted);

        // Derived columns.
        let b = read_batch_expect(&p, WELLS_TAG).unwrap();
        let top = b
            .column_by_name("md_top_ft")
            .unwrap()
            .as_primitive::<Float64Type>();
        let base = b
            .column_by_name("md_base_ft")
            .unwrap()
            .as_primitive::<Float64Type>();
        assert_eq!(top.value(0), 1000.0);
        assert_eq!(base.value(0), 1049.5);
        assert!(top.value(1).is_nan() && base.value(1).is_nan());
        let code = b
            .column_by_name("resd_family_code")
            .unwrap()
            .as_primitive::<UInt8Type>();
        assert_eq!(code.values().to_vec(), vec![2, 0]);
        let n = b
            .column_by_name("n_samples")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(n.values().to_vec(), vec![100, 0]);
    }

    #[test]
    fn wells_rules() {
        let mut row = WellRow {
            header: header(1, "a"),
            grid: None,
            has_survey: false,
            max_inc_deg: f64::NAN,
            n_las_files: 1,
            families: BTreeMap::from([(LogKind::Rhob, "ignored".into())]),
        };
        // Missing keys write "unknown"; other logs are ignored.
        let back = wells_from_batch(&wells_batch(&[row.clone()]).unwrap()).unwrap();
        assert_eq!(back[0].families.len(), 4);
        assert!(back[0].families.values().all(|v| v == "unknown"));
        assert_eq!(back[0].resd_family_code(), 0);
        for (name, code) in [("induction", 1), ("laterolog", 2), ("lateral", 3), ("x", 0)] {
            row.families.insert(LogKind::Resd, name.into());
            assert_eq!(row.resd_family_code(), code);
        }
        // Duplicate id; zero-length grid.
        assert!(matches!(
            wells_batch(&[row.clone(), row.clone()]),
            Err(SxError::InvalidInput(_))
        ));
        row.grid = Some(GridRange { k_start: 0, len: 0 });
        assert!(matches!(wells_batch(&[row]), Err(SxError::InvalidInput(_))));
    }

    fn geometry(id: i32, k0: i32, n: usize, seed: f64) -> WellGeometry {
        let v = |off: f64| {
            (0..n)
                .map(|i| seed + off + i as f64 * 0.1 + 1e-13)
                .collect()
        };
        WellGeometry {
            id: WellId(id),
            grid: GridRange::new(k0, n).unwrap(),
            tvd_ft: v(0.0),
            elev_ft: v(1.0),
            x_m: v(2.0),
            y_m: v(3.0),
        }
    }

    #[test]
    fn example_g_geometry_round_trip() {
        let wells = vec![geometry(8, 40, 5, 0.3), geometry(2, 0, 4, 100.7)];
        let batch = geometry_batch(&wells).unwrap();
        assert_eq!(schema_tag(&batch), Some(GEOMETRY_TAG));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("geometry.parquet");
        write_batch(&p, &batch).unwrap();
        let back = geometry_from_batch(&read_batch_expect(&p, GEOMETRY_TAG).unwrap()).unwrap();
        assert_eq!(back, vec![wells[1].clone(), wells[0].clone()]);

        // Duplicate id; wrong vector length.
        assert!(matches!(
            geometry_batch(&[wells[0].clone(), wells[0].clone()]),
            Err(SxError::InvalidInput(_))
        ));
        let mut short = wells[0].clone();
        short.x_m.pop();
        assert!(matches!(geometry_batch(&[short]), Err(SxError::Shape(_))));
    }

    #[test]
    fn broken_runs_are_format_errors() {
        // A hand-built logs batch with a gap in k.
        let b = geometry_batch(&[geometry(1, 0, 3, 0.0)]).unwrap();
        let ids = Arc::new(Int32Array::from(vec![1, 1, 1]));
        let ks = Arc::new(Int32Array::from(vec![0, 1, 3]));
        let mut cols = b.columns().to_vec();
        cols[0] = ids;
        cols[1] = ks;
        let broken = RecordBatch::try_new(b.schema(), cols).unwrap();
        assert!(matches!(
            geometry_from_batch(&broken),
            Err(SxError::Format(_))
        ));
    }

    fn tops() -> TopSet {
        TopSet::new(vec!["Alpha".into(), "Beta".into()]).unwrap()
    }

    fn pick(well: i32, idx: u16, name: &str, md: f64, src: TopSource) -> TopInputRow {
        TopInputRow {
            well: WellId(well),
            top_idx: idx,
            top_name: name.into(),
            md_ft: md,
            tvd_ft: md - 5.0,
            elev_ft: if well == 2 { f64::NAN } else { 3000.0 - md },
            source: src,
        }
    }

    #[test]
    fn example_g_tops_round_trip() {
        let rows = vec![
            pick(2, 1, "Beta", 1500.0, TopSource::User),
            pick(1, 1, "Beta", 1400.0, TopSource::User),
            pick(1, 0, "Alpha", 1000.0, TopSource::Estimated),
        ];
        let batch = tops_input_batch(&tops(), &rows).unwrap();
        assert_eq!(schema_tag(&batch), Some(TOPS_INPUT_TAG));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tops_input.parquet");
        write_batch(&p, &batch).unwrap();
        let (t, back) =
            tops_input_from_batch(&read_batch_expect(&p, TOPS_INPUT_TAG).unwrap()).unwrap();
        assert_eq!(t, tops());
        assert_eq!(back.len(), 3);
        // File order: (1,0), (1,1), (2,1).
        assert_eq!(back[0].top_name, "Alpha");
        assert_eq!(back[0].source, TopSource::Estimated);
        assert_eq!(back[1].well, WellId(1));
        assert_eq!(back[2].well, WellId(2));
        assert!(back[2].elev_ft.is_nan());
        assert_eq!(back[1].md_ft, 1400.0);
        assert_eq!(back[1].elev_ft, 1600.0);
    }

    #[test]
    fn tops_rules() {
        let ok = pick(1, 0, "Alpha", 1.0, TopSource::User);
        // Top without a pick.
        assert!(matches!(
            tops_input_batch(&tops(), std::slice::from_ref(&ok)),
            Err(SxError::InvalidInput(_))
        ));
        let beta = pick(1, 1, "Beta", 2.0, TopSource::User);
        assert!(tops_input_batch(&tops(), &[ok.clone(), beta.clone()]).is_ok());
        // Duplicate pair, bad index, name mismatch.
        assert!(tops_input_batch(&tops(), &[ok.clone(), beta.clone(), ok.clone()]).is_err());
        let far = pick(1, 2, "Gamma", 3.0, TopSource::User);
        assert!(tops_input_batch(&tops(), &[ok.clone(), beta.clone(), far]).is_err());
        let wrong = pick(2, 1, "Alpha", 3.0, TopSource::User);
        assert!(tops_input_batch(&tops(), &[ok.clone(), beta.clone(), wrong]).is_err());

        // Reader: index gap and unknown source code.
        let gap = TopSet::new(vec!["A".into(), "B".into(), "C".into()]).unwrap();
        let rows: Vec<TopInputRow> = [(0u16, "A"), (1, "B"), (2, "C")]
            .iter()
            .map(|(i, n)| pick(1, *i, n, 1.0, TopSource::User))
            .collect();
        let b = tops_input_batch(&gap, &rows).unwrap();
        let only_two = b.slice(0, 2);
        assert_eq!(tops_input_from_batch(&only_two).unwrap().0.len(), 2);
        let skip_first = b.slice(1, 2);
        assert!(matches!(
            tops_input_from_batch(&skip_first),
            Err(SxError::Format(_))
        ));
        let mut cols = b.columns().to_vec();
        cols[6] = Arc::new(UInt8Array::from(vec![0u8, 9, 0]));
        let bad_src = RecordBatch::try_new(b.schema(), cols).unwrap();
        assert!(matches!(
            tops_input_from_batch(&bad_src),
            Err(SxError::Format(_))
        ));
    }

    /// Cheap deterministic noise in `[0, 1)` (xorshift), so data compresses like real curves.
    fn noise(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Timing budgets of. Run on request, in release:
    /// `cargo test -p sx-core --release io -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn timing_budgets() {
        use std::time::Instant;
        let (n_wells, len) = (600usize, 16_667usize);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let wells: Vec<WellLogs> = (0..n_wells)
            .map(|w| {
                let mut l = WellLogs::new(WellId(w as i32), GridRange::new(2000, len).unwrap());
                for log in LogKind::ALL {
                    let v: Vec<f32> = (0..len)
                        .map(|_| {
                            let x = noise(&mut state);
                            if x < 0.1 {
                                f32::NAN
                            } else {
                                x * 100.0
                            }
                        })
                        .collect();
                    l.curves.insert(log, v);
                    l.provenance.insert(log, vec![0; len]);
                }
                l
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("logs_raw.parquet");

        let t = Instant::now();
        let batch = logs_batch(&wells, LOGS_RAW_TAG).unwrap();
        write_batch(&p, &batch).unwrap();
        let write_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let back = read_batch_expect(&p, LOGS_RAW_TAG).unwrap();
        let logs = logs_from_batch(&back).unwrap();
        let read_s = t.elapsed().as_secs_f64();
        assert_eq!(logs.len(), n_wells);
        assert_eq!(back.num_rows(), n_wells * len);

        // One well, 20,000 samples, six logs, through the LAS writer.
        let grid = GridRange::new(0, 20_000).unwrap();
        let mut las_logs = WellLogs::new(WellId(1), grid);
        for log in LogKind::ALL {
            las_logs.curves.insert(
                log,
                (0..grid.len).map(|_| noise(&mut state) * 10.0).collect(),
            );
            las_logs.provenance.insert(log, vec![0; grid.len]);
        }
        let geo = WellGeometry {
            id: WellId(1),
            grid,
            tvd_ft: (0..grid.len).map(|i| grid.md_ft(i)).collect(),
            elev_ft: (0..grid.len).map(|i| 3000.0 - grid.md_ft(i)).collect(),
            x_m: vec![0.0; grid.len],
            y_m: vec![0.0; grid.len],
        };
        let h = header(1, "W-1");
        let las_path = dir.path().join("w.las");
        let t = Instant::now();
        crate::io::las::write_well_las(&las_path, &h, &geo, &las_logs).unwrap();
        let las_ms = t.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "write (logs_batch + write_batch, {} rows): {write_s:.2} s (budget 4 s)",
            n_wells * len
        );
        eprintln!("read (read_batch + logs_from_batch):        {read_s:.2} s (budget 2 s)");
        eprintln!("write_well_las (20,000 samples, 6 logs):    {las_ms:.1} ms (budget 60 ms)");
        assert!(write_s < 4.0, "write took {write_s:.2} s");
        assert!(read_s < 2.0, "read took {read_s:.2} s");
        assert!(las_ms < 60.0, "LAS took {las_ms:.1} ms");
    }
}
