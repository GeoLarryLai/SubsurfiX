// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Ingest report rows and their batch.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;

/// Column names of `ingest_report.csv`, in order.
pub const INGEST_REPORT_COLUMNS: [&str; 31] = [
    "well_id",
    "well_name",
    "well_status",
    "log",
    "log_status",
    "primary_mnemonic",
    "primary_unit",
    "family",
    "unit_rule",
    "verify",
    "unit_divisor",
    "nphi_offset_frac",
    "n_candidates",
    "merged_from",
    "shifts_ft",
    "skipped",
    "rejected",
    "coverage_before_frac",
    "coverage_after_frac",
    "primary_file",
    "n_files",
    "n_files_used",
    "n_shadowed_files",
    "depth_unit",
    "n_nan_depth",
    "n_dup_depth",
    "n_desc_files",
    "n_grid_finite",
    "md_top_ft",
    "md_base_ft",
    "note",
];

/// Schema tag of the report batch.
pub const INGEST_REPORT_TAG: &str = "subsurfix.ingest_report/1";

/// One report row (one well, one log); fields map 1:1 to
/// [`INGEST_REPORT_COLUMNS`].
///
/// "Does not apply" is `None` for integers, NaN for floats, `""` for text and
/// an empty vector for lists. List columns are joined with `;` in the batch.
/// Depths are in feet; fractions are 0 to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestReportRow {
    /// Well id; `None` when the LAS name has no header.
    pub well_id: Option<i32>,
    /// Well name.
    pub well_name: String,
    /// `WellStatus::as_str`.
    pub well_status: String,
    /// `GR` ... `PEF`.
    pub log: String,
    /// `resolved`, `off_grid`, `rejected` or `absent`.
    pub log_status: String,
    /// Primary mnemonic as in the file.
    pub primary_mnemonic: String,
    /// Raw header unit of the primary.
    pub primary_unit: String,
    /// Family of the primary.
    pub family: String,
    /// `header`, `header_veto`, `blank` or `stats`.
    pub unit_rule: String,
    /// `passed`, `skipped` or `retry_passed`.
    pub verify: String,
    /// Total divisor applied to the primary; NaN when it does not apply.
    pub unit_divisor: f64,
    /// `-0.04` or `0` (NPHI); NaN for other logs.
    pub nphi_offset_frac: f64,
    /// Candidates in the primary file.
    pub n_candidates: Option<i64>,
    /// Secondaries that filled the primary.
    pub merged_from: Vec<String>,
    /// Their shifts, integer feet, same order.
    pub shifts_ft: Vec<i32>,
    /// `MNEM:resid` / `MNEM:cross_family`.
    pub skipped: Vec<String>,
    /// `MNEM:reason`.
    pub rejected: Vec<String>,
    /// Valid primary rows over rows of the primary file.
    pub coverage_before_frac: f64,
    /// Finite spliced rows over rows of the primary file.
    pub coverage_after_frac: f64,
    /// Primary file name (never a path).
    pub primary_file: String,
    /// LAS files of the well.
    pub n_files: Option<i64>,
    /// Files that contributed samples of this log.
    pub n_files_used: Option<i64>,
    /// LAS files of the well found in later folders.
    pub n_shadowed_files: Option<i64>,
    /// `ft`, `m` or `unknown:<raw>`, of the primary file.
    pub depth_unit: String,
    /// Rows with a bad depth, summed over files.
    pub n_nan_depth: Option<i64>,
    /// Rows removed as repeated depths, summed over files.
    pub n_dup_depth: Option<i64>,
    /// Files with a descending index.
    pub n_desc_files: Option<i64>,
    /// Finite grid samples of this log.
    pub n_grid_finite: Option<i64>,
    /// First finite grid MD of this log, feet.
    pub md_top_ft: f64,
    /// Last finite grid MD of this log, feet.
    pub md_base_ft: f64,
    /// Well notes.
    pub note: Vec<String>,
}

impl IngestReportRow {
    /// A row with every "does not apply" value filled in.
    pub(crate) fn blank(
        well_id: Option<i32>,
        well_name: &str,
        well_status: &str,
        log: &str,
    ) -> IngestReportRow {
        IngestReportRow {
            well_id,
            well_name: well_name.to_string(),
            well_status: well_status.to_string(),
            log: log.to_string(),
            log_status: "absent".to_string(),
            primary_mnemonic: String::new(),
            primary_unit: String::new(),
            family: String::new(),
            unit_rule: String::new(),
            verify: String::new(),
            unit_divisor: f64::NAN,
            nphi_offset_frac: f64::NAN,
            n_candidates: None,
            merged_from: Vec::new(),
            shifts_ft: Vec::new(),
            skipped: Vec::new(),
            rejected: Vec::new(),
            coverage_before_frac: f64::NAN,
            coverage_after_frac: f64::NAN,
            primary_file: String::new(),
            n_files: None,
            n_files_used: None,
            n_shadowed_files: None,
            depth_unit: String::new(),
            n_nan_depth: None,
            n_dup_depth: None,
            n_desc_files: None,
            n_grid_finite: None,
            md_top_ft: f64::NAN,
            md_base_ft: f64::NAN,
            note: Vec::new(),
        }
    }
}

fn utf8(rows: &[IngestReportRow], f: impl Fn(&IngestReportRow) -> String) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(rows.iter().map(f)))
}

fn f64s(rows: &[IngestReportRow], f: impl Fn(&IngestReportRow) -> f64) -> ArrayRef {
    Arc::new(Float64Array::from_iter_values(rows.iter().map(f)))
}

fn i64s(rows: &[IngestReportRow], f: impl Fn(&IngestReportRow) -> Option<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(rows.iter().map(f).collect::<Vec<_>>()))
}

/// Report rows as a batch tagged [`INGEST_REPORT_TAG`].
///
/// Column types: `well_id` `Int32`; other integers `Int64`; all nullable, null
/// when the value does not apply. Floats are `Float64` with NaN when they do
/// not apply. Text is `Utf8` with `""` when it does not apply. Lists are
/// `Utf8` joined with `;`. Zero rows give an empty batch.
///
/// # Errors
/// `SxError::InvalidInput` if the batch cannot be built.
pub fn ingest_report_batch(rows: &[IngestReportRow]) -> SxResult<RecordBatch> {
    let join = |v: &[String]| v.join(";");
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.well_id).collect::<Vec<_>>(),
        )),
        utf8(rows, |r| r.well_name.clone()),
        utf8(rows, |r| r.well_status.clone()),
        utf8(rows, |r| r.log.clone()),
        utf8(rows, |r| r.log_status.clone()),
        utf8(rows, |r| r.primary_mnemonic.clone()),
        utf8(rows, |r| r.primary_unit.clone()),
        utf8(rows, |r| r.family.clone()),
        utf8(rows, |r| r.unit_rule.clone()),
        utf8(rows, |r| r.verify.clone()),
        f64s(rows, |r| r.unit_divisor),
        f64s(rows, |r| r.nphi_offset_frac),
        i64s(rows, |r| r.n_candidates),
        utf8(rows, |r| join(&r.merged_from)),
        utf8(rows, |r| {
            r.shifts_ft
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(";")
        }),
        utf8(rows, |r| join(&r.skipped)),
        utf8(rows, |r| join(&r.rejected)),
        f64s(rows, |r| r.coverage_before_frac),
        f64s(rows, |r| r.coverage_after_frac),
        utf8(rows, |r| r.primary_file.clone()),
        i64s(rows, |r| r.n_files),
        i64s(rows, |r| r.n_files_used),
        i64s(rows, |r| r.n_shadowed_files),
        utf8(rows, |r| r.depth_unit.clone()),
        i64s(rows, |r| r.n_nan_depth),
        i64s(rows, |r| r.n_dup_depth),
        i64s(rows, |r| r.n_desc_files),
        i64s(rows, |r| r.n_grid_finite),
        f64s(rows, |r| r.md_top_ft),
        f64s(rows, |r| r.md_base_ft),
        utf8(rows, |r| join(&r.note)),
    ];
    let fields: Vec<Field> = INGEST_REPORT_COLUMNS
        .iter()
        .zip(&columns)
        .map(|(name, col)| {
            let nullable = matches!(col.data_type(), DataType::Int32 | DataType::Int64);
            Field::new(*name, col.data_type().clone(), nullable)
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("cannot build ingest report: {e}")))?;
    with_schema_tag(batch, INGEST_REPORT_TAG)
}

/// Column names of `las_match_report.csv`, in order.
pub const LAS_MATCH_COLUMNS: [&str; 7] = [
    "file",
    "path",
    "header_well",
    "file_name_well",
    "used_well",
    "decision",
    "reason",
];

/// `las_match_report.csv` batch: one row per LAS header/file-name conflict or
/// manual override; `used_well` is empty when the file was left out.
///
/// # Errors
/// `SxError::InvalidInput` if the batch cannot be built.
pub fn las_match_batch(rows: &[super::LasMatchRow]) -> SxResult<RecordBatch> {
    let col = |f: &dyn Fn(&super::LasMatchRow) -> String| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(rows.iter().map(f)))
    };
    let columns: Vec<ArrayRef> = vec![
        col(&|r| r.file.clone()),
        col(&|r| r.path.clone()),
        col(&|r| r.header_well.clone()),
        col(&|r| r.file_name_well.clone()),
        col(&|r| r.used.clone()),
        col(&|r| r.decision.as_str().to_string()),
        col(&|r| r.reason.clone()),
    ];
    let fields: Vec<Field> = LAS_MATCH_COLUMNS
        .iter()
        .map(|n| Field::new(*n, DataType::Utf8, false))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("las_match_report: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::csv::batch_to_csv_string;
    use crate::io::parquet::schema_tag;
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{Float64Type, Int32Type, Int64Type};

    #[test]
    fn columns_and_types() {
        assert_eq!(INGEST_REPORT_COLUMNS.len(), 31);
        let mut row = IngestReportRow::blank(Some(7), "W,1", "ok", "GR");
        row.merged_from = vec!["GRD".into(), "GRN".into()];
        row.shifts_ft = vec![-3, 2];
        row.n_files = Some(2);
        row.unit_divisor = 1.0;
        let none = IngestReportRow::blank(None, "X", "no_header", "PEF");
        let batch = ingest_report_batch(&[row, none]).unwrap();
        assert_eq!(schema_tag(&batch), Some(INGEST_REPORT_TAG));
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 31);
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, INGEST_REPORT_COLUMNS);
        let id = batch.column(0).as_primitive::<Int32Type>();
        assert_eq!(id.value(0), 7);
        assert!(id.is_null(1));
        let n_files = batch.column(20).as_primitive::<Int64Type>();
        assert_eq!(n_files.value(0), 2);
        assert!(n_files.is_null(1));
        assert_eq!(batch.column(10).as_primitive::<Float64Type>().value(0), 1.0);
        assert!(batch
            .column(10)
            .as_primitive::<Float64Type>()
            .value(1)
            .is_nan());
        assert_eq!(batch.column(13).as_string::<i32>().value(0), "GRD;GRN");
        assert_eq!(batch.column(14).as_string::<i32>().value(0), "-3;2");
        assert_eq!(batch.column(14).as_string::<i32>().value(1), "");
        let csv = batch_to_csv_string(&batch).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), INGEST_REPORT_COLUMNS.join(","));
        let first = lines.next().unwrap();
        assert!(first.starts_with("7,\"W,1\",ok,GR,absent,"), "{first}");
        let second = lines.next().unwrap();
        // Null id, NaN floats and unset ints are empty fields.
        assert!(second.starts_with(",X,no_header,PEF,absent,"), "{second}");
        assert_eq!(second.matches(',').count(), 30);
    }

    #[test]
    fn empty_batch() {
        let batch = ingest_report_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 31);
    }
}
