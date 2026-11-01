// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! CSV writer for report batches, and a reader for the files written here.
//!
//! Format: UTF-8, no BOM, `,` separator, `\n` line ends (also after the last
//! row), header row = field names, `.` decimal point. Fields holding `,`, `"`,
//! `\n` or `\r` are quoted with inner quotes doubled. Schema metadata is not
//! written.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;

use super::atomic::io_error;
use crate::error::{SxError, SxResult};

/// CSV text of an `f64`: NaN and ±inf give `""`, else Rust `{}` (shortest
/// round-trip text, for example `0.1`, `-3`).
pub fn csv_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        String::new()
    }
}

/// CSV text of an `f32`: NaN and ±inf give `""`, else Rust `{}`.
pub fn csv_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        String::new()
    }
}

fn push_field(out: &mut String, text: &str) {
    if text.contains([',', '"', '\n', '\r']) {
        out.push('"');
        for c in text.chars() {
            if c == '"' {
                out.push('"');
            }
            out.push(c);
        }
        out.push('"');
    } else {
        out.push_str(text);
    }
}

/// A typed view of one column that can print a cell.
enum Col<'a> {
    I8(&'a arrow::array::Int8Array),
    I16(&'a arrow::array::Int16Array),
    I32(&'a arrow::array::Int32Array),
    I64(&'a arrow::array::Int64Array),
    U8(&'a arrow::array::UInt8Array),
    U16(&'a arrow::array::UInt16Array),
    U32(&'a arrow::array::UInt32Array),
    U64(&'a arrow::array::UInt64Array),
    F32(&'a arrow::array::Float32Array),
    F64(&'a arrow::array::Float64Array),
    Bool(&'a arrow::array::BooleanArray),
    Str(&'a arrow::array::StringArray),
}

impl<'a> Col<'a> {
    fn new(name: &str, array: &'a dyn Array) -> SxResult<Col<'a>> {
        let bad = || {
            SxError::InvalidInput(format!(
                "column {name}: unsupported type {}",
                array.data_type()
            ))
        };
        Ok(match array.data_type() {
            DataType::Int8 => Col::I8(array.as_primitive_opt::<Int8Type>().ok_or_else(bad)?),
            DataType::Int16 => Col::I16(array.as_primitive_opt::<Int16Type>().ok_or_else(bad)?),
            DataType::Int32 => Col::I32(array.as_primitive_opt::<Int32Type>().ok_or_else(bad)?),
            DataType::Int64 => Col::I64(array.as_primitive_opt::<Int64Type>().ok_or_else(bad)?),
            DataType::UInt8 => Col::U8(array.as_primitive_opt::<UInt8Type>().ok_or_else(bad)?),
            DataType::UInt16 => Col::U16(array.as_primitive_opt::<UInt16Type>().ok_or_else(bad)?),
            DataType::UInt32 => Col::U32(array.as_primitive_opt::<UInt32Type>().ok_or_else(bad)?),
            DataType::UInt64 => Col::U64(array.as_primitive_opt::<UInt64Type>().ok_or_else(bad)?),
            DataType::Float32 => Col::F32(array.as_primitive_opt::<Float32Type>().ok_or_else(bad)?),
            DataType::Float64 => Col::F64(array.as_primitive_opt::<Float64Type>().ok_or_else(bad)?),
            DataType::Boolean => Col::Bool(array.as_boolean_opt().ok_or_else(bad)?),
            DataType::Utf8 => Col::Str(array.as_string_opt::<i32>().ok_or_else(bad)?),
            _ => return Err(bad()),
        })
    }

    /// Append the cell of row `r` (already CSV-escaped).
    fn push_cell(&self, out: &mut String, r: usize) {
        macro_rules! int {
            ($a:expr) => {
                if !$a.is_null(r) {
                    let _ = write!(out, "{}", $a.value(r));
                }
            };
        }
        match self {
            Col::I8(a) => int!(a),
            Col::I16(a) => int!(a),
            Col::I32(a) => int!(a),
            Col::I64(a) => int!(a),
            Col::U8(a) => int!(a),
            Col::U16(a) => int!(a),
            Col::U32(a) => int!(a),
            Col::U64(a) => int!(a),
            Col::F32(a) => {
                if !a.is_null(r) {
                    out.push_str(&csv_f32(a.value(r)));
                }
            }
            Col::F64(a) => {
                if !a.is_null(r) {
                    out.push_str(&csv_f64(a.value(r)));
                }
            }
            Col::Bool(a) => {
                if !a.is_null(r) {
                    out.push_str(if a.value(r) { "true" } else { "false" });
                }
            }
            Col::Str(a) => {
                if !a.is_null(r) {
                    push_field(out, a.value(r));
                }
            }
        }
    }
}

/// CSV text of a batch.
///
/// Cells: integers `{}`; floats via [`csv_f32`] / [`csv_f64`] (NaN and ±inf
/// become `""`); booleans `true` / `false`; strings as is; a null cell `""`.
/// Zero rows give the header line only. O(cells).
///
/// # Errors
/// `SxError::InvalidInput` naming a column of an unsupported type.
pub fn batch_to_csv_string(batch: &RecordBatch) -> SxResult<String> {
    let schema = batch.schema_ref();
    let cols = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(f, a)| Col::new(f.name(), a.as_ref()))
        .collect::<SxResult<Vec<_>>>()?;
    let mut out = String::new();
    for (i, f) in schema.fields().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_field(&mut out, f.name());
    }
    out.push('\n');
    for r in 0..batch.num_rows() {
        for (i, c) in cols.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            c.push_cell(&mut out, r);
        }
        out.push('\n');
    }
    Ok(out)
}

/// Write [`batch_to_csv_string`] to exactly `path` (any file name; not
/// atomic). The file is synced before return.
///
/// # Errors
/// `SxError::InvalidInput` (unsupported column type), `SxError::Io`.
pub fn write_batch(path: &Path, batch: &RecordBatch) -> SxResult<()> {
    let text = batch_to_csv_string(batch)?;
    let mut file = std::fs::File::create(path).map_err(|e| io_error(path, e))?;
    file.write_all(text.as_bytes())
        .map_err(|e| io_error(path, e))?;
    file.sync_all().map_err(|e| io_error(path, e))
}

/// Parse CSV text (quoted fields, doubled quotes, `\n` or `\r\n` line ends).
fn parse_csv(text: &str) -> SxResult<Vec<Vec<String>>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    // True once the current record has any content (a field started or ended).
    let mut pending = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                    match chars.peek() {
                        None | Some(',') | Some('\n') | Some('\r') => {}
                        Some(_) => {
                            return Err(SxError::Format(
                                "csv: text after closing quote".to_string(),
                            ))
                        }
                    }
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => {
                in_quotes = true;
                pending = true;
            }
            ',' => {
                record.push(std::mem::take(&mut field));
                pending = true;
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
                pending = false;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            _ => {
                field.push(c);
                pending = true;
            }
        }
    }
    if in_quotes {
        return Err(SxError::Format("csv: unterminated quote".to_string()));
    }
    if pending {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

/// Read a file written by [`write_batch`] (RFC 4180 subset): the header names
/// and the data rows as text. Cells are returned unquoted, `""` stays `""`.
///
/// # Errors
/// `SxError::Io`; `SxError::Format` for an unterminated quote, text after a
/// closing quote, or a file with no header line.
pub fn read_csv(path: &Path) -> SxResult<(Vec<String>, Vec<Vec<String>>)> {
    let bytes = std::fs::read(path).map_err(|e| io_error(path, e))?;
    let text = String::from_utf8(bytes)
        .map_err(|_| SxError::Format(format!("{}: not valid UTF-8", path.display())))?;
    let mut records = parse_csv(&text).map_err(|e| match e {
        SxError::Format(m) => SxError::Format(format!("{}: {m}", path.display())),
        other => other,
    })?;
    if records.is_empty() {
        return Err(SxError::Format(format!(
            "{}: csv has no header line",
            path.display()
        )));
    }
    let header = records.remove(0);
    Ok((header, records))
}
