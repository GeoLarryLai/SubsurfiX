// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Column glossary: one row per column of each stage product.

use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use super::parquet::with_schema_tag;
use crate::error::{SxError, SxResult};
use crate::types::LogKind;

/// Tag of the glossary batch.
pub const GLOSSARY_TAG: &str = "subsurfix.column_glossary/1";

/// Unit text from the column-name suffix (first match in this order).
const UNIT_SUFFIXES: [(&str, &str); 10] = [
    ("_log10ohmm", "log10(ohm.m)"),
    ("_ohmm", "ohm.m"),
    ("_bfrac", "b/e"),
    ("_frac", "v/v"),
    ("_api", "gAPI"),
    ("_gcc", "g/cm3"),
    ("_usft", "us/ft"),
    ("_deg", "deg"),
    ("_ft", "ft"),
    ("_m", "m"),
];

/// Built-in descriptions by exact column name. Where two products use the same
/// column name with different meanings (`x_m`, `y_m`, `md_ft`, `tvd_ft`,
/// `elev_ft`), the entry here is the `wells` (or `geometry` for depth) meaning and
/// [`PRODUCT_DESCRIPTIONS`] overrides it for the other product.
const DESCRIPTIONS: &[(&str, &str)] = &[
    // wells
    ("well_id", "Stable well id"),
    ("well_name", "External well name"),
    ("x_m", "Surface x, project frame"),
    ("y_m", "Surface y, project frame"),
    ("kb_elev_ft", "Kelly bushing elevation above sea level"),
    ("has_logs", "Well has a grid and curves"),
    ("k_start", "First grid index (0 when no logs)"),
    ("n_samples", "Grid length (0 when no logs)"),
    ("md_top_ft", "0.5·k_start; NaN when no logs"),
    (
        "md_base_ft",
        "0.5·(k_start + n_samples − 1); NaN when no logs",
    ),
    ("has_survey", "A deviation survey was used"),
    (
        "max_inc_deg",
        "Largest survey inclination; NaN without survey",
    ),
    ("n_las_files", "LAS files used for the well"),
    ("gr_family", "GR curve family (unknown if absent)"),
    ("nphi_family", "NPHI curve family"),
    ("dtc_family", "DTC curve family"),
    ("resd_family", "RESD curve family"),
    (
        "resd_family_code",
        "0 unknown, 1 induction, 2 laterolog, 3 lateral",
    ),
    // geometry
    ("k", "Global grid index; md_ft = 0.5·k"),
    ("md_ft", "Measured depth below KB"),
    ("tvd_ft", "True vertical depth below KB, positive down"),
    ("elev_ft", "Elevation above sea level, positive up"),
    // tops_input
    (
        "top_idx",
        "Index into the project top list (shallow to deep, 0-based)",
    ),
    ("top_name", "Top name"),
    ("source", "TopSource code (0 = user)"),
    // value columns
    ("gr_api", "Gamma ray"),
    ("rhob_gcc", "Bulk density"),
    ("nphi_frac", "Neutron porosity"),
    ("dtc_usft", "Compressional slowness"),
    ("resd_ohmm", "Deep resistivity"),
    ("resd_log10ohmm", "Deep resistivity, log10"),
    ("pef_bfrac", "Photoelectric factor"),
];

fn dtype_text(dt: &DataType) -> Option<&'static str> {
    Some(match dt {
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::Boolean => "bool",
        DataType::Utf8 => "string",
        _ => return None,
    })
}

fn unit_text(column: &str) -> &'static str {
    UNIT_SUFFIXES
        .iter()
        .find(|(suffix, _)| column.ends_with(suffix))
        .map_or("", |(_, unit)| unit)
}

/// Descriptions that differ by product: `(product, column, text)`. These win over
/// [`DESCRIPTIONS`] when the product name matches.
const PRODUCT_DESCRIPTIONS: &[(&str, &str, &str)] = &[
    ("geometry", "x_m", "Sample x, project frame"),
    ("geometry", "y_m", "Sample y, project frame"),
    ("tops_input", "md_ft", "Pick measured depth"),
    ("tops_input", "tvd_ft", "True vertical depth at the pick"),
    (
        "tops_input",
        "elev_ft",
        "Elevation at the pick; NaN when KB unknown",
    ),
];

fn description_text(product: &str, column: &str) -> String {
    if let Some((_, _, text)) = PRODUCT_DESCRIPTIONS
        .iter()
        .find(|(p, name, _)| *p == product && *name == column)
    {
        return (*text).to_string();
    }
    if let Some((_, text)) = DESCRIPTIONS.iter().find(|(name, _)| *name == column) {
        return (*text).to_string();
    }
    for log in LogKind::ALL {
        let lower = log.name().to_ascii_lowercase();
        if column.strip_suffix("_prov") == Some(lower.as_str()) {
            return format!("{} provenance code", log.name());
        }
    }
    String::new()
}

/// One row per column of each `(product, schema)`, in the given order.
///
/// Columns of the result (all strings): `product`, `column`, `dtype`
/// (`int32`, `int64`, `uint8`, `uint16`, `float32`, `float64`, `bool`,
/// `string`), `unit` (from the name suffix, `""` if none), `description` (from
/// a built-in table by exact column name, `""` if unknown). Tagged
/// `GLOSSARY_TAG`. The pipeline writes it as `column_glossary.csv`.
///
/// # Errors
/// `SxError::InvalidInput` for a column of any other type.
pub fn column_glossary(products: &[(&str, &Schema)]) -> SxResult<RecordBatch> {
    let mut product = Vec::new();
    let mut column = Vec::new();
    let mut dtype = Vec::new();
    let mut unit = Vec::new();
    let mut description = Vec::new();
    for (name, schema) in products {
        for f in schema.fields() {
            let dt = dtype_text(f.data_type()).ok_or_else(|| {
                SxError::InvalidInput(format!(
                    "{name}.{}: unsupported type {}",
                    f.name(),
                    f.data_type()
                ))
            })?;
            product.push((*name).to_string());
            column.push(f.name().clone());
            dtype.push(dt.to_string());
            unit.push(unit_text(f.name()).to_string());
            description.push(description_text(name, f.name()));
        }
    }
    let fields = ["product", "column", "dtype", "unit", "description"]
        .iter()
        .map(|n| Field::new(*n, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = [product, column, dtype, unit, description]
        .into_iter()
        .map(|v| Arc::new(StringArray::from(v)) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("cannot build glossary: {e}")))?;
    with_schema_tag(batch, GLOSSARY_TAG)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::csv::batch_to_csv_string;
    use crate::io::parquet::schema_tag;
    use crate::io::products::{
        geometry_batch, tops_input_batch, wells_batch, TopInputRow, WellRow,
    };
    use crate::types::{GridRange, TopSet, TopSource, WellGeometry, WellHeader, WellId};
    use arrow::array::AsArray;
    use std::collections::BTreeMap;

    fn geometry_schema() -> arrow::datatypes::SchemaRef {
        let g = WellGeometry {
            id: WellId(1),
            grid: GridRange::new(0, 1).unwrap(),
            tvd_ft: vec![1.0],
            elev_ft: vec![2.0],
            x_m: vec![3.0],
            y_m: vec![4.0],
        };
        geometry_batch(&[g]).unwrap().schema()
    }

    #[test]
    fn example_f_geometry_glossary() {
        let schema = geometry_schema();
        let g = column_glossary(&[("geometry", schema.as_ref())]).unwrap();
        assert_eq!(schema_tag(&g), Some(GLOSSARY_TAG));
        assert_eq!(g.num_rows(), 7);
        let text = batch_to_csv_string(&g).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 8);
        assert_eq!(lines[0], "product,column,dtype,unit,description");
        assert_eq!(
            lines[4],
            "geometry,tvd_ft,float64,ft,\"True vertical depth below KB, positive down\""
        );
        assert_eq!(lines[1], "geometry,well_id,int32,,Stable well id");
    }

    #[test]
    fn same_name_columns_get_their_own_product_text() {
        let schema = Schema::new(vec![
            Field::new("x_m", DataType::Float64, false),
            Field::new("md_ft", DataType::Float64, false),
        ]);
        let g = column_glossary(&[
            ("wells", &schema),
            ("geometry", &schema),
            ("tops_input", &schema),
        ])
        .unwrap();
        let desc = g.column(4).as_string::<i32>();
        assert_eq!(desc.value(0), "Surface x, project frame");
        assert_eq!(desc.value(2), "Sample x, project frame");
        assert_eq!(desc.value(1), "Measured depth below KB");
        assert_eq!(desc.value(5), "Pick measured depth");
    }

    #[test]
    fn units_dtypes_and_unknown_columns() {
        let schema = Schema::new(vec![
            Field::new("resd_log10ohmm", DataType::Float32, false),
            Field::new("resd_ohmm", DataType::Float32, false),
            Field::new("pef_bfrac", DataType::Float32, false),
            Field::new("nphi_frac", DataType::Float32, false),
            Field::new("gr_api", DataType::Float32, false),
            Field::new("rhob_gcc", DataType::Float32, false),
            Field::new("dtc_usft", DataType::Float32, false),
            Field::new("max_inc_deg", DataType::Float64, false),
            Field::new("x_m", DataType::Float64, false),
            Field::new("gr_prov", DataType::UInt8, false),
            Field::new("resd_prov", DataType::UInt8, false),
            Field::new("foo", DataType::Utf8, false),
            Field::new("flag", DataType::Boolean, false),
            Field::new("n", DataType::Int64, false),
            Field::new("c", DataType::UInt16, false),
        ]);
        let g = column_glossary(&[("p", &schema)]).unwrap();
        let col = |i: usize| g.column(i).as_string::<i32>();
        let unit = col(3);
        assert_eq!(unit.value(0), "log10(ohm.m)");
        assert_eq!(unit.value(1), "ohm.m");
        assert_eq!(unit.value(2), "b/e");
        assert_eq!(unit.value(3), "v/v");
        assert_eq!(unit.value(4), "gAPI");
        assert_eq!(unit.value(5), "g/cm3");
        assert_eq!(unit.value(6), "us/ft");
        assert_eq!(unit.value(7), "deg");
        assert_eq!(unit.value(8), "m");
        assert_eq!(unit.value(11), "");
        let desc = col(4);
        assert_eq!(desc.value(0), "Deep resistivity, log10");
        assert_eq!(
            desc.value(7),
            "Largest survey inclination; NaN without survey"
        );
        assert_eq!(desc.value(9), "GR provenance code");
        assert_eq!(desc.value(10), "RESD provenance code");
        assert_eq!(desc.value(11), "");
        let dtype: Vec<&str> = (0..g.num_rows()).map(|r| col(2).value(r)).collect();
        assert_eq!(&dtype[11..], ["string", "bool", "int64", "uint16"]);

        let bad = Schema::new(vec![Field::new("d", DataType::Date32, false)]);
        assert!(matches!(
            column_glossary(&[("p", &bad)]),
            Err(SxError::InvalidInput(_))
        ));
        assert_eq!(column_glossary(&[]).unwrap().num_rows(), 0);
    }

    #[test]
    fn every_product_column_has_a_description() {
        let wells = wells_batch(&[WellRow {
            header: WellHeader {
                id: WellId(1),
                name: "w".into(),
                x_m: 0.0,
                y_m: 0.0,
                kb_elev_ft: 0.0,
            },
            grid: None,
            has_survey: false,
            max_inc_deg: f64::NAN,
            n_las_files: 0,
            families: BTreeMap::new(),
        }])
        .unwrap();
        let tops = TopSet::new(vec!["A".into()]).unwrap();
        let tops_in = tops_input_batch(
            &tops,
            &[TopInputRow {
                well: WellId(1),
                top_idx: 0,
                top_name: "A".into(),
                md_ft: 1.0,
                tvd_ft: 1.0,
                elev_ft: 1.0,
                source: TopSource::User,
            }],
        )
        .unwrap();
        let geo = geometry_schema();
        let g = column_glossary(&[
            ("wells", wells.schema().as_ref()),
            ("geometry", geo.as_ref()),
            ("tops_input", tops_in.schema().as_ref()),
        ])
        .unwrap();
        assert_eq!(g.num_rows(), 18 + 7 + 7);
        let desc = g.column(4).as_string::<i32>();
        assert!((0..g.num_rows()).all(|r| !desc.value(r).is_empty()));
    }
}
