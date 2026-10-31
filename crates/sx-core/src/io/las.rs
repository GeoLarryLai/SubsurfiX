// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Per-well LAS 2.0 export through the `las_rs` crate.
//!
//! The text carries no UWI, company, field, service or date items, and no file
//! names, paths, software names or versions.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

use las_rs::core::types::ItemWrapper;
use las_rs::{CurveItem, HeaderItem, LASFile, LasError, Value};

use super::atomic::io_error;
use crate::error::{SxError, SxResult};
use crate::types::{LogKind, WellGeometry, WellHeader, WellLogs, CODE_NONE, GRID_STEP_FT};
use crate::units::canonical_unit;

/// The `NULL` item, written for every NaN cell.
const NULL_VALUE: f64 = -999.25;

/// Row layout passed to `format_las`: line prefix and cell separator.
const LHS_SPACER: &str = " ";
const SPACER: &str = "  ";

/// The three fixed `~Other` lines.
const OTHER_LINES: [&str; 3] = [
    "Provenance codes in <LOG>_SRC curves: 0 measured, 1 model, 9 unfilled gap,",
    "40 weighted neighbor copy, 41 donor well copy, 42 weak neighbor copy, 50 neighbor shape,",
    "51 cross-log fit, 52 donor shape, 53 own texture, 54 smoothed model. NULL = no value.",
];

/// Options for [`well_las_string`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LasExportOptions {
    /// Digits after the decimal point for every curve except `<NAME>_SRC`
    /// (which are written with 0). Default 4.
    pub precision: usize,
    /// The RESD curve in `logs` holds log10(ohm.m) and is written as linear
    /// ohm.m. Default true.
    pub resd_is_log10: bool,
}

impl Default for LasExportOptions {
    /// Precision 4, `resd_is_log10` true.
    fn default() -> Self {
        Self {
            precision: 4,
            resd_is_log10: true,
        }
    }
}

fn log_description(log: LogKind) -> &'static str {
    match log {
        LogKind::Gr => "Gamma ray",
        LogKind::Rhob => "Bulk density",
        LogKind::Nphi => "Neutron porosity",
        LogKind::Dtc => "Compressional slowness",
        LogKind::Resd => "Deep resistivity",
        LogKind::Pef => "Photoelectric factor",
    }
}

fn las_error(e: LasError) -> SxError {
    SxError::Format(e.to_string())
}

fn header_item(mnemonic: &str, unit: &str, value: Value, descr: &str) -> HeaderItem {
    HeaderItem {
        original_mnemonic: mnemonic.to_string(),
        session_mnemonic: mnemonic.to_string(),
        unit: unit.to_string(),
        value,
        descr: descr.to_string(),
        data: Value::default(),
    }
}

fn curve_info(mnemonic: &str, unit: &str, descr: &str) -> HeaderItem {
    header_item(mnemonic, unit, Value::default(), descr)
}

/// Rewrite every NaN cell of the `~A` rows as the exact NULL text.
///
/// `format_las` writes a NaN cell with its column's decimals, so a 0-decimal
/// column would get `-999`, which does not read back as NULL.
fn exact_null_cells(text: &str, data: &[Vec<f64>]) -> SxResult<String> {
    let null_text = NULL_VALUE.to_string();
    let is_null = |c: &Vec<f64>, r: usize| c.get(r).is_some_and(|v| v.is_nan());
    let mut out = String::with_capacity(text.len());
    let mut row: Option<usize> = None;
    for line in text.split_inclusive('\n') {
        let Some(r) = row else {
            out.push_str(line);
            if line.starts_with("~A") {
                row = Some(0);
            }
            continue;
        };
        row = Some(r + 1);
        if !data.iter().any(|c| is_null(c, r)) {
            out.push_str(line);
            continue;
        }
        let cells: Vec<&str> = line.split_whitespace().collect();
        if cells.len() != data.len() {
            return Err(SxError::Format(format!(
                "LAS row {r}: {} cells for {} curves",
                cells.len(),
                data.len()
            )));
        }
        out.push_str(LHS_SPACER);
        for (c, (cell, col)) in cells.iter().zip(data).enumerate() {
            if c > 0 {
                out.push_str(SPACER);
            }
            out.push_str(if is_null(col, r) { &null_text } else { cell });
        }
        out.push('\n');
    }
    Ok(out)
}

/// LAS text for one well.
///
/// Curves in order: `DEPT` (ft, `logs.grid.md_ft(i)`), `TVD`, `ELEV` (ft, from
/// `geometry`), each log present in `logs.curves` in suite order (unit from
/// `units::canonical_unit`; RESD becomes linear ohm.m `10^v` when
/// `opts.resd_is_log10`), then `<NAME>_SRC` provenance curves for the same logs
/// (`CODE_NONE` becomes NULL). The well section holds `STRT/STOP/STEP`,
/// `NULL` (-999.25) and `WELL` (the name with `:` replaced by `_` and line
/// breaks removed). `~Other` holds three fixed lines. Text layout is that of
/// `las_rs::writer::format_las`.
/// NaN rule: NaN samples are written as NULL. O(rows x curves).
///
/// # Errors
/// `SxError::Shape` (ids or grids disagree, vector lengths wrong),
/// `SxError::InvalidInput` (empty grid), `SxError::Format` (rejected by
/// `las_rs`).
pub fn well_las_string(
    header: &WellHeader,
    geometry: &WellGeometry,
    logs: &WellLogs,
    opts: &LasExportOptions,
) -> SxResult<String> {
    if logs.id != header.id || geometry.id != header.id {
        return Err(SxError::Shape(format!(
            "ids differ: header {}, geometry {}, logs {}",
            header.id.0, geometry.id.0, logs.id.0
        )));
    }
    if geometry.grid != logs.grid {
        return Err(SxError::Shape(format!(
            "well {}: geometry grid {:?} differs from logs grid {:?}",
            header.id.0, geometry.grid, logs.grid
        )));
    }
    geometry.validate()?;
    logs.validate()?;
    let grid = logs.grid;
    if grid.is_empty() {
        return Err(SxError::InvalidInput(format!(
            "well {}: empty grid",
            header.id.0
        )));
    }

    let mut curves = vec![
        curve_info("DEPT", "FT", "Measured depth below KB"),
        curve_info("TVD", "FT", "True vertical depth below KB"),
        curve_info("ELEV", "FT", "Elevation above sea level"),
    ];
    let mut data: Vec<Vec<f64>> = vec![
        (0..grid.len).map(|i| grid.md_ft(i)).collect(),
        geometry.tvd_ft.clone(),
        geometry.elev_ft.clone(),
    ];
    let present: Vec<LogKind> = LogKind::ALL
        .iter()
        .copied()
        .filter(|l| logs.curves.contains_key(l))
        .collect();
    for &log in &present {
        let values = logs.curves.get(&log).map(Vec::as_slice).unwrap_or(&[]);
        let linearize = log == LogKind::Resd && opts.resd_is_log10;
        curves.push(curve_info(
            log.name(),
            canonical_unit(log),
            log_description(log),
        ));
        data.push(
            values
                .iter()
                .map(|&v| {
                    let v = f64::from(v);
                    if linearize {
                        10f64.powf(v)
                    } else {
                        v
                    }
                })
                .collect(),
        );
    }
    let mut column_fmt: HashMap<usize, String> = HashMap::new();
    for &log in &present {
        let name = format!("{}_SRC", log.name());
        let codes = logs.provenance.get(&log).map(Vec::as_slice).unwrap_or(&[]);
        column_fmt.insert(curves.len(), "%.0f".to_string());
        curves.push(curve_info(
            &name,
            "",
            &format!("{} provenance code", log.name()),
        ));
        data.push(
            codes
                .iter()
                .map(|&c| {
                    if c == CODE_NONE {
                        f64::NAN
                    } else {
                        f64::from(c)
                    }
                })
                .collect(),
        );
    }

    let well_name: String = header
        .name
        .chars()
        .filter(|c| *c != '\r' && *c != '\n')
        .map(|c| if c == ':' { '_' } else { c })
        .collect();
    let strt = grid.md_ft(0);
    let stop = grid.md_ft(grid.len.saturating_sub(1));
    let step = if grid.len > 1 { GRID_STEP_FT } else { 0.0 };
    let mut las = LASFile::create_default();
    las.well_section.items = vec![
        header_item("STRT", "FT", Value::Float(strt), "START DEPTH"),
        header_item("STOP", "FT", Value::Float(stop), "STOP DEPTH"),
        header_item("STEP", "FT", Value::Float(step), "STEP"),
        header_item("NULL", "", Value::Float(NULL_VALUE), "NULL VALUE"),
        header_item("WELL", "", Value::Str(well_name), "WELL NAME"),
    ]
    .into_iter()
    .map(ItemWrapper::Header)
    .collect();
    las.curves_section.items = curves
        .into_iter()
        .zip(data.iter().cloned())
        .map(|(header, curve_data)| {
            ItemWrapper::Curve(CurveItem {
                header,
                curve_data,
                string_data: None,
                dtype_override: None,
            })
        })
        .collect();
    las.other_section = OTHER_LINES.join("\n");
    let fmt = format!("%.{}f", opts.precision);
    let text = las_rs::writer::format_las(
        &las,
        Some(2.0),
        None,
        false,
        Some(&fmt),
        None,
        LHS_SPACER,
        SPACER,
        None,
        &column_fmt,
        Some(step),
        Some(strt),
        Some(stop),
    )
    .map_err(las_error)?;
    exact_null_cells(&text, &data)
}

/// [`well_las_string`] with default options, written to exactly `path` (any
/// file name; not atomic). The file is synced before return.
///
/// The default takes RESD as log10, which is how `logs_final` stores it.
///
/// # Errors
/// As [`well_las_string`]; `SxError::Io` for the file.
pub fn write_well_las(
    path: &Path,
    header: &WellHeader,
    geometry: &WellGeometry,
    logs: &WellLogs,
) -> SxResult<()> {
    let text = well_las_string(header, geometry, logs, &LasExportOptions::default())?;
    let mut file = std::fs::File::create(path).map_err(|e| io_error(path, e))?;
    file.write_all(text.as_bytes())
        .map_err(|e| io_error(path, e))?;
    file.sync_all().map_err(|e| io_error(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GridRange, WellId};

    fn header(name: &str) -> WellHeader {
        WellHeader {
            id: WellId(1),
            name: name.into(),
            x_m: 1000.0,
            y_m: 2000.0,
            kb_elev_ft: 3000.0,
        }
    }

    fn example_geometry() -> WellGeometry {
        WellGeometry {
            id: WellId(1),
            grid: GridRange::new(2000, 2).unwrap(),
            tvd_ft: vec![1000.0, 1000.5],
            elev_ft: vec![2000.0, 1999.5],
            x_m: vec![1000.0, 1000.0],
            y_m: vec![2000.0, 2000.0],
        }
    }

    fn example_logs() -> WellLogs {
        let mut l = WellLogs::new(WellId(1), GridRange::new(2000, 2).unwrap());
        l.curves.insert(LogKind::Gr, vec![45.25, f32::NAN]);
        l.provenance.insert(LogKind::Gr, vec![0, CODE_NONE]);
        l.curves.insert(LogKind::Resd, vec![1.0, 2.0]);
        l.provenance.insert(LogKind::Resd, vec![0, 1]);
        l
    }

    fn read_back(text: &str) -> LASFile {
        las_rs::parse_with(text, &las_rs::ReadOptions::default()).unwrap()
    }

    #[test]
    fn example_e_read_back() {
        let text = well_las_string(
            &header("W-1"),
            &example_geometry(),
            &example_logs(),
            &LasExportOptions::default(),
        )
        .unwrap();
        let las = read_back(&text);
        let names: Vec<(&str, &str)> = las
            .curves_section
            .items
            .iter()
            .map(|i| (i.session_mnemonic(), i.unit()))
            .collect();
        assert_eq!(
            names,
            [
                ("DEPT", "FT"),
                ("TVD", "FT"),
                ("ELEV", "FT"),
                ("GR", "gAPI"),
                ("RESD", "ohm.m"),
                ("GR_SRC", ""),
                ("RESD_SRC", "")
            ]
        );
        assert_eq!(las.curve_data("DEPT").unwrap(), [1000.0, 1000.5]);
        assert_eq!(las.curve_data("TVD").unwrap(), [1000.0, 1000.5]);
        assert_eq!(las.curve_data("ELEV").unwrap(), [2000.0, 1999.5]);
        assert_eq!(las.curve_data("GR").unwrap()[0], 45.25);
        assert!(las.curve_data("GR").unwrap()[1].is_nan());
        assert_eq!(las.curve_data("RESD").unwrap(), [10.0, 100.0]);
        let gr_src = las.curve_data("GR_SRC").unwrap();
        assert_eq!(gr_src[0], 0.0);
        assert!(gr_src[1].is_nan());
        assert_eq!(las.curve_data("RESD_SRC").unwrap(), [0.0, 1.0]);
        assert_eq!(las.well_value("WELL").as_deref(), Some("W-1"));
        assert_eq!(las.well_value("NULL").as_deref(), Some("-999.25"));
        assert!(las
            .version_section
            .items
            .iter()
            .any(|i| i.original_mnemonic() == "VERS" && i.value().display_str() == "2"));
        assert!(las
            .other_section
            .contains("54 smoothed model. NULL = no value."));
        // NaN cells are the exact NULL text, codes have no decimals.
        let last_row: Vec<&str> = text.lines().last().unwrap().split_whitespace().collect();
        assert_eq!(last_row[3], "-999.25");
        assert_eq!(last_row[5], "-999.25");
        assert_eq!(last_row[6], "1");

        // Same result through the file writer.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("w.las");
        write_well_las(&p, &header("W-1"), &example_geometry(), &example_logs()).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), text);
    }

    #[test]
    fn linear_resd_option_and_precision() {
        let opts = LasExportOptions {
            precision: 2,
            resd_is_log10: false,
        };
        let text =
            well_las_string(&header("W-1"), &example_geometry(), &example_logs(), &opts).unwrap();
        let las = read_back(&text);
        assert_eq!(las.curve_data("RESD").unwrap(), [1.0, 2.0]);
        assert!(text.contains(" 1000.00 "));
    }

    #[test]
    fn checks_and_name_cleanup() {
        let geo = example_geometry();
        let logs = example_logs();
        let opts = LasExportOptions::default();

        let mut other = geo.clone();
        other.id = WellId(2);
        assert!(matches!(
            well_las_string(&header("W-1"), &other, &logs, &opts),
            Err(SxError::Shape(_))
        ));
        let mut other = geo.clone();
        other.grid = GridRange::new(2001, 2).unwrap();
        assert!(matches!(
            well_las_string(&header("W-1"), &other, &logs, &opts),
            Err(SxError::Shape(_))
        ));
        let mut h = header("W-1");
        h.id = WellId(3);
        assert!(matches!(
            well_las_string(&h, &geo, &logs, &opts),
            Err(SxError::Shape(_))
        ));
        let empty_geo = WellGeometry {
            id: WellId(1),
            grid: GridRange::new(0, 0).unwrap(),
            tvd_ft: vec![],
            elev_ft: vec![],
            x_m: vec![],
            y_m: vec![],
        };
        let empty_logs = WellLogs::new(WellId(1), GridRange::new(0, 0).unwrap());
        assert!(matches!(
            well_las_string(&header("W-1"), &empty_geo, &empty_logs, &opts),
            Err(SxError::InvalidInput(_))
        ));

        // Colons become underscores, line breaks vanish; no extra header items.
        let text = well_las_string(&header("A:B\r\nC"), &geo, &logs, &opts).unwrap();
        assert!(text.contains(" WELL."));
        let las = read_back(&text);
        assert_eq!(las.well_value("WELL").as_deref(), Some("A_BC"));
        assert!(las.well_value("UWI").is_none());
        assert_eq!(las.well_section.items.len(), 5);
        assert!(las.params_section.items.is_empty());
    }

    #[test]
    fn curves_without_logs_still_export() {
        let logs = WellLogs::new(WellId(1), GridRange::new(2000, 2).unwrap());
        let text = well_las_string(
            &header("W-1"),
            &example_geometry(),
            &logs,
            &LasExportOptions::default(),
        )
        .unwrap();
        assert_eq!(read_back(&text).curves_section.items.len(), 3);
    }
}
