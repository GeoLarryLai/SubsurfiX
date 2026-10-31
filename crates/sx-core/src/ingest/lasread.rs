// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! LAS reading for ingest: `las_rs` plus the NULL, depth-unit and index-order
//! rules ingest relies on.

use std::path::Path;

use las_rs::core::types::ItemWrapper;
use las_rs::{CurveItem, LASFile, LasError, ReadOptions};

/// Always a null, and the NULL used when a file declares none (or an
/// unparsable one).
const DEFAULT_NULL: f64 = -999.25;

/// A value within this distance of the declared NULL or of `DEFAULT_NULL` is
/// a null.
const NULL_ABS_TOL: f64 = 1e-3;

/// Unit of the depth index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DepthUnit {
    /// Feet.
    Feet,
    /// Meters.
    Meters,
    /// Anything else (the trimmed unit text, possibly empty).
    Unknown(String),
}

/// Classify a unit text as feet, meters or unknown (trimmed, case-insensitive).
pub(crate) fn parse_depth_unit(unit: &str) -> DepthUnit {
    let u = unit.trim().to_ascii_uppercase();
    match u.as_str() {
        "F" | "FT" | "FEET" | "FOOT" => DepthUnit::Feet,
        "M" | "METER" | "METERS" | "METRE" | "METRES" | "MTR" => DepthUnit::Meters,
        _ => DepthUnit::Unknown(unit.trim().to_owned()),
    }
}

/// Direction of an index column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexOrder {
    /// Fewer than two finite values.
    Empty,
    /// Never decreases, increases at least once.
    Ascending,
    /// Never increases, decreases at least once.
    Descending,
    /// All finite values equal.
    Constant,
    /// Both increases and decreases.
    Mixed,
}

/// Direction of `idx` over consecutive finite values (NaN skipped).
pub(crate) fn index_order(idx: &[f64]) -> IndexOrder {
    let mut n_increase = 0usize;
    let mut n_decrease = 0usize;
    let mut n_finite = 0usize;
    let mut prev: Option<f64> = None;
    for &v in idx.iter().filter(|v| v.is_finite()) {
        n_finite += 1;
        if let Some(p) = prev {
            if v > p {
                n_increase += 1;
            } else if v < p {
                n_decrease += 1;
            }
        }
        prev = Some(v);
    }
    if n_finite < 2 {
        IndexOrder::Empty
    } else if n_decrease == 0 && n_increase > 0 {
        IndexOrder::Ascending
    } else if n_increase == 0 && n_decrease > 0 {
        IndexOrder::Descending
    } else if n_increase == 0 && n_decrease == 0 {
        IndexOrder::Constant
    } else {
        IndexOrder::Mixed
    }
}

/// First `~W` item named `mnemonic` (original name, trimmed, case-insensitive).
fn well_item<'a>(las: &'a LASFile, mnemonic: &str) -> Option<&'a ItemWrapper> {
    las.well_section
        .items
        .iter()
        .find(|i| i.original_mnemonic().trim().eq_ignore_ascii_case(mnemonic))
}

/// Unit of the first `~W` item named `mnemonic`.
pub(crate) fn well_unit<'a>(las: &'a LASFile, mnemonic: &str) -> Option<&'a str> {
    well_item(las, mnemonic).map(ItemWrapper::unit)
}

/// Bit size in inches from the `~P` (or `~W`) section (`BS`, `BIT`, `BITSIZE`,
/// `BIT_SIZE`), if it parses to a value between 3 and 30 in. `MM` and `CM`
/// units are converted.
pub(crate) fn param_bit_size_in(las: &LASFile) -> Option<f64> {
    const NAMES: [&str; 4] = ["BS", "BIT", "BITSIZE", "BIT_SIZE"];
    las.params_section
        .items
        .iter()
        .chain(las.well_section.items.iter())
        .filter(|i| {
            NAMES
                .iter()
                .any(|n| i.original_mnemonic().trim().eq_ignore_ascii_case(n))
        })
        .find_map(|i| {
            let v = i.value().display_str().trim().parse::<f64>().ok()?;
            let f = match i.unit().trim().to_ascii_lowercase().as_str() {
                "mm" => 1.0 / 25.4,
                "cm" => 1.0 / 2.54,
                _ => 1.0,
            };
            let v = v * f;
            (v.is_finite() && (3.0..=30.0).contains(&v)).then_some(v)
        })
}

/// The declared `NULL` if it is a finite number, else -999.25.
fn declared_null(las: &LASFile) -> f64 {
    well_item(las, "NULL")
        .and_then(|i| i.value().display_str().trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(DEFAULT_NULL)
}

/// NULL rule: a value within `NULL_ABS_TOL` of `null` or of -999.25, or not
/// finite, becomes NaN.
fn apply_null(values: &mut [f64], null: f64) {
    for v in values.iter_mut() {
        if !v.is_finite()
            || (*v - null).abs() <= NULL_ABS_TOL
            || (*v - DEFAULT_NULL).abs() <= NULL_ABS_TOL
        {
            *v = f64::NAN;
        }
    }
}

/// Read a LAS file with `las_rs` default options, then apply the NULL rule to
/// every curve, the index included.
///
/// # Errors
/// Every `las_rs` read error.
pub(crate) fn read_las(path: &Path) -> Result<LASFile, LasError> {
    let mut las = las_rs::read_file_with(path, &ReadOptions::default())?;
    let null = declared_null(&las);
    for item in &mut las.curves_section.items {
        if let ItemWrapper::Curve(c) = item {
            apply_null(&mut c.curve_data, null);
        }
    }
    Ok(las)
}

/// The `~C` curves in column order.
pub(crate) fn curves(las: &LASFile) -> Vec<&CurveItem> {
    las.curves_section
        .items
        .iter()
        .filter_map(|i| match i {
            ItemWrapper::Curve(c) => Some(c),
            ItemWrapper::Header(_) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_units_and_order() {
        assert_eq!(parse_depth_unit(" ft "), DepthUnit::Feet);
        assert_eq!(parse_depth_unit("Metres"), DepthUnit::Meters);
        assert_eq!(
            parse_depth_unit(" KM "),
            DepthUnit::Unknown("KM".to_string())
        );
        assert_eq!(
            index_order(&[3.0, f64::NAN, 2.0, 1.0]),
            IndexOrder::Descending
        );
        assert_eq!(index_order(&[1.0, 1.0, 2.0]), IndexOrder::Ascending);
        assert_eq!(index_order(&[1.0, 2.0, 1.0]), IndexOrder::Mixed);
        assert_eq!(index_order(&[1.0, 1.0]), IndexOrder::Constant);
        assert_eq!(index_order(&[1.0]), IndexOrder::Empty);
    }

    #[test]
    fn null_rule_within_tolerance() {
        let text =
            "~V\n VERS. 2.0 :\n WRAP. NO :\n~W\n NULL. -9999 :\n~C\n DEPT.FT :\n GR.GAPI :\n\
                    ~A\n-9999 1\n2 -9999.0005\n3 -999.2505\n4 -999.24\n";
        let mut las = las_rs::parse_with(text, &ReadOptions::default()).unwrap();
        let null = declared_null(&las);
        assert_eq!(null, -9999.0);
        for item in &mut las.curves_section.items {
            if let ItemWrapper::Curve(c) = item {
                apply_null(&mut c.curve_data, null);
            }
        }
        let cols = curves(&las);
        assert!(cols[0].curve_data[0].is_nan());
        assert!(cols[1].curve_data[1].is_nan() && cols[1].curve_data[2].is_nan());
        assert_eq!(cols[1].curve_data[3], -999.24);
        assert_eq!(cols[1].curve_data[0], 1.0);
    }
}
