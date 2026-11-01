// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Per-file read: depth curve, depth cleaning, candidates and the unit flow
//! (rules 4.6.3 to 4.6.5 and 4.7).

use std::collections::BTreeMap;
use std::path::Path;

use las_rs::{CurveItem, LASFile};

use super::aliases::AliasTable;
use super::families::curve_family;
use super::lasread::{self, DepthUnit, IndexOrder};
use crate::error::SxResult;
use crate::types::LogKind;
use crate::units::{
    apply_divisor, header_unit, physical_bounds, plausible_median, HeaderUnit, FT_PER_M,
    PLAUSIBLE_MIN_IN_BOUNDS_FRAC, PLAUSIBLE_MIN_SAMPLES,
};
use crate::util::percentile_f32;

/// Offset added to sandstone-scale (`ss` family) neutron porosity.
pub(crate) const NPHI_SS_OFFSET: f64 = -0.04;

/// Median by the percentile rule of 4.11 (`percentile(.., 50)`); NaN skipped,
/// NaN when no finite value exists.
pub(crate) fn median_f32_pct(values: &[f32]) -> f64 {
    percentile_f32(values, 50.0).unwrap_or(f64::NAN)
}

/// DTC median (µs/ft) above which a curve without a known unit is read as µs/m.
pub(crate) const DTC_USM_GUESS_MEDIAN: f64 = 190.0;

/// Divisor guessed from the median of the finite raw values.
///
/// NPHI: `med > 100` gives 1000, `med > 1` gives 100, else 1. DTC: `med > 190`
/// gives `FT_PER_M` (µs/m read as µs/ft; shallow unconsolidated rock reaches
/// about 180 µs/ft, while µs/m logs sit at about 150–600). RESD: `med > 5000` gives 1000. RHOB:
/// `med > 100` gives 1000. GR and PEF: 1. A NaN median gives 1.
pub(crate) fn infer_divisor(log: LogKind, median: f64) -> f64 {
    if median.is_nan() {
        return 1.0;
    }
    match log {
        LogKind::Nphi => {
            if median > 100.0 {
                1000.0
            } else if median > 1.0 {
                100.0
            } else {
                1.0
            }
        }
        LogKind::Dtc => {
            if median > DTC_USM_GUESS_MEDIAN {
                FT_PER_M
            } else {
                1.0
            }
        }
        LogKind::Resd => {
            if median > 5000.0 {
                1000.0
            } else {
                1.0
            }
        }
        LogKind::Rhob => {
            if median > 100.0 {
                1000.0
            } else {
                1.0
            }
        }
        LogKind::Gr | LogKind::Pef => 1.0,
    }
}

/// A candidate that passed the unit flow.
#[derive(Debug, Clone)]
pub(crate) struct Accepted {
    /// Values on the file's cleaned depth, canonical units, NaN = missing.
    pub values: Vec<f32>,
    /// `header`, `header_veto`, `blank` or `stats`.
    pub unit_rule: &'static str,
    /// `passed`, `skipped` or `retry_passed`.
    pub verify: &'static str,
    /// Total divisor applied.
    pub divisor: f64,
    /// `-0.04` for an `ss` NPHI, else `0`.
    pub nphi_offset: f64,
}

/// One column of a file that matches an alias list of one log.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    /// Column index in the file.
    pub col: usize,
    /// Lookup mnemonic as in the file (after duplicate renaming).
    pub mnemonic: String,
    /// Raw header unit.
    pub unit: String,
    /// Alias priority (smaller wins).
    pub priority: usize,
    /// Accepted values, or the rejection reason.
    pub result: Result<Accepted, &'static str>,
}

/// A file read and cleaned.
#[derive(Debug, Clone)]
pub(crate) struct LoadedFile {
    /// File name only.
    pub name: String,
    /// Cleaned depth, feet, strictly ascending.
    pub z: Vec<f64>,
    /// `ft`, `m`, or `unknown:<raw>`.
    pub depth_unit: String,
    /// Rows dropped for a bad depth.
    pub n_nan_depth: usize,
    /// Rows removed as repeated depths.
    pub n_dup_depth: usize,
    /// The file's index column runs downward.
    pub descending: bool,
    /// Report notes of this file.
    pub notes: Vec<String>,
    /// Candidates per log in `(priority, column)` order.
    pub candidates: BTreeMap<LogKind, Vec<Candidate>>,
    /// Caliper (inches), density correction (g/cm³), bit size (inches) and
    /// logging speed (ft/min, absolute) and temperature (°C) on `z`, if present.
    pub aux: [Option<Vec<f32>>; 5],
}

/// Caliper mnemonics, in priority order. `DCAL` is last: it is an absolute
/// density-tool caliper in many vendors' files (KGS: "Caliper from Density
/// Tool", "CDL Caliper") but a differential caliper in others (Schlumberger:
/// "Differential Caliper"); see [`is_differential_caliper`].
const CALIPER: [&str; 12] = [
    "CALI", "CAL", "HCAL", "CALX", "CALY", "CAL1", "C1", "LCAL", "SCAL", "CALS", "CLDC", "DCAL",
];

/// True when a caliper curve is differential (hole size minus bit size), not
/// an absolute hole diameter.
///
/// The curve description says "differential", the mnemonic starts with
/// `DIFF`, or a `DCAL` curve with no description stays small (90th
/// percentile below 4 in) and has at least 5 % negative values. An absolute
/// caliper in open hole reads near the bit size (6–17 in); a closed absolute
/// caliper reads small but is not negative (KGS `DCAL` "CDL Caliper" reads
/// about 0.35 in closed, 8–11 in open).
pub(crate) fn is_differential_caliper(mnemonic: &str, descr: &str, values: &[f32]) -> bool {
    let d = descr.to_ascii_uppercase();
    if d.contains("DIFFERENTIAL") || d.contains("DIFF CAL") || d.contains("DIFF. CAL") {
        return true;
    }
    let m = mnemonic.trim().to_ascii_uppercase();
    if m.starts_with("DIFF") {
        return true;
    }
    if m == "DCAL" && d.trim().is_empty() {
        let finite: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            return false;
        }
        let neg = finite.iter().filter(|&&x| x < 0.0).count() as f64 / finite.len() as f64;
        let p90 = percentile_f32(&finite, 90.0).unwrap_or(f64::NAN);
        return neg >= 0.05 && p90 < 4.0;
    }
    false
}
/// Density-correction mnemonics, in priority order.
const DENS_CORR: [&str; 7] = ["DRHO", "HDRA", "RHOC", "DCOR", "ZCOR", "CORR", "DRH"];
/// Bit-size mnemonics, in priority order.
const BIT_SIZE: [&str; 4] = ["BS", "BIT", "BITSIZE", "BIT_SIZE"];
/// Logging-speed mnemonics, in priority order.
const LOG_SPEED: [&str; 5] = ["LSPD", "SPD", "SPEED", "TSPD", "CS"];

/// Temperature-curve mnemonics (tool, mud, borehole or well temperature),
/// in priority order. Curves only; `BHT` in the ~P section is a parameter.
const TEMPERATURE: [&str; 9] =
    ["TEMP", "WTEP", "ETEMP", "BHT", "TMP", "MTEM", "HTEM", "TEMP_C", "TEMP_F"];

/// Convert a temperature to °C from the LAS unit text. `None` when the unit
/// is not a known temperature unit.
pub(crate) fn temperature_to_c(unit: &str) -> Option<fn(f32) -> f32> {
    let u = unit.trim().trim_start_matches(['°', 'º']).to_ascii_lowercase();
    match u.as_str() {
        "degc" | "c" | "deg c" | "degrees c" | "celsius" | "dc" => Some(|x| x),
        "degf" | "f" | "deg f" | "degrees f" | "fahrenheit" | "df" => {
            Some(|x| (x - 32.0) / 1.8)
        }
        "k" | "degk" | "kelvin" => Some(|x| x - 273.15),
        _ => None,
    }
}

/// Temperature curve in °C. A curve with an unknown unit is read as °C and a
/// `temp_unit_assumed_degc:<mnemonic>:<unit>` note is added.
fn temperature_column(
    curves: &[&CurveItem],
    depth_col: usize,
    order: &[usize],
    starts: &[usize],
    notes: &mut Vec<String>,
) -> Option<Vec<f32>> {
    let mut best: Option<(usize, usize)> = None;
    for (col, c) in curves.iter().enumerate() {
        if col == depth_col {
            continue;
        }
        let m = c.header.session_mnemonic.trim().to_ascii_uppercase();
        let o = c.header.original_mnemonic.trim().to_ascii_uppercase();
        if let Some(p) = TEMPERATURE.iter().position(|a| *a == m || *a == o) {
            if best.is_none_or(|(bp, _)| p < bp) {
                best = Some((p, col));
            }
        }
    }
    let (_, col) = best?;
    let c = curves.get(col)?;
    let mut v = clean_column(&c.curve_data, order, starts);
    let conv = match temperature_to_c(&c.header.unit) {
        Some(f) => f,
        None => {
            let m = c.header.original_mnemonic.trim();
            tracing::warn!(curve = m, unit = c.header.unit.trim(), "unknown temperature unit; read as degC");
            notes.push(format!("temp_unit_assumed_degc:{m}:{}", c.header.unit.trim()));
            |x| x
        }
    };
    v.iter_mut().for_each(|x| *x = conv(*x));
    Some(v)
}

/// Aux column `which` (0 caliper, 1 density correction, 2 bit size, 3 logging
/// speed) in its QC unit.
fn aux_column(
    curves: &[&CurveItem],
    depth_col: usize,
    which: usize,
    order: &[usize],
    starts: &[usize],
) -> Option<Vec<f32>> {
    let list: &[&str] = match which {
        0 => &CALIPER,
        1 => &DENS_CORR,
        2 => &BIT_SIZE,
        _ => &LOG_SPEED,
    };
    let mut best: Option<(usize, usize)> = None;
    for (col, c) in curves.iter().enumerate() {
        if col == depth_col {
            continue;
        }
        let m = c.header.session_mnemonic.trim().to_ascii_uppercase();
        let o = c.header.original_mnemonic.trim().to_ascii_uppercase();
        if which == 0 && caliper_is_differential(c, order, starts) {
            continue;
        }
        if let Some(p) = list.iter().position(|a| *a == m || *a == o) {
            if best.is_none_or(|(bp, _)| p < bp) {
                best = Some((p, col));
            }
        }
    }
    let (_, col) = best?;
    let c = curves.get(col)?;
    let mut v = clean_column(&c.curve_data, order, starts);
    let unit = c.header.unit.trim().to_ascii_lowercase();
    let f = match (which, unit.as_str()) {
        (0 | 2, "mm") => 1.0 / 25.4,
        (0 | 2, "cm") => 1.0 / 2.54,
        (1, "kg/m3") => 0.001,
        (3, "m/min") => 3.280_84,
        (3, "ft/h" | "ft/hr" | "f/hr" | "fph") => 1.0 / 60.0,
        (3, "m/h" | "m/hr") => 3.280_84 / 60.0,
        _ => 1.0,
    };
    if f != 1.0 {
        v.iter_mut().for_each(|x| *x *= f);
    }
    if which == 3 {
        v.iter_mut().for_each(|x| *x = x.abs());
    }
    Some(v)
}

fn caliper_is_differential(c: &CurveItem, order: &[usize], starts: &[usize]) -> bool {
    // Some files put the description in the value column (`DCAL.GAPI  CDL Caliper:`).
    let descr = format!("{} {}", c.header.descr, c.header.value.display_str());
    let d = descr.to_ascii_uppercase();
    let m = c.header.original_mnemonic.trim().to_ascii_uppercase();
    let is_cal = CALIPER.contains(&m.as_str())
        || m.starts_with("DIFF")
        || d.contains("CALIPER");
    if !is_cal {
        return false;
    }
    let v = if m == "DCAL" && d.trim().is_empty() {
        clean_column(&c.curve_data, order, starts)
    } else {
        Vec::new()
    };
    is_differential_caliper(&m, &descr, &v)
}

/// Differential caliper in inches on `z` (first one found), if any.
fn differential_caliper(
    curves: &[&CurveItem],
    depth_col: usize,
    order: &[usize],
    starts: &[usize],
) -> Option<(String, Vec<f32>)> {
    curves.iter().enumerate().find_map(|(col, c)| {
        if col == depth_col || !caliper_is_differential(c, order, starts) {
            return None;
        }
        let mut v = clean_column(&c.curve_data, order, starts);
        let f = match c.header.unit.trim().to_ascii_lowercase().as_str() {
            "mm" => 1.0 / 25.4,
            "cm" => 1.0 / 2.54,
            _ => 1.0,
        };
        if f != 1.0 {
            v.iter_mut().for_each(|x| *x *= f);
        }
        Some((c.header.original_mnemonic.trim().to_string(), v))
    })
}

/// Caliper and bit-size aux columns, with notes.
///
/// An absolute caliper is used as is. A differential caliper is never read as
/// hole size: it becomes `bit size + differential` when the bit size is known
/// (a bit-size curve, else `BS` in `~P`/`~W`), and is dropped otherwise. A
/// parameter bit size fills the bit-size column when there is no curve.
fn caliper_and_bit_size(
    las: &LASFile,
    curves: &[&CurveItem],
    depth_col: usize,
    order: &[usize],
    starts: &[usize],
    n: usize,
    notes: &mut Vec<String>,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut cal = aux_column(curves, depth_col, 0, order, starts);
    let mut bs = aux_column(curves, depth_col, 2, order, starts);
    if bs.is_none() {
        if let Some(b) = lasread::param_bit_size_in(las) {
            bs = Some(vec![b as f32; n]);
            notes.push(format!("bit_size_from_params:{b}"));
        }
    }
    if cal.is_none() {
        if let Some((m, d)) = differential_caliper(curves, depth_col, order, starts) {
            match &bs {
                Some(b) => {
                    cal = Some(d.iter().zip(b).map(|(x, y)| x + y).collect());
                    notes.push(format!("caliper_from_differential:{m}+bit_size"));
                }
                None => notes.push(format!("caliper_differential_dropped_no_bit_size:{m}")),
            }
        }
    }
    (cal, bs)
}

/// Outcome of reading one file.
#[derive(Debug)]
pub(crate) enum LoadResult {
    /// The file was usable.
    Loaded(Box<LoadedFile>),
    /// The file was skipped; the string is the report note.
    Skipped(String),
}

/// Outcome of the verify step.
enum Verdict {
    Skipped,
    Passed,
    Failed { median: f64 },
}

fn finite_values(v: &[f32]) -> Vec<f32> {
    v.iter().copied().filter(|x| x.is_finite()).collect()
}

fn verify(log: LogKind, v: &[f32]) -> Verdict {
    let finite = finite_values(v);
    let n = finite.len();
    if n < PLAUSIBLE_MIN_SAMPLES {
        return Verdict::Skipped;
    }
    let med = median_f32_pct(&finite);
    let bounds = physical_bounds(log);
    let inside = finite
        .iter()
        .filter(|&&x| bounds.contains(f64::from(x)))
        .count();
    let frac = inside as f64 / n as f64;
    if plausible_median(log).contains(med) && frac >= PLAUSIBLE_MIN_IN_BOUNDS_FRAC {
        Verdict::Passed
    } else {
        Verdict::Failed { median: med }
    }
}

fn reject_reason(log: LogKind, median: f64) -> &'static str {
    if plausible_median(log).contains(median) {
        "few_in_bounds"
    } else {
        "median_out_of_range"
    }
}

/// The unit flow of rule 4.7 for one candidate.
///
/// `raw` holds the file's values (finite or NaN) in the raw header unit `unit`.
/// Returns the accepted candidate or the rejection reason (`empty`,
/// `median_out_of_range`, `few_in_bounds`).
pub(crate) fn unit_flow(
    log: LogKind,
    mnemonic: &str,
    unit: &str,
    raw: Vec<f32>,
) -> SxResult<Result<Accepted, &'static str>> {
    let finite = finite_values(&raw);
    if finite.is_empty() {
        return Ok(Err("empty"));
    }
    let med_raw = median_f32_pct(&finite);
    let (mut divisor, unit_rule) = match header_unit(log, unit) {
        HeaderUnit::Unrecognized => {
            let d = infer_divisor(log, med_raw);
            if d != 1.0 {
                tracing::warn!(
                    log = ?log,
                    mnemonic,
                    unit,
                    divisor = d,
                    "header unit not recognised; unit guessed from curve magnitude"
                );
            }
            (d, "stats")
        }
        HeaderUnit::Blank => (1.0, "blank"),
        HeaderUnit::Divisor(x) => {
            if x != 1.0
                && matches!(log, LogKind::Nphi | LogKind::Rhob)
                && infer_divisor(log, med_raw) == 1.0
            {
                (1.0, "header_veto")
            } else {
                (x, "header")
            }
        }
    };
    let mut v = raw;
    if divisor != 1.0 {
        apply_divisor(&mut v, divisor)?;
    }
    let verify_label = match verify(log, &v) {
        Verdict::Skipped => "skipped",
        Verdict::Passed => "passed",
        Verdict::Failed { median } => {
            // An explicit, recognised sonic unit is trusted: no magnitude guess.
            let header_known = matches!(unit_rule, "header" | "header_veto");
            let d2 = if log == LogKind::Dtc && header_known {
                1.0
            } else {
                infer_divisor(log, median)
            };
            if d2 == 1.0 {
                return Ok(Err(reject_reason(log, median)));
            }
            apply_divisor(&mut v, d2)?;
            divisor *= d2;
            match verify(log, &v) {
                Verdict::Failed { median } => return Ok(Err(reject_reason(log, median))),
                Verdict::Passed => {
                    tracing::warn!(
                        log = ?log,
                        mnemonic,
                        unit,
                        divisor,
                        "unit guessed from curve magnitude (header unit missing or not recognised)"
                    );
                    "retry_passed"
                }
                Verdict::Skipped => "skipped",
            }
        }
    };
    let mut nphi_offset = 0.0;
    if log == LogKind::Nphi && curve_family(LogKind::Nphi, mnemonic) == "ss" {
        for x in v.iter_mut().filter(|x| x.is_finite()) {
            *x = (f64::from(*x) + NPHI_SS_OFFSET) as f32;
        }
        nphi_offset = NPHI_SS_OFFSET;
    }
    Ok(Ok(Accepted {
        values: v,
        unit_rule,
        verify: verify_label,
        divisor,
        nphi_offset,
    }))
}

/// Column of the depth curve: first `DEPT`, else `DEPTH`, else `MD` by upper
/// original mnemonic.
fn find_depth_col(curves: &[&CurveItem]) -> Option<usize> {
    ["DEPT", "DEPTH", "MD"].iter().find_map(|name| {
        curves
            .iter()
            .position(|c| c.header.original_mnemonic.trim().to_ascii_uppercase() == *name)
    })
}

/// Depth unit: first non-blank of the depth curve unit and the units of
/// `STRT`, `STOP`, `STEP`.
fn depth_unit_of(las: &LASFile, curves: &[&CurveItem], depth_col: usize) -> DepthUnit {
    let from_curve = curves.get(depth_col).map(|c| c.header.unit.as_str());
    let from_well = ["STRT", "STOP", "STEP"]
        .into_iter()
        .map(|m| lasread::well_unit(las, m));
    std::iter::once(from_curve)
        .chain(from_well)
        .flatten()
        .find(|u| !u.trim().is_empty())
        .map_or_else(
            || DepthUnit::Unknown(String::new()),
            lasread::parse_depth_unit,
        )
}

/// Value of the first finite sample of each group of equal depths, as `f32`.
fn clean_column(col: &[f64], order: &[usize], starts: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(starts.len().saturating_sub(1));
    for w in starts.windows(2) {
        let (a, b) = match w {
            [a, b] => (*a, *b),
            _ => continue,
        };
        let mut value = f32::NAN;
        for &row in order.get(a..b).unwrap_or(&[]) {
            let v = col.get(row).map_or(f32::NAN, |&x| x as f32);
            if v.is_finite() {
                value = v;
                break;
            }
        }
        out.push(value);
    }
    out
}

/// Read one LAS file and prepare its candidates (rules 4.6.3 to 4.7).
///
/// Read failure, or no depth curve, skips the file with a note. Errors only
/// from internal unit math (not reachable for valid divisors).
pub(crate) fn load_file(path: &Path, aliases: &AliasTable) -> SxResult<LoadResult> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let las = match lasread::read_las(path) {
        Ok(l) => l,
        Err(e) => return Ok(LoadResult::Skipped(format!("read_error:{name}:{e}"))),
    };
    let curves = lasread::curves(&las);
    let Some(depth_col) = find_depth_col(&curves) else {
        return Ok(LoadResult::Skipped(format!("no_depth_curve:{name}")));
    };
    let mut notes = Vec::new();
    let unit = depth_unit_of(&las, &curves, depth_col);
    let (factor, depth_unit) = match &unit {
        DepthUnit::Meters => (FT_PER_M, "m".to_string()),
        DepthUnit::Feet => (1.0, "ft".to_string()),
        DepthUnit::Unknown(u) => {
            notes.push(format!("depth_unit_unknown:{u}"));
            (1.0, format!("unknown:{u}"))
        }
    };
    let depth_raw: &[f64] = curves
        .get(depth_col)
        .map_or(&[], |c| c.curve_data.as_slice());
    let md_ft: Vec<f64> = depth_raw
        .iter()
        .map(|&d| if factor == 1.0 { d } else { d * factor })
        .collect();
    // Drop bad depths, stable-sort, collapse exactly equal depths.
    let mut order: Vec<usize> = (0..md_ft.len())
        .filter(|&i| md_ft.get(i).is_some_and(|d| d.is_finite() && *d >= 0.0))
        .collect();
    let n_nan_depth = md_ft.len() - order.len();
    order.sort_by(|&a, &b| {
        let da = md_ft.get(a).copied().unwrap_or(f64::NAN);
        let db = md_ft.get(b).copied().unwrap_or(f64::NAN);
        da.total_cmp(&db)
    });
    let mut z: Vec<f64> = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    for (pos, &row) in order.iter().enumerate() {
        let d = md_ft.get(row).copied().unwrap_or(f64::NAN);
        if z.last() != Some(&d) {
            z.push(d);
            starts.push(pos);
        }
    }
    starts.push(order.len());
    let n_dup_depth = order.len() - z.len();
    let descending = lasread::index_order(curves.first().map_or(&[], |c| c.curve_data.as_slice()))
        == IndexOrder::Descending;

    let mut candidates: BTreeMap<LogKind, Vec<Candidate>> = BTreeMap::new();
    for log in LogKind::ALL {
        let mut found: Vec<(usize, usize)> = Vec::new();
        for (col, curve) in curves.iter().enumerate() {
            if col == depth_col {
                continue;
            }
            let h = &curve.header;
            let p1 = aliases.priority(log, &h.session_mnemonic.trim().to_ascii_uppercase());
            let p2 = aliases.priority(log, &h.original_mnemonic.trim().to_ascii_uppercase());
            let p = match (p1, p2) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if let Some(p) = p {
                found.push((p, col));
            }
        }
        found.sort_unstable();
        let mut list = Vec::with_capacity(found.len());
        for (priority, col) in found {
            let Some(curve) = curves.get(col) else {
                continue;
            };
            let h = &curve.header;
            let raw = clean_column(&curve.curve_data, &order, &starts);
            let result = unit_flow(log, &h.session_mnemonic, &h.unit, raw)?;
            list.push(Candidate {
                col,
                mnemonic: h.session_mnemonic.clone(),
                unit: h.unit.clone(),
                priority,
                result,
            });
        }
        if !list.is_empty() {
            candidates.insert(log, list);
        }
    }
    let (cal, bs) =
        caliper_and_bit_size(&las, &curves, depth_col, &order, &starts, z.len(), &mut notes);
    let temp = temperature_column(&curves, depth_col, &order, &starts, &mut notes);
    Ok(LoadResult::Loaded(Box::new(LoadedFile {
        name,
        z,
        depth_unit,
        n_nan_depth,
        n_dup_depth,
        descending,
        aux: [
            cal,
            aux_column(&curves, depth_col, 1, &order, &starts),
            bs,
            aux_column(&curves, depth_col, 3, &order, &starts),
            temp,
        ],
        notes,
        candidates,
    })))
}
