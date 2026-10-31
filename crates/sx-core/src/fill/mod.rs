// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Gap fill: shared fill types, the fill
//! report and the `logs_filled` batch.
//!
//! [`rules`] holds the model write (range map, pre-write rejection, gaps-only
//! write); [`ladder`] holds the copy steps and the shaped gap fill. Values in
//! the fill state are in working units (RESD in `log10(ohm·m)`).

pub mod ladder;
pub mod rules;

pub mod features;
pub mod holdout;
pub mod lgbm;
pub mod sequential;

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Float64Array, Int32Array, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::config::NeighborsConfig;
use crate::error::{SxError, SxResult};
use crate::io::csv::csv_f64;
use crate::io::parquet::with_schema_tag;
use crate::neighbors::{Coverage, NeighborInputs, WellNeighborCurves};
use crate::types::{GridRange, LogKind, Provenance, WellId, WellLogs, CODE_NONE};

/// Schema tag of `logs_filled.parquet`.
pub(crate) const LOGS_FILLED_TAG: &str = "subsurfix.logs_filled/1";

/// Header line of `fill_report.csv`.
const REPORT_HEADER: &str = "kind,well_id,well_name,log,step,code,interval,donor_well_id,score,\
n_rows,md_top_ft,md_base_ft,range_gain,range_offset,reason";

/// Everything the fill reads besides the working state.
#[derive(Debug, Clone, Copy)]
pub struct FillInputs<'a> {
    /// Donor data, coordinates, coverage, exclusions.
    pub nbr: NeighborInputs<'a>,
    /// Neighbor curves of every well that is filled.
    pub neighbors: &'a BTreeMap<WellId, WellNeighborCurves>,
    /// Resistivity tool family per well (0 unknown, 1 induction, 2 laterolog, 3 lateral).
    pub resd_family: &'a BTreeMap<WellId, u8>,
    /// `layer_coord_source` per well.
    pub coord_source: &'a BTreeMap<WellId, u8>,
    /// Project `[neighbors]` settings, used by the donor helpers.
    pub neighbors_cfg: &'a NeighborsConfig,
}

/// Working state of one well during the fill.
#[derive(Debug, Clone, PartialEq)]
pub struct WellFillState {
    /// Values and provenance for all six logs on the well's grid.
    pub logs: WellLogs,
    /// Rejected model candidates at gap rows, NaN elsewhere. All six logs present.
    pub model_guess: BTreeMap<LogKind, Vec<f32>>,
    /// Coverage window of the well; `None` = no coverage (nothing is filled).
    pub coverage: Option<Coverage>,
}

impl WellFillState {
    /// Keep `Measured` finite samples; gap rows inside coverage get NaN + `Unfilled`;
    /// rows outside coverage get NaN + `CODE_NONE`. Logs missing from `logs` become all-gap.
    /// `None` coverage: every non-measured row is outside coverage.
    ///
    /// Values are copied as given (the fill works in working units, RESD in log10).
    /// NaN rule: skip (a NaN or non-`Measured` sample is discarded). Errors: `Shape`
    /// if `logs` fails [`WellLogs::validate`].
    pub fn reset_from(logs: &WellLogs, coverage: Option<Coverage>) -> SxResult<Self> {
        logs.validate()?;
        let grid = logs.grid;
        let mut out = WellLogs::new(logs.id, grid);
        let mut model_guess = BTreeMap::new();
        for log in LogKind::ALL {
            let values = logs.curves.get(&log);
            let prov = logs.provenance.get(&log);
            let mut v = vec![f32::NAN; grid.len];
            let mut p = vec![CODE_NONE; grid.len];
            for (i, (slot, code)) in v.iter_mut().zip(p.iter_mut()).enumerate() {
                let x = values.and_then(|c| c.get(i)).copied().unwrap_or(f32::NAN);
                let c = prov.and_then(|c| c.get(i)).copied().unwrap_or(CODE_NONE);
                if c == Provenance::Measured.code() && x.is_finite() {
                    *slot = x;
                    *code = c;
                } else if in_coverage(grid, coverage, i) {
                    *code = Provenance::Unfilled.code();
                }
            }
            out.curves.insert(log, v);
            out.provenance.insert(log, p);
            model_guess.insert(log, vec![f32::NAN; grid.len]);
        }
        Ok(Self {
            logs: out,
            model_guess,
            coverage,
        })
    }
}

/// Kind of a `fill_report.csv` row, in report sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReportKind {
    /// Range-map policy of one (well, log) prediction.
    Policy,
    /// Rows written by one step.
    Write,
    /// Candidates rejected by the pre-write rejection.
    Reject,
    /// A gap left unfilled.
    Unfilled,
}

impl ReportKind {
    /// Report text (`policy`, `write`, `reject`, `unfilled`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            ReportKind::Policy => "policy",
            ReportKind::Write => "write",
            ReportKind::Reject => "reject",
            ReportKind::Unfilled => "unfilled",
        }
    }
}

/// Fill step of a report row, in report sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FillStep {
    /// Model write.
    Model,
    /// Trusted neighbor copy.
    NeighborWeighted,
    /// Best real donor copy.
    RealDonor,
    /// Untrusted neighbor copy.
    NeighborWeak,
    /// Neighbor shape tilted to the gap ends.
    NeighborShape,
    /// Linear fit from the well's other logs.
    CrossLogFit,
    /// Donor well shape tilted to the gap ends.
    DonorShape,
    /// Model level plus the well's own texture.
    OwnTexture,
    /// Model guess, tilted if needed.
    SmoothedModel,
    /// No step (unfilled gaps).
    None,
}

impl FillStep {
    /// Report text.
    pub(crate) fn name(self) -> &'static str {
        match self {
            FillStep::Model => "model",
            FillStep::NeighborWeighted => "neighbor_weighted",
            FillStep::RealDonor => "real_donor",
            FillStep::NeighborWeak => "neighbor_weak",
            FillStep::NeighborShape => "neighbor_shape",
            FillStep::CrossLogFit => "cross_log_fit",
            FillStep::DonorShape => "donor_well_shape",
            FillStep::OwnTexture => "own_texture",
            FillStep::SmoothedModel => "smoothed_model",
            FillStep::None => "none",
        }
    }

    /// Provenance this step writes (`Unfilled` for [`FillStep::None`]).
    pub(crate) fn provenance(self) -> Provenance {
        match self {
            FillStep::Model => Provenance::Model,
            FillStep::NeighborWeighted => Provenance::NeighborWeighted,
            FillStep::RealDonor => Provenance::RealDonor,
            FillStep::NeighborWeak => Provenance::NeighborWeak,
            FillStep::NeighborShape => Provenance::NeighborShape,
            FillStep::CrossLogFit => Provenance::CrossLogFit,
            FillStep::DonorShape => Provenance::DonorShape,
            FillStep::OwnTexture => Provenance::OwnTexture,
            FillStep::SmoothedModel => Provenance::SmoothedModel,
            FillStep::None => Provenance::Unfilled,
        }
    }
}

/// One `fill_report.csv` row.
#[derive(Debug, Clone, PartialEq)]
pub struct FillReportRow {
    /// Row kind.
    pub kind: ReportKind,
    /// Well id.
    pub well_id: WellId,
    /// External well name.
    pub well_name: String,
    /// Target log.
    pub log: LogKind,
    /// Fill step.
    pub step: FillStep,
    /// Provenance written (`write`), 9 (`unfilled`), else 255.
    pub code: u8,
    /// Interval id; −1 = none.
    pub interval: i32,
    /// Donor well (`real_donor`, `donor_well_shape`); else −1.
    pub donor_well_id: i32,
    /// Donor score (`real_donor`), `R²` (`cross_log_fit`); else NaN.
    pub score: f64,
    /// Rows counted by the row.
    pub n_rows: usize,
    /// Shallowest row, ft MD; NaN if no row.
    pub md_top_ft: f64,
    /// Deepest row, ft MD; NaN if no row.
    pub md_base_ft: f64,
    /// Range-map gain (`policy` rows); else NaN.
    pub range_gain: f64,
    /// Range-map offset in the log's working unit (`policy` rows); else NaN.
    pub range_offset: f64,
    /// `policy` and `reject` reason text; else empty.
    pub reason: String,
}

impl FillReportRow {
    /// Row over `rows` (local grid rows) with the "none" defaults: code 255,
    /// interval and donor −1, NaN score / gain / offset, empty reason.
    pub(crate) fn over_rows(
        kind: ReportKind,
        state: &WellFillState,
        well_name: &str,
        log: LogKind,
        step: FillStep,
        rows: &[usize],
    ) -> Self {
        let (md_top_ft, md_base_ft) = md_span(state.logs.grid, rows);
        Self {
            kind,
            well_id: state.logs.id,
            well_name: well_name.to_string(),
            log,
            step,
            code: CODE_NONE,
            interval: -1,
            donor_well_id: -1,
            score: f64::NAN,
            n_rows: rows.len(),
            md_top_ft,
            md_base_ft,
            range_gain: f64::NAN,
            range_offset: f64::NAN,
            reason: String::new(),
        }
    }
}

/// Sort in the order of: `well_id`, log in suite order, `kind`, `step`,
/// `md_top_ft` (NaN last). Stable. Complexity: `O(n log n)`.
pub fn sort_report(rows: &mut [FillReportRow]) {
    rows.sort_by(|a, b| {
        a.well_id
            .cmp(&b.well_id)
            .then(a.log.cmp(&b.log))
            .then(a.kind.cmp(&b.kind))
            .then(a.step.cmp(&b.step))
            .then(a.md_top_ft.total_cmp(&b.md_top_ft))
    });
}

/// `fill_summary.csv` text: one row per (well, log) with measured, filled and
/// unfilled sample counts, the filled share of the covered samples
/// (`filled / (measured + filled)`, empty when both are 0) and `mostly_predicted`
/// when that share is above `mostly_predicted_share` (each such row also logs a
/// warning). Errors: none.
pub fn fill_summary_csv(
    logs: &BTreeMap<WellId, WellLogs>,
    names: &BTreeMap<WellId, &str>,
    mostly_predicted_share: f64,
) -> String {
    let mut s = String::from(
        "well_id,well_name,log,n_measured,n_filled,n_unfilled,predicted_share,mostly_predicted\n",
    );
    for (w, l) in logs {
        let name = names.get(w).copied().unwrap_or("");
        for (log, prov) in &l.provenance {
            let n_measured = prov.iter().filter(|&&c| c == 0).count();
            let n_unfilled = prov.iter().filter(|&&c| c == 9).count();
            let n_filled = prov.iter().filter(|&&c| c != 0 && c != 9 && c != 255).count();
            let denom = n_measured + n_filled;
            let share = if denom == 0 { f64::NAN } else { n_filled as f64 / denom as f64 };
            let flag = share.is_finite() && share > mostly_predicted_share;
            if flag {
                tracing::warn!(
                    well = name,
                    log = log.name(),
                    predicted_share = share,
                    "log is mostly predicted, not measured"
                );
            }
            let share_txt = if share.is_finite() { format!("{share:.4}") } else { String::new() };
            let quoted = if name.contains([',', '"']) {
                format!("\"{}\"", name.replace('"', "\"\""))
            } else {
                name.to_string()
            };
            s.push_str(&format!(
                "{},{quoted},{},{n_measured},{n_filled},{n_unfilled},{share_txt},{flag}\n",
                w.0,
                log.name()
            ));
        }
    }
    s
}

/// CSV text with the header of. Floats: shortest round-trip text; NaN as empty field.
///
/// Rows are written in [`sort_report`] order (a sorted copy; `rows` is not changed).
/// Strings holding `,`, `"` or a line break are quoted. `\n` after every line.
/// Empty `rows`: header only. Errors: none.
pub fn fill_report_csv(rows: &[FillReportRow]) -> String {
    let mut sorted = rows.to_vec();
    sort_report(&mut sorted);
    let mut out = String::from(REPORT_HEADER);
    out.push('\n');
    for r in &sorted {
        let fields = [
            r.kind.name().to_string(),
            r.well_id.0.to_string(),
            csv_text(&r.well_name),
            r.log.name().to_string(),
            r.step.name().to_string(),
            r.code.to_string(),
            r.interval.to_string(),
            r.donor_well_id.to_string(),
            csv_f64(r.score),
            r.n_rows.to_string(),
            csv_f64(r.md_top_ft),
            csv_f64(r.md_base_ft),
            csv_f64(r.range_gain),
            csv_f64(r.range_offset),
            csv_text(&r.reason),
        ];
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    out
}

/// `logs_filled.parquet` schema. Errors: `Shape` if a well lacks a log.
///
/// Rows by `well_id` then `k`; columns `well_id` (Int32), `k` (Int32), `md_ft`
/// (Float64), the six values in suite order (Float32, RESD as `resd_log10ohmm`),
/// then the six `<log>_prov` codes (UInt8). Metadata `subsurfix_schema =
/// "subsurfix.logs_filled/1"`. NaN rule: stored as is. Empty map: zero rows.
/// Errors also: `Shape` from [`WellLogs::validate`]; `InvalidInput` if a key
/// differs from its well's id.
pub fn logs_filled_to_batch(logs: &BTreeMap<WellId, WellLogs>) -> SxResult<RecordBatch> {
    let mut total = 0usize;
    for (id, w) in logs {
        if w.id != *id {
            return Err(SxError::InvalidInput(format!(
                "logs_filled: key {} holds well {}",
                id.0, w.id.0
            )));
        }
        w.validate()?;
        for log in LogKind::ALL {
            if !w.curves.contains_key(&log) || !w.provenance.contains_key(&log) {
                return Err(SxError::Shape(format!(
                    "logs_filled: well {} has no {}",
                    id.0,
                    log.name()
                )));
            }
        }
        total += w.grid.len;
    }
    let mut ids = Vec::with_capacity(total);
    let mut ks = Vec::with_capacity(total);
    let mut mds = Vec::with_capacity(total);
    for w in logs.values() {
        for i in 0..w.grid.len {
            ids.push(w.id.0);
            ks.push(w.grid.k(i));
            mds.push(w.grid.md_ft(i));
        }
    }
    let mut fields = vec![
        Field::new("well_id", DataType::Int32, false),
        Field::new("k", DataType::Int32, false),
        Field::new("md_ft", DataType::Float64, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(ids)),
        Arc::new(Int32Array::from(ks)),
        Arc::new(Float64Array::from(mds)),
    ];
    for log in LogKind::ALL {
        let mut v: Vec<f32> = Vec::with_capacity(total);
        for w in logs.values() {
            v.extend_from_slice(w.curves.get(&log).map_or(&[][..], Vec::as_slice));
        }
        fields.push(Field::new(value_column(log), DataType::Float32, false));
        columns.push(Arc::new(Float32Array::from(v)));
    }
    for log in LogKind::ALL {
        let mut p: Vec<u8> = Vec::with_capacity(total);
        for w in logs.values() {
            p.extend_from_slice(w.provenance.get(&log).map_or(&[][..], Vec::as_slice));
        }
        let name = format!("{}_prov", log.name().to_ascii_lowercase());
        fields.push(Field::new(name, DataType::UInt8, false));
        columns.push(Arc::new(UInt8Array::from(p)));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| SxError::InvalidInput(format!("cannot build logs_filled batch: {e}")))?;
    with_schema_tag(batch, LOGS_FILLED_TAG)
}

/// Value column name of `log` in `logs_filled` (RESD holds log10 values).
fn value_column(log: LogKind) -> &'static str {
    match log {
        LogKind::Resd => "resd_log10ohmm",
        other => other.column(),
    }
}

fn csv_text(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// True if local row `i` lies inside `coverage`.
pub(crate) fn in_coverage(grid: GridRange, coverage: Option<Coverage>, i: usize) -> bool {
    coverage.is_some_and(|c| {
        i < grid.len && {
            let k = grid.k(i);
            c.k_first <= k && k <= c.k_last
        }
    })
}

/// Shallowest and deepest MD of local `rows`; NaN if empty.
pub(crate) fn md_span(grid: GridRange, rows: &[usize]) -> (f64, f64) {
    let lo = rows.iter().min().map_or(f64::NAN, |&i| grid.md_ft(i));
    let hi = rows.iter().max().map_or(f64::NAN, |&i| grid.md_ft(i));
    (lo, hi)
}

/// A gap row that is still open: `Unfilled` and NaN.
pub(crate) fn is_open(value: f32, code: u8) -> bool {
    code == Provenance::Unfilled.code() && value.is_nan()
}

/// True for a code written by a fill rule (`Model` and 40–54).
pub(crate) fn is_fill_code(code: u8) -> bool {
    Provenance::from_code(code).is_some_and(Provenance::is_filled)
}

/// Values and provenance of `log`. Errors: `Shape` if missing or of the wrong length.
pub(crate) fn curve(logs: &WellLogs, log: LogKind) -> SxResult<(&[f32], &[u8])> {
    let n = logs.grid.len;
    match (logs.curves.get(&log), logs.provenance.get(&log)) {
        (Some(v), Some(p)) if v.len() == n && p.len() == n => Ok((v, p)),
        _ => Err(missing_log(logs, log)),
    }
}

/// Mutable values and provenance of `log`. Errors as [`curve`].
pub(crate) fn curve_mut(logs: &mut WellLogs, log: LogKind) -> SxResult<(&mut [f32], &mut [u8])> {
    let n = logs.grid.len;
    let err = missing_log(logs, log);
    match (logs.curves.get_mut(&log), logs.provenance.get_mut(&log)) {
        (Some(v), Some(p)) if v.len() == n && p.len() == n => Ok((v, p)),
        _ => Err(err),
    }
}

fn missing_log(logs: &WellLogs, log: LogKind) -> SxError {
    SxError::Shape(format!(
        "fill: well {} has no {} curve of grid length {}",
        logs.id.0,
        log.name(),
        logs.grid.len
    ))
}

/// Population std in `f64`; 0 for fewer than one value.
pub(crate) fn pop_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
}

/// Count and population std of the measured (code 0, finite) values.
pub(crate) fn measured_stats(values: &[f32], prov: &[u8]) -> (usize, f64) {
    let m: Vec<f64> = values
        .iter()
        .zip(prov)
        .filter(|(v, &c)| c == Provenance::Measured.code() && v.is_finite())
        .map(|(v, _)| f64::from(*v))
        .collect();
    (m.len(), pop_std(&m))
}

/// Maximal runs of `true` in `mask`.
pub(crate) fn true_runs(mask: &[bool]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &m) in mask.iter().enumerate() {
        match (m, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..mask.len());
    }
    out
}

#[cfg(test)]
mod summary_tests {
    use super::*;

    #[test]
    fn fill_summary_flags_mostly_predicted() {
        let mut l = WellLogs::new(WellId(3), GridRange::new(0, 4).unwrap());
        l.provenance.insert(LogKind::Dtc, vec![1, 1, 1, 9]);
        l.provenance.insert(LogKind::Gr, vec![0, 0, 0, 1]);
        let logs = BTreeMap::from([(WellId(3), l)]);
        let names = BTreeMap::from([(WellId(3), "W, 3")]);
        let s = fill_summary_csv(&logs, &names, 0.5);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3, "{s}");
        assert!(lines.contains(&"3,\"W, 3\",DTC,0,3,1,1.0000,true"), "{s}");
        assert!(lines.contains(&"3,\"W, 3\",GR,3,1,0,0.2500,false"), "{s}");
    }
}
