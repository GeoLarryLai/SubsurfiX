// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Curve ingest: aliases, load units, splice, merge, MD-grid resampling, and
//! the stage entry [`run_ingest`].
//!
//! Each well's LAS files become one `WellLogs` on the global MD grid (six
//! logs, `Provenance::Measured` where finite, `CODE_NONE` elsewhere) plus a
//! per-(well, log) ingest report. Depths are feet; curve values are `f32`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::{SxError, SxResult};
use crate::types::{LogKind, Provenance, WellAux, WellHeader, WellId, WellLogs, CODE_NONE};

mod aliases;
mod discover;
mod families;
mod lasread;
mod load;
mod merge;
mod report;
mod resample;
mod splice;
mod stage;

pub use aliases::{load_alias_dictionary_csv, AliasTable};
pub use discover::{
    discover_las, discover_las_matched, discover_las_report, well_name_from_path, LasDiscovery,
    LasMatch, LasMatchRow, MatchDecision, WellFiles, WellHint,
};
pub use families::{curve_family, resd_family_code};
pub use report::{
    ingest_report_batch, las_match_batch, IngestReportRow, INGEST_REPORT_COLUMNS, INGEST_REPORT_TAG,
    LAS_MATCH_COLUMNS,
};
pub use stage::{run_ingest, source_files, write_edges, IngestOutput};

use load::{Candidate, LoadResult, LoadedFile};
use merge::{merge_files, Merged};
use resample::{grid_range_for, resample_log};
use splice::{splice_log, Spliced};

/// Settings of one ingest run.
#[derive(Debug, Clone)]
pub struct IngestParams {
    /// LAS folders in priority order (already resolved to usable paths).
    pub las_dirs: Vec<PathBuf>,
    /// `None` = every header well; `Some(set)` = only these names.
    pub wells: Option<BTreeSet<String>>,
    /// Wells to skip.
    pub exclude_wells: BTreeSet<String>,
    /// Mnemonic priority lists.
    pub aliases: AliasTable,
    /// Widest source gap bridged by interpolation, feet. Default 2.0.
    pub max_gap_ft: f64,
    /// How LAS files are matched to header wells.
    pub las_match: LasMatch,
    /// Extra header facts per well key (name, location, depth) that settle
    /// LAS header/file-name conflicts; KB comes from the headers when unset.
    pub hints: BTreeMap<String, WellHint>,
    /// LAS file name -> well key; always wins.
    pub las_overrides: BTreeMap<String, String>,
}

/// Outcome class of one well.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WellStatus {
    /// Logs were built.
    Ok,
    /// The header well has no LAS file.
    NoLas,
    /// A LAS well name has no header.
    NoHeader,
    /// Listed in `exclude_wells`.
    Excluded,
    /// LAS files were read but no grid sample resulted.
    NoResolvedLogs,
    /// Every LAS file of the well was skipped (unreadable or no depth curve).
    ReadError,
}

impl WellStatus {
    /// `ok`, `no_las`, `no_header`, `excluded`, `no_resolved_logs`,
    /// `read_error`.
    pub fn as_str(self) -> &'static str {
        match self {
            WellStatus::Ok => "ok",
            WellStatus::NoLas => "no_las",
            WellStatus::NoHeader => "no_header",
            WellStatus::Excluded => "excluded",
            WellStatus::NoResolvedLogs => "no_resolved_logs",
            WellStatus::ReadError => "read_error",
        }
    }
}

/// Result of ingesting one well.
#[derive(Debug, Clone)]
pub struct WellIngest {
    /// Well id; `None` only for `NoHeader`.
    pub id: Option<WellId>,
    /// Well name.
    pub name: String,
    /// Outcome class.
    pub status: WellStatus,
    /// `Some` iff `status == Ok`.
    pub logs: Option<WellLogs>,
    /// Family of each log's primary mnemonic; "unknown" when the log is absent.
    pub families: BTreeMap<LogKind, &'static str>,
    /// LAS files of the well (in the winning folder).
    pub n_las_files: usize,
    /// Exactly 6 rows, suite order.
    pub report: Vec<IngestReportRow>,
    /// Caliper and density correction on the logs grid (`Some` iff `logs` is).
    pub aux: Option<WellAux>,
}

/// Result of ingesting a fleet of wells.
#[derive(Debug, Clone)]
pub struct FleetIngest {
    /// Header wells ascending by id, then `NoHeader` names by name.
    pub wells: Vec<WellIngest>,
    /// LAS match conflicts and overrides (`las_match_report.csv`).
    pub las_matches: Vec<LasMatchRow>,
}

impl FleetIngest {
    /// Logs of wells with status `Ok`, ascending `WellId`.
    pub fn logs(&self) -> Vec<&WellLogs> {
        let mut out: Vec<&WellLogs> = self
            .wells
            .iter()
            .filter(|w| w.status == WellStatus::Ok)
            .filter_map(|w| w.logs.as_ref())
            .collect();
        out.sort_by_key(|l| l.id);
        out
    }

    /// All report rows in output order (6 per well, wells in fleet order).
    pub fn report_rows(&self) -> Vec<&IngestReportRow> {
        self.wells.iter().flat_map(|w| w.report.iter()).collect()
    }
}

fn check_max_gap(max_gap_ft: f64) -> SxResult<()> {
    if max_gap_ft.is_finite() && max_gap_ft >= 0.0 {
        Ok(())
    } else {
        Err(SxError::Config(format!(
            "ingest.max_gap_ft must be finite and >= 0, got {max_gap_ft}"
        )))
    }
}

fn to_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// A well that was not read: six `absent` rows and `unknown` families.
fn unread_well(
    id: Option<WellId>,
    name: &str,
    status: WellStatus,
    n_files: usize,
    n_shadowed: usize,
) -> WellIngest {
    let report = LogKind::ALL
        .iter()
        .map(|log| {
            let mut row =
                IngestReportRow::blank(id.map(|i| i.0), name, status.as_str(), log.name());
            row.n_files = Some(to_i64(n_files));
            row.n_shadowed_files = Some(to_i64(n_shadowed));
            row
        })
        .collect();
    WellIngest {
        id,
        name: name.to_string(),
        status,
        logs: None,
        families: LogKind::ALL.iter().map(|l| (*l, "unknown")).collect(),
        n_las_files: n_files,
        report,
        aux: None,
    }
}

/// What the primary file said about one log.
struct PrimaryInfo {
    mnemonic: String,
    unit: String,
    unit_rule: &'static str,
    verify: &'static str,
    divisor: f64,
    nphi_offset: f64,
    n_candidates: usize,
    spliced: Spliced,
    rejected: Vec<String>,
    file_name: String,
    depth_unit: String,
}

/// Per-log result over all files of a well.
struct LogResult {
    merged: Option<Merged>,
    primary: Option<PrimaryInfo>,
    had_candidates: bool,
    /// Rejections of every file, in file order (used when there is no primary).
    rejected_all: Vec<String>,
}

fn rejected_entries(cands: &[Candidate], no_valid: &[usize]) -> Vec<String> {
    let mut rows: Vec<(usize, String)> = Vec::new();
    for (pos, c) in cands.iter().enumerate() {
        if let Err(reason) = &c.result {
            rows.push((pos, format!("{}:{reason}", c.mnemonic)));
        }
    }
    for &pos in no_valid {
        if let Some(c) = cands.get(pos) {
            rows.push((pos, format!("{}:no_valid", c.mnemonic)));
        }
    }
    rows.sort_by_key(|r| r.0);
    rows.into_iter().map(|r| r.1).collect()
}

fn resolve_log(log: LogKind, loaded: &[LoadedFile], max_gap_ft: f64) -> LogResult {
    let mut had_candidates = false;
    let mut spliced: Vec<Option<Spliced>> = Vec::with_capacity(loaded.len());
    let mut rejected: Vec<Vec<String>> = Vec::with_capacity(loaded.len());
    for f in loaded {
        match f.candidates.get(&log) {
            None => {
                spliced.push(None);
                rejected.push(Vec::new());
            }
            Some(cands) => {
                had_candidates = true;
                let outcome = splice_log(log, &f.z, cands, max_gap_ft);
                rejected.push(rejected_entries(cands, &outcome.no_valid));
                spliced.push(outcome.spliced);
            }
        }
    }
    let inputs: Vec<Option<(&[f64], &[f32])>> = loaded
        .iter()
        .zip(&spliced)
        .map(|(f, s)| s.as_ref().map(|s| (f.z.as_slice(), s.out.as_slice())))
        .collect();
    let merged = merge_files(&inputs);
    let rejected_all: Vec<String> = rejected.iter().flatten().cloned().collect();
    let primary = merged.as_ref().and_then(|m| {
        let fi = m.primary_file;
        let file = loaded.get(fi)?;
        let sp = spliced.get(fi)?.as_ref()?;
        let cands = file.candidates.get(&log)?;
        let cand = cands.get(sp.primary)?;
        let acc = cand.result.as_ref().ok()?;
        Some(PrimaryInfo {
            mnemonic: cand.mnemonic.clone(),
            unit: cand.unit.clone(),
            unit_rule: acc.unit_rule,
            verify: acc.verify,
            divisor: acc.divisor,
            nphi_offset: acc.nphi_offset,
            n_candidates: cands.len(),
            spliced: sp.clone(),
            rejected: rejected.get(fi).cloned().unwrap_or_default(),
            file_name: file.name.clone(),
            depth_unit: file.depth_unit.clone(),
        })
    });
    LogResult {
        merged,
        primary,
        had_candidates,
        rejected_all,
    }
}

/// Ingest one well from its files. Same rules as [`ingest_fleet`], for one
/// well; used by it.
///
/// Per-file read errors never fail the call (they become report notes).
///
/// # Errors
/// `SxError::Config` if `max_gap_ft` is not finite or `< 0`; `Shape` if the
/// built logs are inconsistent (not expected).
pub fn ingest_well(
    id: WellId,
    name: &str,
    files: &WellFiles,
    params: &IngestParams,
) -> SxResult<WellIngest> {
    check_max_gap(params.max_gap_ft)?;
    let n_files = files.paths.len();
    let mut loaded: Vec<LoadedFile> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for path in &files.paths {
        match load::load_file(path, &params.aliases)? {
            LoadResult::Loaded(f) => {
                notes.extend(f.notes.iter().cloned());
                loaded.push(*f);
            }
            LoadResult::Skipped(note) => notes.push(note),
        }
    }
    let n_nan_depth: usize = loaded.iter().map(|f| f.n_nan_depth).sum();
    let n_dup_depth: usize = loaded.iter().map(|f| f.n_dup_depth).sum();
    let n_desc_files = loaded.iter().filter(|f| f.descending).count();

    let results: Vec<(LogKind, LogResult)> = LogKind::ALL
        .iter()
        .map(|&log| (log, resolve_log(log, &loaded, params.max_gap_ft)))
        .collect();

    // The grid spans every merged sample of every log.
    let mut md_lo = f64::INFINITY;
    let mut md_hi = f64::NEG_INFINITY;
    for (_, r) in &results {
        if let Some(m) = &r.merged {
            if let (Some(&a), Some(&b)) = (m.md_ft.first(), m.md_ft.last()) {
                md_lo = md_lo.min(a);
                md_hi = md_hi.max(b);
            }
        }
    }
    let grid = grid_range_for(md_lo, md_hi);
    let status = if loaded.is_empty() {
        WellStatus::ReadError
    } else if grid.is_none() {
        WellStatus::NoResolvedLogs
    } else {
        WellStatus::Ok
    };

    let mut well_logs = grid.map(|g| WellLogs::new(id, g));
    let mut families: BTreeMap<LogKind, &'static str> = BTreeMap::new();
    let mut report: Vec<IngestReportRow> = Vec::with_capacity(LogKind::ALL.len());
    for (log, r) in &results {
        let log = *log;
        let mut row = IngestReportRow::blank(Some(id.0), name, status.as_str(), log.name());
        row.n_files = Some(to_i64(n_files));
        row.n_shadowed_files = Some(to_i64(files.n_shadowed));
        row.n_nan_depth = Some(to_i64(n_nan_depth));
        row.n_dup_depth = Some(to_i64(n_dup_depth));
        row.n_desc_files = Some(to_i64(n_desc_files));
        row.note = notes.clone();

        // Resample onto the grid.
        let curve: Option<Vec<f32>> = grid.map(|g| match &r.merged {
            Some(m) => resample_log(g, &m.md_ft, &m.values, params.max_gap_ft),
            None => vec![f32::NAN; g.len],
        });
        let first = curve
            .as_ref()
            .and_then(|c| c.iter().position(|v| v.is_finite()));
        let last = curve
            .as_ref()
            .and_then(|c| c.iter().rposition(|v| v.is_finite()));
        let n_grid_finite = curve
            .as_ref()
            .map_or(0, |c| c.iter().filter(|v| v.is_finite()).count());

        row.log_status = if n_grid_finite > 0 {
            "resolved"
        } else if r.merged.is_some() {
            "off_grid"
        } else if r.had_candidates {
            "rejected"
        } else {
            "absent"
        }
        .to_string();

        match (&r.primary, &r.merged) {
            (Some(p), Some(m)) => {
                row.primary_mnemonic = p.mnemonic.clone();
                row.primary_unit = p.unit.clone();
                let fam = curve_family(log, &p.mnemonic);
                row.family = fam.to_string();
                families.insert(log, fam);
                row.unit_rule = p.unit_rule.to_string();
                row.verify = p.verify.to_string();
                row.unit_divisor = p.divisor;
                if log == LogKind::Nphi {
                    row.nphi_offset_frac = p.nphi_offset;
                }
                row.n_candidates = Some(to_i64(p.n_candidates));
                row.merged_from = p.spliced.merged_from.clone();
                row.shifts_ft = p.spliced.shifts_ft.clone();
                row.skipped = p.spliced.skipped.clone();
                row.rejected = p.rejected.clone();
                row.coverage_before_frac = p.spliced.coverage_before_frac;
                row.coverage_after_frac = p.spliced.coverage_after_frac;
                row.primary_file = p.file_name.clone();
                row.n_files_used = Some(to_i64(m.n_files_used));
                row.depth_unit = p.depth_unit.clone();
                row.n_grid_finite = Some(to_i64(n_grid_finite));
                if let (Some(g), Some(a), Some(b)) = (grid, first, last) {
                    row.md_top_ft = g.md_ft(a);
                    row.md_base_ft = g.md_ft(b);
                }
            }
            _ => {
                families.insert(log, "unknown");
                if row.log_status == "rejected" {
                    row.rejected = r.rejected_all.clone();
                }
            }
        }
        report.push(row);

        if let (Some(wl), Some(c)) = (well_logs.as_mut(), curve) {
            let prov: Vec<u8> = c
                .iter()
                .map(|v| {
                    if v.is_finite() {
                        Provenance::Measured.code()
                    } else {
                        CODE_NONE
                    }
                })
                .collect();
            wl.curves.insert(log, c);
            wl.provenance.insert(log, prov);
        }
    }
    // Borehole-condition curves for QC: the first file that has each one.
    let aux = match (status, grid) {
        (WellStatus::Ok, Some(g)) => {
            let pick = |which: usize| -> Vec<f32> {
                loaded
                    .iter()
                    .find_map(|f| {
                        f.aux[which]
                            .as_ref()
                            .map(|v| resample_log(g, &f.z, v, params.max_gap_ft))
                    })
                    .unwrap_or_else(|| vec![f32::NAN; g.len])
            };
            Some(WellAux {
                id,
                grid: g,
                cal_in: pick(0),
                drho: pick(1),
                bs_in: pick(2),
                lspd_ftmin: pick(3),
                temp_c: pick(4),
            })
        }
        _ => None,
    };
    let logs = if status == WellStatus::Ok {
        if let Some(wl) = &well_logs {
            wl.validate()?;
        }
        well_logs
    } else {
        None
    };
    tracing::debug!(
        well = name,
        status = status.as_str(),
        n_files,
        "ingested well"
    );
    Ok(WellIngest {
        id: Some(id),
        name: name.to_string(),
        status,
        logs,
        families,
        n_las_files: n_files,
        report,
        aux,
    })
}

/// One unit of fleet work.
enum Plan<'a> {
    Run(&'a WellHeader, &'a WellFiles),
    Unread(Option<WellId>, String, WellStatus, usize, usize),
}

/// Ingest every selected header well (and report LAS names with no header).
///
/// Order: selected header wells ascending by id, then names that have LAS
/// files (or were asked for in `wells`) but no header, by name. Wells run in
/// parallel on the global rayon pool; results keep that order.
///
/// # Errors
/// `Config` if `max_gap_ft` is not finite or `< 0`; `NotFound`/`Io` from
/// discovery. Per-file read errors never fail the call (they become report
/// notes).
pub fn ingest_fleet(headers: &[WellHeader], params: &IngestParams) -> SxResult<FleetIngest> {
    check_max_gap(params.max_gap_ft)?;
    let known: BTreeSet<String> = headers.iter().map(|h| h.name.clone()).collect();
    let mut hints = params.hints.clone();
    for h in headers {
        if h.kb_elev_ft.is_finite() {
            let e = hints.entry(h.name.clone()).or_default();
            if e.kb_ft.is_none() {
                e.kb_ft = Some(h.kb_elev_ft);
            }
        }
    }
    let LasDiscovery {
        wells: discovered,
        matches: las_matches,
    } = discover_las_report(
        &params.las_dirs,
        &params.exclude_wells,
        &known,
        &hints,
        &params.las_overrides,
        params.las_match,
    )?;
    let mut selected: Vec<&WellHeader> = headers
        .iter()
        .filter(|h| params.wells.as_ref().is_none_or(|w| w.contains(&h.name)))
        .collect();
    selected.sort_by_key(|h| h.id);

    let mut plans: Vec<Plan<'_>> = Vec::new();
    for h in &selected {
        if params.exclude_wells.contains(&h.name) {
            plans.push(Plan::Unread(
                Some(h.id),
                h.name.clone(),
                WellStatus::Excluded,
                0,
                0,
            ));
            continue;
        }
        match discovered.get(&h.name) {
            Some(files) => plans.push(Plan::Run(h, files)),
            None => plans.push(Plan::Unread(
                Some(h.id),
                h.name.clone(),
                WellStatus::NoLas,
                0,
                0,
            )),
        }
    }
    let header_names: BTreeSet<&str> = headers.iter().map(|h| h.name.as_str()).collect();
    let mut orphan_names: BTreeSet<&str> = BTreeSet::new();
    for name in discovered.keys() {
        let wanted = params.wells.as_ref().is_none_or(|w| w.contains(name));
        if wanted && !header_names.contains(name.as_str()) {
            orphan_names.insert(name.as_str());
        }
    }
    if let Some(w) = &params.wells {
        for name in w {
            if !header_names.contains(name.as_str()) && !params.exclude_wells.contains(name) {
                orphan_names.insert(name.as_str());
            }
        }
    }
    for name in orphan_names {
        let (n_files, n_shadowed) = discovered
            .get(name)
            .map_or((0, 0), |f| (f.paths.len(), f.n_shadowed));
        plans.push(Plan::Unread(
            None,
            name.to_string(),
            WellStatus::NoHeader,
            n_files,
            n_shadowed,
        ));
    }

    let results: Vec<SxResult<WellIngest>> = plans
        .par_iter()
        .map(|plan| match plan {
            Plan::Run(h, files) => ingest_well(h.id, &h.name, files, params).map(|mut w| {
                if let Some(row) = w.report.first_mut() {
                    row.note.extend(files.notes.iter().cloned());
                }
                w
            }),
            Plan::Unread(id, name, status, n_files, n_shadowed) => {
                Ok(unread_well(*id, name, *status, *n_files, *n_shadowed))
            }
        })
        .collect();
    let mut wells = Vec::with_capacity(results.len());
    for r in results {
        wells.push(r?);
    }
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for w in &wells {
        *counts.entry(w.status.as_str()).or_default() += 1;
    }
    tracing::info!(?counts, n_wells = wells.len(), "ingest finished");
    Ok(FleetIngest { wells, las_matches })
}

/// Helpers shared by the tests of this module and its children.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};

    /// LAS 2.0 text with one depth curve and the given `(mnemonic, unit,
    /// values)` curves. NaN is written as `-999.25`.
    pub(crate) fn las_text(
        depth_unit: &str,
        depth_mnemonic: &str,
        depth: &[f64],
        curves: &[(&str, &str, &[f64])],
    ) -> String {
        let num = |v: f64| {
            if v.is_nan() {
                "-999.25".to_string()
            } else {
                format!("{v}")
            }
        };
        let first = depth.first().copied().unwrap_or(0.0);
        let last = depth.last().copied().unwrap_or(0.0);
        let mut s = String::new();
        s.push_str("~V\n VERS. 2.0 :\n WRAP. NO :\n~W\n");
        s.push_str(&format!(" STRT.{depth_unit} {} :\n", num(first)));
        s.push_str(&format!(" STOP.{depth_unit} {} :\n", num(last)));
        s.push_str(&format!(" STEP.{depth_unit} 0 :\n"));
        s.push_str(" NULL. -999.25 :\n~C\n");
        s.push_str(&format!(" {depth_mnemonic}.{depth_unit} : Depth\n"));
        for (m, u, _) in curves {
            s.push_str(&format!(" {m}.{u} : curve\n"));
        }
        s.push_str("~A\n");
        for (i, d) in depth.iter().enumerate() {
            s.push_str(&num(*d));
            for (_, _, v) in curves {
                s.push(' ');
                s.push_str(&num(v.get(i).copied().unwrap_or(f64::NAN)));
            }
            s.push('\n');
        }
        s
    }

    /// Write `text` to `dir/name` and return the path.
    pub(crate) fn write_las(dir: &Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{las_text, write_las};
    use super::*;
    use crate::io::csv::write_batch;

    fn params(dirs: Vec<PathBuf>) -> IngestParams {
        IngestParams {
            las_dirs: dirs,
            wells: None,
            exclude_wells: BTreeSet::new(),
            aliases: AliasTable::builtin(),
            max_gap_ft: 2.0,
            las_match: LasMatch::default(),
            hints: Default::default(),
            las_overrides: Default::default(),
        }
    }

    fn header(id: i32, name: &str) -> WellHeader {
        WellHeader {
            id: WellId(id),
            name: name.to_string(),
            x_m: 0.0,
            y_m: 0.0,
            kb_elev_ft: 3000.0,
        }
    }

    /// The example E file: meters and kg/m3.
    fn example_e(dir: &std::path::Path, name: &str) {
        let text = las_text(
            "M",
            "DEPT",
            &[304.8, 305.1048],
            &[("RHOB", "KG/M3", &[2400.0, 2500.0][..])],
        );
        write_las(dir, name, &text);
    }

    #[test]
    fn example_e_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        example_e(dir.path(), "W1.las");
        let got =
            ingest_fleet(&[header(1, "W1")], &params(vec![dir.path().to_path_buf()])).unwrap();
        assert_eq!(got.wells.len(), 1);
        let w = &got.wells[0];
        assert_eq!(w.status, WellStatus::Ok);
        let logs = w.logs.as_ref().unwrap();
        assert_eq!(
            logs.grid,
            crate::types::GridRange {
                k_start: 2000,
                len: 3
            }
        );
        let rhob = &logs.curves[&LogKind::Rhob];
        for (v, want) in rhob.iter().zip([2.4f32, 2.45, 2.5]) {
            assert!((v - want).abs() <= 1e-5 * want, "{v} vs {want}");
        }
        assert_eq!(logs.provenance[&LogKind::Rhob], [0, 0, 0]);
        for log in LogKind::ALL {
            if log != LogKind::Rhob {
                assert!(logs.curves[&log].iter().all(|v| v.is_nan()), "{log:?}");
                assert_eq!(logs.provenance[&log], [CODE_NONE; 3]);
            }
        }
        logs.validate().unwrap();
        assert_eq!(w.report.len(), 6);
        let row = &w.report[1];
        assert_eq!(row.log, "RHOB");
        assert_eq!(row.log_status, "resolved");
        assert_eq!(row.primary_mnemonic, "RHOB");
        assert_eq!(row.primary_unit, "KG/M3");
        assert_eq!(row.unit_rule, "header");
        assert_eq!(row.unit_divisor, 1000.0);
        assert_eq!(row.verify, "skipped");
        assert_eq!(row.depth_unit, "m");
        assert_eq!(row.primary_file, "W1.las");
        assert_eq!(row.n_grid_finite, Some(3));
        assert_eq!(row.md_top_ft, 1000.0);
        assert_eq!(row.md_base_ft, 1001.0);
        assert_eq!(row.n_files, Some(1));
        assert_eq!(row.n_files_used, Some(1));
        assert_eq!(row.n_candidates, Some(1));
        assert_eq!(w.families[&LogKind::Rhob], "unknown");
        assert!(w.report[0].log_status == "absent" && w.report[0].n_files == Some(1));
        assert!(w.report[0].primary_mnemonic.is_empty());
        assert!(w.report[0].unit_divisor.is_nan());
        assert!(w.report[2].nphi_offset_frac.is_nan());
        assert_eq!(got.logs().len(), 1);
        assert_eq!(got.report_rows().len(), 6);
    }

    #[test]
    fn example_d_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let z1: Vec<f64> = (0..201).map(|i| 1000.0 + 0.5 * f64::from(i)).collect();
        let z2: Vec<f64> = (0..501).map(|i| 1050.0 + 0.5 * f64::from(i)).collect();
        let g1 = vec![50.0; 201];
        let g2 = vec![70.0; 501];
        write_las(
            dir.path(),
            "W1_imputed.las",
            &las_text("FT", "DEPT", &z1, &[("GR", "GAPI", &g1[..])]),
        );
        write_las(
            dir.path(),
            "W1_processed.las",
            &las_text("FT", "DEPT", &z2, &[("GR", "GAPI", &g2[..])]),
        );
        let got =
            ingest_fleet(&[header(1, "W1")], &params(vec![dir.path().to_path_buf()])).unwrap();
        let w = &got.wells[0];
        assert_eq!(w.status, WellStatus::Ok);
        let logs = w.logs.as_ref().unwrap();
        assert_eq!(
            logs.grid,
            crate::types::GridRange {
                k_start: 2000,
                len: 601
            }
        );
        let gr = &logs.curves[&LogKind::Gr];
        for (i, v) in gr.iter().enumerate() {
            let md = logs.grid.md_ft(i);
            assert_eq!(*v, if md < 1050.0 { 50.0 } else { 70.0 }, "{md}");
        }
        let row = &w.report[0];
        assert_eq!(row.n_files_used, Some(2));
        assert_eq!(row.n_files, Some(2));
        assert_eq!(row.primary_file, "W1_processed.las");
        assert_eq!(row.log_status, "resolved");
        assert_eq!(row.n_grid_finite, Some(601));
    }

    #[test]
    fn statuses_order_and_report() {
        let d0 = tempfile::tempdir().unwrap();
        let d1 = tempfile::tempdir().unwrap();
        example_e(d0.path(), "W1.las");
        example_e(d0.path(), "ORPHAN_b.las");
        example_e(d0.path(), "W5.las");
        example_e(d1.path(), "W1.las");
        // W4: only a file without a depth curve.
        write_las(
            d0.path(),
            "W4.las",
            &las_text(
                "FT",
                "XDEP",
                &[1.0, 2.0],
                &[("GR", "GAPI", &[1.0, 2.0][..])],
            ),
        );
        // W6: samples that fall between grid points only.
        write_las(
            d0.path(),
            "W6.las",
            &las_text(
                "FT",
                "DEPT",
                &[1000.1, 1000.4],
                &[("GR", "GAPI", &[1.0, 2.0][..])],
            ),
        );
        let headers = vec![
            header(6, "W6"),
            header(4, "W4"),
            header(3, "W3"),
            header(2, "W2"),
            header(1, "W1"),
        ];
        let mut p = params(vec![d0.path().to_path_buf(), d1.path().to_path_buf()]);
        p.exclude_wells.insert("W2".to_string());
        let got = ingest_fleet(&headers, &p).unwrap();
        let summary: Vec<(Option<i32>, &str, WellStatus)> = got
            .wells
            .iter()
            .map(|w| (w.id.map(|i| i.0), w.name.as_str(), w.status))
            .collect();
        assert_eq!(
            summary,
            [
                (Some(1), "W1", WellStatus::Ok),
                (Some(2), "W2", WellStatus::Excluded),
                (Some(3), "W3", WellStatus::NoLas),
                (Some(4), "W4", WellStatus::ReadError),
                (Some(6), "W6", WellStatus::NoResolvedLogs),
                (None, "ORPHAN_b", WellStatus::NoHeader),
                (None, "W5", WellStatus::NoHeader),
            ]
        );
        for w in &got.wells {
            assert_eq!(w.report.len(), 6);
            assert_eq!(w.families.len(), 6);
            assert_eq!(w.logs.is_some(), w.status == WellStatus::Ok);
        }
        // Shadowed file counted on W1.
        assert_eq!(got.wells[0].report[0].n_shadowed_files, Some(1));
        // The no-depth-curve note.
        assert_eq!(got.wells[3].report[0].note, ["no_depth_curve:W4.las"]);
        // W6: samples but no grid point -> grid is None -> off_grid on GR.
        let w6 = &got.wells[4];
        assert_eq!(w6.report[0].log_status, "off_grid");
        assert_eq!(w6.report[0].n_grid_finite, Some(0));
        // Names in `wells` with neither header nor LAS are NoHeader too.
        let mut p2 = params(vec![d0.path().to_path_buf()]);
        p2.wells = Some(["W1", "GHOST"].iter().map(|s| s.to_string()).collect());
        let got2 = ingest_fleet(&headers, &p2).unwrap();
        let names: Vec<&str> = got2.wells.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["W1", "GHOST"]);
        assert_eq!(got2.wells[1].status, WellStatus::NoHeader);
        assert_eq!(got2.wells[1].report[0].n_files, Some(0));

        // The report writes as CSV.
        let rows: Vec<IngestReportRow> = got.report_rows().into_iter().cloned().collect();
        assert_eq!(rows.len(), 42);
        let batch = ingest_report_batch(&rows).unwrap();
        let out = tempfile::tempdir().unwrap();
        let path = out.path().join("ingest_report.csv");
        write_batch(&path, &batch).unwrap();
        let (cols, data) = crate::io::csv::read_csv(&path).unwrap();
        assert_eq!(cols, INGEST_REPORT_COLUMNS);
        assert_eq!(data.len(), 42);
        // No file paths in the report.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains(d0.path().to_str().unwrap()));
    }

    #[test]
    fn rejected_log_lists_reasons() {
        let dir = tempfile::tempdir().unwrap();
        // 2450 g/cc with blank unit and 10 samples: accepted by the unit flow
        // (skipped verify), rejected by the splice.
        let z: Vec<f64> = (0..10).map(|i| 1000.0 + f64::from(i)).collect();
        let v = [2450.0; 10];
        let gr = [80.0; 10];
        write_las(
            dir.path(),
            "W1.las",
            &las_text(
                "FT",
                "DEPT",
                &z,
                &[("RHOB", "", &v[..]), ("GR", "GAPI", &gr[..])],
            ),
        );
        let got =
            ingest_fleet(&[header(1, "W1")], &params(vec![dir.path().to_path_buf()])).unwrap();
        let w = &got.wells[0];
        assert_eq!(w.status, WellStatus::Ok);
        let rhob = &w.report[1];
        assert_eq!(rhob.log_status, "rejected");
        assert_eq!(rhob.rejected, ["RHOB:no_valid"]);
        assert!(rhob.primary_mnemonic.is_empty());
        assert!(w.logs.as_ref().unwrap().curves[&LogKind::Rhob]
            .iter()
            .all(|v| v.is_nan()));
        assert_eq!(w.report[0].log_status, "resolved");
    }

    #[test]
    fn bad_max_gap_is_config_error() {
        let dir = tempfile::tempdir().unwrap();
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            let mut p = params(vec![dir.path().to_path_buf()]);
            p.max_gap_ft = bad;
            assert!(matches!(
                ingest_fleet(&[header(1, "W1")], &p),
                Err(SxError::Config(_))
            ));
        }
        let mut p = params(vec![dir.path().to_path_buf()]);
        p.max_gap_ft = 0.0;
        assert!(ingest_fleet(&[header(1, "W1")], &p).is_ok());
    }

    #[test]
    fn missing_las_folder_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let p = params(vec![dir.path().join("missing")]);
        assert!(matches!(
            ingest_fleet(&[header(1, "W1")], &p),
            Err(SxError::NotFound(_))
        ));
    }

    #[test]
    fn well_status_text() {
        let all = [
            (WellStatus::Ok, "ok"),
            (WellStatus::NoLas, "no_las"),
            (WellStatus::NoHeader, "no_header"),
            (WellStatus::Excluded, "excluded"),
            (WellStatus::NoResolvedLogs, "no_resolved_logs"),
            (WellStatus::ReadError, "read_error"),
        ];
        for (s, t) in all {
            assert_eq!(s.as_str(), t);
        }
    }
}
