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
