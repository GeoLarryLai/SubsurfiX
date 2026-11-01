// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! LAS discovery: folder walk, well names, cross-folder rule.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::error::{SxError, SxResult};
use crate::io::atomic::io_error;
use crate::pipeline::keys::sha256_file;

/// Suffixes removed from a file stem, first match wins (exact case).
const NAME_SUFFIXES: [&str; 6] = [
    "_imputed",
    "_IMPUTED",
    "_processed",
    "_PROCESSED",
    "_normalized",
    "_NORMALIZED",
];

/// Files of one well in its winning folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WellFiles {
    /// Index into `las_dirs` of the winning folder.
    pub folder_index: usize,
    /// LAS files of this well in the winning folder, sorted by file name.
    pub paths: Vec<PathBuf>,
    /// LAS files of this well found in later folders (ignored).
    pub n_shadowed: usize,
    /// Matching notes: header-ID versus file-name conflicts, duplicate files.
    pub notes: Vec<String>,
}

/// Well name from a LAS file path.
///
/// The stem is the file name without its last extension. A stem that starts
/// with 14 ASCII digits and is exactly that long, or continues with `_`, `-`
/// or `.`, gives those 14 digits. Otherwise one trailing suffix (`_imputed`,
/// `_processed`, `_normalized`, either exact case) is removed and whitespace
/// and `_` are trimmed from both ends.
pub fn well_name_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let bytes = stem.as_bytes();
    if bytes.len() >= 14 && bytes[..14].iter().all(u8::is_ascii_digit) {
        let ok = match bytes.get(14) {
            None => true,
            Some(c) => matches!(c, b'_' | b'-' | b'.'),
        };
        if ok {
            return stem.chars().take(14).collect();
        }
    }
    let mut s = stem.as_str();
    for suffix in NAME_SUFFIXES {
        if let Some(rest) = s.strip_suffix(suffix) {
            s = rest;
            break;
        }
    }
    s.trim_matches(|c: char| c.is_whitespace() || c == '_')
        .to_string()
}

/// Direct entries of `dir` that are regular files with extension `las` (any
/// case), sorted by file name in byte order.
///
/// # Errors
/// `SxError::NotFound` when `dir` is not a folder; `SxError::Io` on read
/// failure.
pub(crate) fn list_las_files(dir: &Path) -> SxResult<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(SxError::NotFound(format!(
            "LAS folder {} does not exist",
            dir.display()
        )));
    }
    let entries = std::fs::read_dir(dir).map_err(|e| io_error(dir, e))?;
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| io_error(dir, e))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_las = path
            .extension()
            .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("las"));
        if !is_las {
            continue;
        }
        files.push((entry.file_name().to_string_lossy().into_owned(), path));
    }
    files.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    Ok(files.into_iter().map(|(_, p)| p).collect())
}

/// Walk `las_dirs` in order. Names in `exclude` are skipped.
///
/// The first folder holding a name wins; files in later folders are counted in
/// `n_shadowed` and never merged in.
///
/// # Errors
/// `SxError::NotFound` if a folder does not exist; `SxError::Io` on read
/// failure.
pub fn discover_las(
    las_dirs: &[PathBuf],
    exclude: &BTreeSet<String>,
) -> SxResult<BTreeMap<String, WellFiles>> {
    let mut out: BTreeMap<String, WellFiles> = BTreeMap::new();
    for (i, dir) in las_dirs.iter().enumerate() {
        for path in list_las_files(dir)? {
            let name = well_name_from_path(&path);
            if exclude.contains(&name) {
                continue;
            }
            match out.get_mut(&name) {
                None => {
                    out.insert(
                        name,
                        WellFiles {
                            folder_index: i,
                            paths: vec![path],
                            n_shadowed: 0,
                            notes: Vec::new(),
                        },
                    );
                }
                Some(w) if w.folder_index == i => w.paths.push(path),
                Some(w) => w.n_shadowed += 1,
            }
        }
    }
    Ok(out)
}

/// How a LAS file is matched to a header well.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LasMatch {
    /// Well ID in the LAS header (UWI, API, then WELL) first, then the file name.
    #[default]
    HeaderFirst,
    /// File name only.
    FileName,
}

/// Well identifiers from the `~W` section of a LAS file: `(mnemonic, value)` for
/// UWI, API and WELL, in that order, with empty values skipped. Each gives its
/// value and then its description, since LAS 1.2 files often swap the two.
pub(crate) fn las_header_ids(path: &Path) -> Vec<(String, String)> {
    use std::io::Read;
    let mut buf = Vec::new();
    let Ok(f) = std::fs::File::open(path) else {
        return Vec::new();
    };
    if f.take(65_536).read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let mut in_w = false;
    let mut found: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('~') {
            if in_w {
                break;
            }
            in_w = rest.starts_with(['W', 'w']);
            continue;
        }
        if !in_w || t.starts_with('#') {
            continue;
        }
        let Some((mnem, rest)) = t.split_once('.') else {
            continue;
        };
        let key = match mnem.trim().to_ascii_uppercase().as_str() {
            "UWI" => "UWI",
            "API" => "API",
            "WELL" => "WELL",
            _ => continue,
        };
        // After the dot: unit (no space), then value up to the last colon.
        let after_unit = rest.split_once(char::is_whitespace).map_or("", |(_, v)| v);
        let (value, descr) = after_unit.rsplit_once(':').unwrap_or((after_unit, ""));
        // LAS 1.2 files often swap the two ("UWI . UNIQUE WELL ID: 0402591500"),
        // so both sides are kept as candidates, value first.
        let slot = found.entry(key).or_default();
        if slot.is_empty() {
            slot.extend(
                [value.trim(), descr.trim()]
                    .into_iter()
                    .filter(|v| !v.is_empty())
                    .map(str::to_string),
            );
        }
    }
    ["UWI", "API", "WELL"]
        .iter()
        .flat_map(|k| {
            found
                .get(k)
                .into_iter()
                .flatten()
                .map(|v| ((*k).to_string(), v.clone()))
        })
        .collect()
}

/// Kelly-bushing (or other log reference) elevation in feet from the `~W` or
/// `~P` section (`EKB`, `KB`, `EREF`, `EDF`, first found, in that order of
/// preference), or `None`. A unit of `M` converts from metres.
pub(crate) fn las_kb_ft(path: &Path) -> Option<f64> {
    use std::io::Read;
    let mut buf = Vec::new();
    let f = std::fs::File::open(path).ok()?;
    f.take(65_536).read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf);
    const KEYS: [&str; 4] = ["EKB", "KB", "EREF", "EDF"];
    let mut best: Option<(usize, f64)> = None;
    let mut in_wp = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('~') {
            if rest.starts_with(['A', 'a']) {
                break;
            }
            in_wp = rest.starts_with(['W', 'w', 'P', 'p']);
            continue;
        }
        if !in_wp || t.starts_with('#') {
            continue;
        }
        let Some((mnem, rest)) = t.split_once('.') else {
            continue;
        };
        let Some(rank) = KEYS.iter().position(|k| mnem.trim().eq_ignore_ascii_case(k)) else {
            continue;
        };
        let (unit, after) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let value = after.rsplit_once(':').map_or(after, |(v, _)| v).trim();
        let Ok(v) = value.parse::<f64>() else {
            continue;
        };
        let v = if unit.eq_ignore_ascii_case("m") { v / 0.3048 } else { v };
        if v.is_finite() && best.is_none_or(|(r, _)| rank < r) {
            best = Some((rank, v));
        }
    }
    best.map(|(_, v)| v)
}

/// 14-digit API form of an identifier with at least 10 digits (API-10 and
/// API-12 are padded with zeros), or `None`.
fn api14(s: &str) -> Option<String> {
    let d: String = s.chars().filter(char::is_ascii_digit).collect();
    match d.len() {
        10 => Some(format!("{d}0000")),
        12 => Some(format!("{d}00")),
        n if n >= 14 => Some(d.chars().take(14).collect()),
        _ => None,
    }
}

/// Header well key that a LAS header names, if exactly one matches.
fn match_header_ids(ids: &[(String, String)], known: &BTreeSet<String>) -> Option<String> {
    let by_api: BTreeMap<String, Vec<&String>> = known.iter().fold(BTreeMap::new(), |mut m, k| {
        if let Some(a) = api14(k) {
            m.entry(a).or_insert_with(Vec::new).push(k);
        }
        m
    });
    for (mnem, v) in ids {
        if let Some(k) = known.iter().find(|k| k.eq_ignore_ascii_case(v.trim())) {
            return Some(k.clone());
        }
        if mnem != "WELL" {
            if let Some(a) = api14(v) {
                if let Some(ks) = by_api.get(&a) {
                    if ks.len() == 1 {
                        return ks.first().map(|k| (*k).clone());
                    }
                }
            }
        }
    }
    None
}

/// What the well list says about one well; used only to settle a LAS file
/// whose header ID and file name name different wells. `None` = unknown.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WellHint {
    /// Kelly-bushing elevation, feet.
    pub kb_ft: Option<f64>,
    /// Well label (name, or lease and number).
    pub label: Option<String>,
    /// Surface latitude, degrees.
    pub lat_deg: Option<f64>,
    /// Surface longitude, degrees.
    pub lon_deg: Option<f64>,
    /// Legal location text, for example `"T34S R11W, Sec. 1"`.
    pub location: Option<String>,
    /// Total depth (or deepest log depth), feet.
    pub td_ft: Option<f64>,
}

/// How a LAS file in the match report was assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchDecision {
    /// `[ingest.las_overrides]` named the well.
    Override,
    /// The file's KB elevation matched one candidate only.
    Kb,
    /// Name, location or depth fields matched one candidate clearly better.
    Hints,
    /// Still ambiguous: the file was left out.
    Excluded,
}

impl MatchDecision {
    /// Text used in the report.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::Kb => "kb_elevation",
            Self::Hints => "header_fields",
            Self::Excluded => "excluded",
        }
    }
}

/// One row of the LAS match report (`las_match_report.csv`): a header/file-name
/// conflict or a manual override.
#[derive(Debug, Clone, PartialEq)]
pub struct LasMatchRow {
    /// File name.
    pub file: String,
    /// Full path.
    pub path: String,
    /// Well named by the LAS header ID (empty if none).
    pub header_well: String,
    /// Well named by the file name.
    pub file_name_well: String,
    /// Well the file went to; empty when excluded.
    pub used: String,
    /// How it was decided.
    pub decision: MatchDecision,
    /// Evidence, human readable.
    pub reason: String,
}

/// Output of [`discover_las_report`].
#[derive(Debug, Clone, Default)]
pub struct LasDiscovery {
    /// Files per well, as [`discover_las`].
    pub wells: BTreeMap<String, WellFiles>,
    /// Conflicts and overrides, in discovery order.
    pub matches: Vec<LasMatchRow>,
}

/// `(unit, value)` of the `~W` and `~P` lines of a LAS file, keyed by upper-case
/// mnemonic (first occurrence). The value falls back to the description when
/// empty (LAS 1.2 files often swap them).
fn las_fields(path: &Path) -> BTreeMap<String, (String, String)> {
    use std::io::Read;
    let mut out = BTreeMap::new();
    let mut buf = Vec::new();
    let Ok(f) = std::fs::File::open(path) else {
        return out;
    };
    if f.take(65_536).read_to_end(&mut buf).is_err() {
        return out;
    }
    let text = String::from_utf8_lossy(&buf);
    let mut in_wp = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('~') {
            if rest.starts_with(['A', 'a']) {
                break;
            }
            in_wp = rest.starts_with(['W', 'w', 'P', 'p']);
            continue;
        }
        if !in_wp || t.starts_with('#') {
            continue;
        }
        let Some((mnem, rest)) = t.split_once('.') else {
            continue;
        };
        let (unit, after) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let (value, descr) = after.rsplit_once(':').unwrap_or((after, ""));
        let v = if value.trim().is_empty() { descr.trim() } else { value.trim() };
        out.entry(mnem.trim().to_ascii_uppercase())
            .or_insert_with(|| (unit.trim().to_string(), v.to_string()));
    }
    out
}

fn norm_name(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_uppercase());
        } else if !out.ends_with(' ') && !out.is_empty() && c != '\'' && c != '"' {
            out.push(' ');
        }
    }
    out.trim().to_string()
}

/// `(section, township, range)` parsed from legal-location text such as
/// `"T34S R11W, Sec. 1"`, `"SEC 1 T34S R11W"` or `"1-34S-11W"`.
fn parse_str(s: &str) -> (Option<u32>, Option<String>, Option<String>) {
    let up = s.to_ascii_uppercase();
    let toks: Vec<&str> = up
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-')
        .filter(|t| !t.is_empty())
        .collect();
    let (mut sec, mut twp, mut rng) = (None, None, None);
    let lead_num = |t: &str| -> Option<String> {
        let d: String = t.chars().take_while(char::is_ascii_digit).collect();
        if d.is_empty() {
            return None;
        }
        let rest = &t[d.len()..];
        let n: u32 = d.parse().ok()?;
        Some(format!("{n}{rest}"))
    };
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        let parts: Vec<&str> = t.split('-').collect();
        if parts.len() == 3 && parts.iter().all(|p| p.starts_with(|c: char| c.is_ascii_digit())) {
            sec = parts[0].parse().ok();
            twp = lead_num(parts[1]);
            rng = lead_num(parts[2]);
        } else if t == "SEC" || t == "SECTION" || t == "S" {
            if let Some(n) = toks.get(i + 1).and_then(|n| n.parse::<u32>().ok()) {
                sec = Some(n);
                i += 1;
            }
        } else if let Some(r) = t.strip_prefix("SEC") {
            sec = r.parse().ok().or(sec);
        } else if t == "T" || t == "TWP" {
            twp = toks.get(i + 1).and_then(|n| lead_num(n)).or(twp);
            i += 1;
        } else if t == "R" || t == "RNG" || t == "RGE" {
            rng = toks.get(i + 1).and_then(|n| lead_num(n)).or(rng);
            i += 1;
        } else if let Some(r) = t.strip_prefix('T').filter(|r| r.starts_with(|c: char| c.is_ascii_digit())) {
            twp = lead_num(r);
        } else if let Some(r) = t.strip_prefix('R').filter(|r| r.starts_with(|c: char| c.is_ascii_digit())) {
            rng = lead_num(r);
        }
        i += 1;
    }
    (sec, twp, rng)
}

/// Evidence from the LAS header fields against one candidate well:
/// `(matches, mismatches, text)`.
fn hint_score(f: &BTreeMap<String, (String, String)>, h: &WellHint) -> (i32, i32, String) {
    let get = |k: &str| f.get(k).map(|(u, v)| (u.as_str(), v.trim())).filter(|(_, v)| !v.is_empty());
    let num = |k: &str| get(k).and_then(|(u, v)| v.parse::<f64>().ok().map(|x| (u, x)));
    let mut good = 0;
    let mut bad = 0;
    let mut text: Vec<String> = Vec::new();
    let mut mark = |what: &str, ok: Option<bool>| match ok {
        Some(true) => {
            good += 1;
            text.push(format!("{what}=match"));
        }
        Some(false) => {
            bad += 1;
            text.push(format!("{what}=mismatch"));
        }
        None => {}
    };
    // Name: WELL, LEASE, or LEASE + WELL against the label; numbers-only IDs skipped.
    if let Some(label) = h.label.as_deref().map(norm_name).filter(|l| !l.is_empty()) {
        let well = get("WELL").map(|(_, v)| norm_name(v)).unwrap_or_default();
        let lease = get("LEASE").map(|(_, v)| norm_name(v)).unwrap_or_default();
        let mut names: Vec<String> = [well.clone(), lease.clone(), format!("{lease} {well}")]
            .into_iter()
            .map(|n| n.trim().to_string())
            .filter(|n| n.len() >= 3 && !n.chars().all(|c| c.is_ascii_digit() || c == ' '))
            .collect();
        names.dedup();
        if !names.is_empty() {
            let ok = names.iter().any(|n| n == &label || (n.len() >= 4 && label.contains(n.as_str())) || (label.len() >= 4 && n.contains(label.as_str())));
            mark("name", Some(ok));
        }
    }
    // Latitude / longitude: within 0.5 km match, beyond 2 km mismatch.
    let lat = num("LAT").or_else(|| num("LATI")).or_else(|| num("LATITUDE")).map(|x| x.1);
    let lon = num("LON").or_else(|| num("LONG")).or_else(|| num("LONGITUDE")).map(|x| x.1);
    if let (Some(a), Some(o), Some(b), Some(p)) = (lat, lon, h.lat_deg, h.lon_deg) {
        let dy = (a - b) * 111.2;
        let dx = (o - p) * 111.2 * ((a + b) * 0.5).to_radians().cos();
        let d = (dx * dx + dy * dy).sqrt();
        mark("lat_lon", if d <= 0.5 { Some(true) } else if d > 2.0 { Some(false) } else { None });
    }
    // Section-township-range.
    if let Some(loc) = h.location.as_deref() {
        let mut las_loc = get("LOC").map(|(_, v)| v.to_string()).unwrap_or_default();
        for (k, p) in [("SECT", "SEC "), ("TOWN", " T"), ("RANG", " R")] {
            if let Some((_, v)) = get(k) {
                las_loc.push_str(&format!(" {p}{v}"));
            }
        }
        let a = parse_str(&las_loc);
        let b = parse_str(loc);
        let pairs = [
            (a.0.map(|x| x.to_string()), b.0.map(|x| x.to_string())),
            (a.1.clone(), b.1.clone()),
            (a.2.clone(), b.2.clone()),
        ];
        let both: Vec<bool> = pairs
            .iter()
            .filter_map(|(x, y)| match (x, y) {
                (Some(x), Some(y)) => Some(x.trim_end_matches(|c: char| c.is_ascii_alphabetic()) == y.trim_end_matches(|c: char| c.is_ascii_alphabetic())),
                _ => None,
            })
            .collect();
        if both.len() >= 2 {
            mark("sec_twp_rng", Some(both.iter().all(|&x| x)));
        }
    }
    // Depth: LAS STOP (deepest log depth) against the well's TD.
    if let (Some((u, stop)), Some(td)) = (num("STOP"), h.td_ft) {
        let stop = if u.eq_ignore_ascii_case("m") { stop / 0.3048 } else { stop };
        let d = (stop - td).abs();
        mark("depth", if d <= 100.0 { Some(true) } else if d > 1000.0 { Some(false) } else { None });
    }
    (good, bad, text.join(","))
}

/// Settle a file whose header ID names `h` and whose file name names
/// `by_file` (both known, different): KB elevation first, then the other
/// header fields. `(well or None when ambiguous, decision, reason)`.
fn settle_conflict(
    path: &Path,
    h: &str,
    by_file: &str,
    hints: &BTreeMap<String, WellHint>,
) -> (Option<String>, MatchDecision, String) {
    let none = WellHint::default();
    let hh = hints.get(h).unwrap_or(&none);
    let hf = hints.get(by_file).unwrap_or(&none);
    let las_kb = las_kb_ft(path);
    let near = |w: &WellHint| match (las_kb, w.kb_ft) {
        (Some(a), Some(b)) => (a - b).abs() <= 1.0,
        _ => false,
    };
    let fmt = |x: Option<f64>| x.map_or("?".to_string(), |v| format!("{v}"));
    let kb_text = format!(
        "LAS KB {} ft; {h} KB {}, {by_file} KB {}",
        fmt(las_kb),
        fmt(hh.kb_ft),
        fmt(hf.kb_ft)
    );
    match (near(hh), near(hf)) {
        (true, false) => {
            return (Some(h.to_string()), MatchDecision::Kb, format!("{kb_text}: KB matches the header well"))
        }
        (false, true) => {
            return (
                Some(by_file.to_string()),
                MatchDecision::Kb,
                format!("{kb_text}: KB matches the file-name well"),
            )
        }
        _ => {}
    }
    let f = las_fields(path);
    let (gh, bh, th) = hint_score(&f, hh);
    let (gf, bf, tf) = hint_score(&f, hf);
    let (sh, sf) = (gh - bh, gf - bf);
    let detail = format!(
        "{kb_text}: KB cannot decide; {h} [{}], {by_file} [{}]",
        if th.is_empty() { "no evidence" } else { &th },
        if tf.is_empty() { "no evidence" } else { &tf }
    );
    if sh >= 1 && sh > sf {
        (Some(h.to_string()), MatchDecision::Hints, format!("{detail}: header well matches better"))
    } else if sf >= 1 && sf > sh {
        (Some(by_file.to_string()), MatchDecision::Hints, format!("{detail}: file-name well matches better"))
    } else {
        (None, MatchDecision::Excluded, format!("{detail}: ambiguous, file left out"))
    }
}

/// [`discover_las`] with header-ID matching against the header well keys
/// `known`. A file whose LAS header names exactly one known well goes to that
/// well; otherwise the file name decides. When both name different known
/// wells, the KB elevation (`EKB`/`KB`/`EREF`/`EDF`, within 1 ft of one
/// candidate's `kb_ft` only) decides; else the well name, latitude/longitude,
/// section-township-range and depth fields decide when one candidate clearly
/// matches better; else the file is left out. Both wells get a
/// `las_match_conflict` note. Files named in `overrides` (file name -> well
/// key) always go to that well.
pub fn discover_las_report(
    las_dirs: &[PathBuf],
    exclude: &BTreeSet<String>,
    known: &BTreeSet<String>,
    hints: &BTreeMap<String, WellHint>,
    overrides: &BTreeMap<String, String>,
    mode: LasMatch,
) -> SxResult<LasDiscovery> {
    let mut out: BTreeMap<String, WellFiles> = BTreeMap::new();
    let mut matches: Vec<LasMatchRow> = Vec::new();
    let mut seen_hash: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut extra_notes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (i, dir) in las_dirs.iter().enumerate() {
        for path in list_las_files(dir)? {
            let fname = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            let by_file = well_name_from_path(&path);
            let name = if let Some(w) = overrides.get(&fname) {
                let reason = if known.contains(w) {
                    "[ingest.las_overrides]".to_string()
                } else {
                    format!("[ingest.las_overrides]; {w} is not in the header table")
                };
                tracing::info!(file = fname.as_str(), well = w.as_str(), "LAS file assigned by las_overrides");
                matches.push(LasMatchRow {
                    file: fname.clone(),
                    path: path.display().to_string(),
                    header_well: String::new(),
                    file_name_well: by_file.clone(),
                    used: w.clone(),
                    decision: MatchDecision::Override,
                    reason,
                });
                w.clone()
            } else {
                let by_header = match mode {
                    LasMatch::HeaderFirst => match_header_ids(&las_header_ids(&path), known),
                    LasMatch::FileName => None,
                };
                match &by_header {
                    Some(h) if h != &by_file && known.contains(&by_file) => {
                        let (used, decision, reason) = settle_conflict(&path, h, &by_file, hints);
                        let used_text = used.clone().unwrap_or_else(|| "none (excluded)".into());
                        let note = format!(
                            "las_match_conflict:{fname}: header ID names {h}, file name names {by_file}; used {used_text} ({reason})"
                        );
                        tracing::warn!(
                            file = fname.as_str(),
                            header = h.as_str(),
                            file_name = by_file.as_str(),
                            used = used_text.as_str(),
                            reason = reason.as_str(),
                            "LAS header ID and file name name different wells"
                        );
                        extra_notes.entry(h.clone()).or_default().push(note.clone());
                        extra_notes.entry(by_file.clone()).or_default().push(note);
                        matches.push(LasMatchRow {
                            file: fname.clone(),
                            path: path.display().to_string(),
                            header_well: h.clone(),
                            file_name_well: by_file.clone(),
                            used: used.clone().unwrap_or_default(),
                            decision,
                            reason,
                        });
                        match used {
                            Some(u) => u,
                            None => continue,
                        }
                    }
                    Some(h) => h.clone(),
                    None => by_file,
                }
            };
            if exclude.contains(&name) {
                continue;
            }
            if let Ok(hash) = sha256_file(&path) {
                if let Some((w, f)) = seen_hash.get(&hash) {
                    let note = format!("duplicate_las:{fname} has the same bytes as {f} (well {w})");
                    tracing::warn!(file = fname.as_str(), same_as = f.as_str(), "duplicate LAS file");
                    extra_notes.entry(name.clone()).or_default().push(note.clone());
                    if w == &name {
                        continue;
                    }
                    extra_notes.entry(w.clone()).or_default().push(note);
                } else {
                    seen_hash.insert(hash, (name.clone(), fname.clone()));
                }
            }
            match out.get_mut(&name) {
                None => {
                    out.insert(
                        name,
                        WellFiles {
                            folder_index: i,
                            paths: vec![path],
                            n_shadowed: 0,
                            notes: Vec::new(),
                        },
                    );
                }
                Some(w) if w.folder_index == i => w.paths.push(path),
                Some(w) => w.n_shadowed += 1,
            }
        }
    }
    for (name, notes) in extra_notes {
        if let Some(w) = out.get_mut(&name) {
            w.notes.extend(notes);
        }
    }
    Ok(LasDiscovery { wells: out, matches })
}

/// [`discover_las_report`] without overrides, returning only the files per well.
///
/// # Errors
/// As [`discover_las_report`].
pub fn discover_las_matched(
    las_dirs: &[PathBuf],
    exclude: &BTreeSet<String>,
    known: &BTreeSet<String>,
    hints: &BTreeMap<String, WellHint>,
    mode: LasMatch,
) -> SxResult<BTreeMap<String, WellFiles>> {
    Ok(discover_las_report(las_dirs, exclude, known, hints, &BTreeMap::new(), mode)?.wells)
}
