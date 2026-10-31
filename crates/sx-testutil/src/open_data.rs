// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Open example data: Kansas Geological Survey (KGS) wells in township T34S
//! R11W (Barber County, Kansas).
//!
//! Source: "The source of this material is the Kansas Geological Survey
//! website at http://www.kgs.ku.edu/. All Rights Reserved." KGS terms of use
//! (https://www.kgs.ku.edu/General/copyright.html) allow use, redistribution
//! and modification without fee, with this citation. LAS files from the KGS
//! LAS index (`ks_las_files.txt`); formation tops from `ks_tops.txt` (KGS
//! notes the tops come from various sources and are not confirmed).
//!
//! The data is not stored in the repository. `scripts/fetch_open_data.sh`
//! downloads it; tests find it through `SX_OPEN_DATA` (the folder holding
//! `kgs/T34S_R11W/`) and skip with a note when it is not set.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sx_core::ingest::write_edges;
use sx_core::types::{WellHeader, WellId};
use sx_core::{SxError, SxResult};

/// Env var naming the open-data folder.
pub const OPEN_DATA_ENV: &str = "SX_OPEN_DATA";

/// `$SX_OPEN_DATA/kgs/T34S_R11W` when the variable is set and the folder
/// holds `index.csv`; otherwise `None` (callers print a skip note).
pub fn kgs_dir() -> Option<PathBuf> {
    let root = std::env::var_os(OPEN_DATA_ENV)?;
    let dir = PathBuf::from(root).join("kgs").join("T34S_R11W");
    dir.join("index.csv").is_file().then_some(dir)
}

/// Minimal CSV reader for the KGS exports: comma separated, fields in
/// double quotes, no embedded quotes.
fn read_quoted_csv(path: &Path) -> SxResult<Vec<BTreeMap<String, String>>> {
    let bytes = std::fs::read(path).map_err(|e| SxError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let text: String = bytes.iter().map(|&b| b as char).collect(); // latin-1
    let split = |l: &str| -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut q = false;
        for c in l.chars() {
            match c {
                '"' => q = !q,
                ',' if !q => out.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            }
        }
        out.push(cur);
        out
    };
    let mut lines = text.lines();
    let head = split(lines.next().unwrap_or_default());
    Ok(lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            head.iter()
                .cloned()
                .zip(split(l.trim_end_matches('\r')))
                .collect()
        })
        .collect())
}

/// Canonical name of a KGS formation top (merges "Heebner Shale" and
/// "Heebner", "Cherokee Group"/"Cherokee Shale" and "Cherokee", and so on).
pub fn canonical_top(name: &str) -> String {
    let mut n = name.trim().to_string();
    for suffix in [" Shale", " Group", " Limestone", " Formation", " Fm"] {
        if let Some(s) = n.strip_suffix(suffix) {
            n = s.to_string();
        }
    }
    n
}

/// One KGS well picked for a project.
#[derive(Debug, Clone)]
pub struct KgsWell {
    /// 14-digit API number (file stem of the copied LAS).
    pub api: String,
    /// LAS file in the source folder.
    pub las: PathBuf,
    /// Latitude, longitude (degrees).
    pub lat_lon: (f64, f64),
    /// Kelly bushing (or reference) elevation, feet.
    pub elev_ft: f64,
    /// KGS id.
    pub kid: String,
}

/// Wells of the township, one LAS per API number (the largest file, which in
/// this township is the full logging run), sorted by API, at most `max`.
pub fn kgs_wells(dir: &Path, max: usize) -> SxResult<Vec<KgsWell>> {
    let mut best: BTreeMap<String, (u64, KgsWell)> = BTreeMap::new();
    for r in read_quoted_csv(&dir.join("index.csv"))? {
        let api = r.get("API_NUM_NODASH").cloned().unwrap_or_default();
        if api.len() != 14 {
            continue;
        }
        let file = r["URL"].rsplit('/').next().unwrap_or_default().to_string();
        let las = dir.join(&file);
        let Ok(meta) = std::fs::metadata(&las) else {
            continue;
        };
        let p = |k: &str| r.get(k).and_then(|v| v.trim().parse::<f64>().ok());
        let w = KgsWell {
            api: api.clone(),
            las,
            lat_lon: (
                p("Latitude").unwrap_or(f64::NAN),
                p("Longitude").unwrap_or(f64::NAN),
            ),
            elev_ft: p("Elevation").unwrap_or(0.0),
            kid: r["KGS_ID"].clone(),
        };
        if best.get(&api).is_none_or(|(s, _)| meta.len() > *s) {
            best.insert(api, (meta.len(), w));
        }
    }
    Ok(best.into_values().map(|(_, w)| w).take(max).collect())
}

/// Write a project (`project.toml`, `edges/`, `las/<API>.las`) from up to
/// `max_wells` KGS wells. Coordinates: local equirectangular metres around
/// the mean location. Tops: canonical KGS tops present in at least
/// `min_top_wells` of the chosen wells, measured depth in feet.
/// Returns the `project.toml` path.
///
/// # Errors
/// `SxError::Io` and the errors of the edge writer.
pub fn write_kgs_project(
    dir: &Path,
    src: &Path,
    max_wells: usize,
    min_top_wells: usize,
) -> SxResult<PathBuf> {
    let io = |path: &Path, e: std::io::Error| SxError::Io {
        path: path.display().to_string(),
        source: e,
    };
    let wells = kgs_wells(src, max_wells)?;
    let las_dir = dir.join("las");
    std::fs::create_dir_all(&las_dir).map_err(|e| io(&las_dir, e))?;
    let n = wells.len().max(1) as f64;
    let lat0 = wells.iter().map(|w| w.lat_lon.0).sum::<f64>() / n;
    let lon0 = wells.iter().map(|w| w.lat_lon.1).sum::<f64>() / n;
    let m_per_deg = 111_320.0;
    let mut headers = Vec::new();
    let mut id_of_kid = BTreeMap::new();
    for (i, w) in wells.iter().enumerate() {
        let dst = las_dir.join(format!("{}.las", w.api));
        std::fs::copy(&w.las, &dst).map_err(|e| io(&dst, e))?;
        let id = WellId(i as i32 + 1);
        id_of_kid.insert(w.kid.clone(), id);
        headers.push(WellHeader {
            id,
            name: w.api.clone(),
            x_m: (w.lat_lon.1 - lon0) * m_per_deg * lat0.to_radians().cos(),
            y_m: (w.lat_lon.0 - lat0) * m_per_deg,
            kb_elev_ft: w.elev_ft,
        });
    }
    // Tops: first pick per (well, canonical name); keep names seen in enough wells.
    let mut picks: BTreeMap<(WellId, String), f64> = BTreeMap::new();
    let tops_csv = src.join("tops.csv");
    if tops_csv.is_file() {
        for r in read_quoted_csv(&tops_csv)? {
            let Some(id) = id_of_kid.get(&r["KID"]) else {
                continue;
            };
            let Ok(md) = r["TOP"].trim().parse::<f64>() else {
                continue;
            };
            picks
                .entry((*id, canonical_top(&r["FORMATION"])))
                .or_insert(md);
        }
    }
    let mut count: BTreeMap<&str, BTreeSet<WellId>> = BTreeMap::new();
    for (w, name) in picks.keys() {
        count.entry(name.as_str()).or_default().insert(*w);
    }
    let keep: BTreeSet<String> = count
        .iter()
        .filter(|(_, s)| s.len() >= min_top_wells)
        .map(|(k, _)| (*k).to_string())
        .collect();
    let tops: Vec<(WellId, String, f64, u8)> = picks
        .iter()
        .filter(|((_, n), _)| keep.contains(n))
        .map(|((w, n), md)| (*w, n.clone(), *md, 0u8))
        .collect();
    write_edges(dir, &headers, &BTreeMap::new(), &tops)?;
    let toml = dir.join("project.toml");
    std::fs::write(&toml, crate::synth_project::project_toml()).map_err(|e| io(&toml, e))?;
    Ok(toml)
}
