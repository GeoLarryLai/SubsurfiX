// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Real-data checks on open Kansas Geological Survey (KGS) wells (T34S R11W,
//! Barber County). They replace the earlier private-fixture tests. Without
//! `SX_OPEN_DATA` each test prints a skip note and passes; fetch the data
//! with `scripts/fetch_open_data.sh`.
//! "The source of this material is the Kansas Geological Survey website at
//! http://www.kgs.ku.edu/. All Rights Reserved."

use std::collections::{BTreeMap, BTreeSet};

use sx_core::config::load_project;
use sx_core::fill::holdout::score_holdout;
use sx_core::ingest::{ingest_fleet, run_ingest, AliasTable, IngestParams};
use sx_core::pipeline::{run, RunOptions};
use sx_core::types::{LogKind, WellHeader, WellId};
use sx_testutil::open_data::{kgs_dir, kgs_wells, write_kgs_project};

macro_rules! need_data {
    () => {
        match kgs_dir() {
            Some(d) => d,
            None => {
                eprintln!("SX_OPEN_DATA not set (or no kgs/T34S_R11W); skipping");
                return;
            }
        }
    };
}

/// Replaces `primaries_match_expected`: the alias table resolves the main
/// curves of real vendor LAS files.
#[test]
fn kgs_primaries_resolve() {
    let src = need_data!();
    let tmp = tempfile::tempdir().unwrap();
    let wells = kgs_wells(&src, 200).unwrap();
    let las = tmp.path().join("las");
    std::fs::create_dir_all(&las).unwrap();
    let headers: Vec<WellHeader> = wells
        .iter()
        .enumerate()
        .map(|(i, w)| {
            std::fs::copy(&w.las, las.join(format!("{}.las", w.api))).unwrap();
            WellHeader {
                id: WellId(i as i32 + 1),
                name: w.api.clone(),
                x_m: 0.0,
                y_m: 0.0,
                kb_elev_ft: 0.0,
            }
        })
        .collect();
    let params = IngestParams {
        las_dirs: vec![las],
        wells: None,
        exclude_wells: BTreeSet::new(),
        aliases: AliasTable::builtin(),
        max_gap_ft: 2.0,
        las_match: Default::default(),
        hints: Default::default(),
        las_overrides: Default::default(),
    };
    let fleet = ingest_fleet(&headers, &params).unwrap();
    let mut resolved: BTreeMap<String, usize> = BTreeMap::new();
    for w in &fleet.wells {
        for r in &w.report {
            if r.log_status == "resolved" {
                *resolved.entry(r.log.clone()).or_default() += 1;
            }
        }
    }
    let n = headers.len();
    eprintln!("{n} wells; resolved per log: {resolved:?}");
    // Survey (2026-10-28): GR 130, RHOB 129, RES 130, NPHI 95, PE 66, DT 12 of 132 wells.
    for (log, min_frac) in [("GR", 0.9), ("RESD", 0.85), ("NPHI", 0.6), ("PEF", 0.4)] {
        let got = resolved.get(log).copied().unwrap_or(0);
        assert!(got as f64 >= min_frac * n as f64, "{log}: {got}/{n}");
    }
}

/// Replaces the private-fixture ingest smoke test: the ingest stage on a KGS project.
#[test]
fn kgs_ingest_smoke() {
    let src = need_data!();
    let tmp = tempfile::tempdir().unwrap();
    let toml = write_kgs_project(tmp.path(), &src, 40, 15).unwrap();
    let cfg = load_project(&toml).unwrap();
    let out = run_ingest(&cfg.ingest, tmp.path()).unwrap();
    eprintln!(
        "wells={} logs={} picks={}",
        out.wells.len(),
        out.logs.len(),
        out.picks.len()
    );
    assert_eq!(out.wells.len(), 40);
    assert!(out.logs.len() >= 38, "logs for {} wells", out.logs.len());
    for g in &out.geometry {
        g.validate().unwrap();
    }
    for l in &out.logs {
        assert!(l.grid.len > 0);
        let gr = &l.curves[&LogKind::Gr];
        let fin: Vec<f32> = gr.iter().copied().filter(|v| v.is_finite()).collect();
        if !fin.is_empty() {
            let mean = fin.iter().sum::<f32>() / fin.len() as f32;
            assert!((5.0..300.0).contains(&mean), "GR mean {mean}");
        }
    }
    assert!(!out.picks.is_empty());
}

/// Replaces the private-fixture holdout test: full pipeline, then leak-free holdout scoring.
#[test]
fn kgs_pipeline_and_holdout() {
    let src = need_data!();
    let tmp = tempfile::tempdir().unwrap();
    let toml = write_kgs_project(tmp.path(), &src, 25, 10).unwrap();
    let cfg = load_project(&toml).unwrap();
    run(&cfg, &RunOptions::new(tmp.path())).unwrap();
    let report = score_holdout(tmp.path(), &cfg).unwrap();
    eprintln!("{}", report.summary_table());
    assert!(report.scores_path.is_file() && report.summary_path.is_file());
    assert!(!report.summary.is_empty());
}
