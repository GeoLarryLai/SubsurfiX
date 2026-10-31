// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Acceptance tests for `sx_core::config`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sx_core::config::*;
use sx_core::types::LogKind;
use sx_core::SxError;

fn expect_bad(edit: impl FnOnce(&mut ProjectConfig), path: &str, message: &str) {
    let mut cfg = ProjectConfig::default_with_crs("EPSG:32614");
    edit(&mut cfg);
    match validate(&cfg) {
        Err(SxError::Config(got)) => assert_eq!(got, format!("{path}: {message}"), "{path}"),
        other => panic!("{path}: {other:?}"),
    }
}

fn expect_toml(body: &str, path: &str, message: &str) {
    let text = format!("[project]\ncrs = \"EPSG:32614\"\n{body}");
    match from_toml_str(&text) {
        Err(SxError::Config(got)) => assert_eq!(got, format!("{path}: {message}"), "{text}"),
        other => panic!("{text}\n{other:?}"),
    }
}

#[test]
fn minimal_file_loads() {
    let text = "seed = 42\n[project]\ncrs = \"EPSG:32614\"\n";
    let cfg = from_toml_str(text).unwrap();
    let mut expect = ProjectConfig::default_with_crs("EPSG:32614");
    expect.seed = 42;
    assert_eq!(cfg, expect);
}

#[test]
fn five_expanded_defaults() {
    let cfg = from_toml_str("seed = 42\n[project]\ncrs = \"EPSG:32614\"\n").unwrap();
    assert_eq!(cfg.qc.tail_pct, [0.25, 99.75]);
    assert_eq!(cfg.dtw.window.band_frac, 0.25);
    assert_eq!(cfg.dtw.full.band_frac, 0.12);
    assert_eq!(cfg.fill.lightgbm.learning_rate, 0.05);
    assert_eq!(cfg.correlate.graph.k_graph, 6);
    assert_eq!(cfg.refresh.percentile, 95.0);
}

#[test]
fn missing_project_fails() {
    match from_toml_str("seed = 1\n") {
        Err(SxError::Config(message)) => assert!(message.contains("project"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn missing_crs_fails() {
    match from_toml_str("[project]\n") {
        Err(SxError::Config(message)) => assert!(message.contains("crs"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unknown_keys_fail() {
    let cases = [
        (
            "sed = 1\n[project]\ncrs = \"EPSG:32614\"\n",
            "sed",
            Some("parse error at line 1"),
        ),
        (
            "[project]\ncrs = \"EPSG:32614\"\n[qc]\ntail_pctt = [0.25, 99.75]\n",
            "tail_pctt",
            Some("parse error at line 4"),
        ),
        (
            "[project]\ncrs = \"EPSG:32614\"\n[qc.jury]\nquorun = 1\n",
            "quorun",
            None,
        ),
        (
            "[project]\ncrs = \"EPSG:32614\"\n[dtw.window]\nfoo = 1\n",
            "foo",
            None,
        ),
        (
            "[project]\ncrs = \"EPSG:32614\"\n[[edges.headers]]\npath = \"h.csv\"\nkb_unit = \"ft\"\nxy_unit = \"m\"\nbogus = 1\n",
            "bogus",
            None,
        ),
    ];
    for (text, typo, line) in cases {
        match from_toml_str(text) {
            Err(SxError::Config(message)) => {
                assert!(message.contains(typo), "{message}");
                if let Some(prefix) = line {
                    assert!(message.starts_with(prefix), "{message}");
                }
            }
            other => panic!("{typo}: {other:?}"),
        }
    }
}

#[test]
fn parse_error_has_line_and_column() {
    match from_toml_str("[project]\ncrs = ") {
        Err(SxError::Config(message)) => {
            assert!(
                message.starts_with("parse error at line 2, column"),
                "{message}"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn window_partial_table_keeps_window_defaults() {
    let cfg =
        from_toml_str("[project]\ncrs = \"EPSG:32614\"\n[dtw.window]\ngr_weight = 0.4\n").unwrap();
    let mut expect = ProjectConfig::default_with_crs("EPSG:32614");
    expect.dtw.window.gr_weight = 0.4;
    assert_eq!(cfg.dtw.window, expect.dtw.window);
    assert_eq!(cfg, expect);
}

#[test]
fn non_finite_rejected() {
    let inf = from_toml_str("[project]\ncrs = \"EPSG:32614\"\n[refresh]\nrefresh_tol_ft = inf\n");
    match inf {
        Err(SxError::Config(message)) => {
            assert_eq!(message, "refresh.refresh_tol_ft: must be finite (got inf)");
        }
        other => panic!("{other:?}"),
    }
    let nan = from_toml_str("[project]\ncrs = \"EPSG:32614\"\n[refresh]\nrefresh_tol_ft = nan\n");
    match nan {
        Err(SxError::Config(message)) => {
            assert_eq!(message, "refresh.refresh_tol_ft: must be finite (got NaN)");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn default_is_valid() {
    validate(&ProjectConfig::default_with_crs("EPSG:32614")).unwrap();
}

#[test]
fn validation_rules_table() {
    plain_assignment_rules();
    doc_row_rules();
}

fn plain_assignment_rules() {
    expect_bad(
        |c| c.seed = 9_223_372_036_854_775_808,
        "seed",
        "must be <= 9223372036854775807 (got 9223372036854775808)",
    );
    expect_bad(
        |c| c.project.crs = String::new(),
        "project.crs",
        "must not be empty",
    );
    expect_bad(
        |c| c.ingest.max_gap_ft = -1.0,
        "ingest.max_gap_ft",
        "must be >= 0 (got -1)",
    );

    expect_bad(
        |c| c.qc.constant_min_count = 0,
        "qc.constant_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.tail_min_count = 0,
        "qc.tail_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.rescue_min_count = 0,
        "qc.rescue_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.rescue_ref_wells = 0,
        "qc.rescue_ref_wells",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.rescue_min_ref_wells = 0,
        "qc.rescue_min_ref_wells",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.repeat_min_len = 0,
        "qc.repeat_min_len",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.repeat_min_count = 0,
        "qc.repeat_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.plateau_window = 0,
        "qc.plateau_window",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.plateau_window_min = 0,
        "qc.plateau_window_min",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.plateau_min_len = 0,
        "qc.plateau_min_len",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.plateau_min_count = 0,
        "qc.plateau_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.run_min_len = 0,
        "qc.run_min_len",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.stats_min_count = 0,
        "qc.stats_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.fleet_min_wells = 0,
        "qc.fleet_min_wells",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.post_flat_min_count = 0,
        "qc.post_flat_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.near_count = 0,
        "qc.near_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(|c| c.qc.near_min = 0, "qc.near_min", "must be >= 1 (got 0)");
    expect_bad(
        |c| c.qc.ring_first = 0,
        "qc.ring_first",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.ring_last = 0,
        "qc.ring_last",
        "must be >= 1 (got 0)",
    );
    expect_bad(|c| c.qc.ring_min = 0, "qc.ring_min", "must be >= 1 (got 0)");
    expect_bad(|c| c.qc.jsd_k = 0, "qc.jsd_k", "must be >= 1 (got 0)");
    expect_bad(
        |c| c.qc.jsd_neighbors = 0,
        "qc.jsd_neighbors",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jsd_min_neighbors = 0,
        "qc.jsd_min_neighbors",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jsd_min_wells = 0,
        "qc.jsd_min_wells",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jsd_max_count = 0,
        "qc.jsd_max_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.pileup_min_len = 0,
        "qc.pileup_min_len",
        "must be >= 1 (got 0)",
    );

    expect_bad(
        |c| c.qc.constant_rel_tol = -1.0,
        "qc.constant_rel_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.tail_pad_frac = -1.0,
        "qc.tail_pad_frac",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.rescue_min_cv = -1.0,
        "qc.rescue_min_cv",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.repeat_rel_tol = -1.0,
        "qc.repeat_rel_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.plateau_std_share = -1.0,
        "qc.plateau_std_share",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.mad_floor = -1.0,
        "qc.mad_floor",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.lowvar_ratio = -1.0,
        "qc.lowvar_ratio",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.post_flat_ratio = -1.0,
        "qc.post_flat_ratio",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.dev_limit = -1.0,
        "qc.dev_limit",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.jsd_mad_mult = -1.0,
        "qc.jsd_mad_mult",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.jsd_floor = -1.0,
        "qc.jsd_floor",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.gr_span_frac = -1.0,
        "qc.gr_span_frac",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.resd_decade_log10 = -1.0,
        "qc.resd_decade_log10",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.pileup_min_share = -1.0,
        "qc.pileup_min_share",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.pileup_rel_tol = -1.0,
        "qc.pileup_rel_tol",
        "must be >= 0 (got -1)",
    );

    expect_bad(
        |c| c.qc.tail_pct[0] = 101.0,
        "qc.tail_pct[0]",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.qc.tail_pct[1] = 101.0,
        "qc.tail_pct[1]",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.qc.stats_pct[0] = 101.0,
        "qc.stats_pct[0]",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.qc.stats_pct[1] = 101.0,
        "qc.stats_pct[1]",
        "must be in [0, 100] (got 101)",
    );

    expect_bad(
        |c| c.qc.rescue_candidate_band[0] = 0.0,
        "qc.rescue_candidate_band[0]",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.qc.rescue_accept_band[0] = 0.0,
        "qc.rescue_accept_band[0]",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.qc.unit_band[0] = 0.0,
        "qc.unit_band[0]",
        "must be > 0 (got 0)",
    );

    let decreasing = [99.75, 0.25];
    let increasing = "must be strictly increasing (got [99.75, 0.25])";
    expect_bad(|c| c.qc.tail_pct = decreasing, "qc.tail_pct", increasing);
    expect_bad(
        |c| c.qc.rescue_candidate_band = decreasing,
        "qc.rescue_candidate_band",
        increasing,
    );
    expect_bad(
        |c| c.qc.rescue_accept_band = decreasing,
        "qc.rescue_accept_band",
        increasing,
    );
    expect_bad(|c| c.qc.stats_pct = decreasing, "qc.stats_pct", increasing);
    expect_bad(|c| c.qc.unit_band = decreasing, "qc.unit_band", increasing);

    expect_bad(
        |c| c.qc.ring_first = 65,
        "qc.ring_first",
        "must be <= qc.ring_last (got 65, limit 64)",
    );
    expect_bad(
        |c| {
            c.qc.rescue_factors.insert("GR".to_string(), 1.0);
        },
        "qc.rescue_factors.GR",
        "must be finite, > 0 and != 1 (got 1)",
    );
    expect_bad(
        |c| {
            c.qc.rescue_factors.insert("GR".to_string(), 0.0);
        },
        "qc.rescue_factors.GR",
        "must be finite, > 0 and != 1 (got 0)",
    );

    expect_bad(
        |c| c.qc.jury.min_count = 0,
        "qc.jury.min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.near_jurors = 0,
        "qc.jury.near_jurors",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.max_jurors = 0,
        "qc.jury.max_jurors",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.quorum = 0,
        "qc.jury.quorum",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.window_min = 0,
        "qc.jury.window_min",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.hop_min = 0,
        "qc.jury.hop_min",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.joint_floor = 0,
        "qc.jury.joint_floor",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.min_streak = 0,
        "qc.jury.min_streak",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.corr_min_count = 0,
        "qc.jury.corr_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.max_dist_m = 0.0,
        "qc.jury.max_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.dist_floor_m = 0.0,
        "qc.jury.dist_floor_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.dist_power = -1.0,
        "qc.jury.dist_power",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.jury.flat_std_share = -1.0,
        "qc.jury.flat_std_share",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.qc.jury.path_min_quality = 2.0,
        "qc.jury.path_min_quality",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.qc.jury.unaligned_weight = 2.0,
        "qc.jury.unaligned_weight",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.qc.jury.window_ft[0] = 0.0,
        "qc.jury.window_ft[0]",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.window_ft[1] = 0.0,
        "qc.jury.window_ft[1]",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.joint_share = 0.0,
        "qc.jury.joint_share",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.bad_share = 0.0,
        "qc.jury.bad_share",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.qc.jury.anticorr_limit = 1.0,
        "qc.jury.anticorr_limit",
        "must be in [-1, 0] (got 1)",
    );
    expect_bad(
        |c| c.qc.jury.quorum = 9,
        "qc.jury.quorum",
        "must be <= qc.jury.near_jurors (got 9, limit 8)",
    );
    expect_bad(
        |c| c.qc.jury.near_jurors = 21,
        "qc.jury.near_jurors",
        "must be <= qc.jury.max_jurors (got 21, limit 20)",
    );

    expect_bad(
        |c| c.align.pad_ft = 0.0,
        "align.pad_ft",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.align.k_pairs = 0,
        "align.k_pairs",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.align.tops.k = 0,
        "align.tops.k",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.align.tops.quorum = 0,
        "align.tops.quorum",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.align.tops.max_dist_m = 0.0,
        "align.tops.max_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.align.tops.min_dist_m = 0.0,
        "align.tops.min_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.align.tops.power = -1.0,
        "align.tops.power",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.align.pairs.min_span_ratio = 0.0,
        "align.pairs.min_span_ratio",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.align.pairs.max_span_ratio = 0.25,
        "align.pairs.max_span_ratio",
        "must be > align.pairs.min_span_ratio (got 0.25, limit 0.25)",
    );
    expect_bad(
        |c| c.align.pairs.depth_nodes = 1,
        "align.pairs.depth_nodes",
        "must be >= 2 (got 1)",
    );
    expect_bad(
        |c| c.align.pairs.depth_min_shared = 0,
        "align.pairs.depth_min_shared",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.align.pairs.depth_band_frac = 0.0,
        "align.pairs.depth_band_frac",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.align.pairs.depth_quality = 2.0,
        "align.pairs.depth_quality",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.pairs.fail_quality = 2.0,
        "align.pairs.fail_quality",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.pairs.fail_quality_no_coord = 2.0,
        "align.pairs.fail_quality_no_coord",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.confidence.observed = 2.0,
        "align.confidence.observed",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.confidence.pinched_user = 2.0,
        "align.confidence.pinched_user",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.confidence.pinched_other = 2.0,
        "align.confidence.pinched_other",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.confidence.estimated_one = 2.0,
        "align.confidence.estimated_one",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.confidence.estimated_none = 2.0,
        "align.confidence.estimated_none",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.align.confidence.pad_floor = 2.0,
        "align.confidence.pad_floor",
        "must be in [0, 1] (got 2)",
    );

    expect_bad(
        |c| c.dtw.interval.grid_budget = 0,
        "dtw.interval.grid_budget",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_nodes = 0,
        "dtw.interval.min_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.max_nodes = 0,
        "dtw.interval.max_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_shared_floor = 0,
        "dtw.interval.min_shared_floor",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_shared_cap = 0,
        "dtw.interval.min_shared_cap",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_shared_div = 0,
        "dtw.interval.min_shared_div",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_channel_samples = 0,
        "dtw.interval.min_channel_samples",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_channel_nodes = 0,
        "dtw.interval.min_channel_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.band_frac = 0.0,
        "dtw.interval.band_frac",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.gap_cost = -1.0,
        "dtw.interval.gap_cost",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.interval.max_gap_steps = -1.0,
        "dtw.interval.max_gap_steps",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.interval.min_span = -1.0,
        "dtw.interval.min_span",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.interval.pct_low = 101.0,
        "dtw.interval.pct_low",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.dtw.interval.pct_high = 101.0,
        "dtw.interval.pct_high",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.dtw.interval.pct_low = 99.0,
        "dtw.interval.pct_low",
        "must be < dtw.interval.pct_high (got 99, limit 99)",
    );
    expect_bad(
        |c| c.dtw.interval.dtw_factor = 2.0,
        "dtw.interval.dtw_factor",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.dtw.interval.linear_factor = 2.0,
        "dtw.interval.linear_factor",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.dtw.interval.resd_floor_ohmm = 0.0,
        "dtw.interval.resd_floor_ohmm",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.interval.min_nodes = 129,
        "dtw.interval.min_nodes",
        "must be <= dtw.interval.max_nodes (got 129, limit 128)",
    );
    expect_bad(
        |c| c.dtw.interval.min_shared_floor = 65,
        "dtw.interval.min_shared_floor",
        "must be <= dtw.interval.min_shared_cap (got 65, limit 64)",
    );

    expect_bad(
        |c| c.dtw.full.min_nodes = 0,
        "dtw.full.min_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.max_nodes = 0,
        "dtw.full.max_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.band_frac = 0.0,
        "dtw.full.band_frac",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.gr_weight = 2.0,
        "dtw.full.gr_weight",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.dtw.full.z_cap = 0.0,
        "dtw.full.z_cap",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.mad_factor = 0.0,
        "dtw.full.mad_factor",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.scale_floor = 0.0,
        "dtw.full.scale_floor",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.resd_floor_ohmm = 0.0,
        "dtw.full.resd_floor_ohmm",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.full.max_gap_steps = -1.0,
        "dtw.full.max_gap_steps",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.full.min_gap_ft = -1.0,
        "dtw.full.min_gap_ft",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.full.min_nodes = 2001,
        "dtw.full.min_nodes",
        "must be <= dtw.full.max_nodes (got 2001, limit 2000)",
    );

    expect_bad(
        |c| c.dtw.window.min_nodes = 0,
        "dtw.window.min_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.max_nodes = 0,
        "dtw.window.max_nodes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.band_frac = 0.0,
        "dtw.window.band_frac",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.gr_weight = 2.0,
        "dtw.window.gr_weight",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.dtw.window.z_cap = 0.0,
        "dtw.window.z_cap",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.mad_factor = 0.0,
        "dtw.window.mad_factor",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.scale_floor = 0.0,
        "dtw.window.scale_floor",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.resd_floor_ohmm = 0.0,
        "dtw.window.resd_floor_ohmm",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.dtw.window.max_gap_steps = -1.0,
        "dtw.window.max_gap_steps",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.window.min_gap_ft = -1.0,
        "dtw.window.min_gap_ft",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.dtw.window.min_nodes = 2001,
        "dtw.window.min_nodes",
        "must be <= dtw.window.max_nodes (got 2001, limit 2000)",
    );

    expect_bad(
        |c| c.normalize.ref_k = 0,
        "normalize.ref_k",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.ref_min_voters = 0,
        "normalize.ref_min_voters",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.pick_min_count = 0,
        "normalize.pick_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.depth_bins = 0,
        "normalize.depth_bins",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.depth_candidates = 0,
        "normalize.depth_candidates",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.scope = "bad".to_string(),
        "normalize.scope",
        "must be one of \"layer\", \"all\" (got \"bad\")",
    );
    expect_bad(
        |c| c.normalize.ref_power = -1.0,
        "normalize.ref_power",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.normalize.ref_dist_floor_m = 0.0,
        "normalize.ref_dist_floor_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.normalize.ref_span_ratio_max = 0.0,
        "normalize.ref_span_ratio_max",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.stretch_min_corr = 2.0,
        "normalize.stretch_min_corr",
        "must be in [-1, 1] (got 2)",
    );
    expect_bad(
        |c| c.normalize.gain_cap = 0.0,
        "normalize.gain_cap",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.normalize.gain_cap_extended = 1.5,
        "normalize.gain_cap_extended",
        "must be >= normalize.gain_cap (got 1.5, limit 2)",
    );
    expect_bad(
        |c| c.normalize.shift_only_frac = 0.0,
        "normalize.shift_only_frac",
        "must be in (0, 1) (got 0)",
    );
    expect_bad(
        |c| c.normalize.degenerate_span = -1.0,
        "normalize.degenerate_span",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.normalize.depth_max_cover = 0.0,
        "normalize.depth_max_cover",
        "must be in (0, 1] (got 0)",
    );

    expect_bad(
        |c| c.neighbors.knn_list_len = 0,
        "neighbors.knn_list_len",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.search_cap = 0,
        "neighbors.search_cap",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.k_primary = 0,
        "neighbors.k_primary",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.batch_size = 0,
        "neighbors.batch_size",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.quorum = 0,
        "neighbors.quorum",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.min_finite_donor = 0,
        "neighbors.min_finite_donor",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.validate_min_run = 0,
        "neighbors.validate_min_run",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.range_map_min_n = 0,
        "neighbors.range_map_min_n",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.trusted_min_n = 0,
        "neighbors.trusted_min_n",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.max_dist_m = 0.0,
        "neighbors.max_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.min_dist_m = 0.0,
        "neighbors.min_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.trusted_max_dist_m = 0.0,
        "neighbors.trusted_max_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.neighbors.idw_power = -1.0,
        "neighbors.idw_power",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.neighbors.donor_max_bracket_ft = -1.0,
        "neighbors.donor_max_bracket_ft",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.neighbors.interval_eps = -1.0,
        "neighbors.interval_eps",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.neighbors.validate_split_ft = -1.0,
        "neighbors.validate_split_ft",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.neighbors.flat_rel_tol = -1.0,
        "neighbors.flat_rel_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.neighbors.saturation_rel_tol = -1.0,
        "neighbors.saturation_rel_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.neighbors.iq_no_coord = 2.0,
        "neighbors.iq_no_coord",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.aq_unaligned = 2.0,
        "neighbors.aq_unaligned",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.conf_observed = 2.0,
        "neighbors.conf_observed",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.conf_estimated_one_user = 2.0,
        "neighbors.conf_estimated_one_user",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.conf_estimated_no_user = 2.0,
        "neighbors.conf_estimated_no_user",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.conf_pad_floor = 2.0,
        "neighbors.conf_pad_floor",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.trusted_min_quality = 2.0,
        "neighbors.trusted_min_quality",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.min_span_frac = 2.0,
        "neighbors.min_span_frac",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.neighbors.saturation_frac = 0.0,
        "neighbors.saturation_frac",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.neighbors.range_p_low = 101.0,
        "neighbors.range_p_low",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.neighbors.range_p_high = 101.0,
        "neighbors.range_p_high",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.neighbors.range_p_low = 90.0,
        "neighbors.range_p_low",
        "must be < neighbors.range_p_high (got 90, limit 90)",
    );
    expect_bad(
        |c| c.neighbors.max_gain = 0.0,
        "neighbors.max_gain",
        "must be >= 1 (got 0)",
    );

    expect_bad(
        |c| c.fill.min_train_rows = 0,
        "fill.min_train_rows",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.local_min_count = 0,
        "fill.local_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.range_min_n = 0,
        "fill.range_min_n",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.flat_min_run = 0,
        "fill.flat_min_run",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.plateau_window = 0,
        "fill.plateau_window",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.plateau_min_count = 0,
        "fill.plateau_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.quiet_min_count = 0,
        "fill.quiet_min_count",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.line_min_rows = 0,
        "fill.line_min_rows",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.holdout_n_wells = 0,
        "fill.holdout_n_wells",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.holdout_min_samples = 0,
        "fill.holdout_min_samples",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.holdout_blocks = 0,
        "fill.holdout_blocks",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.holdout_min_block = 0,
        "fill.holdout_min_block",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.holdout_max_attempts = 0,
        "fill.holdout_max_attempts",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.dropout_frac = 2.0,
        "fill.dropout_frac",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.fill.dropout_layer_prob = 2.0,
        "fill.dropout_layer_prob",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.fill.holdout_frac = 0.0,
        "fill.holdout_frac",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.fill.saturation_frac = 0.0,
        "fill.saturation_frac",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.fill.range_p_low = 101.0,
        "fill.range_p_low",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.fill.range_p_high = 101.0,
        "fill.range_p_high",
        "must be in [0, 100] (got 101)",
    );
    expect_bad(
        |c| c.fill.range_p_low = 90.0,
        "fill.range_p_low",
        "must be < fill.range_p_high (got 90, limit 90)",
    );
    expect_bad(
        |c| c.fill.max_gain = 0.0,
        "fill.max_gain",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.min_span_frac = -1.0,
        "fill.min_span_frac",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.degenerate_span = -1.0,
        "fill.degenerate_span",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.near_const_rel = -1.0,
        "fill.near_const_rel",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.near_const_abs = -1.0,
        "fill.near_const_abs",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.plateau_std_frac = -1.0,
        "fill.plateau_std_frac",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.quiet_std_frac = -1.0,
        "fill.quiet_std_frac",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.exact_flat_tol = -1.0,
        "fill.exact_flat_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.saturation_rel_tol = -1.0,
        "fill.saturation_rel_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.line_rel_tol = -1.0,
        "fill.line_rel_tol",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.scale_floor = 0.0,
        "fill.scale_floor",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.fill.holdout_mode = "bad".to_string(),
        "fill.holdout_mode",
        "must be one of \"interval\", \"whole_log\" (got \"bad\")",
    );

    expect_bad(
        |c| c.fill.lightgbm.num_iterations = 0,
        "fill.lightgbm.num_iterations",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.max_depth = 0,
        "fill.lightgbm.max_depth",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.min_data_in_leaf = 0,
        "fill.lightgbm.min_data_in_leaf",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.bin_construct_sample_cnt = 0,
        "fill.lightgbm.bin_construct_sample_cnt",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.num_threads = 0,
        "fill.lightgbm.num_threads",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.num_leaves = 1,
        "fill.lightgbm.num_leaves",
        "must be >= 2 (got 1)",
    );
    expect_bad(
        |c| c.fill.lightgbm.max_bin = 1,
        "fill.lightgbm.max_bin",
        "must be >= 2 (got 1)",
    );
    expect_bad(
        |c| c.fill.lightgbm.max_bin_wide = 1,
        "fill.lightgbm.max_bin_wide",
        "must be >= 2 (got 1)",
    );
    expect_bad(
        |c| c.fill.lightgbm.learning_rate = 0.0,
        "fill.lightgbm.learning_rate",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.feature_fraction = 0.0,
        "fill.lightgbm.feature_fraction",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.bagging_fraction = 0.0,
        "fill.lightgbm.bagging_fraction",
        "must be in (0, 1] (got 0)",
    );
    expect_bad(
        |c| c.fill.lightgbm.lambda_l1 = -1.0,
        "fill.lightgbm.lambda_l1",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.lightgbm.lambda_l2 = -1.0,
        "fill.lightgbm.lambda_l2",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.lightgbm.min_gain_to_split = -1.0,
        "fill.lightgbm.min_gain_to_split",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.lightgbm.min_sum_hessian_in_leaf = -1.0,
        "fill.lightgbm.min_sum_hessian_in_leaf",
        "must be >= 0 (got -1)",
    );

    expect_bad(
        |c| c.fill.ladder.real_donor_max_n = 0,
        "fill.ladder.real_donor_max_n",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.donor_max_n = 0,
        "fill.ladder.donor_max_n",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.anchor_rows = 0,
        "fill.ladder.anchor_rows",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.exact_flat_min_rows = 0,
        "fill.ladder.exact_flat_min_rows",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.crosslog_min_rows = 0,
        "fill.ladder.crosslog_min_rows",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.texture_window = 0,
        "fill.ladder.texture_window",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.scale_min_measured = 0,
        "fill.ladder.scale_min_measured",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.donor_max_dist_m = 0.0,
        "fill.ladder.donor_max_dist_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.tilt_max_scale = 0.0,
        "fill.ladder.tilt_max_scale",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.crosslog_window_ft = 0.0,
        "fill.ladder.crosslog_window_ft",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.scale_fallback = 0.0,
        "fill.ladder.scale_fallback",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.fill.ladder.anchor_tol_scale = -1.0,
        "fill.ladder.anchor_tol_scale",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.ladder.guard_quiet_frac = -1.0,
        "fill.ladder.guard_quiet_frac",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.ladder.crosslog_min_std = -1.0,
        "fill.ladder.crosslog_min_std",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.fill.ladder.physical_max_out_frac = 2.0,
        "fill.ladder.physical_max_out_frac",
        "must be in [0, 1] (got 2)",
    );
    expect_bad(
        |c| c.fill.ladder.crosslog_min_r2 = 2.0,
        "fill.ladder.crosslog_min_r2",
        "must be in [0, 1] (got 2)",
    );

    expect_bad(
        |c| c.correlate.graph.k_graph = 0,
        "correlate.graph.k_graph",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.correlate.graph.max_edge_m = 0.0,
        "correlate.graph.max_edge_m",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.correlate.solve.knot_step_ft = 0.0,
        "correlate.solve.knot_step_ft",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.correlate.solve.lambda_ridge = 0.0,
        "correlate.solve.lambda_ridge",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.correlate.solve.tukey_c = 0.0,
        "correlate.solve.tukey_c",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.correlate.solve.match_step = 0,
        "correlate.solve.match_step",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.correlate.solve.lambda_smooth = -1.0,
        "correlate.solve.lambda_smooth",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.correlate.tops.k_check = 0,
        "correlate.tops.k_check",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.correlate.tops.min_votes = 0,
        "correlate.tops.min_votes",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.correlate.tops.min_votes = 9,
        "correlate.tops.min_votes",
        "must be <= correlate.tops.k_check (got 9, limit 8)",
    );
    expect_bad(
        |c| c.correlate.tops.tol_ft = -1.0,
        "correlate.tops.tol_ft",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.correlate.tops.spread_mult = -1.0,
        "correlate.tops.spread_mult",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.correlate.tops.side_share = 0.0,
        "correlate.tops.side_share",
        "must be in (0, 1] (got 0)",
    );

    expect_bad(
        |c| c.subhorizons.n_grid = 1,
        "subhorizons.n_grid",
        "must be >= 2 (got 1)",
    );
    expect_bad(
        |c| c.subhorizons.min_wells = 0,
        "subhorizons.min_wells",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.subhorizons.min_segment_samples = 0,
        "subhorizons.min_segment_samples",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.subhorizons.kernel_half_width_mult = 0,
        "subhorizons.kernel_half_width_mult",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.subhorizons.min_spacing_samples = 0,
        "subhorizons.min_spacing_samples",
        "must be >= 1 (got 0)",
    );
    expect_bad(
        |c| c.subhorizons.link_tol_mult = 0.0,
        "subhorizons.link_tol_mult",
        "must be > 0 (got 0)",
    );
    expect_bad(
        |c| c.subhorizons.min_strength_api = -1.0,
        "subhorizons.min_strength_api",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.subhorizons.min_persistence = 6,
        "subhorizons.min_persistence",
        "must be in [1, 5] (got 6)",
    );
    expect_bad(
        |c| c.refresh.refresh_tol_ft = -1.0,
        "refresh.refresh_tol_ft",
        "must be >= 0 (got -1)",
    );
    expect_bad(
        |c| c.refresh.percentile = 101.0,
        "refresh.percentile",
        "must be in [0, 100] (got 101)",
    );
}

fn doc_row_rules() {
    expect_toml(
        "[[edges.headers]]\npath = \"\"\nkb_unit = \"ft\"\nxy_unit = \"m\"\n",
        "edges.headers[0].path",
        "must not be empty",
    );
    expect_toml(
        "[[edges.headers]]\npath = \"h.csv\"\nkb_unit = \"km\"\nxy_unit = \"m\"\n",
        "edges.headers[0].kb_unit",
        "must be one of \"ft\", \"m\" (got \"km\")",
    );
    expect_toml(
        "[[edges.headers]]\npath = \"h.csv\"\nkb_unit = \"ft\"\ncrs = \"\"\n",
        "edges.headers[0].crs",
        "must not be empty when set",
    );
    expect_toml(
        "[[edges.headers]]\npath = \"h.csv\"\nkb_unit = \"ft\"\n",
        "edges.headers[0].crs",
        "exactly one of crs and xy_unit must be set",
    );
    expect_toml(
        "[[edges.headers]]\npath = \"h.csv\"\nkb_unit = \"ft\"\ncrs = \"EPSG:32613\"\nxy_unit = \"m\"\n",
        "edges.headers[0].crs",
        "exactly one of crs and xy_unit must be set",
    );
    expect_toml(
        "[[edges.headers]]\npath = \"h.csv\"\nkb_unit = \"ft\"\nxy_unit = \"km\"\n",
        "edges.headers[0].xy_unit",
        "must be one of \"m\", \"ft\", \"us-ft\" (got \"km\")",
    );
    expect_toml(
        "[[edges.tops]]\npath = \"\"\ndepth_ref = \"md\"\ndepth_unit = \"ft\"\n",
        "edges.tops[0].path",
        "must not be empty",
    );
    expect_toml(
        "[[edges.tops]]\npath = \"t.csv\"\ndepth_ref = \"kb\"\ndepth_unit = \"ft\"\n",
        "edges.tops[0].depth_ref",
        "must be one of \"md\", \"tvd\", \"tvdss\", \"elev\" (got \"kb\")",
    );
    expect_toml(
        "[[edges.tops]]\npath = \"t.csv\"\ndepth_ref = \"md\"\ndepth_unit = \"km\"\n",
        "edges.tops[0].depth_unit",
        "must be one of \"ft\", \"m\" (got \"km\")",
    );
    expect_toml(
        "[[edges.tops]]\npath = \"t.csv\"\ndepth_ref = \"md\"\ndepth_unit = \"ft\"\nlayout = \"tall\"\n",
        "edges.tops[0].layout",
        "must be one of \"long\", \"wide\" (got \"tall\")",
    );
    expect_toml(
        "[[edges.tops]]\npath = \"t.csv\"\ndepth_ref = \"md\"\ndepth_unit = \"ft\"\nrepeat_pick = \"last\"\n",
        "edges.tops[0].repeat_pick",
        "must be one of \"shallowest\", \"first\", \"error\" (got \"last\")",
    );
    expect_toml(
        "[[edges.surveys]]\npath = \"\"\nmd_unit = \"ft\"\n",
        "edges.surveys[0].path",
        "must not be empty",
    );
    expect_toml(
        "[[edges.surveys]]\npath = \"s.csv\"\nmd_unit = \"km\"\n",
        "edges.surveys[0].md_unit",
        "must be one of \"ft\", \"m\" (got \"km\")",
    );
    expect_toml(
        "[edges.area]\npath = \"\"\n",
        "edges.area.path",
        "must not be empty",
    );
    expect_toml(
        "[ingest.alias_extra]\nXYZ = [\"A\"]\n",
        "ingest.alias_extra.XYZ",
        "unknown log name \"XYZ\"",
    );
    expect_toml(
        "[qc.rescue_factors]\nXYZ = 2.0\n",
        "qc.rescue_factors.XYZ",
        "unknown log name \"XYZ\"",
    );
    expect_toml(
        "[normalize]\nlogs = []\n",
        "normalize.logs",
        "must not be empty",
    );
    expect_toml(
        "[normalize]\nlogs = [\"RHOB\"]\n",
        "normalize.logs",
        "RHOB is not allowed (RHOB, DTC and RESD are not normalized)",
    );
    expect_toml(
        "[normalize]\nlogs = [\"DTC\"]\n",
        "normalize.logs",
        "DTC is not allowed (RHOB, DTC and RESD are not normalized)",
    );
    expect_toml(
        "[normalize]\nlogs = [\"RESD\"]\n",
        "normalize.logs",
        "RESD is not allowed (RHOB, DTC and RESD are not normalized)",
    );
    expect_toml(
        "[ingest.las_overrides]\n\"las/a.las\" = \"W1\"\n",
        "ingest.las_overrides",
        "key \"las/a.las\" must be a bare LAS file name (no folder)",
    );
    expect_toml(
        "[ingest.las_overrides]\n\"a.las\" = \" \"\n",
        "ingest.las_overrides.a.las",
        "well key must not be empty",
    );
    expect_toml("[fill]\norder = []\n", "fill.order", "must not be empty");
    expect_toml(
        "[fill]\norder = [\"GR\", \"GR\"]\n",
        "fill.order",
        "duplicate log GR",
    );
    expect_toml(
        "[fill]\nlocal_windows_ft = [2.5]\n",
        "fill.local_windows_ft",
        "must have exactly 2 entries (got 1)",
    );
    expect_toml(
        "[fill]\nlocal_windows_ft = [2.5, 0.0]\n",
        "fill.local_windows_ft[1]",
        "must be > 0 (got 0)",
    );
    expect_toml(
        "[subhorizons]\nscales = []\n",
        "subhorizons.scales",
        "must not be empty",
    );
    expect_toml(
        "[subhorizons]\nscales = [1, 1, 4]\n",
        "subhorizons.scales",
        "must be strictly ascending with every entry >= 1 (got [1, 1, 4])",
    );
    expect_toml(
        "[subhorizons]\nmin_persistence = 1\nscales = [0, 2]\n",
        "subhorizons.scales",
        "must be strictly ascending with every entry >= 1 (got [0, 2])",
    );
}

#[test]
fn template_equals_default() {
    let cfg = from_toml_str(&template_toml()).unwrap();
    assert_eq!(cfg, ProjectConfig::default_with_crs("EPSG:32614"));
}

#[test]
fn template_lists_every_key() {
    let template_val: toml::Value = toml::from_str(&template_toml()).unwrap();
    let default_val = toml::Value::try_from(ProjectConfig::default_with_crs("EPSG:32614")).unwrap();
    let template_paths = collect_paths(&template_val);
    let default_paths = collect_paths(&default_val);
    let missing: BTreeSet<_> = default_paths.difference(&template_paths).cloned().collect();
    let extra: BTreeSet<_> = template_paths.difference(&default_paths).cloned().collect();
    let expect: BTreeSet<_> = ["edges.headers", "edges.tops", "edges.surveys", "edges.pts"]
        .into_iter()
        .map(str::to_string)
        .collect();
    assert_eq!(missing, expect, "extra template paths: {extra:?}");
    assert!(extra.is_empty(), "{extra:?}");
}

fn collect_paths(value: &toml::Value) -> BTreeSet<String> {
    fn walk(prefix: &str, value: &toml::Value, out: &mut BTreeSet<String>) {
        match value {
            toml::Value::Table(table) if table.is_empty() => {
                if !prefix.is_empty() {
                    out.insert(prefix.to_string());
                }
            }
            toml::Value::Table(table) => {
                for (key, child) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(&path, child, out);
                }
            }
            _ => {
                if !prefix.is_empty() {
                    out.insert(prefix.to_string());
                }
            }
        }
    }
    let mut out = BTreeSet::new();
    walk("", value, &mut out);
    out
}

#[test]
fn toml_round_trip_default() {
    let cfg = ProjectConfig::default_with_crs("EPSG:32614");
    let text = toml::to_string(&cfg).unwrap();
    assert_eq!(from_toml_str(&text).unwrap(), cfg);
}

#[test]
fn json_round_trip_default() {
    let cfg = ProjectConfig::default_with_crs("EPSG:32614");
    let text = serde_json::to_string(&cfg).unwrap();
    let back: ProjectConfig = serde_json::from_str(&text).unwrap();
    assert_eq!(back, cfg);
}

#[test]
fn stage_sections() {
    let cfg = ProjectConfig::default_with_crs("EPSG:32614");
    assert_eq!(
        cfg.stage_section("refresh").unwrap(),
        serde_json::json!({"refresh_tol_ft": 5.0, "percentile": 95.0})
    );
    assert_eq!(
        cfg.stage_section("ingest").unwrap(),
        serde_json::to_value(&cfg.ingest).unwrap()
    );
    assert_eq!(
        cfg.stage_section("qc").unwrap(),
        serde_json::to_value(&cfg.qc).unwrap()
    );
    assert!(cfg.stage_section("qc").unwrap().get("jury").is_some());
    assert_eq!(
        cfg.stage_section("neighbors").unwrap(),
        serde_json::to_value(&cfg.neighbors).unwrap()
    );
    assert_eq!(
        cfg.stage_section("fill").unwrap(),
        serde_json::to_value(&cfg.fill).unwrap()
    );
    assert_eq!(
        cfg.stage_section("subhorizons").unwrap(),
        serde_json::to_value(&cfg.subhorizons).unwrap()
    );
    assert_eq!(
        cfg.stage_section("export").unwrap(),
        serde_json::to_value(&cfg.export).unwrap()
    );
    let rough = cfg.stage_section("rough_align").unwrap();
    assert_eq!(
        rough,
        serde_json::json!({
            "align": serde_json::to_value(&cfg.align).unwrap(),
            "dtw": serde_json::to_value(&cfg.dtw).unwrap(),
        })
    );
    let normalize = cfg.stage_section("normalize").unwrap();
    let normalize_keys: BTreeSet<_> = normalize.as_object().unwrap().keys().cloned().collect();
    assert_eq!(
        normalize_keys,
        BTreeSet::from(["normalize".to_string(), "qc".to_string()])
    );
    let correlate = cfg.stage_section("correlate").unwrap();
    let correlate_keys: BTreeSet<_> = correlate.as_object().unwrap().keys().cloned().collect();
    assert_eq!(
        correlate_keys,
        BTreeSet::from(["correlate".to_string(), "dtw".to_string()])
    );
    match cfg.stage_section("qc2") {
        Err(SxError::InvalidInput(_)) => {}
        other => panic!("{other:?}"),
    }

    let mut threads = cfg.clone();
    threads.runtime.threads = 4;
    let mut rate = cfg.clone();
    rate.fill.lightgbm.learning_rate = 0.1;
    for stage in [
        "ingest",
        "qc",
        "rough_align",
        "normalize",
        "neighbors",
        "fill",
        "correlate",
        "subhorizons",
        "refresh",
        "export",
    ] {
        assert_eq!(
            cfg.stage_section(stage).unwrap(),
            threads.stage_section(stage).unwrap(),
            "{stage}"
        );
        let changed = cfg.stage_section(stage).unwrap() != rate.stage_section(stage).unwrap();
        assert_eq!(changed, stage == "fill", "{stage}");
    }
}

#[test]
fn load_project_reads_file() {
    let path = std::env::temp_dir().join(format!("sx-core-config-{}.toml", std::process::id()));
    let _guard = TempFile(path.clone());
    std::fs::write(&path, template_toml()).unwrap();
    assert_eq!(
        load_project(&path).unwrap(),
        ProjectConfig::default_with_crs("EPSG:32614")
    );
    std::fs::write(&path, "[qc").unwrap();
    match load_project(&path) {
        Err(SxError::Config(message)) => {
            assert!(message.starts_with("parse error at line"), "{message}")
        }
        other => panic!("{other:?}"),
    }
    let missing =
        std::env::temp_dir().join(format!("sx-core-config-missing-{}", std::process::id()));
    match load_project(&missing) {
        Err(SxError::Io { .. }) => {}
        other => panic!("{other:?}"),
    }
}

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn log_lists() {
    let cfg = from_toml_str(
        "[project]\ncrs = \"EPSG:32614\"\n[fill]\norder = [\"gr\", \"DTC\", \"RHOB\", \"NPHI\", \"RESD\", \"PEF\"]\n",
    )
    .unwrap();
    assert_eq!(
        cfg.fill.order,
        vec![
            LogKind::Gr,
            LogKind::Dtc,
            LogKind::Rhob,
            LogKind::Nphi,
            LogKind::Resd,
            LogKind::Pef
        ]
    );
    assert_eq!(
        serde_json::to_value(&cfg.fill).unwrap()["order"],
        serde_json::json!(["GR", "DTC", "RHOB", "NPHI", "RESD", "PEF"])
    );
    match from_toml_str("[project]\ncrs = \"EPSG:32614\"\n[fill]\norder = [\"XYZ\"]\n") {
        Err(SxError::Config(message)) => assert!(message.contains("XYZ"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn resolve_path_cases() {
    let dir = Path::new("/proj");
    assert_eq!(
        resolve_path(dir, Path::new("a/b.csv")),
        PathBuf::from("/proj/a/b.csv")
    );
    assert_eq!(
        resolve_path(dir, Path::new("headers.xlsx")),
        PathBuf::from("/proj/headers.xlsx")
    );
    assert_eq!(
        resolve_path(dir, Path::new("/abs/x.toml")),
        PathBuf::from("/abs/x.toml")
    );
}

#[test]
fn closed_caliper_mode_switch() {
    let cfg = ProjectConfig::default_with_crs("EPSG:32614");
    assert_eq!(cfg.qc.closed_caliper.mode, "rescue");
    let rej =
        from_toml_str("[project]\ncrs = \"EPSG:32614\"\n[qc.closed_caliper]\nmode = \"reject\"\n")
            .unwrap();
    assert_eq!(rej.qc.closed_caliper.mode, "reject");
    expect_bad(
        |c| c.qc.closed_caliper.mode = "shift".into(),
        "qc.closed_caliper.mode",
        "must be \"rescue\" or \"reject\" (got \"shift\")",
    );
    expect_bad(
        |c| c.qc.closed_caliper.min_neighbors = 9,
        "qc.closed_caliper.min_neighbors",
        "must be <= k_neighbors (8)",
    );
}
