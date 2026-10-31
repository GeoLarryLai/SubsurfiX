// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Rule table 4.5. The first failure is returned.

use std::collections::BTreeSet;
use std::fmt::Display;
use std::path::Path;

use super::dtw::SpanParams;
use super::project::ProjectConfig;
use crate::types::LogKind;
use crate::{SxError, SxResult};

/// Check every rule of table 4.5, in table order; the first failure is returned.
///
/// The error is `SxError::Config` and starts with the dotted key (`qc.tail_pct`,
/// `edges.headers[0].kb_unit`). Floats must be finite before any other bound.
/// Counts are dimensionless. Distances are metres, depths feet, unless the key
/// says otherwise. An empty config is not special: the defaults pass, and a
/// broken value is `Config` rather than `EmptyInput`.
pub fn validate(cfg: &ProjectConfig) -> SxResult<()> {
    validate_seed(cfg)?;
    validate_project(cfg)?;
    validate_edges(cfg)?;
    validate_ingest(cfg)?;
    validate_qc(cfg)?;
    validate_jury(cfg)?;
    validate_closed_caliper(cfg)?;
    validate_align(cfg)?;
    validate_dtw(cfg)?;
    validate_normalize(cfg)?;
    validate_nphi_offset(cfg)?;
    validate_neighbors(cfg)?;
    validate_fill(cfg)?;
    validate_lightgbm(cfg)?;
    validate_ladder(cfg)?;
    validate_correlate(cfg)?;
    validate_subhorizons(cfg)?;
    validate_refresh(cfg)?;
    validate_propagation(cfg)?;
    Ok(())
}

fn validate_propagation(cfg: &ProjectConfig) -> SxResult<()> {
    let p = &cfg.propagation;
    crate::sim::PropagationModel::parse(&p.method).map_err(|_| {
        bad(
            "propagation.method",
            format!("must be \"qrf\" or \"lightgbm\" (got {:?})", p.method),
        )
    })?;
    if p.quantiles.is_empty() || p.quantiles.iter().any(|q| !(*q > 0.0 && *q < 1.0)) {
        return Err(bad(
            "propagation.quantiles",
            format!("need one or more levels in (0, 1) (got {:?})", p.quantiles),
        ));
    }
    check_count("propagation.qrf_trees", p.qrf_trees)?;
    check_count("propagation.qrf_min_leaf", p.qrf_min_leaf)?;
    check_count("propagation.lgbm_iterations", p.lgbm_iterations)?;
    if p.conformal_folds < 2 {
        return Err(bad(
            "propagation.conformal_folds",
            format!("must be at least 2 (got {})", p.conformal_folds),
        ));
    }
    Ok(())
}

fn bad(path: &str, reason: String) -> SxError {
    SxError::Config(format!("{path}: {reason}"))
}

fn check_finite<T>(path: &str, value: T) -> SxResult<()>
where
    T: Into<f64> + Copy + Display,
{
    let as_float = value.into();
    if as_float.is_finite() {
        Ok(())
    } else {
        Err(bad(path, format!("must be finite (got {value})")))
    }
}

fn check_count(path: &str, value: usize) -> SxResult<()> {
    check_count_min(path, value, 1)
}

fn check_count_min(path: &str, value: usize, minimum: usize) -> SxResult<()> {
    if value >= minimum {
        Ok(())
    } else {
        Err(bad(path, format!("must be >= {minimum} (got {value})")))
    }
}

fn check_ge<T>(path: &str, value: T, limit: f64) -> SxResult<()>
where
    T: Into<f64> + Copy + Display,
{
    check_finite(path, value)?;
    if value.into() >= limit {
        Ok(())
    } else {
        Err(bad(path, format!("must be >= {limit} (got {value})")))
    }
}

fn check_gt<T>(path: &str, value: T, limit: f64) -> SxResult<()>
where
    T: Into<f64> + Copy + Display,
{
    check_finite(path, value)?;
    if value.into() > limit {
        Ok(())
    } else {
        Err(bad(path, format!("must be > {limit} (got {value})")))
    }
}

fn check_range<T>(path: &str, value: T, low: f64, high: f64) -> SxResult<()>
where
    T: Into<f64> + Copy + Display,
{
    check_finite(path, value)?;
    let as_float = value.into();
    if as_float >= low && as_float <= high {
        Ok(())
    } else {
        Err(bad(
            path,
            format!("must be in [{low}, {high}] (got {value})"),
        ))
    }
}

fn check_open_closed<T>(path: &str, value: T, low: f64, high: f64) -> SxResult<()>
where
    T: Into<f64> + Copy + Display,
{
    check_finite(path, value)?;
    let as_float = value.into();
    if as_float > low && as_float <= high {
        Ok(())
    } else {
        Err(bad(
            path,
            format!("must be in ({low}, {high}] (got {value})"),
        ))
    }
}

fn check_open_open<T>(path: &str, value: T, low: f64, high: f64) -> SxResult<()>
where
    T: Into<f64> + Copy + Display,
{
    check_finite(path, value)?;
    let as_float = value.into();
    if as_float > low && as_float < high {
        Ok(())
    } else {
        Err(bad(
            path,
            format!("must be in ({low}, {high}) (got {value})"),
        ))
    }
}

fn check_increasing(path: &str, pair: [f64; 2]) -> SxResult<()> {
    if !pair[0].is_finite() {
        return Err(bad(path, format!("must be finite (got {})", pair[0])));
    }
    if !pair[1].is_finite() {
        return Err(bad(path, format!("must be finite (got {})", pair[1])));
    }
    if pair[0] < pair[1] {
        Ok(())
    } else {
        Err(bad(
            path,
            format!(
                "must be strictly increasing (got [{}, {}])",
                pair[0], pair[1]
            ),
        ))
    }
}

fn check_one_of(path: &str, value: &str, allowed: &[&str]) -> SxResult<()> {
    if allowed.contains(&value) {
        return Ok(());
    }
    let list = allowed
        .iter()
        .map(|item| format!("\"{item}\""))
        .collect::<Vec<_>>()
        .join(", ");
    Err(bad(
        path,
        format!("must be one of {list} (got \"{value}\")"),
    ))
}

fn rel_f64(path: &str, value: f64, op: &str, other: &str, limit: f64, pass: bool) -> SxResult<()> {
    check_finite(path, value)?;
    if pass {
        Ok(())
    } else {
        Err(bad(
            path,
            format!("must be {op} {other} (got {value}, limit {limit})"),
        ))
    }
}

fn rel_usize(
    path: &str,
    value: usize,
    op: &str,
    other: &str,
    limit: usize,
    pass: bool,
) -> SxResult<()> {
    if pass {
        Ok(())
    } else {
        Err(bad(
            path,
            format!("must be {op} {other} (got {value}, limit {limit})"),
        ))
    }
}

fn check_nonempty_path(path: &str, value: &Path) -> SxResult<()> {
    if value.as_os_str().is_empty() {
        Err(bad(path, "must not be empty".into()))
    } else {
        Ok(())
    }
}

fn validate_seed(cfg: &ProjectConfig) -> SxResult<()> {
    const SEED_LIMIT: u64 = 9_223_372_036_854_775_807;
    if cfg.seed > SEED_LIMIT {
        return Err(bad(
            "seed",
            format!("must be <= {SEED_LIMIT} (got {})", cfg.seed),
        ));
    }
    Ok(())
}

fn validate_project(cfg: &ProjectConfig) -> SxResult<()> {
    if cfg.project.crs.trim().is_empty() {
        return Err(bad("project.crs", "must not be empty".into()));
    }
    Ok(())
}

fn validate_edges(cfg: &ProjectConfig) -> SxResult<()> {
    let edges = &cfg.edges;
    for (index, header) in edges.headers.iter().enumerate() {
        check_nonempty_path(&format!("edges.headers[{index}].path"), &header.path)?;
    }
    for (index, header) in edges.headers.iter().enumerate() {
        check_one_of(
            &format!("edges.headers[{index}].kb_unit"),
            &header.kb_unit,
            &["ft", "m"],
        )?;
    }
    for (index, header) in edges.headers.iter().enumerate() {
        if let Some(crs) = &header.crs {
            if crs.trim().is_empty() {
                return Err(bad(
                    &format!("edges.headers[{index}].crs"),
                    "must not be empty when set".into(),
                ));
            }
        }
    }
    for (index, header) in edges.headers.iter().enumerate() {
        let crs_set = header.crs.is_some();
        let xy_set = header.xy_unit.is_some();
        if crs_set == xy_set {
            return Err(bad(
                &format!("edges.headers[{index}].crs"),
                "exactly one of crs and xy_unit must be set".into(),
            ));
        }
    }
    for (index, header) in edges.headers.iter().enumerate() {
        if let Some(unit) = &header.xy_unit {
            check_one_of(
                &format!("edges.headers[{index}].xy_unit"),
                unit,
                &["m", "ft", "us-ft"],
            )?;
        }
    }
    for (index, tops) in edges.tops.iter().enumerate() {
        check_nonempty_path(&format!("edges.tops[{index}].path"), &tops.path)?;
    }
    for (index, tops) in edges.tops.iter().enumerate() {
        check_one_of(
            &format!("edges.tops[{index}].depth_ref"),
            &tops.depth_ref,
            &["md", "tvd", "tvdss", "elev"],
        )?;
    }
    for (index, tops) in edges.tops.iter().enumerate() {
        check_one_of(
            &format!("edges.tops[{index}].depth_unit"),
            &tops.depth_unit,
            &["ft", "m"],
        )?;
    }
    for (index, tops) in edges.tops.iter().enumerate() {
        check_one_of(
            &format!("edges.tops[{index}].layout"),
            &tops.layout,
            &["long", "wide"],
        )?;
    }
    for (index, tops) in edges.tops.iter().enumerate() {
        check_one_of(
            &format!("edges.tops[{index}].repeat_pick"),
            &tops.repeat_pick,
            &["shallowest", "first", "error"],
        )?;
    }
    for (index, survey) in edges.surveys.iter().enumerate() {
        check_nonempty_path(&format!("edges.surveys[{index}].path"), &survey.path)?;
    }
    for (index, survey) in edges.surveys.iter().enumerate() {
        check_one_of(
            &format!("edges.surveys[{index}].md_unit"),
            &survey.md_unit,
            &["ft", "m"],
        )?;
    }
    for (index, pts) in edges.pts.iter().enumerate() {
        check_nonempty_path(&format!("edges.pts[{index}].path"), &pts.path)?;
        if pts.well.is_some() == pts.key_col.is_some() {
            return Err(SxError::Config(format!(
                "edges.pts[{index}]: set exactly one of well or key_col"
            )));
        }
    }
    if let Some(area) = &edges.area {
        check_nonempty_path("edges.area.path", &area.path)?;
    }
    Ok(())
}

fn validate_ingest(cfg: &ProjectConfig) -> SxResult<()> {
    check_ge("ingest.max_gap_ft", cfg.ingest.max_gap_ft, 0.0)?;
    for key in cfg.ingest.alias_extra.keys() {
        if LogKind::from_name(key).is_err() {
            return Err(bad(
                &format!("ingest.alias_extra.{key}"),
                format!("unknown log name {key:?}"),
            ));
        }
    }
    for (file, well) in &cfg.ingest.las_overrides {
        if file.trim().is_empty() || file.contains(['/', '\\']) {
            return Err(bad(
                "ingest.las_overrides",
                format!("key {file:?} must be a bare LAS file name (no folder)"),
            ));
        }
        if well.trim().is_empty() {
            return Err(bad(
                &format!("ingest.las_overrides.{file}"),
                "well key must not be empty".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_qc(cfg: &ProjectConfig) -> SxResult<()> {
    let qc = &cfg.qc;
    check_count("qc.constant_min_count", qc.constant_min_count)?;
    check_count("qc.tail_min_count", qc.tail_min_count)?;
    check_count("qc.rescue_min_count", qc.rescue_min_count)?;
    check_count("qc.rescue_ref_wells", qc.rescue_ref_wells)?;
    check_count("qc.rescue_min_ref_wells", qc.rescue_min_ref_wells)?;
    check_count("qc.repeat_min_len", qc.repeat_min_len)?;
    check_count("qc.repeat_min_count", qc.repeat_min_count)?;
    check_count("qc.plateau_window", qc.plateau_window)?;
    check_count("qc.plateau_window_min", qc.plateau_window_min)?;
    check_count("qc.plateau_min_len", qc.plateau_min_len)?;
    check_count("qc.plateau_min_count", qc.plateau_min_count)?;
    check_count("qc.run_min_len", qc.run_min_len)?;
    check_count("qc.stats_min_count", qc.stats_min_count)?;
    check_count("qc.fleet_min_wells", qc.fleet_min_wells)?;
    check_count("qc.post_flat_min_count", qc.post_flat_min_count)?;
    check_count("qc.near_count", qc.near_count)?;
    check_count("qc.near_min", qc.near_min)?;
    check_count("qc.ring_first", qc.ring_first)?;
    check_count("qc.ring_last", qc.ring_last)?;
    check_count("qc.ring_min", qc.ring_min)?;
    check_count("qc.jsd_k", qc.jsd_k)?;
    check_count("qc.jsd_neighbors", qc.jsd_neighbors)?;
    check_count("qc.jsd_min_neighbors", qc.jsd_min_neighbors)?;
    check_count("qc.jsd_min_wells", qc.jsd_min_wells)?;
    check_count("qc.jsd_max_count", qc.jsd_max_count)?;
    check_count("qc.pileup_min_len", qc.pileup_min_len)?;

    check_ge("qc.constant_rel_tol", qc.constant_rel_tol, 0.0)?;
    check_ge("qc.tail_pad_frac", qc.tail_pad_frac, 0.0)?;
    check_ge("qc.rescue_min_cv", qc.rescue_min_cv, 0.0)?;
    check_ge("qc.repeat_rel_tol", qc.repeat_rel_tol, 0.0)?;
    check_ge("qc.plateau_std_share", qc.plateau_std_share, 0.0)?;
    check_ge("qc.mad_floor", qc.mad_floor, 0.0)?;
    check_ge("qc.lowvar_ratio", qc.lowvar_ratio, 0.0)?;
    check_ge("qc.post_flat_ratio", qc.post_flat_ratio, 0.0)?;
    check_ge("qc.dev_limit", qc.dev_limit, 0.0)?;
    check_ge("qc.jsd_mad_mult", qc.jsd_mad_mult, 0.0)?;
    check_ge("qc.jsd_floor", qc.jsd_floor, 0.0)?;
    check_ge("qc.gr_span_frac", qc.gr_span_frac, 0.0)?;
    check_ge("qc.resd_decade_log10", qc.resd_decade_log10, 0.0)?;
    check_ge("qc.pileup_min_share", qc.pileup_min_share, 0.0)?;
    check_ge("qc.pileup_rel_tol", qc.pileup_rel_tol, 0.0)?;

    check_range("qc.tail_pct[0]", qc.tail_pct[0], 0.0, 100.0)?;
    check_range("qc.tail_pct[1]", qc.tail_pct[1], 0.0, 100.0)?;
    check_range("qc.stats_pct[0]", qc.stats_pct[0], 0.0, 100.0)?;
    check_range("qc.stats_pct[1]", qc.stats_pct[1], 0.0, 100.0)?;

    check_gt(
        "qc.rescue_candidate_band[0]",
        qc.rescue_candidate_band[0],
        0.0,
    )?;
    check_gt("qc.rescue_accept_band[0]", qc.rescue_accept_band[0], 0.0)?;
    check_gt("qc.unit_band[0]", qc.unit_band[0], 0.0)?;

    check_increasing("qc.tail_pct", qc.tail_pct)?;
    check_increasing("qc.rescue_candidate_band", qc.rescue_candidate_band)?;
    check_increasing("qc.rescue_accept_band", qc.rescue_accept_band)?;
    check_increasing("qc.stats_pct", qc.stats_pct)?;
    check_increasing("qc.unit_band", qc.unit_band)?;

    rel_usize(
        "qc.ring_first",
        qc.ring_first,
        "<=",
        "qc.ring_last",
        qc.ring_last,
        qc.ring_first <= qc.ring_last,
    )?;

    for key in qc.rescue_factors.keys() {
        if LogKind::from_name(key).is_err() {
            return Err(bad(
                &format!("qc.rescue_factors.{key}"),
                format!("unknown log name {key:?}"),
            ));
        }
    }
    for (key, value) in &qc.rescue_factors {
        if !(value.is_finite() && *value > 0.0 && *value != 1.0) {
            return Err(bad(
                &format!("qc.rescue_factors.{key}"),
                format!("must be finite, > 0 and != 1 (got {value})"),
            ));
        }
    }
    if !(qc.low_density_rhob.is_finite() && qc.low_density_rhob >= 0.0) {
        return Err(bad(
            "qc.low_density_rhob",
            format!("must be finite and >= 0 (got {})", qc.low_density_rhob),
        ));
    }
    for key in &qc.badhole_logs {
        if LogKind::from_name(key).is_err() {
            return Err(bad("qc.badhole_logs", format!("unknown log name {key:?}")));
        }
    }
    for (name, v) in [
        ("qc.cal_open_min_in", qc.cal_open_min_in),
        ("qc.cal_undergauge_in", qc.cal_undergauge_in),
        ("qc.cal_washout_in", qc.cal_washout_in),
        ("qc.drho_limit", qc.drho_limit),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(bad(name, format!("must be finite and > 0 (got {v})")));
        }
    }
    for (table, map) in [
        ("qc.plausible_range", &qc.plausible_range),
        ("qc.plausible_median", &qc.plausible_median),
    ] {
        for (key, [lo, hi]) in map {
            if LogKind::from_name(key).is_err() {
                return Err(bad(
                    &format!("{table}.{key}"),
                    format!("unknown log name {key:?}"),
                ));
            }
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(bad(
                    &format!("{table}.{key}"),
                    "must be two finite values, strictly increasing".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_closed_caliper(cfg: &ProjectConfig) -> SxResult<()> {
    let c = &cfg.qc.closed_caliper;
    if c.mode != "rescue" && c.mode != "reject" {
        return Err(bad(
            "qc.closed_caliper.mode",
            format!("must be \"rescue\" or \"reject\" (got {:?})", c.mode),
        ));
    }
    check_count("qc.closed_caliper.k_neighbors", c.k_neighbors)?;
    check_count("qc.closed_caliper.min_neighbors", c.min_neighbors)?;
    check_count("qc.closed_caliper.min_overlap", c.min_overlap)?;
    check_count("qc.closed_caliper.despike_window", c.despike_window)?;
    check_count("qc.closed_caliper.mad_window", c.mad_window)?;
    if c.min_neighbors > c.k_neighbors {
        return Err(bad(
            "qc.closed_caliper.min_neighbors",
            format!("must be <= k_neighbors ({})", c.k_neighbors),
        ));
    }
    for (name, v) in [
        ("qc.closed_caliper.bin_ft", c.bin_ft),
        ("qc.closed_caliper.drho_limit", c.drho_limit),
        ("qc.closed_caliper.min_rhob", c.min_rhob),
        ("qc.closed_caliper.mad_k", c.mad_k),
        ("qc.closed_caliper.spike_floor", c.spike_floor),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(bad(name, format!("must be finite and > 0 (got {v})")));
        }
    }
    let [lo, hi] = c.envelope_pct;
    if !(lo.is_finite() && hi.is_finite() && 0.0 <= lo && lo < hi && hi <= 100.0) {
        return Err(bad(
            "qc.closed_caliper.envelope_pct",
            format!("must satisfy 0 <= lo < hi <= 100 (got [{lo}, {hi}])"),
        ));
    }
    Ok(())
}

fn validate_jury(cfg: &ProjectConfig) -> SxResult<()> {
    let jury = &cfg.qc.jury;
    check_count("qc.jury.min_count", jury.min_count)?;
    check_count("qc.jury.near_jurors", jury.near_jurors)?;
    check_count("qc.jury.max_jurors", jury.max_jurors)?;
    check_count("qc.jury.quorum", jury.quorum)?;
    check_count("qc.jury.window_min", jury.window_min)?;
    check_count("qc.jury.hop_min", jury.hop_min)?;
    check_count("qc.jury.joint_floor", jury.joint_floor)?;
    check_count("qc.jury.min_streak", jury.min_streak)?;
    check_count("qc.jury.corr_min_count", jury.corr_min_count)?;

    check_gt("qc.jury.max_dist_m", jury.max_dist_m, 0.0)?;
    check_gt("qc.jury.dist_floor_m", jury.dist_floor_m, 0.0)?;

    check_ge("qc.jury.dist_power", jury.dist_power, 0.0)?;
    check_ge("qc.jury.flat_std_share", jury.flat_std_share, 0.0)?;

    check_range("qc.jury.path_min_quality", jury.path_min_quality, 0.0, 1.0)?;
    check_range("qc.jury.unaligned_weight", jury.unaligned_weight, 0.0, 1.0)?;

    check_gt("qc.jury.window_ft[0]", jury.window_ft[0], 0.0)?;
    check_gt("qc.jury.window_ft[1]", jury.window_ft[1], 0.0)?;

    check_open_closed("qc.jury.joint_share", jury.joint_share, 0.0, 1.0)?;
    check_open_closed("qc.jury.bad_share", jury.bad_share, 0.0, 1.0)?;

    check_range("qc.jury.anticorr_limit", jury.anticorr_limit, -1.0, 0.0)?;

    rel_usize(
        "qc.jury.quorum",
        jury.quorum,
        "<=",
        "qc.jury.near_jurors",
        jury.near_jurors,
        jury.quorum <= jury.near_jurors,
    )?;
    rel_usize(
        "qc.jury.near_jurors",
        jury.near_jurors,
        "<=",
        "qc.jury.max_jurors",
        jury.max_jurors,
        jury.near_jurors <= jury.max_jurors,
    )?;
    Ok(())
}

fn validate_align(cfg: &ProjectConfig) -> SxResult<()> {
    let align = &cfg.align;
    check_gt("align.pad_ft", align.pad_ft, 0.0)?;
    check_count("align.k_pairs", align.k_pairs)?;

    let tops = &align.tops;
    check_count("align.tops.k", tops.k)?;
    check_count("align.tops.quorum", tops.quorum)?;
    check_gt("align.tops.max_dist_m", tops.max_dist_m, 0.0)?;
    check_gt("align.tops.min_dist_m", tops.min_dist_m, 0.0)?;
    check_ge("align.tops.power", tops.power, 0.0)?;

    let pairs = &align.pairs;
    check_gt("align.pairs.min_span_ratio", pairs.min_span_ratio, 0.0)?;
    rel_f64(
        "align.pairs.max_span_ratio",
        pairs.max_span_ratio,
        ">",
        "align.pairs.min_span_ratio",
        pairs.min_span_ratio,
        pairs.max_span_ratio > pairs.min_span_ratio,
    )?;
    check_count_min("align.pairs.depth_nodes", pairs.depth_nodes, 2)?;
    check_count("align.pairs.depth_min_shared", pairs.depth_min_shared)?;
    check_gt("align.pairs.depth_band_frac", pairs.depth_band_frac, 0.0)?;
    check_range("align.pairs.depth_quality", pairs.depth_quality, 0.0, 1.0)?;
    check_range("align.pairs.fail_quality", pairs.fail_quality, 0.0, 1.0)?;
    check_range(
        "align.pairs.fail_quality_no_coord",
        pairs.fail_quality_no_coord,
        0.0,
        1.0,
    )?;

    let confidence = &align.confidence;
    check_range("align.confidence.observed", confidence.observed, 0.0, 1.0)?;
    check_range(
        "align.confidence.pinched_user",
        confidence.pinched_user,
        0.0,
        1.0,
    )?;
    check_range(
        "align.confidence.pinched_other",
        confidence.pinched_other,
        0.0,
        1.0,
    )?;
    check_range(
        "align.confidence.estimated_one",
        confidence.estimated_one,
        0.0,
        1.0,
    )?;
    check_range(
        "align.confidence.estimated_none",
        confidence.estimated_none,
        0.0,
        1.0,
    )?;
    check_range("align.confidence.pad_floor", confidence.pad_floor, 0.0, 1.0)?;
    Ok(())
}

fn validate_dtw(cfg: &ProjectConfig) -> SxResult<()> {
    let interval = &cfg.dtw.interval;
    check_count("dtw.interval.grid_budget", interval.grid_budget)?;
    check_count("dtw.interval.min_nodes", interval.min_nodes)?;
    check_count("dtw.interval.max_nodes", interval.max_nodes)?;
    check_count("dtw.interval.min_shared_floor", interval.min_shared_floor)?;
    check_count("dtw.interval.min_shared_cap", interval.min_shared_cap)?;
    check_count("dtw.interval.min_shared_div", interval.min_shared_div)?;
    check_count(
        "dtw.interval.min_channel_samples",
        interval.min_channel_samples,
    )?;
    check_count("dtw.interval.min_channel_nodes", interval.min_channel_nodes)?;
    check_gt("dtw.interval.band_frac", interval.band_frac, 0.0)?;
    check_ge("dtw.interval.gap_cost", interval.gap_cost, 0.0)?;
    check_ge("dtw.interval.max_gap_steps", interval.max_gap_steps, 0.0)?;
    check_ge("dtw.interval.min_span", interval.min_span, 0.0)?;
    check_range("dtw.interval.pct_low", interval.pct_low, 0.0, 100.0)?;
    check_range("dtw.interval.pct_high", interval.pct_high, 0.0, 100.0)?;
    rel_f64(
        "dtw.interval.pct_low",
        interval.pct_low,
        "<",
        "dtw.interval.pct_high",
        interval.pct_high,
        interval.pct_low < interval.pct_high,
    )?;
    check_range("dtw.interval.dtw_factor", interval.dtw_factor, 0.0, 1.0)?;
    check_range(
        "dtw.interval.linear_factor",
        interval.linear_factor,
        0.0,
        1.0,
    )?;
    check_gt(
        "dtw.interval.resd_floor_ohmm",
        interval.resd_floor_ohmm,
        0.0,
    )?;
    rel_usize(
        "dtw.interval.min_nodes",
        interval.min_nodes,
        "<=",
        "dtw.interval.max_nodes",
        interval.max_nodes,
        interval.min_nodes <= interval.max_nodes,
    )?;
    rel_usize(
        "dtw.interval.min_shared_floor",
        interval.min_shared_floor,
        "<=",
        "dtw.interval.min_shared_cap",
        interval.min_shared_cap,
        interval.min_shared_floor <= interval.min_shared_cap,
    )?;
    validate_span("dtw.full", &cfg.dtw.full)?;
    validate_span("dtw.window", &cfg.dtw.window)?;
    Ok(())
}

fn validate_span(prefix: &str, span: &SpanParams) -> SxResult<()> {
    check_count(&format!("{prefix}.min_nodes"), span.min_nodes)?;
    check_count(&format!("{prefix}.max_nodes"), span.max_nodes)?;
    check_gt(&format!("{prefix}.band_frac"), span.band_frac, 0.0)?;
    check_range(&format!("{prefix}.gr_weight"), span.gr_weight, 0.0, 1.0)?;
    check_gt(&format!("{prefix}.z_cap"), span.z_cap, 0.0)?;
    check_gt(&format!("{prefix}.mad_factor"), span.mad_factor, 0.0)?;
    check_gt(&format!("{prefix}.scale_floor"), span.scale_floor, 0.0)?;
    check_gt(
        &format!("{prefix}.resd_floor_ohmm"),
        span.resd_floor_ohmm,
        0.0,
    )?;
    check_ge(&format!("{prefix}.max_gap_steps"), span.max_gap_steps, 0.0)?;
    check_ge(&format!("{prefix}.min_gap_ft"), span.min_gap_ft, 0.0)?;
    rel_usize(
        &format!("{prefix}.min_nodes"),
        span.min_nodes,
        "<=",
        &format!("{prefix}.max_nodes"),
        span.max_nodes,
        span.min_nodes <= span.max_nodes,
    )?;
    Ok(())
}

fn validate_nphi_offset(cfg: &ProjectConfig) -> SxResult<()> {
    let c = &cfg.nphi_offset;
    check_finite("nphi_offset.gr_max", c.gr_max)?;
    check_range("nphi_offset.phi_max", c.phi_max, -0.1, 0.5)?;
    check_finite("nphi_offset.pef_min", c.pef_min)?;
    check_gt("nphi_offset.pef_max", c.pef_max - c.pef_min, 0.0)?;
    check_gt("nphi_offset.rho_matrix", c.rho_matrix - c.rho_fluid, 0.0)?;
    check_count("nphi_offset.min_well_samples", c.min_well_samples)?;
    check_count("nphi_offset.min_wells", c.min_wells)?;
    check_count("nphi_offset.n_boot", c.n_boot)?;
    check_range("nphi_offset.ci_level", c.ci_level, 0.5, 0.999)?;
    check_gt("nphi_offset.outlier_limit", c.outlier_limit, 0.0)?;
    check_gt("nphi_offset.fit_phi_max", c.fit_phi_max - c.phi_max, 0.0)?;
    check_gt("nphi_offset.fit_bin", c.fit_bin, 0.0)?;
    check_gt("nphi_offset.slope_tol", c.slope_tol, 0.0)?;
    check_count("nphi_offset.n_boot_slope", c.n_boot_slope)?;
    Ok(())
}

fn validate_normalize(cfg: &ProjectConfig) -> SxResult<()> {
    let normalize = &cfg.normalize;
    check_count("normalize.ref_k", normalize.ref_k)?;
    check_count("normalize.ref_min_voters", normalize.ref_min_voters)?;
    check_count("normalize.pick_min_count", normalize.pick_min_count)?;
    check_count("normalize.depth_bins", normalize.depth_bins)?;
    check_count("normalize.depth_candidates", normalize.depth_candidates)?;
    check_count(
        "normalize.nphi_offset_min_count",
        normalize.nphi_offset_min_count,
    )?;
    check_gt(
        "normalize.nphi_offset_limit",
        normalize.nphi_offset_limit,
        0.0,
    )?;
    check_range("normalize.nphi_shift", normalize.nphi_shift, -0.2, 0.2)?;
    check_finite("normalize.nphi_offset_gr_max", normalize.nphi_offset_gr_max)?;
    check_finite(
        "normalize.nphi_offset_rhob_min",
        normalize.nphi_offset_rhob_min,
    )?;
    rel_f64(
        "normalize.nphi_offset_rho_matrix",
        normalize.nphi_offset_rho_matrix,
        ">",
        "normalize.nphi_offset_rho_fluid",
        normalize.nphi_offset_rho_fluid,
        normalize.nphi_offset_rho_matrix > normalize.nphi_offset_rho_fluid,
    )?;
    check_one_of("normalize.scope", &normalize.scope, &["layer", "all"])?;
    check_ge("normalize.ref_power", normalize.ref_power, 0.0)?;
    check_gt(
        "normalize.ref_dist_floor_m",
        normalize.ref_dist_floor_m,
        0.0,
    )?;
    check_ge(
        "normalize.ref_span_ratio_max",
        normalize.ref_span_ratio_max,
        1.0,
    )?;
    check_range(
        "normalize.stretch_min_corr",
        normalize.stretch_min_corr,
        -1.0,
        1.0,
    )?;
    check_ge("normalize.gain_cap", normalize.gain_cap, 1.0)?;
    rel_f64(
        "normalize.gain_cap_extended",
        normalize.gain_cap_extended,
        ">=",
        "normalize.gain_cap",
        normalize.gain_cap,
        normalize.gain_cap_extended >= normalize.gain_cap,
    )?;
    check_open_open(
        "normalize.shift_only_frac",
        normalize.shift_only_frac,
        0.0,
        1.0,
    )?;
    check_ge("normalize.degenerate_span", normalize.degenerate_span, 0.0)?;
    check_open_closed(
        "normalize.depth_max_cover",
        normalize.depth_max_cover,
        0.0,
        1.0,
    )?;
    if normalize.logs.is_empty() {
        return Err(bad("normalize.logs", "must not be empty".into()));
    }
    for kind in &normalize.logs {
        if matches!(kind, LogKind::Rhob | LogKind::Dtc | LogKind::Resd) {
            return Err(bad(
                "normalize.logs",
                format!(
                    "{} is not allowed (RHOB, DTC and RESD are not normalized)",
                    kind.name()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_neighbors(cfg: &ProjectConfig) -> SxResult<()> {
    let neighbors = &cfg.neighbors;
    check_count("neighbors.knn_list_len", neighbors.knn_list_len)?;
    check_count("neighbors.search_cap", neighbors.search_cap)?;
    check_count("neighbors.k_primary", neighbors.k_primary)?;
    check_count("neighbors.batch_size", neighbors.batch_size)?;
    check_count("neighbors.quorum", neighbors.quorum)?;
    check_count("neighbors.min_finite_donor", neighbors.min_finite_donor)?;
    check_count("neighbors.validate_min_run", neighbors.validate_min_run)?;
    check_count("neighbors.range_map_min_n", neighbors.range_map_min_n)?;
    check_count("neighbors.trusted_min_n", neighbors.trusted_min_n)?;

    check_gt("neighbors.max_dist_m", neighbors.max_dist_m, 0.0)?;
    check_gt("neighbors.min_dist_m", neighbors.min_dist_m, 0.0)?;
    check_gt(
        "neighbors.trusted_max_dist_m",
        neighbors.trusted_max_dist_m,
        0.0,
    )?;

    check_ge("neighbors.idw_power", neighbors.idw_power, 0.0)?;
    check_ge(
        "neighbors.donor_max_bracket_ft",
        neighbors.donor_max_bracket_ft,
        0.0,
    )?;
    check_ge("neighbors.interval_eps", neighbors.interval_eps, 0.0)?;
    check_ge(
        "neighbors.validate_split_ft",
        neighbors.validate_split_ft,
        0.0,
    )?;
    check_ge("neighbors.flat_rel_tol", neighbors.flat_rel_tol, 0.0)?;
    check_ge(
        "neighbors.saturation_rel_tol",
        neighbors.saturation_rel_tol,
        0.0,
    )?;

    check_range("neighbors.iq_no_coord", neighbors.iq_no_coord, 0.0, 1.0)?;
    check_range("neighbors.aq_unaligned", neighbors.aq_unaligned, 0.0, 1.0)?;
    check_range("neighbors.conf_observed", neighbors.conf_observed, 0.0, 1.0)?;
    check_range(
        "neighbors.conf_estimated_one_user",
        neighbors.conf_estimated_one_user,
        0.0,
        1.0,
    )?;
    check_range(
        "neighbors.conf_estimated_no_user",
        neighbors.conf_estimated_no_user,
        0.0,
        1.0,
    )?;
    check_range(
        "neighbors.conf_pad_floor",
        neighbors.conf_pad_floor,
        0.0,
        1.0,
    )?;
    check_range(
        "neighbors.trusted_min_quality",
        neighbors.trusted_min_quality,
        0.0,
        1.0,
    )?;
    check_range("neighbors.min_span_frac", neighbors.min_span_frac, 0.0, 1.0)?;

    check_open_closed(
        "neighbors.saturation_frac",
        neighbors.saturation_frac,
        0.0,
        1.0,
    )?;
    check_range("neighbors.range_p_low", neighbors.range_p_low, 0.0, 100.0)?;
    check_range("neighbors.range_p_high", neighbors.range_p_high, 0.0, 100.0)?;
    rel_f64(
        "neighbors.range_p_low",
        neighbors.range_p_low,
        "<",
        "neighbors.range_p_high",
        neighbors.range_p_high,
        neighbors.range_p_low < neighbors.range_p_high,
    )?;
    check_ge("neighbors.max_gain", neighbors.max_gain, 1.0)?;
    Ok(())
}

fn validate_fill(cfg: &ProjectConfig) -> SxResult<()> {
    let fill = &cfg.fill;
    check_count("fill.min_train_rows", fill.min_train_rows)?;
    check_count("fill.local_min_count", fill.local_min_count)?;
    check_count("fill.range_min_n", fill.range_min_n)?;
    check_count("fill.flat_min_run", fill.flat_min_run)?;
    check_count("fill.plateau_window", fill.plateau_window)?;
    check_count("fill.plateau_min_count", fill.plateau_min_count)?;
    check_count("fill.quiet_min_count", fill.quiet_min_count)?;
    check_count("fill.line_min_rows", fill.line_min_rows)?;
    check_count("fill.holdout_n_wells", fill.holdout_n_wells)?;
    check_count("fill.holdout_min_samples", fill.holdout_min_samples)?;
    check_count("fill.holdout_blocks", fill.holdout_blocks)?;
    check_count("fill.holdout_min_block", fill.holdout_min_block)?;
    check_count("fill.holdout_max_attempts", fill.holdout_max_attempts)?;

    check_range("fill.dropout_frac", fill.dropout_frac, 0.0, 1.0)?;
    check_range("fill.dropout_layer_prob", fill.dropout_layer_prob, 0.0, 1.0)?;

    check_open_closed("fill.holdout_frac", fill.holdout_frac, 0.0, 1.0)?;
    check_open_closed("fill.saturation_frac", fill.saturation_frac, 0.0, 1.0)?;

    check_range("fill.range_p_low", fill.range_p_low, 0.0, 100.0)?;
    check_range("fill.range_p_high", fill.range_p_high, 0.0, 100.0)?;
    rel_f64(
        "fill.range_p_low",
        fill.range_p_low,
        "<",
        "fill.range_p_high",
        fill.range_p_high,
        fill.range_p_low < fill.range_p_high,
    )?;
    check_ge("fill.max_gain", fill.max_gain, 1.0)?;

    check_ge("fill.min_span_frac", fill.min_span_frac, 0.0)?;
    check_ge("fill.degenerate_span", fill.degenerate_span, 0.0)?;
    check_ge("fill.near_const_rel", fill.near_const_rel, 0.0)?;
    check_ge("fill.near_const_abs", fill.near_const_abs, 0.0)?;
    check_ge("fill.plateau_std_frac", fill.plateau_std_frac, 0.0)?;
    check_ge("fill.quiet_std_frac", fill.quiet_std_frac, 0.0)?;
    check_ge("fill.exact_flat_tol", fill.exact_flat_tol, 0.0)?;
    check_ge("fill.saturation_rel_tol", fill.saturation_rel_tol, 0.0)?;
    check_ge("fill.line_rel_tol", fill.line_rel_tol, 0.0)?;
    check_gt("fill.scale_floor", fill.scale_floor, 0.0)?;
    check_one_of(
        "fill.holdout_mode",
        &fill.holdout_mode,
        &["interval", "whole_log"],
    )?;

    if fill.order.is_empty() {
        return Err(bad("fill.order", "must not be empty".into()));
    }
    let mut seen = BTreeSet::new();
    for kind in &fill.order {
        if !seen.insert(*kind) {
            return Err(bad("fill.order", format!("duplicate log {}", kind.name())));
        }
    }
    if fill.local_windows_ft.len() != 2 {
        return Err(bad(
            "fill.local_windows_ft",
            format!(
                "must have exactly 2 entries (got {})",
                fill.local_windows_ft.len()
            ),
        ));
    }
    for (index, value) in fill.local_windows_ft.iter().enumerate() {
        check_gt(&format!("fill.local_windows_ft[{index}]"), *value, 0.0)?;
    }
    Ok(())
}

fn validate_lightgbm(cfg: &ProjectConfig) -> SxResult<()> {
    let lightgbm = &cfg.fill.lightgbm;
    check_count("fill.lightgbm.num_iterations", lightgbm.num_iterations)?;
    check_count("fill.lightgbm.max_depth", lightgbm.max_depth)?;
    check_count("fill.lightgbm.min_data_in_leaf", lightgbm.min_data_in_leaf)?;
    check_count(
        "fill.lightgbm.bin_construct_sample_cnt",
        lightgbm.bin_construct_sample_cnt,
    )?;
    check_count("fill.lightgbm.num_threads", lightgbm.num_threads)?;
    check_count_min("fill.lightgbm.num_leaves", lightgbm.num_leaves, 2)?;
    check_count_min("fill.lightgbm.max_bin", lightgbm.max_bin, 2)?;
    check_count_min("fill.lightgbm.max_bin_wide", lightgbm.max_bin_wide, 2)?;
    check_gt("fill.lightgbm.learning_rate", lightgbm.learning_rate, 0.0)?;
    if !matches!(
        lightgbm.objective.as_str(),
        "regression" | "huber" | "quantile"
    ) {
        return Err(SxError::Config(format!(
            "fill.lightgbm.objective must be regression, huber or quantile, got {:?}",
            lightgbm.objective
        )));
    }
    check_gt("fill.lightgbm.alpha", lightgbm.alpha, 0.0)?;
    check_open_closed(
        "fill.lightgbm.feature_fraction",
        lightgbm.feature_fraction,
        0.0,
        1.0,
    )?;
    check_open_closed(
        "fill.lightgbm.bagging_fraction",
        lightgbm.bagging_fraction,
        0.0,
        1.0,
    )?;
    check_ge("fill.lightgbm.lambda_l1", lightgbm.lambda_l1, 0.0)?;
    check_ge("fill.lightgbm.lambda_l2", lightgbm.lambda_l2, 0.0)?;
    check_ge(
        "fill.lightgbm.min_gain_to_split",
        lightgbm.min_gain_to_split,
        0.0,
    )?;
    check_ge(
        "fill.lightgbm.min_sum_hessian_in_leaf",
        lightgbm.min_sum_hessian_in_leaf,
        0.0,
    )?;
    Ok(())
}

fn validate_ladder(cfg: &ProjectConfig) -> SxResult<()> {
    let ladder = &cfg.fill.ladder;
    check_count("fill.ladder.real_donor_max_n", ladder.real_donor_max_n)?;
    check_count("fill.ladder.donor_max_n", ladder.donor_max_n)?;
    check_count("fill.ladder.anchor_rows", ladder.anchor_rows)?;
    check_count(
        "fill.ladder.exact_flat_min_rows",
        ladder.exact_flat_min_rows,
    )?;
    check_count("fill.ladder.crosslog_min_rows", ladder.crosslog_min_rows)?;
    check_count("fill.ladder.texture_window", ladder.texture_window)?;
    check_count("fill.ladder.scale_min_measured", ladder.scale_min_measured)?;
    check_gt("fill.ladder.donor_max_dist_m", ladder.donor_max_dist_m, 0.0)?;
    check_gt("fill.ladder.tilt_max_scale", ladder.tilt_max_scale, 0.0)?;
    check_gt(
        "fill.ladder.crosslog_window_ft",
        ladder.crosslog_window_ft,
        0.0,
    )?;
    check_gt("fill.ladder.scale_fallback", ladder.scale_fallback, 0.0)?;
    check_ge("fill.ladder.anchor_tol_scale", ladder.anchor_tol_scale, 0.0)?;
    check_ge("fill.ladder.guard_quiet_frac", ladder.guard_quiet_frac, 0.0)?;
    check_ge("fill.ladder.crosslog_min_std", ladder.crosslog_min_std, 0.0)?;
    check_range(
        "fill.ladder.physical_max_out_frac",
        ladder.physical_max_out_frac,
        0.0,
        1.0,
    )?;
    check_range(
        "fill.ladder.crosslog_min_r2",
        ladder.crosslog_min_r2,
        0.0,
        1.0,
    )?;
    Ok(())
}

fn validate_correlate(cfg: &ProjectConfig) -> SxResult<()> {
    let graph = &cfg.correlate.graph;
    check_count("correlate.graph.k_graph", graph.k_graph)?;
    check_gt("correlate.graph.max_edge_m", graph.max_edge_m, 0.0)?;

    let solve = &cfg.correlate.solve;
    check_gt("correlate.solve.knot_step_ft", solve.knot_step_ft, 0.0)?;
    check_gt("correlate.solve.lambda_ridge", solve.lambda_ridge, 0.0)?;
    check_gt("correlate.solve.tukey_c", solve.tukey_c, 0.0)?;
    check_count("correlate.solve.match_step", solve.match_step)?;
    check_ge("correlate.solve.lambda_smooth", solve.lambda_smooth, 0.0)?;

    let tops = &cfg.correlate.tops;
    check_count("correlate.tops.k_check", tops.k_check)?;
    check_count("correlate.tops.min_votes", tops.min_votes)?;
    rel_usize(
        "correlate.tops.min_votes",
        tops.min_votes,
        "<=",
        "correlate.tops.k_check",
        tops.k_check,
        tops.min_votes <= tops.k_check,
    )?;
    check_ge("correlate.tops.tol_ft", tops.tol_ft, 0.0)?;
    check_ge("correlate.tops.spread_mult", tops.spread_mult, 0.0)?;
    check_open_closed("correlate.tops.side_share", tops.side_share, 0.0, 1.0)?;
    Ok(())
}

fn validate_subhorizons(cfg: &ProjectConfig) -> SxResult<()> {
    let sub = &cfg.subhorizons;
    check_count_min("subhorizons.n_grid", sub.n_grid, 2)?;
    check_count("subhorizons.min_wells", sub.min_wells)?;
    check_count("subhorizons.min_segment_samples", sub.min_segment_samples)?;
    check_count(
        "subhorizons.kernel_half_width_mult",
        sub.kernel_half_width_mult,
    )?;
    check_count("subhorizons.min_spacing_samples", sub.min_spacing_samples)?;
    check_gt("subhorizons.link_tol_mult", sub.link_tol_mult, 0.0)?;
    check_ge("subhorizons.min_strength_api", sub.min_strength_api, 0.0)?;
    let scale_count = sub.scales.len();
    if scale_count > 0 && (sub.min_persistence < 1 || sub.min_persistence > scale_count) {
        return Err(bad(
            "subhorizons.min_persistence",
            format!(
                "must be in [1, {scale_count}] (got {})",
                sub.min_persistence
            ),
        ));
    }
    if sub.scales.is_empty() {
        return Err(bad("subhorizons.scales", "must not be empty".into()));
    }
    if !scales_strictly_ascending(&sub.scales) {
        return Err(bad(
            "subhorizons.scales",
            format!(
                "must be strictly ascending with every entry >= 1 (got {})",
                format_usize_list(&sub.scales)
            ),
        ));
    }
    Ok(())
}

fn scales_strictly_ascending(scales: &[usize]) -> bool {
    let mut previous: Option<usize> = None;
    for value in scales {
        if *value < 1 {
            return false;
        }
        if let Some(earlier) = previous {
            if *value <= earlier {
                return false;
            }
        }
        previous = Some(*value);
    }
    true
}

fn format_usize_list(values: &[usize]) -> String {
    let body = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn validate_refresh(cfg: &ProjectConfig) -> SxResult<()> {
    check_ge("refresh.refresh_tol_ft", cfg.refresh.refresh_tol_ft, 0.0)?;
    check_range("refresh.percentile", cfg.refresh.percentile, 0.0, 100.0)?;
    Ok(())
}
