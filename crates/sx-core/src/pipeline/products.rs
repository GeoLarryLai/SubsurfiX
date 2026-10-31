// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Per-stage input and output paths, relative to the project folder.
//!
//! Dynamic entries are not listed here: ingest reads the files from
//! `ingest::source_files`, fill writes one `models/<LOG>.txt` per trained
//! model, refresh reads those model files, export writes one LAS per well.

use super::stage::Stage;

pub(crate) const WELLS: &str = "stages/ingest/wells.parquet";
pub(crate) const GEOMETRY: &str = "stages/ingest/geometry.parquet";
pub(crate) const LOGS_RAW: &str = "stages/ingest/logs_raw.parquet";
pub(crate) const TOPS_INPUT: &str = "stages/ingest/tops_input.parquet";
pub(crate) const AUX_RAW: &str = "stages/ingest/aux_raw.parquet";
pub(crate) const INGEST_REPORT: &str = "stages/ingest/ingest_report.csv";
pub(crate) const LAS_MATCH: &str = "stages/ingest/las_match.csv";

pub(crate) const LOGS_QC: &str = "stages/qc/logs_qc.parquet";
pub(crate) const QC_FLAGS: &str = "stages/qc/qc_flags.parquet";
pub(crate) const QC_REPORT: &str = "stages/qc/qc_report.csv";
pub(crate) const QC_LOW_DENSITY: &str = "stages/qc/low_density_unverified.csv";
pub(crate) const QC_RHOB_CLOSED_CAL: &str = "stages/qc/rhob_closed_caliper.parquet";
pub(crate) const QC_CLOSED_CAL_RUNS: &str = "stages/qc/closed_caliper_runs.csv";
pub(crate) const QC_LAS_MATCH: &str = "stages/qc/las_match_report.csv";

pub(crate) const INTERVALS: &str = "stages/rough_align/intervals.parquet";
pub(crate) const TOPS_ESTIMATED: &str = "stages/rough_align/tops_estimated.parquet";
pub(crate) const PAIRS_ROUGH: &str = "stages/rough_align/pairs_rough.parquet";
pub(crate) const LAYER_COORD: &str = "stages/rough_align/layer_coord.parquet";

pub(crate) const LOGS_NORM: &str = "stages/normalize/logs_norm.parquet";
pub(crate) const QC_FLAGS_POST: &str = "stages/normalize/qc_flags_post.parquet";
pub(crate) const NORM_ANCHORS: &str = "stages/normalize/norm_anchors.csv";
pub(crate) const NORM_REPORT: &str = "stages/normalize/norm_report.csv";

pub(crate) const NEIGHBOR_CURVES: &str = "stages/neighbors/neighbor_curves.parquet";

pub(crate) const LOGS_FILLED: &str = "stages/fill/logs_filled.parquet";
pub(crate) const FILL_REPORT: &str = "stages/fill/fill_report.csv";
pub(crate) const FILL_SUMMARY: &str = "stages/fill/fill_summary.csv";
pub(crate) const MODELS_JSON: &str = "stages/fill/models.json";
/// Folder of the per-log model texts `<LOG>.txt`.
pub(crate) const MODELS_DIR: &str = "stages/fill/models";

pub(crate) const LAYER_TIME: &str = "stages/correlate/layer_time.parquet";
pub(crate) const TOPS: &str = "stages/correlate/tops.parquet";
pub(crate) const TOPS_CSV: &str = "stages/correlate/tops.csv";
pub(crate) const SUSPECTS: &str = "stages/correlate/suspects.csv";
pub(crate) const SOLVE_REPORT: &str = "stages/correlate/solve_report.csv";

pub(crate) const SUBHORIZONS: &str = "stages/subhorizons/subhorizons.parquet";
pub(crate) const SUBHORIZONS_CSV: &str = "stages/subhorizons/subhorizons.csv";

pub(crate) const LOGS_FINAL: &str = "stages/refresh/logs_final.parquet";
pub(crate) const REFRESH_REPORT: &str = "stages/refresh/refresh_report.csv";

/// Folder of the per-well LAS files.
pub(crate) const EXPORT_LAS_DIR: &str = "export/las";
pub(crate) const EXPORT_GLOSSARY: &str = "export/column_glossary.csv";

/// `(product name, source, export copy)` of the Parquet files export copies.
pub(crate) const EXPORT_COPIES: [(&str, &str, &str); 6] = [
    ("wells", WELLS, "export/wells.parquet"),
    ("geometry", GEOMETRY, "export/geometry.parquet"),
    ("logs_final", LOGS_FINAL, "export/logs_final.parquet"),
    ("tops", TOPS, "export/tops.parquet"),
    ("layer_time", LAYER_TIME, "export/layer_time.parquet"),
    ("subhorizons", SUBHORIZONS, "export/subhorizons.parquet"),
];

/// Folder a stage writes into (cleaned of stale products after it runs).
pub(crate) fn output_dir(stage: Stage) -> String {
    match stage {
        Stage::Export => "export".to_string(),
        other => format!("stages/{}", other.name()),
    }
}

const NEIGHBOR_INPUTS: [&str; 12] = [
    WELLS,
    GEOMETRY,
    LOGS_RAW,
    TOPS_INPUT,
    LOGS_NORM,
    QC_FLAGS_POST,
    NORM_ANCHORS,
    NORM_REPORT,
    INTERVALS,
    TOPS_ESTIMATED,
    PAIRS_ROUGH,
    LAYER_COORD,
];

/// Fixed inputs of `stage`.
pub(crate) fn fixed_inputs(stage: Stage) -> Vec<&'static str> {
    match stage {
        Stage::Ingest => Vec::new(),
        Stage::Qc => vec![WELLS, LOGS_RAW, AUX_RAW, LAS_MATCH],
        Stage::RoughAlign => vec![WELLS, GEOMETRY, TOPS_INPUT, LOGS_QC],
        Stage::Normalize => vec![
            WELLS,
            GEOMETRY,
            LOGS_QC,
            QC_FLAGS,
            QC_REPORT,
            INTERVALS,
            PAIRS_ROUGH,
            LAYER_COORD,
        ],
        Stage::Neighbors => NEIGHBOR_INPUTS.to_vec(),
        Stage::Fill => {
            let mut v = NEIGHBOR_INPUTS.to_vec();
            v.push(NEIGHBOR_CURVES);
            v
        }
        Stage::Correlate => vec![
            WELLS,
            GEOMETRY,
            TOPS_INPUT,
            INTERVALS,
            TOPS_ESTIMATED,
            LOGS_FILLED,
        ],
        Stage::Subhorizons => vec![
            WELLS,
            GEOMETRY,
            TOPS_INPUT,
            INTERVALS,
            LOGS_FILLED,
            LAYER_TIME,
        ],
        Stage::Refresh => {
            let mut v = NEIGHBOR_INPUTS.to_vec();
            v.extend([NEIGHBOR_CURVES, LOGS_FILLED, LAYER_TIME]);
            v
        }
        Stage::Export => vec![WELLS, GEOMETRY, LOGS_FINAL, TOPS, LAYER_TIME, SUBHORIZONS],
    }
}

/// Fixed outputs of `stage`.
pub(crate) fn fixed_outputs(stage: Stage) -> Vec<&'static str> {
    match stage {
        Stage::Ingest => vec![
            WELLS,
            GEOMETRY,
            LOGS_RAW,
            AUX_RAW,
            TOPS_INPUT,
            INGEST_REPORT,
            LAS_MATCH,
        ],
        Stage::Qc => vec![
            LOGS_QC,
            QC_FLAGS,
            QC_REPORT,
            QC_LOW_DENSITY,
            QC_RHOB_CLOSED_CAL,
            QC_CLOSED_CAL_RUNS,
            QC_LAS_MATCH,
        ],
        Stage::RoughAlign => vec![INTERVALS, TOPS_ESTIMATED, PAIRS_ROUGH, LAYER_COORD],
        Stage::Normalize => vec![LOGS_NORM, QC_FLAGS_POST, NORM_ANCHORS, NORM_REPORT],
        Stage::Neighbors => vec![NEIGHBOR_CURVES],
        Stage::Fill => vec![LOGS_FILLED, FILL_REPORT, FILL_SUMMARY, MODELS_JSON],
        Stage::Correlate => vec![LAYER_TIME, TOPS, TOPS_CSV, SUSPECTS, SOLVE_REPORT],
        Stage::Subhorizons => vec![SUBHORIZONS, SUBHORIZONS_CSV],
        Stage::Refresh => vec![LOGS_FINAL, REFRESH_REPORT],
        Stage::Export => {
            let mut v: Vec<&'static str> = EXPORT_COPIES.iter().map(|c| c.2).collect();
            v.push(EXPORT_GLOSSARY);
            v
        }
    }
}
