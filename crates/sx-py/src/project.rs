// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Pipeline, config and logging entry points of `subsurfix._native`.
//!
//! Engine work runs inside `py.detach`: plug-in models (rayon workers calling
//! `Python::attach`) can run, and other Python threads keep going.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use sx_core::config::{
    self, AreaSource, EdgesConfig, HeaderSource, PtsSource, ProjectConfig, ProjectSection, RuntimeConfig,
    SurveySource, TopsSource,
};
use sx_core::pipeline::{self, RunOptions, RunSummary, Stage, StageOutcome, StageStatus};
use sx_core::{SxError, SxResult};
use tracing_subscriber::filter::LevelFilter;

use crate::errors::to_py_err;
use crate::plugin::snapshot_overrides;

/// One row per stage, as handed to Python.
type Rows = Vec<BTreeMap<String, Py<PyAny>>>;

/// `(config section, stage whose hashed config holds it)`, in `ProjectConfig`
/// field order after `seed`, `project`, `runtime`, `edges`.
const STAGE_SECTIONS: [(&str, &str); 11] = [
    ("ingest", "ingest"),
    ("qc", "qc"),
    ("align", "rough_align"),
    ("dtw", "rough_align"),
    ("normalize", "normalize"),
    ("neighbors", "neighbors"),
    ("fill", "fill"),
    ("correlate", "correlate"),
    ("subhorizons", "subhorizons"),
    ("refresh", "refresh"),
    ("export", "export"),
];

/// Engine version, e.g. `"0.0.2-dev"`.
#[pyfunction]
pub(crate) fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The ten stage names, in run order.
#[pyfunction]
pub(crate) fn stage_names() -> Vec<&'static str> {
    Stage::ALL.iter().map(|s| s.name()).collect()
}

/// `{stage: [file, ...]}`: fixed product files of each stage, relative to
/// `stages/<stage>/` (to `export/` for export).
#[pyfunction]
pub(crate) fn stage_products() -> BTreeMap<String, Vec<String>> {
    pipeline::stage_products()
        .into_iter()
        .map(|(stage, files)| {
            let files = files.into_iter().map(str::to_owned).collect();
            (stage.name().to_owned(), files)
        })
        .collect()
}

/// The project file at `config_path`, parsed and validated, as compact JSON
/// with every default filled in. Errors: `IoError` (unreadable file),
/// `ConfigError` (parse or validation).
#[pyfunction]
pub(crate) fn config_json(py: Python<'_>, config_path: PathBuf) -> PyResult<String> {
    py.detach(move || config::load_project(&config_path).and_then(|cfg| project_json(&cfg)))
        .map_err(to_py_err)
}

/// Run the pipeline of the project at `config_path` with the registered
/// plug-ins (one snapshot), GIL released. Returns one dict per stage:
/// `stage`, `action`, `seconds`, `n_wells`, `stage_key` (None when out of range).
/// Errors: `InvalidInputError` (unknown stage name, `from_stage` after
/// `to_stage`, an earlier stage not up to date), and every error of
/// `config::load_project` and `pipeline::run`.
#[pyfunction]
#[pyo3(signature = (config_path, from_stage=None, to_stage=None, force=false, threads=None))]
pub(crate) fn run_pipeline(
    py: Python<'_>,
    config_path: PathBuf,
    from_stage: Option<String>,
    to_stage: Option<String>,
    force: bool,
    threads: Option<usize>,
) -> PyResult<Rows> {
    let from = parse_stage(from_stage.as_deref())?;
    let to = parse_stage(to_stage.as_deref())?;
    let models = snapshot_overrides(py)?;
    let dir = project_dir(&config_path);
    let result: SxResult<RunSummary> = py.detach(move || {
        let cfg = config::load_project(&config_path)?;
        let mut o = RunOptions::new(dir);
        o.from = from;
        o.to = to;
        o.force = force;
        o.threads = threads;
        o.models = models;
        pipeline::run(&cfg, &o)
    });
    let summary = result.map_err(to_py_err)?;
    summary.stages.iter().map(|o| outcome_row(py, o)).collect()
}

/// Holdout scoring of the project at `config_path` (GIL released): masks
/// intervals of a few wells, refills them and scores the fill against the
/// measured values. The project must have run through `fill`. Writes
/// `stages/fill/holdout/holdout_scores.csv` and `holdout_summary.csv` and
/// returns their paths `(scores, summary)`. Errors: those of
/// `config::load_project` and `fill::holdout::score_holdout`.
#[pyfunction]
#[pyo3(signature = (config_path, threads=None))]
pub(crate) fn score_holdout(
    py: Python<'_>,
    config_path: PathBuf,
    threads: Option<usize>,
) -> PyResult<(PathBuf, PathBuf)> {
    let dir = project_dir(&config_path);
    let result: SxResult<(PathBuf, PathBuf)> = py.detach(move || {
        let cfg = config::load_project(&config_path)?;
        sx_core::runtime::with_threads(threads.unwrap_or(cfg.runtime.threads), || {
            let r = sx_core::fill::holdout::score_holdout(&dir, &cfg)?;
            Ok((r.scores_path, r.summary_path))
        })
    });
    result.map_err(to_py_err)
}

/// What `run_pipeline` would do now, per stage, counting registered plug-ins.
/// One dict per stage: `stage`, `version`, `state`, `recorded_key`,
/// `finished_utc` (the last two None without a record). Errors: those of
/// `config::load_project` and `pipeline::status`.
#[pyfunction]
pub(crate) fn pipeline_status(py: Python<'_>, config_path: PathBuf) -> PyResult<Rows> {
    let models = snapshot_overrides(py)?;
    let dir = project_dir(&config_path);
    let result: SxResult<Vec<StageStatus>> = py.detach(move || {
        let cfg = config::load_project(&config_path)?;
        let mut o = RunOptions::new(dir);
        o.models = models;
        pipeline::status(&cfg, &o)
    });
    let status = result.map_err(to_py_err)?;
    status.iter().map(|s| status_row(py, s)).collect()
}

/// Print engine log events at `level` (`error`, `warn`, `info`, `debug`,
/// `trace`, any case) and above to stderr. Returns `True` if installed now,
/// `False` if a subscriber already existed. Errors: `InvalidInputError` for
/// another level.
#[pyfunction]
#[pyo3(signature = (level="info"))]
pub(crate) fn init_logging(level: &str) -> PyResult<bool> {
    let max = match level.to_ascii_lowercase().as_str() {
        "error" => LevelFilter::ERROR,
        "warn" => LevelFilter::WARN,
        "info" => LevelFilter::INFO,
        "debug" => LevelFilter::DEBUG,
        "trace" => LevelFilter::TRACE,
        _ => {
            return Err(to_py_err(SxError::InvalidInput(format!(
                "unknown log level {level:?}; valid levels: error, warn, info, debug, trace"
            ))))
        }
    };
    Ok(tracing_subscriber::fmt()
        .with_max_level(max)
        .with_writer(std::io::stderr)
        .try_init()
        .is_ok())
}

fn parse_stage(name: Option<&str>) -> PyResult<Option<Stage>> {
    name.map(Stage::from_name).transpose().map_err(to_py_err)
}

/// Parent of `config_path`, `"."` when it has none.
fn project_dir(config_path: &Path) -> PathBuf {
    match config_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn outcome_row(py: Python<'_>, o: &StageOutcome) -> PyResult<BTreeMap<String, Py<PyAny>>> {
    Ok(BTreeMap::from([
        ("stage".to_owned(), o.stage.name().into_py_any(py)?),
        ("action".to_owned(), o.action.as_str().into_py_any(py)?),
        ("seconds".to_owned(), o.seconds.into_py_any(py)?),
        ("n_wells".to_owned(), o.n_wells.into_py_any(py)?),
        ("stage_key".to_owned(), o.stage_key.clone().into_py_any(py)?),
    ]))
}

fn status_row(py: Python<'_>, s: &StageStatus) -> PyResult<BTreeMap<String, Py<PyAny>>> {
    Ok(BTreeMap::from([
        ("stage".to_owned(), s.stage.name().into_py_any(py)?),
        ("version".to_owned(), s.version.into_py_any(py)?),
        ("state".to_owned(), s.state.as_str().into_py_any(py)?),
        (
            "recorded_key".to_owned(),
            s.recorded_key.clone().into_py_any(py)?,
        ),
        (
            "finished_utc".to_owned(),
            s.finished_utc.clone().into_py_any(py)?,
        ),
    ]))
}

/// Whole config as JSON, keys in `ProjectConfig` field order. Stage sections
/// are serde's text via `ProjectConfig::stage_section`; `project`, `runtime`
/// and `edges` are written here (sx-py has no JSON serializer of its own),
/// following their serde attributes.
fn project_json(cfg: &ProjectConfig) -> SxResult<String> {
    let ProjectConfig {
        seed,
        project: ProjectSection { crs },
        runtime: RuntimeConfig { threads },
        edges,
        ingest: _,
        qc: _,
        align: _,
        dtw: _,
        normalize: _,
        neighbors: _,
        fill: _,
        correlate: _,
        subhorizons: _,
        refresh: _,
        export: _,
        propagation,
    } = cfg;
    let mut out = Obj::default()
        .raw("seed", seed.to_string())
        .raw("project", Obj::default().str("crs", crs).finish())
        .raw(
            "runtime",
            Obj::default().raw("threads", threads.to_string()).finish(),
        )
        .raw("edges", edges_json(edges));
    for (section, stage) in STAGE_SECTIONS {
        let value = cfg.stage_section(stage)?;
        let text = match Stage::from_name(stage)?.config_sections() {
            [_] => value.to_string(),
            _ => value
                .get(section)
                .map(ToString::to_string)
                .ok_or_else(|| SxError::Config(format!("config section {section:?} missing")))?,
        };
        out = out.raw(section, text);
    }
    let q: Vec<String> = propagation
        .quantiles
        .iter()
        .map(|v| format!("{v:?}"))
        .collect();
    out = out.raw(
        "propagation",
        Obj::default()
            .str("method", &propagation.method)
            .raw("quantiles", format!("[{}]", q.join(",")))
            .raw("qrf_trees", propagation.qrf_trees.to_string())
            .raw("qrf_min_leaf", propagation.qrf_min_leaf.to_string())
            .raw("lgbm_iterations", propagation.lgbm_iterations.to_string())
            .finish(),
    );
    Ok(out.finish())
}

fn edges_json(e: &EdgesConfig) -> String {
    let EdgesConfig {
        allow_network,
        allow_ballpark,
        headers,
        tops,
        surveys,
        pts,
        area,
    } = e;
    let out = Obj::default()
        .raw("allow_network", allow_network.to_string())
        .raw("allow_ballpark", allow_ballpark.to_string())
        .raw("headers", json_array(headers.iter().map(header_json)))
        .raw("tops", json_array(tops.iter().map(tops_json)))
        .raw("surveys", json_array(surveys.iter().map(survey_json)))
        .raw("pts", json_array(pts.iter().map(pts_json)));
    match area {
        Some(a) => out.raw("area", area_json(a)),
        None => out,
    }
    .finish()
}

fn pts_json(p: &PtsSource) -> String {
    let PtsSource {
        path,
        well,
        key_col,
        sheets,
        delimiter,
        temp_unit,
        pressure_unit,
        depth_unit,
    } = p;
    Obj::default()
        .path("path", path)
        .opt("well", well.as_deref())
        .opt("key_col", key_col.as_deref())
        .raw("sheets", json_array(sheets.iter().map(|s| json_string(s))))
        .str("delimiter", delimiter)
        .opt("temp_unit", temp_unit.as_deref())
        .opt("pressure_unit", pressure_unit.as_deref())
        .opt("depth_unit", depth_unit.as_deref())
        .finish()
}

fn header_json(h: &HeaderSource) -> String {
    let HeaderSource {
        path,
        kb_unit,
        sheet,
        delimiter,
        crs,
        xy_unit,
        key_col,
        x_col,
        y_col,
        label_col,
        kb_col,
        kb_match,
        datum_col,
        lat_col,
        lon_col,
        location_col,
        td_col,
    } = h;
    Obj::default()
        .path("path", path)
        .str("kb_unit", kb_unit)
        .opt("sheet", sheet.as_deref())
        .str("delimiter", delimiter)
        .opt("crs", crs.as_deref())
        .opt("xy_unit", xy_unit.as_deref())
        .strs("key_col", key_col)
        .strs("x_col", x_col)
        .strs("y_col", y_col)
        .strs("label_col", label_col)
        .strs("kb_col", kb_col)
        .opt("kb_match", kb_match.as_deref())
        .strs("datum_col", datum_col)
        .strs("lat_col", lat_col)
        .strs("lon_col", lon_col)
        .strs("location_col", location_col)
        .strs("td_col", td_col)
        .finish()
}

fn tops_json(t: &TopsSource) -> String {
    let TopsSource {
        path,
        depth_ref,
        depth_unit,
        sheet,
        delimiter,
        layout,
        repeat_pick,
        key_col,
        top_col,
        depth_col,
        top_columns,
        exclude_columns,
    } = t;
    let out = Obj::default()
        .path("path", path)
        .str("depth_ref", depth_ref)
        .str("depth_unit", depth_unit)
        .opt("sheet", sheet.as_deref())
        .str("delimiter", delimiter)
        .str("layout", layout)
        .str("repeat_pick", repeat_pick)
        .strs("key_col", key_col)
        .strs("top_col", top_col);
    let out = match depth_col {
        Some(v) => out.strs("depth_col", v),
        None => out,
    };
    let out = match top_columns {
        Some(v) => out.strs("top_columns", v),
        None => out,
    };
    out.strs("exclude_columns", exclude_columns).finish()
}

fn survey_json(s: &SurveySource) -> String {
    let SurveySource {
        path,
        md_unit,
        sheet,
        delimiter,
        key_col,
        md_col,
        inc_col,
        azi_col,
    } = s;
    Obj::default()
        .path("path", path)
        .str("md_unit", md_unit)
        .opt("sheet", sheet.as_deref())
        .str("delimiter", delimiter)
        .strs("key_col", key_col)
        .strs("md_col", md_col)
        .strs("inc_col", inc_col)
        .strs("azi_col", azi_col)
        .finish()
}

fn area_json(a: &AreaSource) -> String {
    let AreaSource {
        path,
        crs,
        layer,
        need_tops,
    } = a;
    Obj::default()
        .path("path", path)
        .opt("crs", crs.as_deref())
        .opt("layer", layer.as_deref())
        .raw("need_tops", need_tops.to_string())
        .finish()
}

/// Compact JSON object text, members in insertion order.
#[derive(Default)]
struct Obj(Vec<String>);

impl Obj {
    /// Member whose value is already JSON text.
    fn raw(mut self, key: &str, json: String) -> Self {
        self.0.push(format!("{}:{json}", json_string(key)));
        self
    }

    fn str(self, key: &str, value: &str) -> Self {
        self.raw(key, json_string(value))
    }

    /// Omitted when `None` (serde `skip_serializing_if = "Option::is_none"`).
    fn opt(self, key: &str, value: Option<&str>) -> Self {
        match value {
            Some(v) => self.str(key, v),
            None => self,
        }
    }

    fn strs(self, key: &str, values: &[String]) -> Self {
        self.raw(key, json_array(values.iter().map(|v| json_string(v))))
    }

    /// Path as text; non-UTF-8 parts are replaced with U+FFFD.
    fn path(self, key: &str, value: &Path) -> Self {
        self.str(key, &value.to_string_lossy())
    }

    fn finish(self) -> String {
        format!("{{{}}}", self.0.join(","))
    }
}

fn json_array(items: impl Iterator<Item = String>) -> String {
    format!("[{}]", items.collect::<Vec<_>>().join(","))
}

/// `s` as a JSON string literal (RFC 8259 escapes).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
