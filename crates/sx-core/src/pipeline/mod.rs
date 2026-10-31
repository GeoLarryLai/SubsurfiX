// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Pipeline runner.
//!
//! Runs the ten stages in fixed order on a project folder, skips a stage when
//! its stage key and its recorded outputs are unchanged, writes every product
//! through `<name>.tmp` + rename, and records each run in `manifest.json` and
//! `logs/run_<utc>.log`. Stages always read their inputs from files.

pub(crate) mod assemble;
mod atomic;
pub(crate) mod keys;
mod manifest;
mod models;
pub(crate) mod products;
pub(crate) mod readers;
mod stage;
mod stages;
mod utc;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub use stage::Stage;

use atomic::{remove_stale, ProjectLock};
use keys::{canonical_json, sha256_hex, stage_config_value, stage_key, stamped_key, HashCache};
use manifest::{HistoryEntry, Manifest, StageRecord};
use products::{fixed_inputs, fixed_outputs, output_dir, MODELS_DIR};
use stages::{run_stage, Ctx};
use utc::{now_secs, utc_compact, utc_text};

use crate::config::ProjectConfig;
use crate::error::{SxError, SxResult};
use crate::fill::sequential::{RegressorFactory, TrainedModels};
use crate::io::atomic::io_error;
use crate::types::LogKind;

/// Per-log plug-in models: (descriptor such as `"py:mypkg.MeanModel"`, factory).
pub type ModelOverrides = BTreeMap<LogKind, (String, Arc<dyn RegressorFactory>)>;

/// Options for one run. `RunOptions::new(dir)` gives the defaults shown.
#[derive(Clone)]
pub struct RunOptions {
    /// Folder holding `project.toml`.
    pub project_dir: PathBuf,
    /// First stage to run; `None` = ingest.
    pub from: Option<Stage>,
    /// Last stage to run; `None` = export.
    pub to: Option<Stage>,
    /// Rerun every in-range stage even when up to date. Default false.
    pub force: bool,
    /// Rayon pool size; `None` = `config.runtime.threads`.
    pub threads: Option<usize>,
    /// Plug-in fill models; empty except from the Python face.
    pub models: ModelOverrides,
}

impl RunOptions {
    /// Defaults: whole range, no force, config threads, no plug-ins.
    pub fn new(project_dir: impl Into<PathBuf>) -> Self {
        Self {
            project_dir: project_dir.into(),
            from: None,
            to: None,
            force: false,
            threads: None,
            models: ModelOverrides::new(),
        }
    }
}

impl std::fmt::Debug for RunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let models: BTreeMap<&str, &str> = self
            .models
            .iter()
            .map(|(log, (d, _))| (log.name(), d.as_str()))
            .collect();
        f.debug_struct("RunOptions")
            .field("project_dir", &self.project_dir)
            .field("from", &self.from)
            .field("to", &self.to)
            .field("force", &self.force)
            .field("threads", &self.threads)
            .field("models", &models)
            .finish()
    }
}

/// What a run did with one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageAction {
    /// The stage ran and wrote its outputs.
    Ran,
    /// In range and up to date; nothing written.
    Skipped,
    /// Before `from`; verified up to date.
    Checked,
    /// After `to`; untouched.
    OutOfRange,
}

impl StageAction {
    /// `"ran"`, `"skipped"`, `"checked"`, `"out_of_range"`.
    pub fn as_str(self) -> &'static str {
        match self {
            StageAction::Ran => "ran",
            StageAction::Skipped => "skipped",
            StageAction::Checked => "checked",
            StageAction::OutOfRange => "out_of_range",
        }
    }
}

/// One stage of a [`RunSummary`].
#[derive(Debug, Clone, PartialEq)]
pub struct StageOutcome {
    /// Stage.
    pub stage: Stage,
    /// What happened.
    pub action: StageAction,
    /// Stage key (64 hex); `None` for `OutOfRange`.
    pub stage_key: Option<String>,
    /// Wall time in seconds, hashing included; 0.0 for `OutOfRange`.
    pub seconds: f64,
    /// Rows of `stages/ingest/wells.parquet` after the stage; 0 when unknown.
    pub n_wells: usize,
}

/// Result of [`run`].
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    /// Project folder.
    pub project_dir: PathBuf,
    /// `<project>/logs/run_<utc_compact>.log`.
    pub run_log: PathBuf,
    /// All ten stages, in order.
    pub stages: Vec<StageOutcome>,
}

/// Whether a stage would run now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageState {
    /// Recorded key and outputs match, and every earlier stage is up to date.
    UpToDate,
    /// Recorded, but the key, an output, or an earlier stage changed.
    Stale,
    /// No record in the manifest.
    NeverRun,
}

impl StageState {
    /// `"up_to_date"`, `"stale"`, `"never_run"`.
    pub fn as_str(self) -> &'static str {
        match self {
            StageState::UpToDate => "up_to_date",
            StageState::Stale => "stale",
            StageState::NeverRun => "never_run",
        }
    }
}

/// One stage of [`status`].
#[derive(Debug, Clone, PartialEq)]
pub struct StageStatus {
    /// Stage.
    pub stage: Stage,
    /// Code version of the stage.
    pub version: u32,
    /// State now.
    pub state: StageState,
    /// Stage key in the manifest, if any.
    pub recorded_key: Option<String>,
    /// Finish time in the manifest, if any.
    pub finished_utc: Option<String>,
}

/// Fixed product file names per stage, relative to `stages/<stage>/` (to
/// `export/` for export), without per-well LAS files and per-log model files.
pub fn stage_products() -> Vec<(Stage, Vec<&'static str>)> {
    Stage::ALL
        .iter()
        .map(|&s| {
            let dir = format!("{}/", output_dir(s));
            let names = fixed_outputs(s)
                .into_iter()
                .filter_map(|p| p.strip_prefix(dir.as_str()))
                .collect();
            (s, names)
        })
        .collect()
}

static NONCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `<utc_text>/<pid>/<n>`: unique within and across processes.
fn new_nonce() -> String {
    format!(
        "{}/{}/{}",
        utc_text(now_secs()),
        std::process::id(),
        NONCE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Manifest key of `path`: relative with `/` separators inside the project,
/// else the absolute path.
fn manifest_key(project: &Path, path: &Path) -> String {
    match path.strip_prefix(project) {
        Ok(rel) => rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => path.display().to_string(),
    }
}

/// Model text files listed in the fill record, as `(log, absolute path)`.
fn model_files(project: &Path, manifest: &Manifest) -> Vec<(LogKind, PathBuf)> {
    let prefix = format!("{MODELS_DIR}/");
    manifest
        .stages
        .get(Stage::Fill.name())
        .map(|rec| {
            rec.outputs
                .keys()
                .filter_map(|k| {
                    let name = k.strip_prefix(prefix.as_str())?.strip_suffix(".txt")?;
                    let log = LogKind::from_name(name).ok()?;
                    Some((log, project.join(k)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Inputs of `stage` as `(manifest key, path)`, sorted by key (byte order).
fn stage_inputs(
    stage: Stage,
    project: &Path,
    cfg: &ProjectConfig,
    manifest: &Manifest,
) -> SxResult<Vec<(String, PathBuf)>> {
    let mut v: Vec<(String, PathBuf)> = fixed_inputs(stage)
        .into_iter()
        .map(|r| (r.to_string(), project.join(r)))
        .collect();
    match stage {
        Stage::Ingest => {
            for path in crate::ingest::source_files(&cfg.ingest, project)? {
                v.push((manifest_key(project, &path), path));
            }
        }
        Stage::Refresh => {
            for (_, path) in model_files(project, manifest) {
                v.push((manifest_key(project, &path), path));
            }
        }
        _ => {}
    }
    v.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    v.dedup_by(|a, b| a.0 == b.0);
    Ok(v)
}

/// Hashes of `inputs` in order, or `Err(key)` of the first missing file.
fn hash_inputs(
    inputs: &[(String, PathBuf)],
    cache: &mut HashCache,
) -> SxResult<Result<Vec<String>, String>> {
    let mut out = Vec::with_capacity(inputs.len());
    for (key, path) in inputs {
        if !path.is_file() {
            return Ok(Err(key.clone()));
        }
        out.push(cache.hash(path)?);
    }
    Ok(Ok(out))
}

/// `(config_hash, stage_key)`.
fn compute_key(
    stage: Stage,
    cfg: &ProjectConfig,
    models: &ModelOverrides,
    nonce: &str,
    input_hashes: &[String],
) -> SxResult<(String, String)> {
    let value = stage_config_value(cfg, stage, models, nonce)?;
    let config_hash = sha256_hex(canonical_json(&value).as_bytes());
    let key = stage_key(stage.name(), stage.version(), &config_hash, input_hashes);
    Ok((config_hash, key))
}

/// Package version recorded in every stage stamp.
pub(crate) const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Why a saved stage is stale, or `None` when it can be reused: the record
/// must carry the same package version, format version, config hash,
/// upstream stamp and stage key, and every recorded output must exist with
/// the recorded hash. A record without a stamp (older manifest) is stale.
fn stale_reason(
    stage: Stage,
    record: Option<&StageRecord>,
    config_hash: &str,
    upstream_key: &str,
    key: &str,
    project: &Path,
    cache: &mut HashCache,
) -> SxResult<Option<String>> {
    let Some(rec) = record else {
        return Ok(Some("no saved output".into()));
    };
    if rec.package_version.is_empty() {
        return Ok(Some(
            "saved output has no version stamp (older cache)".into(),
        ));
    }
    if rec.package_version != PACKAGE_VERSION {
        return Ok(Some(format!(
            "package version {} -> {PACKAGE_VERSION}",
            rec.package_version
        )));
    }
    if rec.version != stage.version() {
        return Ok(Some(format!(
            "format version {} -> {}",
            rec.version,
            stage.version()
        )));
    }
    if rec.config_hash != config_hash {
        return Ok(Some("config changed".into()));
    }
    if rec.upstream_key != upstream_key {
        return Ok(Some("upstream stage changed".into()));
    }
    if rec.stage_key != key {
        return Ok(Some("inputs changed".into()));
    }
    for (rel, hash) in &rec.outputs {
        let path = project.join(rel);
        if !path.is_file() || cache.hash(&path)? != *hash {
            return Ok(Some(format!("output {rel} missing or modified")));
        }
    }
    Ok(None)
}

/// Rows of `stages/ingest/wells.parquet`; 0 when it cannot be read.
fn count_wells(project: &Path) -> usize {
    File::open(project.join(products::WELLS))
        .ok()
        .and_then(|f| ParquetRecordBatchReaderBuilder::try_new(f).ok())
        .and_then(|b| usize::try_from(b.metadata().file_metadata().num_rows()).ok())
        .unwrap_or(0)
}

/// `logs/run_<utc_compact>.log`, one flushed line per event.
struct RunLog {
    path: PathBuf,
    file: File,
}

impl RunLog {
    fn create(project: &Path) -> SxResult<Self> {
        let dir = project.join("logs");
        std::fs::create_dir_all(&dir).map_err(|e| io_error(&dir, e))?;
        let base = format!("run_{}", utc_compact(now_secs()));
        let mut n = 1u32;
        loop {
            let name = if n == 1 {
                format!("{base}.log")
            } else {
                format!("{base}_{n}.log")
            };
            let path = dir.join(name);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok(Self { path, file }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
                Err(e) => return Err(io_error(&path, e)),
            }
        }
    }

    /// `<utc_text> <stage|-> <event> <detail>`.
    fn event(&mut self, stage: &str, event: &str, detail: &str) -> SxResult<()> {
        let line = format!("{} {stage} {event} {detail}\n", utc_text(now_secs()));
        self.file
            .write_all(line.as_bytes())
            .and_then(|()| self.file.flush())
            .map_err(|e| io_error(&self.path, e))
    }
}

/// Run state carried from stage to stage.
struct RunState<'a> {
    cfg: &'a ProjectConfig,
    options: &'a RunOptions,
    project: &'a Path,
    from: Stage,
    nonce: String,
    manifest: Manifest,
    cache: HashCache,
    log: RunLog,
    ran_any: bool,
    trained: Option<Arc<TrainedModels>>,
    /// Stamp of the previous stage in this run (empty before ingest).
    upstream_key: String,
}

impl RunState<'_> {
    /// Check, skip or run one in-range or earlier stage.
    fn step(&mut self, stage: Stage) -> SxResult<(StageAction, String)> {
        let name = stage.name();
        let inputs = stage_inputs(stage, self.project, self.cfg, &self.manifest)?;
        let hashes = match hash_inputs(&inputs, &mut self.cache)? {
            Ok(h) => h,
            Err(_) if stage < self.from => return Err(not_up_to_date(stage, self.from)),
            Err(missing) => {
                return Err(SxError::NotFound(format!(
                    "{missing} (input of {name}); run earlier stages"
                )))
            }
        };
        let (config_hash, base_key) =
            compute_key(stage, self.cfg, &self.options.models, &self.nonce, &hashes)?;
        let upstream_key = self.upstream_key.clone();
        let key = stamped_key(&base_key, PACKAGE_VERSION, &upstream_key);
        self.upstream_key = key.clone();
        let reason = stale_reason(
            stage,
            self.manifest.stages.get(name),
            &config_hash,
            &upstream_key,
            &key,
            self.project,
            &mut self.cache,
        )?;
        if stage < self.from {
            if let Some(why) = reason {
                return Err(SxError::InvalidInput(format!(
                    "stage {name} is not up to date ({why}); cannot start at {}",
                    self.from.name()
                )));
            }
            return Ok((StageAction::Checked, key));
        }
        if reason.is_none() && !self.options.force && !self.ran_any {
            return Ok((StageAction::Skipped, key));
        }
        let why = if self.options.force {
            "forced".to_string()
        } else if self.ran_any {
            "an upstream stage re-ran".to_string()
        } else {
            reason.unwrap_or_else(|| "forced".to_string())
        };
        tracing::info!(stage = name, reason = %why, "stage stale, re-running");
        self.log
            .event(name, "stale", &format!("re-running: {why}"))?;

        let started_utc = utc_text(now_secs());
        let ctx = Ctx {
            project: self.project,
            cfg: self.cfg,
            models: &self.options.models,
            trained: self.trained.clone(),
            model_files: model_files(self.project, &self.manifest),
        };
        let run = match run_stage(stage, &ctx) {
            Ok(run) => run,
            Err(e) => {
                self.manifest.stages.remove(name);
                self.manifest.history.push(HistoryEntry {
                    stage: name.to_string(),
                    action: "failed".into(),
                    stage_key: key,
                    started_utc,
                    finished_utc: utc_text(now_secs()),
                    n_wells: count_wells(self.project),
                    error: Some(e.to_string()),
                });
                self.manifest.save(self.project)?;
                return Err(e);
            }
        };
        let mut keep = BTreeSet::new();
        for (rel, hash) in &run.outputs {
            let path = self.project.join(rel);
            self.cache.insert(path.clone(), hash.clone());
            keep.insert(path);
        }
        for gone in remove_stale(&self.project.join(output_dir(stage)), &keep)? {
            self.cache.forget(&gone);
        }
        if run.trained.is_some() {
            self.trained = run.trained;
        }
        let n_wells = count_wells(self.project);
        let finished_utc = utc_text(now_secs());
        self.manifest.stages.insert(
            name.to_string(),
            StageRecord {
                version: stage.version(),
                package_version: PACKAGE_VERSION.to_string(),
                upstream_key,
                stage_key: key.clone(),
                config_hash,
                inputs: inputs.iter().map(|(k, _)| k.clone()).zip(hashes).collect(),
                outputs: run.outputs.into_iter().collect(),
                started_utc: started_utc.clone(),
                finished_utc: finished_utc.clone(),
                n_wells,
            },
        );
        self.manifest.history.push(HistoryEntry {
            stage: name.to_string(),
            action: StageAction::Ran.as_str().into(),
            stage_key: key.clone(),
            started_utc,
            finished_utc,
            n_wells,
            error: None,
        });
        self.manifest.save(self.project)?;
        self.ran_any = true;
        Ok((StageAction::Ran, key))
    }
}

fn not_up_to_date(stage: Stage, from: Stage) -> SxError {
    SxError::InvalidInput(format!(
        "stage {} is not up to date; cannot start at {}",
        stage.name(),
        from.name()
    ))
}

/// Run the pipeline on `options.project_dir`.
///
/// Builds the thread pool (`options.threads`, else `config.runtime.threads`),
/// takes the project lock, then per stage in order: before `from` it must be
/// up to date (`Checked`); after `to` it is untouched (`OutOfRange`); in range
/// it is `Skipped` when up to date, not forced and no earlier stage ran in
/// this run, else it runs (`Ran`), its outputs are written atomically, stale
/// products in its folder are deleted and the manifest is saved. No samples
/// are touched here, so there is no NaN rule.
///
/// # Errors
/// `SxError::InvalidInput` (`from` after `to`, a stage before `from` not up to
/// date, project locked), `SxError::NotFound` (an input of an in-range stage
/// is missing), `SxError::Config` (no thread pool can be built; a second
/// project with another thread count runs on its own pool), `SxError::Format` (bad manifest), `SxError::Io`, and any stage error
/// (recorded as a `failed` history entry; the stage record is removed).
pub fn run(config: &ProjectConfig, options: &RunOptions) -> SxResult<RunSummary> {
    let config = &crate::nphi_offset::effective_config(&options.project_dir, config)?;
    crate::runtime::with_threads(options.threads.unwrap_or(config.runtime.threads), || {
        run_on_pool(config, options)
    })
}

fn run_on_pool(config: &ProjectConfig, options: &RunOptions) -> SxResult<RunSummary> {
    let from = options.from.unwrap_or(Stage::Ingest);
    let to = options.to.unwrap_or(Stage::Export);
    if from > to {
        return Err(SxError::InvalidInput(format!(
            "--from {} comes after --to {}",
            from.name(),
            to.name()
        )));
    }
    let project = options.project_dir.as_path();
    let _lock = ProjectLock::acquire(project)?;
    let manifest = Manifest::load(project)?;
    let mut log = RunLog::create(project)?;
    log.event(
        "-",
        "start",
        &format!(
            "from={} to={} force={}",
            from.name(),
            to.name(),
            options.force
        ),
    )?;
    let mut st = RunState {
        cfg: config,
        options,
        project,
        from,
        nonce: new_nonce(),
        manifest,
        cache: HashCache::default(),
        log,
        ran_any: false,
        trained: None,
        upstream_key: String::new(),
    };
    let mut outcomes = Vec::with_capacity(Stage::ALL.len());
    for stage in Stage::ALL {
        if stage > to {
            outcomes.push(StageOutcome {
                stage,
                action: StageAction::OutOfRange,
                stage_key: None,
                seconds: 0.0,
                n_wells: 0,
            });
            continue;
        }
        let span = tracing::info_span!("stage", stage = stage.name());
        let _guard = span.enter();
        let t0 = Instant::now();
        let (action, key) = match st.step(stage) {
            Ok(done) => done,
            Err(e) => {
                st.log.event(stage.name(), "failed", &e.to_string())?;
                return Err(e);
            }
        };
        let seconds = t0.elapsed().as_secs_f64();
        let n_wells = count_wells(project);
        tracing::info!(action = action.as_str(), seconds, n_wells, "stage done");
        st.log.event(
            stage.name(),
            action.as_str(),
            &format!("seconds={seconds:.3} n_wells={n_wells} key={key}"),
        )?;
        outcomes.push(StageOutcome {
            stage,
            action,
            stage_key: Some(key),
            seconds,
            n_wells,
        });
    }
    let ran = outcomes
        .iter()
        .filter(|o| o.action == StageAction::Ran)
        .count();
    st.log.event("-", "end", &format!("ran={ran}"))?;
    Ok(RunSummary {
        project_dir: options.project_dir.clone(),
        run_log: st.log.path.clone(),
        stages: outcomes,
    })
}

/// What [`run`] would do now, per stage, without running, locking or building
/// the thread pool. Uses `options.project_dir` and `options.models` only.
///
/// A stage after one that is not up to date is `Stale` when it has a record
/// (the run would rerun it). No samples are touched, so there is no NaN rule.
///
/// # Errors
/// `SxError::Format` (bad manifest), `SxError::Io`, `SxError::Config`.
pub fn status(config: &ProjectConfig, options: &RunOptions) -> SxResult<Vec<StageStatus>> {
    let project = options.project_dir.as_path();
    let config = &crate::nphi_offset::effective_config(project, config)?;
    let manifest = Manifest::load(project)?;
    let nonce = new_nonce();
    let mut cache = HashCache::default();
    let mut dirty = false;
    let mut upstream = String::new();
    let mut out = Vec::with_capacity(Stage::ALL.len());
    for stage in Stage::ALL {
        let record = manifest.stages.get(stage.name());
        let state = match record {
            None => StageState::NeverRun,
            Some(_) if dirty => StageState::Stale,
            Some(_) => {
                let fresh = match stage_inputs(stage, project, config, &manifest)
                    .and_then(|inputs| hash_inputs(&inputs, &mut cache))
                {
                    Ok(Ok(hashes)) => {
                        let (config_hash, base) =
                            compute_key(stage, config, &options.models, &nonce, &hashes)?;
                        let key = stamped_key(&base, PACKAGE_VERSION, &upstream);
                        let why = stale_reason(
                            stage,
                            record,
                            &config_hash,
                            &upstream,
                            &key,
                            project,
                            &mut cache,
                        )?;
                        upstream = key;
                        why.is_none()
                    }
                    Ok(Err(_)) | Err(_) => false,
                };
                if fresh {
                    StageState::UpToDate
                } else {
                    StageState::Stale
                }
            }
        };
        dirty |= state != StageState::UpToDate;
        out.push(StageStatus {
            stage,
            version: stage.version(),
            state,
            recorded_key: record.map(|r| r.stage_key.clone()),
            finished_utc: record.map(|r| r.finished_utc.clone()),
        });
    }
    Ok(out)
}
