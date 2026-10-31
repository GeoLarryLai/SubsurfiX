// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `subsurfix status`.

use std::path::Path;

use serde_json::{Map, Value};
use sx_core::{SxError, SxResult};

use crate::cli::{io_error, resolve_config};
use crate::stages::Stage;

/// List stages and whether manifest entries and output files exist.
///
/// The project file is not parsed. A missing manifest prints every stage as
/// `missing`. This does not hash files and does not judge staleness.
pub fn execute(config: &Path) -> SxResult<()> {
    let (_, project_dir) = resolve_config(config)?;
    let manifest_path = project_dir.join("manifest.json");
    let manifest_present = manifest_path.is_file();
    let rows = if manifest_present {
        let stages = read_manifest(&manifest_path)?;
        stage_rows(&stages, &project_dir)?
    } else {
        Stage::ALL
            .iter()
            .map(|stage| columns(stage.name(), "no", "-", "missing"))
            .collect()
    };
    println!("project: {}", project_dir.display());
    if manifest_present {
        println!("manifest: manifest.json");
    } else {
        println!("manifest: none");
    }
    println!("{}", header());
    for row in rows {
        println!("{row}");
    }
    Ok(())
}

fn header() -> String {
    columns("stage", "manifest", "outputs", "state")
}

fn columns(stage: &str, manifest: &str, outputs: &str, state: &str) -> String {
    format!("{:<12} {:<8} {:<8} {}", stage, manifest, outputs, state)
}

fn stage_rows(stages: &Map<String, Value>, project_dir: &Path) -> SxResult<Vec<String>> {
    let mut rows = Vec::with_capacity(Stage::ALL.len());
    for stage in Stage::ALL {
        rows.push(stage_row(stage, stages, project_dir)?);
    }
    Ok(rows)
}

fn stage_row(stage: Stage, stages: &Map<String, Value>, project_dir: &Path) -> SxResult<String> {
    let name = stage.name();
    let Some(entry) = stages.get(name) else {
        return Ok(columns(name, "no", "-", "missing"));
    };
    let outputs = outputs_object(entry, name)?;
    let n = outputs.len();
    let mut present = 0usize;
    for key in outputs.keys() {
        if project_dir.join(key).is_file() {
            present += 1;
        }
    }
    let ratio = format!("{present}/{n}");
    let state = if n > 0 && present == n {
        "done"
    } else {
        "incomplete"
    };
    Ok(columns(name, "yes", &ratio, state))
}

fn read_manifest(path: &Path) -> SxResult<Map<String, Value>> {
    let bytes = std::fs::read(path).map_err(|source| io_error(path, source))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|err| SxError::Format(format!("manifest.json: {err}")))?;
    let Some(obj) = value.as_object() else {
        return Err(SxError::Format(
            "manifest.json: top level is not an object".to_owned(),
        ));
    };
    if obj.get("schema").and_then(Value::as_str) != Some("subsurfix.manifest/1") {
        return Err(SxError::Format(
            "manifest.json: schema is not \"subsurfix.manifest/1\"".to_owned(),
        ));
    }
    let Some(stages) = obj.get("stages").and_then(Value::as_object) else {
        return Err(SxError::Format(
            "manifest.json: stages is missing or not an object".to_owned(),
        ));
    };
    for (name, entry) in stages {
        // Keys that are not known stage names are ignored.
        if Stage::from_name(name).is_ok() {
            outputs_object(entry, name)?;
        }
    }
    Ok(stages.clone())
}

fn outputs_object<'a>(entry: &'a Value, name: &str) -> SxResult<&'a Map<String, Value>> {
    let Some(obj) = entry.as_object() else {
        return Err(SxError::Format(format!(
            "manifest.json: stage {name} is not an object"
        )));
    };
    obj.get("outputs")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            SxError::Format(format!(
                "manifest.json: stage {name} outputs is missing or not an object"
            ))
        })
}
