// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! CLI integration tests. Each test runs the `subsurfix` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const STAGES: [&str; 10] = [
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
];

struct Project {
    dir: tempfile::TempDir,
    config: PathBuf,
}

fn subsurfix() -> Command {
    Command::new(env!("CARGO_BIN_EXE_subsurfix"))
}

fn run(cmd: &mut Command) -> Output {
    cmd.output().expect("spawn subsurfix")
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf-8")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr utf-8")
}

fn init_project() -> Project {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run(subsurfix().arg("init").arg(dir.path()));
    assert_eq!(
        output.status.code(),
        Some(0),
        "init stderr: {}",
        stderr(&output)
    );
    let config = dir.path().join("project.toml");
    Project { dir, config }
}

fn columns(stage: &str, manifest: &str, outputs: &str, state: &str) -> String {
    format!("{:<12} {:<8} {:<8} {}", stage, manifest, outputs, state)
}

#[test]
fn version_command_and_flag() {
    let expected = format!("subsurfix {}\n", env!("CARGO_PKG_VERSION"));
    let version = run(subsurfix().arg("version"));
    assert_eq!(version.status.code(), Some(0));
    assert_eq!(stdout(&version), expected);
    let flag = run(subsurfix().arg("--version"));
    assert_eq!(flag.status.code(), Some(0), "stderr: {}", stderr(&flag));
    assert_eq!(stdout(&flag), expected, "stderr: {}", stderr(&flag));
}

#[test]
fn init_creates_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run(subsurfix().arg("init").arg(dir.path()));
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    assert!(dir.path().join("project.toml").is_file());
    assert!(
        stdout(&output).starts_with("created "),
        "{}",
        stdout(&output)
    );
}

#[test]
fn init_refuses_overwrite() {
    let project = init_project();
    let output = run(subsurfix().arg("init").arg(project.dir.path()));
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(
        stderr(&output).contains("already exists"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn init_template_loads() {
    let project = init_project();
    sx_core::config::load_project(&project.config).expect("template loads");
}

#[test]
fn run_full_pipeline() {
    let dir = tempfile::tempdir().expect("tempdir");
    let spec = sx_testutil::synth_project::SynthProjectSpec {
        n_wells: 6,
        seed: 0,
    };
    let config = sx_testutil::synth_project::write_synthetic_project(dir.path(), &spec)
        .expect("synthetic project");
    let output = run(subsurfix().args(["run", "--config"]).arg(&config));
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    assert_eq!(
        stdout(&output),
        format!("ran: {}\nskipped: -\n", STAGES.join(", "))
    );
}

#[test]
fn run_unknown_stage() {
    let project = init_project();
    let output = run(subsurfix()
        .args(["run", "--config"])
        .arg(&project.config)
        .args(["--from", "bogus"]));
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(
        stderr(&output).contains("unknown stage"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn run_from_after_to() {
    let project = init_project();
    let output = run(subsurfix()
        .args(["run", "--config"])
        .arg(&project.config)
        .args(["--from", "fill", "--to", "qc"]));
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(
        stderr(&output).contains("comes after"),
        "{}",
        stderr(&output)
    );

    let output = run(subsurfix()
        .args(["run", "--config"])
        .arg(&project.config)
        .args(["--threads", "0"]));
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(stderr(&output).contains("--threads"), "{}", stderr(&output));
}

#[test]
fn run_missing_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run(subsurfix()
        .current_dir(dir.path())
        .args(["run", "--config", "nope.toml"]));
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(stderr(&output).contains("not found"), "{}", stderr(&output));
}

#[test]
fn status_without_manifest() {
    let project = init_project();
    let output = run(subsurfix()
        .args(["status", "--config"])
        .arg(&project.config));
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    let text = stdout(&output);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 13, "{text}");
    let project_dir = std::fs::canonicalize(project.dir.path()).expect("canonicalize");
    assert_eq!(lines[0], format!("project: {}", project_dir.display()));
    assert_eq!(lines[1], "manifest: none");
    assert_eq!(lines[2], columns("stage", "manifest", "outputs", "state"));
    for (offset, name) in STAGES.iter().enumerate() {
        let line = lines[3 + offset];
        assert!(line.starts_with(name), "{line}");
        assert!(line.ends_with("missing"), "{line}");
    }
}

#[test]
fn status_with_manifest() {
    let project = init_project();
    write_manifest(project.dir.path());
    let wells = project.dir.path().join("stages/ingest/wells.parquet");
    std::fs::create_dir_all(wells.parent().expect("parent")).expect("mkdir");
    std::fs::write(&wells, b"x").expect("write wells");

    let incomplete = run(subsurfix()
        .args(["status", "--config"])
        .arg(&project.config));
    assert_eq!(
        incomplete.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&incomplete)
    );
    let ingest = columns("ingest", "yes", "1/2", "incomplete");
    assert!(
        stdout(&incomplete).lines().any(|line| line == ingest),
        "{}",
        stdout(&incomplete)
    );

    let geometry = project.dir.path().join("stages/ingest/geometry.parquet");
    std::fs::write(&geometry, b"x").expect("write geometry");
    let done = run(subsurfix()
        .args(["status", "--config"])
        .arg(&project.config));
    assert_eq!(done.status.code(), Some(0), "stderr: {}", stderr(&done));
    let ingest_done = columns("ingest", "yes", "2/2", "done");
    assert!(
        stdout(&done).lines().any(|line| line == ingest_done),
        "{}",
        stdout(&done)
    );
}

#[test]
fn status_bad_manifest() {
    let project = init_project();
    std::fs::write(
        project.dir.path().join("manifest.json"),
        r#"{"schema": "other"}"#,
    )
    .expect("write manifest");
    let output = run(subsurfix()
        .args(["status", "--config"])
        .arg(&project.config));
    assert_eq!(output.status.code(), Some(1), "stderr: {}", stderr(&output));
    assert!(
        stderr(&output).contains("format error"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn status_ignores_unknown_stage_keys() {
    let project = init_project();
    std::fs::write(
        project.dir.path().join("manifest.json"),
        r#"{"schema": "subsurfix.manifest/1", "stages": {"zzz": {"nope": 1}}}"#,
    )
    .expect("write manifest");
    let output = run(subsurfix()
        .args(["status", "--config"])
        .arg(&project.config));
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("missing"), "{}", stdout(&output));
}

#[test]
fn no_subcommand() {
    let output = run(&mut subsurfix());
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
}

fn write_manifest(dir: &Path) {
    let body = r#"{"schema": "subsurfix.manifest/1", "stages": {"ingest": {"outputs": {"stages/ingest/wells.parquet": "00", "stages/ingest/geometry.parquet": "00"}}}}"#;
    std::fs::write(dir.join("manifest.json"), body).expect("write manifest");
}
