// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! End-to-end pipeline run on a small synthetic project.

use std::path::{Path, PathBuf};
use std::time::Instant;

use sx_core::config::load_project;
use sx_core::pipeline::{run, stage_products, RunOptions, RunSummary, Stage, StageAction};
use sx_testutil::synth_project::{write_synthetic_project, SynthProjectSpec};

fn actions(s: &RunSummary) -> Vec<StageAction> {
    s.stages.iter().map(|o| o.action).collect()
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            files(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn history_len(dir: &Path) -> usize {
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    let m: serde_json::Value = serde_json::from_str(&text).unwrap();
    m["history"].as_array().unwrap().len()
}

#[test]
fn fresh_skip_and_learning_rate_rerun() {
    let t0 = Instant::now();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let spec = SynthProjectSpec {
        n_wells: 8,
        seed: 0,
    };
    let toml = write_synthetic_project(root, &spec).unwrap();
    let opts = RunOptions::new(root);

    // Fresh run: every stage runs and writes every listed product.
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    assert_eq!(actions(&s), vec![StageAction::Ran; 10]);
    for (stage, names) in stage_products() {
        let folder = match stage {
            Stage::Export => root.join("export"),
            other => root.join("stages").join(other.name()),
        };
        for name in names {
            assert!(folder.join(name).is_file(), "{stage:?}: {name} missing");
        }
    }
    assert!(root.join("stages/fill/models/GR.txt").is_file());
    let las: Vec<_> = std::fs::read_dir(root.join("export/las"))
        .unwrap()
        .collect();
    assert_eq!(las.len(), spec.n_wells);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    for (stage, rec) in manifest["stages"].as_object().unwrap() {
        for rel in rec["outputs"].as_object().unwrap().keys() {
            assert!(root.join(rel).is_file(), "{stage}: {rel} missing");
        }
    }
    assert_eq!(history_len(root), 10);
    assert!(s.run_log.is_file());
    assert!(!root.join(".subsurfix.lock").exists());
    let mut all = Vec::new();
    files(root, &mut all);
    assert!(all.iter().all(|p| p.extension().is_none_or(|e| e != "tmp")));

    // Second run, nothing changed: every stage skips.
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    assert_eq!(actions(&s), vec![StageAction::Skipped; 10]);
    assert_eq!(history_len(root), 10);

    // A new learning rate reruns fill and everything after it.
    let text = std::fs::read_to_string(&toml).unwrap();
    assert!(text.contains("learning_rate = 0.1\n"));
    std::fs::write(
        &toml,
        text.replace("learning_rate = 0.1\n", "learning_rate = 0.05\n"),
    )
    .unwrap();
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    let mut want = vec![StageAction::Skipped; 5];
    want.extend([StageAction::Ran; 5]);
    assert_eq!(actions(&s), want);
    let log = std::fs::read_to_string(&s.run_log).unwrap();
    assert!(
        log.contains("fill stale re-running: config changed"),
        "{log}"
    );
    assert!(
        log.contains("refresh stale re-running: an upstream stage re-ran"),
        "{log}"
    );

    // Unchanged config again: everything reused.
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    assert_eq!(actions(&s), vec![StageAction::Skipped; 10]);

    // A schema / format bump recorded for qc (older format saved) reruns qc
    // and everything after it.
    let edit_manifest = |f: &dyn Fn(&mut serde_json::Value)| {
        let path = root.join("manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        f(&mut m);
        std::fs::write(&path, serde_json::to_string_pretty(&m).unwrap()).unwrap();
    };
    let qc_version = Stage::Qc.version();
    edit_manifest(&|m| m["stages"]["qc"]["version"] = (qc_version - 1).into());
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    let mut want = vec![StageAction::Skipped];
    want.extend([StageAction::Ran; 9]);
    assert_eq!(actions(&s), want);
    let log = std::fs::read_to_string(&s.run_log).unwrap();
    assert!(
        log.contains(&format!(
            "qc stale re-running: format version {} -> {qc_version}",
            qc_version - 1
        )),
        "{log}"
    );

    // A package version change reruns everything.
    edit_manifest(&|m| m["stages"]["ingest"]["package_version"] = "0.0.0-old".into());
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    assert_eq!(actions(&s), vec![StageAction::Ran; 10]);
    let log = std::fs::read_to_string(&s.run_log).unwrap();
    assert!(
        log.contains("ingest stale re-running: package version 0.0.0-old -> "),
        "{log}"
    );

    // An old cache without version stamps counts as stale.
    edit_manifest(&|m| {
        for rec in m["stages"].as_object_mut().unwrap().values_mut() {
            let r = rec.as_object_mut().unwrap();
            r.remove("package_version");
            r.remove("upstream_key");
        }
    });
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    assert_eq!(actions(&s), vec![StageAction::Ran; 10]);
    let log = std::fs::read_to_string(&s.run_log).unwrap();
    assert!(
        log.contains("ingest stale re-running: saved output has no version stamp"),
        "{log}"
    );
    let s = run(&load_project(&toml).unwrap(), &opts).unwrap();
    assert_eq!(actions(&s), vec![StageAction::Skipped; 10]);

    eprintln!(
        "pipeline_e2e: three runs in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
}
