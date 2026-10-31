// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Paths and JSON loaders for committed and external fixtures.

use std::path::{Path, PathBuf};

/// Environment variable that overrides [`data_fixtures_dir`].
pub const DATA_FIXTURES_ENV: &str = "SX_DATA_FIXTURES";

/// `subsurfix/` directory: two parents above this crate's manifest.
///
/// Panics if either parent directory is missing.
pub fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Some(crates_dir) = manifest.parent() else {
        panic!("workspace root: {} has no parent", manifest.display());
    };
    let Some(root) = crates_dir.parent() else {
        panic!("workspace root: {} has no parent", crates_dir.display());
    };
    root.to_path_buf()
}

/// `workspace_root()/tests/fixtures/<name>`.
///
/// `name` may contain `/` (for example `qc/ring_dev.json`).
/// Panics if the file does not exist.
pub fn fixture_path(name: &str) -> PathBuf {
    let path = workspace_root().join("tests/fixtures").join(name);
    if !path.exists() {
        panic!(
            "fixture {} not found (tiny fixtures live in subsurfix/tests/fixtures/)",
            path.display()
        );
    }
    path
}

/// Directory of bigger fixtures.
///
/// Uses `SX_DATA_FIXTURES` when that variable is set and non-empty. Otherwise
/// `workspace_root()/../data/fixtures`. Does not check that the directory exists.
pub fn data_fixtures_dir() -> PathBuf {
    match std::env::var(DATA_FIXTURES_ENV) {
        Ok(value) if !value.is_empty() => PathBuf::from(value),
        _ => workspace_root().join("../data/fixtures"),
    }
}

/// `data_fixtures_dir()/<name>`.
///
/// Panics, with setup advice, if the file does not exist. A missing data
/// fixture fails the test; it does not skip.
pub fn data_fixture_path(name: &str) -> PathBuf {
    let path = data_fixtures_dir().join(name);
    if !path.exists() {
        let default = workspace_root().join("../data/fixtures");
        panic!(
            "data fixture {} not found. Set SX_DATA_FIXTURES or use the default {}. Tests that need data fixtures are #[ignore]; run them with: cargo test --workspace -- --ignored data_",
            path.display(),
            default.display()
        );
    }
    path
}

/// Read and parse a JSON file.
///
/// Panics with the path and the error if the file cannot be read or parsed.
pub fn read_json(path: &Path) -> serde_json::Value {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => panic!("failed to read {}: {err}", path.display()),
    };
    match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(err) => panic!("failed to parse {}: {err}", path.display()),
    }
}

/// `read_json` of [`fixture_path`]`(name)`.
pub fn load_json(name: &str) -> serde_json::Value {
    read_json(&fixture_path(name))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        data_fixtures_dir, fixture_path, load_json, read_json, workspace_root, DATA_FIXTURES_ENV,
    };

    #[test]
    fn smoke_fixture_values_null() {
        let path = fixture_path("testutil_smoke.json");
        assert!(path.is_file(), "{}", path.display());
        for value in [read_json(&path), load_json("testutil_smoke.json")] {
            let values = value
                .get("values")
                .and_then(|item| item.as_array())
                .expect("values array");
            assert_eq!(values.get(2).map(serde_json::Value::is_null), Some(true));
        }
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn missing_fixture_panics() {
        let _ = fixture_path("nope.json");
    }

    #[test]
    fn data_dir_follows_env_or_default() {
        let got = data_fixtures_dir();
        match std::env::var(DATA_FIXTURES_ENV) {
            Ok(value) if !value.is_empty() => assert_eq!(got, PathBuf::from(value)),
            _ => assert_eq!(got, workspace_root().join("../data/fixtures")),
        }
    }
}
