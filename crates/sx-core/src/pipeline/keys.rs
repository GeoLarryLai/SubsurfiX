// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Canonical JSON, config hash, file SHA-256, stage key and the in-run hash cache.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::stage::Stage;
use super::ModelOverrides;
use crate::config::ProjectConfig;
use crate::error::{SxError, SxResult};
use crate::io::atomic::io_error;

/// Read buffer of [`sha256_file`].
const HASH_BUFFER: usize = 1 << 20;

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push(':');
                if let Some(item) = map.get(*key) {
                    write_canonical(item, out);
                }
            }
            out.push('}');
        }
        leaf => out.push_str(&leaf.to_string()),
    }
}

/// Canonical JSON: object keys sorted by UTF-8 bytes at every depth, no
/// whitespace, strings escaped as `serde_json` does, numbers as their
/// `serde_json::Number` text (`5.0`, `0.01`, `8000.0`).
pub(crate) fn canonical_json(v: &Value) -> String {
    let mut out = String::new();
    write_canonical(v, &mut out);
    out
}

/// Lowercase hex SHA-256 of `bytes` (64 chars).
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Lowercase hex SHA-256 of a file's bytes, read in 1 MiB blocks.
///
/// # Errors
/// `SxError::Io` when the file cannot be opened or read.
pub(crate) fn sha256_file(path: &Path) -> SxResult<String> {
    let mut file = std::fs::File::open(path).map_err(|e| io_error(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUFFER];
    loop {
        let n = file.read(&mut buf).map_err(|e| io_error(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(buf.get(..n).unwrap_or_default());
    }
    Ok(hex::encode(hasher.finalize()))
}

/// `sha256_hex(name + "\n" + version + "\n" + config_hash + "\n" + input_hashes.join("\n"))`.
pub(crate) fn stage_key(
    name: &str,
    version: u32,
    config_hash: &str,
    input_hashes: &[String],
) -> String {
    let text = format!(
        "{name}\n{version}\n{config_hash}\n{}",
        input_hashes.join("\n")
    );
    sha256_hex(text.as_bytes())
}

/// Version stamp of a stage: [`stage_key`] extended with the package
/// version and the upstream stage's stamp, so a new release, a stage format
/// bump, a config change or any upstream change gives a new stamp.
pub(crate) fn stamped_key(base_key: &str, package_version: &str, upstream_key: &str) -> String {
    sha256_hex(format!("{base_key}\n{package_version}\n{upstream_key}").as_bytes())
}

/// The JSON value hashed into a stage's config hash.
///
/// One listed section gives that object, several give `{section: value, ...}`;
/// then `"_seed"` is added. For `fill` with plug-ins, `"_plugins"` (log name →
/// descriptor) and `"_nonce"` are added so the fill key never matches.
///
/// # Errors
/// `SxError::Config` when the config does not serialize or a section is missing
/// or not an object.
pub(crate) fn stage_config_value(
    cfg: &ProjectConfig,
    stage: Stage,
    models: &ModelOverrides,
    nonce: &str,
) -> SxResult<Value> {
    let root = serde_json::to_value(cfg).map_err(|e| SxError::Config(e.to_string()))?;
    let section = |name: &str| -> SxResult<Value> {
        match root.get(name) {
            Some(v) if v.is_object() => Ok(v.clone()),
            _ => Err(SxError::Config(format!(
                "config section {name:?} is missing or not a table"
            ))),
        }
    };
    let mut value = match stage.config_sections() {
        [one] => section(one)?,
        many => {
            let mut map = Map::new();
            for name in many {
                map.insert((*name).to_string(), section(name)?);
            }
            Value::Object(map)
        }
    };
    let Some(obj) = value.as_object_mut() else {
        return Err(SxError::Config(format!(
            "config of stage {} is not a table",
            stage.name()
        )));
    };
    obj.insert("_seed".into(), Value::from(cfg.seed));
    if stage == Stage::Fill && !models.is_empty() {
        let plugins: Map<String, Value> = models
            .iter()
            .map(|(log, (descriptor, _))| (log.name().to_string(), Value::from(descriptor.clone())))
            .collect();
        obj.insert("_plugins".into(), Value::Object(plugins));
        obj.insert("_nonce".into(), Value::from(nonce));
    }
    Ok(value)
}

/// SHA-256 of files, one run only: a file is hashed once; files the run writes
/// enter with the hash computed at write time. Nothing is kept across runs.
#[derive(Debug, Default)]
pub(crate) struct HashCache {
    map: BTreeMap<PathBuf, String>,
}

impl HashCache {
    /// Hash of `path` (from the cache, or hashed now).
    ///
    /// # Errors
    /// As [`sha256_file`].
    pub(crate) fn hash(&mut self, path: &Path) -> SxResult<String> {
        if let Some(h) = self.map.get(path) {
            return Ok(h.clone());
        }
        let h = sha256_file(path)?;
        self.map.insert(path.to_path_buf(), h.clone());
        Ok(h)
    }

    /// Record the hash of a file this run just wrote.
    pub(crate) fn insert(&mut self, path: PathBuf, hash: String) {
        self.map.insert(path, hash);
    }

    /// Drop a deleted file.
    pub(crate) fn forget(&mut self, path: &Path) {
        self.map.remove(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProjectConfig;

    #[test]
    fn examples_1_to_4() {
        let refresh = serde_json::json!({"refresh_tol_ft": 5.0, "percentile": 95.0, "_seed": 0});
        assert_eq!(
            canonical_json(&refresh),
            r#"{"_seed":0,"percentile":95.0,"refresh_tol_ft":5.0}"#
        );
        let mut cfg = ProjectConfig::default_with_crs("EPSG:32614");
        cfg.refresh.refresh_tol_ft = 5.0;
        cfg.refresh.percentile = 95.0;
        let v = stage_config_value(&cfg, Stage::Refresh, &ModelOverrides::new(), "").unwrap();
        let config_hash = sha256_hex(canonical_json(&v).as_bytes());
        assert_eq!(
            config_hash,
            "f0aad4e9062fab3dffadab03ca643ee2162fca12c88710aa24cd587a1c5c0ffb"
        );
        let inputs = vec![sha256_hex(b"tau"), sha256_hex(b"filled")];
        assert_eq!(
            inputs[0],
            "2bfcf64c38aeb71a5f36b5aa43dd96cd81c76452533e46061a35e4c46c87eaf1"
        );
        assert_eq!(
            stage_key("refresh", 1, &config_hash, &inputs),
            "660704c338431d747f188817e28e1a96bc7c37dbbbc7ac745acee0935363c531"
        );
        assert_eq!(
            stage_key("ingest", 1, &config_hash, &[]),
            "5fcb83de2bee8b1594ae9615c3315cea9dfa3f41fd92f5c0c241db567939a8e7"
        );
        let nested: Value =
            serde_json::from_str(r#"{"b":{"z":1,"a":[2.5,{"y":true,"x":null}]},"a":"q"}"#).unwrap();
        assert_eq!(
            canonical_json(&nested),
            r#"{"a":"q","b":{"a":[2.5,{"x":null,"y":true}],"z":1}}"#
        );
        use super::super::utc::{utc_compact, utc_text};
        assert_eq!(utc_text(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_text(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_text(1_792_288_740), "2026-10-18T01:59:00Z");
        assert_eq!(utc_compact(1_792_288_740), "20261018T015900Z");
    }
}
