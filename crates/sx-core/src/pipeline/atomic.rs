// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Atomic product writes with their hash, the project lock, and stale-file cleanup.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::keys::sha256_file;
use super::utc::{now_secs, utc_text};
use crate::error::{SxError, SxResult};
use crate::io::atomic::io_error;

/// Lock file name inside the project folder.
pub(crate) const LOCK_FILE: &str = ".subsurfix.lock";

/// Extensions of files the cleanup may delete.
const PRODUCT_EXTENSIONS: [&str; 6] = ["parquet", "csv", "las", "json", "txt", "tmp"];

/// Write `final_path` through `<name>.tmp` in the same folder, then rename.
///
/// Steps of `io::atomic::write_atomic` (stale tmp removed, `write(tmp)`, tmp
/// synced, renamed over `final_path`, folder synced; tmp removed on error),
/// plus the SHA-256 of the written bytes, taken from the tmp file.
///
/// # Errors
/// Whatever `write` returns; `SxError::Io` for file system failures.
pub(crate) fn write_atomic<F: FnOnce(&Path) -> SxResult<()>>(
    final_path: &Path,
    write: F,
) -> SxResult<String> {
    let mut hash = String::new();
    crate::io::atomic::write_atomic(final_path, |tmp| {
        write(tmp)?;
        hash = sha256_file(tmp)?;
        Ok(())
    })?;
    Ok(hash)
}

/// `<project>/.subsurfix.lock`, held for one run; removed on drop.
#[derive(Debug)]
pub(crate) struct ProjectLock {
    path: PathBuf,
}

impl ProjectLock {
    /// Create the lock file (`create_new`), content `pid=<pid>\nstarted_utc=<utc>\n`.
    ///
    /// # Errors
    /// `SxError::InvalidInput` when the lock exists; `SxError::Io` otherwise.
    pub(crate) fn acquire(project_dir: &Path) -> SxResult<Self> {
        let path = project_dir.join(LOCK_FILE);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                let text = format!(
                    "pid={}\nstarted_utc={}\n",
                    std::process::id(),
                    utc_text(now_secs())
                );
                let lock = Self { path };
                f.write_all(text.as_bytes())
                    .map_err(|e| io_error(&lock.path, e))?;
                Ok(lock)
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let content = std::fs::read_to_string(&path).unwrap_or_default();
                let content = content.trim().replace('\n', ", ");
                Err(SxError::InvalidInput(format!(
                    "project is locked by another run ({content}); delete {LOCK_FILE} if no run is active"
                )))
            }
            Err(e) => Err(io_error(&path, e)),
        }
    }
}

impl Drop for ProjectLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> SxResult<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_error(dir, e)),
    };
    for entry in entries {
        let path = entry.map_err(|e| io_error(dir, e))?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Delete files under `dir` (recursive) with a product extension that are not
/// in `keep`. Other files are left alone. Returns the deleted paths.
///
/// # Errors
/// `SxError::Io` when the folder cannot be listed or a file cannot be removed.
pub(crate) fn remove_stale(dir: &Path, keep: &BTreeSet<PathBuf>) -> SxResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files(dir, &mut files)?;
    files.sort();
    let mut removed = Vec::new();
    for path in files {
        let product = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| PRODUCT_EXTENSIONS.contains(&e));
        if product && !keep.contains(&path) {
            std::fs::remove_file(&path).map_err(|e| io_error(&path, e))?;
            removed.push(path);
        }
    }
    Ok(removed)
}
