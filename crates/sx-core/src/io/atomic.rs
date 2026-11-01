// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tmp-then-rename file writes.

use std::fs::File;
use std::path::Path;

use crate::error::{SxError, SxResult};

/// Build an `SxError::Io` for `path` (path as text).
pub(crate) fn io_error(path: &Path, source: std::io::Error) -> SxError {
    SxError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// Write a file by `write(tmp)` and then rename `tmp` to `final_path`.
///
/// Steps: create the parent folder if missing; `tmp` is `final_path` with
/// `.tmp` appended to the file name (a stale `tmp` is removed first); call
/// `write(&tmp)`; `sync_all` the tmp file; rename it over `final_path`
/// (replacing an existing file); best-effort `sync_all` of the parent folder.
/// On an error in the write or rename step, `tmp` is removed and the first
/// error is returned.
///
/// Units: none. NaN rule: not applicable.
///
/// # Errors
/// Whatever `write` returns; `SxError::Io` (path = `final_path`) for folder,
/// sync and rename failures; `SxError::InvalidInput` when `final_path` has no
/// file name.
pub fn write_atomic<F: FnOnce(&Path) -> SxResult<()>>(final_path: &Path, write: F) -> SxResult<()> {
    let file_name = final_path.file_name().ok_or_else(|| {
        SxError::InvalidInput(format!("{}: path has no file name", final_path.display()))
    })?;
    let parent = final_path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = parent {
        std::fs::create_dir_all(dir).map_err(|e| io_error(final_path, e))?;
    }
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(".tmp");
    let tmp = final_path.with_file_name(tmp_name);
    if tmp.exists() {
        std::fs::remove_file(&tmp).map_err(|e| io_error(final_path, e))?;
    }

    let result = write(&tmp)
        .and_then(|()| {
            File::open(&tmp)
                .and_then(|f| f.sync_all())
                .map_err(|e| io_error(final_path, e))
        })
        .and_then(|()| std::fs::rename(&tmp, final_path).map_err(|e| io_error(final_path, e)));
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    let dir = parent.unwrap_or_else(|| Path::new("."));
    if let Ok(f) = File::open(dir) {
        let _ = f.sync_all();
    }
    Ok(())
}
