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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{csv, parquet};
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn small_batch() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .unwrap()
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn example_a_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("a.csv");
        let err = write_atomic(&target, |_| Err(SxError::InvalidInput("boom".into())));
        assert!(matches!(err, Err(SxError::InvalidInput(m)) if m == "boom"));
        assert!(!target.exists());
        assert!(!dir.path().join("a.csv.tmp").exists());

        let b = small_batch();
        write_atomic(&target, |t| csv::write_batch(t, &b)).unwrap();
        assert_eq!(names(dir.path()), vec!["a.csv"]);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "a\n1\n2\n");

        // Failure after writing the tmp file removes it and keeps the old file.
        let err = write_atomic(&target, |t| {
            std::fs::write(t, "partial").map_err(|e| io_error(t, e))?;
            Err(SxError::InvalidInput("late".into()))
        });
        assert!(err.is_err());
        assert_eq!(names(dir.path()), vec!["a.csv"]);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "a\n1\n2\n");
    }

    #[test]
    fn replaces_creates_parent_and_removes_stale_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("sub").join("deeper").join("x.txt");
        write_atomic(&target, |t| {
            std::fs::write(t, "one").map_err(|e| io_error(t, e))
        })
        .unwrap();
        std::fs::write(dir.path().join("sub/deeper/x.txt.tmp"), "stale").unwrap();
        write_atomic(&target, |t| {
            std::fs::write(t, "two").map_err(|e| io_error(t, e))
        })
        .unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "two");
        assert_eq!(names(&dir.path().join("sub/deeper")), vec!["x.txt"]);
    }

    #[test]
    fn file_writers_use_exact_name() {
        let dir = tempfile::tempdir().unwrap();
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let b = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let b = parquet::with_schema_tag(b, "t/1").unwrap();
        parquet::write_batch(&dir.path().join("x.parquet.tmp"), &b).unwrap();
        assert_eq!(names(dir.path()), vec!["x.parquet.tmp"]);
    }
}
