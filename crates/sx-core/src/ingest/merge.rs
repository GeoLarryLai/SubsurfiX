// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Multi-file merge of one log.

/// Merged samples of one log of one well.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Merged {
    /// Depth, feet, strictly ascending.
    pub md_ft: Vec<f64>,
    /// Values, same length.
    pub values: Vec<f32>,
    /// Index (in the input slice) of the primary file.
    pub primary_file: usize,
    /// Files that contributed at least one sample.
    pub n_files_used: usize,
}

/// Finite `(z, value)` samples of one file's spliced output.
fn finite_samples(z: &[f64], out: &[f32]) -> Vec<(f64, f32)> {
    z.iter()
        .zip(out)
        .filter(|(_, v)| v.is_finite())
        .map(|(&z, &v)| (z, v))
        .collect()
}

/// Merge the spliced outputs of one log over the well's files.
///
/// `files[i]` is `Some((z, out))` when file `i` has the log. The primary file
/// has the most finite samples (tie: earlier file) and keeps all of them.
/// Other files, by finite count descending then file order, add only samples
/// outside the range covered so far (`z < lo` or `z > hi`); the range then
/// widens to everything kept. No shift or calibration. `None` when no file
/// has a finite sample.
pub(crate) fn merge_files(files: &[Option<(&[f64], &[f32])>]) -> Option<Merged> {
    let samples: Vec<Vec<(f64, f32)>> = files
        .iter()
        .map(|f| f.map_or_else(Vec::new, |(z, out)| finite_samples(z, out)))
        .collect();
    // Largest finite count; the earlier file wins a tie.
    let mut primary: Option<usize> = None;
    for (i, s) in samples.iter().enumerate() {
        let better = match primary {
            None => !s.is_empty(),
            Some(p) => s.len() > samples.get(p).map_or(0, Vec::len),
        };
        if better {
            primary = Some(i);
        }
    }
    let primary = primary?;
    let mut merged: Vec<(f64, f32)> = samples.get(primary)?.clone();
    let mut lo = merged.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
    let mut hi = merged.iter().map(|s| s.0).fold(f64::NEG_INFINITY, f64::max);
    let mut n_files_used = 1;
    let mut others: Vec<usize> = (0..samples.len())
        .filter(|&i| i != primary && samples.get(i).is_some_and(|s| !s.is_empty()))
        .collect();
    others.sort_by(|&a, &b| {
        let na = samples.get(a).map_or(0, Vec::len);
        let nb = samples.get(b).map_or(0, Vec::len);
        nb.cmp(&na).then(a.cmp(&b))
    });
    for i in others {
        let kept: Vec<(f64, f32)> = samples
            .get(i)
            .map(|s| {
                s.iter()
                    .copied()
                    .filter(|&(z, _)| z < lo || z > hi)
                    .collect()
            })
            .unwrap_or_default();
        if kept.is_empty() {
            continue;
        }
        n_files_used += 1;
        for &(z, _) in &kept {
            lo = lo.min(z);
            hi = hi.max(z);
        }
        merged.extend(kept);
    }
    merged.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(Merged {
        md_ft: merged.iter().map(|s| s.0).collect(),
        values: merged.iter().map(|s| s.1).collect(),
        primary_file: primary,
        n_files_used,
    })
}
