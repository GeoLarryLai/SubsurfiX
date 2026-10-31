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

#[cfg(test)]
mod tests {
    use super::*;

    fn series(lo: f64, hi: f64, v: f32) -> (Vec<f64>, Vec<f32>) {
        let n = ((hi - lo) / 0.5).round() as usize + 1;
        let z: Vec<f64> = (0..n).map(|i| lo + 0.5 * i as f64).collect();
        let out = vec![v; n];
        (z, out)
    }

    #[test]
    fn example_d() {
        let (z1, v1) = series(1000.0, 1100.0, 50.0);
        let (z2, v2) = series(1050.0, 1300.0, 70.0);
        assert_eq!((v1.len(), v2.len()), (201, 501));
        let files = [Some((&z1[..], &v1[..])), Some((&z2[..], &v2[..]))];
        let m = merge_files(&files).unwrap();
        assert_eq!(m.primary_file, 1);
        assert_eq!(m.n_files_used, 2);
        assert_eq!(m.md_ft.len(), 100 + 501);
        assert_eq!(m.md_ft[0], 1000.0);
        assert_eq!(m.md_ft[99], 1049.5);
        assert_eq!(m.md_ft[100], 1050.0);
        assert!(m.values[..100].iter().all(|&v| v == 50.0));
        assert!(m.values[100..].iter().all(|&v| v == 70.0));
        assert!(m.md_ft.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn holes_inside_are_not_filled_and_ties_go_to_earlier() {
        // Primary has a hole; the other file only has samples inside the range.
        let z1 = [1000.0, 1001.0, 1004.0, 1005.0];
        let v1 = [1.0f32, 2.0, f32::NAN, 5.0];
        let z2 = [1002.0, 1003.0];
        let v2 = [3.0f32, 4.0];
        let files = [Some((&z1[..], &v1[..])), Some((&z2[..], &v2[..])), None];
        let m = merge_files(&files).unwrap();
        // File 0 has 3 finite samples, file 1 has 2.
        assert_eq!(m.primary_file, 0);
        assert_eq!(m.n_files_used, 1);
        assert_eq!(m.md_ft, [1000.0, 1001.0, 1005.0]);

        // Equal counts: earlier file is primary.
        let za = [1.0, 2.0];
        let zb = [10.0, 11.0];
        let va = [1.0f32, 1.0];
        let vb = [2.0f32, 2.0];
        let files = [Some((&za[..], &va[..])), Some((&zb[..], &vb[..]))];
        let m = merge_files(&files).unwrap();
        assert_eq!(m.primary_file, 0);
        assert_eq!(m.n_files_used, 2);
        assert_eq!(m.md_ft, [1.0, 2.0, 10.0, 11.0]);
    }

    #[test]
    fn running_range_widens() {
        // The second-largest file widens the range before the smallest is judged.
        let (z1, v1) = series(1000.0, 1010.0, 1.0); // 21 samples, primary
        let (z2, v2) = series(1005.0, 1014.0, 2.0); // 19 samples: keeps > 1010
        let (z3, v3) = series(1011.0, 1016.0, 3.0); // 11 samples: keeps > 1014 only
        let files = [
            Some((&z1[..], &v1[..])),
            Some((&z3[..], &v3[..])),
            Some((&z2[..], &v2[..])),
        ];
        let m = merge_files(&files).unwrap();
        assert_eq!(m.primary_file, 0);
        assert_eq!(m.n_files_used, 3);
        let from3 = m
            .md_ft
            .iter()
            .zip(&m.values)
            .filter(|(_, &v)| v == 3.0)
            .map(|(&z, _)| z)
            .collect::<Vec<_>>();
        assert_eq!(from3, [1014.5, 1015.0, 1015.5, 1016.0]);
    }

    #[test]
    fn nothing_to_merge() {
        assert!(merge_files(&[]).is_none());
        let z = [1.0];
        let v = [f32::NAN];
        assert!(merge_files(&[Some((&z[..], &v[..])), None]).is_none());
    }
}
