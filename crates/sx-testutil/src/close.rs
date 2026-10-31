// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tolerance comparison for tests.

/// Elementwise comparison within absolute and relative tolerances.
///
/// Units: the caller's units, unchanged by this helper.
/// NaN rule: both NaN match; a NaN next to any other value does not.
/// Infinities match only when equal (`+inf` with `+inf`, `-inf` with `-inf`).
/// A finite pair matches when `|actual - expected| <= atol + rtol * |expected|`.
///
/// `Err` when `atol` or `rtol` is negative or NaN, when the lengths differ, or
/// when any element does not match. The mismatch report lists at most the first
/// 10 indexes. `max |diff|` ignores NaN-position and infinity mismatches.
/// Complexity: O(n).
pub fn check_close<T: Copy + Into<f64>>(
    actual: &[T],
    expected: &[T],
    atol: f64,
    rtol: f64,
) -> Result<(), String> {
    if atol.is_nan() || rtol.is_nan() || atol < 0.0 || rtol < 0.0 {
        return Err("atol and rtol must be >= 0".to_string());
    }
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }

    let n = actual.len();
    let mut n_bad = 0usize;
    let mut max_finite: Option<(f64, usize)> = None;
    let mut lines = Vec::new();

    for (i, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
        let a: f64 = (*left).into();
        let b: f64 = (*right).into();
        let matched = if a.is_nan() || b.is_nan() {
            a.is_nan() && b.is_nan()
        } else if a.is_infinite() || b.is_infinite() {
            a == b
        } else {
            (a - b).abs() <= atol + rtol * b.abs()
        };
        if matched {
            continue;
        }
        n_bad += 1;
        let abs_diff = (a - b).abs();
        if a.is_finite() && b.is_finite() {
            let replace = match max_finite {
                Some((best, _)) => abs_diff > best,
                None => true,
            };
            if replace {
                max_finite = Some((abs_diff, i));
            }
        }
        if lines.len() < 10 {
            let diff_text = if a.is_nan() || b.is_nan() {
                "NaN-position".to_string()
            } else {
                format!("{abs_diff:e}")
            };
            lines.push(format!("  [{i}] actual={a} expected={b} diff={diff_text}"));
        }
    }

    if n_bad == 0 {
        return Ok(());
    }

    let max_note = match max_finite {
        Some((diff, index)) => format!("max |diff| = {diff:e} at index {index}"),
        None => "max |diff| = n/a".to_string(),
    };
    let mut msg =
        format!("{n_bad} of {n} elements differ (atol={atol:e}, rtol={rtol:e}); {max_note}");
    for line in &lines {
        msg.push('\n');
        msg.push_str(line);
    }
    Err(msg)
}

/// Panics with the [`check_close`] message when the slices differ.
///
/// Units and NaN rule: see [`check_close`]. Complexity: O(n).
pub fn assert_close<T: Copy + Into<f64>>(actual: &[T], expected: &[T], atol: f64, rtol: f64) {
    if let Err(msg) = check_close(actual, expected, atol, rtol) {
        panic!("{msg}");
    }
}

/// Panics with the [`check_close`] message when the scalars differ.
///
/// Units and NaN rule: see [`check_close`]. Same as [`assert_close`] on
/// one-element slices.
pub fn assert_close_scalar(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert_close(&[actual], &[expected], atol, rtol);
}

#[cfg(test)]
mod tests {
    use super::{assert_close, assert_close_scalar, check_close};

    #[test]
    fn nan_pair_within_tolerance() {
        assert_eq!(
            check_close(&[1.0f32, f32::NAN], &[1.000001f32, f32::NAN], 1e-5, 1e-5),
            Ok(())
        );
    }

    #[test]
    fn finite_mismatch_message() {
        let err = check_close(&[1.0, 2.0], &[1.0, 2.1], 1e-3, 0.0).unwrap_err();
        let prefix = "1 of 2 elements differ (atol=1e-3, rtol=0e0); max |diff| = 1.0000000000000009e-1 at index 1";
        assert!(err.starts_with(prefix), "{err}");
        assert!(
            err.contains("  [1] actual=2 expected=2.1 diff=1.0000000000000009e-1"),
            "{err}"
        );
    }

    #[test]
    fn nan_position_is_err() {
        let err = check_close(&[f64::NAN], &[0.0], 1.0, 1.0).unwrap_err();
        assert!(err.contains("diff=NaN-position"), "{err}");
        assert!(err.contains("max |diff| = n/a"), "{err}");
    }

    #[test]
    fn matching_infinities() {
        assert_eq!(
            check_close(&[f64::INFINITY], &[f64::INFINITY], 0.0, 0.0),
            Ok(())
        );
    }

    #[test]
    fn length_mismatch() {
        let err = check_close(&[1.0_f64], &[1.0, 2.0], 0.0, 0.0).unwrap_err();
        assert_eq!(err, "length mismatch: actual 1, expected 2");
    }

    #[test]
    fn negative_or_nan_tolerance() {
        for (atol, rtol) in [(-1.0, 0.0), (0.0, -1.0), (f64::NAN, 0.0), (1.0, f64::NAN)] {
            assert_eq!(
                check_close(&[1.0_f64], &[1.0], atol, rtol).unwrap_err(),
                "atol and rtol must be >= 0"
            );
        }
    }

    #[test]
    #[should_panic(expected = "elements differ")]
    fn assert_close_panics() {
        assert_close(&[1.0_f64], &[2.0], 0.0, 0.0);
    }

    #[test]
    fn assert_close_scalar_accepts_equal() {
        assert_close_scalar(3.0, 3.0, 0.0, 0.0);
    }

    #[test]
    fn relative_tolerance_uses_expected() {
        assert!(check_close(&[2.0_f64], &[1.0], 0.0, 0.5).is_err());
        assert_eq!(check_close(&[1.0_f64], &[2.0], 0.0, 0.5), Ok(()));
    }

    #[test]
    fn lists_only_first_ten_mismatches() {
        let actual = [0.0_f64; 12];
        let expected = [1.0_f64; 12];
        let err = check_close(&actual, &expected, 0.0, 0.0).unwrap_err();
        assert!(err.starts_with("12 of 12 elements differ "), "{err}");
        assert_eq!(err.matches(" actual=").count(), 10, "{err}");
    }

    #[test]
    #[ignore = "timing budget; run on request"]
    fn timing_check_close_million() {
        let n = 1_000_000;
        let actual = vec![1.0f32; n];
        let expected = vec![1.0f32; n];
        let start = std::time::Instant::now();
        check_close(&actual, &expected, 1e-5, 1e-5).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(20),
            "check_close on {n} f32 pairs took {elapsed:?}"
        );
    }

    #[test]
    fn max_diff_skips_nan_and_inf() {
        let err =
            check_close(&[f64::NAN, 1.0, f64::INFINITY], &[0.0, 1.5, 0.0], 0.0, 0.0).unwrap_err();
        assert!(
            err.starts_with(
                "3 of 3 elements differ (atol=0e0, rtol=0e0); max |diff| = 5e-1 at index 1"
            ),
            "{err}"
        );
        assert!(err.contains("diff=NaN-position"), "{err}");
    }
}
