// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Interval table and pad confidence.

use crate::config::ConfidenceParams;
use crate::error::{SxError, SxResult};
use crate::types::{IntervalRow, IntervalState, TopSource, WellId};

/// One interval of one well with its thickness and confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalInfo {
    /// Well, interval index, kept boundary MDs (ft, NaN if absent) and state.
    pub row: IntervalRow,
    /// `base_md_ft − top_md_ft` (ft, MD); NaN when `Unknown`.
    pub thickness_ft: f64,
    /// Interval confidence in `[0, 1]`.
    pub confidence: f32,
}

/// Interval rows for one well from kept boundary MDs (NaN = absent) and their
/// sources.
///
/// `boundary_md` and `source` hold boundaries `0..=n+1` (pads included), so
/// there are `n + 1` intervals. States are checked in this order: `Unknown`
/// (a boundary absent or base < top; thickness NaN, confidence 0), `Pinched`
/// (thickness exactly 0; `pinched_user` if both boundaries are user picks,
/// else `pinched_other`), `Pad` (interval 0 or `n`), `Observed` (both user;
/// `observed`), `Estimated` (`estimated_one` with one user boundary, else
/// `estimated_none`). A `Pad` takes the confidence of its neighbor (interval 1
/// or `n − 1`) when that neighbor is not a pad interval and its confidence is
/// finite and > 0; otherwise `pad_floor`.
///
/// Units: ft. NaN rule: a non-finite MD is an absent boundary. Errors:
/// [`SxError::Shape`] when the two slices differ in length,
/// [`SxError::InvalidInput`] for fewer than 3 boundaries or more than
/// `u16::MAX + 1` intervals. Complexity: `O(n)`.
pub fn interval_table(
    well: WellId,
    boundary_md: &[f64],
    source: &[Option<TopSource>],
    p: &ConfidenceParams,
) -> SxResult<Vec<IntervalInfo>> {
    if boundary_md.len() != source.len() {
        return Err(SxError::Shape(format!(
            "well {}: {} boundary MDs but {} sources",
            well.0,
            boundary_md.len(),
            source.len()
        )));
    }
    let n = boundary_md
        .len()
        .checked_sub(2)
        .filter(|&n| n >= 1)
        .ok_or_else(|| {
            SxError::InvalidInput(format!(
                "well {}: interval table needs at least 3 boundaries",
                well.0
            ))
        })?;
    let is_user = |s: Option<TopSource>| s == Some(TopSource::User);
    let mut rows: Vec<IntervalInfo> = Vec::with_capacity(n + 1);
    for (i, (mds, srcs)) in boundary_md.windows(2).zip(source.windows(2)).enumerate() {
        let (&[top, base], &[src_top, src_base]) = (mds, srcs) else {
            continue;
        };
        let interval = u16::try_from(i)
            .map_err(|_| SxError::InvalidInput(format!("well {}: too many intervals", well.0)))?;
        let n_user = usize::from(is_user(src_top)) + usize::from(is_user(src_base));
        let thickness = base - top;
        let (state, thickness_ft, confidence) =
            if !top.is_finite() || !base.is_finite() || base < top {
                (IntervalState::Unknown, f64::NAN, 0.0)
            } else if thickness == 0.0 {
                let c = if n_user == 2 {
                    p.pinched_user
                } else {
                    p.pinched_other
                };
                (IntervalState::Pinched, thickness, c)
            } else if i == 0 || i == n {
                (IntervalState::Pad, thickness, p.pad_floor)
            } else if n_user == 2 {
                (IntervalState::Observed, thickness, p.observed)
            } else if n_user == 1 {
                (IntervalState::Estimated, thickness, p.estimated_one)
            } else {
                (IntervalState::Estimated, thickness, p.estimated_none)
            };
        rows.push(IntervalInfo {
            row: IntervalRow {
                well,
                interval,
                top_md_ft: top,
                base_md_ft: base,
                state,
            },
            thickness_ft,
            confidence,
        });
    }
    let pad_confidence = |adjacent: usize, rows: &[IntervalInfo]| -> f32 {
        let adjacent_is_pad = adjacent == 0 || adjacent == n;
        match rows.get(adjacent) {
            Some(r) if !adjacent_is_pad && r.confidence.is_finite() && r.confidence > 0.0 => {
                r.confidence
            }
            _ => p.pad_floor,
        }
    };
    let upper = pad_confidence(1, &rows);
    let lower = pad_confidence(n - 1, &rows);
    for (i, c) in [(0, upper), (n, lower)] {
        if let Some(r) = rows.get_mut(i) {
            if r.row.state == IntervalState::Pad {
                r.confidence = c;
            }
        }
    }
    Ok(rows)
}
