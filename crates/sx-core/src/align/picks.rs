// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! User picks: cleaning, canonical top order and study range.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use tracing::debug;

use crate::error::{SxError, SxResult};
use crate::types::{WellHeader, WellId};
use crate::util::median;

/// One user pick as read from the tops table.
#[derive(Debug, Clone, PartialEq)]
pub struct UserPick {
    /// Well of the pick.
    pub well: WellId,
    /// Top name as written in the tops table.
    pub name: String,
    /// Measured depth, ft. A non-finite value is not a depth but still marks
    /// the top as present in that well.
    pub md_ft: f64,
}

/// Cleaned picks per well (same order as `headers`): name → MD in ft, NaN when
/// the pick row has no finite MD.
///
/// Picks of wells not in `headers` are dropped; several picks for one
/// `(well, name)` keep the smallest finite MD. `headers` must be sorted by id.
pub(crate) fn clean_picks(
    headers: &[WellHeader],
    picks: &[UserPick],
) -> Vec<BTreeMap<String, f64>> {
    let mut out: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new(); headers.len()];
    let mut n_unknown = 0usize;
    for pick in picks {
        let Some(map) = headers
            .binary_search_by_key(&pick.well, |h| h.id)
            .ok()
            .and_then(|w| out.get_mut(w))
        else {
            n_unknown += 1;
            continue;
        };
        let md = if pick.md_ft.is_finite() {
            pick.md_ft
        } else {
            f64::NAN
        };
        match map.get_mut(&pick.name) {
            Some(old) => {
                if md.is_finite() && (old.is_nan() || md < *old) {
                    *old = md;
                }
            }
            None => {
                map.insert(pick.name.clone(), md);
            }
        }
    }
    if n_unknown > 0 {
        debug!(n_unknown, "rough alignment: dropped picks of unknown wells");
    }
    out
}

/// Canonical top order.
///
/// `pick_elevations` holds `(name, pick elevation in ft)` for every finite
/// pick. Per name the median elevation is taken (even count: mean of the two
/// middle values); names are sorted by median **descending** (shallowest
/// first), tie by name ascending (byte order). Index 0 = shallowest. A name
/// with no finite elevation does not appear.
///
/// NaN rule: skip (non-finite elevations are ignored). Errors: none; empty
/// input gives an empty list. Complexity: `O(n log n)`.
pub fn canonical_order(pick_elevations: &[(String, f64)]) -> Vec<String> {
    let mut by_name: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (name, elev) in pick_elevations {
        if elev.is_finite() {
            by_name.entry(name.as_str()).or_default().push(*elev);
        }
    }
    let mut medians: Vec<(&str, f64)> = by_name
        .into_iter()
        .map(|(name, elevs)| (name, median(&elevs)))
        .collect();
    medians.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    medians.into_iter().map(|(n, _)| n.to_string()).collect()
}

/// Study range as inclusive canonical indices. An empty name
/// means the first / last canonical name.
///
/// Errors: [`SxError::Config`] for an unknown name or `study_top` after
/// `study_base`; [`SxError::EmptyInput`] when `names` is empty.
pub(crate) fn study_range(
    names: &[String],
    study_top: &str,
    study_base: &str,
) -> SxResult<(usize, usize)> {
    let last = names
        .len()
        .checked_sub(1)
        .ok_or_else(|| SxError::EmptyInput("no canonical top names".into()))?;
    let find = |name: &str, default: usize, key: &str| -> SxResult<usize> {
        if name.is_empty() {
            return Ok(default);
        }
        names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| SxError::Config(format!("align.{key}: unknown top name {name:?}")))
    };
    let top = find(study_top, 0, "study_top")?;
    let base = find(study_base, last, "study_base")?;
    if top > base {
        return Err(SxError::Config(format!(
            "align.study_top {study_top:?} lies below align.study_base {study_base:?}"
        )));
    }
    Ok((top, base))
}
