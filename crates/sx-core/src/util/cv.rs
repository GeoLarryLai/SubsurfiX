// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Well-level cross-validation splits.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{SxError, SxResult};
use crate::types::WellId;

use super::rng::{rng_for, shuffle};

/// One train/test split of well ids.
///
/// Both lists are sorted ascending and do not overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// Wells used to train this fold.
    pub train: Vec<WellId>,
    /// Wells held out for this fold.
    pub test: Vec<WellId>,
}

/// A well and its map coordinates, in meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WellXY {
    /// Well id.
    pub id: WellId,
    /// Easting, meters.
    pub x_m: f64,
    /// Northing, meters.
    pub y_m: f64,
}

/// K-fold split on well ids.
///
/// Duplicate ids are merged. Ids are shuffled with [`rng_for`]`(seed,
/// "cv_kfold", 0)` and assigned to fold `i % k`. Each fold's `test` and
/// `train` lists are sorted ascending.
///
/// NaN rule: none (ids only). Empty `wells` is [`SxError::EmptyInput`].
/// `k < 2` or `k` greater than the number of unique ids is
/// [`SxError::InvalidInput`].
///
/// Complexity: `O(n log n)`.
pub fn kfold_by_well(wells: &[WellId], k: usize, seed: u64) -> SxResult<Vec<Fold>> {
    if wells.is_empty() {
        return Err(SxError::EmptyInput("k-fold needs at least one well".into()));
    }
    let mut ids = wells.to_vec();
    ids.sort();
    ids.dedup();
    if k < 2 || k > ids.len() {
        return Err(SxError::InvalidInput(format!(
            "k must be in 2..={}",
            ids.len()
        )));
    }
    let sorted = ids.clone();
    let mut rng = rng_for(seed, "cv_kfold", 0);
    shuffle(&mut rng, &mut ids);
    let mut tests = vec![Vec::<WellId>::new(); k];
    for (i, id) in ids.iter().enumerate() {
        let fold = i % k;
        match tests.get_mut(fold) {
            Some(bucket) => bucket.push(*id),
            None => {
                return Err(SxError::Numerical("k-fold index is out of range".into()));
            }
        }
    }
    Ok(folds_from_tests(&sorted, tests))
}

/// Blocked spatial K-fold split.
///
/// `block_m` is the block size in meters. The block origin is the minimum
/// `x_m` and `y_m`. Blocks are shuffled with [`rng_for`]`(seed, "cv_spatial",
/// 0)`. Fold sizes in wells may differ. Output layout matches
/// [`kfold_by_well`].
///
/// NaN rule: error. Empty `wells` is [`SxError::EmptyInput`].
///
/// Errors after the empty check, in order: `block_m` not finite or `<= 0`,
/// a non-finite coordinate, a repeated id, then `k < 2` or `k` greater than
/// the number of blocks. Those are [`SxError::InvalidInput`].
///
/// Complexity: `O(n log n)`.
pub fn spatial_block_folds(
    wells: &[WellXY],
    block_m: f64,
    k: usize,
    seed: u64,
) -> SxResult<Vec<Fold>> {
    let prepared = prepare_spatial(wells, block_m)?;
    if k < 2 || k > prepared.blocks.len() {
        return Err(SxError::InvalidInput(format!(
            "k must be in 2..={}",
            prepared.blocks.len()
        )));
    }
    let mut blocks = prepared.blocks;
    let mut rng = rng_for(seed, "cv_spatial", 0);
    shuffle(&mut rng, &mut blocks);
    let mut fold_of = BTreeMap::new();
    for (i, block) in blocks.iter().enumerate() {
        fold_of.insert(*block, i % k);
    }
    let mut tests = vec![Vec::<WellId>::new(); k];
    for (id, bx, by) in prepared.placed {
        let fold = match fold_of.get(&(bx, by)) {
            Some(fold) => *fold,
            None => {
                return Err(SxError::Numerical(
                    "spatial block was not assigned a fold".into(),
                ));
            }
        };
        match tests.get_mut(fold) {
            Some(bucket) => bucket.push(id),
            None => {
                return Err(SxError::Numerical(
                    "spatial fold index is out of range".into(),
                ));
            }
        }
    }
    Ok(folds_from_tests(&prepared.sorted_ids, tests))
}

struct SpatialInput {
    placed: Vec<(WellId, i64, i64)>,
    blocks: Vec<(i64, i64)>,
    sorted_ids: Vec<WellId>,
}

fn prepare_spatial(wells: &[WellXY], block_m: f64) -> SxResult<SpatialInput> {
    if wells.is_empty() {
        return Err(SxError::EmptyInput(
            "spatial folds need at least one well".into(),
        ));
    }
    if !block_m.is_finite() || block_m <= 0.0 {
        return Err(SxError::InvalidInput(
            "block_m must be finite and > 0".into(),
        ));
    }
    for w in wells {
        if !w.x_m.is_finite() || !w.y_m.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "well {} has a non-finite coordinate",
                w.id.0
            )));
        }
    }
    let mut seen = BTreeSet::new();
    for w in wells {
        if !seen.insert(w.id) {
            return Err(SxError::InvalidInput(format!(
                "repeated well id {}",
                w.id.0
            )));
        }
    }
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;
    for w in wells {
        if w.x_m < x0 {
            x0 = w.x_m;
        }
        if w.y_m < y0 {
            y0 = w.y_m;
        }
    }
    if !x0.is_finite() || !y0.is_finite() {
        return Err(SxError::InvalidInput(
            "well coordinates have no finite minimum".into(),
        ));
    }
    let mut placed = Vec::with_capacity(wells.len());
    let mut blocks = Vec::with_capacity(wells.len());
    for w in wells {
        let bx = ((w.x_m - x0) / block_m).floor() as i64;
        let by = ((w.y_m - y0) / block_m).floor() as i64;
        placed.push((w.id, bx, by));
        blocks.push((bx, by));
    }
    blocks.sort_unstable();
    blocks.dedup();
    let mut sorted_ids: Vec<WellId> = wells.iter().map(|w| w.id).collect();
    sorted_ids.sort();
    Ok(SpatialInput {
        placed,
        blocks,
        sorted_ids,
    })
}

fn folds_from_tests(sorted_ids: &[WellId], tests: Vec<Vec<WellId>>) -> Vec<Fold> {
    tests
        .into_iter()
        .map(|mut test| {
            test.sort();
            let train = sorted_ids
                .iter()
                .copied()
                .filter(|id| test.binary_search(id).is_err())
                .collect();
            Fold { train, test }
        })
        .collect()
}
