// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Log-to-cell upscaling: average the log samples that fall in each cell.
//!
//! Arithmetic suits additive properties (porosity, density); harmonic suits
//! flow across layers (vertical permeability); geometric suits horizontal
//! permeability in heterogeneous rock. Facies use the majority code.

use std::collections::BTreeMap;

use crate::error::{SxError, SxResult};
use crate::grid::RegularGrid;

/// Averaging rule for a continuous log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Averaging {
    /// Mean of values.
    Arithmetic,
    /// `n / Σ 1/v`; needs every value > 0.
    Harmonic,
    /// `exp(mean(ln v))`; needs every value > 0.
    Geometric,
}

impl Averaging {
    /// Parse "arithmetic" / "harmonic" / "geometric" (case-insensitive).
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for any other name.
    pub fn parse(s: &str) -> SxResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "arithmetic" => Ok(Self::Arithmetic),
            "harmonic" => Ok(Self::Harmonic),
            "geometric" => Ok(Self::Geometric),
            _ => Err(SxError::InvalidInput(format!("unknown averaging {s:?}"))),
        }
    }
}

/// Upscaled cells: `(cell index, value, sample count)`, sorted by cell.
pub type CellValues<T> = Vec<(usize, T, usize)>;

fn check(points: &[[f64; 3]], n: usize) -> SxResult<()> {
    if points.len() != n {
        return Err(SxError::Shape(format!(
            "{} points vs {n} values",
            points.len()
        )));
    }
    Ok(())
}

/// Anything that maps a point to a cell index (regular or corner-point grid).
pub trait CellLocator {
    /// Cell containing `p`, or `None`.
    fn locate_cell(&self, p: [f64; 3]) -> Option<usize>;
}

impl CellLocator for RegularGrid {
    fn locate_cell(&self, p: [f64; 3]) -> Option<usize> {
        self.locate(p)
    }
}

/// Average continuous samples per cell. Samples outside the grid or NaN
/// are skipped; cells with fewer than `min_count` samples are left out.
///
/// # Errors
/// [`SxError::Shape`] on a length mismatch; [`SxError::InvalidInput`] if a
/// harmonic or geometric average meets a value ≤ 0.
pub fn upscale_continuous(
    grid: &impl CellLocator,
    points: &[[f64; 3]],
    values: &[f64],
    method: Averaging,
    min_count: usize,
) -> SxResult<CellValues<f64>> {
    check(points, values.len())?;
    let mut acc: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for (p, &v) in points.iter().zip(values) {
        if !v.is_finite() {
            continue;
        }
        let Some(c) = grid.locate_cell(*p) else {
            continue;
        };
        let t = match method {
            Averaging::Arithmetic => v,
            Averaging::Harmonic | Averaging::Geometric if v <= 0.0 => {
                return Err(SxError::InvalidInput(format!(
                    "{method:?} average needs values > 0, got {v}"
                )))
            }
            Averaging::Harmonic => 1.0 / v,
            Averaging::Geometric => v.ln(),
        };
        let e = acc.entry(c).or_insert((0.0, 0));
        e.0 += t;
        e.1 += 1;
    }
    Ok(acc
        .into_iter()
        .filter(|(_, (_, n))| *n >= min_count.max(1))
        .map(|(c, (s, n))| {
            let m = s / n as f64;
            let v = match method {
                Averaging::Arithmetic => m,
                Averaging::Harmonic => 1.0 / m,
                Averaging::Geometric => m.exp(),
            };
            (c, v, n)
        })
        .collect())
}

/// Majority facies code per cell; ties go to the smallest code. Negative
/// codes mean missing and are skipped.
///
/// # Errors
/// [`SxError::Shape`] on a length mismatch.
pub fn upscale_facies(
    grid: &impl CellLocator,
    points: &[[f64; 3]],
    codes: &[i32],
    min_count: usize,
) -> SxResult<CellValues<i32>> {
    check(points, codes.len())?;
    let mut acc: BTreeMap<usize, BTreeMap<i32, usize>> = BTreeMap::new();
    for (p, &c) in points.iter().zip(codes) {
        if c < 0 {
            continue;
        }
        if let Some(cell) = grid.locate_cell(*p) {
            *acc.entry(cell).or_default().entry(c).or_default() += 1;
        }
    }
    Ok(acc
        .into_iter()
        .filter_map(|(cell, counts)| {
            let n: usize = counts.values().sum();
            // max_by_key keeps the last max; iterate codes descending so the smallest wins.
            let (code, _) = counts.iter().rev().max_by_key(|(_, k)| **k)?;
            (n >= min_count.max(1)).then_some((cell, *code, n))
        })
        .collect())
}
