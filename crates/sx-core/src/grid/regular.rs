// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Regular 3D grid: `nx × ny × nz` box cells, axis aligned, no rotation.

use serde::{Deserialize, Serialize};

use crate::error::{SxError, SxResult};

/// Axis-aligned regular grid. Cell `(i, j, k)` spans
/// `[x0 + i·dx, x0 + (i+1)·dx) × [y0 + j·dy, …) × [z0 + k·dz, …)`.
/// Linear index `i + nx·(j + ny·k)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegularGrid {
    /// Origin (minimum corner) x, y, z (z = depth of the top face).
    pub origin: [f64; 3],
    /// Cell size along x, y, z; all > 0.
    pub cell: [f64; 3],
    /// Cell counts along x, y, z; all ≥ 1.
    pub dims: [usize; 3],
}

impl RegularGrid {
    /// Checked constructor.
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] if a size is not finite and positive, a count is
    /// 0, or the origin is not finite.
    pub fn new(origin: [f64; 3], cell: [f64; 3], dims: [usize; 3]) -> SxResult<Self> {
        if origin.iter().any(|v| !v.is_finite())
            || cell.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || dims.contains(&0)
        {
            return Err(SxError::InvalidInput(format!(
                "grid: origin {origin:?}, cell {cell:?}, dims {dims:?}"
            )));
        }
        Ok(Self { origin, cell, dims })
    }

    /// Grid covering `[min, max]` on each axis with the given cell size
    /// (counts rounded up).
    ///
    /// # Errors
    /// As [`RegularGrid::new`]; also when `max < min`.
    pub fn covering(min: [f64; 3], max: [f64; 3], cell: [f64; 3]) -> SxResult<Self> {
        let mut dims = [0usize; 3];
        for a in 0..3 {
            if !(max[a] >= min[a]) {
                return Err(SxError::InvalidInput(format!(
                    "grid: max {max:?} < min {min:?}"
                )));
            }
            dims[a] = (((max[a] - min[a]) / cell[a]).ceil() as usize).max(1);
        }
        Self::new(min, cell, dims)
    }

    /// Number of cells.
    pub fn n_cells(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Linear index of `(i, j, k)`.
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// `(i, j, k)` of a linear index.
    pub fn ijk(&self, idx: usize) -> (usize, usize, usize) {
        let i = idx % self.dims[0];
        let j = (idx / self.dims[0]) % self.dims[1];
        (i, j, idx / (self.dims[0] * self.dims[1]))
    }

    /// Cell containing a point; `None` outside the grid or for NaN. The max
    /// faces belong to the last cell.
    pub fn locate(&self, p: [f64; 3]) -> Option<usize> {
        let mut c = [0usize; 3];
        for a in 0..3 {
            let t = (p[a] - self.origin[a]) / self.cell[a];
            if !t.is_finite() || t < 0.0 || t > self.dims[a] as f64 {
                return None;
            }
            c[a] = (t.floor() as usize).min(self.dims[a] - 1);
        }
        Some(self.index(c[0], c[1], c[2]))
    }

    /// Centre of a cell.
    pub fn center(&self, idx: usize) -> [f64; 3] {
        let (i, j, k) = self.ijk(idx);
        [
            self.origin[0] + (i as f64 + 0.5) * self.cell[0],
            self.origin[1] + (j as f64 + 0.5) * self.cell[1],
            self.origin[2] + (k as f64 + 0.5) * self.cell[2],
        ]
    }
}
