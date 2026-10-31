// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Depth helpers: elevation at an MD and the smallest MD at an elevation

use crate::types::{WellGeometry, WellHeader, GRID_STEP_FT};

/// Usable geometry samples: the grid length, guarded by the vector length.
fn geom_len(geom: &WellGeometry) -> usize {
    geom.grid.len.min(geom.elev_ft.len())
}

/// Elevation (ft, positive up) of a well at measured depth `md_ft`
///
/// Geometry sample `i` sits at `md_i = 0.5 (k_start + i)` with elevation
/// `geom.elev_ft[i]`. No samples: `kb_elev_ft − md`. Above the first sample:
/// `elev_0 + (md_0 − md)`; below the last: `elev_last − (md − md_last)`
/// (vertical extension). Otherwise exact at a sample, else linear between the
/// two bracketing samples.
///
/// Units: feet. NaN rule: propagate (NaN `md_ft` or a NaN bracketing
/// elevation gives NaN). Errors: none. Complexity: `O(1)`.
pub fn elev_at_md(header: &WellHeader, geom: &WellGeometry, md_ft: f64) -> f64 {
    if md_ft.is_nan() {
        return f64::NAN;
    }
    let n = geom_len(geom);
    if n == 0 {
        return header.kb_elev_ft - md_ft;
    }
    let grid = geom.grid;
    let (md_first, md_last) = (grid.md_ft(0), grid.md_ft(n - 1));
    let (Some(&e_first), Some(&e_last)) = (geom.elev_ft.first(), geom.elev_ft.get(n - 1)) else {
        return f64::NAN;
    };
    if md_ft < md_first {
        return e_first + (md_first - md_ft);
    }
    if md_ft > md_last {
        return e_last - (md_ft - md_last);
    }
    let mut i = (((md_ft - md_first) / GRID_STEP_FT).floor() as usize).min(n - 1);
    if i > 0 && grid.md_ft(i) > md_ft {
        i -= 1;
    }
    if i + 1 < n && grid.md_ft(i + 1) <= md_ft {
        i += 1;
    }
    let md_i = grid.md_ft(i);
    let Some(&e_i) = geom.elev_ft.get(i) else {
        return f64::NAN;
    };
    if md_ft == md_i {
        return e_i;
    }
    let Some(&e_j) = geom.elev_ft.get(i + 1) else {
        return f64::NAN;
    };
    let md_j = grid.md_ft(i + 1);
    e_i + (md_ft - md_i) / (md_j - md_i) * (e_j - e_i)
}

/// Smallest measured depth (ft) at which the well reaches elevation `elev_ft`
///
/// No samples: `kb_elev_ft − e`. `e ≥ elev_0`: `md_0 − (e − elev_0)`. Else the
/// first sample `i` with `elev_i ≤ e` gives
/// `md_{i−1} + (elev_{i−1} − e) / (elev_{i−1} − elev_i) × 0.5`; no such sample:
/// `md_last + (elev_last − e)`.
///
/// Units: feet. NaN rule: propagate (NaN `elev_ft`, or a NaN elevation next
/// to the crossing, gives NaN; NaN samples never count as a crossing).
/// Errors: none. Complexity: `O(n)`.
pub fn md_at_elev(header: &WellHeader, geom: &WellGeometry, elev_ft: f64) -> f64 {
    if elev_ft.is_nan() {
        return f64::NAN;
    }
    let n = geom_len(geom);
    let Some(elev) = geom.elev_ft.get(..n).filter(|e| !e.is_empty()) else {
        return header.kb_elev_ft - elev_ft;
    };
    let grid = geom.grid;
    let md_first = grid.md_ft(0);
    let (Some(&e_first), Some(&e_last)) = (elev.first(), elev.last()) else {
        return f64::NAN;
    };
    if elev_ft >= e_first {
        return md_first - (elev_ft - e_first);
    }
    match elev.iter().position(|&e| e <= elev_ft) {
        Some(i) => match (i.checked_sub(1), i.checked_sub(1).and_then(|p| elev.get(p))) {
            (Some(p), Some(&e_p)) => {
                let e_i = elev.get(i).copied().unwrap_or(f64::NAN);
                grid.md_ft(p) + (e_p - elev_ft) / (e_p - e_i) * GRID_STEP_FT
            }
            _ => md_first - (elev_ft - e_first),
        },
        None => grid.md_ft(n - 1) + (e_last - elev_ft),
    }
}
