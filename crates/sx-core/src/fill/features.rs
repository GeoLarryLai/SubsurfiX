// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Model features and feature-dropout augmentation.
//!
//! Every target has 46 columns: eight per predictor slot (the other five logs
//! in suite order), then six well and position columns. The target's own
//! neighbor curve is never read. Values in the fill state are in working units
//! (RESD in `log10(ohm·m)`) and are used as given.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::debug;

use crate::config::FillConfig;
use crate::error::{SxError, SxResult};
use crate::fill::{curve, in_coverage, FillInputs, WellFillState};
use crate::hash::stream_seed;
use crate::neighbors::NeighborCurve;
use crate::types::{FeatureMatrix, LogKind, Provenance, TopPick, TopSource, WellId};

/// Number of feature columns per target.
pub const N_FEATURES: usize = 46;

/// Predictor slots per target.
pub(crate) const N_SLOTS: usize = 5;

/// Columns per predictor slot.
const SLOT_WIDTH: usize = 8;

/// Column-name suffixes of one slot, in order.
const SLOT_SUFFIXES: [&str; SLOT_WIDTH] = [
    "value",
    "is_nbr",
    "nbr_n",
    "nbr_dist_m",
    "is_prior",
    "dev2p5",
    "dev10",
    "grad",
];

/// Well and position columns after the slots, in order.
const WELL_COLUMNS: [&str; 6] = [
    "resd_family",
    "x_m",
    "y_m",
    "elev_ft",
    "layer_coord",
    "layer_coord_source",
];

/// First well column (`resd_family`).
const COL_WELL: usize = N_SLOTS * SLOT_WIDTH;
/// `layer_coord` column.
const COL_LAYER: usize = COL_WELL + 4;
/// `layer_coord_source` column.
const COL_LAYER_SOURCE: usize = COL_WELL + 5;

/// Dropout stream name.
const DROPOUT_STREAM: &str = "fill.dropout";

/// The five predictors of `target`: the other logs in suite order.
fn predictors(target: LogKind) -> impl Iterator<Item = LogKind> {
    LogKind::ALL.into_iter().filter(move |&l| l != target)
}

/// Exact feature names in order for `target`.
///
/// Per predictor `p` (lower-case short name): `{p}_value`, `{p}_is_nbr`, `{p}_nbr_n`,
/// `{p}_nbr_dist_m`, `{p}_is_prior`, `{p}_dev2p5`, `{p}_dev10`, `{p}_grad`; then
/// `resd_family`, `x_m`, `y_m`, `elev_ft`, `layer_coord`, `layer_coord_source`.
/// Always [`N_FEATURES`] names. Errors: none.
pub fn feature_names(target: LogKind) -> Vec<String> {
    let mut out = Vec::with_capacity(N_FEATURES);
    for p in predictors(target) {
        let p = p.name().to_ascii_lowercase();
        out.extend(SLOT_SUFFIXES.iter().map(|s| format!("{p}_{s}")));
    }
    out.extend(WELL_COLUMNS.iter().map(|s| (*s).to_string()));
    out
}

/// `layer_coord_source` for one well: 0 if `coord` is all NaN or empty,
/// 1 if any pick of `well` has source `Estimated`, else 2.
///
/// NaN rule: skip (only finite coordinate values count). Errors: none.
pub fn layer_coord_source(coord: &[f64], picks: &[TopPick], well: WellId) -> u8 {
    if !coord.iter().any(|u| u.is_finite()) {
        return 0;
    }
    if picks
        .iter()
        .any(|p| p.well == well && p.source == TopSource::Estimated)
    {
        1
    } else {
        2
    }
}

/// Features of one well for `target`, one row per coverage row (in grid order).
#[derive(Debug, Clone, PartialEq)]
pub struct WellFeatures {
    /// Well id.
    pub well: WellId,
    /// Local grid index of each feature row.
    pub rows: Vec<usize>,
    /// Feature rows, columns named by [`feature_names`].
    pub x: FeatureMatrix,
}

/// From the current state. A well without coverage gives zero rows.
///
/// Slot values: measured value; else a `Model` fill (`is_prior = 1`); else
/// the predictor's finite neighbor value (`is_nbr = 1`, `nbr_n`, `nbr_dist_m`);
/// else NaN. Depth-local columns run over the slot series on all coverage
/// rows with the half-windows of `local_windows_ft`. `resd_family` missing from
/// `inputs` counts as 0; `layer_coord_source` missing is taken from the
/// coordinate alone (0 or 2). `layer_coord` is NaN without a coordinate.
/// NaN rule: skip (missing values become NaN features).
/// Errors: `NotFound` if the well is missing from `inputs` (header, geometry or
/// neighbor curves); `Shape` if a neighbor curve, the coordinate or a state
/// curve does not have the grid length; `Config` if `local_windows_ft` does not
/// hold two values. Complexity: `O(rows × window rows)`.
pub fn build_well_features(
    inputs: &FillInputs<'_>,
    state: &WellFillState,
    target: LogKind,
    cfg: &FillConfig,
) -> SxResult<WellFeatures> {
    let well = state.logs.id;
    let missing =
        |map: &str| SxError::NotFound(format!("fill features: well {} has no {map}", well.0));
    let header = inputs
        .nbr
        .headers
        .get(&well)
        .ok_or_else(|| missing("header"))?;
    let geometry = inputs
        .nbr
        .geometry
        .get(&well)
        .ok_or_else(|| missing("geometry"))?;
    let curves = inputs
        .neighbors
        .get(&well)
        .ok_or_else(|| missing("neighbor curves"))?;
    let grid = state.logs.grid;
    let n_grid = grid.len;
    let coord = inputs.nbr.coord.get(&well).map(Vec::as_slice);
    if let Some(u) = coord {
        if u.len() != n_grid {
            return Err(SxError::Shape(format!(
                "fill features: well {}: coordinate has {} rows, grid has {n_grid}",
                well.0,
                u.len()
            )));
        }
    }
    let names = feature_names(target);
    let rows: Vec<usize> = (0..n_grid)
        .filter(|&i| in_coverage(grid, state.coverage, i))
        .collect();
    let n = rows.len();
    if n == 0 {
        return Ok(WellFeatures {
            well,
            rows,
            x: FeatureMatrix::new(0, names, Vec::new())?,
        });
    }
    let [half_a, half_b] = local_windows(cfg)?;
    let md: Vec<f64> = rows.iter().map(|&i| grid.md_ft(i)).collect();
    let mut data = vec![f32::NAN; n * N_FEATURES];

    for (s, p) in predictors(target).enumerate() {
        let (values, prov) = curve(&state.logs, p)?;
        let nbr = curves.curves.get(&p);
        if let Some(c) = nbr {
            if c.value.len() != n_grid || c.n_donors.len() != n_grid || c.dist_m.len() != n_grid {
                return Err(SxError::Shape(format!(
                    "fill features: well {}: {} neighbor curve has {} rows, grid has {n_grid}",
                    well.0,
                    p.name(),
                    c.value.len()
                )));
            }
        }
        let slots: Vec<[f32; 5]> = rows
            .iter()
            .map(|&i| slot_at(values, prov, nbr, i))
            .collect();
        let series: Vec<f32> = slots.iter().map(|s| s[0]).collect();
        let dev_a = depth_local_dev(&series, &md, half_a, cfg.local_min_count)?;
        let dev_b = depth_local_dev(&series, &md, half_b, cfg.local_min_count)?;
        let grad = depth_gradient(&series, &md, cfg.local_min_count)?;
        let base = s * SLOT_WIDTH;
        let cols = slots.iter().zip(dev_a.iter().zip(dev_b.iter().zip(&grad)));
        for (row, (sl, (&da, (&db, &g)))) in
            data.as_chunks_mut::<N_FEATURES>().0.iter_mut().zip(cols)
        {
            if let Some(dst) = row.get_mut(base..base + SLOT_WIDTH) {
                dst.copy_from_slice(&[sl[0], sl[1], sl[2], sl[3], sl[4], da, db, g]);
            }
        }
    }

    let family = f32::from(inputs.resd_family.get(&well).copied().unwrap_or(0));
    let source = inputs
        .coord_source
        .get(&well)
        .copied()
        .unwrap_or_else(|| layer_coord_source(coord.unwrap_or(&[]), &[], well));
    let (x_m, y_m) = (header.x_m as f32, header.y_m as f32);
    for (row, &i) in data.as_chunks_mut::<N_FEATURES>().0.iter_mut().zip(&rows) {
        let elev = geometry
            .grid
            .index_of_k(grid.k(i))
            .and_then(|j| geometry.elev_ft.get(j))
            .map_or(f32::NAN, |&e| e as f32);
        let u = coord.and_then(|c| c.get(i)).map_or(f32::NAN, |&u| u as f32);
        if let Some(dst) = row.get_mut(COL_WELL..N_FEATURES) {
            dst.copy_from_slice(&[family, x_m, y_m, elev, u, f32::from(source)]);
        }
    }
    debug!(
        well = well.0,
        log = target.name(),
        rows = n,
        "fill features"
    );
    Ok(WellFeatures {
        well,
        rows,
        x: FeatureMatrix::new(n, names, data)?,
    })
}

/// Slot `[value, is_nbr, nbr_n, nbr_dist_m, is_prior]` at grid row `i`.
fn slot_at(values: &[f32], prov: &[u8], nbr: Option<&NeighborCurve>, i: usize) -> [f32; 5] {
    let v = values.get(i).copied().unwrap_or(f32::NAN);
    let p = prov.get(i).copied();
    if v.is_finite() && p == Some(Provenance::Measured.code()) {
        return [v, 0.0, 0.0, f32::NAN, 0.0];
    }
    if v.is_finite() && p == Some(Provenance::Model.code()) {
        return [v, 0.0, 0.0, f32::NAN, 1.0];
    }
    if let Some(c) = nbr {
        let nv = c.value.get(i).copied().unwrap_or(f32::NAN);
        if nv.is_finite() {
            let count = c.n_donors.get(i).map_or(0.0, |&k| k as f32);
            let dist = c.dist_m.get(i).map_or(f32::NAN, |&d| d as f32);
            return [nv, 1.0, count, dist, 0.0];
        }
    }
    [f32::NAN, 0.0, 0.0, f32::NAN, 0.0]
}

/// The two depth-local half-windows (ft). Errors: `Config` unless exactly two.
fn local_windows(cfg: &FillConfig) -> SxResult<[f64; 2]> {
    match cfg.local_windows_ft.as_slice() {
        [a, b] => Ok([*a, *b]),
        other => Err(SxError::Config(format!(
            "fill.local_windows_ft must hold two half-windows, got {}",
            other.len()
        ))),
    }
}

fn check_series(values: &[f32], md_ft: &[f64]) -> SxResult<()> {
    if values.len() != md_ft.len() {
        return Err(SxError::Shape(format!(
            "depth-local feature: {} values, {} depths",
            values.len(),
            md_ft.len()
        )));
    }
    if md_ft.windows(2).any(|w| !matches!(w, [a, b] if a <= b)) {
        return Err(SxError::InvalidInput(
            "depth-local feature: md_ft must be non-decreasing".into(),
        ));
    }
    Ok(())
}

/// `dev_h` over one series. `md_ft` and `values` have equal length.
///
/// `dev_h(r) = v(r) − mean{v(s) finite : md(s) ∈ [md(r) − h, md(r) + h]}` (both ends
/// inclusive, the row itself included), in `f64`; NaN if fewer than `min_count`
/// values lie in the window. `md_ft` must be non-decreasing (grid order).
/// NaN rule: skip (NaN rows give NaN). Errors: `Shape` on a length mismatch;
/// `InvalidInput` if `md_ft` decreases or holds NaN. Empty input: empty output.
/// Complexity: `O(n × window rows)`.
pub fn depth_local_dev(
    values: &[f32],
    md_ft: &[f64],
    half_window_ft: f64,
    min_count: usize,
) -> SxResult<Vec<f32>> {
    check_series(values, md_ft)?;
    let min_count = min_count.max(1);
    let finite: Vec<(f64, f64)> = values
        .iter()
        .zip(md_ft)
        .filter(|(v, _)| v.is_finite())
        .map(|(&v, &m)| (m, f64::from(v)))
        .collect();
    let mut out = vec![f32::NAN; values.len()];
    let (mut lo, mut hi) = (0usize, 0usize);
    for (o, (&v, &m)) in out.iter_mut().zip(values.iter().zip(md_ft)) {
        let (top, base) = (m - half_window_ft, m + half_window_ft);
        while finite.get(lo).is_some_and(|&(x, _)| x < top) {
            lo += 1;
        }
        hi = hi.max(lo);
        while finite.get(hi).is_some_and(|&(x, _)| x <= base) {
            hi += 1;
        }
        if !v.is_finite() {
            continue;
        }
        let win = finite.get(lo..hi).unwrap_or(&[]);
        if win.len() < min_count {
            continue;
        }
        let mean = win.iter().map(|&(_, y)| y).sum::<f64>() / win.len() as f64;
        *o = (f64::from(v) - mean) as f32;
    }
    Ok(out)
}

/// `grad` over one series, per ft. NaN rule: skip.
///
/// Over the finite rows only (in MD order): second-order central differences
/// for uneven spacing inside, one-sided differences at the two ends, in `f64`.
/// Fewer than `max(min_count, 2)` finite rows: all NaN. `md_ft` must be
/// non-decreasing. Errors: `Shape` on a length mismatch; `InvalidInput` if
/// `md_ft` decreases or holds NaN. Empty input: empty output.
pub fn depth_gradient(values: &[f32], md_ft: &[f64], min_count: usize) -> SxResult<Vec<f32>> {
    check_series(values, md_ft)?;
    let mut out = vec![f32::NAN; values.len()];
    let pts: Vec<(usize, f64, f64)> = values
        .iter()
        .zip(md_ft)
        .enumerate()
        .filter(|(_, (v, _))| v.is_finite())
        .map(|(i, (&v, &m))| (i, m, f64::from(v)))
        .collect();
    if pts.len() < min_count.max(2) {
        return Ok(out);
    }
    for (j, &(i, x, f)) in pts.iter().enumerate() {
        let prev = j.checked_sub(1).and_then(|k| pts.get(k));
        let next = pts.get(j + 1);
        let g = match (prev, next) {
            (Some(&(_, xm, fm)), Some(&(_, xp, fp))) => {
                let (hs, hd) = (x - xm, xp - x);
                (hs * hs * fp + (hd * hd - hs * hs) * f - hd * hd * fm) / (hs * hd * (hd + hs))
            }
            (None, Some(&(_, xp, fp))) => (fp - f) / (xp - x),
            (Some(&(_, xm, fm)), None) => (f - fm) / (x - xm),
            (None, None) => f64::NAN,
        };
        if let Some(o) = out.get_mut(i) {
            *o = g as f32;
        }
    }
    Ok(out)
}

/// True if any of the five slot values in a feature row is finite.
pub(crate) fn any_slot_value(row: &[f32]) -> bool {
    (0..N_SLOTS).any(|s| row.get(s * SLOT_WIDTH).is_some_and(|v| v.is_finite()))
}

/// Mask slot `s` of a feature row.
fn mask_slot(row: &mut [f32], s: usize) {
    let base = s * SLOT_WIDTH;
    if let Some(dst) = row.get_mut(base..base + SLOT_WIDTH) {
        dst.copy_from_slice(&[
            f32::NAN,
            0.0,
            0.0,
            f32::NAN,
            0.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ]);
    }
}

/// Returns originals followed by copies. Errors: `Shape` if `y.len != x.n_rows`.
///
/// `m = round(dropout_frac × N)` (half away from zero, at most `N`) rows are picked by a
/// partial Fisher–Yates with `ChaCha8Rng::seed_from_u64(stream_seed(project_seed,
/// "fill.dropout", target as i32))`; per pick the draws are `s1`, `s2` (slots
/// `0..5`) and `a` (`f64` in `[0, 1)`). A copy masks slots `s1` and `s2`, and
/// the layer columns if `a < dropout_layer_prob`; copies keep their target and
/// follow the originals in pick order. NaN rule: values copied as given.
/// Errors also: `Shape` if `x` does not have [`N_FEATURES`] columns.
/// Empty input: empty output.
pub fn apply_dropout(
    x: &FeatureMatrix,
    y: &[f32],
    target: LogKind,
    project_seed: u64,
    cfg: &FillConfig,
) -> SxResult<(FeatureMatrix, Vec<f32>)> {
    let n = x.n_rows;
    if y.len() != n {
        return Err(SxError::Shape(format!(
            "dropout: {} targets for {n} feature rows",
            y.len()
        )));
    }
    if x.n_cols != N_FEATURES || x.data.len() != n * N_FEATURES {
        return Err(SxError::Shape(format!(
            "dropout: feature matrix is {n} x {} with {} values, want {N_FEATURES} columns",
            x.n_cols,
            x.data.len()
        )));
    }
    let m_f = (cfg.dropout_frac * n as f64).round();
    let m = if m_f > 0.0 { (m_f as usize).min(n) } else { 0 };
    let mut rng =
        ChaCha8Rng::seed_from_u64(stream_seed(project_seed, DROPOUT_STREAM, target as i32));
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..m {
        let j = rng.random_range(i..n);
        idx.swap(i, j);
    }
    let mut data = Vec::with_capacity((n + m) * N_FEATURES);
    data.extend_from_slice(&x.data);
    let mut out_y = Vec::with_capacity(n + m);
    out_y.extend_from_slice(y);
    for &pick in idx.get(..m).unwrap_or(&[]) {
        let s1: usize = rng.random_range(0..N_SLOTS);
        let s2: usize = rng.random_range(0..N_SLOTS);
        let a = rng.random::<f64>();
        let (Some(row), Some(&t)) = (x.row(pick), y.get(pick)) else {
            return Err(SxError::Shape(format!("dropout: row {pick} out of range")));
        };
        let mut copy = row.to_vec();
        mask_slot(&mut copy, s1);
        mask_slot(&mut copy, s2);
        if a < cfg.dropout_layer_prob {
            if let Some(u) = copy.get_mut(COL_LAYER) {
                *u = f32::NAN;
            }
            if let Some(src) = copy.get_mut(COL_LAYER_SOURCE) {
                *src = 0.0;
            }
        }
        data.extend_from_slice(&copy);
        out_y.push(t);
    }
    Ok((FeatureMatrix::new(n + m, x.names.clone(), data)?, out_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fill::sequential::testkit::Project;

    fn bits(m: &FeatureMatrix) -> Vec<u32> {
        m.data.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn fill_feature_order_matches_sheet() {
        let gr = feature_names(LogKind::Gr);
        assert_eq!(
            &gr[0..8],
            &[
                "rhob_value",
                "rhob_is_nbr",
                "rhob_nbr_n",
                "rhob_nbr_dist_m",
                "rhob_is_prior",
                "rhob_dev2p5",
                "rhob_dev10",
                "rhob_grad"
            ]
        );
        assert_eq!(
            &gr[40..46],
            &[
                "resd_family",
                "x_m",
                "y_m",
                "elev_ft",
                "layer_coord",
                "layer_coord_source"
            ]
        );
        for target in LogKind::ALL {
            let names = feature_names(target);
            assert_eq!(names.len(), N_FEATURES);
            let own = format!("{}_", target.name().to_ascii_lowercase());
            assert!(
                !names[..COL_WELL].iter().any(|n| n.starts_with(&own)),
                "{target:?}"
            );
            let slots: Vec<LogKind> = LogKind::ALL.into_iter().filter(|&l| l != target).collect();
            for (s, p) in slots.iter().enumerate() {
                let want = format!("{}_value", p.name().to_ascii_lowercase());
                assert_eq!(names[s * SLOT_WIDTH], want);
            }
        }

        let md: Vec<f64> = (0..9).map(|i| 100.0 + 0.5 * f64::from(i)).collect();
        let v = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0];
        let dev = depth_local_dev(&v, &md, 2.5, 3).unwrap();
        assert_eq!((dev[0], dev[4]), (-2.5, 1.0));
        let g = depth_gradient(&v, &md, 3).unwrap();
        assert_eq!((g[0], g[1]), (4.0, 1.0));
        let g = depth_gradient(&[1.0, 4.0, 2.0, 5.0], &[0.0, 0.5, 1.5, 2.0], 3).unwrap();
        for (got, want) in g.iter().zip([6.0, 3.333_333, 3.333_333, 6.0]) {
            assert!((got - want).abs() < 1e-5, "{g:?}");
        }

        let mut p = Project::new();
        let state = p.state(1);
        let cfg = FillConfig::default();
        let before = build_well_features(&p.inputs(), &state, LogKind::Gr, &cfg).unwrap();
        assert_eq!(before.x.n_rows, state.logs.grid.len);
        p.shift_neighbor(1, LogKind::Gr, 100.0);
        let after = build_well_features(&p.inputs(), &state, LogKind::Gr, &cfg).unwrap();
        assert_eq!(bits(&before.x), bits(&after.x));
        assert_eq!(before.rows, after.rows);
    }

    #[test]
    fn fill_dropout_same_seed_same_rows() {
        assert_eq!(stream_seed(0, DROPOUT_STREAM, 0), 0x0a05_e365_0bc9_2932);
        let n = 8;
        let data: Vec<f32> = (0..n * N_FEATURES).map(|i| i as f32 * 0.5 + 1.0).collect();
        let x = FeatureMatrix::new(n, feature_names(LogKind::Gr), data).unwrap();
        let y: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cfg = FillConfig::default();
        let (xa, ya) = apply_dropout(&x, &y, LogKind::Gr, 7, &cfg).unwrap();
        let (xb, yb) = apply_dropout(&x, &y, LogKind::Gr, 7, &cfg).unwrap();
        assert_eq!(xa.n_rows, 10);
        assert_eq!(bits(&xa), bits(&xb));
        assert_eq!(ya, yb);
        assert_eq!(&xa.data[..n * N_FEATURES], &x.data[..]);
        assert_eq!(&ya[..n], &y[..]);
    }
}
