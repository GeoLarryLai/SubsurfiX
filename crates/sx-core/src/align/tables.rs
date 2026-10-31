// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Arrow record batches of the four rough-alignment outputs.
//!
//! Each batch carries schema metadata `subsurfix_schema` = `subsurfix.<name>/1`.
//! Writing the Parquet files belongs to the stage runner.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, ListArray, StringArray, UInt8Array,
};
use arrow::datatypes::{Field, Float32Type, Int32Type, Schema, UInt8Type};
use arrow::record_batch::RecordBatch;

use crate::align::intervals::IntervalInfo;
use crate::align::pairs::PairMap;
use crate::align::run::{BoundaryRow, RoughAlignment};
use crate::error::{SxError, SxResult};
use crate::io::parquet::with_schema_tag;
use crate::types::{TopSet, WellId, CODE_NONE};

const INTERVALS_TAG: &str = "subsurfix.intervals/1";
const TOPS_ESTIMATED_TAG: &str = "subsurfix.tops_estimated/1";
const PAIRS_ROUGH_TAG: &str = "subsurfix.pairs_rough/1";
const LAYER_COORD_TAG: &str = "subsurfix.layer_coord/1";

/// Name of boundary `b`: `pad_above`, the top name, or `pad_below`.
fn boundary_name(top_set: &TopSet, b: usize) -> String {
    let n = top_set.len();
    if b == 0 {
        "pad_above".to_string()
    } else if b == n + 1 {
        "pad_below".to_string()
    } else {
        top_set.names.get(b - 1).cloned().unwrap_or_default()
    }
}

fn finish(cols: Vec<(&str, ArrayRef)>, tag: &str) -> SxResult<RecordBatch> {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(name, a)| Field::new(*name, a.data_type().clone(), false))
        .collect();
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| SxError::InvalidInput(format!("cannot build batch: {e}")))?;
    with_schema_tag(batch, tag)
}

fn i32s(values: impl Iterator<Item = i32>) -> ArrayRef {
    Arc::new(Int32Array::from_iter_values(values))
}

fn f64s(values: impl Iterator<Item = f64>) -> ArrayRef {
    Arc::new(Float64Array::from_iter_values(values))
}

fn u8s(values: impl Iterator<Item = u8>) -> ArrayRef {
    Arc::new(UInt8Array::from_iter_values(values))
}

/// `intervals` table: one row per well per interval, sorted by
/// `(well_id, interval)`. Columns `well_id` int32, `interval` int32,
/// `top_name` utf8, `base_name` utf8, `top_md_ft`, `base_md_ft`,
/// `thickness_ft` float64, `state` uint8, `confidence` float32.
///
/// NaN rule: NaN MDs and thicknesses are stored as is. Errors:
/// [`SxError::InvalidInput`] if the batch cannot be built. Complexity:
/// `O(n log n)`.
pub fn intervals_batch(r: &RoughAlignment) -> SxResult<RecordBatch> {
    let mut rows: Vec<&IntervalInfo> = r.intervals.iter().collect();
    rows.sort_by_key(|i| (i.row.well, i.row.interval));
    let name = |offset: usize| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(rows.iter().map(|i| {
            boundary_name(&r.top_set, usize::from(i.row.interval) + offset)
        })))
    };
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("well_id", i32s(rows.iter().map(|i| i.row.well.0))),
        (
            "interval",
            i32s(rows.iter().map(|i| i32::from(i.row.interval))),
        ),
        ("top_name", name(0)),
        ("base_name", name(1)),
        ("top_md_ft", f64s(rows.iter().map(|i| i.row.top_md_ft))),
        ("base_md_ft", f64s(rows.iter().map(|i| i.row.base_md_ft))),
        ("thickness_ft", f64s(rows.iter().map(|i| i.thickness_ft))),
        ("state", u8s(rows.iter().map(|i| i.row.state as u8))),
        (
            "confidence",
            Arc::new(Float32Array::from_iter_values(
                rows.iter().map(|i| i.confidence),
            )),
        ),
    ];
    finish(cols, INTERVALS_TAG)
}

/// `tops_estimated` table: one row per well per boundary
/// `0..=n+1`, sorted by `(well_id, boundary)`. Columns `well_id` int32,
/// `boundary` int32, `top_name` utf8, `md_ft`, `elev_ft` float64, `source`
/// uint8 (255 = none), `status` uint8, `n_donors` int32.
///
/// NaN rule: NaN depths are stored as is. Errors: [`SxError::InvalidInput`] if
/// the batch cannot be built. Complexity: `O(n log n)`.
pub fn tops_estimated_batch(r: &RoughAlignment) -> SxResult<RecordBatch> {
    let mut rows: Vec<&BoundaryRow> = r.boundaries.iter().collect();
    rows.sort_by_key(|b| (b.well, b.boundary));
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("well_id", i32s(rows.iter().map(|b| b.well.0))),
        ("boundary", i32s(rows.iter().map(|b| i32::from(b.boundary)))),
        (
            "top_name",
            Arc::new(StringArray::from_iter_values(
                rows.iter()
                    .map(|b| boundary_name(&r.top_set, usize::from(b.boundary))),
            )),
        ),
        ("md_ft", f64s(rows.iter().map(|b| b.md_ft))),
        ("elev_ft", f64s(rows.iter().map(|b| b.elev_ft))),
        (
            "source",
            u8s(rows.iter().map(|b| b.source.map_or(CODE_NONE, |s| s as u8))),
        ),
        ("status", u8s(rows.iter().map(|b| b.status as u8))),
        ("n_donors", i32s(rows.iter().map(|b| i32::from(b.n_donors)))),
    ];
    finish(cols, TOPS_ESTIMATED_TAG)
}

/// `pairs_rough` table: one row per directed pair, sorted by
/// `(target_id, rank)`. Columns `target_id`, `donor_id`, `rank` int32,
/// `distance_m` float64, `status` uint8, `quality` float32, `span_ratio`,
/// `total_cost`, `mean_cost` float64, `n_dtw_intervals`,
/// `n_linear_intervals` int32, `intervals` list<int32>, `interval_quality`
/// list<float32>, `interval_method` list<uint8>, `k_target`, `k_donor`
/// list<int32>.
///
/// NaN rule: NaN ratios and costs are stored as is. Errors:
/// [`SxError::InvalidInput`] if the batch cannot be built. Complexity:
/// `O(n log n + path pairs)`.
pub fn pairs_rough_batch(r: &RoughAlignment) -> SxResult<RecordBatch> {
    let mut rows: Vec<&PairMap> = r.pairs.iter().collect();
    rows.sort_by_key(|m| (m.path.a, m.rank));
    let list_i32 = |f: &dyn Fn(&PairMap) -> Vec<i32>| -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
            rows.iter().map(|m| Some(f(m).into_iter().map(Some))),
        ))
    };
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("target_id", i32s(rows.iter().map(|m| m.path.a.0))),
        ("donor_id", i32s(rows.iter().map(|m| m.path.b.0))),
        ("rank", i32s(rows.iter().map(|m| i32::from(m.rank)))),
        ("distance_m", f64s(rows.iter().map(|m| m.distance_m))),
        ("status", u8s(rows.iter().map(|m| m.status as u8))),
        (
            "quality",
            Arc::new(Float32Array::from_iter_values(
                rows.iter().map(|m| m.path.quality),
            )),
        ),
        ("span_ratio", f64s(rows.iter().map(|m| m.span_ratio))),
        ("total_cost", f64s(rows.iter().map(|m| m.path.cost))),
        ("mean_cost", f64s(rows.iter().map(|m| m.mean_cost))),
        (
            "n_dtw_intervals",
            i32s(rows.iter().map(|m| i32::from(m.n_dtw_intervals))),
        ),
        (
            "n_linear_intervals",
            i32s(rows.iter().map(|m| i32::from(m.n_linear_intervals))),
        ),
        (
            "intervals",
            list_i32(&|m| m.intervals.iter().map(|&i| i32::from(i)).collect()),
        ),
        (
            "interval_quality",
            Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(
                rows.iter()
                    .map(|m| Some(m.interval_quality.iter().map(|&q| Some(q)))),
            )),
        ),
        (
            "interval_method",
            Arc::new(ListArray::from_iter_primitive::<UInt8Type, _, _>(
                rows.iter()
                    .map(|m| Some(m.interval_method.iter().map(|&c| Some(c)))),
            )),
        ),
        (
            "k_target",
            list_i32(&|m| m.path.path.iter().map(|p| p.0).collect()),
        ),
        (
            "k_donor",
            list_i32(&|m| m.path.path.iter().map(|p| p.1).collect()),
        ),
    ];
    finish(cols, PAIRS_ROUGH_TAG)
}

/// `layer_coord` table: one row per `(well_id, k)` on each well's
/// logs grid, sorted by `well_id` then `k` (row-order contract). Columns
/// `well_id`, `k` int32, `md_ft`, `layer_coord` float64, `interval` int32
/// (−1 none), `interval_state` uint8 (255 none), `coord_source` uint8.
///
/// NaN rule: NaN coordinates are stored as is. Errors:
/// [`SxError::InvalidInput`] if the batch cannot be built. Complexity:
/// `O(samples + n log n)`.
pub fn layer_coord_batch(r: &RoughAlignment) -> SxResult<RecordBatch> {
    let mut states: BTreeMap<WellId, Vec<u8>> = BTreeMap::new();
    for info in &r.intervals {
        let v = states.entry(info.row.well).or_default();
        let i = usize::from(info.row.interval);
        if v.len() <= i {
            v.resize(i + 1, CODE_NONE);
        }
        if let Some(slot) = v.get_mut(i) {
            *slot = info.row.state as u8;
        }
    }
    let mut coords: Vec<_> = r.coords.iter().collect();
    coords.sort_by_key(|c| c.well);
    let total: usize = coords.iter().map(|c| c.grid.len).sum();
    let mut well_id = Vec::with_capacity(total);
    let mut k = Vec::with_capacity(total);
    let mut md_ft = Vec::with_capacity(total);
    let mut coord = Vec::with_capacity(total);
    let mut interval = Vec::with_capacity(total);
    let mut state = Vec::with_capacity(total);
    let mut source = Vec::with_capacity(total);
    for c in coords {
        let well_states = states.get(&c.well);
        for i in 0..c.grid.len {
            let iv = c.interval.get(i).copied().unwrap_or(-1);
            well_id.push(c.well.0);
            k.push(c.grid.k(i));
            md_ft.push(c.grid.md_ft(i));
            coord.push(c.layer_coord.get(i).copied().unwrap_or(f64::NAN));
            interval.push(iv);
            state.push(
                usize::try_from(iv)
                    .ok()
                    .and_then(|j| well_states?.get(j).copied())
                    .unwrap_or(CODE_NONE),
            );
            source.push(c.source as u8);
        }
    }
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("well_id", Arc::new(Int32Array::from(well_id))),
        ("k", Arc::new(Int32Array::from(k))),
        ("md_ft", Arc::new(Float64Array::from(md_ft))),
        ("layer_coord", Arc::new(Float64Array::from(coord))),
        ("interval", Arc::new(Int32Array::from(interval))),
        ("interval_state", Arc::new(UInt8Array::from(state))),
        ("coord_source", Arc::new(UInt8Array::from(source))),
    ];
    finish(cols, LAYER_COORD_TAG)
}
