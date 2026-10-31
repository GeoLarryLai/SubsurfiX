// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `neighbor_curves.parquet` as an Arrow batch.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    PrimitiveArray, UInt8Array,
};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type, Schema,
    UInt8Type,
};
use arrow::record_batch::RecordBatch;

use crate::error::{SxError, SxResult};
use crate::io::parquet::{schema_tag, with_schema_tag};
use crate::neighbors::{NeighborCurve, WellNeighborCurves};
use crate::types::{GridRange, LogKind, WellId, GRID_STEP_FT};

/// Schema tag of `neighbor_curves.parquet`.
const NEIGHBOR_CURVES_TAG: &str = "subsurfix.neighbor_curves/1";

/// Columns per log.
const PER_LOG: usize = 7;

/// Column prefix and unit suffix of a log.
fn prefix_unit(log: LogKind) -> (&'static str, &'static str) {
    match log {
        LogKind::Gr => ("gr", "api"),
        LogKind::Rhob => ("rhob", "gcc"),
        LogKind::Nphi => ("nphi", "frac"),
        LogKind::Dtc => ("dtc", "usft"),
        LogKind::Resd => ("resd", "log10ohmm"),
        LogKind::Pef => ("pef", "bfrac"),
    }
}

/// Every field of the table, in order.
fn fields() -> Vec<Field> {
    let mut out = vec![
        Field::new("well_id", DataType::Int32, false),
        Field::new("k", DataType::Int32, false),
        Field::new("md_ft", DataType::Float64, false),
    ];
    for log in LogKind::ALL {
        let (p, u) = prefix_unit(log);
        out.extend([
            Field::new(format!("{p}_nbr_{u}"), DataType::Float32, false),
            Field::new(format!("{p}_nbr_spread_{u}"), DataType::Float32, false),
            Field::new(format!("{p}_nbr_n"), DataType::Int64, false),
            Field::new(format!("{p}_nbr_dist_m"), DataType::Float64, false),
            Field::new(format!("{p}_nbr_quality_frac"), DataType::Float32, false),
            Field::new(format!("{p}_nbr_tier"), DataType::UInt8, false),
            Field::new(format!("{p}_nbr_trusted"), DataType::Boolean, false),
        ]);
    }
    out
}

/// `neighbor_curves.parquet` schema, rows sorted by (well_id, k).
///
/// One row per `(well_id, k)` over each well's whole grid; per log in suite
/// order: value and spread (float32, working units, RESD in log10), `n` (int64),
/// distance trust (float64, m), quality (float32), tier (uint8, 255 = none),
/// trusted (bool). Schema metadata `subsurfix_schema = "subsurfix.neighbor_curves/1"`.
/// NaN rule: NaN stored as is. Errors: `InvalidInput` if a map key differs
/// from its well id or the batch cannot be built; `Shape` if a log is missing
/// or a vector length differs from the grid. Empty map: a batch with no rows.
/// Complexity: `O(rows)`.
pub fn neighbor_curves_to_batch(
    curves: &BTreeMap<WellId, WellNeighborCurves>,
) -> SxResult<RecordBatch> {
    let total: usize = curves.values().map(|w| w.grid.len).sum();
    let mut well_id = Vec::with_capacity(total);
    let mut k = Vec::with_capacity(total);
    let mut md_ft = Vec::with_capacity(total);
    for (&id, w) in curves {
        if id != w.id {
            return Err(SxError::InvalidInput(format!(
                "neighbor curves: key {} holds well {}",
                id.0, w.id.0
            )));
        }
        for i in 0..w.grid.len {
            well_id.push(id.0);
            k.push(w.grid.k(i));
            md_ft.push(w.grid.md_ft(i));
        }
    }
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(well_id)),
        Arc::new(Int32Array::from(k)),
        Arc::new(Float64Array::from(md_ft)),
    ];
    for log in LogKind::ALL {
        let mut value = Vec::with_capacity(total);
        let mut spread = Vec::with_capacity(total);
        let mut n = Vec::with_capacity(total);
        let mut dist = Vec::with_capacity(total);
        let mut quality = Vec::with_capacity(total);
        let mut tier = Vec::with_capacity(total);
        let mut trusted = Vec::with_capacity(total);
        for w in curves.values() {
            let c = w.curves.get(&log).ok_or_else(|| {
                SxError::Shape(format!("well {}: no {} neighbor curve", w.id.0, log.name()))
            })?;
            let len = w.grid.len;
            if [
                c.value.len(),
                c.spread.len(),
                c.n_donors.len(),
                c.dist_m.len(),
                c.quality.len(),
                c.tier.len(),
                c.trusted.len(),
            ]
            .iter()
            .any(|&l| l != len)
            {
                return Err(SxError::Shape(format!(
                    "well {}: {} neighbor curve length differs from grid of {len}",
                    w.id.0,
                    log.name()
                )));
            }
            value.extend_from_slice(&c.value);
            spread.extend_from_slice(&c.spread);
            n.extend(c.n_donors.iter().map(|&x| i64::from(x)));
            dist.extend_from_slice(&c.dist_m);
            quality.extend_from_slice(&c.quality);
            tier.extend_from_slice(&c.tier);
            trusted.extend_from_slice(&c.trusted);
        }
        cols.extend([
            Arc::new(Float32Array::from(value)) as ArrayRef,
            Arc::new(Float32Array::from(spread)),
            Arc::new(Int64Array::from(n)),
            Arc::new(Float64Array::from(dist)),
            Arc::new(Float32Array::from(quality)),
            Arc::new(UInt8Array::from(tier)),
            Arc::new(BooleanArray::from(trusted)),
        ]);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields())), cols)
        .map_err(|e| SxError::InvalidInput(format!("cannot build neighbor curves batch: {e}")))?;
    with_schema_tag(batch, NEIGHBOR_CURVES_TAG)
}

fn format_err(msg: impl Into<String>) -> SxError {
    SxError::Format(format!("neighbor curves: {}", msg.into()))
}

fn prim<T: ArrowPrimitiveType>(batch: &RecordBatch, i: usize) -> SxResult<&PrimitiveArray<T>> {
    batch
        .columns()
        .get(i)
        .and_then(|c| c.as_primitive_opt::<T>())
        .ok_or_else(|| format_err(format!("column {i} has the wrong type")))
}

fn slice<T: Clone>(values: &[T], start: usize, len: usize) -> SxResult<Vec<T>> {
    values
        .get(start..start + len)
        .map(<[T]>::to_vec)
        .ok_or_else(|| format_err("row range outside column"))
}

/// Inverse of `neighbor_curves_to_batch`. Errors: `Format` on a schema mismatch.
///
/// Also `Format` for a missing or different schema tag, nulls, a broken
/// row-order contract (`md_ft == 0.5 · k`, ids ascending, `k` rising by 1
/// within a well) or an `n` outside `u32`. NaN rule: NaN read as is. Empty
/// batch: empty map. Complexity: `O(rows)`.
pub fn neighbor_curves_from_batch(
    batch: &RecordBatch,
) -> SxResult<BTreeMap<WellId, WellNeighborCurves>> {
    let want = fields();
    let got = batch.schema_ref().fields();
    if got.len() != want.len()
        || got
            .iter()
            .zip(&want)
            .any(|(g, w)| g.name() != w.name() || g.data_type() != w.data_type())
    {
        return Err(format_err("schema mismatch"));
    }
    if schema_tag(batch) != Some(NEIGHBOR_CURVES_TAG) {
        return Err(format_err("schema tag mismatch"));
    }
    if batch.columns().iter().any(|c| c.null_count() > 0) {
        return Err(format_err("null values"));
    }
    let well = prim::<Int32Type>(batch, 0)?.values();
    let k = prim::<Int32Type>(batch, 1)?.values();
    let md = prim::<Float64Type>(batch, 2)?.values();
    // (well, first row, row count, first k)
    let mut groups: Vec<(i32, usize, usize, i32)> = Vec::new();
    for (r, ((&w, &kk), &m)) in well.iter().zip(k.iter()).zip(md.iter()).enumerate() {
        if m != GRID_STEP_FT * f64::from(kk) {
            return Err(format_err(format!("row {r}: md_ft is not 0.5 * k")));
        }
        match groups.last_mut() {
            Some(g) if g.0 == w => {
                if i64::from(kk) != i64::from(g.3) + g.2 as i64 {
                    return Err(format_err(format!("row {r}: k does not follow")));
                }
                g.2 += 1;
            }
            Some(g) if w < g.0 => {
                return Err(format_err(format!("row {r}: well_id decreases")));
            }
            _ => groups.push((w, r, 1, kk)),
        }
    }
    let mut columns = Vec::with_capacity(LogKind::ALL.len());
    for (j, log) in LogKind::ALL.into_iter().enumerate() {
        let base = 3 + PER_LOG * j;
        let trusted: Vec<bool> = batch
            .columns()
            .get(base + 6)
            .and_then(|c| c.as_boolean_opt())
            .ok_or_else(|| format_err("trusted column has the wrong type"))?
            .values()
            .iter()
            .collect();
        columns.push((
            log,
            prim::<Float32Type>(batch, base)?.values(),
            prim::<Float32Type>(batch, base + 1)?.values(),
            prim::<Int64Type>(batch, base + 2)?.values(),
            prim::<Float64Type>(batch, base + 3)?.values(),
            prim::<Float32Type>(batch, base + 4)?.values(),
            prim::<UInt8Type>(batch, base + 5)?.values(),
            trusted,
        ));
    }
    let mut out = BTreeMap::new();
    for &(w, start, len, k_start) in &groups {
        let grid = GridRange::new(k_start, len).map_err(|e| format_err(e.to_string()))?;
        let mut curves = BTreeMap::new();
        for (log, value, spread, n, dist, quality, tier, trusted) in &columns {
            let n_donors = slice(n, start, len)?
                .into_iter()
                .map(|x| u32::try_from(x).map_err(|_| format_err(format!("n {x} outside u32"))))
                .collect::<SxResult<Vec<u32>>>()?;
            curves.insert(
                *log,
                NeighborCurve {
                    value: slice(value, start, len)?,
                    spread: slice(spread, start, len)?,
                    n_donors,
                    dist_m: slice(dist, start, len)?,
                    quality: slice(quality, start, len)?,
                    tier: slice(tier, start, len)?,
                    trusted: slice(trusted, start, len)?,
                },
            );
        }
        let id = WellId(w);
        out.insert(id, WellNeighborCurves { id, grid, curves });
    }
    Ok(out)
}
