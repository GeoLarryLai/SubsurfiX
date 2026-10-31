// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[dtw]` interval and full-well / window span settings.

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

/// Node budget and cost settings for interval DTW (`[dtw.interval]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntervalParams {
    /// total node budget shared by the intervals of a pair (nodes per pair)
    pub grid_budget: usize,
    /// fewest nodes per interval (nodes)
    pub min_nodes: usize,
    /// most nodes per interval (nodes)
    pub max_nodes: usize,
    /// floor of the shared-node requirement (nodes)
    pub min_shared_floor: usize,
    /// cap of the shared-node requirement (nodes)
    pub min_shared_cap: usize,
    /// nodes divided by this give the shared-node requirement
    pub min_shared_div: usize,
    /// DTW band as a fraction of the node count (fraction of `N`)
    pub band_frac: f64,
    /// cost of a cell with no log finite at both nodes (cost units)
    pub gap_cost: f64,
    /// whole-well finite samples a log needs (samples)
    pub min_channel_samples: usize,
    /// finite nodes a log needs in an interval (nodes)
    pub min_channel_nodes: usize,
    /// widest NaN gap bridged when resampling (node steps)
    pub max_gap_steps: f64,
    /// low percentile of the linear rescale (percent)
    pub pct_low: f64,
    /// high percentile of the linear rescale (percent)
    pub pct_high: f64,
    /// smallest percentile span kept (working units)
    pub min_span: f64,
    /// quality factor of a DTW match
    pub dtw_factor: f64,
    /// quality factor of a linear match
    pub linear_factor: f64,
    /// floor applied before log10 of RESD (ohm·m)
    pub resd_floor_ohmm: f64,
}

impl Default for IntervalParams {
    fn default() -> Self {
        Self {
            grid_budget: 512,
            min_nodes: 24,
            max_nodes: 128,
            min_shared_floor: 8,
            min_shared_cap: 64,
            min_shared_div: 4,
            band_frac: 0.12,
            gap_cost: 1.5,
            min_channel_samples: 8,
            min_channel_nodes: 8,
            max_gap_steps: 4.0,
            pct_low: 1.0,
            pct_high: 99.0,
            min_span: 1e-9,
            dtw_factor: 1.0,
            linear_factor: 0.45,
            resd_floor_ohmm: 1e-6,
        }
    }
}

/// Resampling and local-cost settings shared by `[dtw.full]` and `[dtw.window]`.
///
/// [`Default`] is the full-span column. [`SpanParams::window_default`] is the window column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpanParams {
    /// fewest resampled nodes (nodes)
    pub min_nodes: usize,
    /// most resampled nodes (nodes)
    pub max_nodes: usize,
    /// DTW band as a fraction of the node count (fraction of `N`)
    pub band_frac: f64,
    /// cost weight of GR when valid (other logs share the rest)
    pub gr_weight: f64,
    /// cap of the robust z in the local cost (robust z)
    pub z_cap: f64,
    /// MAD to sigma factor (see 4.9)
    pub mad_factor: f64,
    /// floor of the pooled robust scale (working units)
    pub scale_floor: f64,
    /// widest NaN gap bridged when resampling (node steps)
    pub max_gap_steps: f64,
    /// smallest gap width bridged (ft)
    pub min_gap_ft: f64,
    /// floor applied before log10 of RESD (ohm·m)
    pub resd_floor_ohmm: f64,
}

impl Default for SpanParams {
    fn default() -> Self {
        Self {
            min_nodes: 24,
            max_nodes: 2000,
            band_frac: 0.12,
            gr_weight: 0.5,
            z_cap: 3.0,
            mad_factor: 1.482602218505602, // = MAD_TO_SIGMA
            scale_floor: 1e-6,
            max_gap_steps: 4.0,
            min_gap_ft: 1.0,
            resd_floor_ohmm: 1e-6,
        }
    }
}

impl SpanParams {
    /// Window-mode defaults. Same as [`Default`] except `band_frac` is 0.25.
    pub fn window_default() -> Self {
        Self {
            band_frac: 0.25,
            ..Self::default()
        }
    }
}

/// `[dtw]` container: interval params, full-well span, and window span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtwConfig {
    /// interval DTW
    pub interval: IntervalParams,
    /// full-well span DTW
    pub full: SpanParams,
    /// window span DTW; a partial table keeps window defaults (4.4)
    #[serde(default = "SpanParams::window_default", deserialize_with = "de_window")]
    pub window: SpanParams,
}

impl Default for DtwConfig {
    fn default() -> Self {
        Self {
            interval: IntervalParams::default(),
            full: SpanParams::default(),
            window: SpanParams::window_default(),
        }
    }
}

fn de_window<'de, D: Deserializer<'de>>(deserializer: D) -> Result<SpanParams, D::Error> {
    let given = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
    let base_val = serde_json::to_value(SpanParams::window_default())
        .map_err(|_| <D::Error as serde::de::Error>::custom("internal"))?;
    let serde_json::Value::Object(mut base) = base_val else {
        return Err(<D::Error as serde::de::Error>::custom("internal"));
    };
    for (key, value) in given {
        base.insert(key, value);
    }
    serde_json::from_value(serde_json::Value::Object(base)).map_err(serde::de::Error::custom)
}
