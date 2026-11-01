// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Frozen core types shared by every module.
//!
//! Changing anything here needs an Opus decision and a ledger entry.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{SxError, SxResult};

/// Grid step of the global measured-depth grid, in feet.
pub const GRID_STEP_FT: f64 = 0.5;

/// Provenance / code value meaning "no value here".
pub const CODE_NONE: u8 = 255;

/// Stable integer id of a well inside one project.
///
/// Assigned once when a well first enters the project (ascending order of the
/// well's name string at creation), then never changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WellId(pub i32);

/// A contiguous run of the global MD grid: samples `k_start .. k_start + len`.
///
/// Sample `i` (0-based within the range) sits at grid index `k_start + i` and
/// `md_ft = 0.5 * (k_start + i)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridRange {
    /// Global grid index of the first sample.
    pub k_start: i32,
    /// Number of samples.
    pub len: usize,
}

impl GridRange {
    /// Build a range; errors if `len` overflows the `i32` grid index.
    pub fn new(k_start: i32, len: usize) -> SxResult<Self> {
        let len_i64 = i64::try_from(len)
            .map_err(|_| SxError::InvalidInput("grid length too large".into()))?;
        if i64::from(k_start) + len_i64 > i64::from(i32::MAX) {
            return Err(SxError::InvalidInput("grid range exceeds i32 index".into()));
        }
        Ok(Self { k_start, len })
    }

    /// Global grid index of local sample `i`. No bounds check.
    pub fn k(&self, i: usize) -> i32 {
        self.k_start + i as i32
    }

    /// Measured depth (ft) of local sample `i`. No bounds check.
    pub fn md_ft(&self, i: usize) -> f64 {
        GRID_STEP_FT * f64::from(self.k(i))
    }

    /// Local index of global grid index `k`, or `None` if outside the range.
    pub fn index_of_k(&self, k: i32) -> Option<usize> {
        let i = i64::from(k) - i64::from(self.k_start);
        if i >= 0 && (i as u64) < self.len as u64 {
            Some(i as usize)
        } else {
            None
        }
    }

    /// True if the range has no samples.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The logs the engine knows by name.
///
/// Column name and unit suffix are fixed by [`LogKind::column`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LogKind {
    /// Gamma ray, API units.
    Gr = 0,
    /// Bulk density, g/cm³.
    Rhob = 1,
    /// Neutron porosity, fraction (m³/m³).
    Nphi = 2,
    /// Compressional slowness, µs/ft.
    Dtc = 3,
    /// Deep resistivity, ohm·m (math in log10).
    Resd = 4,
    /// Photoelectric factor, barns/electron.
    Pef = 5,
}

impl LogKind {
    /// All known logs in suite order (the `u8` id order). Specs that say
    /// "in suite order" mean this order.
    pub const ALL: [LogKind; 6] = [
        LogKind::Gr,
        LogKind::Rhob,
        LogKind::Nphi,
        LogKind::Dtc,
        LogKind::Resd,
        LogKind::Pef,
    ];

    /// Short upper-case name, e.g. `"GR"`.
    pub fn name(self) -> &'static str {
        match self {
            LogKind::Gr => "GR",
            LogKind::Rhob => "RHOB",
            LogKind::Nphi => "NPHI",
            LogKind::Dtc => "DTC",
            LogKind::Resd => "RESD",
            LogKind::Pef => "PEF",
        }
    }

    /// Parquet column name with unit suffix, e.g. `"gr_api"`.
    pub fn column(self) -> &'static str {
        match self {
            LogKind::Gr => "gr_api",
            LogKind::Rhob => "rhob_gcc",
            LogKind::Nphi => "nphi_frac",
            LogKind::Dtc => "dtc_usft",
            LogKind::Resd => "resd_ohmm",
            LogKind::Pef => "pef_bfrac",
        }
    }

    /// Parse a short name (case-insensitive).
    pub fn from_name(s: &str) -> SxResult<Self> {
        let up = s.trim().to_ascii_uppercase();
        LogKind::ALL
            .iter()
            .copied()
            .find(|k| k.name() == up)
            .ok_or_else(|| SxError::NotFound(format!("unknown log name {s:?}")))
    }
}

/// Where a curve sample came from. Stored as `u8`; [`CODE_NONE`] = outside the
/// well's coverage (never filled).
///
/// Codes `40..=49` are neighbor/donor copies and `50..=59` are shaped gap
/// fills, so `code / 10` gives the fill stage (4 or 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Provenance {
    /// Measured value that passed QC.
    Measured = 0,
    /// Gradient-boosted tree prediction written into a gap.
    Model = 1,
    /// Gap inside coverage that no rule could fill; value stays NaN.
    Unfilled = 9,
    /// Trusted weighted neighbor curve copied in.
    NeighborWeighted = 40,
    /// Best single real donor well copied in.
    RealDonor = 41,
    /// Untrusted neighbor curve copied in (last copy option).
    NeighborWeak = 42,
    /// Shaped gap fill: neighbor curve shape tilted to the gap ends.
    NeighborShape = 50,
    /// Shaped gap fill: linear fit from the well's other logs.
    CrossLogFit = 51,
    /// Shaped gap fill: donor well shape tilted to the gap ends.
    DonorShape = 52,
    /// Shaped gap fill: model level plus the well's own texture.
    OwnTexture = 53,
    /// Shaped gap fill: model guess, tilted to the gap ends if needed.
    SmoothedModel = 54,
}

impl Provenance {
    /// `u8` code.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// Parse a code; `None` for [`CODE_NONE`] or an unknown code.
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Provenance::Measured,
            1 => Provenance::Model,
            9 => Provenance::Unfilled,
            40 => Provenance::NeighborWeighted,
            41 => Provenance::RealDonor,
            42 => Provenance::NeighborWeak,
            50 => Provenance::NeighborShape,
            51 => Provenance::CrossLogFit,
            52 => Provenance::DonorShape,
            53 => Provenance::OwnTexture,
            54 => Provenance::SmoothedModel,
            _ => return None,
        })
    }

    /// True for any value that was written by a fill rule.
    pub fn is_filled(self) -> bool {
        !matches!(self, Provenance::Measured | Provenance::Unfilled)
    }

    /// Fill stage: 0 measured or unfilled, 1 model, 4 copy, 5 shaped fill.
    pub fn stage(self) -> u8 {
        match self {
            Provenance::Measured | Provenance::Unfilled => 0,
            Provenance::Model => 1,
            other => other.code() / 10,
        }
    }
}

/// Header of one well. Coordinates are in the project frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WellHeader {
    /// Stable id.
    pub id: WellId,
    /// External well name (for example a 14-digit API number), as text.
    pub name: String,
    /// Surface x, meters, project frame.
    pub x_m: f64,
    /// Surface y, meters, project frame.
    pub y_m: f64,
    /// Kelly bushing elevation above sea level, feet.
    pub kb_elev_ft: f64,
}

/// Per-sample geometry of one well on its grid range.
///
/// All vectors have length `grid.len`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WellGeometry {
    /// Well id.
    pub id: WellId,
    /// Grid range covered.
    pub grid: GridRange,
    /// True vertical depth below KB, feet, positive down.
    pub tvd_ft: Vec<f64>,
    /// Elevation, feet, positive up (`kb_elev_ft - tvd_ft`).
    pub elev_ft: Vec<f64>,
    /// Sample x, meters, project frame.
    pub x_m: Vec<f64>,
    /// Sample y, meters, project frame.
    pub y_m: Vec<f64>,
}

impl WellGeometry {
    /// Check that every vector has length `grid.len`.
    pub fn validate(&self) -> SxResult<()> {
        let n = self.grid.len;
        for (name, len) in [
            ("tvd_ft", self.tvd_ft.len()),
            ("elev_ft", self.elev_ft.len()),
            ("x_m", self.x_m.len()),
            ("y_m", self.y_m.len()),
        ] {
            if len != n {
                return Err(SxError::Shape(format!(
                    "well {}: {name} has {len} samples, grid has {n}",
                    self.id.0
                )));
            }
        }
        Ok(())
    }
}

/// Borehole-condition curves of one well on its grid, used only by QC
/// (bad-hole flags). Not part of the six modelled logs.
#[derive(Debug, Clone, PartialEq)]
pub struct WellAux {
    /// Well id.
    pub id: WellId,
    /// Grid range (same as the well's logs).
    pub grid: GridRange,
    /// Caliper, inches; NaN = missing (all NaN when the well has none).
    pub cal_in: Vec<f32>,
    /// Density correction (DRHO, RHOC, HDRA ...), g/cm³; NaN = missing.
    pub drho: Vec<f32>,
    /// Bit size (BS), inches; NaN = missing. Nominal hole size for bad-hole QC.
    pub bs_in: Vec<f32>,
    /// Logging speed (LSPD ...), ft/min, absolute; NaN = missing. Report only.
    pub lspd_ftmin: Vec<f32>,
    /// Temperature curve (TEMP, MTEM, BHT ...), °C; NaN = missing. Report only:
    /// a logging-tool or mud temperature, not a formation temperature.
    pub temp_c: Vec<f32>,
}

/// Curves of one well on its grid range, with one provenance code per sample.
///
/// Every curve and every provenance vector has length `grid.len`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WellLogs {
    /// Well id.
    pub id: WellId,
    /// Grid range covered.
    pub grid: GridRange,
    /// Curve values; NaN = missing.
    pub curves: BTreeMap<LogKind, Vec<f32>>,
    /// Provenance code per sample per log; [`CODE_NONE`] where the value is NaN
    /// and no fill was attempted.
    pub provenance: BTreeMap<LogKind, Vec<u8>>,
}

impl WellLogs {
    /// Empty set of curves on `grid`.
    pub fn new(id: WellId, grid: GridRange) -> Self {
        Self {
            id,
            grid,
            curves: BTreeMap::new(),
            provenance: BTreeMap::new(),
        }
    }

    /// Check lengths and that every curve has a provenance vector.
    pub fn validate(&self) -> SxResult<()> {
        let n = self.grid.len;
        for (log, values) in &self.curves {
            if values.len() != n {
                return Err(SxError::Shape(format!(
                    "well {}: {} has {} samples, grid has {n}",
                    self.id.0,
                    log.name(),
                    values.len()
                )));
            }
            match self.provenance.get(log) {
                Some(p) if p.len() == n => {}
                Some(p) => {
                    return Err(SxError::Shape(format!(
                        "well {}: {} provenance has {} samples, grid has {n}",
                        self.id.0,
                        log.name(),
                        p.len()
                    )))
                }
                None => {
                    return Err(SxError::Shape(format!(
                        "well {}: {} has no provenance",
                        self.id.0,
                        log.name()
                    )))
                }
            }
        }
        Ok(())
    }
}

/// Ordered list of top (surface) names, shallow to deep.
///
/// Boundary layout used everywhere:
/// boundary 0 = upper pad, boundaries `1..=n` = tops `0..n`, boundary `n+1` =
/// lower pad. Interval `i` lies between boundary `i` and boundary `i+1`, so
/// there are `n+1` intervals and interval 0 / interval `n` are the pads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopSet {
    /// Names, shallow to deep. Unique.
    pub names: Vec<String>,
}

impl TopSet {
    /// Build and check names are unique and non-empty.
    pub fn new(names: Vec<String>) -> SxResult<Self> {
        if names.is_empty() {
            return Err(SxError::EmptyInput("top set has no names".into()));
        }
        let mut seen = std::collections::BTreeSet::new();
        for n in &names {
            if n.trim().is_empty() || !seen.insert(n.as_str()) {
                return Err(SxError::InvalidInput(format!(
                    "bad or repeated top name {n:?}"
                )));
            }
        }
        Ok(Self { names })
    }

    /// Number of tops `n`.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Always false for a valid set.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Number of intervals including the two pads: `n + 1`.
    pub fn n_intervals(&self) -> usize {
        self.names.len() + 1
    }

    /// Index of a top name.
    pub fn index_of(&self, name: &str) -> SxResult<u16> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|i| u16::try_from(i).ok())
            .ok_or_else(|| SxError::NotFound(format!("top {name:?}")))
    }
}

/// Where a top depth came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TopSource {
    /// Picked by the user. Never overwritten.
    User = 0,
    /// Estimated by rough alignment (interpolated from neighbors).
    Estimated = 1,
    /// Virtual pad boundary.
    Pad = 2,
    /// Suggested by the full correlation where the user has no pick.
    Suggested = 3,
}

/// One top depth in one well.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopPick {
    /// Well id.
    pub well: WellId,
    /// Index into [`TopSet::names`].
    pub top: u16,
    /// Measured depth, feet.
    pub md_ft: f64,
    /// Origin of the depth.
    pub source: TopSource,
}

/// State of one interval in one well.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum IntervalState {
    /// Both bounding boundaries are user picks and thickness > 0.
    Observed = 0,
    /// Thickness is zero (or below the pinch threshold). Never interpolate through it.
    Pinched = 1,
    /// At least one boundary is estimated.
    Estimated = 2,
    /// Boundaries cannot be placed.
    Unknown = 3,
    /// Pad interval above the first or below the last top.
    Pad = 4,
}

/// One interval of one well.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntervalRow {
    /// Well id.
    pub well: WellId,
    /// Interval index `0..=n` (see [`TopSet`] for the layout).
    pub interval: u16,
    /// MD of the upper boundary, feet (NaN if unknown).
    pub top_md_ft: f64,
    /// MD of the lower boundary, feet (NaN if unknown).
    pub base_md_ft: f64,
    /// State.
    pub state: IntervalState,
}

/// A warping path between two wells from DTW.
///
/// `path` holds pairs of global grid indices `(k_a, k_b)`, monotone
/// non-decreasing in both, first pair = start anchors, last pair = end anchors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairPath {
    /// First well.
    pub a: WellId,
    /// Second well.
    pub b: WellId,
    /// Interval index the path covers, or `None` for a multi-interval window.
    pub interval: Option<u16>,
    /// Matched grid-index pairs.
    pub path: Vec<(i32, i32)>,
    /// Total accumulated cost along the path.
    pub cost: f64,
    /// Match quality in `[0, 1]`, higher is better.
    pub quality: f32,
}

/// Dense row-major `f32` matrix of model features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureMatrix {
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
    /// Row-major values, length `n_rows * n_cols`. NaN = missing.
    pub data: Vec<f32>,
    /// Column names, length `n_cols`.
    pub names: Vec<String>,
}

impl FeatureMatrix {
    /// Build and check shape.
    pub fn new(n_rows: usize, names: Vec<String>, data: Vec<f32>) -> SxResult<Self> {
        let n_cols = names.len();
        let want = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| SxError::Shape("feature matrix too large".into()))?;
        if data.len() != want {
            return Err(SxError::Shape(format!(
                "feature matrix data has {} values, want {n_rows} x {n_cols}",
                data.len()
            )));
        }
        Ok(Self {
            n_rows,
            n_cols,
            data,
            names,
        })
    }

    /// Row `r` as a slice, or `None` if out of range.
    pub fn row(&self, r: usize) -> Option<&[f32]> {
        if r >= self.n_rows {
            return None;
        }
        let start = r * self.n_cols;
        self.data.get(start..start + self.n_cols)
    }
}

/// A regression model the gap-fill engine can train and use.
///
/// The built-in implementation wraps LightGBM. The Python plug-in slot wraps
/// any object with scikit-learn style `fit` / `predict`.
pub trait Regressor: Send + Sync {
    /// Short name for reports.
    fn name(&self) -> &str;

    /// Train on `x` (rows = samples) and target `y` (length `x.n_rows`).
    /// `weight`, if given, has length `x.n_rows`.
    fn fit(&mut self, x: &FeatureMatrix, y: &[f32], weight: Option<&[f32]>) -> SxResult<()>;

    /// Predict one value per row of `x`.
    fn predict(&self, x: &FeatureMatrix) -> SxResult<Vec<f32>>;
}
