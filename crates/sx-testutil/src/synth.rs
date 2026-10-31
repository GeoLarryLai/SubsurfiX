// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Seeded vertical synthetic wells with known tops and known layer time.
//!
//! Every random draw goes through [`sx_core::util::rng`]. Same config and seed
//! produce identical [`SynthProject::to_bytes`] output.
//!
//! NaN rule for curves: a sample is a finite `f32`, or NaN where a gap was
//! written. Non-finite config fields are rejected by [`generate`].

use std::collections::BTreeSet;

use sx_core::types::{
    GridRange, LogKind, Provenance, TopPick, TopSet, TopSource, WellGeometry, WellHeader, WellId,
    WellLogs, CODE_NONE, GRID_STEP_FT,
};
use sx_core::util::rng::{
    rng_for, shuffle, standard_normal, uniform_below, uniform_int_inclusive, uniform_range,
};
use sx_core::{SxError, SxResult};

/// Regional datum, dip, and two sinusoidal undulations for the top surfaces.
///
/// Elevations are feet. Wavelengths and the dip denominators are meters.
/// NaN rule: [`generate`] rejects any non-finite field.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceConfig {
    /// Elevation of top 0 before dip and waves, feet.
    pub top0_elev_ft: f64,
    /// Nominal thickness of each interval, feet. Positive.
    pub thickness_ft: f64,
    /// East-west dip, feet of elevation per meter of x.
    pub dip_x_ft_per_m: f64,
    /// North-south dip, feet of elevation per meter of y.
    pub dip_y_ft_per_m: f64,
    /// Amplitude of the x-direction wave, feet. Non-negative.
    pub amp1_ft: f64,
    /// Wavelength of the x-direction wave, meters. Positive.
    pub wavelength1_m: f64,
    /// Amplitude of the y-direction wave, feet. Non-negative.
    pub amp2_ft: f64,
    /// Wavelength of the y-direction wave, meters. Positive.
    pub wavelength2_m: f64,
    /// Pad thickness above top 0, feet. Positive.
    pub pad_above_ft: f64,
    /// Pad thickness below the deepest top, feet. Positive.
    pub pad_below_ft: f64,
}

/// One interval collapsed to zero thickness inside a horizontal radius.
///
/// `interval` is a [`TopSet`] interval index (`1..=n_tops-1`), not a top index.
/// NaN rule: [`generate`] rejects a non-finite center or radius.
#[derive(Debug, Clone, PartialEq)]
pub struct PinchConfig {
    /// Interval index pinched out, in `1..=n_tops-1`.
    pub interval: u16,
    /// Circle center x, meters.
    pub center_x_m: f64,
    /// Circle center y, meters.
    pub center_y_m: f64,
    /// Circle radius, meters. Non-negative. Zero pinches only the center point.
    pub radius_m: f64,
}

/// Random NaN runs painted onto curves after the noise sequence is drawn.
///
/// Lengths are sample counts on the well grid. NaN rule: none (integer fields).
#[derive(Debug, Clone, PartialEq)]
pub struct GapConfig {
    /// Minimum number of runs per log per well.
    pub runs_min: u64,
    /// Maximum number of runs per log per well.
    pub runs_max: u64,
    /// Minimum run length, samples. At least 1.
    pub len_min: u64,
    /// Maximum run length, samples.
    pub len_max: u64,
}

/// Inputs for [`generate`].
///
/// Distances are meters, elevations are feet, `seed` selects the random
/// streams. NaN rule: [`generate`] returns [`SxError::InvalidInput`] when any
/// `f64` field is non-finite or outside the range stated on that field.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthConfig {
    /// Seed mixed into every random stream.
    pub seed: u64,
    /// Number of wells. At least 1.
    pub n_wells: usize,
    /// Grid spacing of the well lattice, meters. Positive.
    pub spacing_m: f64,
    /// Half-width of the uniform xy jitter, meters. Non-negative.
    pub jitter_m: f64,
    /// Mean kelly-bushing elevation, feet.
    pub kb_elev_mean_ft: f64,
    /// Half-width of the uniform KB jitter, feet. Non-negative.
    pub kb_elev_jitter_ft: f64,
    /// Number of tops, in `1..=999`.
    pub n_tops: usize,
    /// Shape of the top surfaces.
    pub surface: SurfaceConfig,
    /// Optional zero-thickness interval. `None` keeps every interval open.
    pub pinch: Option<PinchConfig>,
    /// Logs to store, in any order. Empty and repeated kinds are rejected.
    /// Layer means are still drawn for every kind in [`LogKind::ALL`].
    pub logs: Vec<LogKind>,
    /// Optional gap painter. `None` leaves every curve finite.
    pub gaps: Option<GapConfig>,
    /// Fraction of true tops removed from [`SynthProject::tops`], in `[0, 1]`.
    pub drop_top_frac: f64,
}

/// One synthetic well plus the truth used to score it.
///
/// Elevations and layer time are `f64`. Curve samples are `f32` (NaN in gaps).
/// NaN rule: layer time is finite for a project [`generate`] accepted;
/// curve NaNs are gaps and carry provenance [`CODE_NONE`].
#[derive(Debug, Clone, PartialEq)]
pub struct SynthWell {
    /// Surface location, name, and kelly bushing.
    pub header: WellHeader,
    /// Vertical geometry on the global 0.5 ft grid.
    pub geometry: WellGeometry,
    /// Curves requested by [`SynthConfig::logs`], suite order.
    pub logs: WellLogs,
    /// Boundary elevations `B_0..=B_{n+1}`, feet, non-increasing.
    pub boundaries_elev_ft: Vec<f64>,
    /// True layer time per sample, length `geometry.grid.len`.
    pub layer_time: Vec<f64>,
}

/// A synthetic project: tops, wells in ascending [`WellId`], and picks.
///
/// NaN rule: picks are finite measured depths. Wells follow [`SynthWell`].
#[derive(Debug, Clone, PartialEq)]
pub struct SynthProject {
    /// Top names `T01` … in shallow-to-deep order.
    pub top_set: TopSet,
    /// Wells in ascending id order.
    pub wells: Vec<SynthWell>,
    /// Every pick, including pinched tops and tops later dropped.
    pub tops_truth: Vec<TopPick>,
    /// `tops_truth` with dropped rows removed, same relative order.
    pub tops: Vec<TopPick>,
}

impl Default for SurfaceConfig {
    /// Spec defaults: 2000 ft datum, 60 ft layers, mild dip, 100 ft pads.
    fn default() -> Self {
        Self {
            top0_elev_ft: 2000.0,
            thickness_ft: 60.0,
            dip_x_ft_per_m: 0.005,
            dip_y_ft_per_m: -0.0025,
            amp1_ft: 10.0,
            wavelength1_m: 6000.0,
            amp2_ft: 5.0,
            wavelength2_m: 2500.0,
            pad_above_ft: 100.0,
            pad_below_ft: 100.0,
        }
    }
}

impl Default for GapConfig {
    /// One to three runs, each 20 to 200 samples.
    fn default() -> Self {
        Self {
            runs_min: 1,
            runs_max: 3,
            len_min: 20,
            len_max: 200,
        }
    }
}

impl Default for SynthConfig {
    /// 25 wells on a 1000 m lattice, six tops, every log, no pinch or gaps.
    fn default() -> Self {
        Self {
            seed: 0,
            n_wells: 25,
            spacing_m: 1000.0,
            jitter_m: 200.0,
            kb_elev_mean_ft: 3000.0,
            kb_elev_jitter_ft: 50.0,
            n_tops: 6,
            surface: SurfaceConfig::default(),
            pinch: None,
            logs: LogKind::ALL.to_vec(),
            gaps: None,
            drop_top_frac: 0.0,
        }
    }
}

impl SynthProject {
    /// Little-endian fingerprint of every field.
    ///
    /// NaN rule: curve NaNs are stored with [`f32::to_bits`]; nothing is
    /// skipped. Empty input: an empty project writes no well or pick bytes.
    /// Complexity: O(total curve samples).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for name in &self.top_set.names {
            push_str(&mut buf, name);
        }
        for well in &self.wells {
            push_i32(&mut buf, well.header.id.0);
            push_str(&mut buf, &well.header.name);
            push_f64(&mut buf, well.header.x_m);
            push_f64(&mut buf, well.header.y_m);
            push_f64(&mut buf, well.header.kb_elev_ft);
            push_i32(&mut buf, well.geometry.grid.k_start);
            push_u64(&mut buf, well.geometry.grid.len as u64);
            push_f64s(&mut buf, &well.geometry.tvd_ft);
            push_f64s(&mut buf, &well.geometry.elev_ft);
            push_f64s(&mut buf, &well.geometry.x_m);
            push_f64s(&mut buf, &well.geometry.y_m);
            for (log, values) in &well.logs.curves {
                buf.push(*log as u8);
                for value in values {
                    buf.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                if let Some(prov) = well.logs.provenance.get(log) {
                    buf.extend_from_slice(prov);
                }
            }
            push_f64s(&mut buf, &well.boundaries_elev_ft);
            push_f64s(&mut buf, &well.layer_time);
        }
        push_picks(&mut buf, &self.tops_truth);
        push_picks(&mut buf, &self.tops);
        buf
    }
}

/// Build the synthetic project.
///
/// Draw order is part of the result: wells (`synth_wells`), layer means
/// (`synth_layers`), per-log noise (`synth_noise_{LOG}`), gaps (`synth_gaps`),
/// then the top shuffle (`synth_tops`).
///
/// NaN rule: error. A non-finite config field returns [`SxError::InvalidInput`].
/// Curve NaNs are produced only by [`SynthConfig::gaps`].
///
/// Errors: [`SxError::InvalidInput`] for a count, range, pinch, or gap rule
/// broken by `cfg`, and when a well's grid does not fit the `i32` index.
///
/// Complexity: O(`n_wells` · samples · logs).
pub fn generate(cfg: &SynthConfig) -> SxResult<SynthProject> {
    validate(cfg)?;
    let (side, placed) = place_wells(cfg)?;
    let names = top_names(cfg.n_tops);
    let top_set = TopSet::new(names)?;
    let means = draw_layer_means(cfg);
    let logs = sorted_logs(&cfg.logs);

    let mut wells = Vec::with_capacity(placed.len());
    let mut tops_truth = Vec::with_capacity(placed.len().saturating_mul(cfg.n_tops));
    for place in placed {
        let id = place.id;
        let x_m = place.x_m;
        let y_m = place.y_m;
        let kb = place.kb_elev_ft;
        let surfaces = surface_elevations(cfg, side, x_m, y_m);
        let boundaries = boundary_elevations(
            &surfaces,
            cfg.surface.pad_above_ft,
            cfg.surface.pad_below_ft,
        );
        let grid = grid_for_well(kb, &boundaries)?;
        let geometry = build_geometry(id, grid, x_m, y_m, kb);
        let layer_time: Vec<f64> = geometry
            .elev_ft
            .iter()
            .map(|&elev| layer_time_at_elev(&boundaries, elev))
            .collect();
        let mut well_logs = sample_logs(cfg, id, grid, &layer_time, &means, &logs);
        if let Some(gaps) = &cfg.gaps {
            apply_gaps(cfg.seed, id.0, gaps, &logs, &mut well_logs);
        }
        for (j, &elev) in surfaces.iter().enumerate() {
            let top = u16::try_from(j).map_err(|_| invalid("n_tops", "does not fit in u16"))?;
            tops_truth.push(TopPick {
                well: id,
                top,
                md_ft: kb - elev,
                source: TopSource::User,
            });
        }
        wells.push(SynthWell {
            header: WellHeader {
                id,
                name: place.name,
                x_m,
                y_m,
                kb_elev_ft: kb,
            },
            geometry,
            logs: well_logs,
            boundaries_elev_ft: boundaries,
            layer_time,
        });
    }
    let tops = drop_tops(cfg, &tops_truth);
    Ok(SynthProject {
        top_set,
        wells,
        tops_truth,
        tops,
    })
}

fn invalid(field: &str, detail: &str) -> SxError {
    SxError::InvalidInput(format!("{field} {detail}"))
}

fn require_finite(field: &str, value: f64) -> SxResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid(field, "is not finite"))
    }
}

fn require_positive(field: &str, value: f64) -> SxResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(field, "must be finite and > 0"))
    }
}

fn require_nonnegative(field: &str, value: f64) -> SxResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid(field, "must be finite and >= 0"))
    }
}

fn validate(cfg: &SynthConfig) -> SxResult<()> {
    if cfg.n_wells == 0 {
        return Err(invalid("n_wells", "must be > 0"));
    }
    if cfg.n_tops == 0 || cfg.n_tops > 999 {
        return Err(invalid("n_tops", "must be in 1..=999"));
    }
    if cfg.logs.is_empty() {
        return Err(invalid("logs", "must be non-empty"));
    }
    let mut seen = BTreeSet::new();
    for log in &cfg.logs {
        if !seen.insert(*log) {
            return Err(invalid("logs", "has a repeat"));
        }
    }

    let positive = [
        ("spacing_m", cfg.spacing_m),
        ("thickness_ft", cfg.surface.thickness_ft),
        ("wavelength1_m", cfg.surface.wavelength1_m),
        ("wavelength2_m", cfg.surface.wavelength2_m),
        ("pad_above_ft", cfg.surface.pad_above_ft),
        ("pad_below_ft", cfg.surface.pad_below_ft),
    ];
    for (field, value) in positive {
        require_positive(field, value)?;
    }
    let nonnegative = [
        ("jitter_m", cfg.jitter_m),
        ("kb_elev_jitter_ft", cfg.kb_elev_jitter_ft),
        ("amp1_ft", cfg.surface.amp1_ft),
        ("amp2_ft", cfg.surface.amp2_ft),
    ];
    for (field, value) in nonnegative {
        require_nonnegative(field, value)?;
    }
    require_finite("top0_elev_ft", cfg.surface.top0_elev_ft)?;
    require_finite("dip_x_ft_per_m", cfg.surface.dip_x_ft_per_m)?;
    require_finite("dip_y_ft_per_m", cfg.surface.dip_y_ft_per_m)?;
    require_finite("kb_elev_mean_ft", cfg.kb_elev_mean_ft)?;
    if let Some(pinch) = &cfg.pinch {
        require_finite("pinch.center_x_m", pinch.center_x_m)?;
        require_finite("pinch.center_y_m", pinch.center_y_m)?;
        require_finite("pinch.radius_m", pinch.radius_m)?;
    }
    if !cfg.drop_top_frac.is_finite() || !(0.0..=1.0).contains(&cfg.drop_top_frac) {
        return Err(invalid("drop_top_frac", "must be in [0, 1]"));
    }
    if let Some(pinch) = &cfg.pinch {
        let interval = usize::from(pinch.interval);
        let max_interval = cfg.n_tops - 1;
        if !(1..=max_interval).contains(&interval) {
            return Err(invalid("pinch.interval", "must be in 1..=n_tops-1"));
        }
        if pinch.radius_m < 0.0 {
            return Err(invalid("pinch.radius_m", "must be >= 0"));
        }
    }
    if let Some(gaps) = &cfg.gaps {
        if gaps.runs_min > gaps.runs_max {
            return Err(invalid("gaps.runs_min", "must be <= gaps.runs_max"));
        }
        if gaps.len_min == 0 {
            return Err(invalid("gaps.len_min", "must be > 0"));
        }
        if gaps.len_min > gaps.len_max {
            return Err(invalid("gaps.len_min", "must be <= gaps.len_max"));
        }
    }
    Ok(())
}

struct PlacedWell {
    id: WellId,
    name: String,
    x_m: f64,
    y_m: f64,
    kb_elev_ft: f64,
}

fn grid_side(n_wells: usize) -> usize {
    let mut side = 0usize;
    while side.saturating_mul(side) < n_wells {
        side = side.saturating_add(1);
    }
    side
}

fn place_wells(cfg: &SynthConfig) -> SxResult<(usize, Vec<PlacedWell>)> {
    let side = grid_side(cfg.n_wells);
    if side == 0 {
        return Err(invalid("n_wells", "must be > 0"));
    }
    let mut rng = rng_for(cfg.seed, "synth_wells", 0);
    let mut wells = Vec::with_capacity(cfg.n_wells);
    for i in 0..cfg.n_wells {
        let jx = uniform_range(&mut rng, -cfg.jitter_m, cfg.jitter_m);
        let jy = uniform_range(&mut rng, -cfg.jitter_m, cfg.jitter_m);
        let kb_lo = cfg.kb_elev_mean_ft - cfg.kb_elev_jitter_ft;
        let kb_hi = cfg.kb_elev_mean_ft + cfg.kb_elev_jitter_ft;
        let kb = uniform_range(&mut rng, kb_lo, kb_hi);
        let col = i % side;
        let row = i / side;
        let id_num = i32::try_from(i).map_err(|_| invalid("n_wells", "does not fit in i32"))?;
        wells.push(PlacedWell {
            id: WellId(id_num),
            name: format!("SYN{i:04}"),
            x_m: col as f64 * cfg.spacing_m + jx,
            y_m: row as f64 * cfg.spacing_m + jy,
            kb_elev_ft: kb,
        });
    }
    Ok((side, wells))
}

fn top_names(n_tops: usize) -> Vec<String> {
    (0..n_tops)
        .map(|j| {
            let label = j + 1;
            format!("T{label:02}")
        })
        .collect()
}

fn sorted_logs(logs: &[LogKind]) -> Vec<LogKind> {
    let mut out = logs.to_vec();
    out.sort();
    out
}

/// Top elevations `S_0..S_{n-1}` at `(x, y)`, after the monotone clamp and pinch.
fn surface_elevations(cfg: &SynthConfig, side: usize, x: f64, y: f64) -> Vec<f64> {
    let surf = &cfg.surface;
    let xc = (side.saturating_sub(1) as f64) * cfg.spacing_m / 2.0;
    let yc = xc;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut surfaces = Vec::with_capacity(cfg.n_tops);
    for j in 0..cfg.n_tops {
        let jf = j as f64;
        let wave_x = two_pi * x / surf.wavelength1_m + 0.7 * jf;
        let wave_y = two_pi * y / surf.wavelength2_m + 1.3 * jf;
        let elev = surf.top0_elev_ft - jf * surf.thickness_ft
            + surf.dip_x_ft_per_m * (x - xc)
            + surf.dip_y_ft_per_m * (y - yc)
            + surf.amp1_ft * wave_x.sin()
            + surf.amp2_ft * wave_y.sin();
        surfaces.push(elev);
    }
    enforce_monotone(&mut surfaces);
    if let Some(pinch) = &cfg.pinch {
        apply_pinch(&mut surfaces, pinch, x, y);
    }
    surfaces
}

fn enforce_monotone(surfaces: &mut [f64]) {
    let mut prev = None;
    for elev in surfaces.iter_mut() {
        if let Some(above) = prev {
            *elev = elev.min(above);
        }
        prev = Some(*elev);
    }
}

fn apply_pinch(surfaces: &mut [f64], pinch: &PinchConfig, x: f64, y: f64) {
    let t = usize::from(pinch.interval);
    if t == 0 || t >= surfaces.len() {
        return;
    }
    let dist = (x - pinch.center_x_m).hypot(y - pinch.center_y_m);
    if dist > pinch.radius_m {
        return;
    }
    let deeper = surfaces[t];
    surfaces[t - 1] = deeper;
}

fn boundary_elevations(surfaces: &[f64], pad_above_ft: f64, pad_below_ft: f64) -> Vec<f64> {
    let (Some(&top), Some(&bottom)) = (surfaces.first(), surfaces.last()) else {
        return Vec::new();
    };
    let mut boundaries = Vec::with_capacity(surfaces.len().saturating_add(2));
    boundaries.push(top + pad_above_ft);
    boundaries.extend_from_slice(surfaces);
    boundaries.push(bottom - pad_below_ft);
    boundaries
}

fn grid_for_well(kb_elev_ft: f64, boundaries: &[f64]) -> SxResult<GridRange> {
    let (Some(&b0), Some(&b_last)) = (boundaries.first(), boundaries.last()) else {
        return Err(invalid("n_tops", "must be > 0"));
    };
    let md_top = kb_elev_ft - b0;
    let md_base = kb_elev_ft - b_last;
    let k_start = sat_i32((md_top / GRID_STEP_FT).ceil())
        .ok_or_else(|| invalid("grid", "k_start is not finite"))?;
    let k_end = sat_i32((md_base / GRID_STEP_FT).floor())
        .ok_or_else(|| invalid("grid", "k_end is not finite"))?;
    let len_i64 = i64::from(k_end) - i64::from(k_start) + 1;
    if len_i64 < 0 {
        return Err(invalid("grid", "len is negative"));
    }
    let len = usize::try_from(len_i64).map_err(|_| invalid("grid", "len does not fit"))?;
    GridRange::new(k_start, len)
}

fn sat_i32(value: f64) -> Option<i32> {
    if !value.is_finite() {
        return None;
    }
    if value >= f64::from(i32::MAX) {
        Some(i32::MAX)
    } else if value <= f64::from(i32::MIN) {
        Some(i32::MIN)
    } else {
        Some(value as i32)
    }
}

fn build_geometry(
    id: WellId,
    grid: GridRange,
    x_m: f64,
    y_m: f64,
    kb_elev_ft: f64,
) -> WellGeometry {
    let n = grid.len;
    let mut tvd_ft = Vec::with_capacity(n);
    let mut elev_ft = Vec::with_capacity(n);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let md = grid.md_ft(i);
        tvd_ft.push(md);
        elev_ft.push(kb_elev_ft - md);
        xs.push(x_m);
        ys.push(y_m);
    }
    WellGeometry {
        id,
        grid,
        tvd_ft,
        elev_ft,
        x_m: xs,
        y_m: ys,
    }
}

/// Layer time at elevation `e` from boundary elevations `B_0..=B_{n+1}`.
///
/// `i` is the largest index in `0..=n` with `B_i >= e`, and the fraction is
/// `(B_i - e) / (B_i - B_{i+1})`. A sample exactly on a top gets fraction 0
/// in the deeper interval. A zero denominator (pinched boundary with no
/// deeper sample) contributes fraction 0.
fn layer_time_at_elev(boundaries: &[f64], e: f64) -> f64 {
    let n = boundaries.len().saturating_sub(2);
    let mut chosen = 0usize;
    for idx in (0..=n).rev() {
        if boundaries.get(idx).is_some_and(|bound| *bound >= e) {
            chosen = idx;
            break;
        }
    }
    let upper = boundaries.get(chosen).copied().unwrap_or(e);
    let lower = boundaries
        .get(chosen.saturating_add(1))
        .copied()
        .unwrap_or(upper);
    let denom = upper - lower;
    let frac = if denom == 0.0 {
        0.0
    } else {
        (upper - e) / denom
    };
    chosen as f64 + frac
}

#[derive(Clone, Copy)]
struct CurveLaw {
    lo: f64,
    hi: f64,
    amp: f64,
    sign: f64,
    phi: f64,
    sigma: f64,
}

fn curve_law(log: LogKind) -> CurveLaw {
    match log {
        LogKind::Gr => CurveLaw {
            lo: 40.0,
            hi: 130.0,
            amp: 15.0,
            sign: 1.0,
            phi: 0.80,
            sigma: 4.0,
        },
        LogKind::Rhob => CurveLaw {
            lo: 2.20,
            hi: 2.65,
            amp: 0.05,
            sign: -1.0,
            phi: 0.70,
            sigma: 0.015,
        },
        LogKind::Nphi => CurveLaw {
            lo: 0.05,
            hi: 0.35,
            amp: 0.03,
            sign: 1.0,
            phi: 0.70,
            sigma: 0.010,
        },
        LogKind::Dtc => CurveLaw {
            lo: 55.0,
            hi: 100.0,
            amp: 5.0,
            sign: 1.0,
            phi: 0.80,
            sigma: 1.5,
        },
        LogKind::Resd => CurveLaw {
            lo: 0.0,
            hi: 2.0,
            amp: 0.20,
            sign: -1.0,
            phi: 0.85,
            sigma: 0.05,
        },
        LogKind::Pef => CurveLaw {
            lo: 2.0,
            hi: 5.0,
            amp: 0.30,
            sign: 1.0,
            phi: 0.60,
            sigma: 0.10,
        },
    }
}

fn draw_layer_means(cfg: &SynthConfig) -> Vec<[f64; 6]> {
    let n_intervals = cfg.n_tops.saturating_add(1);
    let mut means = vec![[0.0; 6]; n_intervals];
    let mut rng = rng_for(cfg.seed, "synth_layers", 0);
    for row in &mut means {
        for log in LogKind::ALL {
            let law = curve_law(log);
            let draw = uniform_range(&mut rng, law.lo, law.hi);
            if let Some(slot) = row.get_mut(log as usize) {
                *slot = draw;
            }
        }
    }
    means
}

fn interval_of(tau: f64, n_tops: usize) -> usize {
    if !tau.is_finite() {
        return 0;
    }
    let floored = tau.floor();
    if floored <= 0.0 {
        return 0;
    }
    let n = n_tops as f64;
    if floored >= n {
        return n_tops;
    }
    floored as usize
}

fn thin_bed(tau: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    0.7 * (two_pi * 3.0 * tau).sin() + 0.3 * (two_pi * 7.0 * tau + 1.0).sin()
}

fn mean_of(means: &[[f64; 6]], interval: usize, log: LogKind) -> f64 {
    means
        .get(interval)
        .and_then(|row| row.get(log as usize))
        .copied()
        .unwrap_or(0.0)
}

fn sample_logs(
    cfg: &SynthConfig,
    id: WellId,
    grid: GridRange,
    layer_time: &[f64],
    means: &[[f64; 6]],
    logs: &[LogKind],
) -> WellLogs {
    let mut out = WellLogs::new(id, grid);
    let n = grid.len;
    for &log in logs {
        let law = curve_law(log);
        let innov = law.sigma * (1.0 - law.phi * law.phi).sqrt();
        let log_name = log.name();
        let stage = format!("synth_noise_{log_name}");
        let mut rng = rng_for(cfg.seed, &stage, id.0);
        let mut values = Vec::with_capacity(n);
        let mut prov = Vec::with_capacity(n);
        let mut noise = 0.0;
        for s in 0..n {
            let z = standard_normal(&mut rng);
            if s == 0 {
                noise = law.sigma * z;
            } else {
                noise = law.phi * noise + innov * z;
            }
            let tau = layer_time.get(s).copied().unwrap_or(0.0);
            let interval = interval_of(tau, cfg.n_tops);
            let w = mean_of(means, interval, log) + law.amp * law.sign * thin_bed(tau) + noise;
            let stored = if log == LogKind::Resd {
                10.0_f64.powf(w) as f32
            } else {
                w as f32
            };
            let code = if stored.is_finite() {
                Provenance::Measured.code()
            } else {
                CODE_NONE
            };
            values.push(stored);
            prov.push(code);
        }
        out.curves.insert(log, values);
        out.provenance.insert(log, prov);
    }
    out
}

fn apply_gaps(seed: u64, well_id: i32, gaps: &GapConfig, logs: &[LogKind], well: &mut WellLogs) {
    let len = u64::try_from(well.grid.len).unwrap_or(u64::MAX);
    let mut rng = rng_for(seed, "synth_gaps", well_id);
    for log in logs {
        let runs = uniform_int_inclusive(&mut rng, gaps.runs_min, gaps.runs_max);
        for _ in 0..runs {
            let drawn = uniform_int_inclusive(&mut rng, gaps.len_min, gaps.len_max);
            let run_len = drawn.min(len);
            let start = uniform_below(&mut rng, len - run_len + 1);
            paint_gap(well, log, start, run_len);
        }
    }
}

fn paint_gap(well: &mut WellLogs, log: &LogKind, start: u64, run_len: u64) {
    let Some(start_i) = usize::try_from(start).ok() else {
        return;
    };
    let Ok(run_i) = usize::try_from(run_len) else {
        return;
    };
    if let Some(values) = well.curves.get_mut(log) {
        for value in values.iter_mut().skip(start_i).take(run_i) {
            *value = f32::NAN;
        }
    }
    if let Some(prov) = well.provenance.get_mut(log) {
        for code in prov.iter_mut().skip(start_i).take(run_i) {
            *code = CODE_NONE;
        }
    }
}

fn drop_count(frac: f64, n: usize) -> usize {
    let rounded = (frac * n as f64).round();
    if !rounded.is_finite() || rounded <= 0.0 {
        return 0;
    }
    if rounded >= n as f64 {
        return n;
    }
    rounded as usize
}

fn drop_tops(cfg: &SynthConfig, tops_truth: &[TopPick]) -> Vec<TopPick> {
    let n = tops_truth.len();
    let n_drop = drop_count(cfg.drop_top_frac, n);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rng_for(cfg.seed, "synth_tops", 0);
    shuffle(&mut rng, &mut idx);
    let mut dropped = vec![false; n];
    for &i in idx.iter().take(n_drop) {
        if let Some(flag) = dropped.get_mut(i) {
            *flag = true;
        }
    }
    tops_truth
        .iter()
        .enumerate()
        .filter(|(i, _)| !dropped.get(*i).copied().unwrap_or(false))
        .map(|(_, pick)| *pick)
        .collect()
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(buf: &mut Vec<u8>, value: f64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f64s(buf: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        push_f64(buf, *value);
    }
}

fn push_str(buf: &mut Vec<u8>, text: &str) {
    push_u64(buf, text.len() as u64);
    buf.extend_from_slice(text.as_bytes());
}

fn push_picks(buf: &mut Vec<u8>, picks: &[TopPick]) {
    push_u64(buf, picks.len() as u64);
    for pick in picks {
        push_i32(buf, pick.well.0);
        buf.extend_from_slice(&pick.top.to_le_bytes());
        push_f64(buf, pick.md_ft);
        buf.push(pick.source as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_within(label: &str, got: f64, expect: f64, tol: f64) {
        let diff = (got - expect).abs();
        assert!(
            diff <= tol,
            "{label}: got {got:.16} expected {expect:.16} diff {diff:.3e}"
        );
    }

    fn featured_cfg() -> SynthConfig {
        SynthConfig {
            gaps: Some(GapConfig::default()),
            drop_top_frac: 0.2,
            pinch: Some(PinchConfig {
                interval: 2,
                center_x_m: 2000.0,
                center_y_m: 2000.0,
                radius_m: 1200.0,
            }),
            ..SynthConfig::default()
        }
    }

    fn md_of(picks: &[TopPick], well: WellId, top: u16) -> f64 {
        picks
            .iter()
            .find(|pick| pick.well == well && pick.top == top)
            .map(|pick| pick.md_ft)
            .unwrap_or_else(|| panic!("missing top {top} for well {}", well.0))
    }

    #[test]
    fn determinism_same_seed() {
        let cfg = featured_cfg();
        let a = generate(&cfg).unwrap().to_bytes();
        let b = generate(&cfg).unwrap().to_bytes();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_differs() {
        let cfg0 = SynthConfig::default();
        let cfg1 = SynthConfig {
            seed: 1,
            ..SynthConfig::default()
        };
        let a = generate(&cfg0).unwrap().to_bytes();
        let b = generate(&cfg1).unwrap().to_bytes();
        assert_ne!(a, b);
    }

    #[test]
    fn well_zero_and_one_reference() {
        let project = generate(&SynthConfig::default()).unwrap();
        let w0 = &project.wells[0].header;
        let w1 = &project.wells[1].header;
        assert_within("well0 x_m", w0.x_m, -174.11954157491292, 1e-9);
        assert_within("well0 y_m", w0.y_m, -36.358159978080636, 1e-9);
        assert_within("well0 kb", w0.kb_elev_ft, 3043.11375438437, 1e-9);
        assert_within("well1 x_m", w1.x_m, 961.1741043910825, 1e-9);
        assert_within("well1 y_m", w1.y_m, 194.0135175967369, 1e-9);
        assert_within("well1 kb", w1.kb_elev_ft, 3033.4560467937185, 1e-9);
    }

    #[test]
    fn surface_reference() {
        let cfg = SynthConfig::default();
        let got = surface_elevations(&cfg, 5, 1000.0, 2000.0);
        let expect = [
            1998.904971,
            1945.061563,
            1886.270449,
            1817.333307,
            1744.922114,
            1680.824560,
        ];
        assert_eq!(got.len(), expect.len());
        for (i, (got_s, expect_s)) in got.iter().zip(expect).enumerate() {
            assert_within(&format!("S{i}"), *got_s, expect_s, 1e-6);
        }
    }

    #[test]
    fn layer_time_rules() {
        let project = generate(&SynthConfig::default()).unwrap();
        let n_tops = project.top_set.len();
        for well in &project.wells {
            assert_eq!(well.layer_time.len(), well.geometry.grid.len);
            assert!(!well.layer_time.is_empty());
            for pair in well.layer_time.windows(2) {
                assert!(
                    pair[1] >= pair[0],
                    "well {} layer time decreased",
                    well.header.id.0
                );
            }
            let first = well.layer_time[0];
            let last = well.layer_time[well.layer_time.len() - 1];
            assert!(
                (0.0..1.0).contains(&first),
                "well {} first layer time {first}",
                well.header.id.0
            );
            assert!(
                last <= n_tops as f64 + 1.0,
                "well {} last layer time {last}",
                well.header.id.0
            );
            for pick in project
                .tops_truth
                .iter()
                .filter(|pick| pick.well == well.header.id)
            {
                let elev = well.header.kb_elev_ft - pick.md_ft;
                let tau = layer_time_at_elev(&well.boundaries_elev_ft, elev);
                assert_within(
                    &format!("well {} top {}", well.header.id.0, pick.top),
                    tau,
                    f64::from(pick.top) + 1.0,
                    1e-9,
                );
            }
        }

        let surfaces = [
            1998.904971,
            1945.061563,
            1886.270449,
            1817.333307,
            1744.922114,
            1680.824560,
        ];
        let boundaries = boundary_elevations(&surfaces, 100.0, 100.0);
        let kb = 3000.0;
        assert_within(
            "k 1803",
            layer_time_at_elev(&boundaries, kb - 901.5),
            0.004049714563689113,
            1e-6,
        );
        assert_within(
            "k 2838",
            layer_time_at_elev(&boundaries, kb - 1419.0),
            6.998245595890105,
            1e-6,
        );
        assert_within(
            "S0-30",
            layer_time_at_elev(&boundaries, surfaces[0] - 30.0),
            1.5571712631958081,
            1e-6,
        );
    }

    #[test]
    fn pinch_zero_thickness() {
        let cfg = SynthConfig {
            pinch: Some(PinchConfig {
                interval: 2,
                center_x_m: 2000.0,
                center_y_m: 2000.0,
                radius_m: 1200.0,
            }),
            ..SynthConfig::default()
        };
        let project = generate(&cfg).unwrap();
        let mut inside = 0usize;
        for well in &project.wells {
            let dist = (well.header.x_m - 2000.0).hypot(well.header.y_m - 2000.0);
            if dist > 1200.0 {
                continue;
            }
            inside += 1;
            let md1 = md_of(&project.tops_truth, well.header.id, 1);
            let md2 = md_of(&project.tops_truth, well.header.id, 2);
            assert_eq!(md1, md2, "well {} pinch tops", well.header.id.0);
            for &tau in &well.layer_time {
                assert_ne!(
                    tau.floor(),
                    2.0,
                    "well {} interval 2 has a sample",
                    well.header.id.0
                );
            }
        }
        assert!(inside > 0, "expected a well inside the pinch radius");
    }

    #[test]
    fn drop_count() {
        let cfg = SynthConfig {
            drop_top_frac: 0.2,
            ..SynthConfig::default()
        };
        let project = generate(&cfg).unwrap();
        assert_eq!(project.tops_truth.len(), 150);
        assert_eq!(project.tops.len(), 120);
    }

    #[test]
    fn gaps_provenance() {
        let cfg = SynthConfig {
            gaps: Some(GapConfig::default()),
            ..SynthConfig::default()
        };
        let project = generate(&cfg).unwrap();
        let mut nans = 0usize;
        for well in &project.wells {
            for log in well.logs.curves.keys() {
                let values = &well.logs.curves[log];
                let prov = &well.logs.provenance[log];
                assert_eq!(values.len(), prov.len());
                for (&value, &code) in values.iter().zip(prov.iter()) {
                    if value.is_nan() {
                        assert_eq!(code, 255);
                        nans += 1;
                    } else if value.is_finite() {
                        assert_eq!(code, 0);
                    }
                }
            }
        }
        assert!(nans > 0, "gaps produced no NaN");
    }

    #[test]
    fn values_in_bounds() {
        let project = generate(&SynthConfig::default()).unwrap();
        for well in &project.wells {
            for (log, values) in &well.logs.curves {
                let (lo, hi) = physical_bounds(*log);
                for &value in values {
                    assert!(
                        value.is_finite() && value >= lo && value <= hi,
                        "well {} {} value {value} outside {lo}..={hi}",
                        well.header.id.0,
                        log.name()
                    );
                }
            }
        }
    }

    fn physical_bounds(log: LogKind) -> (f32, f32) {
        match log {
            LogKind::Gr => (0.0, 500.0),
            LogKind::Rhob => (1.0, 3.5),
            LogKind::Nphi => (-0.15, 1.0),
            LogKind::Dtc => (30.0, 250.0),
            LogKind::Resd => (0.01, 50_000.0),
            LogKind::Pef => (0.5, 10.0),
        }
    }

    #[test]
    fn grid_on_global_steps() {
        let project = generate(&SynthConfig::default()).unwrap();
        for well in &project.wells {
            assert_eq!(well.geometry.tvd_ft.len(), well.geometry.grid.len);
            for (i, &tvd) in well.geometry.tvd_ft.iter().enumerate() {
                let k = i64::from(well.geometry.grid.k_start) + i64::try_from(i).unwrap();
                assert_eq!(tvd, 0.5 * k as f64);
            }
        }
    }
}
