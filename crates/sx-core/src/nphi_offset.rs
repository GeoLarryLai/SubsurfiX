// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Field-wide NPHI offset against limestone density porosity.
//!
//! The difference d = NPHI - DPHI_ls, DPHI_ls = (rho_matrix - RHOB) / (rho_matrix - rho_fluid),
//! is taken on measured logs after QC (bad hole masked), before normalization and fill.
//! Porosity axis: PHI_ND = (NPHI + DPHI_ls) / 2, so noise in either log does not bias d
//! through the selection or the fit. Offset: clean (GR <= `gr_max`), tight
//! (PHI_ND <= `phi_max`) limestone (PEF in
//! [`pef_min`, `pef_max`] where measured). Field offset = median of the well medians,
//! with a bootstrap interval over wells. Trend: Theil-Sen fit of d against PHI_ND on
//! per-well, per-porosity-bin medians of clean limestone up to `fit_phi_max`; a constant
//! offset has a slope near 0, a unit or scale mismatch does not.
//! The offset is applicable when enough wells qualify, its interval excludes 0 and
//! |slope| <= `slope_tol`. Normalize then subtracts it from NPHI (`[nphi_offset] apply`);
//! a nonzero `[normalize] nphi_shift` overrides it.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{NphiOffsetConfig, ProjectConfig};
use crate::error::{SxError, SxResult};
use crate::types::LogKind;
use crate::util::rng::{rng_for, uniform_f64, ChaCha8Rng};

/// Stored result, relative to the project folder.
pub const RESULT_JSON: &str = "stages/nphi_offset/result.json";
/// Per-well table.
pub const WELLS_CSV: &str = "stages/nphi_offset/wells.csv";
/// Subsample of clean-limestone samples for plots.
pub const SAMPLES_CSV: &str = "stages/nphi_offset/samples.csv";
/// Schema tag of [`RESULT_JSON`].
pub const SCHEMA: &str = "subsurfix.nphi_offset/1";

/// P10 / P50 / P90 of d (v/v).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pct {
    /// 10th percentile
    pub p10: f64,
    /// median
    pub p50: f64,
    /// 90th percentile
    pub p90: f64,
}

/// One well.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WellOffset {
    /// well id
    pub well_id: i32,
    /// qualifying clean, tight limestone samples
    pub n: usize,
    /// percentiles of d; None without samples
    pub pct: Option<Pct>,
    /// enough samples to count in the field estimate
    pub counted: bool,
    /// counted and |P50 - field offset| > `outlier_limit`
    pub outlier: bool,
}

/// Theil-Sen fit of d against an explanatory log.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Trend {
    /// d at x = 0
    pub intercept: f64,
    /// change of d per unit x
    pub slope: f64,
    /// bootstrap interval of the slope (wells resampled); None when not computed
    pub slope_ci: Option<(f64, f64)>,
    /// binned medians used
    pub n_points: usize,
}

/// Evaluation result, stored as [`RESULT_JSON`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NphiOffsetResult {
    /// [`SCHEMA`]
    pub schema: String,
    /// data used
    pub input: String,
    /// `ok`, `too_few_wells`, `ci_includes_zero` or `not_constant`
    pub status: String,
    /// shift to subtract from NPHI when applied (field offset if `ok`, else 0)
    pub shift: f64,
    /// field offset: median of the counted well medians; None without counted wells
    pub field_offset: Option<f64>,
    /// bootstrap interval of the field offset
    pub ci: Option<(f64, f64)>,
    /// interval level
    pub ci_level: f64,
    /// percentiles of d over all counted samples
    pub field_pct: Option<Pct>,
    /// counted samples
    pub n_samples: usize,
    /// counted wells
    pub n_wells: usize,
    /// d against PHI_ND (clean limestone up to `fit_phi_max`)
    pub trend_phi: Option<Trend>,
    /// d against GR per API (tight limestone, GR up to 150)
    pub trend_gr: Option<Trend>,
    /// d against PEF per b/e (clean, tight rock)
    pub trend_pef: Option<Trend>,
    /// per well
    pub wells: Vec<WellOffset>,
}

impl NphiOffsetResult {
    /// JSON text of the result.
    ///
    /// # Errors
    /// `SxError::Format` when serialization fails.
    pub fn to_json(&self) -> SxResult<String> {
        serde_json::to_string(self).map_err(|e| SxError::Format(e.to_string()))
    }

    /// True when the evaluation supports a constant offset.
    pub fn applicable(&self) -> bool {
        self.status == "ok"
    }
}

/// Input arrays, one entry per sample; NaN = missing.
#[derive(Debug, Clone, Copy)]
pub struct NphiOffsetInput<'a> {
    /// well id
    pub well: &'a [i32],
    /// GR (API)
    pub gr: &'a [f64],
    /// RHOB (g/cm³)
    pub rhob: &'a [f64],
    /// NPHI, limestone units (v/v)
    pub nphi: &'a [f64],
    /// PEF (b/e)
    pub pef: &'a [f64],
}

fn pct_sorted(v: &[f64], q: f64) -> f64 {
    let x = q * (v.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (x - lo as f64)
}

fn pct(v: &mut [f64]) -> Option<Pct> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(Pct {
        p10: pct_sorted(v, 0.1),
        p50: pct_sorted(v, 0.5),
        p90: pct_sorted(v, 0.9),
    })
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    pct_sorted(v, 0.5)
}

/// Theil-Sen slope and intercept; None with fewer than 3 points or one x.
pub fn theil_sen(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let mut s = Vec::with_capacity(x.len() * x.len() / 2);
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let dx = x[j] - x[i];
            if dx.abs() > 1e-12 {
                s.push((y[j] - y[i]) / dx);
            }
        }
    }
    if x.len() < 3 || s.is_empty() {
        return None;
    }
    let b = median(&mut s);
    let mut r: Vec<f64> = x.iter().zip(y).map(|(xi, yi)| yi - b * xi).collect();
    Some((median(&mut r), b))
}

/// Per-well, per-bin medians of d: well -> [(bin centre, median)].
type Binned = BTreeMap<i32, Vec<(f64, f64)>>;

fn binned(rows: &[(i32, f64, f64)], width: f64, min_n: usize) -> Binned {
    let mut cells: BTreeMap<(i32, i64), Vec<f64>> = BTreeMap::new();
    for &(w, x, d) in rows {
        cells
            .entry((w, (x / width).floor() as i64))
            .or_default()
            .push(d);
    }
    let mut out = Binned::new();
    for ((w, b), mut v) in cells {
        if v.len() >= min_n {
            out.entry(w)
                .or_default()
                .push(((b as f64 + 0.5) * width, median(&mut v)));
        }
    }
    out
}

fn fit(b: &Binned, wells: &[i32]) -> Option<(f64, f64, usize)> {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for w in wells {
        for &(xi, yi) in b.get(w).map_or(&[][..], |v| &v[..]) {
            x.push(xi);
            y.push(yi);
        }
    }
    theil_sen(&x, &y).map(|(a, s)| (a, s, x.len()))
}

fn resample(rng: &mut ChaCha8Rng, keys: &[i32]) -> Vec<i32> {
    (0..keys.len())
        .map(|_| keys[((uniform_f64(rng) * keys.len() as f64) as usize).min(keys.len() - 1)])
        .collect()
}

fn trend(b: &Binned, n_boot: usize, level: f64, rng: &mut ChaCha8Rng) -> Option<Trend> {
    let keys: Vec<i32> = b.keys().copied().collect();
    let (intercept, slope, n_points) = fit(b, &keys)?;
    let mut s: Vec<f64> = (0..n_boot)
        .filter_map(|_| fit(b, &resample(rng, &keys)).map(|f| f.1))
        .collect();
    let slope_ci = (s.len() >= 10).then(|| {
        s.sort_by(f64::total_cmp);
        (
            pct_sorted(&s, (1.0 - level) / 2.0),
            pct_sorted(&s, (1.0 + level) / 2.0),
        )
    });
    Some(Trend {
        intercept,
        slope,
        slope_ci,
        n_points,
    })
}

/// Evaluate the field-wide offset. `seed` drives the bootstraps.
///
/// # Errors
/// `SxError::Shape` when the arrays differ in length.
pub fn evaluate(
    inp: &NphiOffsetInput<'_>,
    c: &NphiOffsetConfig,
    seed: u64,
) -> SxResult<NphiOffsetResult> {
    let n = inp.well.len();
    if [inp.gr.len(), inp.rhob.len(), inp.nphi.len(), inp.pef.len()]
        .iter()
        .any(|&l| l != n)
    {
        return Err(SxError::Shape(
            "nphi_offset arrays must have equal length".into(),
        ));
    }
    let span = c.rho_matrix - c.rho_fluid;
    let pef_ok = |p: f64| {
        if p.is_finite() {
            p >= c.pef_min && p <= c.pef_max
        } else {
            !c.require_pef
        }
    };
    let mut tight: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
    let (mut by_phi, mut by_gr, mut by_pef) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (g, r, x, p) = (inp.gr[i], inp.rhob[i], inp.nphi[i], inp.pef[i]);
        if !(r.is_finite() && x.is_finite()) {
            continue;
        }
        let dphi = (c.rho_matrix - r) / span;
        let d = x - dphi;
        // Porosity axis: neutron-density mean, so log noise in NPHI or RHOB does not
        // leak into d through the selection or the fit (as it would with DPHI alone).
        let phi = 0.5 * (x + dphi);
        let w = inp.well[i];
        let clean = g.is_finite() && g <= c.gr_max;
        let is_tight = phi <= c.phi_max;
        if clean && is_tight && pef_ok(p) {
            tight.entry(w).or_default().push(d);
        }
        if clean && pef_ok(p) && phi <= c.fit_phi_max {
            by_phi.push((w, phi, d));
        }
        if is_tight && pef_ok(p) && g.is_finite() && g <= 150.0 {
            by_gr.push((w, g, d));
        }
        if clean && is_tight && p.is_finite() {
            by_pef.push((w, p, d));
        }
    }
    let mut wells: Vec<WellOffset> = tight
        .iter()
        .map(|(&w, v)| WellOffset {
            well_id: w,
            n: v.len(),
            pct: pct(&mut v.clone()),
            counted: v.len() >= c.min_well_samples,
            outlier: false,
        })
        .collect();
    let counted: Vec<(i32, f64)> = wells
        .iter()
        .filter(|w| w.counted)
        .filter_map(|w| w.pct.map(|p| (w.well_id, p.p50)))
        .collect();
    let mut all: Vec<f64> = counted
        .iter()
        .flat_map(|(w, _)| tight[w].iter().copied())
        .collect();
    let n_samples = all.len();
    let field_pct = pct(&mut all);
    let mut rng = rng_for(seed, "nphi_offset", 0);
    let (field_offset, ci) = if counted.is_empty() {
        (None, None)
    } else {
        let mut m: Vec<f64> = counted.iter().map(|x| x.1).collect();
        let f = median(&mut m);
        let mut boot: Vec<f64> = (0..c.n_boot)
            .map(|_| {
                let mut s: Vec<f64> = (0..m.len())
                    .map(|_| {
                        m[((uniform_f64(&mut rng) * m.len() as f64) as usize).min(m.len() - 1)]
                    })
                    .collect();
                median(&mut s)
            })
            .collect();
        boot.sort_by(f64::total_cmp);
        (
            Some(f),
            Some((
                pct_sorted(&boot, (1.0 - c.ci_level) / 2.0),
                pct_sorted(&boot, (1.0 + c.ci_level) / 2.0),
            )),
        )
    };
    if let Some(f) = field_offset {
        for w in wells.iter_mut().filter(|w| w.counted) {
            w.outlier = w.pct.is_some_and(|p| (p.p50 - f).abs() > c.outlier_limit);
        }
    }
    let trend_phi = trend(
        &binned(&by_phi, c.fit_bin, 5),
        c.n_boot_slope,
        c.ci_level,
        &mut rng,
    );
    let trend_gr = trend(&binned(&by_gr, 10.0, 5), 0, c.ci_level, &mut rng);
    let trend_pef = trend(&binned(&by_pef, 0.25, 5), 0, c.ci_level, &mut rng);
    let status = match (field_offset, ci) {
        _ if counted.len() < c.min_wells => "too_few_wells",
        (Some(_), Some((lo, hi))) if lo <= 0.0 && hi >= 0.0 => "ci_includes_zero",
        _ if !trend_phi.is_some_and(|t| t.slope.abs() <= c.slope_tol) => "not_constant",
        _ => "ok",
    };
    let shift = if status == "ok" {
        field_offset.unwrap_or(0.0)
    } else {
        0.0
    };
    if status != "ok" {
        tracing::warn!(
            status,
            "NPHI offset not applicable; NPHI is left as measured"
        );
    }
    Ok(NphiOffsetResult {
        schema: SCHEMA.into(),
        input: "measured logs after QC (bad hole masked), before normalization and fill".into(),
        status: status.into(),
        shift,
        field_offset,
        ci,
        ci_level: c.ci_level,
        field_pct,
        n_samples,
        n_wells: counted.len(),
        trend_phi,
        trend_gr,
        trend_pef,
        wells,
    })
}

fn io_err(path: &Path, source: std::io::Error) -> SxError {
    SxError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// Evaluate on `stages/qc/logs_qc.parquet` and write [`RESULT_JSON`], [`WELLS_CSV`]
/// and [`SAMPLES_CSV`].
///
/// # Errors
/// Missing or unreadable QC logs (run the pipeline through `qc` first), write errors.
pub fn evaluate_project(project: &Path, cfg: &ProjectConfig) -> SxResult<NphiOffsetResult> {
    let logs = crate::pipeline::readers::read_logs(
        &project.join(crate::pipeline::products::LOGS_QC),
        crate::qc::LOGS_QC_TAG,
    )?;
    let (mut well, mut gr, mut rhob, mut nphi, mut pef) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for l in &logs {
        let len = l.grid.len;
        let col = |k: LogKind| -> Vec<f64> {
            l.curves.get(&k).map_or(vec![f64::NAN; len], |v| {
                v.iter().map(|&x| f64::from(x)).collect()
            })
        };
        well.extend(std::iter::repeat_n(l.id.0, len));
        gr.extend(col(LogKind::Gr));
        rhob.extend(col(LogKind::Rhob));
        nphi.extend(col(LogKind::Nphi));
        pef.extend(col(LogKind::Pef));
    }
    let c = &cfg.nphi_offset;
    let inp = NphiOffsetInput {
        well: &well,
        gr: &gr,
        rhob: &rhob,
        nphi: &nphi,
        pef: &pef,
    };
    let r = evaluate(&inp, c, cfg.seed)?;
    let dir = project.join("stages/nphi_offset");
    std::fs::create_dir_all(&dir).map_err(|e| io_err(&dir, e))?;
    let json = serde_json::to_string_pretty(&r).map_err(|e| SxError::Format(e.to_string()))?;
    write(&project.join(RESULT_JSON), json)?;
    let mut csv = String::from("well_id,n,p10,p50,p90,counted,outlier\n");
    for w in &r.wells {
        let p = w.pct.map_or([f64::NAN; 3], |p| [p.p10, p.p50, p.p90]);
        csv += &format!(
            "{},{},{:.5},{:.5},{:.5},{},{}\n",
            w.well_id, w.n, p[0], p[1], p[2], w.counted, w.outlier
        );
    }
    write(&project.join(WELLS_CSV), csv)?;
    // Clean-limestone samples of the trend fit, evenly thinned.
    let span = c.rho_matrix - c.rho_fluid;
    let keep: Vec<usize> = (0..well.len())
        .filter(|&i| {
            let dphi = (c.rho_matrix - rhob[i]) / span;
            let p = pef[i];
            rhob[i].is_finite()
                && nphi[i].is_finite()
                && gr[i] <= c.gr_max
                && 0.5 * (dphi + nphi[i]) <= c.fit_phi_max
                && (if p.is_finite() {
                    p >= c.pef_min && p <= c.pef_max
                } else {
                    !c.require_pef
                })
        })
        .collect();
    let step = keep.len().div_ceil(c.max_plot_samples.max(1)).max(1);
    let mut s = String::from("well_id,dphi_ls,nphi,gr,pef\n");
    for &i in keep.iter().step_by(step) {
        s += &format!(
            "{},{:.5},{:.5},{:.2},{:.3}\n",
            well[i],
            (c.rho_matrix - rhob[i]) / span,
            nphi[i],
            gr[i],
            pef[i]
        );
    }
    write(&project.join(SAMPLES_CSV), s)?;
    Ok(r)
}

fn write(path: &Path, text: String) -> SxResult<()> {
    crate::io::atomic::write_atomic(path, |tmp| {
        std::fs::write(tmp, text.as_bytes()).map_err(|e| io_err(tmp, e))
    })
}

/// Stored result of the project, if evaluated.
///
/// # Errors
/// An unreadable or malformed [`RESULT_JSON`].
pub fn load(project: &Path) -> SxResult<Option<NphiOffsetResult>> {
    let p = project.join(RESULT_JSON);
    if !p.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p).map_err(|e| io_err(&p, e))?;
    let r: NphiOffsetResult =
        serde_json::from_str(&text).map_err(|e| SxError::Format(format!("{RESULT_JSON}: {e}")))?;
    if r.schema != SCHEMA {
        return Err(SxError::Format(format!(
            "{RESULT_JSON}: schema is not {SCHEMA}"
        )));
    }
    Ok(Some(r))
}

/// JSON text of a `[nphi_offset]` section.
///
/// # Errors
/// `SxError::Format` when serialization fails.
pub fn config_json(c: &NphiOffsetConfig) -> SxResult<String> {
    serde_json::to_string(c).map_err(|e| SxError::Format(e.to_string()))
}

/// Where the NPHI shift used by normalize comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftSource {
    /// `[normalize] nphi_shift` is nonzero
    Manual,
    /// stored evaluation, applicable, `[nphi_offset] apply = true`
    Evaluated,
    /// no shift
    None,
}

impl ShiftSource {
    /// `manual`, `evaluated` or `none`.
    pub fn name(self) -> &'static str {
        match self {
            ShiftSource::Manual => "manual",
            ShiftSource::Evaluated => "evaluated",
            ShiftSource::None => "none",
        }
    }
}

/// Shift to subtract from NPHI: manual if nonzero, else the evaluated offset when
/// `apply` and the evaluation is applicable, else 0.
pub fn resolve_shift(
    manual: f64,
    apply: bool,
    stored: Option<&NphiOffsetResult>,
) -> (f64, ShiftSource) {
    if manual != 0.0 {
        return (manual, ShiftSource::Manual);
    }
    match stored {
        Some(r) if apply && r.applicable() => (r.shift, ShiftSource::Evaluated),
        _ => (0.0, ShiftSource::None),
    }
}

/// `cfg` with `[normalize] nphi_shift` set to the resolved shift of `project`.
///
/// # Errors
/// Those of [`load`].
pub fn effective_config(project: &Path, cfg: &ProjectConfig) -> SxResult<ProjectConfig> {
    let stored = load(project)?;
    let (shift, source) = resolve_shift(
        cfg.normalize.nphi_shift,
        cfg.nphi_offset.apply,
        stored.as_ref(),
    );
    let mut out = cfg.clone();
    out.normalize.nphi_shift = shift;
    if source != ShiftSource::None {
        tracing::info!(shift, source = source.name(), "NPHI shift");
    }
    Ok(out)
}
