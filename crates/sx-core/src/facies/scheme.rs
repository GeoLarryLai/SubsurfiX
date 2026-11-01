// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Lithofacies from logs: a three-class roll-up (sandstone, mudstone,
//! carbonate) and a five-facies interpretation layer (sandstone, shale,
//! limestone, dolomite, cherty carbonate) derived from cutoffs or from a
//! mineral-mixing solve.
//!
//! - [`Method::Cutoffs`]: an ordered rules table (first match wins) on GR,
//!   RHOB, NPHI, PEF and the apparent matrix values RHOMAA and UMAA.
//! - [`Method::Mixing`]: non-negative least squares for quartz, calcite,
//!   dolomite, clay and porosity from RHOB, NPHI, U = PEF·RHOB and closure.
//!   Rows without PEF (or NPHI) use a reduced system and are flagged coarse.
//! - [`RollupClassifier`]: a supervised Gaussian classifier for the
//!   three roll-up classes, trained on labelled wells.
//!
//! NPHI is in limestone units (v/v).

use super::separability::Gaussian;
use crate::{SxError, SxResult};

/// Five-facies codes, plus an undifferentiated carbonate for rows that lack
/// the logs needed to split carbonates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Facies {
    Sandstone = 0,
    Shale = 1,
    Limestone = 2,
    Dolomite = 3,
    ChertyCarbonate = 4,
    Carbonate = 5,
}

impl Facies {
    pub const ALL: [Facies; 6] = [
        Self::Sandstone,
        Self::Shale,
        Self::Limestone,
        Self::Dolomite,
        Self::ChertyCarbonate,
        Self::Carbonate,
    ];
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sandstone => "sandstone",
            Self::Shale => "shale",
            Self::Limestone => "limestone",
            Self::Dolomite => "dolomite",
            Self::ChertyCarbonate => "cherty_carbonate",
            Self::Carbonate => "carbonate",
        }
    }
    /// Parse a name.
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for unknown names.
    pub fn parse(s: &str) -> SxResult<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.name() == s.trim().to_ascii_lowercase())
            .ok_or_else(|| SxError::InvalidInput(format!("unknown facies {s:?}")))
    }
    /// Roll-up class: 0 sandstone, 1 mudstone, 2 carbonate.
    pub fn rollup(self) -> Rollup {
        match self {
            Self::Sandstone => Rollup::Sandstone,
            Self::Shale => Rollup::Mudstone,
            _ => Rollup::Carbonate,
        }
    }
}

/// Roll-up classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Rollup {
    Sandstone = 0,
    Mudstone = 1,
    Carbonate = 2,
}

/// Interpretation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Cutoffs,
    Mixing,
}

impl Method {
    /// Parse "cutoffs" / "mixing".
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for other names.
    pub fn parse(s: &str) -> SxResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cutoffs" | "rules" => Ok(Self::Cutoffs),
            "mixing" | "nnls" => Ok(Self::Mixing),
            _ => Err(SxError::InvalidInput(format!(
                "unknown facies method {s:?}"
            ))),
        }
    }
}

/// Log or derived variable used by a rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    Gr,
    Rhob,
    Nphi,
    Pef,
    Rhomaa,
    Umaa,
}

impl Var {
    fn name(self) -> &'static str {
        match self {
            Self::Gr => "gr",
            Self::Rhob => "rhob",
            Self::Nphi => "nphi",
            Self::Pef => "pef",
            Self::Rhomaa => "rhomaa",
            Self::Umaa => "umaa",
        }
    }
    fn parse(s: &str) -> SxResult<Self> {
        [
            Self::Gr,
            Self::Rhob,
            Self::Nphi,
            Self::Pef,
            Self::Rhomaa,
            Self::Umaa,
        ]
        .into_iter()
        .find(|v| v.name() == s.trim().to_ascii_lowercase())
        .ok_or_else(|| SxError::InvalidInput(format!("unknown variable {s:?}")))
    }
}

/// One rule: all conditions `min <= var < max` must hold. A rule whose
/// variables are missing on a row is skipped. `coarse` marks fallback rules
/// that do not use PEF.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub facies: Facies,
    pub conditions: Vec<(Var, f64, f64)>,
    pub coarse: bool,
}

/// Ordered rules table.
#[derive(Clone, Debug, PartialEq)]
pub struct Cutoffs {
    pub rules: Vec<Rule>,
}

impl Default for Cutoffs {
    fn default() -> Self {
        use Facies::*;
        use Var::*;
        let inf = f64::INFINITY;
        let r = |f, c: Vec<(Var, f64, f64)>, coarse| Rule {
            facies: f,
            conditions: c,
            coarse,
        };
        Self {
            rules: vec![
                r(Shale, vec![(Gr, 75.0, inf)], false),
                r(Sandstone, vec![(Umaa, -inf, 6.5)], false),
                r(
                    Dolomite,
                    vec![(Umaa, 6.5, 11.5), (Rhomaa, 2.76, inf)],
                    false,
                ),
                r(ChertyCarbonate, vec![(Umaa, 6.5, 11.5)], false),
                r(Limestone, vec![(Umaa, 11.5, inf)], false),
                r(Sandstone, vec![(Rhomaa, -inf, 2.68)], true),
                r(Carbonate, vec![(Rhomaa, 2.68, inf)], true),
            ],
        }
    }
}

impl Cutoffs {
    /// CSV with header `order,facies,variable,min,max,coarse`. Rows that
    /// share `order` form one rule.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("order,facies,variable,min,max,coarse\n");
        for (i, r) in self.rules.iter().enumerate() {
            for (v, lo, hi) in &r.conditions {
                s += &format!(
                    "{},{},{},{},{},{}\n",
                    i + 1,
                    r.facies.name(),
                    v.name(),
                    lo,
                    hi,
                    r.coarse
                );
            }
        }
        s
    }

    /// Parse the CSV written by [`Cutoffs::to_csv`]. Empty min/max mean open.
    ///
    /// # Errors
    /// [`SxError::InvalidInput`] for malformed rows.
    pub fn from_csv(text: &str) -> SxResult<Self> {
        let num = |t: &str, d: f64| -> SxResult<f64> {
            let t = t.trim();
            if t.is_empty() {
                Ok(d)
            } else {
                t.parse::<f64>()
                    .map_err(|_| SxError::InvalidInput(format!("bad number {t:?}")))
            }
        };
        let mut rules: Vec<(String, Rule)> = Vec::new();
        for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 5 {
                return Err(SxError::InvalidInput(format!("bad cutoffs row {line:?}")));
            }
            let cond = (
                Var::parse(f[2])?,
                num(f[3], f64::NEG_INFINITY)?,
                num(f[4], f64::INFINITY)?,
            );
            let coarse = f
                .get(5)
                .is_some_and(|t| t.trim().eq_ignore_ascii_case("true"));
            let fac = Facies::parse(f[1])?;
            match rules.last_mut() {
                Some((o, r)) if o == f[0].trim() => {
                    if r.facies != fac {
                        return Err(SxError::InvalidInput(format!("rule {o} mixes facies")));
                    }
                    r.conditions.push(cond);
                }
                _ => rules.push((
                    f[0].trim().to_string(),
                    Rule {
                        facies: fac,
                        conditions: vec![cond],
                        coarse,
                    },
                )),
            }
        }
        Ok(Self {
            rules: rules.into_iter().map(|(_, r)| r).collect(),
        })
    }
}

/// Mineral and fluid endpoints: (RHOB g/cc, NPHI v/v limestone units, PEF b/e).
#[derive(Clone, Debug, PartialEq)]
pub struct Endpoints {
    pub quartz: [f64; 3],
    pub calcite: [f64; 3],
    pub dolomite: [f64; 3],
    pub clay: [f64; 3],
    pub fluid: [f64; 3],
}

impl Default for Endpoints {
    // Dolomite PEF is 3.14 b/e, not pi.
    #[allow(clippy::approx_constant)]
    fn default() -> Self {
        Self {
            quartz: [2.65, -0.02, 1.81],
            calcite: [2.71, 0.0, 5.08],
            dolomite: [2.87, 0.02, 3.14],
            clay: [2.52, 0.30, 3.45],
            fluid: [1.0, 1.0, 0.36],
        }
    }
}

/// Parameters.
#[derive(Clone, Debug)]
pub struct FaciesParams {
    pub method: Method,
    pub cutoffs: Cutoffs,
    pub endpoints: Endpoints,
    /// Measurement scales used to weight the mixing equations
    /// (RHOB, NPHI, U, closure).
    pub sigma: [f64; 4],
    /// Mixing: clay fraction of solids at or above which a row is shale.
    pub shale_clay: f64,
    /// Mixing: quartz fraction of solids at or above which a carbonate is cherty.
    pub chert_quartz: f64,
    /// Fluid density used for RHOMAA / UMAA.
    pub fluid_rho: f64,
    /// Subtracted from NPHI before interpretation (v/v; 0 = logs as given).
    pub nphi_shift: f64,
}

impl Default for FaciesParams {
    fn default() -> Self {
        Self {
            method: Method::Cutoffs,
            cutoffs: Cutoffs::default(),
            endpoints: Endpoints::default(),
            sigma: [0.025, 0.03, 0.5, 0.01],
            shale_clay: 0.5,
            chert_quartz: 0.2,
            fluid_rho: 1.0,
            nphi_shift: 0.0,
        }
    }
}

/// Logs per row; NaN = missing.
#[derive(Clone, Copy, Debug)]
pub struct FaciesInput<'a> {
    pub gr: &'a [f64],
    pub rhob: &'a [f64],
    pub nphi: &'a [f64],
    pub pef: &'a [f64],
}

/// Per-row result. `facies` and `rollup` are -1 where no rule applies.
#[derive(Clone, Debug, Default)]
pub struct FaciesResult {
    pub facies: Vec<i32>,
    pub rollup: Vec<i32>,
    /// True where PEF (or NPHI for mixing) was missing and a reduced rule
    /// or system was used.
    pub coarse: Vec<bool>,
    pub rhomaa: Vec<f64>,
    pub umaa: Vec<f64>,
    /// Mixing only: volumes (quartz, calcite, dolomite, clay, porosity).
    pub volumes: Vec<[f64; 5]>,
    /// Mixing only: weighted RMS misfit of the solve.
    pub misfit: Vec<f64>,
}

/// Apparent matrix density and volumetric cross-section from RHOB, NPHI, PEF.
fn apparent_matrix(rhob: f64, nphi: f64, pef: f64, fluid_rho: f64) -> (f64, f64) {
    if !(rhob.is_finite() && nphi.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let dphi = (2.71 - rhob) / (2.71 - fluid_rho);
    let phi = (0.5 * (dphi + nphi)).clamp(0.0, 0.9);
    let rhomaa = (rhob - phi * fluid_rho) / (1.0 - phi);
    let umaa = if pef.is_finite() {
        (pef * rhob - phi * 0.398) / (1.0 - phi)
    } else {
        f64::NAN
    };
    (rhomaa, umaa)
}

/// Lawson–Hanson non-negative least squares, `min |A x - b|, x >= 0`.
/// `a` is row-major m × n.
#[allow(clippy::needless_range_loop)]
pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let n = if m == 0 { 0 } else { a[0].len() };
    let mut x = vec![0.0; n];
    let mut passive = vec![false; n];
    let grad = |x: &[f64]| -> Vec<f64> {
        let r: Vec<f64> = (0..m)
            .map(|i| b[i] - (0..n).map(|j| a[i][j] * x[j]).sum::<f64>())
            .collect();
        (0..n)
            .map(|j| (0..m).map(|i| a[i][j] * r[i]).sum())
            .collect()
    };
    let solve_passive = |p: &[bool]| -> Vec<f64> {
        let idx: Vec<usize> = (0..n).filter(|&j| p[j]).collect();
        let k = idx.len();
        let mut g = vec![vec![0.0; k + 1]; k];
        for (r, &jr) in idx.iter().enumerate() {
            for (c, &jc) in idx.iter().enumerate() {
                g[r][c] = (0..m).map(|i| a[i][jr] * a[i][jc]).sum();
            }
            g[r][r] += 1e-12;
            g[r][k] = (0..m).map(|i| a[i][jr] * b[i]).sum();
        }
        for c in 0..k {
            let piv = (c..k)
                .max_by(|&p, &q| g[p][c].abs().total_cmp(&g[q][c].abs()))
                .unwrap_or(c);
            g.swap(c, piv);
            let d = g[c][c];
            if d.abs() < 1e-300 {
                continue;
            }
            for r in 0..k {
                if r != c {
                    let f = g[r][c] / d;
                    for cc in c..=k {
                        g[r][cc] -= f * g[c][cc];
                    }
                }
            }
        }
        let mut z = vec![0.0; n];
        for (r, &j) in idx.iter().enumerate() {
            z[j] = if g[r][r].abs() > 1e-300 {
                g[r][k] / g[r][r]
            } else {
                0.0
            };
        }
        z
    };
    for _ in 0..3 * n + 10 {
        let w = grad(&x);
        let cand = (0..n)
            .filter(|&j| !passive[j] && w[j] > 1e-10)
            .max_by(|&p, &q| w[p].total_cmp(&w[q]));
        let Some(j) = cand else { break };
        passive[j] = true;
        loop {
            let z = solve_passive(&passive);
            if (0..n).all(|j| !passive[j] || z[j] > 0.0) {
                x = z;
                break;
            }
            let mut alpha = 1.0f64;
            for j in 0..n {
                if passive[j] && z[j] <= 0.0 {
                    alpha = alpha.min(x[j] / (x[j] - z[j]));
                }
            }
            for j in 0..n {
                x[j] += alpha * (z[j] - x[j]);
                if passive[j] && x[j] <= 1e-12 {
                    passive[j] = false;
                    x[j] = 0.0;
                }
            }
        }
    }
    x
}

/// Mixing solve for one row. Returns volumes, misfit and whether the
/// system was reduced.
fn mix_row(rhob: f64, nphi: f64, pef: f64, p: &FaciesParams) -> Option<([f64; 5], f64, bool)> {
    if !rhob.is_finite() {
        return None;
    }
    let e = &p.endpoints;
    let comps = [e.quartz, e.calcite, e.dolomite, e.clay, e.fluid];
    let has_n = nphi.is_finite();
    let has_u = pef.is_finite();
    // Reduced unknown sets: without PEF dolomite is not resolvable;
    // without NPHI clay and porosity are not separable from the matrix
    // minerals, so clay is dropped.
    let use_comp = [true, true, has_u, has_n, true];
    let cols: Vec<usize> = (0..5).filter(|&j| use_comp[j]).collect();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut row = |coef: &dyn Fn(&[f64; 3]) -> f64, val: f64, s: f64| {
        a.push(
            cols.iter()
                .map(|&j| coef(&comps[j]) / s)
                .collect::<Vec<f64>>(),
        );
        b.push(val / s);
    };
    row(&|c| c[0], rhob, p.sigma[0]);
    if has_n {
        row(&|c| c[1], nphi, p.sigma[1]);
    }
    if has_u {
        row(&|c| c[2] * c[0], pef * rhob, p.sigma[2]);
    }
    row(&|_| 1.0, 1.0, p.sigma[3]);
    let xs = nnls(&a, &b);
    let mut v = [0.0; 5];
    for (k, &j) in cols.iter().enumerate() {
        v[j] = xs[k];
    }
    let misfit = (a
        .iter()
        .zip(&b)
        .map(|(r, bi)| (r.iter().zip(&xs).map(|(x, y)| x * y).sum::<f64>() - bi).powi(2))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt();
    Some((v, misfit, !(has_n && has_u)))
}

fn mix_facies(v: &[f64; 5], gr: f64, coarse: bool, p: &FaciesParams) -> Facies {
    let solids = (v[0] + v[1] + v[2] + v[3]).max(1e-9);
    let (q, c, d, cl) = (v[0] / solids, v[1] / solids, v[2] / solids, v[3] / solids);
    let shale_gr = p
        .cutoffs
        .rules
        .iter()
        .find(|r| r.facies == Facies::Shale)
        .and_then(|r| r.conditions.iter().find(|c| c.0 == Var::Gr).map(|c| c.1));
    if cl >= p.shale_clay || matches!((shale_gr, gr.is_finite()), (Some(t), true) if gr >= t) {
        return Facies::Shale;
    }
    if q >= c + d {
        return Facies::Sandstone;
    }
    if coarse {
        return Facies::Carbonate;
    }
    if q >= p.chert_quartz {
        Facies::ChertyCarbonate
    } else if d > c {
        Facies::Dolomite
    } else {
        Facies::Limestone
    }
}

/// Classify rows into the five facies and the roll-up.
///
/// # Errors
/// [`SxError::Shape`] when the log lengths differ.
pub fn interpret(inp: &FaciesInput, p: &FaciesParams) -> SxResult<FaciesResult> {
    let n = inp.rhob.len();
    if [inp.gr.len(), inp.nphi.len(), inp.pef.len()]
        .iter()
        .any(|&l| l != n)
    {
        return Err(SxError::Shape("facies logs must have equal length".into()));
    }
    let mut out = FaciesResult::default();
    for i in 0..n {
        let (gr, rhob, nphi, pef) = (
            inp.gr[i],
            inp.rhob[i],
            inp.nphi[i] - p.nphi_shift,
            inp.pef[i],
        );
        let (rhomaa, umaa) = apparent_matrix(rhob, nphi, pef, p.fluid_rho);
        out.rhomaa.push(rhomaa);
        out.umaa.push(umaa);
        let (fac, coarse) = match p.method {
            Method::Cutoffs => {
                let val = |v: Var| match v {
                    Var::Gr => gr,
                    Var::Rhob => rhob,
                    Var::Nphi => nphi,
                    Var::Pef => pef,
                    Var::Rhomaa => rhomaa,
                    Var::Umaa => umaa,
                };
                p.cutoffs
                    .rules
                    .iter()
                    .find(|r| {
                        r.conditions.iter().all(|&(v, lo, hi)| {
                            let x = val(v);
                            x.is_finite() && x >= lo && x < hi
                        })
                    })
                    .map_or((None, false), |r| (Some(r.facies), r.coarse))
            }
            Method::Mixing => match mix_row(rhob, nphi, pef, p) {
                Some((v, mis, coarse)) => {
                    out.volumes.push(v);
                    out.misfit.push(mis);
                    (Some(mix_facies(&v, gr, coarse, p)), coarse)
                }
                None => {
                    out.volumes.push([f64::NAN; 5]);
                    out.misfit.push(f64::NAN);
                    (None, false)
                }
            },
        };
        out.facies.push(fac.map_or(-1, |f| f as i32));
        out.rollup.push(fac.map_or(-1, |f| f.rollup() as i32));
        out.coarse.push(coarse);
    }
    Ok(out)
}

/// Supervised Gaussian classifier for the three roll-up classes.
pub struct RollupClassifier {
    model: Gaussian,
}

impl RollupClassifier {
    /// Fit on complete rows with labels 0..3; other rows are ignored.
    ///
    /// # Errors
    /// [`SxError::EmptyInput`] when fewer than two rows of each class remain.
    pub fn fit(x: &[Vec<f64>], y: &[i32], shrink: f64) -> SxResult<Self> {
        let keep: Vec<usize> = (0..x.len())
            .filter(|&i| (0..3).contains(&y[i]) && x[i].iter().all(|v| v.is_finite()))
            .collect();
        for c in 0..3 {
            if keep.iter().filter(|&&i| y[i] == c).count() < 2 {
                return Err(SxError::EmptyInput(format!(
                    "roll-up class {c} has <2 rows"
                )));
            }
        }
        let yu: Vec<usize> = y.iter().map(|&v| v.max(0) as usize).collect();
        Ok(Self {
            model: Gaussian::fit(x, &yu, &keep, 3, shrink),
        })
    }

    /// Predict; -1 for rows with missing values.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i32> {
        x.iter()
            .map(|r| {
                if r.iter().all(|v| v.is_finite()) {
                    self.model.predict(r) as i32
                } else {
                    -1
                }
            })
            .collect()
    }
}
