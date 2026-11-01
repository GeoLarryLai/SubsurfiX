// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! CO2-brine fluid properties for carbon storage (EXPERIMENTAL prototype).
//!
//! Pure functions of state (no globals, no allocation in the hot path), so
//! they are thread-safe and suitable for building operator-based
//! linearization (OBL) tables in parallel.
//!
//! Models, chosen to follow the open-DARTS CCS property set where possible:
//!
//! * Mutual solubility of CO2 and H2O in NaCl brine:
//!   Spycher, Pruess & Ennis-King (2003), *Geochim. Cosmochim. Acta* 67,
//!   3015-3031 (Redlich-Kwong gas phase, equilibrium constants, average
//!   partial molar volumes), with the salinity extension of Spycher & Pruess
//!   (2005), *Geochim. Cosmochim. Acta* 69, 3309-3320, using the CO2 activity
//!   coefficient of Duan & Sun (2003), *Chem. Geol.* 193, 257-271.
//!   Valid 12-100 °C, up to 600 bar, up to 6 mol/kg NaCl. The high-temperature
//!   model of Spycher & Pruess (2010) is NOT implemented.
//! * [`SolubilityModel::ZiabakhshGanjiKooi2012`] (the Henry's-law aqueous
//!   model open-DARTS uses): Ziabakhsh-Ganji & Kooi (2012), *Int. J.
//!   Greenhouse Gas Control* 11, S21-S34, as reprinted in Ziabakhsh-Ganji
//!   (2015), PhD thesis, VU Amsterdam, ch. 2 (http://hdl.handle.net/1871/52539):
//!   Peng-Robinson gas phase (eqs. 15-23, Table 2.5 k_ij), the Henry's
//!   constant of Akinfiev & Diamond (2003), *Geochim. Cosmochim. Acta* 67,
//!   613-629, with ZGK Table 2.2 parameters (eqs. 24-25), Fine & Millero
//!   (1973) water density and King et al. (1992) water fugacity with the
//!   Shibue (2003) vapour pressure (Appendix 2-A), the activity coefficient
//!   of eqs. 26-28 (Tables 2.3-2.4) and the closed form eq. 29. CO2 only.
//! * CO2 density and enthalpy: Peng & Robinson (1976), *Ind. Eng. Chem.
//!   Fundam.* 15, 59-64, without volume shift (as open-DARTS), with the NIST
//!   Shomate ideal-gas heat capacity of CO2. Span-Wagner is not implemented.
//! * CO2 viscosity: Fenghour, Wakeham & Vesovic (1998), *J. Phys. Chem. Ref.
//!   Data* 27, 31-44 (zero-density + excess terms; critical enhancement
//!   neglected).
//! * Brine density: pure water from IAPWS-IF97 ([`super::water`]), NaCl
//!   correction of Batzle & Wang (1992), *Geophysics* 57, 1396-1408, and
//!   dissolved CO2 through the apparent molar volume of Garcia (2001),
//!   LBNL-49023.
//! * Brine viscosity: Mao & Duan (2009), *Int. J. Thermophys.* 30, 1510-1523
//!   (relative viscosity of NaCl brine over IF97 water), times the CO2 factor
//!   of Islam & Carlson (2012), *Energy Fuels* 26, 5330-5336.
//! * Aqueous enthalpy: IF97 water plus dissolved CO2 at its ideal-gas
//!   enthalpy plus the dissolution enthalpy from the van't Hoff derivative of
//!   the Spycher et al. (2003) CO2(g) equilibrium constant. The NaCl effect on
//!   enthalpy is neglected.
//!
//! Units: Pa, K, mol/kg (molality), kg/m³, Pa·s, J/kg. Enthalpy reference:
//! IF97 for water; CO2 ideal gas at 298.15 K is zero.

use super::water;
use crate::error::{SxError, SxResult};

/// Molar mass of CO2 (kg/mol).
pub const M_CO2: f64 = 44.0095e-3;
/// Molar mass of H2O (kg/mol).
pub const M_H2O: f64 = 18.015_268e-3;
/// Molar mass of NaCl (kg/mol).
pub const M_NACL: f64 = 58.443e-3;
const R_SI: f64 = 8.314_462_618;
const R_BAR: f64 = 83.144_7; // bar cm³ /(mol K), as in Spycher et al. (2003)
const MOL_H2O_PER_KG: f64 = 55.508;

/// Aqueous-phase solubility model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolubilityModel {
    /// Spycher et al. (2003) + Spycher & Pruess (2005) salinity (default).
    #[default]
    SpycherPruess2005,
    /// Ziabakhsh-Ganji & Kooi (2012), CO2-H2O-NaCl only.
    ZiabakhshGanjiKooi2012,
}

/// Equilibrium compositions of the two coexisting phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solubility {
    /// CO2 mole fraction in the aqueous phase, salt-free basis (CO2 + H2O).
    pub x_co2: f64,
    /// CO2 molality in the aqueous phase (mol per kg H2O).
    pub m_co2: f64,
    /// H2O mole fraction in the CO2-rich phase.
    pub y_h2o: f64,
    /// Whether the CO2-rich phase is liquid-like (T < 31 °C, liquid root).
    pub co2_liquid: bool,
}

fn check_range(p_pa: f64, t_k: f64, m_nacl: f64, model: SolubilityModel) -> SxResult<()> {
    let t_c = t_k - 273.15;
    if !(p_pa.is_finite() && t_k.is_finite() && m_nacl.is_finite())
        || !(12.0..=100.0).contains(&t_c)
        || !(1e5..=600e5).contains(&p_pa)
        || !(0.0..=6.0).contains(&m_nacl)
    {
        return Err(SxError::InvalidInput(format!(
            "co2_brine: {model:?} is limited here to 12-100 °C, 1-600 bar, 0-6 mol/kg NaCl; got {t_c:.2} °C, {:.2} bar, {m_nacl} mol/kg",
            p_pa / 1e5
        )));
    }
    Ok(())
}

/// Real roots of x³ + a x² + b x + c = 0, ascending.
fn cubic_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let q = (a * a - 3.0 * b) / 9.0;
    let r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;
    if r * r < q * q * q {
        let th = (r / (q * q * q).sqrt()).acos();
        let s = -2.0 * q.sqrt();
        let mut v = vec![
            s * (th / 3.0).cos() - a / 3.0,
            s * ((th + 2.0 * std::f64::consts::PI) / 3.0).cos() - a / 3.0,
            s * ((th - 2.0 * std::f64::consts::PI) / 3.0).cos() - a / 3.0,
        ];
        v.sort_by(f64::total_cmp);
        v
    } else {
        let aa = -r.signum() * (r.abs() + (r * r - q * q * q).sqrt()).cbrt();
        let bb = if aa == 0.0 { 0.0 } else { q / aa };
        vec![aa + bb - a / 3.0]
    }
}

/// Redlich-Kwong fugacity coefficients (φ_H2O, φ_CO2) of the CO2-rich phase,
/// Spycher et al. (2003) eqs. 1-2 and A-8 (infinite dilution of H2O).
fn rk_fugacity(p_bar: f64, t_k: f64) -> (f64, f64, bool) {
    let a = 7.54e7 - 4.13e4 * t_k;
    let b = 27.80;
    let (b_h2o, a_h2o_co2) = (18.18, 7.89e7);
    let rt = R_BAR * t_k;
    let sq = t_k.sqrt();
    let roots: Vec<f64> = cubic_roots(
        -rt / p_bar,
        -(rt * b / p_bar - a / (p_bar * sq) + b * b),
        -a * b / (p_bar * sq),
    )
    .into_iter()
    .filter(|&v| v > b)
    .collect();
    let lnphi = |v: f64, bk: f64, aik: f64| {
        (v / (v - b)).ln() + bk / (v - b) - 2.0 * aik / (rt * sq * b) * ((v + b) / v).ln()
            + a * bk / (rt * sq * b * b) * (((v + b) / v).ln() - b / (v + b))
            - (p_bar * v / rt).ln()
    };
    // Choose the root with the lower CO2 fugacity (stable phase).
    let (mut best, mut best_l) = (roots[roots.len() - 1], f64::INFINITY);
    for &v in &roots {
        let l = lnphi(v, b, a);
        if l < best_l {
            best_l = l;
            best = v;
        }
    }
    let liquid = roots.len() > 1 && best == roots[0];
    (lnphi(best, b_h2o, a_h2o_co2).exp(), best_l.exp(), liquid)
}

/// Duan & Sun (2003) parameter form, T in K, P in bar.
fn duan_sun_par(c: &[f64; 11], t: f64, p: f64) -> f64 {
    c[0] + c[1] * t
        + c[2] / t
        + c[3] * t * t
        + c[4] / (630.0 - t)
        + c[5] * p
        + c[6] * p * t.ln()
        + c[7] * p / t
        + c[8] * p / (630.0 - t)
        + c[9] * p * p / (630.0 - t).powi(2)
        + c[10] * t * p.ln()
}

/// CO2 activity coefficient in NaCl brine, molality scale, Duan & Sun (2003)
/// as given in Spycher & Pruess (2005) eq. A-4: ln γ = 2λ m_Na + ξ m_Cl m_Na.
fn gamma_co2_m(p_bar: f64, t_k: f64, m: f64) -> f64 {
    const LAMBDA: [f64; 11] = [
        -0.411_370_585,
        6.076_320_13e-4,
        97.534_770_8,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.023_762_246_9,
        0.017_065_623_6,
        0.0,
        1.413_358_34e-5,
    ];
    const ZETA: [f64; 11] = [
        3.363_897_23e-4,
        -1.982_989_80e-5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.122_208_30e-3,
        -5.248_733_03e-3,
        0.0,
        0.0,
    ];
    let lam = duan_sun_par(&LAMBDA, t_k, p_bar);
    let zeta = duan_sun_par(&ZETA, t_k, p_bar);
    (2.0 * lam * m + zeta * m * m).exp()
}

/// CO2 saturation pressure (Pa), Span & Wagner (1996) eq. 3.13 ancillary.
pub fn co2_saturation_pressure(t_k: f64) -> f64 {
    let tau = 1.0 - t_k / TC;
    if tau <= 0.0 {
        return PC;
    }
    PC * (TC / t_k
        * (-7.060_208_7 * tau + 1.939_121_8 * tau.powf(1.5)
            - 1.646_359_7 * tau * tau
            - 3.299_563_4 * tau.powi(4)))
    .exp()
}

/// log10 K0 of CO2(g or l) and H2O, Spycher et al. (2003) Table 2, t in °C.
fn log_k0(t_c: f64, co2_liquid: bool) -> (f64, f64) {
    let k_h2o = -2.209 + 3.097e-2 * t_c - 1.098e-4 * t_c * t_c + 2.048e-7 * t_c.powi(3);
    let k_co2 = if co2_liquid {
        1.169 + 1.368e-2 * t_c - 5.380e-5 * t_c * t_c
    } else {
        1.189 + 1.304e-2 * t_c - 5.446e-5 * t_c * t_c
    };
    (k_h2o, k_co2)
}

/// Mutual solubility of CO2 and H2O at `p_pa`, `t_k` and NaCl molality
/// `m_nacl` (Spycher et al. 2003; Spycher & Pruess 2005).
///
/// # Errors
/// `InvalidInput` out of range (12-100 °C, 1-600 bar, 0-6 mol/kg for both
/// models; the 100 °C cap comes from the Spycher et al. (2003) K0_H2O that
/// ZGK also uses).
pub fn mutual_solubility(
    p_pa: f64,
    t_k: f64,
    m_nacl: f64,
    model: SolubilityModel,
) -> SxResult<Solubility> {
    check_range(p_pa, t_k, m_nacl, model)?;
    if model == SolubilityModel::ZiabakhshGanjiKooi2012 {
        return Ok(zgk_solubility(p_pa / 1e5, t_k, m_nacl));
    }
    let p = p_pa / 1e5;
    let t_c = t_k - 273.15;
    let (phi_w, phi_c, _) = rk_fugacity(p, t_k);
    // Spycher et al. (2003): below the CO2 critical temperature and above its
    // saturation pressure the CO2-rich phase is liquid (K0 of CO2(l)).
    let liq = t_k < TC && p_pa > co2_saturation_pressure(t_k);
    let (lk_w, lk_c) = log_k0(t_c, liq);
    let rt = R_BAR * t_k;
    let dp = p - 1.0;
    let a_ = 10f64.powf(lk_w) / (phi_w * p) * (dp * 18.1 / rt).exp();
    let gm = gamma_co2_m(p, t_k, m_nacl);
    let nm = 2.0 * m_nacl;
    // Spycher & Pruess (2005) eqs. 4, 7, 9, 11, with the molality-to-mole-
    // fraction conversion A-8, which depends on m_CO2: fixed-point iteration
    // (converges to machine precision in a few passes; exact in pure water).
    let mut m_co2 = 0.0;
    let mut y_h2o = 0.0;
    for _ in 0..30 {
        let gx = gm * (1.0 + (m_co2 + nm) / MOL_H2O_PER_KG) / (1.0 + m_co2 / MOL_H2O_PER_KG);
        let b_ = phi_c * p / (MOL_H2O_PER_KG * gx * 10f64.powf(lk_c)) * (-dp * 32.6 / rt).exp();
        y_h2o = (1.0 - b_) * MOL_H2O_PER_KG / ((1.0 / a_ - b_) * (nm + MOL_H2O_PER_KG) + nm * b_);
        let x = b_ * (1.0 - y_h2o);
        let m_new = x * (MOL_H2O_PER_KG + nm) / (1.0 - x);
        let done = (m_new - m_co2).abs() < 1e-13;
        m_co2 = m_new;
        if done {
            break;
        }
    }
    let x_co2 = m_co2 / (m_co2 + MOL_H2O_PER_KG);
    Ok(Solubility {
        x_co2,
        m_co2,
        y_h2o,
        co2_liquid: liq,
    })
}

// ---------------- Ziabakhsh-Ganji & Kooi (2012) ----------------
// Equation and table numbers refer to Ziabakhsh-Ganji (2015) thesis ch. 2,
// a reprint of Ziabakhsh-Ganji & Kooi (2012). Units there: bar, K, cm³, g.

/// Peng & Robinson (1976) a(T), b in bar cm⁶/mol² and cm³/mol (eqs. 18-20).
/// The thesis prints the κ(ω) polynomial as 0.37646 + 1.4522ω − 0.26992ω²;
/// that differs from Peng & Robinson (1976), whose 0.37464 + 1.54226ω −
/// 0.26992ω² is used here.
fn zgk_pr_ab(tc: f64, pc_bar: f64, omega: f64, t_k: f64) -> (f64, f64) {
    let kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
    let s = 1.0 + kappa * (1.0 - (t_k / tc).sqrt());
    (
        0.45724 * R_BAR * R_BAR * tc * tc / pc_bar * s * s,
        0.07780 * R_BAR * tc / pc_bar,
    )
}

/// PR fugacity coefficients (φ_CO2, φ_H2O) of the CO2-rich phase with water
/// at infinite dilution in the mixing rule (eqs. 15-23), k_CO2-H2O = 0.19014
/// (Table 2.5). Critical constants: CO2 as above; H2O 647.096 K, 220.64 bar,
/// ω 0.3443 (not listed in the paper).
fn zgk_fugacity(p_bar: f64, t_k: f64) -> (f64, f64) {
    let (ac, bc) = zgk_pr_ab(TC, PC / 1e5, OMEGA, t_k);
    let (aw, bw) = zgk_pr_ab(647.096, 220.64, 0.3443, t_k);
    let rt = R_BAR * t_k;
    let a = ac * p_bar / (rt * rt);
    let b = bc * p_bar / rt;
    let roots: Vec<f64> = cubic_roots(
        -(1.0 - b),
        a - 2.0 * b - 3.0 * b * b,
        -(a * b - b * b - b * b * b),
    )
    .into_iter()
    .filter(|&z| z > b)
    .collect();
    let mut z = roots[roots.len() - 1];
    if roots.len() > 1 {
        // Eq. 23 (Danesh 1998): pick the root of lower Gibbs energy.
        let (zl, zh) = (roots[0], roots[roots.len() - 1]);
        let s2 = std::f64::consts::SQRT_2;
        let (d1, d2) = (1.0 + s2, 1.0 - s2);
        let dg = (zh - zl) + ((zl - b) / (zh - b)).ln()
            - a / (b * (d2 - d1))
                * (((zl + d1 * b) * (zh + d2 * b)) / ((zl + d2 * b) * (zh + d1 * b))).ln();
        if dg > 0.0 {
            z = zl;
        }
    }
    let l = ((z + 2.414 * b) / (z - 0.414 * b)).ln();
    let lnphi = |bi: f64, sum_ya: f64| {
        bi / bc * (z - 1.0) - (z - b).ln() + a / (2.828 * b) * (bi / bc - 2.0 * sum_ya / ac) * l
    };
    let awc = (aw * ac).sqrt() * (1.0 - 0.19014);
    (lnphi(bc, ac).exp(), lnphi(bw, awc).exp())
}

/// Pure water density (g/cm³) of Fine & Millero (1973) and fugacity (bar)
/// of King et al. (1992) with the Shibue (2003) vapour pressure, thesis
/// Appendix 2-A eqs. A1-A7 (H2O critical point 647.096 K, 220.64 bar).
fn zgk_water(p_bar: f64, t_k: f64) -> (f64, f64) {
    let th = t_k - 273.15;
    let v0 = (1.0 + 18.159_725e-3 * th)
        / (0.999_839_6 + 18.224_944e-3 * th
            - 7.922_210e-6 * th.powi(2)
            - 55.448_46e-9 * th.powi(3)
            + 149.756_2e-12 * th.powi(4)
            - 393.295_2e-15 * th.powi(5));
    let bq = 19_654.320 + 147.037 * th - 2.215_54 * th.powi(2) + 1.0478e-2 * th.powi(3)
        - 2.2789e-5 * th.powi(4);
    let a1 = 3.2891 - 2.3910e-3 * th + 2.8446e-4 * th.powi(2) - 2.8200e-6 * th.powi(3)
        + 8.477e-9 * th.powi(4);
    let a2 = 6.245e-5 - 3.913e-6 * th - 3.499e-8 * th.powi(2) + 7.942e-10 * th.powi(3)
        - 3.299e-12 * th.powi(4);
    let v = v0 - v0 * p_bar / (bq + a1 * p_bar + a2 * p_bar * p_bar);
    let (tcw, pcw) = (647.096, 220.64);
    let tau = 1.0 - t_k / tcw;
    let ps = pcw
        * (tcw / t_k
            * (-7.859_517_83 * tau + 1.844_082_59 * tau.powf(1.5) - 11.786_649_7 * tau.powi(3)
                + 22.680_741_1 * tau.powf(3.5)
                - 15.961_871_9 * tau.powi(4)
                + 1.801_225_02 * tau.powf(7.5)))
        .exp();
    let f0 = ps * ((p_bar - ps) * 18.0152 * v / (R_BAR * t_k)).exp();
    (1.0 / v, f0)
}

/// ZGK eq. 27, T in K, P in bar.
fn zgk_par(c: &[f64; 10], t: f64, p: f64) -> f64 {
    c[0] + c[1] * t
        + c[2] / t
        + c[3] * p
        + c[4] / p
        + c[5] * p / t
        + c[6] * t / (p * p)
        + c[7] * p / (630.0 - t)
        + c[8] * t * p.ln()
        + c[9] * p / (t * t)
}

fn zgk_solubility(p: f64, t_k: f64, m_nacl: f64) -> Solubility {
    // Table 2.3 (λ_CO2-Na) and Table 2.4 (ζ_CO2-Na-Cl).
    const LAMBDA: [f64; 10] = [
        -0.0653, 1.6791e-4, 40.839, 0.0, 0.0, -3.9267e-2, 0.0, 2.1157e-2, 6.5486e-6, 0.0,
    ];
    const ZETA: [f64; 10] = [
        -1.1446e-2, 2.828e-5, 0.0, 0.0, 0.0, 1.398e-2, 0.0, -1.4349e-2, 0.0, 0.0,
    ];
    // Table 2.2, CO2: η, τ, β, Γ.
    let (eta, tau, beta, gamma_p) = (-0.114_535, -5.279_063, 6.187_967, 0.0);
    let rt = R_BAR * t_k;
    let (rho_w, f0_w) = zgk_water(p, t_k);
    let db = tau + gamma_p * p + beta * (1e3 / t_k).sqrt(); // eq. 25
    let ln_kh = (1.0 - eta) * f0_w.ln() + eta * (rt * rho_w / 18.0152).ln() + 2.0 * rho_w * db; // eq. 24
                                                                                                // Eq. 28 for NaCl: ln γ = 2 λ m_Na + ζ m_Cl m_Na.
    let ln_g = 2.0 * zgk_par(&LAMBDA, t_k, p) * m_nacl + zgk_par(&ZETA, t_k, p) * m_nacl * m_nacl;
    let (phi_c, phi_w) = zgk_fugacity(p, t_k);
    let (lk_w, _) = log_k0(t_k - 273.15, false); // eq. 12
                                                 // Eq. 29 with B = P φ_CO2 / (k_H γ), A = K0_H2O/(φ_H2O P) exp((P-1)V/RT).
    let bb = p * phi_c / (ln_kh + ln_g).exp();
    let aa = 10f64.powf(lk_w) / (phi_w * p) * ((p - 1.0) * 18.1 / rt).exp();
    let y_h2o = (1.0 - bb) / (1.0 / aa - bb);
    // x is the CO2 mole fraction with Σx = 1 over CO2 and H2O (eq. 30).
    let x_co2 = bb * (1.0 - y_h2o);
    Solubility {
        x_co2,
        m_co2: x_co2 * MOL_H2O_PER_KG / (1.0 - x_co2),
        y_h2o,
        co2_liquid: t_k < TC && p * 1e5 > co2_saturation_pressure(t_k),
    }
}

// ---------------- CO2: Peng-Robinson ----------------
const TC: f64 = 304.1282;
const PC: f64 = 7.3773e6;
const OMEGA: f64 = 0.22394;

fn pr_params(t_k: f64) -> (f64, f64, f64) {
    let kappa = 0.37464 + 1.54226 * OMEGA - 0.26992 * OMEGA * OMEGA;
    let ac = 0.457_235_53 * R_SI * R_SI * TC * TC / PC;
    let b = 0.077_796_07 * R_SI * TC / PC;
    let s = 1.0 + kappa * (1.0 - (t_k / TC).sqrt());
    let a = ac * s * s;
    let dadt = -ac * kappa * s / (t_k * TC).sqrt();
    (a, b, dadt)
}

/// (Z, A, B) of the stable PR root for pure CO2.
fn pr_z(p_pa: f64, t_k: f64) -> (f64, f64, f64) {
    let (a, b, _) = pr_params(t_k);
    let aa = a * p_pa / (R_SI * t_k).powi(2);
    let bb = b * p_pa / (R_SI * t_k);
    let roots: Vec<f64> = cubic_roots(
        -(1.0 - bb),
        aa - 3.0 * bb * bb - 2.0 * bb,
        -(aa * bb - bb * bb - bb * bb * bb),
    )
    .into_iter()
    .filter(|&z| z > bb)
    .collect();
    let s2 = std::f64::consts::SQRT_2;
    let lnphi = |z: f64| {
        z - 1.0
            - (z - bb).ln()
            - aa / (2.0 * s2 * bb) * ((z + (1.0 + s2) * bb) / (z + (1.0 - s2) * bb)).ln()
    };
    let z = roots
        .iter()
        .copied()
        .min_by(|x, y| lnphi(*x).total_cmp(&lnphi(*y)))
        .unwrap_or(1.0);
    (z, aa, bb)
}

/// Pure CO2 density (kg/m³), Peng-Robinson (1976) without volume shift.
pub fn co2_density(p_pa: f64, t_k: f64) -> f64 {
    let (z, _, _) = pr_z(p_pa, t_k);
    p_pa * M_CO2 / (z * R_SI * t_k)
}

/// Ideal-gas CO2 enthalpy (J/mol) relative to 298.15 K, NIST Shomate fit
/// (Chase 1998, NIST-JANAF; 298-1200 K, extrapolated mildly below 298 K).
fn co2_h_ideal(t_k: f64) -> f64 {
    let h = |t: f64| {
        let t = t / 1000.0;
        1000.0
            * (24.997_35 * t + 55.186_96 * t * t / 2.0 - 33.691_37 * t.powi(3) / 3.0
                + 7.948_387 * t.powi(4) / 4.0
                + 0.136_638 / t)
    };
    h(t_k) - h(298.15)
}

/// Pure CO2 specific enthalpy (J/kg): ideal gas + PR departure.
pub fn co2_enthalpy(p_pa: f64, t_k: f64) -> f64 {
    let (a, b, dadt) = pr_params(t_k);
    let (z, _, bb) = pr_z(p_pa, t_k);
    let s2 = std::f64::consts::SQRT_2;
    let dep = R_SI * t_k * (z - 1.0)
        + (t_k * dadt - a) / (2.0 * s2 * b) * ((z + (1.0 + s2) * bb) / (z + (1.0 - s2) * bb)).ln();
    (co2_h_ideal(t_k) + dep) / M_CO2
}

/// CO2 viscosity (Pa·s) at density `rho` (kg/m³) and `t_k`, Fenghour et al.
/// (1998) zero-density and excess terms; critical enhancement neglected.
pub fn co2_viscosity(rho: f64, t_k: f64) -> f64 {
    const A: [f64; 5] = [
        0.235_156,
        -0.491_266,
        5.211_155e-2,
        5.347_906e-2,
        -1.537_102e-2,
    ];
    let ts = t_k / 251.196;
    let lt = ts.ln();
    let ln_omega: f64 = A
        .iter()
        .enumerate()
        .map(|(i, a)| a * lt.powi(i as i32))
        .sum();
    let eta0 = 1.006_97 * t_k.sqrt() / ln_omega.exp();
    let d_eta = 0.407_111_9e-2 * rho
        + 0.719_803_7e-4 * rho * rho
        + 0.241_169_7e-16 * rho.powi(6) / ts.powi(3)
        + 0.297_107_2e-22 * rho.powi(8)
        - 0.162_788_8e-22 * rho.powi(8) / ts;
    (eta0 + d_eta) * 1e-6
}

// ---------------- Brine ----------------

/// NaCl mass fraction from molality.
fn salt_mass_fraction(m_nacl: f64) -> f64 {
    m_nacl * M_NACL / (1.0 + m_nacl * M_NACL)
}

/// Density (kg/m³) of NaCl brine with dissolved CO2 at molality `m_co2`.
/// IF97 water, Batzle & Wang (1992) salt term, Garcia (2001) CO2 apparent
/// molar volume.
///
/// # Errors
/// From [`water::liquid`] outside IF97 region 1.
pub fn brine_density(p_pa: f64, t_k: f64, m_nacl: f64, m_co2: f64) -> SxResult<f64> {
    let w = water::liquid(p_pa, t_k)?;
    let (s, pm, t) = (salt_mass_fraction(m_nacl), p_pa / 1e6, t_k - 273.15);
    let rho_b = w.density
        + 1000.0
            * s
            * (0.668
                + 0.44 * s
                + 1e-6
                    * (300.0 * pm - 2400.0 * pm * s
                        + t * (80.0 + 3.0 * t - 3300.0 * s - 13.0 * pm + 47.0 * pm * s)));
    let v_phi = (37.51 - 9.585e-2 * t + 8.740e-4 * t * t - 5.044e-7 * t.powi(3)) * 1e-6; // m³/mol
                                                                                         // per kg of salt-free water: mass and volume of brine, plus CO2
    let mass_b = 1.0 + m_nacl * M_NACL;
    Ok((mass_b + m_co2 * M_CO2) / (mass_b / rho_b + m_co2 * v_phi))
}

/// Viscosity (Pa·s) of NaCl brine with dissolved CO2 mole fraction `x_co2`:
/// Mao & Duan (2009) relative viscosity × Islam & Carlson (2012) CO2 factor.
///
/// # Errors
/// From [`water::liquid`] outside IF97 region 1.
pub fn brine_viscosity(p_pa: f64, t_k: f64, m_nacl: f64, x_co2: f64) -> SxResult<f64> {
    let w = water::liquid(p_pa, t_k)?;
    let t = t_k;
    let a = -0.213_192_13 + 0.136_515_89e-2 * t - 0.121_917_56e-5 * t * t;
    let b = 0.691_619_45e-1 - 0.272_922_63e-3 * t + 0.208_524_48e-6 * t * t;
    let c = -0.259_888_55e-2 + 0.779_892_27e-5 * t;
    let m = m_nacl;
    let mu_b = w.viscosity * (a * m + b * m * m + c * m.powi(3)).exp();
    Ok(mu_b * (1.0 + 4.65 * x_co2.max(0.0).powf(1.0134)))
}

/// Dissolution enthalpy of CO2(g) into water (J/mol, negative = exothermic),
/// van't Hoff from Spycher et al. (2003) log K0_CO2(g).
fn co2_dissolution_enthalpy(t_k: f64) -> f64 {
    let t_c = t_k - 273.15;
    let dlogk_dt = 1.304e-2 - 2.0 * 5.446e-5 * t_c;
    // K0 here is for CO2(g) -> CO2(aq) written as aq/g? Spycher K0 = f/(a):
    // gas from aqueous; ΔH(aq->g) = R T² d lnK/dT, so dissolution is minus.
    -R_SI * t_k * t_k * std::f64::consts::LN_10 * dlogk_dt
}

/// Specific enthalpy (J/kg) of the aqueous phase: IF97 water + dissolved
/// CO2 (ideal-gas enthalpy + dissolution enthalpy). Salt effect neglected.
///
/// # Errors
/// From [`water::liquid`].
pub fn brine_enthalpy(p_pa: f64, t_k: f64, m_co2: f64) -> SxResult<f64> {
    let w = water::liquid(p_pa, t_k)?;
    let h_co2 = co2_h_ideal(t_k) + co2_dissolution_enthalpy(t_k); // J/mol
    Ok((w.enthalpy + m_co2 * h_co2) / (1.0 + m_co2 * M_CO2))
}

// ---------------- Flash ----------------

/// Two-phase split result. Mole fractions are on a salt-free CO2/H2O basis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashResult {
    /// CO2-rich (gas/supercritical) phase mole fraction; outside [0, 1]
    /// for a single-phase state (negative flash, Whitson & Michelsen 1989).
    pub nu_gas_raw: f64,
    /// Gas phase mole fraction clipped to [0, 1].
    pub nu_gas: f64,
    /// CO2 mole fraction in the aqueous phase (equilibrium, or z if single aqueous).
    pub x_co2: f64,
    /// H2O mole fraction in the gas phase (equilibrium, or 1 − z if single gas).
    pub y_h2o: f64,
    /// Densities (kg/m³) of aqueous and gas phases.
    pub rho_aq: f64,
    pub rho_gas: f64,
    /// Viscosities (Pa·s).
    pub mu_aq: f64,
    pub mu_gas: f64,
    /// Specific enthalpies (J/kg).
    pub h_aq: f64,
    pub h_gas: f64,
    /// Gas saturation (volume fraction of the gas phase).
    pub s_gas: f64,
}

/// Negative two-phase flash of CO2/H2O at overall CO2 mole fraction `z_co2`
/// (salt-free basis) in brine of NaCl molality `m_nacl`. Since the binary
/// has fixed equilibrium compositions at (p, T, m), the split is the lever
/// rule. Gas-phase properties use pure CO2 correlations (water in the gas,
/// < 1 mol% here, is neglected in density/viscosity).
///
/// # Errors
/// Out-of-range state, `z_co2` outside [0, 1], or unimplemented model.
pub fn flash(
    p_pa: f64,
    t_k: f64,
    z_co2: f64,
    m_nacl: f64,
    model: SolubilityModel,
) -> SxResult<FlashResult> {
    if !(0.0..=1.0).contains(&z_co2) {
        return Err(SxError::InvalidInput(format!(
            "co2_brine: z_co2 must be in [0, 1], got {z_co2}"
        )));
    }
    let sol = mutual_solubility(p_pa, t_k, m_nacl, model)?;
    let y_co2 = 1.0 - sol.y_h2o;
    let nu_raw = (z_co2 - sol.x_co2) / (y_co2 - sol.x_co2);
    let nu = nu_raw.clamp(0.0, 1.0);
    let (x, y_h2o) = if nu_raw <= 0.0 {
        (z_co2, sol.y_h2o)
    } else if nu_raw >= 1.0 {
        (sol.x_co2, 1.0 - z_co2)
    } else {
        (sol.x_co2, sol.y_h2o)
    };
    let m_co2 = MOL_H2O_PER_KG * x / (1.0 - x);
    let rho_aq = brine_density(p_pa, t_k, m_nacl, m_co2)?;
    let rho_gas = co2_density(p_pa, t_k);
    let mu_aq = brine_viscosity(p_pa, t_k, m_nacl, x)?;
    let mu_gas = co2_viscosity(rho_gas, t_k);
    let h_aq = brine_enthalpy(p_pa, t_k, m_co2)?;
    let h_gas = co2_enthalpy(p_pa, t_k);
    // phase molar masses (salt carried by the aqueous phase)
    let mw_aq = x * M_CO2 + (1.0 - x) * (M_H2O + M_NACL * m_nacl * M_H2O);
    let mw_g = (1.0 - y_h2o) * M_CO2 + y_h2o * M_H2O;
    let v_aq = (1.0 - nu) * mw_aq / rho_aq;
    let v_g = nu * mw_g / rho_gas;
    let s_gas = if v_aq + v_g > 0.0 {
        v_g / (v_aq + v_g)
    } else {
        0.0
    };
    Ok(FlashResult {
        nu_gas_raw: nu_raw,
        nu_gas: nu,
        x_co2: x,
        y_h2o,
        rho_aq,
        rho_gas,
        mu_aq,
        mu_gas,
        h_aq,
        h_gas,
        s_gas,
    })
}
