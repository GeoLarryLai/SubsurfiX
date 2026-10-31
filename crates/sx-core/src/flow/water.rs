// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Liquid water properties from the IAPWS (International Association for
//! the Properties of Water and Steam) formulations.
//!
//! - Density, specific enthalpy and isobaric heat capacity: IAPWS-IF97
//!   (Industrial Formulation 1997) region 1, the dimensionless Gibbs free
//!   energy `γ(π, τ)` with `π = p / 16.53 MPa`, `τ = 1386 K / T`.
//! - Saturation pressure: IF97 region 4 (the region-1 bound `p ≥ p_sat(T)`).
//! - Viscosity: IAPWS 2008, without the critical enhancement.
//! - Thermal conductivity: IAPWS 2011, without the critical enhancement.
//!
//! Valid for compressed liquid with 273.15 K ≤ T ≤ 623.15 K and
//! `p_sat(T) ≤ p ≤ 100 MPa`; outside that range an error is returned.
//! Coefficients are the published values of the IAPWS releases.

use crate::error::{SxError, SxResult};

const R: f64 = 461.526; // J/(kg K), IF97 specific gas constant
const T_C: f64 = 647.096; // K, critical temperature
const RHO_C: f64 = 322.0; // kg/m³, critical density

/// IF97 region 1 (I, J, n).
const REGION1: [(i32, i32, f64); 34] = [
    (0, -2, 0.146_329_712_131_67),
    (0, -1, -0.845_481_871_691_14),
    (0, 0, -0.375_636_036_720_40e1),
    (0, 1, 0.338_551_691_683_85e1),
    (0, 2, -0.957_919_633_878_72),
    (0, 3, 0.157_720_385_132_28),
    (0, 4, -0.166_164_171_995_01e-1),
    (0, 5, 0.812_146_299_835_68e-3),
    (1, -9, 0.283_190_801_238_04e-3),
    (1, -7, -0.607_063_015_658_74e-3),
    (1, -1, -0.189_900_682_184_19e-1),
    (1, 0, -0.325_297_487_705_05e-1),
    (1, 1, -0.218_417_171_754_14e-1),
    (1, 3, -0.528_383_579_699_30e-4),
    (2, -3, -0.471_843_210_732_67e-3),
    (2, 0, -0.300_017_807_930_26e-3),
    (2, 1, 0.476_613_939_069_87e-4),
    (2, 3, -0.441_418_453_308_46e-5),
    (2, 17, -0.726_949_962_975_94e-15),
    (3, -4, -0.316_796_448_450_54e-4),
    (3, 0, -0.282_707_979_853_12e-5),
    (3, 6, -0.852_051_281_201_03e-9),
    (4, -5, -0.224_252_819_080_00e-5),
    (4, -2, -0.651_712_228_956_01e-6),
    (4, 10, -0.143_417_299_379_24e-12),
    (5, -8, -0.405_169_968_601_17e-6),
    (8, -11, -0.127_343_017_416_41e-8),
    (8, -6, -0.174_248_712_306_34e-9),
    (21, -29, -0.687_621_312_955_31e-18),
    (23, -31, 0.144_783_078_285_21e-19),
    (29, -38, 0.263_357_816_627_95e-22),
    (30, -39, -0.119_476_226_400_71e-22),
    (31, -40, 0.182_280_945_814_04e-23),
    (32, -41, -0.935_370_872_924_58e-25),
];

/// IF97 region 4 saturation-line coefficients n1..n10.
const REGION4: [f64; 10] = [
    0.116_705_214_527_67e4,
    -0.724_213_167_032_06e6,
    -0.170_738_469_400_92e2,
    0.120_208_247_024_70e5,
    -0.323_255_503_223_33e7,
    0.149_151_086_135_30e2,
    -0.482_326_573_615_91e4,
    0.405_113_405_420_57e6,
    -0.238_555_575_678_49,
    0.650_175_348_447_98e3,
];

/// Saturation pressure (Pa) at temperature `t_k` (273.15–647.096 K), IF97 region 4.
///
/// # Errors
/// `InvalidInput` outside 273.15–647.096 K.
pub fn saturation_pressure(t_k: f64) -> SxResult<f64> {
    if !(273.15..=T_C).contains(&t_k) {
        return Err(SxError::InvalidInput(format!(
            "water: saturation pressure needs 273.15 K <= T <= 647.096 K, got {t_k} K"
        )));
    }
    let n = &REGION4;
    let th = t_k + n[8] / (t_k - n[9]);
    let a = th * th + n[0] * th + n[1];
    let b = n[2] * th * th + n[3] * th + n[4];
    let c = n[5] * th * th + n[6] * th + n[7];
    let x = 2.0 * c / (-b + (b * b - 4.0 * a * c).sqrt());
    Ok(x.powi(4) * 1e6)
}

/// Liquid water state at one pressure and temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidWater {
    /// Density (kg/m³).
    pub density: f64,
    /// Specific enthalpy (J/kg).
    pub enthalpy: f64,
    /// Isobaric specific heat capacity (J/(kg K)).
    pub cp: f64,
    /// Isothermal compressibility (1/Pa).
    pub compressibility: f64,
    /// Dynamic viscosity (Pa s).
    pub viscosity: f64,
    /// Thermal conductivity (W/(m K)).
    pub conductivity: f64,
}

/// Region-1 Gibbs derivatives (γ_π, γ_τ, γ_ππ, γ_ττ).
fn gibbs1(pi: f64, tau: f64) -> [f64; 4] {
    let (a, b) = (7.1 - pi, tau - 1.222);
    let mut g = [0.0; 4];
    for &(i, j, n) in &REGION1 {
        let (fi, fj) = (f64::from(i), f64::from(j));
        let ai = a.powi(i);
        let bj = b.powi(j);
        // ∂/∂π (7.1 − π)^I = −I (7.1 − π)^(I−1)
        let dai = if i > 0 { -fi * a.powi(i - 1) } else { 0.0 };
        let ddai = if i > 1 { fi * (fi - 1.0) * a.powi(i - 2) } else { 0.0 };
        let dbj = if j != 0 { fj * b.powi(j - 1) } else { 0.0 };
        let ddbj = if j != 0 && j != 1 { fj * (fj - 1.0) * b.powi(j - 2) } else { 0.0 };
        g[0] += n * dai * bj;
        g[1] += n * ai * dbj;
        g[2] += n * ddai * bj;
        g[3] += n * ai * ddbj;
    }
    g
}

/// Liquid water properties at pressure `p_pa` (Pa) and temperature `t_k` (K).
///
/// # Errors
/// `InvalidInput` outside the IF97 region-1 range (273.15–623.15 K,
/// `p_sat(T)`–100 MPa) or for non-finite input.
pub fn liquid(p_pa: f64, t_k: f64) -> SxResult<LiquidWater> {
    if !(p_pa.is_finite() && t_k.is_finite()) || !(273.15..=623.15).contains(&t_k) || p_pa > 100e6 {
        return Err(SxError::InvalidInput(format!(
            "water: IF97 region 1 needs 273.15-623.15 K and p <= 100 MPa, got {t_k} K, {p_pa} Pa"
        )));
    }
    let psat = saturation_pressure(t_k)?;
    if p_pa < psat {
        return Err(SxError::InvalidInput(format!(
            "water: {p_pa} Pa is below the saturation pressure {psat:.1} Pa at {t_k} K (not liquid)"
        )));
    }
    let (pi, tau) = (p_pa / 16.53e6, 1386.0 / t_k);
    let [gp, gt, gpp, gtt] = gibbs1(pi, tau);
    let v = pi * gp * R * t_k / p_pa;
    let density = 1.0 / v;
    let enthalpy = tau * gt * R * t_k;
    let cp = -tau * tau * gtt * R;
    let compressibility = -pi * gpp / gp / p_pa;
    // Viscosity, IAPWS 2008 (μ0 · μ1), in Pa s.
    let (tr, dr) = (t_k / T_C, density / RHO_C);
    let h0 = [1.677_52, 2.204_62, 0.636_656_4, -0.241_605];
    let mu0 = 100.0 * tr.sqrt() / h0.iter().enumerate().map(|(i, h)| h / tr.powi(i as i32)).sum::<f64>();
    const HIJ: [(i32, i32, f64); 21] = [
        (0, 0, 0.520_094),
        (1, 0, 0.085_089_5),
        (2, 0, -1.083_74),
        (3, 0, -0.289_555),
        (0, 1, 0.222_531),
        (1, 1, 0.999_115),
        (2, 1, 1.887_97),
        (3, 1, 1.266_13),
        (5, 1, 0.120_573),
        (0, 2, -0.281_378),
        (1, 2, -0.906_851),
        (2, 2, -0.772_479),
        (3, 2, -0.489_837),
        (4, 2, -0.257_040),
        (0, 3, 0.161_913),
        (1, 3, 0.257_399),
        (0, 4, -0.032_537_2),
        (3, 4, 0.069_845_2),
        (4, 5, 0.008_721_02),
        (3, 6, -0.004_356_73),
        (5, 6, -0.000_593_264),
    ];
    let (x, y) = (1.0 / tr - 1.0, dr - 1.0);
    let mu1 = (dr * HIJ.iter().map(|&(i, j, h)| h * x.powi(i) * y.powi(j)).sum::<f64>()).exp();
    let viscosity = mu0 * mu1 * 1e-6;
    // Thermal conductivity, IAPWS 2011 (λ0 · λ1), in W/(m K).
    let l0 = [2.443_221e-3, 1.323_095e-2, 6.770_357e-3, -3.454_586e-3, 4.096_266e-4];
    let lam0 = tr.sqrt() / l0.iter().enumerate().map(|(i, l)| l / tr.powi(i as i32)).sum::<f64>();
    const LIJ: [[f64; 6]; 5] = [
        [1.603_973_57, -0.646_013_523, 0.111_443_906, 0.102_997_357, -0.050_412_363_4, 0.006_098_592_58],
        [2.337_718_42, -2.788_437_78, 1.536_161_67, -0.463_045_512, 0.083_282_701_9, -0.007_192_012_45],
        [2.196_505_29, -4.545_807_85, 3.557_772_44, -1.409_449_78, 0.275_418_278, -0.020_593_881_6],
        [-1.210_513_78, 1.608_129_89, -0.621_178_141, 0.071_637_322_4, 0.0, 0.0],
        [-2.720_337_0, 4.575_863_31, -3.183_692_45, 1.116_834_8, -0.192_683_05, 0.012_913_842],
    ];
    let mut s = 0.0;
    for (i, row) in LIJ.iter().enumerate() {
        for (j, l) in row.iter().enumerate() {
            s += l * x.powi(i as i32) * y.powi(j as i32);
        }
    }
    let conductivity = lam0 * (dr * s).exp() * 1e-3;
    Ok(LiquidWater {
        density,
        enthalpy,
        cp,
        compressibility,
        viscosity,
        conductivity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a / b - 1.0).abs()
    }

    #[test]
    fn if97_region1_verification_values() {
        // IF97 Table 5: v (m³/kg), h (kJ/kg), cp (kJ/(kg K)).
        for (t, p, v, h, cp) in [
            (300.0, 3e6, 0.100_215_168e-2, 0.115_331_273e3, 0.417_301_218e1),
            (300.0, 80e6, 0.971_180_894e-3, 0.184_142_828e3, 0.401_008_987e1),
            (500.0, 3e6, 0.120_241_800e-2, 0.975_542_239e3, 0.465_580_682e1),
        ] {
            let w = liquid(p, t).unwrap();
            assert!(rel(1.0 / w.density, v) < 1e-8, "v at {t} {p}");
            assert!(rel(w.enthalpy, h * 1e3) < 1e-8, "h at {t} {p}");
            assert!(rel(w.cp, cp * 1e3) < 1e-8, "cp at {t} {p}");
        }
        // Isothermal compressibility at 500 K, 3 MPa: 0.112892188e-2 1/MPa.
        assert!(rel(liquid(3e6, 500.0).unwrap().compressibility, 0.112_892_188e-8) < 1e-7);
    }

    #[test]
    fn saturation_pressure_verification_values() {
        // IF97 Table 35.
        for (t, p) in [(300.0, 0.353_658_941e-2), (500.0, 0.263_889_776e1), (600.0, 0.123_443_146e2)] {
            assert!(rel(saturation_pressure(t).unwrap(), p * 1e6) < 1e-8, "{t}");
        }
    }

    #[test]
    fn transport_property_checks() {
        // Viscosity (IAPWS 2008) and conductivity (IAPWS 2011) without the critical
        // enhancement, at IF97 densities; reference values from an independent implementation.
        let w = liquid(0.101_325e6, 298.15).unwrap();
        assert!(rel(w.density, 997.048_03) < 1e-7);
        assert!(rel(w.viscosity, 890.022_37e-6) < 1e-6, "{}", w.viscosity);
        assert!(rel(w.conductivity, 0.606_516_58) < 1e-6, "{}", w.conductivity);
        let w = liquid(10e6, 423.15).unwrap();
        assert!(rel(w.density, 922.318_76) < 1e-7);
        assert!(rel(w.viscosity, 185.021_82e-6) < 1e-6, "{}", w.viscosity);
        assert!(rel(w.conductivity, 0.687_436_37) < 1e-6, "{}", w.conductivity);
        assert!(liquid(1e5, 400.0).is_err(), "steam at 1 bar, 400 K");
        assert!(liquid(1e6, 700.0).is_err());
    }
}
