// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Analytic reference solutions used by the validation tests.

use super::transport::Corey;

/// Thiem steady radial flow: the pressure rise `p(r) − p_w` away from a
/// producer of rate `q` (> 0): `q μ / (2π k h) ln(r / r_w)`.
pub fn thiem_drop(q: f64, mu: f64, k: f64, h: f64, r: f64, rw: f64) -> f64 {
    q * mu / (2.0 * std::f64::consts::PI * k * h) * (r / rw).ln()
}

/// Buckley-Leverett water saturation at dimensionless position `xd`
/// (x / L) and time `td` (injected pore volumes), initial saturation `swc`.
/// Uses the Welge tangent for the shock.
pub fn buckley_leverett(c: &Corey, xd: f64, td: f64) -> f64 {
    let (a, b) = (c.swc, 1.0 - c.sor);
    let n = 20000;
    let h = (b - a) / n as f64;
    let dfds = |s: f64| {
        (c.fw((s + 1e-6).min(b)) - c.fw((s - 1e-6).max(a)))
            / ((s + 1e-6).min(b) - (s - 1e-6).max(a))
    };
    // Shock saturation: tangent from (swc, 0).
    let mut s_shock = b;
    let mut best = 0.0;
    for i in 1..=n {
        let s = a + h * i as f64;
        let slope = c.fw(s) / (s - a);
        if slope > best {
            best = slope;
            s_shock = s;
        }
    }
    let front = td * best;
    if xd > front {
        return a;
    }
    // Rarefaction: find s in [s_shock, b] with td f'(s) = xd (f' decreasing there).
    let (mut lo, mut hi) = (s_shock, b);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if td * dfds(mid) > xd {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

/// Exponential integral E1(x), x > 0 (series below 1, continued fraction above).
#[allow(clippy::approx_constant)] // Euler-Mascheroni literal; the std constant is not stable on all toolchains
pub fn exp_int_e1(x: f64) -> f64 {
    if !(x > 0.0) {
        return f64::INFINITY;
    }
    if x < 1.0 {
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..200 {
            term *= -x / k as f64;
            let add = -term / k as f64;
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        -0.577_215_664_901_532_9 - x.ln() + sum
    } else {
        // Lentz continued fraction.
        let (mut b, mut c, mut d) = (x + 1.0, 1e300, 1.0 / (x + 1.0));
        let mut h = d;
        for i in 1..300 {
            let an = -((i * i) as f64);
            b += 2.0;
            d = 1.0 / (an * d + b);
            c = b + an / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// Theis / line-source drawdown (Pa) at radius `r` and time `t` for a well
/// producing `q` (m³/s) in an infinite slightly compressible reservoir:
/// `q μ / (4π k h) E1(r² φ μ c_t / (4 k t))`.
#[allow(clippy::too_many_arguments)]
pub fn theis_drawdown(q: f64, mu: f64, k: f64, h: f64, phi: f64, ct: f64, r: f64, t: f64) -> f64 {
    q * mu / (4.0 * std::f64::consts::PI * k * h)
        * exp_int_e1(r * r * phi * mu * ct / (4.0 * k * t))
}

/// `ln erfc(z)` for `z >= 0`, Chebyshev fit (fractional error < 1.2e-7
/// everywhere); the log form avoids underflow in [`ogata_banks`].
fn ln_erfc_nonneg(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    t.ln() - z * z + poly
}

/// Complementary error function (fractional error < 1.2e-7).
pub fn erfc(z: f64) -> f64 {
    if z >= 0.0 {
        ln_erfc_nonneg(z).exp()
    } else {
        2.0 - ln_erfc_nonneg(-z).exp()
    }
}

/// Ogata–Banks (1961): 1D advection–dispersion from a fixed value at
/// `x = 0` into a semi-infinite domain at zero, velocity `v` (m/s),
/// dispersion `d` (m²/s). Returns the normalised value (0..1) at `x`, `t`.
pub fn ogata_banks(x: f64, t: f64, v: f64, d: f64) -> f64 {
    if t <= 0.0 {
        return if x <= 0.0 { 1.0 } else { 0.0 };
    }
    let s = 2.0 * (d * t).sqrt();
    let first = erfc((x - v * t) / s);
    let second = (v * x / d + ln_erfc_nonneg((x + v * t) / s)).exp();
    0.5 * (first + second)
}

/// Semi-infinite pure conduction from a step to `1` at `x = 0`:
/// `erfc(x / (2 √(α t)))`, diffusivity `alpha` (m²/s).
pub fn conduction_step(x: f64, t: f64, alpha: f64) -> f64 {
    erfc(x / (2.0 * (alpha * t).sqrt()))
}
