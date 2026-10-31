// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! **Experimental** physics for the OBL fully implicit engine
//! ([`super::implicit`]).
//!
//! * [`DeadOil`]: water and oil components, each in its own phase
//!   (immiscible, no mass transfer), Brooks–Corey/Corey relative
//!   permeabilities, optional capillary pressure, slightly compressible
//!   phases and rock. State `[p, z_w]` (water mass fraction).
//! * [`BlackOil`]: gas, oil and water components; gas dissolves in oil with
//!   the solution gas–oil ratio `Rs(p)` and formation volume factors
//!   `Bo(p)`, `Bg(p)` from a PVT table (black-oil model, Aziz & Settari
//!   1979; operator form after Khait & Voskov 2017). State `[p, z_g, z_o]`
//!   (mass fractions), phases `[gas, oil, water]`.
//!
//! Mass basis throughout: with total mass 1 and phase volumes `V_p`, the
//! accumulation operator of component `c` is `α_c = (1 + c_r (p − p_ref)) z_c / Σ V_p`
//! (kg per m³ of reference pore volume) and the flux operator is
//! `β_{c,p} = x_{c,p} ρ_p k_{r,p} / μ_p`.

use super::implicit::{OblPhysics, OpLayout};
use super::multiphase::Capillary;
use super::transport::Corey;

/// Water–oil, dead-oil style (immiscible) physics.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadOil {
    /// Relative permeability end points, exponents and viscosities.
    pub corey: Corey,
    /// Capillary pressure `Pc = p_o − p_w` as a function of Sw.
    pub capillary: Capillary,
    /// Water density at `p_ref` (kg/m³).
    pub rho_w: f64,
    /// Oil density at `p_ref` (kg/m³).
    pub rho_o: f64,
    /// Water compressibility (1/Pa).
    pub c_w: f64,
    /// Oil compressibility (1/Pa).
    pub c_o: f64,
    /// Rock compressibility (1/Pa).
    pub c_rock: f64,
    /// Reference pressure (Pa).
    pub p_ref: f64,
}

impl Default for DeadOil {
    fn default() -> Self {
        Self {
            corey: Corey::default(),
            capillary: Capillary::None,
            rho_w: 1000.0,
            rho_o: 800.0,
            c_w: 0.0,
            c_o: 0.0,
            c_rock: 0.0,
            p_ref: 1e5,
        }
    }
}

impl DeadOil {
    fn dens(&self, p: f64) -> (f64, f64) {
        let dp = p - self.p_ref;
        (self.rho_w * (1.0 + self.c_w * dp), self.rho_o * (1.0 + self.c_o * dp))
    }

    /// Water mass fraction giving water saturation `sw` at pressure `p`.
    pub fn z_from_sw(&self, p: f64, sw: f64) -> f64 {
        let (rw, ro) = self.dens(p);
        sw * rw / (sw * rw + (1.0 - sw) * ro)
    }
}

impl OblPhysics for DeadOil {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_phases(&self) -> usize {
        2
    }
    fn phase_names(&self) -> Vec<String> {
        vec!["water".into(), "oil".into()]
    }
    fn component_names(&self) -> Vec<String> {
        vec!["water".into(), "oil".into()]
    }
    fn saturations(&self, s: &[f64]) -> Vec<f64> {
        let (rw, ro) = self.dens(s[0]);
        let z = s[1].clamp(0.0, 1.0);
        let (vw, vo) = (z / rw, (1.0 - z) / ro);
        let sw = vw / (vw + vo);
        vec![sw, 1.0 - sw]
    }
    fn evaluate(&self, s: &[f64], out: &mut [f64]) {
        let lay = OpLayout { ne: 2, np: 2, thermal: false };
        let p = s[0];
        let z = s[1].clamp(0.0, 1.0);
        let (rw, ro) = self.dens(p);
        let (vw, vo) = (z / rw, (1.0 - z) / ro);
        let vt = vw + vo;
        let sw = vw / vt;
        let phi = 1.0 + self.c_rock * (p - self.p_ref);
        let (lw, lo) = self.corey.mobility(sw);
        out.iter_mut().for_each(|v| *v = 0.0);
        out[lay.alpha(0)] = phi * z / vt;
        out[lay.alpha(1)] = phi * (1.0 - z) / vt;
        out[lay.beta(0, 0)] = rw * lw;
        out[lay.beta(1, 1)] = ro * lo;
        out[lay.rho(0)] = rw;
        out[lay.rho(1)] = ro;
        out[lay.pc(0)] = self.capillary.pc(&self.corey, sw);
        out[lay.pc(1)] = 0.0;
        out[lay.lambda(0)] = lw;
        out[lay.lambda(1)] = lo;
    }
}

/// One black-oil PVT row: pressure (Pa), saturated solution gas–oil ratio
/// Rs (sm³/sm³), oil formation volume factor Bo (rm³/sm³), oil viscosity
/// (Pa·s), gas formation volume factor Bg (rm³/sm³), gas viscosity (Pa·s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PvtRow {
    /// Pressure (Pa).
    pub p: f64,
    /// Rs (sm³/sm³).
    pub rs: f64,
    /// Bo (rm³/sm³).
    pub bo: f64,
    /// μo (Pa·s).
    pub mu_o: f64,
    /// Bg (rm³/sm³).
    pub bg: f64,
    /// μg (Pa·s).
    pub mu_g: f64,
}

/// Black-oil physics with dissolved gas (live oil, no vaporized oil).
///
/// Relative permeabilities: Corey per phase on normalized saturations,
/// `krw(Sw)`, `krg(Sg)` and `kro(So)` (a simple saturation-weighted model,
/// not Stone's). No capillary pressure. Undersaturated oil takes Bo and μo at its
/// bubble point with a constant undersaturated compressibility
/// `Bo = Bo(p_b) (1 − c_o (p − p_b))`.
#[derive(Debug, Clone, PartialEq)]
pub struct BlackOil {
    /// PVT table, increasing pressure.
    pub pvt: Vec<PvtRow>,
    /// Surface densities (kg/m³): gas, oil, water.
    pub rho_sc: [f64; 3],
    /// Water formation volume factor at `p_ref` and compressibility (1/Pa).
    pub bw: f64,
    /// Water compressibility (1/Pa).
    pub c_w: f64,
    /// Water viscosity (Pa·s).
    pub mu_w: f64,
    /// Residual saturations: gas, oil, water (connate).
    pub s_res: [f64; 3],
    /// Corey exponents: gas, oil, water.
    pub n_exp: [f64; 3],
    /// End-point relative permeabilities: gas, oil, water.
    pub kr_max: [f64; 3],
    /// Undersaturated oil compressibility (1/Pa) above the bubble point.
    pub c_o_undersat: f64,
    /// Rock compressibility (1/Pa).
    pub c_rock: f64,
    /// Reference pressure (Pa).
    pub p_ref: f64,
}

impl BlackOil {
    fn pvt_at(&self, p: f64) -> PvtRow {
        let t = &self.pvt;
        if p <= t[0].p {
            return t[0];
        }
        if p >= t[t.len() - 1].p {
            return t[t.len() - 1];
        }
        let k = t.partition_point(|r| r.p <= p).max(1);
        let (a, b) = (t[k - 1], t[k]);
        let f = (p - a.p) / (b.p - a.p);
        let l = |x: f64, y: f64| x + f * (y - x);
        PvtRow { p, rs: l(a.rs, b.rs), bo: l(a.bo, b.bo), mu_o: l(a.mu_o, b.mu_o), bg: l(a.bg, b.bg), mu_g: l(a.mu_g, b.mu_g) }
    }

    /// Phase split at `[p, zg, zo]`: (saturations [g, o, w], phase
    /// densities, dissolved-gas mass fraction in oil, viscosities, total volume).
    fn flash(&self, s: &[f64]) -> ([f64; 3], [f64; 3], f64, [f64; 3], f64) {
        let p = s[0];
        let zg = s[1].max(0.0);
        let zo = s[2].max(0.0);
        let zw = (1.0 - zg - zo).max(0.0);
        let r = self.pvt_at(p);
        let [rg_sc, ro_sc, rw_sc] = self.rho_sc;
        let rs_mass = r.rs * rg_sc / ro_sc; // kg gas / kg stock-tank oil, saturated
        let dissolved = zg.min(zo * rs_mass);
        let free = zg - dissolved;
        let rs_act = if zo > 0.0 { dissolved / zo * ro_sc / rg_sc } else { 0.0 };
        // Undersaturated oil: Bo and μo at the bubble point of its Rs, Bo
        // shrinking with the undersaturated compressibility above it.
        let (bo, mu_o) = if rs_act < r.rs * (1.0 - 1e-12) {
            let rb = self.pvt_at(self.bubble_point(rs_act));
            (rb.bo * (1.0 - self.c_o_undersat * (p - rb.p)).max(0.5), rb.mu_o)
        } else {
            (r.bo, r.mu_o)
        };
        let rho_o = (ro_sc + rs_act * rg_sc) / bo;
        let rho_g = rg_sc / r.bg;
        let rho_w = rw_sc / self.bw * (1.0 + self.c_w * (p - self.p_ref));
        let m_o = zo + dissolved;
        let (vg, vo, vw) = (free / rho_g, m_o / rho_o, zw / rho_w);
        let vt = (vg + vo + vw).max(1e-30);
        let xg_o = if m_o > 0.0 { dissolved / m_o } else { 0.0 };
        ([vg / vt, vo / vt, vw / vt], [rho_g, rho_o, rho_w], xg_o, [r.mu_g, mu_o, self.mu_w], vt)
    }

    /// Bubble-point pressure for solution ratio `rs` (inverse of the
    /// increasing Rs column, linear between rows).
    pub fn bubble_point(&self, rs: f64) -> f64 {
        let t = &self.pvt;
        if rs <= t[0].rs {
            return t[0].p;
        }
        for w in t.windows(2) {
            if rs <= w[1].rs {
                return w[0].p + (rs - w[0].rs) / (w[1].rs - w[0].rs).max(1e-300) * (w[1].p - w[0].p);
            }
        }
        t[t.len() - 1].p
    }

    fn kr(&self, sat: &[f64; 3]) -> [f64; 3] {
        let den = (1.0 - self.s_res.iter().sum::<f64>()).max(1e-9);
        let mut k = [0.0; 3];
        for i in 0..3 {
            let se = ((sat[i] - self.s_res[i]) / den).clamp(0.0, 1.0);
            k[i] = self.kr_max[i] * se.powf(self.n_exp[i]);
        }
        k
    }
}

impl OblPhysics for BlackOil {
    fn n_vars(&self) -> usize {
        3
    }
    fn n_phases(&self) -> usize {
        3
    }
    fn phase_names(&self) -> Vec<String> {
        vec!["gas".into(), "oil".into(), "water".into()]
    }
    fn component_names(&self) -> Vec<String> {
        vec!["gas".into(), "oil".into(), "water".into()]
    }
    fn saturations(&self, s: &[f64]) -> Vec<f64> {
        self.flash(s).0.to_vec()
    }
    fn evaluate(&self, s: &[f64], out: &mut [f64]) {
        let lay = OpLayout { ne: 3, np: 3, thermal: false };
        let (sat, rho, xg_o, mu, vt) = self.flash(s);
        let zg = s[1].max(0.0);
        let zo = s[2].max(0.0);
        let zw = (1.0 - zg - zo).max(0.0);
        let phi = 1.0 + self.c_rock * (s[0] - self.p_ref);
        let kr = self.kr(&sat);
        let lam = [kr[0] / mu[0], kr[1] / mu[1], kr[2] / mu[2]];
        out.iter_mut().for_each(|v| *v = 0.0);
        out[lay.alpha(0)] = phi * zg / vt;
        out[lay.alpha(1)] = phi * zo / vt;
        out[lay.alpha(2)] = phi * zw / vt;
        // gas component: free gas phase + dissolved in oil
        out[lay.beta(0, 0)] = rho[0] * lam[0];
        out[lay.beta(0, 1)] = xg_o * rho[1] * lam[1];
        out[lay.beta(1, 1)] = (1.0 - xg_o) * rho[1] * lam[1];
        out[lay.beta(2, 2)] = rho[2] * lam[2];
        for p in 0..3 {
            out[lay.rho(p)] = rho[p];
            out[lay.lambda(p)] = lam[p];
        }
    }
}

// ---------------- thermal water ----------------

/// Counts property evaluations that failed and used a fallback.
#[derive(Debug, Default)]
struct Fallbacks(std::sync::atomic::AtomicUsize);

impl Fallbacks {
    fn hit(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn get(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Rock thermal properties for thermal physics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RockThermal {
    /// Grain density (kg/m³).
    pub rho: f64,
    /// Heat capacity (J/(kg K)).
    pub cp: f64,
    /// Thermal conductivity (W/(m K)).
    pub conductivity: f64,
}

impl Default for RockThermal {
    /// Sandstone: 2650 kg/m³, 900 J/(kg K), 2.5 W/(m K).
    fn default() -> Self {
        Self { rho: 2650.0, cp: 900.0, conductivity: 2.5 }
    }
}

/// Rock internal energy reference temperature (K).
const T_REF_ROCK: f64 = 273.15;

/// **Experimental.** Single-phase liquid water with an energy equation:
/// state `[p, T]`, equations `[water mass, energy]`. Density, enthalpy,
/// viscosity and conductivity from IAPWS-IF97 region 1 / IAPWS 2011
/// ([`super::water::liquid`]); internal energy `U = h − p/ρ`. A state
/// outside region 1 (steam, T > 623.15 K) is evaluated at the nearest
/// liquid state and counted as a fallback.
#[derive(Debug, Default)]
pub struct ThermalWater {
    /// Rock thermal properties.
    pub rock: RockThermal,
    /// Rock compressibility (1/Pa).
    pub c_rock: f64,
    /// Reference pressure of `c_rock` (Pa).
    pub p_ref: f64,
    fallbacks: Fallbacks,
}

impl ThermalWater {
    /// Water with the given rock.
    pub fn new(rock: RockThermal, c_rock: f64, p_ref: f64) -> Self {
        Self { rock, c_rock, p_ref, fallbacks: Fallbacks::default() }
    }

    fn props(&self, p: f64, t: f64) -> super::water::LiquidWater {
        if let Ok(w) = super::water::liquid(p, t) {
            return w;
        }
        self.fallbacks.hit();
        let tc = t.clamp(273.16, 623.15);
        let ps = super::water::saturation_pressure(tc).unwrap_or(1e5);
        super::water::liquid(p.clamp(ps * 1.001 + 1.0, 1e8), tc)
            .unwrap_or(super::water::LiquidWater { density: 1000.0, enthalpy: 4180.0 * (tc - 273.15), cp: 4180.0, compressibility: 4.5e-10, viscosity: 1e-3, conductivity: 0.6 })
    }
}

impl OblPhysics for ThermalWater {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_phases(&self) -> usize {
        1
    }
    fn phase_names(&self) -> Vec<String> {
        vec!["water".into()]
    }
    fn component_names(&self) -> Vec<String> {
        vec!["water".into(), "energy".into()]
    }
    fn thermal(&self) -> bool {
        true
    }
    fn rock_conductivity(&self) -> f64 {
        self.rock.conductivity
    }
    fn n_fallbacks(&self) -> usize {
        self.fallbacks.get()
    }
    fn saturations(&self, _s: &[f64]) -> Vec<f64> {
        vec![1.0]
    }
    fn evaluate(&self, s: &[f64], out: &mut [f64]) {
        let lay = self.layout();
        let (p, t) = (s[0], s[1]);
        let w = self.props(p, t);
        let phi = 1.0 + self.c_rock * (p - self.p_ref);
        let u = w.enthalpy - p / w.density;
        out.iter_mut().for_each(|v| *v = 0.0);
        out[lay.alpha(0)] = phi * w.density;
        out[lay.alpha(1)] = phi * w.density * u;
        out[lay.beta(0, 0)] = w.density / w.viscosity;
        out[lay.beta(1, 0)] = w.density * w.enthalpy / w.viscosity;
        out[lay.rho(0)] = w.density;
        out[lay.lambda(0)] = 1.0 / w.viscosity;
        out[lay.rock_energy()] = self.rock.rho * self.rock.cp * (t - T_REF_ROCK);
        out[lay.fluid_cond()] = w.conductivity;
    }
}

// ---------------- CO2–brine ----------------

use super::co2_brine::{self, SolubilityModel, M_CO2, M_H2O, M_NACL};

/// **Experimental.** CO₂–brine OBL physics around [`co2_brine::flash`].
///
/// Components `[CO2, H2O]` (salt-free mass basis: NaCl of fixed molality
/// rides with the aqueous phase and is not a component), phases
/// `[aqueous, gas]` (CO₂-rich). State `[p, z_CO2]` (overall CO₂ mass
/// fraction on a salt-free basis), plus `T` (K) if `thermal`, else the
/// fixed `temperature`. The mass fraction is converted to the salt-free
/// mole fraction for the flash; the phase compositions it returns are
/// converted back to mass fractions and the split is the mass lever rule.
/// The aqueous mass operators use the salt-free aqueous density
/// `ρ_aq (1 + m_CO2 M_CO2)/(1 + m_CO2 M_CO2 + m_NaCl M_NaCl)`; gravity uses
/// the full brine density. Relative permeability: Corey per phase with
/// residual saturations; no capillary pressure. Energy (thermal): phase
/// internal energy `U = h − p/ρ` per salt-free kg with the flash enthalpies;
/// fluid conductivity `0.6 S_aq + 0.085 S_gas` W/(m K) (constants).
///
/// A failed flash (state outside the property models' range) is retried at
/// the nearest state inside 12–100 °C, 1–600 bar; if that fails too the
/// node gets fixed fallback values. Every fallback is counted
/// ([`OblPhysics::n_fallbacks`]).
#[derive(Debug)]
pub struct Co2Brine {
    /// NaCl molality (mol/kg H2O).
    pub m_nacl: f64,
    /// Solubility model (default Spycher & Pruess 2005).
    pub model: SolubilityModel,
    /// Energy equation (temperature as the last state variable).
    pub thermal: bool,
    /// Temperature (K) of an isothermal run.
    pub temperature: f64,
    /// Residual aqueous and gas saturations.
    pub s_res: [f64; 2],
    /// Corey exponents, aqueous and gas.
    pub n_exp: [f64; 2],
    /// End-point relative permeabilities, aqueous and gas.
    pub kr_max: [f64; 2],
    /// Rock compressibility (1/Pa) and its reference pressure (Pa).
    pub c_rock: f64,
    /// Reference pressure (Pa).
    pub p_ref: f64,
    /// Rock thermal properties (thermal only).
    pub rock: RockThermal,
    fallbacks: Fallbacks,
}

impl Default for Co2Brine {
    fn default() -> Self {
        Self {
            m_nacl: 0.0,
            model: SolubilityModel::SpycherPruess2005,
            thermal: false,
            temperature: 323.15,
            s_res: [0.2, 0.05],
            n_exp: [2.0, 2.0],
            kr_max: [1.0, 1.0],
            c_rock: 0.0,
            p_ref: 1e5,
            rock: RockThermal::default(),
            fallbacks: Fallbacks::default(),
        }
    }
}

/// Phase state of [`Co2Brine`] at one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Co2BrineState {
    /// Gas saturation.
    pub s_gas: f64,
    /// Gas phase mass fraction (salt-free).
    pub l_gas: f64,
    /// CO₂ mass fraction in the aqueous phase (salt-free).
    pub x_co2_mass: f64,
    /// CO₂ mass fraction in the gas phase.
    pub y_co2_mass: f64,
    /// Salt-free and full aqueous densities, gas density (kg/m³).
    pub rho_aq_sf: f64,
    /// Full brine density (kg/m³).
    pub rho_aq: f64,
    /// Gas density (kg/m³).
    pub rho_gas: f64,
    /// Viscosities (Pa·s).
    pub mu_aq: f64,
    /// Gas viscosity (Pa·s).
    pub mu_gas: f64,
    /// Enthalpies (J/kg, salt-free aqueous).
    pub h_aq: f64,
    /// Gas enthalpy (J/kg).
    pub h_gas: f64,
    /// Volume per kg of salt-free fluid (m³/kg).
    pub v_tot: f64,
}

/// Overall CO₂ mole fraction (salt-free) of a CO₂ mass fraction.
pub fn co2_mass_to_mole(z: f64) -> f64 {
    let (a, b) = (z / M_CO2, (1.0 - z) / M_H2O);
    a / (a + b)
}

fn co2_mole_to_mass(x: f64) -> f64 {
    x * M_CO2 / (x * M_CO2 + (1.0 - x) * M_H2O)
}

impl Co2Brine {
    /// Physics with the default (Spycher & Pruess 2005) solubility model.
    pub fn new(m_nacl: f64, model: SolubilityModel, thermal: bool, temperature: f64) -> Self {
        Self { m_nacl, model, thermal, temperature, ..Self::default() }
    }

    fn temp(&self, s: &[f64]) -> f64 {
        if self.thermal { s[2] } else { self.temperature }
    }

    /// Phase split and properties at `(p, z_CO2 mass, T)`; failures fall
    /// back as described on [`Co2Brine`] and are counted.
    pub fn state(&self, p: f64, z: f64, t: f64) -> Co2BrineState {
        let z = z.clamp(0.0, 1.0);
        let zm = co2_mass_to_mole(z);
        let f = match co2_brine::flash(p, t, zm, self.m_nacl, self.model) {
            Ok(f) => f,
            Err(_) => {
                self.fallbacks.hit();
                match co2_brine::flash(p.clamp(1e5, 6e7), t.clamp(285.15, 373.15), zm, self.m_nacl, self.model) {
                    Ok(f) => f,
                    Err(_) => co2_brine::FlashResult { nu_gas_raw: 0.0, nu_gas: 0.0, x_co2: zm.min(0.02), y_h2o: 0.005, rho_aq: 1000.0, rho_gas: 600.0, mu_aq: 5e-4, mu_gas: 5e-5, h_aq: 4180.0 * (t - 273.15), h_gas: 0.0, s_gas: 0.0 },
                }
            }
        };
        let x_m = co2_mole_to_mass(f.x_co2);
        let y_m = co2_mole_to_mass(1.0 - f.y_h2o);
        let l_gas = if f.nu_gas_raw <= 0.0 {
            0.0
        } else if f.nu_gas_raw >= 1.0 {
            1.0
        } else {
            ((z - x_m) / (y_m - x_m)).clamp(0.0, 1.0)
        };
        let m_co2 = 55.508 * f.x_co2 / (1.0 - f.x_co2).max(1e-12);
        let sf = (1.0 + m_co2 * M_CO2) / (1.0 + m_co2 * M_CO2 + self.m_nacl * M_NACL);
        let rho_aq_sf = f.rho_aq * sf;
        let (va, vg) = ((1.0 - l_gas) / rho_aq_sf, l_gas / f.rho_gas);
        let v_tot = (va + vg).max(1e-30);
        Co2BrineState {
            s_gas: vg / v_tot,
            l_gas,
            x_co2_mass: if l_gas < 1.0 { x_m } else { 0.0 },
            y_co2_mass: y_m,
            rho_aq_sf,
            rho_aq: f.rho_aq,
            rho_gas: f.rho_gas,
            mu_aq: f.mu_aq,
            mu_gas: f.mu_gas,
            h_aq: f.h_aq,
            h_gas: f.h_gas,
            v_tot,
        }
    }

    fn kr(&self, s_gas: f64) -> [f64; 2] {
        let den = (1.0 - self.s_res[0] - self.s_res[1]).max(1e-9);
        let sw = 1.0 - s_gas;
        let se_w = ((sw - self.s_res[0]) / den).clamp(0.0, 1.0);
        let se_g = ((s_gas - self.s_res[1]) / den).clamp(0.0, 1.0);
        [self.kr_max[0] * se_w.powf(self.n_exp[0]), self.kr_max[1] * se_g.powf(self.n_exp[1])]
    }
}

impl OblPhysics for Co2Brine {
    fn n_vars(&self) -> usize {
        if self.thermal { 3 } else { 2 }
    }
    fn n_phases(&self) -> usize {
        2
    }
    fn phase_names(&self) -> Vec<String> {
        vec!["aqueous".into(), "gas".into()]
    }
    fn component_names(&self) -> Vec<String> {
        let mut v = vec!["CO2".to_string(), "H2O".to_string()];
        if self.thermal {
            v.push("energy".into());
        }
        v
    }
    fn thermal(&self) -> bool {
        self.thermal
    }
    fn rock_conductivity(&self) -> f64 {
        self.rock.conductivity
    }
    fn n_fallbacks(&self) -> usize {
        self.fallbacks.get()
    }
    fn saturations(&self, s: &[f64]) -> Vec<f64> {
        let st = self.state(s[0], s[1], self.temp(s));
        vec![1.0 - st.s_gas, st.s_gas]
    }
    fn evaluate(&self, s: &[f64], out: &mut [f64]) {
        let lay = self.layout();
        let (p, z, t) = (s[0], s[1].clamp(0.0, 1.0), self.temp(s));
        let st = self.state(p, z, t);
        let phi = 1.0 + self.c_rock * (p - self.p_ref);
        let kr = self.kr(st.s_gas);
        let lam = [kr[0] / st.mu_aq, kr[1] / st.mu_gas];
        out.iter_mut().for_each(|v| *v = 0.0);
        out[lay.alpha(0)] = phi * z / st.v_tot;
        out[lay.alpha(1)] = phi * (1.0 - z) / st.v_tot;
        out[lay.beta(0, 0)] = st.x_co2_mass * st.rho_aq_sf * lam[0];
        out[lay.beta(1, 0)] = (1.0 - st.x_co2_mass) * st.rho_aq_sf * lam[0];
        out[lay.beta(0, 1)] = st.y_co2_mass * st.rho_gas * lam[1];
        out[lay.beta(1, 1)] = (1.0 - st.y_co2_mass) * st.rho_gas * lam[1];
        out[lay.rho(0)] = st.rho_aq;
        out[lay.rho(1)] = st.rho_gas;
        out[lay.lambda(0)] = lam[0];
        out[lay.lambda(1)] = lam[1];
        if self.thermal {
            let u_aq = st.h_aq - p / st.rho_aq_sf;
            let u_g = st.h_gas - p / st.rho_gas;
            out[lay.alpha(2)] = phi * ((1.0 - st.l_gas) * u_aq + st.l_gas * u_g) / st.v_tot;
            out[lay.beta(2, 0)] = st.h_aq * st.rho_aq_sf * lam[0];
            out[lay.beta(2, 1)] = st.h_gas * st.rho_gas * lam[1];
            out[lay.rock_energy()] = self.rock.rho * self.rock.cp * (t - T_REF_ROCK);
            out[lay.fluid_cond()] = 0.6 * (1.0 - st.s_gas) + 0.085 * st.s_gas;
        }
    }
}
