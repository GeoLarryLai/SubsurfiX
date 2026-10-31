// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[nphi_offset]` settings: field-wide NPHI offset evaluation.

use serde::{Deserialize, Serialize};

/// Field-wide NPHI offset on clean, tight limestone, and whether to apply it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NphiOffsetConfig {
    /// subtract the evaluated offset from NPHI in normalize when an evaluation is stored
    pub apply: bool,
    /// clean rock: GR at or below (API)
    pub gr_max: f64,
    /// tight rock: neutron-density porosity (NPHI + DPHI_ls) / 2 at or below (v/v)
    pub phi_max: f64,
    /// limestone PEF window, lower bound, where PEF is measured (b/e)
    pub pef_min: f64,
    /// limestone PEF window, upper bound, where PEF is measured (b/e)
    pub pef_max: f64,
    /// reject samples without PEF
    pub require_pef: bool,
    /// matrix density of the density porosity (g/cm³)
    pub rho_matrix: f64,
    /// fluid density of the density porosity (g/cm³)
    pub rho_fluid: f64,
    /// qualifying samples a well needs to count (samples)
    pub min_well_samples: usize,
    /// qualifying wells needed for a field estimate (wells)
    pub min_wells: usize,
    /// bootstrap resamples of the wells
    pub n_boot: usize,
    /// two-sided confidence level of the interval
    pub ci_level: f64,
    /// well offsets further than this from the field offset are outliers (v/v)
    pub outlier_limit: f64,
    /// trend fit: clean limestone up to this neutron-density porosity (v/v)
    pub fit_phi_max: f64,
    /// trend fit: porosity bin width (v/v)
    pub fit_bin: f64,
    /// largest |slope| of (NPHI - DPHI_ls) vs neutron-density porosity accepted as a constant offset
    pub slope_tol: f64,
    /// bootstrap resamples for the slope interval
    pub n_boot_slope: usize,
    /// qualifying samples written for plots (samples)
    pub max_plot_samples: usize,
}

impl Default for NphiOffsetConfig {
    fn default() -> Self {
        Self {
            apply: true,
            gr_max: 30.0,
            phi_max: 0.05,
            pef_min: 4.0,
            pef_max: 6.0,
            require_pef: false,
            rho_matrix: 2.71,
            rho_fluid: 1.0,
            min_well_samples: 30,
            min_wells: 5,
            n_boot: 1000,
            ci_level: 0.9,
            outlier_limit: 0.04,
            fit_phi_max: 0.25,
            fit_bin: 0.01,
            slope_tol: 0.1,
            n_boot_slope: 200,
            max_plot_samples: 20000,
        }
    }
}
