// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `subsurfix nphi-offset`.

use std::path::Path;

use sx_core::SxResult;

use crate::cli::resolve_config;

/// Evaluate the field-wide NPHI offset and store it in the project.
pub fn execute(config: &Path) -> SxResult<()> {
    let (config_path, project_dir) = resolve_config(config)?;
    let cfg = sx_core::config::load_project(&config_path)?;
    let r = sx_core::nphi_offset::evaluate_project(&project_dir, &cfg)?;
    let f = |x: Option<f64>| x.map_or("-".to_owned(), |v| format!("{v:.4}"));
    println!("input: {}", r.input);
    println!("wells: {}  samples: {}", r.n_wells, r.n_samples);
    println!(
        "field offset: {}  {:.0}% CI: {}",
        f(r.field_offset),
        100.0 * r.ci_level,
        r.ci.map_or("-".to_owned(), |(a, b)| format!("{a:.4} .. {b:.4}"))
    );
    if let Some(p) = r.field_pct {
        println!("P10/P50/P90: {:.4} / {:.4} / {:.4}", p.p10, p.p50, p.p90);
    }
    if let Some(t) = r.trend_phi {
        println!(
            "trend vs PHI_ND: intercept {:.4}, slope {:.3}",
            t.intercept, t.slope
        );
    }
    println!(
        "outlier wells: {}",
        r.wells.iter().filter(|w| w.outlier).count()
    );
    let (shift, source) = sx_core::nphi_offset::resolve_shift(
        cfg.normalize.nphi_shift,
        cfg.nphi_offset.apply,
        Some(&r),
    );
    println!(
        "status: {}  shift used by normalize: {shift:.4} ({})",
        r.status,
        source.name()
    );
    println!("wrote: {}", sx_core::nphi_offset::RESULT_JSON);
    Ok(())
}
