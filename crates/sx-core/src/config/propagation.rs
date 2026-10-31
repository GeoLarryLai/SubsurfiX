// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[propagation]` settings of `ml_propagation`.

use serde::{Deserialize, Serialize};

/// `ml_propagation` model and quantiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PropagationConfig {
    /// model: "qrf" (quantile regression forest, default) or "lightgbm" (quantile LightGBM)
    pub method: String,
    /// predicted quantile levels, each in (0, 1)
    pub quantiles: Vec<f64>,
    /// QRF trees
    pub qrf_trees: usize,
    /// QRF minimum rows per leaf
    pub qrf_min_leaf: usize,
    /// LightGBM boosting rounds
    pub lgbm_iterations: usize,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            method: "qrf".to_string(),
            quantiles: vec![0.1, 0.5, 0.9],
            qrf_trees: 200,
            qrf_min_leaf: 10,
            lgbm_iterations: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{from_toml_str, template_toml, validate};
    use crate::sim::{PropagationModel, PropagationParams};

    #[test]
    fn template_default_is_qrf_and_method_key_switches() {
        let cfg = from_toml_str(&template_toml()).unwrap();
        let p = PropagationParams::from_config(&cfg.propagation, cfg.seed).unwrap();
        assert_eq!(p.model, PropagationModel::QuantileForest);
        let t = template_toml().replace("method = \"qrf\"", "method = \"lightgbm\"");
        let cfg = from_toml_str(&t).unwrap();
        assert_eq!(
            PropagationParams::from_config(&cfg.propagation, 1)
                .unwrap()
                .model,
            PropagationModel::LightGbm
        );
        let mut bad = cfg.clone();
        bad.propagation.method = "svm".into();
        assert!(validate(&bad)
            .unwrap_err()
            .to_string()
            .contains("propagation.method"));
        bad.propagation.method = "qrf".into();
        bad.propagation.quantiles = vec![1.5];
        assert!(validate(&bad)
            .unwrap_err()
            .to_string()
            .contains("propagation.quantiles"));
    }
}
