// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Change-based adaptive time-step control.
//!
//! After each step the largest change of the controlled variable (e.g.
//! temperature or saturation) is compared with a target: the next step is
//! scaled by `safety · target / change`, bounded by `[shrink, growth]`, and a
//! step whose change exceeds `reject_factor · target` is repeated with the
//! smaller step. The step never drops below `dt_min`.

use crate::error::{SxError, SxResult};

/// Adaptive step controller settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepControl {
    /// First step (s).
    pub dt_init: f64,
    /// Smallest step (s); a rejected step below it is an error.
    pub dt_min: f64,
    /// Target largest change per step of the controlled variable.
    pub target_change: f64,
    /// Largest growth factor per step.
    pub growth: f64,
    /// Smallest shrink factor per step.
    pub shrink: f64,
    /// Safety factor on the predicted step.
    pub safety: f64,
    /// A step whose change exceeds `reject_factor · target_change` is repeated.
    pub reject_factor: f64,
}

impl StepControl {
    /// Controller with the given first step and target change, and default
    /// bounds (growth 2, shrink 0.25, safety 0.9, reject at 2× the target,
    /// `dt_min = dt_init · 1e-6`).
    #[must_use]
    pub fn new(dt_init: f64, target_change: f64) -> Self {
        Self {
            dt_init,
            dt_min: dt_init * 1e-6,
            target_change,
            growth: 2.0,
            shrink: 0.25,
            safety: 0.9,
            reject_factor: 2.0,
        }
    }

    /// Check the settings.
    ///
    /// # Errors
    /// `InvalidInput` for non-positive or inconsistent values.
    pub fn validate(&self) -> SxResult<()> {
        let pos = |v: f64| v > 0.0 && v.is_finite();
        if !(pos(self.dt_init)
            && pos(self.dt_min)
            && self.dt_min <= self.dt_init
            && pos(self.target_change)
            && self.growth >= 1.0
            && self.growth.is_finite()
            && self.shrink > 0.0
            && self.shrink < 1.0
            && self.safety > 0.0
            && self.safety <= 1.0
            && self.reject_factor >= 1.0
            && self.reject_factor.is_finite())
        {
            return Err(SxError::InvalidInput(
                "step control: need dt_init >= dt_min > 0, target_change > 0, growth >= 1, 0 < shrink < 1, 0 < safety <= 1, reject_factor >= 1"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Judge a step of length `dt` with largest change `change`: returns
    /// `(accepted, next_dt)`.
    ///
    /// # Errors
    /// `Numerical` when a rejected step would fall below `dt_min`.
    pub fn judge(&self, dt: f64, change: f64) -> SxResult<(bool, f64)> {
        let change = change.abs();
        let factor = if change > 0.0 {
            (self.safety * self.target_change / change).clamp(self.shrink, self.growth)
        } else {
            self.growth
        };
        let accept = change <= self.reject_factor * self.target_change;
        let next = dt * factor;
        if !accept && next < self.dt_min {
            return Err(SxError::Numerical(format!(
                "step control: step {next:.3e} s below dt_min {:.3e} s (change {change:.3e} per step)",
                self.dt_min
            )));
        }
        Ok((accept, next.max(self.dt_min)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_shrinks_and_rejects() {
        let c = StepControl::new(1.0, 1.0);
        assert_eq!(c.judge(1.0, 0.0).unwrap(), (true, 2.0));
        let (ok, dt) = c.judge(1.0, 0.9).unwrap();
        assert!(ok && (dt - 1.0).abs() < 1e-12);
        let (ok, dt) = c.judge(1.0, 3.0).unwrap();
        assert!(!ok && (dt - 0.3).abs() < 1e-12);
        assert!(c.judge(1e-6, 100.0).is_err());
        assert!(StepControl { shrink: 1.0, ..c }.validate().is_err());
    }
}
