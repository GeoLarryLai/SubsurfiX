// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Seeded ChaCha8 helpers with a fixed bit recipe.
//!
//! Every draw uses [`ChaCha8Rng::seed_from_u64`] and `next_u64` only.

use rand::{Rng, SeedableRng};

use crate::hash::stream_seed;

/// ChaCha8 generator used for every seeded stream in the engine.
pub use rand_chacha::ChaCha8Rng;

/// Build the generator for one `(seed, stage, well)` stream.
///
/// `well_id` is the well's integer id, or `0` when the stream is not per well.
/// The generator is fully determined by [`crate::hash::stream_seed`].
///
/// NaN rule: none (no floating inputs). Empty input: none.
#[must_use]
pub fn rng_for(seed: u64, stage: &str, well_id: i32) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(stream_seed(seed, stage, well_id))
}

/// Uniform draw in `[0, 1)`.
///
/// Consumes one `next_u64`. NaN rule: none. Empty input: none.
#[must_use]
pub fn uniform_f64(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Uniform draw in `[lo, hi)` via [`uniform_f64`].
///
/// NaN rule: propagate (`lo` or `hi` non-finite follows IEEE arithmetic).
/// Empty input: none.
#[must_use]
pub fn uniform_range(rng: &mut ChaCha8Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * uniform_f64(rng)
}

/// Uniform integer in `[0, n)`.
///
/// `n == 0` returns `0` and still consumes one `next_u64`.
/// NaN rule: none. Empty input: none.
#[must_use]
pub fn uniform_below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    ((rng.next_u64() as u128 * u128::from(n)) >> 64) as u64
}

/// Uniform integer in `lo..=hi`.
///
/// Valid for `hi - lo < u64::MAX`. If `hi < lo`, returns `lo` and still
/// consumes one `next_u64`. NaN rule: none. Empty input: none.
#[must_use]
pub fn uniform_int_inclusive(rng: &mut ChaCha8Rng, lo: u64, hi: u64) -> u64 {
    if hi < lo {
        let _draw = rng.next_u64();
        return lo;
    }
    let span = hi.wrapping_sub(lo).wrapping_add(1);
    lo.wrapping_add(uniform_below(rng, span))
}

/// One standard-normal draw (Box–Muller, sine half discarded).
///
/// Consumes exactly two [`uniform_f64`] draws, `u1` then `u2`.
/// NaN rule: none (the bit recipe stays finite). Empty input: none.
#[must_use]
pub fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1 = uniform_f64(rng);
    let u2 = uniform_f64(rng);
    (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Fisher–Yates shuffle from the end.
///
/// Length 0 or 1 draws nothing. NaN rule: none. Empty input: no-op.
pub fn shuffle<T>(rng: &mut ChaCha8Rng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        let j = uniform_below(rng, (i + 1) as u64) as usize;
        v.swap(i, j);
    }
}
