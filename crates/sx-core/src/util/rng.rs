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

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn chacha8_seed_zero_first_three_u64() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert_eq!(rng.next_u64(), 13_080_132_717_333_068_652);
        assert_eq!(rng.next_u64(), 8_594_738_769_458_413_623);
        assert_eq!(rng.next_u64(), 12_896_916_468_484_187_878);
    }

    #[test]
    fn uniform_f64_fresh_seed_zero() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert_eq!(uniform_f64(&mut rng), 0.7090754154265618);
    }

    #[test]
    fn stream_seed_cv_kfold_example() {
        assert_eq!(
            crate::hash::stream_seed(0, "cv_kfold", 0),
            14_598_824_572_957_819_918
        );
    }

    #[test]
    fn rng_for_example_draws() {
        let mut rng = rng_for(7, "test", 3);
        let u = uniform_f64(&mut rng);
        let below = uniform_below(&mut rng, 10);
        let z = standard_normal(&mut rng);
        assert_eq!(u, 0.29247540809159345);
        assert_eq!(below, 7);
        let diff = (z - (-0.30400902640991795)).abs();
        assert!(diff <= 1e-15, "got {z} diff {diff}");
    }

    #[test]
    fn shuffle_matches_kfold_example_order() {
        let mut ids: Vec<i32> = (0..10).collect();
        let mut rng = rng_for(0, "cv_kfold", 0);
        shuffle(&mut rng, &mut ids);
        assert_eq!(ids, vec![8, 6, 2, 5, 1, 9, 3, 0, 4, 7]);
    }

    #[test]
    fn uniform_range_uses_uniform_f64() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let got = uniform_range(&mut rng, -2.0, 4.0);
        let mut fresh = ChaCha8Rng::seed_from_u64(4);
        let expect = -2.0 + 6.0 * uniform_f64(&mut fresh);
        assert_eq!(got, expect);
    }

    #[test]
    fn uniform_below_zero_returns_zero_and_draws() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        assert_eq!(uniform_below(&mut rng, 0), 0);
        let mut fresh = ChaCha8Rng::seed_from_u64(11);
        let _draw = fresh.next_u64();
        assert_eq!(rng.next_u64(), fresh.next_u64());
    }

    #[test]
    fn uniform_int_inclusive_matches_below_and_inverted_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let got = uniform_int_inclusive(&mut rng, 3, 12);
        let mut fresh = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(got, 3 + uniform_below(&mut fresh, 10));

        let mut rng = ChaCha8Rng::seed_from_u64(11);
        assert_eq!(uniform_int_inclusive(&mut rng, 5, 3), 5);
        let mut fresh = ChaCha8Rng::seed_from_u64(11);
        let _draw = fresh.next_u64();
        assert_eq!(rng.next_u64(), fresh.next_u64());
    }

    #[test]
    fn uniform_int_inclusive_large_span_does_not_panic() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let v = uniform_int_inclusive(&mut rng, 1, u64::MAX);
        assert!(v >= 1);
    }

    #[test]
    fn standard_normal_draws_twice() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let _z = standard_normal(&mut rng);
        let mut fresh = ChaCha8Rng::seed_from_u64(5);
        let _u1 = uniform_f64(&mut fresh);
        let _u2 = uniform_f64(&mut fresh);
        assert_eq!(rng.next_u64(), fresh.next_u64());
    }

    #[test]
    fn shuffle_empty_and_singleton_draw_nothing() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut empty: [u32; 0] = [];
        shuffle(&mut rng, &mut empty);
        let mut one = [9_u32];
        shuffle(&mut rng, &mut one);
        assert_eq!(one, [9]);
        let mut fresh = ChaCha8Rng::seed_from_u64(11);
        assert_eq!(rng.next_u64(), fresh.next_u64());
    }
}
