// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Stable hashing for seeds and stream ids.
//!
//! Uses 64-bit FNV-1a, which is fixed across platforms and Rust versions
//! (unlike `std::hash`).

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over the concatenation of `parts`, with a `0xff` byte between parts
/// so that `["ab", "c"]` and `["a", "bc"]` differ.
pub fn stable_hash(parts: &[&[u8]]) -> u64 {
    let mut h = FNV_OFFSET;
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            h ^= 0xff;
            h = h.wrapping_mul(FNV_PRIME);
        }
        for &b in *part {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Seed for one random stream: `seed ^ stable_hash([stage, well_id as little-endian i32])`.
pub fn stream_seed(seed: u64, stage: &str, well_id: i32) -> u64 {
    seed ^ stable_hash(&[stage.as_bytes(), &well_id.to_le_bytes()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_offset() {
        assert_eq!(stable_hash(&[]), FNV_OFFSET);
    }

    #[test]
    fn known_value_for_a() {
        assert_eq!(stable_hash(&[b"a"]), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn separator_matters() {
        assert_ne!(stable_hash(&[b"ab", b"c"]), stable_hash(&[b"a", b"bc"]));
    }
}
