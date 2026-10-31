// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! UTC text from whole seconds since the epoch, without a date crate.
//!
//! Days-to-civil conversion on the proleptic Gregorian calendar
//! (era / day-of-era arithmetic).

use std::time::{SystemTime, UNIX_EPOCH};

/// `(year, month, day, hour, minute, second)` of `secs` seconds after the epoch.
fn civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX / 2);
    let rem = secs % 86_400;
    let (hh, mm, ss) = (
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    );
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day, hh, mm, ss)
}

/// `"YYYY-MM-DDTHH:MM:SSZ"`, e.g. `"2026-10-18T01:59:00Z"`.
pub(crate) fn utc_text(secs_since_epoch: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs_since_epoch);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `"YYYYMMDDTHHMMSSZ"`, e.g. `"20261018T015900Z"`.
pub(crate) fn utc_compact(secs_since_epoch: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs_since_epoch);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

/// Whole seconds since the epoch now (0 if the clock is before the epoch).
pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
