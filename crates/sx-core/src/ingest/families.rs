// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Curve families. Plain string tests, no regex.

use crate::types::LogKind;

fn starts_any(m: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| m.starts_with(p))
}

fn contains_any(m: &str, parts: &[&str]) -> bool {
    parts.iter().any(|p| m.contains(p))
}

fn ends_any(m: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|s| m.ends_with(s))
}

/// `RESD` family of an upper-case mnemonic.
fn resd_family(m: &str) -> &'static str {
    let b = m.as_bytes();
    let is_laterolog = starts_any(m, &["LL", "HLL", "RLLD", "DLL"])
        || contains_any(m, &["LATEROLOG", "_LL"])
        || (m.starts_with("RLA") && b.get(3).is_some_and(u8::is_ascii_digit));
    if is_laterolog {
        return "laterolog";
    }
    // `E` followed by two ASCII digits, nothing else (free choice: exact form).
    let is_e_digits = b.len() == 3
        && b.first() == Some(&b'E')
        && b.get(1).is_some_and(u8::is_ascii_digit)
        && b.get(2).is_some_and(u8::is_ascii_digit);
    let is_lateral = starts_any(m, &["LAT", "ES_"])
        || contains_any(m, &["LATERAL", "18F8"])
        || ends_any(m, &["LN", "SN"])
        || m == "LS"
        || is_e_digits;
    if is_lateral {
        return "lateral";
    }
    let is_at_pattern = b.len() >= 4
        && b.first() == Some(&b'A')
        && b.get(1)
            .is_some_and(|c| matches!(c, b'T' | b'H' | b'O' | b'F'))
        && b.get(2).is_some_and(u8::is_ascii_digit)
        && b.get(3) == Some(&b'0');
    let is_induction = starts_any(
        m,
        &[
            "ILD", "RILD", "IL", "AHT", "M2R", "AIT", "RT9", "IDPH", "DIL",
        ],
    ) || m.contains("INDUCT")
        || m == "ID"
        || is_at_pattern;
    if is_induction {
        return "induction";
    }
    "unknown"
}

/// Family label of a curve mnemonic for `log`. Case-insensitive.
///
/// Empty, `NAN` or `NONE` give `"unknown"`. Labels: GR `cgr sgr grn mwd gr`;
/// RESD `laterolog lateral induction unknown`; NPHI `ss ls unknown`; DTC
/// `lwd wireline`; RHOB and PEF `unknown`. The first matching rule wins (for
/// RESD in the order laterolog, lateral, induction).
pub fn curve_family(log: LogKind, mnemonic: &str) -> &'static str {
    let m = mnemonic.trim().to_ascii_uppercase();
    if m.is_empty() || m == "NAN" || m == "NONE" {
        return "unknown";
    }
    let m = m.as_str();
    match log {
        LogKind::Gr => {
            if m.contains("CGR") {
                "cgr"
            } else if starts_any(m, &["SGR", "HSGR", "GRK"]) {
                "sgr"
            } else if m.starts_with("GRN") {
                "grn"
            } else if contains_any(m, &["MWD", "LWD"]) {
                "mwd"
            } else {
                "gr"
            }
        }
        LogKind::Resd => resd_family(m),
        LogKind::Nphi => {
            if m.ends_with("SS") || contains_any(m, &["_SS", "SAN"]) {
                "ss"
            } else if m.ends_with("LS") || contains_any(m, &["_LS", "LIM", "NPLS"]) {
                "ls"
            } else {
                "unknown"
            }
        }
        LogKind::Dtc => {
            if contains_any(m, &["MWD", "LWD"]) {
                "lwd"
            } else {
                "wireline"
            }
        }
        LogKind::Rhob | LogKind::Pef => "unknown",
    }
}

/// RESD family code: unknown 0, induction 1, laterolog 2, lateral 3; other
/// labels 0.
pub fn resd_family_code(family: &str) -> u8 {
    match family {
        "induction" => 1,
        "laterolog" => 2,
        "lateral" => 3,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_examples() {
        let rows = [
            (LogKind::Resd, "LLD", "laterolog"),
            (LogKind::Resd, "RLA5", "laterolog"),
            (LogKind::Resd, "ILD", "induction"),
            (LogKind::Resd, "AO90", "induction"),
            (LogKind::Resd, "AHT90", "induction"),
            (LogKind::Resd, "M2R9", "induction"),
            (LogKind::Resd, "RD", "unknown"),
            (LogKind::Nphi, "NPSS", "ss"),
            (LogKind::Nphi, "CNCLS", "ls"),
            (LogKind::Nphi, "TNPH_LIM", "ls"),
            (LogKind::Nphi, "TNPH", "unknown"),
            (LogKind::Gr, "GRN", "grn"),
            (LogKind::Gr, "GRD", "gr"),
            (LogKind::Dtc, "DT", "wireline"),
        ];
        for (log, m, fam) in rows {
            assert_eq!(curve_family(log, m), fam, "{log:?} {m}");
            assert_eq!(curve_family(log, &m.to_ascii_lowercase()), fam);
        }
    }

    #[test]
    fn other_rules() {
        assert_eq!(curve_family(LogKind::Gr, "ECGR"), "cgr");
        assert_eq!(curve_family(LogKind::Gr, "SGR"), "sgr");
        assert_eq!(curve_family(LogKind::Gr, "GRK"), "sgr");
        assert_eq!(curve_family(LogKind::Gr, "LWD_GR"), "mwd");
        assert_eq!(curve_family(LogKind::Gr, "GR_MWD"), "mwd");
        assert_eq!(curve_family(LogKind::Dtc, "DT_LWD"), "lwd");
        assert_eq!(curve_family(LogKind::Rhob, "RHOB"), "unknown");
        assert_eq!(curve_family(LogKind::Pef, "PEF"), "unknown");
        assert_eq!(curve_family(LogKind::Resd, "RLLD"), "laterolog");
        assert_eq!(curve_family(LogKind::Resd, "RES_LL"), "laterolog");
        assert_eq!(curve_family(LogKind::Resd, "RLAX"), "unknown");
        assert_eq!(curve_family(LogKind::Resd, "LAT"), "lateral");
        assert_eq!(curve_family(LogKind::Resd, "ES_16"), "lateral");
        assert_eq!(curve_family(LogKind::Resd, "R18F8"), "lateral");
        assert_eq!(curve_family(LogKind::Resd, "R64LN"), "lateral");
        assert_eq!(curve_family(LogKind::Resd, "LS"), "lateral");
        assert_eq!(curve_family(LogKind::Resd, "E16"), "lateral");
        assert_eq!(curve_family(LogKind::Resd, "ID"), "induction");
        assert_eq!(curve_family(LogKind::Resd, "DEEP_INDUCTION"), "induction");
        assert_eq!(curve_family(LogKind::Resd, "AF90"), "induction");
        assert_eq!(curve_family(LogKind::Resd, "AT11"), "unknown");
        assert_eq!(curve_family(LogKind::Nphi, "TNPH_SAN"), "ss");
        assert_eq!(curve_family(LogKind::Nphi, "NPLS"), "ls");
        for log in LogKind::ALL {
            for m in ["", "  ", "nan", "NONE"] {
                assert_eq!(curve_family(log, m), "unknown");
            }
        }
    }

    #[test]
    fn resd_codes() {
        assert_eq!(resd_family_code("unknown"), 0);
        assert_eq!(resd_family_code("induction"), 1);
        assert_eq!(resd_family_code("laterolog"), 2);
        assert_eq!(resd_family_code("lateral"), 3);
        assert_eq!(resd_family_code("ss"), 0);
        assert_eq!(resd_family_code(""), 0);
    }
}
