// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Alias tables: built-in priority lists, dictionary build, extras, exclusions.
//!
//! No regular expressions: every rule is a plain string test on the upper-case
//! mnemonic `m`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::error::{SxError, SxResult};
use crate::io::atomic::io_error;
use crate::types::LogKind;

/// Per-log ordered alias lists; index 0 = highest priority. Entries upper case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasTable {
    lists: BTreeMap<LogKind, Vec<String>>,
}

/// Substrings that mark a mnemonic as a derived curve.
const DERIVED_MARKS: [&str; 18] = [
    "NRM", "CLINE", "EDIT", "SHIFT", "SMOOTH", "GARDNER", "DENSHIFT", "LOG10", "_XS", "_OH", "_CH",
    "CHI", "ORIG", "BEST", "_NEW", "_REP", "SLDT", "SLDS",
];

/// Substrings that mark a mnemonic as low priority.
const LOW_MARKS: [&str; 4] = ["DIGITIZED", "MERGED", "CORR", "RE25"];

/// Substrings that remove a `DENSITY` row.
const DENSITY_DROP_CONTAINS: [&str; 17] = [
    "POR", "PHI", "DPH", "DPRS", "DPRD", "DPRL", "DPLS", "DPSS", "DPDL", "DLIM", "UPHI", "HDPH",
    "EDPH", "DPO", "DPL", "PDD", "ZDNC",
];

/// Preferred head entries per log.
fn preferred_heads(log: LogKind) -> &'static [&'static str] {
    match log {
        LogKind::Gr => &[
            "GR", "GRD", "GRN", "GRS", "GRR", "GR2", "GR1", "GAMMA", "GRM",
        ],
        LogKind::Rhob => &[
            "RHOB", "RHOZ", "ZDEN", "DEN", "RHOB1", "RHOM", "RHOS", "RHO8",
        ],
        LogKind::Nphi => &[
            "NPHI", "NPLS", "CNPHI", "SNP", "TNPH", "NPOR", "NPHI1", "CNC", "CNL", "CNPOR",
        ],
        LogKind::Dtc => &["DTC", "DT", "DTCO", "DT2", "DT1", "DTCO1"],
        LogKind::Resd => &[
            "RESD", "ILD", "LLD", "AT90", "HLLD", "RD", "RLA5", "ILD1", "AF90", "RT90", "RT",
            "RES_DEEP", "RILD",
        ],
        LogKind::Pef => &["PEF", "PE", "PEFZ", "PEFL", "PEF8"],
    }
}

fn starts_any(m: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| m.starts_with(p))
}

fn contains_any(m: &str, parts: &[&str]) -> bool {
    parts.iter().any(|p| m.contains(p))
}

fn is_any(m: &str, words: &[&str]) -> bool {
    words.contains(&m)
}

/// Derived-curve flag of an upper-case mnemonic.
pub(crate) fn is_derived(m: &str) -> bool {
    contains_any(m, &DERIVED_MARKS)
}

/// Low-priority flag of an upper-case mnemonic.
pub(crate) fn is_low(m: &str) -> bool {
    contains_any(m, &LOW_MARKS)
}

/// `NEUT` followed by an optional `_` and one or more ASCII digits.
fn is_neut_digits(m: &str) -> bool {
    match m.strip_prefix("NEUT") {
        Some(rest) => {
            let digits = rest.strip_prefix('_').unwrap_or(rest);
            !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        }
        None => false,
    }
}

/// True when the upper-case mnemonic can never be this log.
///
/// Spectral, computed and component curves are not total GR; count rates and
/// generic porosity are not neutron porosity; sonic porosity, travel time,
/// velocity and shear are not compressional slowness; shallow or unspecified
/// resistivity is not deep. NaN rule: not applicable (text).
pub(crate) fn is_excluded(log: LogKind, m: &str) -> bool {
    match log {
        LogKind::Gr => {
            if is_any(m, &["CGR", "ECGR", "HCGR", "SGR", "HSGR"])
                || starts_any(m, &["SGR", "HSGR", "CGR", "ECGR", "HCGR", "MCGR", "GK"])
                || contains_any(m, &["POTA", "URAN"])
            {
                return true;
            }
            let rest = ["GRN_", "GR_", "GR"]
                .iter()
                .find_map(|p| m.strip_prefix(p))
                .unwrap_or(m);
            starts_any(rest, &["K", "TH", "UR", "UT"])
        }
        LogKind::Nphi => {
            starts_any(m, &["PHI", "TPHI", "NCNL", "FCNL", "NRAT"])
                || contains_any(m, &["PHIT", "PHIE"])
                || is_any(
                    m,
                    &[
                        "NEUT",
                        "NEUTRON",
                        "NEUT_DIGITIZED",
                        "NEAR",
                        "FAR",
                        "NRAW",
                        "NCR",
                        "POROSITY",
                    ],
                )
                || is_neut_digits(m)
        }
        LogKind::Dtc => {
            contains_any(
                m,
                &["DTS", "DT4S", "DTST", "DTTS", "SHEAR", "STON", "PHI", "POR"],
            ) || m.starts_with("SPH")
                || is_any(m, &["SPLS", "SPDL", "TT", "ITT", "ITTT", "INTT", "VP"])
                || (m.starts_with("DT") && m.ends_with('S'))
        }
        LogKind::Rhob => {
            m.starts_with("DRH")
                || contains_any(
                    m,
                    &[
                        "DRHO", "HDRA", "ZCOR", "DCOR", "RHG", "RHOG", "RHOF", "RHOMA",
                    ],
                )
                || is_any(m, &["CORR", "AIPD"])
        }
        LogKind::Resd => {
            starts_any(m, &["CI", "CND", "COND"])
                || m.contains("MCGRD")
                || is_any(
                    m,
                    &["LAT", "LS", "LL", "LS1", "LL7", "GUARD", "LGUARD", "RES"],
                )
        }
        LogKind::Pef => false,
    }
}

impl AliasTable {
    /// The built-in lists of rule 4.2, exactly.
    pub fn builtin() -> AliasTable {
        let mut lists = BTreeMap::new();
        for (log, list) in [
            (LogKind::Gr, GR_LIST),
            (LogKind::Rhob, RHOB_LIST),
            (LogKind::Nphi, NPHI_LIST),
            (LogKind::Dtc, DTC_LIST),
            (LogKind::Resd, RESD_LIST),
            (LogKind::Pef, PEF_LIST),
        ] {
            lists.insert(log, list.iter().map(|s| (*s).to_string()).collect());
        }
        AliasTable { lists }
    }

    /// Build from dictionary rows `(category, mnemonic)` by the rules of 4.3.
    ///
    /// Both values are trimmed and upper-cased first. A log with no rows gets
    /// an empty list.
    ///
    /// # Errors
    /// `SxError::Format("alias dictionary has no usable rows")` when all six
    /// lists end up empty.
    pub fn from_dictionary(rows: &[(String, String)]) -> SxResult<AliasTable> {
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        // Per log: mnemonics in file order, de-duplicated.
        let mut per_log: BTreeMap<LogKind, Vec<String>> = BTreeMap::new();
        for (category, mnemonic) in rows {
            let category = category.trim().to_ascii_uppercase();
            let m = mnemonic.trim().to_ascii_uppercase();
            if category.is_empty() || m.is_empty() {
                continue;
            }
            if !seen.insert((category.clone(), m.clone())) {
                continue;
            }
            if category == "DPHI" {
                continue;
            }
            if category == "DENSITY"
                && (contains_any(&m, &DENSITY_DROP_CONTAINS) || m.starts_with("CDL"))
            {
                continue;
            }
            let log = match category.as_str() {
                "GR" => LogKind::Gr,
                "DENSITY" => LogKind::Rhob,
                "NEUTRON" => LogKind::Nphi,
                "SONIC" => LogKind::Dtc,
                "RES_DEEP" => LogKind::Resd,
                "PEF" => LogKind::Pef,
                _ => continue,
            };
            if is_excluded(log, &m) || is_derived(&m) {
                continue;
            }
            let list = per_log.entry(log).or_default();
            if !list.contains(&m) {
                list.push(m);
            }
        }
        let mut lists = BTreeMap::new();
        let mut any = false;
        for log in LogKind::ALL {
            let rows = per_log.remove(&log).unwrap_or_default();
            let mut ordered: Vec<String> = Vec::with_capacity(rows.len());
            for head in preferred_heads(log) {
                if rows.iter().any(|m| m == head) {
                    ordered.push((*head).to_string());
                }
            }
            for m in rows.iter().filter(|m| !is_low(m)) {
                if !ordered.contains(m) {
                    ordered.push(m.clone());
                }
            }
            for m in rows.iter().filter(|m| is_low(m)) {
                if !ordered.contains(m) {
                    ordered.push(m.clone());
                }
            }
            any |= !ordered.is_empty();
            lists.insert(log, ordered);
        }
        if !any {
            return Err(SxError::Format(
                "alias dictionary has no usable rows".into(),
            ));
        }
        Ok(AliasTable { lists })
    }

    /// Insert extra mnemonics per log. Returns the new table.
    ///
    /// Each extra (trimmed, upper-cased, empty skipped), in the given order, is
    /// skipped when already in the list or excluded; otherwise it goes just
    /// before the first low-priority entry (or at the end when there is none).
    pub fn with_extra(&self, extra: &BTreeMap<LogKind, Vec<String>>) -> AliasTable {
        let mut lists = self.lists.clone();
        for (log, names) in extra {
            let list = lists.entry(*log).or_default();
            for name in names {
                let m = name.trim().to_ascii_uppercase();
                if m.is_empty() || list.contains(&m) || is_excluded(*log, &m) {
                    continue;
                }
                match list.iter().position(|e| is_low(e)) {
                    Some(at) => list.insert(at, m),
                    None => list.push(m),
                }
            }
        }
        AliasTable { lists }
    }

    /// The ordered list for `log` (empty when the log has none).
    pub fn list(&self, log: LogKind) -> &[String] {
        self.lists.get(&log).map_or(&[], Vec::as_slice)
    }

    /// Priority index of an upper-case mnemonic for `log`, or `None`.
    pub fn priority(&self, log: LogKind, mnemonic_upper: &str) -> Option<usize> {
        self.list(log).iter().position(|e| e == mnemonic_upper)
    }
}

/// Read a dictionary CSV.
///
/// UTF-8, else Latin-1. First line = header. The category column is the first
/// header cell whose upper text is `ALIAS`, else column 0; the mnemonic column
/// is the first cell whose upper text is `OLD_NAME`, `MNEMONIC` or `NAME`,
/// else column 1. Lines split on `,`; fields are trimmed and lose one pair of
/// surrounding `"`. Short lines are skipped. Values are upper-cased; rows with
/// an empty value or `NAN` are dropped.
///
/// # Errors
/// `SxError::Io` (cannot read), `SxError::Format` (no rows).
pub fn load_alias_dictionary_csv(path: &Path) -> SxResult<Vec<(String, String)>> {
    let bytes = std::fs::read(path).map_err(|e| io_error(path, e))?;
    let text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => e.into_bytes().iter().map(|&b| char::from(b)).collect(),
    };
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut lines = text.lines();
    let header: Vec<String> = match lines.next() {
        Some(l) => l.split(',').map(clean_field).collect(),
        None => Vec::new(),
    };
    let cat_col = header
        .iter()
        .position(|h| h.eq_ignore_ascii_case("ALIAS"))
        .unwrap_or(0);
    let mn_col = header
        .iter()
        .position(|h| {
            let u = h.to_ascii_uppercase();
            u == "OLD_NAME" || u == "MNEMONIC" || u == "NAME"
        })
        .unwrap_or(1);
    let need = cat_col.max(mn_col) + 1;
    let mut rows = Vec::new();
    for line in lines {
        let fields: Vec<String> = line.split(',').map(clean_field).collect();
        if fields.len() < need {
            continue;
        }
        let (Some(cat), Some(mn)) = (fields.get(cat_col), fields.get(mn_col)) else {
            continue;
        };
        let cat = cat.to_ascii_uppercase();
        let mn = mn.to_ascii_uppercase();
        if cat.is_empty() || mn.is_empty() || cat == "NAN" || mn == "NAN" {
            continue;
        }
        rows.push((cat, mn));
    }
    if rows.is_empty() {
        return Err(SxError::Format(format!(
            "{}: alias dictionary has no rows",
            path.display()
        )));
    }
    Ok(rows)
}

/// Trim a field and remove one pair of surrounding double quotes.
fn clean_field(raw: &str) -> String {
    let t = raw.trim();
    let t = t
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(t);
    t.trim().to_string()
}

// ---------------------------------------------------------------- built-in lists
// Index 0 is the highest priority. Text taken from rule 4.2; the SHA-256 test
// pins every list.
const GR_LIST: &[&str] = &[
    "GR",
    "GRD",
    "GRN",
    "GRS",
    "GRR",
    "GR2",
    "GAMMA",
    "GRM",
    "GR_EDTC",
    "GR_EC",
    "GR_DT",
    "GR_RES",
    "RGR",
    "GR_WSI",
    "GR_1",
    "GR_3CARBS",
    "GR_RES3",
    "HGR",
    "GR_PND",
    "MSGR",
    "GRN1",
    "GRN2",
    "GRN3",
    "GRN4",
    "GRNC",
    "GRNPC_T",
    "GRL",
    "GRLL",
    "GRM1",
    "GRMB",
    "GRML",
    "GRMR",
    "GRMU",
    "GRN_",
    "GRHI",
    "GRHT",
    "GRG",
    "GRG1",
    "GRGC",
    "GRE",
    "GRC",
    "GRC_P",
    "GRCD",
    "GRCN",
    "GRCO",
    "GRCRX",
    "GRCX",
    "GRAP",
    "GRAPI",
    "GRARX",
    "GRA",
    "GR8",
    "GR-1",
    "GR100",
    "GR1AM",
    "GR1AX",
    "GR2_1",
    "GR25",
    "GR3",
    "GR4",
    "GR5",
    "GR7",
    "GRAD",
    "CED_GR",
    "GRWS",
    "GR_10",
    "GRTO",
    "GRT",
    "GR_T",
    "GRR_1",
    "GR_RM",
    "GR_MWD",
    "GRR1",
    "GRR2",
    "GRSG",
    "GRTN",
    "GRS1",
    "GAMA",
    "GRN_1",
    "GRTC",
    "GR_2",
    "GRN_EDTC",
    "GRD_1",
    "GRS_2",
    "GRS2",
    "GRD1",
    "GRD2",
    "GRR_EDTC",
    "GRS_EDTC",
    "GRTR",
    "GRSGS",
    "GRSL",
    "GR_OBS",
    "GR_S",
    "GR_EDT",
    "GR_MXW",
    "GR_EDTC1",
    "GRAM",
    "GRAX",
    "PGXRC-T",
    "PGRG",
    "LWD_GR",
    "LWD_GRM",
    "GRAX_MWD",
    "GREXT",
    "GRST",
    "GR_OPEN",
    "GR_CNL",
    "GR_FDC",
    "GRTT",
    "GRTOTAL",
    "GRR_T",
    "GRN_T",
    "GR_CAL",
    "GRSD",
    "GR_DN",
    "GR_LF",
    "GR_RG",
    "GR_UP",
    "GRRAX",
    "GR_TOTAL",
    "GR_MUD",
    "GR_RT",
    "GRW",
    "GRWA",
    "GR_RM1",
    "GR_RM2",
    "GR_RM4",
    "GAMAA",
    "GRM_MWD",
    "GR_R10",
    "GRTON",
    "GR_RM3",
    "GAMM",
    "GR_MXW_RT",
    "GR_GNT",
    "GR_3",
    "GRR1_1",
    "GRS_1",
    "GRR_2",
    "GRS3",
    "GRS4",
    "GR_IMP",
    "RE26_NDP_GR",
    "RE_NDPR_GR",
    "RE_SNDP_GR",
    "RE_SNDPR_GR",
    "GRCORR",
    "GR_RE25",
];

const RHOB_LIST: &[&str] = &[
    "RHOB",
    "RHOZ",
    "ZDEN",
    "DEN",
    "RHOB1",
    "RHOM",
    "RHOS",
    "RHO8",
    "DENSITY",
    "ERHO",
    "ZDEN_1",
    "ZDEN1",
    "ZDEN2",
    "RHOB_2",
    "RHOB_SHFT_1",
    "RHOB1_1",
    "RHOB1_SHFT_1",
    "BULK_DEN",
    "DEN_1",
    "DENS",
    "RE26_NDP_RHOB",
    "RE_NDPR_RHOB",
    "RE_SNDP_RHOB",
    "RE_SNDPR_RHOB",
];

const NPHI_LIST: &[&str] = &[
    "NPHI",
    "NPLS",
    "CNPHI",
    "SNP",
    "TNPH",
    "NPOR",
    "CNC",
    "CNL",
    "CNPOR",
    "NPHI_DEC",
    "NPHI_N",
    "NPHI_F",
    "NLBC",
    "CN",
    "CN_LS",
    "CNC_LS",
    "CNEC",
    "CNEC_1",
    "CNLS",
    "CNLS_WSI",
    "CNLS_WSI_RAW",
    "CNLSC",
    "CNPOR_LS",
    "CNS",
    "CNS_",
    "CNS_LS",
    "CNSSC",
    "CNTC",
    "CRFI",
    "DCSF",
    "CRNI",
    "ENLI",
    "FBAC",
    "FDSN",
    "FTMD",
    "NDSN",
    "NEU",
    "NEUT_PHI_NOR",
    "NHPI",
    "NLIM",
    "NPDL",
    "NPHD",
    "NPHI_LS",
    "NPHI_PND",
    "NPHI_S",
    "NPHI_SHFT_1",
    "NPHI1_1",
    "NPHI2",
    "NPHIAVG_1",
    "NPHICH",
    "NPHL",
    "NPHS",
    "NPL",
    "NPL_1",
    "NPOR_1",
    "NPOR_LS",
    "PND_1",
    "PND_SHFT_1",
    "NPRL",
    "NPRS",
    "NPSS",
    "NTMD",
    "SNP_1",
    "SNP_2",
    "SNP_LS",
    "SNPAVG_1",
    "SNPAVG_SHFT_1",
    "SNPC_1",
    "SNPCDOL_1",
    "TNPH_1",
    "TNPH_LS",
    "TNPH_SHFT_1",
    "NPOR_LIM",
    "NPLC",
    "NPOR_DOL",
    "NPRD",
    "CNLS_DEC",
    "CNCLS",
    "CNCL",
    "CNL_LS",
    "CND",
    "CNDF",
    "CNDL_DEC",
    "CNDL",
    "CNL_LIM",
    "CNLF",
    "CNPOR_DEC",
    "CNPOR_LIM",
    "CNPORD",
    "CNPO",
    "CNSS_DEC",
    "CNSS",
    "TNPHLS",
    "TNPH_LIM",
    "TNPHL",
    "TNPLS",
    "TNPL",
    "TNPH_DL",
    "TNPHDL",
    "TNPH_DOL",
    "TNPH_SS",
    "TNPH_SAN",
    "TNPS",
    "TNPH1",
    "TNPH2",
    "CN_LIM",
    "CNC25",
    "CNCDL",
    "CNCL_1",
    "CNCSS",
    "CNEUT",
    "CNEUT1",
    "CNEUT2",
    "CNP",
    "CNPLS",
    "CNPORLS",
    "CNS_LIM",
    "HTNP",
    "HTNP1",
    "NEU1",
    "NPHI_1",
    "NPLS_1",
    "NPOR_2",
    "TNPH_2",
    "NPHI_2",
    "NPHI_3",
    "NPHI_4",
    "NPHI_LIM",
    "NPHID",
    "NPHIL",
    "NPHILS",
    "NPLS1",
    "NPOR_3",
    "NPOR_4",
    "TNPH_LS_1",
    "TNPH_3",
    "TNPH_4",
    "HTNP_SAN",
    "NPHLS",
    "CNC_SS",
    "CNC_LIM",
    "CNC_DL",
    "RE26_NDP_NPHI",
    "RE_NDPR_NPHI",
    "RE_SNDP_NPHI",
    "RE_SNDPR_NPHI",
    "TNPH_MERGED",
    "NPHI_RE25",
];

const DTC_LIST: &[&str] = &[
    "DTC",
    "DT",
    "DTCO",
    "DT2",
    "DTCO1",
    "DT_1",
    "AC",
    "DTLF",
    "DTLN",
    "SON",
    "CPSL",
    "DTL",
    "DT24",
    "DT_CORE",
    "DT0P",
    "DT1R",
    "DT2R",
    "DT35",
    "DT4P",
    "SONIC",
    "DT_3",
    "DT_4",
    "DT_5",
    "DT_6",
    "DT_7",
    "DT_C",
    "DT24I",
    "DT24QA",
    "DT24QI",
    "DTRP",
    "DTD",
    "DTP",
    "DT4PR",
    "DT4PT",
    "DTC_RA_FT",
    "DTC1",
    "DTC26",
    "DTCF",
    "DTCO3",
    "DT:1",
    "DT:2",
    "DT24:1",
    "DT24:2",
    "DT4P:1",
    "DT4P:2",
    "DTCO:1",
    "DTCO:2",
    "DTLF:1",
    "DTLF:2",
    "AC:1",
    "AC:2",
    "DT0P:1",
    "DT0P:2",
    "DT35:1",
    "DT35:2",
    "DT_2",
    "DT_X",
    "DTCO31",
    "DT5-6",
    "DT4",
    "RE_SNDP_DTC",
    "RE_SNDPR_DTC",
    "DTC_RE25",
];

const RESD_LIST: &[&str] = &[
    "RESD",
    "ILD",
    "LLD",
    "AT90",
    "HLLD",
    "RD",
    "RLA5",
    "AF90",
    "RT90",
    "RT",
    "RES_DEEP",
    "RILD",
    "DEEP_LAT",
    "LLD_N",
    "RLLD",
    "RES_ILD",
    "DEEP_IND",
    "AIT90",
    "AHT90",
    "M1R9",
    "M2R9",
    "M2R9_1",
    "AHTCO90",
    "AO90",
    "R90",
    "RLA5_1",
    "RLA51",
    "RLL",
    "ILD_1",
    "RES90",
    "RT90_1",
    "ILD_3",
    "ILD4",
    "ILD1_1",
    "ILD2_1",
    "RT_HRLT",
    "REID",
    "RLA4",
    "RLA3",
    "ILD2",
    "AHORT",
    "DDLL",
    "IL",
    "LLD_1",
    "DVC2",
    "DVR2",
    "RES_ILS",
    "RLLA",
    "ILD_2",
    "RES_445",
    "RES_LAT_D",
    "DRES",
    "RT_2",
    "RLA31",
    "M2R9QI",
    "RE_NDPR_RTSP",
    "RE_SNDPR_RTSP",
    "RTSP",
    "LLD_DIGITIZED",
    "RLA3_MERGED",
    "RT_RE25",
];

const PEF_LIST: &[&str] = &[
    "PEF",
    "PE",
    "PEFZ",
    "PEFL",
    "PEF8",
    "PHOTO_ELEC",
    "PEF_1",
    "PEC",
    "PEFS",
    "PEL",
    "PEDN",
    "PEF_S",
    "PEF_SHFT_1",
    "PES",
    "PE__L",
    "PE_0_3",
    "PE_1",
    "PE_HAL",
    "PE_I",
    "PE_L",
    "PE_LONG",
    "PE_S",
    "PE_SHORT",
    "PE1",
    "PE2",
    "PEA",
    "PEB",
    "PECLNX",
    "PECOR",
    "PEDF",
    "PEDF_1",
    "PEDN_1",
    "PEF_2",
    "PEF_HAL",
    "PEF2",
    "PEFBA",
    "PEFC",
    "PEFLA",
    "PEFSA",
    "PEI",
    "PE-L",
    "PELONG",
    "PEFZ1",
    "PEFZ_1",
    "PEF_3",
    "RE26_NDP_PE",
    "RE_NDPR_PE",
    "RE_NDPR_PEF",
    "RE_SNDP_PE",
    "RE_SNDPR_PE",
    "RE_SNDPR_PEF",
    "PEF_DIGITIZED",
    "PE_CORR",
    "PEFZ_MERGED",
    "PE_RE25",
];

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha(list: &[String]) -> String {
        hex::encode(Sha256::digest(list.join("\n").as_bytes()))
    }

    #[test]
    fn builtin_lists_exact() {
        let t = AliasTable::builtin();
        let expected = [
            (
                LogKind::Gr,
                149,
                "650005a286f96b69bef74a5b2bbb2633f23025f00f4780cc1cdf11caa01d0055",
            ),
            (
                LogKind::Rhob,
                24,
                "3904b83c73d212013b66bd702038753e28d5736ab893265d73fbb3c38118d7c7",
            ),
            (
                LogKind::Nphi,
                148,
                "300cfa13f8e5ef69bceda35c88deb83edc1c3cebd6868c7c61e402fc15793b98",
            ),
            (
                LogKind::Dtc,
                63,
                "076dc271c4ca371ba2ca7244234d559e533cae97eaebcb6ca67e61832930ac3a",
            ),
            (
                LogKind::Resd,
                61,
                "6f28fc79e0b8acd260714ccdc49f884217ea2ee324ecaf35c37fda0eeb3a3e57",
            ),
            (
                LogKind::Pef,
                55,
                "927ceb24e64923127b4eacdde2d3608d5cca279f58c4fb5794ca50fc628f1d82",
            ),
        ];
        for (log, n, digest) in expected {
            assert_eq!(t.list(log).len(), n, "{log:?}");
            assert_eq!(sha(t.list(log)), digest, "{log:?}");
        }
    }

    #[test]
    fn builtin_has_no_excluded_or_derived_and_low_is_tail() {
        let t = AliasTable::builtin();
        for log in LogKind::ALL {
            let list = t.list(log);
            let mut seen_low = false;
            for m in list {
                assert!(!is_excluded(log, m), "{log:?} {m} excluded");
                assert!(!is_derived(m), "{log:?} {m} derived");
                if is_low(m) {
                    seen_low = true;
                } else {
                    assert!(!seen_low, "{log:?} {m} after a low entry");
                }
            }
            let uniq: BTreeSet<&String> = list.iter().collect();
            assert_eq!(uniq.len(), list.len(), "{log:?} has repeats");
        }
        assert_eq!(t.priority(LogKind::Gr, "GR"), Some(0));
        assert_eq!(t.priority(LogKind::Gr, "GRD"), Some(1));
        assert_eq!(t.priority(LogKind::Gr, "XYZ"), None);
    }

    #[test]
    fn excluded_examples() {
        for m in ["HSGR_1", "CGR2", "GR_POTA", "GRTH", "GR_K", "GKUT"] {
            assert!(is_excluded(LogKind::Gr, m), "GR {m}");
        }
        for m in [
            "NCNL", "POROSITY", "NEUT", "NEUT_2", "NEUT3", "PHIE", "TPHI1",
        ] {
            assert!(is_excluded(LogKind::Nphi, m), "NPHI {m}");
        }
        for m in ["SPHI", "DT_S", "ITT"] {
            assert!(is_excluded(LogKind::Dtc, m), "DTC {m}");
        }
        assert!(is_excluded(LogKind::Rhob, "AIPD"));
        assert!(is_excluded(LogKind::Resd, "LL7"));
        for m in ["GR", "GRN", "GRTO", "GR_UP"] {
            assert!(!is_excluded(LogKind::Gr, m), "GR {m}");
        }
        for m in ["NPHI", "NPSS", "NEUTRONX"] {
            assert!(!is_excluded(LogKind::Nphi, m), "NPHI {m}");
        }
        for m in ["DT", "DT0P"] {
            assert!(!is_excluded(LogKind::Dtc, m), "DTC {m}");
        }
        for m in ["ILD", "IL"] {
            assert!(!is_excluded(LogKind::Resd, m), "RESD {m}");
        }
        assert!(!is_excluded(LogKind::Rhob, "RHOB"));
        assert!(!is_excluded(LogKind::Pef, "SGR"));
    }

    #[test]
    fn extra_goes_before_first_low() {
        let t = AliasTable::builtin();
        let mut extra = BTreeMap::new();
        extra.insert(
            LogKind::Resd,
            vec![
                "xres".to_string(),
                "ILD".to_string(),
                "LL7".to_string(),
                "  ".to_string(),
            ],
        );
        extra.insert(LogKind::Pef, vec!["PEX".to_string()]);
        let t2 = t.with_extra(&extra);
        let resd = t2.list(LogKind::Resd);
        assert_eq!(resd.len(), 62);
        let n = resd.len();
        assert_eq!(
            &resd[n - 5..],
            ["RTSP", "XRES", "LLD_DIGITIZED", "RLA3_MERGED", "RT_RE25"]
        );
        // No low entry in PEF is at the very end, so an extra goes before it.
        let pef = t2.list(LogKind::Pef);
        assert_eq!(pef.len(), 56);
        let at = pef.iter().position(|m| m == "PEX");
        let first_low = pef.iter().position(|m| is_low(m));
        assert_eq!(at.map(|a| a + 1), first_low);
        // The original table is untouched.
        assert_eq!(t.list(LogKind::Resd).len(), 61);
        // An empty extra map changes nothing.
        assert_eq!(t.with_extra(&BTreeMap::new()), t);
    }

    fn rows(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
            .collect()
    }

    #[test]
    fn dictionary_build() {
        let t = AliasTable::from_dictionary(&rows(&[
            ("GR", "gr_edtc"),
            ("GR", "GRN"),
            ("GR", "GRN"),
            ("GR", "SGR"),
            ("GR", "GR_NRM"),
            ("GR", "GR"),
            ("GR", "GR_MERGED"),
            ("DENSITY", "RHOB"),
            ("DENSITY", "DPHI_X"),
            ("DENSITY", "CDL1"),
            ("DPHI", "DPHI"),
            ("NEUTRON", "TNPH"),
            ("NEUTRON", "NEUT"),
            ("SONIC", "DT"),
            ("SONIC", "DTS"),
            ("RES_DEEP", "ILD"),
            ("RES_DEEP", "LL"),
            ("PEF", "PEF"),
            ("OTHER", "XX"),
        ]))
        .unwrap();
        assert_eq!(t.list(LogKind::Gr), ["GR", "GRN", "GR_EDTC", "GR_MERGED"]);
        assert_eq!(t.list(LogKind::Rhob), ["RHOB"]);
        assert_eq!(t.list(LogKind::Nphi), ["TNPH"]);
        assert_eq!(t.list(LogKind::Dtc), ["DT"]);
        assert_eq!(t.list(LogKind::Resd), ["ILD"]);
        assert_eq!(t.list(LogKind::Pef), ["PEF"]);
    }

    #[test]
    fn dictionary_empty_is_format_error() {
        let err = AliasTable::from_dictionary(&rows(&[("OTHER", "X"), ("DPHI", "Y")]));
        match err {
            Err(SxError::Format(m)) => assert_eq!(m, "alias dictionary has no usable rows"),
            other => panic!("expected Format, got {other:?}"),
        }
        assert!(AliasTable::from_dictionary(&[]).is_err());
    }

    #[test]
    fn dictionary_csv_utf8_and_latin1() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.csv");
        std::fs::write(
            &p,
            "id,Alias,old_name\n1,gr,\"gr\"\n2, DENSITY ,rhob\nshort\n3,GR,\n4,,X\n5,gr,nan\n",
        )
        .unwrap();
        let got = load_alias_dictionary_csv(&p).unwrap();
        assert_eq!(got, rows(&[("GR", "GR"), ("DENSITY", "RHOB")]));

        // Default columns 0 and 1; Latin-1 bytes (0xE9 is not UTF-8).
        let p2 = dir.path().join("l.csv");
        std::fs::write(&p2, b"cat,mn\nGR,GR\xe9X\n").unwrap();
        let got = load_alias_dictionary_csv(&p2).unwrap();
        assert_eq!(got, rows(&[("GR", "GR\u{e9}X")]));

        let p3 = dir.path().join("e.csv");
        std::fs::write(&p3, "a,b\n").unwrap();
        assert!(matches!(
            load_alias_dictionary_csv(&p3),
            Err(SxError::Format(_))
        ));
        assert!(matches!(
            load_alias_dictionary_csv(&dir.path().join("none.csv")),
            Err(SxError::Io { .. })
        ));
    }

    /// Placed in `with_extra` terms: a name present in both, no repeat.
    #[test]
    fn extra_skips_duplicates() {
        let t = AliasTable::builtin();
        let mut extra = BTreeMap::new();
        extra.insert(LogKind::Gr, vec!["GR".to_string(), "mygr".to_string()]);
        let t2 = t.with_extra(&extra);
        assert_eq!(t2.list(LogKind::Gr).len(), 150);
        assert_eq!(
            t2.list(LogKind::Gr).iter().filter(|m| *m == "GR").count(),
            1
        );
    }
}
