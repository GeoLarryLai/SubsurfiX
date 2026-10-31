// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `[edges]` sources. Rust parses and validates the keys; it does not open the files.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

fn owned(text: &str) -> String {
    text.to_string()
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().copied().map(str::to_string).collect()
}

fn def_comma() -> String {
    owned(",")
}

fn def_none_string() -> Option<String> {
    None
}

fn def_none_vec() -> Option<Vec<String>> {
    None
}

fn def_empty_vec() -> Vec<String> {
    Vec::new()
}

fn def_uwi() -> Vec<String> {
    strings(&["UWI"])
}

fn def_true() -> bool {
    true
}

fn def_header_x() -> Vec<String> {
    strings(&["Surface X", "X"])
}

fn def_header_y() -> Vec<String> {
    strings(&["Surface Y", "Y"])
}

fn def_header_label() -> Vec<String> {
    strings(&["Name", "well_name", "Well Name"])
}

fn def_kb_match() -> Option<String> {
    Some(owned("datum value"))
}

fn def_datum_col() -> Vec<String> {
    strings(&["Well datum name", "Datum"])
}

fn def_lat_col() -> Vec<String> {
    strings(&["Latitude", "Lat", "LAT"])
}

fn def_lon_col() -> Vec<String> {
    strings(&["Longitude", "Lon", "Long", "LON"])
}

fn def_location_col() -> Vec<String> {
    strings(&["Location", "Legal Location", "STR"])
}

fn def_td_col() -> Vec<String> {
    strings(&["Depth_stop", "TD", "Total Depth", "Driller TD"])
}

fn def_long() -> String {
    owned("long")
}

fn def_shallowest() -> String {
    owned("shallowest")
}

fn def_top_col() -> Vec<String> {
    strings(&["Top", "Surface", "top_name", "topname"])
}

fn def_md_col() -> Vec<String> {
    strings(&["Measured Depth_ft", "Measured Depth", "MD"])
}

fn def_inc_col() -> Vec<String> {
    strings(&["Deviation Angle_deg", "Inclination", "Inc"])
}

fn def_azi_col() -> Vec<String> {
    strings(&["Deviation Azimuth_deg", "Azimuth", "Azi"])
}

/// One `[[edges.headers]]` table. `path` and `kb_unit` are required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderSource {
    /// header table file (CSV or XLSX), relative to the project folder (path)
    pub path: PathBuf,
    /// "ft" or "m"; unit of the datum elevation
    pub kb_unit: String,
    /// worksheet name (XLSX)
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    /// CSV delimiter
    #[serde(default = "def_comma")]
    pub delimiter: String,
    /// CRS of x/y; exactly one of `crs` and `xy_unit` must be set
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub crs: Option<String>,
    /// "m", "ft" or "us-ft" when x/y are already in the project frame
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub xy_unit: Option<String>,
    /// candidate column names for the well key
    #[serde(default = "def_uwi")]
    pub key_col: Vec<String>,
    /// candidate column names for x
    #[serde(default = "def_header_x")]
    pub x_col: Vec<String>,
    /// candidate column names for y
    #[serde(default = "def_header_y")]
    pub y_col: Vec<String>,
    /// candidate column names for the well label
    #[serde(default = "def_header_label")]
    pub label_col: Vec<String>,
    /// candidate column names for the datum elevation
    #[serde(default = "def_empty_vec")]
    pub kb_col: Vec<String>,
    /// text matched against the datum-value column; Rust passes it through
    #[serde(default = "def_kb_match", skip_serializing_if = "Option::is_none")]
    pub kb_match: Option<String>,
    /// candidate column names for the datum type
    #[serde(default = "def_datum_col")]
    pub datum_col: Vec<String>,
    /// candidate column names for latitude, degrees (LAS match hint)
    #[serde(default = "def_lat_col")]
    pub lat_col: Vec<String>,
    /// candidate column names for longitude, degrees (LAS match hint)
    #[serde(default = "def_lon_col")]
    pub lon_col: Vec<String>,
    /// candidate column names for a legal location such as section-township-range (LAS match hint)
    #[serde(default = "def_location_col")]
    pub location_col: Vec<String>,
    /// candidate column names for total depth, kb_unit (LAS match hint)
    #[serde(default = "def_td_col")]
    pub td_col: Vec<String>,
}

/// One `[[edges.tops]]` table. `path`, `depth_ref`, and `depth_unit` are required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopsSource {
    /// tops table file (path)
    pub path: PathBuf,
    /// "md", "tvd", "tvdss" or "elev"
    pub depth_ref: String,
    /// "ft" or "m"
    pub depth_unit: String,
    /// worksheet name (XLSX)
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    /// CSV delimiter
    #[serde(default = "def_comma")]
    pub delimiter: String,
    /// "long" or "wide"
    #[serde(default = "def_long")]
    pub layout: String,
    /// "shallowest", "first" or "error"
    #[serde(default = "def_shallowest")]
    pub repeat_pick: String,
    /// candidate column names for the well key
    #[serde(default = "def_uwi")]
    pub key_col: Vec<String>,
    /// candidate column names for the top name (long layout)
    #[serde(default = "def_top_col")]
    pub top_col: Vec<String>,
    /// candidate depth columns; None = default list for `depth_ref`
    #[serde(default = "def_none_vec", skip_serializing_if = "Option::is_none")]
    pub depth_col: Option<Vec<String>>,
    /// wide layout: the top columns; None = every other column
    #[serde(default = "def_none_vec", skip_serializing_if = "Option::is_none")]
    pub top_columns: Option<Vec<String>>,
    /// wide layout: columns that are not tops
    #[serde(default = "def_empty_vec")]
    pub exclude_columns: Vec<String>,
}

/// One `[[edges.surveys]]` table. `path` and `md_unit` are required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurveySource {
    /// survey table file (path)
    pub path: PathBuf,
    /// "ft" or "m"
    pub md_unit: String,
    /// worksheet name (XLSX)
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    /// CSV delimiter
    #[serde(default = "def_comma")]
    pub delimiter: String,
    /// candidate column names for the well key
    #[serde(default = "def_uwi")]
    pub key_col: Vec<String>,
    /// candidate column names for MD
    #[serde(default = "def_md_col")]
    pub md_col: Vec<String>,
    /// candidate column names for inclination
    #[serde(default = "def_inc_col")]
    pub inc_col: Vec<String>,
    /// candidate column names for azimuth
    #[serde(default = "def_azi_col")]
    pub azi_col: Vec<String>,
}

/// One `[[edges.pts]]` table: a pressure-temperature-spinner (PTS) survey,
/// read by the Python edge into `edges/pts.parquet`. `path` is required and
/// exactly one of `well` or `key_col`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PtsSource {
    /// survey file: CSV, XLSX, or XLS (path)
    pub path: PathBuf,
    /// well key of the whole file
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub well: Option<String>,
    /// column holding the well key
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub key_col: Option<String>,
    /// sheets to read (all when empty)
    #[serde(default = "def_empty_vec")]
    pub sheets: Vec<String>,
    /// CSV delimiter
    #[serde(default = "def_comma")]
    pub delimiter: String,
    /// "degC", "degF" or "K" (else from the column name)
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub temp_unit: Option<String>,
    /// "psia", "psig", "kPa", "MPa" or "bar" (else from the column name)
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub pressure_unit: Option<String>,
    /// "ft" or "m" (else from the column name)
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub depth_unit: Option<String>,
}

/// Optional `[edges.area]` study-area file. `path` is required when the table is present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AreaSource {
    /// study-area file (.shp, .geojson, .json, .gpkg) (path)
    pub path: PathBuf,
    /// CRS of the file when it carries none
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub crs: Option<String>,
    /// GeoPackage layer name
    #[serde(default = "def_none_string", skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// a well must have tops to be selected
    #[serde(default = "def_true")]
    pub need_tops: bool,
}

/// Header, tops, survey, and study-area sources, plus PROJ flags.
///
/// Empty source lists are valid here so the template loads with no data files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct EdgesConfig {
    /// allow PROJ to download datum grids (Python edge only)
    pub allow_network: bool,
    /// allow ballpark datum transforms (Python edge only)
    pub allow_ballpark: bool,
    /// array of tables `[[edges.headers]]`
    pub headers: Vec<HeaderSource>,
    /// array of tables `[[edges.tops]]`
    pub tops: Vec<TopsSource>,
    /// array of tables `[[edges.surveys]]`
    pub surveys: Vec<SurveySource>,
    /// array of tables `[[edges.pts]]` (PTS surveys)
    pub pts: Vec<PtsSource>,
    /// optional table `[edges.area]` (study area)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area: Option<AreaSource>,
}
