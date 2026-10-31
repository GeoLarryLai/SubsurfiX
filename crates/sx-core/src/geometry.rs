// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Well geometry: minimum curvature, TVD, elevation, per-sample x/y.
//!
//! Depths and survey offsets are in feet, project coordinates in meters.
//! Feet are converted to meters only with [`crate::units::M_PER_FT`]
//! (0.3048 exactly).

use crate::error::{SxError, SxResult};
use crate::types::{GridRange, WellGeometry, WellHeader};
use crate::units::M_PER_FT;

/// Below this dogleg (radians) the ratio factor is 1 and tangents are blended
/// linearly.
const BETA_EPS: f64 = 1e-7;

/// A dogleg this close to π (radians) cannot be represented.
const REVERSAL_EPS: f64 = 1e-6;

/// A deviation survey as delivered by the Python edge. Same length vectors.
/// MD in feet below KB; inclination in degrees from vertical (0 = down);
/// azimuth in degrees clockwise from project-frame north.
#[derive(Debug, Clone, PartialEq)]
pub struct Survey {
    /// Measured depth, feet below KB.
    pub md_ft: Vec<f64>,
    /// Inclination, degrees from vertical.
    pub inc_deg: Vec<f64>,
    /// Azimuth, degrees clockwise from project-frame north.
    pub azi_deg: Vec<f64>,
}

/// What cleaning removed or added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SurveyCleanReport {
    /// Number of input stations.
    pub n_input: usize,
    /// NaN/inf in any field, or inclination outside [0, 180].
    pub n_bad: usize,
    /// Stations dropped because `md_ft < 0`.
    pub n_negative_md: usize,
    /// Stations dropped because an earlier station has exactly the same MD.
    pub n_duplicate_md: usize,
    /// A surface station (MD 0) was added in front.
    pub prepended_surface: bool,
}

/// One computed station. Angles in radians. Offsets in feet from the wellhead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Station {
    /// Measured depth, feet below KB.
    pub md_ft: f64,
    /// Inclination, radians.
    pub inc_rad: f64,
    /// Azimuth, radians, in `[0, 2π)`.
    pub azi_rad: f64,
    /// True vertical depth below KB, feet, positive down.
    pub tvd_ft: f64,
    /// Northing offset from the wellhead, feet.
    pub north_ft: f64,
    /// Easting offset from the wellhead, feet.
    pub east_ft: f64,
    /// Dogleg angle from the previous station, radians (0 for the first).
    pub dogleg_rad: f64,
}

/// Position along the hole at one MD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// True vertical depth below KB, feet.
    pub tvd_ft: f64,
    /// Northing offset from the wellhead, feet.
    pub north_ft: f64,
    /// Easting offset from the wellhead, feet.
    pub east_ft: f64,
    /// Inclination, degrees.
    pub inc_deg: f64,
    /// Azimuth, degrees, in `[0, 360)`.
    pub azi_deg: f64,
}

/// Minimum-curvature trajectory of one well.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    stations: Vec<Station>,
    has_survey: bool,
}

/// Increments over one minimum-curvature step.
#[derive(Debug, Clone, Copy)]
struct Increment {
    dtvd: f64,
    dnorth: f64,
    deast: f64,
    beta: f64,
}

/// Dogleg angle between two directions, radians.
fn dogleg(i1: f64, a1: f64, i2: f64, a2: f64) -> f64 {
    let c = (i2 - i1).cos() - i1.sin() * i2.sin() * (1.0 - (a2 - a1).cos());
    c.clamp(-1.0, 1.0).acos()
}

/// Minimum-curvature step of length `dmd` from `(i1, a1)` to `(i2, a2)`.
fn mc_increment(dmd: f64, i1: f64, a1: f64, i2: f64, a2: f64) -> Increment {
    let beta = dogleg(i1, a1, i2, a2);
    let rf = if beta > BETA_EPS {
        (2.0 / beta) * (beta / 2.0).tan()
    } else {
        1.0
    };
    let half = dmd / 2.0;
    Increment {
        dtvd: half * (i1.cos() + i2.cos()) * rf,
        dnorth: half * (i1.sin() * a1.cos() + i2.sin() * a2.cos()) * rf,
        deast: half * (i1.sin() * a1.sin() + i2.sin() * a2.sin()) * rf,
        beta,
    }
}

/// Unit tangent `(north, east, down)`.
fn tangent(inc: f64, azi: f64) -> [f64; 3] {
    [inc.sin() * azi.cos(), inc.sin() * azi.sin(), inc.cos()]
}

fn nan_position() -> Position {
    Position {
        tvd_ft: f64::NAN,
        north_ft: f64::NAN,
        east_ft: f64::NAN,
        inc_deg: f64::NAN,
        azi_deg: f64::NAN,
    }
}

fn station_position(s: &Station) -> Position {
    Position {
        tvd_ft: s.tvd_ft,
        north_ft: s.north_ft,
        east_ft: s.east_ft,
        inc_deg: s.inc_rad.to_degrees(),
        azi_deg: s.azi_rad.to_degrees(),
    }
}

/// Straight line below (or at) station `s`, along its tangent.
fn tangent_extension(s: &Station, md: f64) -> Position {
    let d = md - s.md_ft;
    Position {
        tvd_ft: s.tvd_ft + d * s.inc_rad.cos(),
        north_ft: s.north_ft + d * s.inc_rad.sin() * s.azi_rad.cos(),
        east_ft: s.east_ft + d * s.inc_rad.sin() * s.azi_rad.sin(),
        inc_deg: s.inc_rad.to_degrees(),
        azi_deg: s.azi_rad.to_degrees(),
    }
}

/// Point on the arc between `s1` and `s2` at `md` (`s1.md < md < s2.md`).
fn interpolate(s1: &Station, s2: &Station, md: f64) -> Position {
    let dmd = s2.md_ft - s1.md_ft;
    let f = (md - s1.md_ft) / dmd;
    let t1 = tangent(s1.inc_rad, s1.azi_rad);
    let t2 = tangent(s2.inc_rad, s2.azi_rad);
    let beta = s2.dogleg_rad;
    let mut t = [0.0; 3];
    if beta > BETA_EPS {
        let sb = beta.sin();
        let w1 = ((1.0 - f) * beta).sin() / sb;
        let w2 = (f * beta).sin() / sb;
        for k in 0..3 {
            t[k] = w1 * t1[k] + w2 * t2[k];
        }
    } else {
        for k in 0..3 {
            t[k] = (1.0 - f) * t1[k] + f * t2[k];
        }
        let norm = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if norm > 0.0 {
            for v in t.iter_mut() {
                *v /= norm;
            }
        }
    }
    let inc_f = t[2].clamp(-1.0, 1.0).acos();
    let azi_f = if t[0].hypot(t[1]) < 1e-12 {
        s1.azi_rad
    } else {
        wrap_two_pi(t[1].atan2(t[0]))
    };
    let inc = mc_increment(md - s1.md_ft, s1.inc_rad, s1.azi_rad, inc_f, azi_f);
    Position {
        tvd_ft: s1.tvd_ft + inc.dtvd,
        north_ft: s1.north_ft + inc.dnorth,
        east_ft: s1.east_ft + inc.deast,
        inc_deg: inc_f.to_degrees(),
        azi_deg: azi_f.to_degrees(),
    }
}

/// Map an angle in `[-π, π]` (radians) to `[0, 2π)`.
fn wrap_two_pi(a: f64) -> f64 {
    let w = if a < 0.0 {
        a + std::f64::consts::TAU
    } else {
        a
    };
    if w >= std::f64::consts::TAU {
        0.0
    } else {
        w
    }
}

impl Trajectory {
    /// Vertical well: one station at MD 0, inclination 0. `has_survey() == false`.
    pub fn vertical() -> Trajectory {
        Trajectory {
            stations: vec![Station {
                md_ft: 0.0,
                inc_rad: 0.0,
                azi_rad: 0.0,
                tvd_ft: 0.0,
                north_ft: 0.0,
                east_ft: 0.0,
                dogleg_rad: 0.0,
            }],
            has_survey: false,
        }
    }

    /// Clean the survey and compute minimum-curvature stations.
    ///
    /// Cleaning: drop stations with a non-finite field or inclination outside
    /// `[0, 180]` (`n_bad`), drop `md_ft < 0` (`n_negative_md`), wrap azimuth
    /// into `[0, 360)`, stable-sort by MD, keep the first of exactly equal MDs
    /// (`n_duplicate_md`), and prepend a surface station (MD 0, inclination 0,
    /// azimuth of the first station) when the first MD is above 0.
    ///
    /// Units: feet in, feet out. NaN rule: stations with NaN are dropped.
    /// Errors: [`SxError::Shape`] if the three vectors differ in length;
    /// [`SxError::InvalidInput`] if a dogleg between consecutive stations is
    /// `>= π − 1e-6`. Zero usable stations gives `Ok(vertical)` with the
    /// report. Complexity: O(n log n).
    pub fn from_survey(survey: &Survey) -> SxResult<(Trajectory, SurveyCleanReport)> {
        let n = survey.md_ft.len();
        if survey.inc_deg.len() != n || survey.azi_deg.len() != n {
            return Err(SxError::Shape(format!(
                "survey lengths differ: md {}, inc {}, azi {}",
                n,
                survey.inc_deg.len(),
                survey.azi_deg.len()
            )));
        }
        let mut report = SurveyCleanReport {
            n_input: n,
            ..SurveyCleanReport::default()
        };

        // (md_ft, inc_deg, azi_deg) of kept rows.
        let mut rows: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        for ((&md, &inc), &azi) in survey
            .md_ft
            .iter()
            .zip(survey.inc_deg.iter())
            .zip(survey.azi_deg.iter())
        {
            if !md.is_finite()
                || !inc.is_finite()
                || !azi.is_finite()
                || !(0.0..=180.0).contains(&inc)
            {
                report.n_bad += 1;
                continue;
            }
            if md < 0.0 {
                report.n_negative_md += 1;
                continue;
            }
            // Fold -0.0 into 0.0 so equal MDs compare and sort as equal.
            let md = if md == 0.0 { 0.0 } else { md };
            let mut azi = azi.rem_euclid(360.0);
            if azi >= 360.0 {
                azi = 0.0;
            }
            rows.push((md, inc, azi));
        }

        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<(f64, f64, f64)> = Vec::with_capacity(rows.len() + 1);
        for row in rows {
            match kept.last() {
                Some(last) if last.0 == row.0 => report.n_duplicate_md += 1,
                _ => kept.push(row),
            }
        }

        let first = match kept.first() {
            Some(first) => *first,
            None => return Ok((Trajectory::vertical(), report)),
        };
        if first.0 > 0.0 {
            kept.insert(0, (0.0, 0.0, first.2));
            report.prepended_surface = true;
        }

        let mut stations: Vec<Station> = Vec::with_capacity(kept.len());
        for (md, inc_deg, azi_deg) in kept {
            let inc = inc_deg.to_radians();
            let azi = azi_deg.to_radians();
            let station = match stations.last() {
                None => Station {
                    md_ft: md,
                    inc_rad: inc,
                    azi_rad: azi,
                    tvd_ft: 0.0,
                    north_ft: 0.0,
                    east_ft: 0.0,
                    dogleg_rad: 0.0,
                },
                Some(p) => {
                    let step = mc_increment(md - p.md_ft, p.inc_rad, p.azi_rad, inc, azi);
                    if step.beta >= std::f64::consts::PI - REVERSAL_EPS {
                        return Err(SxError::InvalidInput(format!(
                            "dogleg between MD {} ft and MD {} ft is a reversal",
                            p.md_ft, md
                        )));
                    }
                    Station {
                        md_ft: md,
                        inc_rad: inc,
                        azi_rad: azi,
                        tvd_ft: p.tvd_ft + step.dtvd,
                        north_ft: p.north_ft + step.dnorth,
                        east_ft: p.east_ft + step.deast,
                        dogleg_rad: step.beta,
                    }
                }
            };
            stations.push(station);
        }

        Ok((
            Trajectory {
                stations,
                has_survey: true,
            },
            report,
        ))
    }

    /// True when the trajectory came from a survey (not [`Trajectory::vertical`]).
    pub fn has_survey(&self) -> bool {
        self.has_survey
    }

    /// Stations, strictly increasing in MD, first at MD 0.
    pub fn stations(&self) -> &[Station] {
        &self.stations
    }

    /// Largest station inclination, degrees; NaN when `!has_survey()`.
    pub fn max_inc_deg(&self) -> f64 {
        if !self.has_survey {
            return f64::NAN;
        }
        self.stations
            .iter()
            .map(|s| s.inc_rad.to_degrees())
            .fold(f64::NAN, f64::max)
    }

    /// Position at `md_ft`, feet and degrees.
    ///
    /// On a station: that station. Between stations: the same circular arc
    /// the stations assume. Below the last station: a straight line along its
    /// tangent. NaN rule: NaN `md_ft` or `md_ft < 0` gives all-NaN fields.
    /// Complexity: O(log stations).
    pub fn position_at(&self, md_ft: f64) -> Position {
        if md_ft.is_nan() || md_ft < 0.0 {
            return nan_position();
        }
        let n = self.stations.partition_point(|s| s.md_ft <= md_ft);
        match n.checked_sub(1) {
            Some(idx) => self.position_in(idx, md_ft),
            None => nan_position(),
        }
    }

    /// `position_at(md_ft).tvd_ft`. NaN rule: propagate (negative MD is NaN).
    pub fn tvd_at(&self, md_ft: f64) -> f64 {
        self.position_at(md_ft).tvd_ft
    }

    /// Smallest MD >= 0 whose TVD equals `tvd_ft`, feet; `None` if never
    /// reached.
    ///
    /// NaN rule: NaN or negative `tvd_ft` gives `None`; `0` gives `Some(0.0)`.
    /// Each station segment is split into `max(1, ceil(dogleg_deg))` equal
    /// pieces, and the first piece whose ends straddle the target is bisected
    /// to 1e-9 ft (at most 100 steps). Past the last station the tangent line
    /// is used when it descends. Complexity: O(stations + total dogleg degrees).
    pub fn md_at_tvd(&self, tvd_ft: f64) -> Option<f64> {
        if tvd_ft.is_nan() || tvd_ft < 0.0 {
            return None;
        }
        if tvd_ft == 0.0 {
            return Some(0.0);
        }
        for (idx, pair) in self.stations.windows(2).enumerate() {
            let (s1, s2) = match pair {
                [a, b] => (a, b),
                _ => continue,
            };
            let m = s2.dogleg_rad.to_degrees().ceil().max(1.0);
            let m_count = m as usize;
            let mut a = s1.md_ft;
            let mut ga = s1.tvd_ft - tvd_ft;
            for j in 1..=m_count {
                let (b, gb) = if j == m_count {
                    (s2.md_ft, s2.tvd_ft - tvd_ft)
                } else {
                    let b = s1.md_ft + (s2.md_ft - s1.md_ft) * (j as f64 / m);
                    (b, self.position_in(idx, b).tvd_ft - tvd_ft)
                };
                if gb == 0.0 {
                    return Some(b);
                }
                if ga * gb < 0.0 {
                    return Some(self.bisect_tvd(idx, a, b, ga, tvd_ft));
                }
                a = b;
                ga = gb;
            }
        }
        let last = self.stations.last()?;
        let c = last.inc_rad.cos();
        if last.tvd_ft <= tvd_ft && c > 1e-12 {
            Some(last.md_ft + (tvd_ft - last.tvd_ft) / c)
        } else {
            None
        }
    }

    /// Bisect `tvd_at(md) = target` on `[a, b]` inside segment `idx`;
    /// `ga` is the (nonzero) residual at `a`.
    fn bisect_tvd(&self, idx: usize, mut a: f64, mut b: f64, ga: f64, target: f64) -> f64 {
        for _ in 0..100 {
            if b - a <= 1e-9 {
                break;
            }
            let mid = 0.5 * (a + b);
            let gm = self.position_in(idx, mid).tvd_ft - target;
            if gm == 0.0 {
                return mid;
            }
            if (gm < 0.0) == (ga < 0.0) {
                a = mid;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    }

    /// Position at `md`, where station `idx` is the last station with
    /// `md_ft <= md`. All-NaN if `idx` is out of range.
    fn position_in(&self, idx: usize, md: f64) -> Position {
        let Some(s1) = self.stations.get(idx) else {
            return nan_position();
        };
        match self.stations.get(idx + 1) {
            None => tangent_extension(s1, md),
            Some(s2) => {
                if md == s1.md_ft {
                    station_position(s1)
                } else {
                    interpolate(s1, s2, md)
                }
            }
        }
    }
}

/// Per-sample geometry on `grid`.
///
/// `tvd_ft` from the trajectory, `elev_ft = kb_elev_ft - tvd_ft`, and
/// `x_m`/`y_m` = header coordinates plus survey offsets (feet) times
/// [`M_PER_FT`]. Units: feet for depths, meters for coordinates.
/// NaN rule: propagate (a NaN header x/y/KB gives NaN in that output only;
/// a sample at negative MD is NaN in every output). Empty grid gives empty
/// vectors. Errors: none beyond `GridRange` limits. Complexity: O(n + stations).
pub fn build_geometry(
    header: &WellHeader,
    grid: GridRange,
    traj: &Trajectory,
) -> SxResult<WellGeometry> {
    let n = grid.len;
    let mut tvd_ft = Vec::with_capacity(n);
    let mut elev_ft = Vec::with_capacity(n);
    let mut x_m = Vec::with_capacity(n);
    let mut y_m = Vec::with_capacity(n);
    let mut idx = 0usize;
    for i in 0..n {
        let md = grid.md_ft(i);
        let p = if md.is_nan() || md < 0.0 {
            nan_position()
        } else {
            while let Some(next) = traj.stations.get(idx + 1) {
                if next.md_ft <= md {
                    idx += 1;
                } else {
                    break;
                }
            }
            traj.position_in(idx, md)
        };
        tvd_ft.push(p.tvd_ft);
        elev_ft.push(header.kb_elev_ft - p.tvd_ft);
        x_m.push(header.x_m + p.east_ft * M_PER_FT);
        y_m.push(header.y_m + p.north_ft * M_PER_FT);
    }
    let geometry = WellGeometry {
        id: header.id,
        grid,
        tvd_ft,
        elev_ft,
        x_m,
        y_m,
    };
    geometry.validate()?;
    Ok(geometry)
}

/// TVD and elevation of one MD (tops): `(tvd_ft, kb_elev_ft − tvd_ft)`, feet.
///
/// Wells without a survey get `tvd = md`. NaN rule: propagate (NaN MD or
/// negative MD gives NaN both; NaN KB gives NaN elevation only).
pub fn md_to_tvd_elev(traj: &Trajectory, kb_elev_ft: f64, md_ft: f64) -> (f64, f64) {
    let tvd = traj.tvd_at(md_ft);
    (tvd, kb_elev_ft - tvd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WellId;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 + 1e-9 * b.abs()
    }

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!(close(a, b), "{what}: got {a}, want {b}");
    }

    fn header() -> WellHeader {
        WellHeader {
            id: WellId(7),
            name: "w".into(),
            x_m: 1000.0,
            y_m: 2000.0,
            kb_elev_ft: 3000.0,
        }
    }

    fn survey3() -> Survey {
        Survey {
            md_ft: vec![0.0, 1000.0, 1500.0],
            inc_deg: vec![0.0, 0.0, 30.0],
            azi_deg: vec![0.0, 0.0, 90.0],
        }
    }

    fn traj3() -> Trajectory {
        Trajectory::from_survey(&survey3()).unwrap().0
    }

    fn survey_of(rows: &[(f64, f64, f64)]) -> Survey {
        Survey {
            md_ft: rows.iter().map(|r| r.0).collect(),
            inc_deg: rows.iter().map(|r| r.1).collect(),
            azi_deg: rows.iter().map(|r| r.2).collect(),
        }
    }

    #[test]
    fn station_two_numbers() {
        let t = traj3();
        assert!(t.has_survey());
        let s = t.stations();
        assert_eq!(s.len(), 3);
        let s2 = s[2];
        assert_close(s2.dogleg_rad, 30f64.to_radians(), "beta");
        let rf = (2.0 / (std::f64::consts::PI / 6.0)) * 15f64.to_radians().tan();
        assert_close(rf, 1.0234905233494715, "rf");
        assert_close(s2.tvd_ft, 1477.464829276, "tvd");
        assert_close(s2.east_ft, 127.936315419, "east");
        assert_close(s2.north_ft, 0.0, "north");
        assert_close(s2.tvd_ft - 1000.0, 477.464829276, "delta tvd");
    }

    #[test]
    fn three_station_table() {
        let t = traj3();
        // md, tvd, elev, north, east, x, y, inc, azi
        let rows = [
            (500.0, 500.0, 2500.0, 0.0, 0.0, 1000.0, 2000.0, 0.0, 0.0),
            (1000.0, 1000.0, 2000.0, 0.0, 0.0, 1000.0, 2000.0, 0.0, 0.0),
            (
                1250.0,
                1247.153982366,
                1752.846017634,
                0.0,
                32.538439067,
                1009.917716228,
                2000.0,
                15.0,
                90.0,
            ),
            (
                1500.0,
                1477.464829276,
                1522.535170724,
                0.0,
                127.936315419,
                1038.994988940,
                2000.0,
                30.0,
                90.0,
            ),
            (
                2000.0,
                1910.477531168,
                1089.522468832,
                0.0,
                377.936315419,
                1115.194988940,
                2000.0,
                30.0,
                90.0,
            ),
        ];
        for (md, tvd, elev, north, east, x, y, inc, azi) in rows {
            let p = t.position_at(md);
            assert_close(p.tvd_ft, tvd, "tvd");
            assert_close(p.north_ft, north, "north");
            assert_close(p.east_ft, east, "east");
            assert_close(p.inc_deg, inc, "inc");
            assert_close(p.azi_deg, azi, "azi");
            let k = (md / 0.5) as i32;
            let g = build_geometry(&header(), GridRange::new(k, 1).unwrap(), &t).unwrap();
            assert_close(g.tvd_ft[0], tvd, "g tvd");
            assert_close(g.elev_ft[0], elev, "g elev");
            assert_close(g.x_m[0], x, "g x");
            assert_close(g.y_m[0], y, "g y");
        }
    }

    #[test]
    fn grid_walk_matches_point_queries() {
        let t = traj3();
        let g = build_geometry(&header(), GridRange::new(0, 4001).unwrap(), &t).unwrap();
        g.validate().unwrap();
        assert_eq!(g.id, WellId(7));
        for i in 0..4001 {
            let md = 0.5 * i as f64;
            let p = t.position_at(md);
            assert_eq!(g.tvd_ft[i], p.tvd_ft);
            assert_eq!(g.elev_ft[i], 3000.0 - p.tvd_ft);
            assert_eq!(g.x_m[i], 1000.0 + p.east_ft * M_PER_FT);
            assert_eq!(g.y_m[i], 2000.0 + p.north_ft * M_PER_FT);
        }
    }

    #[test]
    fn md_at_tvd_and_tops() {
        let t = traj3();
        let md = t.md_at_tvd(1247.153982366).unwrap();
        assert!((md - 1250.0).abs() <= 1e-6, "md {md}");
        let (tvd, elev) = md_to_tvd_elev(&t, 3000.0, 1250.0);
        assert_close(tvd, 1247.153982366, "tvd");
        assert_close(elev, 1752.846017634, "elev");
    }

    #[test]
    fn md_at_tvd_edge_cases() {
        let t = traj3();
        assert_eq!(t.md_at_tvd(f64::NAN), None);
        assert_eq!(t.md_at_tvd(-1.0), None);
        assert_eq!(t.md_at_tvd(0.0), Some(0.0));
        // Exactly at a station TVD.
        let md = t.md_at_tvd(1000.0).unwrap();
        assert_close(md, 1000.0, "station md");
        // Past the last station: tangent line at 30 degrees.
        let md = t.md_at_tvd(1910.477531168).unwrap();
        assert!((md - 2000.0).abs() <= 1e-6, "md {md}");
        // Round trip over the whole well.
        for md in [10.0, 999.0, 1100.0, 1250.0, 1499.0, 1750.0, 5000.0] {
            let tvd = t.tvd_at(md);
            let back = t.md_at_tvd(tvd).unwrap();
            assert!((back - md).abs() <= 1e-6, "md {md} back {back}");
        }
        // Vertical well: md equals tvd.
        let v = Trajectory::vertical();
        assert_close(v.md_at_tvd(321.5).unwrap(), 321.5, "vertical");
        // Horizontal end: never reaches deeper.
        let (h, _) =
            Trajectory::from_survey(&survey_of(&[(0.0, 90.0, 0.0), (1000.0, 90.0, 0.0)])).unwrap();
        assert_eq!(h.md_at_tvd(10.0), None);
    }

    #[test]
    fn md_at_tvd_returns_shallowest_crossing() {
        // Goes down to inc 90, then turns back up: tvd rises then falls.
        let (t, _) = Trajectory::from_survey(&survey_of(&[
            (0.0, 0.0, 0.0),
            (1000.0, 90.0, 0.0),
            (2000.0, 135.0, 0.0),
        ]))
        .unwrap();
        let end = t.stations()[2].tvd_ft;
        let peak = t.stations()[1].tvd_ft;
        assert!(end < peak);
        let target = 0.5 * (end + peak);
        // The same TVD is reached again on the way back up.
        let md = t.md_at_tvd(target).unwrap();
        assert!(md < 1000.0, "md {md}");
        assert_close(t.tvd_at(md), target, "tvd at md");
        // A TVD deeper than the peak is never reached (inc > 90 at the end).
        assert_eq!(t.md_at_tvd(peak + 1.0), None);
    }

    #[test]
    fn prepended_surface_station() {
        let (t, rep) =
            Trajectory::from_survey(&survey_of(&[(500.0, 3.0, 45.0), (1000.0, 6.0, 45.0)]))
                .unwrap();
        assert!(rep.prepended_surface);
        assert_eq!(rep.n_input, 2);
        let s = t.stations();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].md_ft, 0.0);
        assert_eq!(s[0].inc_rad, 0.0);
        assert_close(s[0].azi_rad, 45f64.to_radians(), "surface azi");
        assert_close(s[1].tvd_ft, 499.771568250, "tvd 500");
        assert_close(s[1].north_ft, 9.253891659, "north 500");
        assert_close(s[1].east_ft, 9.253891659, "east 500");
        assert_close(s[2].tvd_ft, 998.173297371, "tvd 1000");
        assert_close(s[2].north_ft, 36.990202362, "north 1000");
        assert_close(s[2].east_ft, 36.990202362, "east 1000");
    }

    #[test]
    fn interpolated_position_continues_to_next_station() {
        let (t, _) = Trajectory::from_survey(&survey_of(&[
            (0.0, 0.0, 0.0),
            (1000.0, 10.0, 30.0),
            (2000.0, 40.0, 120.0),
        ]))
        .unwrap();
        let p = t.position_at(1600.0);
        assert_close(p.tvd_ft, 1572.271099887, "tvd");
        assert_close(p.north_ft, 78.785032650, "north");
        assert_close(p.east_ft, 188.998080073, "east");
        // Reference inc and azi are rounded to 6 decimals.
        assert!((p.inc_deg - 24.499212).abs() <= 1e-6, "inc {}", p.inc_deg);
        assert!((p.azi_deg - 109.616773).abs() <= 1e-6, "azi {}", p.azi_deg);

        let s3 = t.stations()[2];
        let step = mc_increment(
            2000.0 - 1600.0,
            p.inc_deg.to_radians(),
            p.azi_deg.to_radians(),
            s3.inc_rad,
            s3.azi_rad,
        );
        assert_close(p.tvd_ft + step.dtvd, s3.tvd_ft, "tvd to s3");
        assert_close(p.north_ft + step.dnorth, s3.north_ft, "north to s3");
        assert_close(p.east_ft + step.deast, s3.east_ft, "east to s3");
    }

    #[test]
    fn vertical_default() {
        let v = Trajectory::vertical();
        assert!(!v.has_survey());
        assert_eq!(v.stations().len(), 1);
        assert!(v.max_inc_deg().is_nan());
        let p = v.position_at(1234.5);
        assert_eq!(
            (p.tvd_ft, p.north_ft, p.east_ft, p.inc_deg, p.azi_deg),
            (1234.5, 0.0, 0.0, 0.0, 0.0)
        );
        let (tvd, elev) = md_to_tvd_elev(&v, 100.0, 40.0);
        assert_eq!((tvd, elev), (40.0, 60.0));
        let g = build_geometry(&header(), GridRange::new(0, 3).unwrap(), &v).unwrap();
        assert_eq!(g.tvd_ft, vec![0.0, 0.5, 1.0]);
        assert_eq!(g.x_m, vec![1000.0; 3]);
        assert_eq!(g.y_m, vec![2000.0; 3]);
    }

    #[test]
    fn max_inc() {
        assert_close(traj3().max_inc_deg(), 30.0, "max inc");
    }

    #[test]
    fn cleaning_counts() {
        let s = survey_of(&[
            (0.0, 0.0, 0.0),
            (f64::NAN, 5.0, 10.0),
            (-10.0, 5.0, 10.0),
            (500.0, 5.0, 10.0),
            (500.0, 9.0, 99.0),
            (1000.0, 10.0, 10.0),
        ]);
        let (t, rep) = Trajectory::from_survey(&s).unwrap();
        assert_eq!(
            rep,
            SurveyCleanReport {
                n_input: 6,
                n_bad: 1,
                n_negative_md: 1,
                n_duplicate_md: 1,
                prepended_surface: false,
            }
        );
        let st = t.stations();
        assert_eq!(st.len(), 3);
        // First of the duplicates (inc 5, azi 10) is kept.
        assert_close(st[1].inc_rad, 5f64.to_radians(), "kept inc");
        assert_close(st[1].azi_rad, 10f64.to_radians(), "kept azi");
    }

    #[test]
    fn cleaning_bad_inclination_sort_and_azimuth() {
        let s = survey_of(&[
            (1000.0, 10.0, -90.0),
            (0.0, 0.0, 0.0),
            (500.0, 181.0, 0.0),
            (600.0, -1.0, 0.0),
            (700.0, 5.0, f64::INFINITY),
            (800.0, 5.0, 450.0),
        ]);
        let (t, rep) = Trajectory::from_survey(&s).unwrap();
        assert_eq!(rep.n_bad, 3);
        assert_eq!(rep.n_negative_md, 0);
        let st = t.stations();
        assert_eq!(st.len(), 3);
        assert_eq!(st[1].md_ft, 800.0);
        assert_close(st[1].azi_rad, 90f64.to_radians(), "450 wraps to 90");
        assert_close(st[2].azi_rad, 270f64.to_radians(), "-90 wraps to 270");
    }

    #[test]
    fn first_station_at_zero_used_as_is() {
        let (t, rep) =
            Trajectory::from_survey(&survey_of(&[(0.0, 10.0, 0.0), (100.0, 10.0, 0.0)])).unwrap();
        assert!(!rep.prepended_surface);
        let s = t.stations();
        assert_eq!(s.len(), 2);
        assert_close(s[0].inc_rad, 10f64.to_radians(), "inc");
        assert_eq!(s[0].tvd_ft, 0.0);
        assert_close(s[1].tvd_ft, 100.0 * 10f64.to_radians().cos(), "tvd");
        assert_close(s[1].north_ft, 100.0 * 10f64.to_radians().sin(), "north");
    }

    #[test]
    fn negative_zero_md_counts_as_duplicate_of_zero() {
        let (t, rep) = Trajectory::from_survey(&survey_of(&[
            (0.0, 1.0, 0.0),
            (-0.0, 2.0, 0.0),
            (100.0, 2.0, 0.0),
        ]))
        .unwrap();
        assert_eq!(rep.n_duplicate_md, 1);
        assert_close(t.stations()[0].inc_rad, 1f64.to_radians(), "first kept");
    }

    #[test]
    fn zero_usable_stations_is_vertical() {
        let (t, rep) =
            Trajectory::from_survey(&survey_of(&[(f64::NAN, 1.0, 1.0), (-5.0, 1.0, 1.0)])).unwrap();
        assert!(!t.has_survey());
        assert_eq!(t, Trajectory::vertical());
        assert_eq!(rep.n_input, 2);
        assert_eq!(rep.n_bad, 1);
        assert_eq!(rep.n_negative_md, 1);
        assert!(!rep.prepended_surface);
        let (t, rep) = Trajectory::from_survey(&survey_of(&[])).unwrap();
        assert!(!t.has_survey());
        assert_eq!(rep, SurveyCleanReport::default());
    }

    #[test]
    fn survey_errors() {
        let bad = Survey {
            md_ft: vec![0.0, 1.0],
            inc_deg: vec![0.0],
            azi_deg: vec![0.0, 0.0],
        };
        assert!(matches!(
            Trajectory::from_survey(&bad),
            Err(SxError::Shape(_))
        ));
        let reversal = survey_of(&[(0.0, 90.0, 0.0), (100.0, 90.0, 180.0)]);
        assert!(matches!(
            Trajectory::from_survey(&reversal),
            Err(SxError::InvalidInput(_))
        ));
    }

    #[test]
    fn position_nan_rules() {
        let t = traj3();
        for md in [f64::NAN, -0.5] {
            let p = t.position_at(md);
            assert!(p.tvd_ft.is_nan() && p.north_ft.is_nan() && p.east_ft.is_nan());
            assert!(p.inc_deg.is_nan() && p.azi_deg.is_nan());
            assert!(t.tvd_at(md).is_nan());
            let (tvd, elev) = md_to_tvd_elev(&t, 3000.0, md);
            assert!(tvd.is_nan() && elev.is_nan());
        }
    }

    #[test]
    fn nan_kb_only_affects_elevation() {
        let mut h = header();
        h.kb_elev_ft = f64::NAN;
        let g = build_geometry(&h, GridRange::new(0, 100).unwrap(), &traj3()).unwrap();
        assert!(g.elev_ft.iter().all(|v| v.is_nan()));
        assert!(g.tvd_ft.iter().all(|v| v.is_finite()));
        assert!(g.x_m.iter().all(|v| v.is_finite()));
        assert!(g.y_m.iter().all(|v| v.is_finite()));
        let (tvd, elev) = md_to_tvd_elev(&traj3(), f64::NAN, 1250.0);
        assert!(tvd.is_finite() && elev.is_nan());
    }

    #[test]
    fn nan_header_xy_only_affects_that_axis() {
        let mut h = header();
        h.x_m = f64::NAN;
        let g = build_geometry(&h, GridRange::new(0, 10).unwrap(), &traj3()).unwrap();
        assert!(g.x_m.iter().all(|v| v.is_nan()));
        assert!(g.y_m.iter().all(|v| v.is_finite()));
        assert!(g.tvd_ft.iter().all(|v| v.is_finite()));
        assert!(g.elev_ft.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empty_grid_and_negative_md_samples() {
        let g = build_geometry(&header(), GridRange::new(0, 0).unwrap(), &traj3()).unwrap();
        assert!(g.tvd_ft.is_empty() && g.elev_ft.is_empty());
        assert!(g.x_m.is_empty() && g.y_m.is_empty());
        // Grid starting above MD 0: negative-MD samples are NaN, rest follow.
        let g = build_geometry(&header(), GridRange::new(-2, 4).unwrap(), &traj3()).unwrap();
        assert!(g.tvd_ft[0].is_nan() && g.tvd_ft[1].is_nan());
        assert_eq!(g.tvd_ft[2], 0.0);
        assert_eq!(g.tvd_ft[3], 0.5);
    }

    #[test]
    fn small_dogleg_uses_unit_ratio_factor() {
        let (t, _) = Trajectory::from_survey(&survey_of(&[
            (0.0, 10.0, 20.0),
            (100.0, 10.0, 20.0),
            (200.0, 10.0 + 1e-9, 20.0),
        ]))
        .unwrap();
        let p = t.position_at(150.0);
        assert!(p.tvd_ft.is_finite());
        assert_close(p.inc_deg, 10.0, "inc");
        assert_close(p.azi_deg, 20.0, "azi");
    }

    #[test]
    #[ignore = "timing budget; run on request"]
    fn timing_build_geometry_20k_samples() {
        let rows: Vec<(f64, f64, f64)> = (0..200)
            .map(|i| (100.0 * f64::from(i), 0.3 * f64::from(i), 2.0 * f64::from(i)))
            .collect();
        let (t, _) = Trajectory::from_survey(&survey_of(&rows)).unwrap();
        let grid = GridRange::new(0, 20_000).unwrap();
        let start = std::time::Instant::now();
        let g = build_geometry(&header(), grid, &t).unwrap();
        let dt = start.elapsed();
        assert_eq!(g.tvd_ft.len(), 20_000);
        assert!(dt.as_secs_f64() < 0.002, "took {dt:?}");
    }
}
