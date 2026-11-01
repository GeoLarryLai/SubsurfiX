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
