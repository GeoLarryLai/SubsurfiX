# SPDX-License-Identifier: AGPL-3.0-or-later
"""PTS (pressure-temperature-spinner) surveys: units, passes, depth tie-in."""
import math

import pytest

from subsurfix.edges.pts import PtsSource, load_pts
from subsurfix.errors import ConfigError


def test_units_and_tie_in(tmp_path):
    a = tmp_path / "rih.csv"
    a.write_text("Depth(Feet),Pressure (psia),Temperature (F),Flow (RPM)\n0,14.696,212,0\n100,50,32,1\n")
    b = tmp_path / "station.csv"
    b.write_text("Time,Pressure kPa,Temp degC\n08:00,1000,150\n")
    with pytest.warns(UserWarning, match="no depth tie-in"):
        t = load_pts([PtsSource(path=str(a), well="W1"), PtsSource(path=str(b), well="W1")])
    rows = t.to_pylist()
    assert [r["pass_name"] for r in rows] == ["rih", "rih", "station"]
    assert rows[0]["temp_c"] == pytest.approx(100.0)
    assert rows[1]["temp_c"] == pytest.approx(0.0)
    assert rows[0]["pressure_mpa"] == pytest.approx(0.101325, rel=1e-4)
    assert rows[1]["md_ft"] == 100.0 and rows[1]["depth_tied"]
    assert math.isnan(rows[2]["md_ft"]) and not rows[2]["depth_tied"]
    assert rows[2]["pressure_mpa"] == pytest.approx(1.0)


def test_unknown_unit_is_an_error(tmp_path):
    a = tmp_path / "x.csv"
    a.write_text("MD,Temperature\n10,50\n")
    with pytest.raises(ConfigError, match="temp_unit"):
        load_pts([PtsSource(path=str(a), well="W1", depth_unit="ft")])
    t = load_pts([PtsSource(path=str(a), well="W1", depth_unit="m", temp_unit="degC")])
    assert t.column("md_ft").to_pylist()[0] == pytest.approx(10 / 0.3048)
    with pytest.raises(ConfigError, match="temp_unit"):
        load_pts([PtsSource(path=str(a), well="W1", temp_unit="rankine")])
