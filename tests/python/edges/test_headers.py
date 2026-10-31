# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from subsurfix.edges.headers import HeaderSource, load_headers
from subsurfix.errors import ConfigError


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def _source(path: Path, **kwargs: object) -> HeaderSource:
    values: dict[str, object] = {
        "path": str(path),
        "kb_unit": "ft",
        "xy_unit": "m",
        "key_col": ("UWI",),
        "x_col": ("X",),
        "y_col": ("Y",),
        "kb_col": ("KB",),
        "kb_match": None,
        "label_col": (),
        "datum_col": (),
    }
    values.update(kwargs)
    return HeaderSource(**values)  # type: ignore[arg-type]


def test_header_units(tmp_path: Path) -> None:
    feet = tmp_path / "feet.csv"
    _write_csv(feet, [["UWI", "X", "Y", "KB"], ["W1", "1000", "1000", "914.4"]])
    table = load_headers([_source(feet, xy_unit="ft", kb_unit="m")], "EPSG:32614")
    assert table.column("x_m").to_pylist() == [304.8]
    assert table.column("y_m").to_pylist() == [304.8]
    assert table.column("kb_elev_ft").to_pylist() == [3000.0]

    us_feet = tmp_path / "usft.csv"
    _write_csv(us_feet, [["UWI", "X", "Y", "KB"], ["W1", "3937", "3937", "1"]])
    table = load_headers([_source(us_feet, xy_unit="us-ft")], "EPSG:32614")
    assert table.column("x_m").to_pylist() == [1200.0]
    assert table.column("y_m").to_pylist() == [1200.0]


def test_header_merge_keeps_first_finite_values(tmp_path: Path) -> None:
    first = tmp_path / "A.csv"
    second = tmp_path / "B.csv"
    _write_csv(first, [["UWI", "X", "Y", "KB"], ["W1", "", "", "3000"]])
    _write_csv(second, [["UWI", "X", "Y", "KB"], ["W1", "100", "200", "3100"]])
    table = load_headers([_source(first), _source(second)], "EPSG:32614")
    assert table.column("well_key").to_pylist() == ["W1"]
    assert table.column("x_m").to_pylist() == [100.0]
    assert table.column("y_m").to_pylist() == [200.0]
    assert table.column("xy_source").to_pylist() == ["B.csv"]
    assert table.column("kb_elev_ft").to_pylist() == [3000.0]
    assert table.column("kb_source").to_pylist() == ["A.csv"]
    assert "well_id" not in table.column_names


def test_headers_require_one_source_and_one_xy_declaration(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="at least one"):
        load_headers([], "EPSG:32614")
    path = tmp_path / "both.csv"
    _write_csv(path, [["UWI", "X", "Y"], ["W1", "1", "2"]])
    source = HeaderSource(
        path=str(path),
        kb_unit="ft",
        crs="EPSG:32614",
        xy_unit="m",
        key_col=("UWI",),
        x_col=("X",),
        y_col=("Y",),
    )
    with pytest.raises(ConfigError, match="exactly one"):
        load_headers([source], "EPSG:32614")


def test_header_match_hints(tmp_path: Path) -> None:
    """Latitude, longitude, legal location and TD are kept as LAS-match hints."""
    path = tmp_path / "kgs.csv"
    _write_csv(
        path,
        [
            ["UWI", "X", "Y", "KB", "Latitude", "Longitude", "Location", "Depth_stop"],
            ["W1", "500000", "4100000", "1342", "37.1", "-98.4", "T34S R11W, Sec. 1", "5142.5"],
            ["W2", "500100", "4100100", "1336", "", "", "", ""],
        ],
    )
    table = load_headers([_source(path)], "EPSG:32614")
    assert table.column("lat_deg").to_pylist()[0] == 37.1
    assert table.column("lon_deg").to_pylist()[0] == -98.4
    assert table.column("location").to_pylist() == ["T34S R11W, Sec. 1", None]
    assert table.column("td_ft").to_pylist() == [5142.5, None]
    # No latitude column: derived from the project frame (pyproj), else null.
    plain = tmp_path / "plain.csv"
    _write_csv(plain, [["UWI", "X", "Y", "KB"], ["W1", "500000", "4100000", "1"]])
    table = load_headers([_source(plain)], "EPSG:32614")
    lat = table.column("lat_deg").to_pylist()[0]
    try:
        import pyproj  # noqa: F401
    except ImportError:
        assert lat is None
    else:
        assert lat == pytest.approx(37.04, abs=0.01)
