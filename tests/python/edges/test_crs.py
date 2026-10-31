# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import sys

import pytest

from subsurfix.edges.crs import make_transformer, project_crs, reproject_xy
from subsurfix.errors import ConfigError, MissingDependencyError


def test_reproject_lonlat_to_epsg_32614() -> None:
    x_m, y_m = reproject_xy([-102.0], [31.5], "EPSG:4326", "EPSG:32614")
    assert abs(float(x_m[0]) - 215047.49164992862) <= 1e-3
    assert abs(float(y_m[0]) - 3488916.7150485055) <= 1e-3


def test_project_crs_must_be_projected_metres() -> None:
    project_crs("EPSG:32614")
    with pytest.raises(ConfigError, match="projected CRS in metres"):
        project_crs("EPSG:4326")


def test_nad27_without_grid_is_config_error() -> None:
    try:
        make_transformer(
            "+proj=utm +zone=13 +datum=NAD27 +units=us-ft +no_defs",
            "EPSG:32614",
            allow_network=False,
        )
    except ConfigError as exc:
        message = str(exc)
        if "us_noaa_conus.tif" not in message:
            pytest.skip("datum grids are installed; NAD27 did not report us_noaa_conus.tif")
        assert "allow_ballpark" in message
        assert "allow_network" in message
    else:
        pytest.skip("datum grids are installed; NAD27 transformation did not need a missing grid")


def test_missing_pyproj(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyproj", None)
    with pytest.raises(MissingDependencyError):
        reproject_xy([0.0], [0.0], "EPSG:4326", "EPSG:32614")
