# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from subsurfix.edges.area import AreaSource, load_area, points_in_area
from subsurfix.edges.crs import reproject_xy
from subsurfix.errors import ConfigError, MissingDependencyError


def test_points_in_area_square_with_hole() -> None:
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=np.float64)
    hole = np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]], dtype=np.float64)
    x_m = np.array([1.0, 5.0, 20.0, np.nan, 10.0])
    y_m = np.array([1.0, 5.0, 5.0, 1.0, 5.0])
    inside = points_in_area(x_m, y_m, [outer, hole])
    assert inside.tolist() == [True, False, False, False, False]


def test_missing_shapefile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "shapefile", None)
    path = tmp_path / "area.shp"
    path.write_bytes(b"")
    with pytest.raises(MissingDependencyError):
        load_area(AreaSource(path=str(path)), "EPSG:32614")


def test_shapefile_prj_and_missing_prj(tmp_path: Path) -> None:
    import pyproj
    import shapefile

    with_prj = tmp_path / "area.shp"
    writer = shapefile.Writer(str(with_prj), shapeType=shapefile.POLYGON)
    writer.field("name", "C")
    writer.poly([[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]])
    writer.record("a")
    writer.close()
    with_prj.with_suffix(".prj").write_text(pyproj.CRS("EPSG:32614").to_wkt(), encoding="utf-8")
    rings = load_area(AreaSource(path=str(with_prj)), "EPSG:32614")
    assert rings[0][0, 0] == 0.0
    assert rings[0][1, 0] == 10.0
    assert rings[0][0, 1] == 0.0
    assert np.array_equal(rings[0][0], rings[0][-1])

    bare = tmp_path / "bare.shp"
    writer = shapefile.Writer(str(bare), shapeType=shapefile.POLYGON)
    writer.field("name", "C")
    writer.poly([[[0, 0], [1, 0], [1, 1], [0, 0]]])
    writer.record("a")
    writer.close()
    with pytest.raises(ConfigError, match="no .prj"):
        load_area(AreaSource(path=str(bare)), "EPSG:32614")


def test_geojson_legacy_crs_and_default_wgs84(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.geojson"
    legacy.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:EPSG::32614"},
                },
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rings = load_area(AreaSource(path=str(legacy)), "EPSG:32614")
    assert rings[0][0, 0] == 0.0
    assert rings[0][2, 0] == 10.0

    geographic = tmp_path / "wgs84.geojson"
    geographic.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [[-102.0, 31.5], [-102.0, 31.6], [-101.9, 31.6], [-101.9, 31.5], [-102.0, 31.5]]
                ],
            }
        ),
        encoding="utf-8",
    )
    rings = load_area(AreaSource(path=str(geographic)), "EPSG:32614")
    expected_x, expected_y = reproject_xy([-102.0], [31.5], "EPSG:4326", "EPSG:32614")
    assert abs(float(rings[0][0, 0]) - float(expected_x[0])) <= 1e-3
    assert abs(float(rings[0][0, 1]) - float(expected_y[0])) <= 1e-3


def _write_gpkg(path: Path, layers: tuple[str, ...]) -> None:
    import shapely
    from shapely.geometry import Polygon

    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (7, 3), (7, 7), (3, 7)]])
    blob = b"GP" + bytes([0, 1]) + struct.pack("<i", 32614) + shapely.to_wkb(polygon)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE gpkg_spatial_ref_sys ("
        "srs_name TEXT, srs_id INTEGER PRIMARY KEY, organization TEXT, "
        "organization_coordsys_id INTEGER, definition TEXT, description TEXT)"
    )
    connection.execute(
        "CREATE TABLE gpkg_contents ("
        "table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT, description TEXT, "
        "last_change TEXT, min_x REAL, min_y REAL, max_x REAL, max_y REAL, srs_id INTEGER)"
    )
    connection.execute(
        "CREATE TABLE gpkg_geometry_columns ("
        "table_name TEXT, column_name TEXT, geometry_type_name TEXT, "
        "srs_id INTEGER, z INTEGER, m INTEGER)"
    )
    connection.execute(
        "INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
        ("EPSG:32614", 32614, "EPSG", 32614, "undefined", ""),
    )
    for layer in layers:
        connection.execute(f'CREATE TABLE "{layer}" (geom BLOB)')
        connection.execute(
            "INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?,?,?,?)",
            (layer, "features", layer, "", "2020-01-01T00:00:00Z", 0, 0, 10, 10, 32614),
        )
        connection.execute(
            "INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)",
            (layer, "geom", "POLYGON", 32614, 0, 0),
        )
        connection.execute(f'INSERT INTO "{layer}" (geom) VALUES (?)', (blob,))
    connection.commit()
    connection.close()


def test_geopackage_rings_and_layer_choice(tmp_path: Path) -> None:
    path = tmp_path / "area.gpkg"
    _write_gpkg(path, ("area",))
    rings = load_area(AreaSource(path=str(path)), "EPSG:32614")
    x_m = np.array([1.0, 5.0, 20.0, np.nan, 10.0])
    y_m = np.array([1.0, 5.0, 5.0, 1.0, 5.0])
    inside = points_in_area(x_m, y_m, rings)
    assert inside.tolist() == [True, False, False, False, False]

    many = tmp_path / "many.gpkg"
    _write_gpkg(many, ("beta", "alpha"))
    with pytest.raises(ConfigError, match="alpha"):
        load_area(AreaSource(path=str(many)), "EPSG:32614")
    rings = load_area(AreaSource(path=str(many), layer="beta"), "EPSG:32614")
    assert len(rings) == 2
