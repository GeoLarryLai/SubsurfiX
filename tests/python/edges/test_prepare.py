# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import csv
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from subsurfix.edges import prepare_edges
from subsurfix.edges.__main__ import main
from subsurfix.edges.schemas import (
    HEADERS_SCHEMA,
    SELECTION_SCHEMA,
    SURVEYS_SCHEMA,
    TOPS_SCHEMA,
    WELL_IDS_SCHEMA,
)
from subsurfix.errors import ConfigError


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_prepare_edges_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_csv(
        tmp_path / "headers.csv",
        [
            ["UWI", "X", "Y", "KB", "Name", "Datum"],
            ["A", "1", "1", "1000", "Alpha", "KB"],
            ["C", "100", "100", "1100", "Charlie", "KB"],
            ["B", "5", "5", "1050", "Bravo", "KB"],
        ],
    )
    _write_csv(
        tmp_path / "tops.csv",
        [
            ["UWI", "Top", "MD"],
            ["A", "T1", "1000"],
            ["B", "T1", "2000"],
        ],
    )
    _write_csv(
        tmp_path / "surveys.csv",
        [
            ["UWI", "MD", "Inc", "Azi"],
            ["A", "0", "0", "0"],
            ["A", "500", "10", "90"],
            ["B", "0", "0", "180"],
        ],
    )
    # Square (0, 0)-(10, 10) covers A (1, 1) and B (5, 5). C (100, 100) is outside.
    # A and B have tops, so 2 wells are selected.
    (tmp_path / "area.geojson").write_text(
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
    toml = tmp_path / "project.toml"
    toml.write_text(
        "\n".join(
            [
                "[project]",
                'crs = "EPSG:32614"',
                "",
                "[edges]",
                "allow_network = false",
                "allow_ballpark = false",
                "",
                "[[edges.headers]]",
                'path = "headers.csv"',
                'kb_unit = "ft"',
                'xy_unit = "m"',
                'key_col = ["UWI"]',
                'x_col = ["X"]',
                'y_col = ["Y"]',
                'label_col = ["Name"]',
                'kb_col = ["KB"]',
                'datum_col = ["Datum"]',
                "",
                "[[edges.tops]]",
                'path = "tops.csv"',
                'depth_ref = "md"',
                'depth_unit = "ft"',
                'layout = "long"',
                'key_col = ["UWI"]',
                'top_col = ["Top"]',
                'depth_col = ["MD"]',
                "",
                "[[edges.surveys]]",
                'path = "surveys.csv"',
                'md_unit = "ft"',
                'key_col = ["UWI"]',
                'md_col = ["MD"]',
                'inc_col = ["Inc"]',
                'azi_col = ["Azi"]',
                "",
                "[edges.area]",
                'path = "area.geojson"',
                "need_tops = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert main(["subsurfix.edges", str(toml)]) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(printed) == 5

    schemas = {
        "well_ids.parquet": ("subsurfix.edges_well_ids/1", WELL_IDS_SCHEMA),
        "headers.parquet": ("subsurfix.edges_headers/1", HEADERS_SCHEMA),
        "tops.parquet": ("subsurfix.edges_tops/1", TOPS_SCHEMA),
        "surveys.parquet": ("subsurfix.edges_surveys/1", SURVEYS_SCHEMA),
        "selection.parquet": ("subsurfix.edges_selection/1", SELECTION_SCHEMA),
    }
    edges = tmp_path / "edges"
    for filename, (schema_name, schema) in schemas.items():
        path = edges / filename
        got = pq.read_schema(path)
        assert got.names == schema.names
        for field, expected in zip(got, schema, strict=True):
            assert field.name == expected.name
            assert field.type == expected.type
            assert field.nullable is expected.nullable
        assert pq.ParquetFile(path).metadata.metadata == {
            b"subsurfix_schema": schema_name.encode("ascii")
        }

    selection = pq.read_table(edges / "selection.parquet")
    assert selection.column("well_key").to_pylist() == ["A", "B", "C"]
    assert selection.column("in_area").to_pylist() == [True, True, False]
    assert selection.column("has_tops").to_pylist() == [True, True, False]
    assert selection.column("selected").to_pylist() == [True, True, False]
    assert sum(selection.column("selected").to_pylist()) == 2

    well_bytes = (edges / "well_ids.parquet").read_bytes()
    header_bytes = (edges / "headers.parquet").read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_edges(toml)
    assert (edges / "well_ids.parquet").read_bytes() == well_bytes
    assert (edges / "headers.parquet").read_bytes() == header_bytes


def test_unknown_edges_key(tmp_path: Path) -> None:
    toml = tmp_path / "project.toml"
    toml.write_text(
        '[project]\ncrs = "EPSG:32614"\n\n[edges]\nnope = true\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="nope"):
        prepare_edges(toml)


def test_unknown_header_key(tmp_path: Path) -> None:
    toml = tmp_path / "project.toml"
    toml.write_text(
        "\n".join(
            [
                "[project]",
                'crs = "EPSG:32614"',
                "",
                "[[edges.headers]]",
                'path = "headers.csv"',
                'kb_unit = "ft"',
                'xy_unit = "m"',
                "bogus = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="bogus"):
        prepare_edges(toml)


def test_missing_depth_ref(tmp_path: Path) -> None:
    toml = tmp_path / "project.toml"
    toml.write_text(
        "\n".join(
            [
                "[project]",
                'crs = "EPSG:32614"',
                "",
                "[[edges.headers]]",
                'path = "headers.csv"',
                'kb_unit = "ft"',
                'xy_unit = "m"',
                "",
                "[[edges.tops]]",
                'path = "tops.csv"',
                'depth_unit = "ft"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="depth_ref"):
        prepare_edges(toml)


def test_cli_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prog"]) == 2
    assert main(["prog", "a", "b"]) == 2
    captured = capsys.readouterr()
    assert "usage: python -m subsurfix.edges PROJECT_TOML" in captured.err
    assert captured.out == ""


def test_import_edges_without_gis_packages() -> None:
    code = (
        "import sys\n"
        "for name in ('pyproj', 'shapely', 'shapefile', 'openpyxl'):\n"
        "    sys.modules[name] = None\n"
        "import subsurfix.edges\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
