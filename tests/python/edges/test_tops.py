# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from subsurfix.edges.ids import attach_well_ids
from subsurfix.edges.schemas import TOPS_SCHEMA, without_well_id
from subsurfix.edges.tops import TopsSource, load_tops
from subsurfix.errors import ConfigError, FormatError


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def _source(path: Path, **kwargs: object) -> TopsSource:
    values: dict[str, object] = {
        "path": str(path),
        "depth_ref": "md",
        "depth_unit": "ft",
        "layout": "long",
        "repeat_pick": "shallowest",
        "key_col": ("UWI",),
        "top_col": ("Top",),
        "depth_col": ("MD",),
    }
    values.update(kwargs)
    return TopsSource(**values)  # type: ignore[arg-type]


def test_repeat_picks(tmp_path: Path) -> None:
    path = tmp_path / "tops.csv"
    _write_csv(
        path,
        [
            ["UWI", "Top", "MD"],
            ["W1", "T1", "1000"],
            ["W1", "T1", "990"],
            ["W1", "T1", "990"],
            ["W1", "T2", ""],
        ],
    )
    with pytest.warns(UserWarning) as caught:
        shallow = load_tops([_source(path, repeat_pick="shallowest")])
    text = " ".join(str(item.message) for item in caught)
    assert "repeated picks: 2" in text
    assert "rows with NaN depth: 1" in text
    assert shallow.column("well_key").to_pylist() == ["W1"]
    assert shallow.column("top_name").to_pylist() == ["T1"]
    assert shallow.column("depth_ft").to_pylist() == [990.0]
    assert shallow.column("depth_ref").to_pylist() == [0]

    with pytest.warns(UserWarning):
        first = load_tops([_source(path, repeat_pick="first")])
    assert first.column("depth_ft").to_pylist() == [1000.0]

    with pytest.raises(FormatError):
        load_tops([_source(path, repeat_pick="error")])

    with pytest.warns(UserWarning):
        elev = load_tops([_source(path, repeat_pick="shallowest", depth_ref="elev")])
    assert elev.column("depth_ft").to_pylist() == [1000.0]
    assert elev.column("depth_ref").to_pylist() == [3]


def test_tops_metres_to_feet(tmp_path: Path) -> None:
    path = tmp_path / "tops.csv"
    _write_csv(path, [["UWI", "Top", "MD"], ["W1", "T1", "304.8"]])
    table = load_tops([_source(path, depth_unit="m")])
    assert table.column("depth_ft").to_pylist() == [1000.0]


def test_depth_column_suffix_must_match_unit(tmp_path: Path) -> None:
    path = tmp_path / "tops.csv"
    _write_csv(path, [["UWI", "Top", "MD_ft"], ["W1", "T1", "10"]])
    with pytest.raises(ConfigError, match="MD_ft"):
        load_tops([_source(path, depth_unit="m", depth_col=("MD_ft",))])


def test_later_tops_source_does_not_replace(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    _write_csv(first, [["UWI", "Top", "MD"], ["W1", "T1", "100"]])
    _write_csv(second, [["UWI", "Top", "MD"], ["W1", "T1", "50"], ["W1", "T2", "80"]])
    with pytest.warns(UserWarning, match="earlier source"):
        table = load_tops([_source(first), _source(second)])
    rows = list(
        zip(
            table.column("top_name").to_pylist(),
            table.column("depth_ft").to_pylist(),
            table.column("source").to_pylist(),
            strict=True,
        )
    )
    assert rows == [("T1", 100.0, "a.csv"), ("T2", 80.0, "b.csv")]


def test_wide_layout_and_empty_tops(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    _write_csv(
        path,
        [
            ["UWI", "T1", "T2", "Skip"],
            ["W1", "10", "20", "5"],
            ["W1", "12", "", "7"],
        ],
    )
    with pytest.warns(UserWarning, match="repeated picks: 1"):
        table = load_tops(
            [
                TopsSource(
                    path=str(path),
                    depth_ref="md",
                    depth_unit="ft",
                    layout="wide",
                    key_col=("UWI",),
                    exclude_columns=("Skip",),
                )
            ]
        )
    assert table.column("top_name").to_pylist() == ["T1", "T2"]
    assert table.column("depth_ft").to_pylist() == [10.0, 20.0]

    empty = load_tops([])
    assert empty.num_rows == 0
    assert empty.schema.equals(without_well_id(TOPS_SCHEMA), check_metadata=False)
    attached = attach_well_ids(empty, {})
    assert attached.num_rows == 0
    assert attached.schema.equals(TOPS_SCHEMA, check_metadata=False)
