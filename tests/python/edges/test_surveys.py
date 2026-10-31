# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from subsurfix.edges.schemas import SURVEYS_SCHEMA, without_well_id
from subsurfix.edges.surveys import SurveySource, load_surveys
from subsurfix.errors import FormatError


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def _source(path: Path, **kwargs: object) -> SurveySource:
    values: dict[str, object] = {
        "path": str(path),
        "md_unit": "ft",
        "key_col": ("UWI",),
        "md_col": ("MD",),
        "inc_col": ("Inc",),
        "azi_col": ("Azi",),
    }
    values.update(kwargs)
    return SurveySource(**values)  # type: ignore[arg-type]


def test_azimuth_wraps_into_0_360(tmp_path: Path) -> None:
    path = tmp_path / "survey.csv"
    _write_csv(
        path,
        [
            ["UWI", "MD", "Inc", "Azi"],
            ["W1", "0", "0", "-10"],
            ["W1", "100", "1", "370"],
        ],
    )
    table = load_surveys([_source(path)])
    assert table.column("azi_deg").to_pylist() == [350.0, 10.0]
    assert table.column("md_ft").to_pylist() == [0.0, 100.0]
    assert "well_id" not in table.column_names


def test_survey_metres_duplicate_md_and_bad_angles(tmp_path: Path) -> None:
    metres = tmp_path / "metres.csv"
    _write_csv(metres, [["UWI", "MD", "Inc", "Azi"], ["W1", "304.8", "0", "0"]])
    table = load_surveys([_source(metres, md_unit="m")])
    assert table.column("md_ft").to_pylist() == [1000.0]

    repeated = tmp_path / "repeated.csv"
    _write_csv(
        repeated,
        [
            ["UWI", "MD", "Inc", "Azi"],
            ["W1", "100", "1", "0"],
            ["W1", "100", "5", "90"],
        ],
    )
    with pytest.warns(UserWarning, match="repeated md: 1"):
        table = load_surveys([_source(repeated)])
    assert table.column("inc_deg").to_pylist() == [1.0]

    steep = tmp_path / "steep.csv"
    _write_csv(steep, [["UWI", "MD", "Inc", "Azi"], ["W1", "10", "181", "0"]])
    with pytest.raises(FormatError, match="inclination"):
        load_surveys([_source(steep)])

    missing = tmp_path / "missing.csv"
    _write_csv(missing, [["UWI", "MD", "Inc", "Azi"], ["W1", "10", "1", ""]])
    with pytest.raises(FormatError, match="inclination or azimuth missing"):
        load_surveys([_source(missing)])


def test_first_survey_source_owns_the_well(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    _write_csv(
        first,
        [["UWI", "MD", "Inc", "Azi"], ["W1", "0", "0", "0"], ["W1", "10", "1", "5"]],
    )
    _write_csv(
        second,
        [["UWI", "MD", "Inc", "Azi"], ["W1", "5", "2", "9"], ["W2", "0", "0", "1"]],
    )
    with pytest.warns(UserWarning, match="earlier source"):
        table = load_surveys([_source(first), _source(second)])
    rows = list(
        zip(
            table.column("well_key").to_pylist(),
            table.column("md_ft").to_pylist(),
            table.column("source").to_pylist(),
            strict=True,
        )
    )
    assert rows == [("W1", 0.0, "a.csv"), ("W1", 10.0, "a.csv"), ("W2", 0.0, "b.csv")]


def test_empty_surveys() -> None:
    table = load_surveys([])
    assert table.num_rows == 0
    assert table.schema.equals(without_well_id(SURVEYS_SCHEMA), check_metadata=False)
