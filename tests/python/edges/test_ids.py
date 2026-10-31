# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from subsurfix.edges.headers import HeaderSource, load_headers
from subsurfix.edges.ids import assign_well_ids, attach_well_ids
from subsurfix.edges.schemas import HEADERS_SCHEMA
from subsurfix.errors import ConfigError


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_assign_well_ids_keeps_existing_ids() -> None:
    ids = assign_well_ids(["B2", "A1", "C3", "A1"])
    assert ids == {"A1": 0, "B2": 1, "C3": 2}
    again = assign_well_ids(["A0", "D4", "B2"], ids)
    assert again["A0"] == 3
    assert again["D4"] == 4
    assert again["A1"] == 0
    assert again["B2"] == 1
    assert again["C3"] == 2
    assert set(again) == {"A0", "A1", "B2", "C3", "D4"}


def test_assign_rejects_duplicate_existing_ids() -> None:
    with pytest.raises(ConfigError):
        assign_well_ids(["A"], {"A": 1, "B": 1})


def test_attach_sorts_by_well_id(tmp_path: Path) -> None:
    path = tmp_path / "headers.csv"
    _write_csv(path, [["UWI", "X", "Y", "KB"], ["B", "2", "2", "1"], ["A", "1", "1", "1"]])
    headers = load_headers(
        [
            HeaderSource(
                path=str(path),
                kb_unit="ft",
                xy_unit="m",
                key_col=("UWI",),
                x_col=("X",),
                y_col=("Y",),
                kb_col=("KB",),
                kb_match=None,
                label_col=(),
                datum_col=(),
            )
        ],
        "EPSG:32614",
    )
    attached = attach_well_ids(headers, {"B": 0, "A": 1})
    assert attached.column("well_id").to_pylist() == [0, 1]
    assert attached.column("well_key").to_pylist() == ["B", "A"]
    assert attached.schema.equals(HEADERS_SCHEMA, check_metadata=False)
    assert attached.schema.field("well_id").nullable is False


def test_attach_drops_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "headers.csv"
    _write_csv(path, [["UWI", "X", "Y", "KB"], ["A", "1", "1", "1"], ["Z", "2", "2", "1"]])
    headers = load_headers(
        [
            HeaderSource(
                path=str(path),
                kb_unit="ft",
                xy_unit="m",
                key_col=("UWI",),
                x_col=("X",),
                y_col=("Y",),
                kb_col=("KB",),
                kb_match=None,
                label_col=(),
                datum_col=(),
            )
        ],
        "EPSG:32614",
    )
    with pytest.warns(UserWarning, match="Z"):
        attached = attach_well_ids(headers, {"A": 0})
    assert attached.column("well_key").to_pylist() == ["A"]
