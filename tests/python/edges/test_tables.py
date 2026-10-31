# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subsurfix.edges._tables import cell_to_key, find_column, read_rows
from subsurfix.errors import FormatError, MissingDependencyError, NotFoundError


def test_cell_to_key_examples() -> None:
    assert cell_to_key(15007012340000) == "15007012340000"
    assert cell_to_key(15007012340000.0) == "15007012340000"
    assert cell_to_key(" 0042 ") == "0042"
    assert cell_to_key(None) is None
    with pytest.raises(FormatError):
        cell_to_key(1.5)
    with pytest.raises(FormatError):
        cell_to_key(True)


def test_find_column_exact_beats_substring_and_is_case_insensitive() -> None:
    names = ["Surface X", "X", "uwi"]
    assert find_column(names, ["X"], "x", required=True, what="x", source="t.csv") == 1
    assert find_column(names, ["UWI"], required=True, what="key", source="t.csv") == 2
    assert find_column(["Surface X"], ["Nope"], "x", required=True, what="x", source="t.csv") == 0
    assert find_column(["A"], ["Nope"], required=False, what="nope", source="t.csv") is None
    with pytest.raises(FormatError) as caught:
        find_column(["A"], ["Nope"], required=True, what="nope", source="t.csv")
    assert str(caught.value).startswith("t.csv:")


def test_csv_pads_short_rows_skips_blanks_and_rejects_long_rows(tmp_path: Path) -> None:
    padded = tmp_path / "pad.csv"
    padded.write_text("A,B\n1\n", encoding="utf-8")
    header, rows = read_rows(padded, None, ",")
    assert header == ["A", "B"]
    assert rows == [["1", None]]

    blanks = tmp_path / "blanks.csv"
    blanks.write_text("A,B\n\n1,2\n,\n3,4\n", encoding="utf-8")
    _header, rows = read_rows(blanks, None, ",")
    assert rows == [["1", "2"], ["3", "4"]]

    long_row = tmp_path / "long.csv"
    long_row.write_text("A,B\n1\n1,2,3\n", encoding="utf-8")
    with pytest.raises(FormatError, match="line 3"):
        read_rows(long_row, None, ",")


def test_missing_and_unsupported_tables(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        read_rows(tmp_path / "missing.csv", None, ",")
    xls = tmp_path / "old.xls"
    xls.write_bytes(b"not a workbook")
    with pytest.raises(FormatError, match="unsupported table type"):
        read_rows(xls, None, ",")


def test_missing_openpyxl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    path = tmp_path / "t.xlsx"
    path.write_bytes(b"")
    with pytest.raises(MissingDependencyError, match=r"pip install openpyxl"):
        read_rows(path, None, ",")


def test_xlsx_stores_integer_keys_as_int_and_float(tmp_path: Path) -> None:
    import openpyxl

    path = tmp_path / "keys.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "UWI"
    sheet["B1"] = "X"
    sheet["A2"] = 15007012340000
    sheet["B2"] = 1
    sheet["A3"] = 15007012340000.0
    sheet["B3"] = 2
    assert type(sheet["A2"].value) is int
    assert type(sheet["A3"].value) is float
    workbook.save(path)
    workbook.close()
    _header, rows = read_rows(path, None, ",")
    assert cell_to_key(rows[0][0]) == "15007012340000"
    assert cell_to_key(rows[1][0]) == "15007012340000"
