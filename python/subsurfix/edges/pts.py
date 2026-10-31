# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Pressure-temperature-spinner (PTS) surveys and other temperature tables.

A PTS file often holds several passes, one per sheet (run in hole, pull out of
hole, a static bottom-hole station), each with a title row above the column
names. Every sheet is read; the column-name row is the first row naming a
temperature column. Units come from the source settings, else from the column
names (``Temperature (F)``, ``Pressure (psia)``, ``Depth(Feet)``); a temperature
or pressure column with no known unit is a ``ConfigError``.

A sheet with no depth column (a station recorded against time) has no depth
tie-in: its rows are kept with NaN ``md_ft`` and ``depth_tied = False``, and a
warning names the sheet.
"""
from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from subsurfix.edges._tables import read_rows, source_name
from subsurfix.edges.schemas import PTS_SCHEMA, array_table, without_well_id
from subsurfix.errors import ConfigError, FormatError, NotFoundError

_BODY = without_well_id(PTS_SCHEMA)
_PSI_TO_MPA = 0.006894757293168361
_ATM_MPA = 0.101325
_TEMP_UNITS = {"degc": "degC", "c": "degC", "degf": "degF", "f": "degF", "k": "K"}
_PRES_UNITS = {"psia": "psia", "psig": "psig", "psi": "psia", "kpa": "kPa", "mpa": "MPa", "bar": "bar"}
_DEPTH_UNITS = {"ft": "ft", "feet": "ft", "foot": "ft", "f": "ft", "m": "m", "meter": "m", "meters": "m", "metres": "m"}


@dataclass(frozen=True, kw_only=True)
class PtsSource:
    """One PTS or temperature table (CSV, XLSX, or XLS with python-calamine).

    ``well`` names the well (its header key) when the file has no well column;
    ``key_col`` names such a column. ``sheets`` limits the sheets read (all by
    default). Units (``temp_unit`` degC/degF/K, ``pressure_unit``
    psia/psig/kPa/MPa/bar, ``depth_unit`` ft/m) override the column names.
    """

    path: str
    well: str | None = None
    key_col: str | None = None
    sheets: tuple[str, ...] = ()
    delimiter: str = ","
    temp_unit: str | None = None
    pressure_unit: str | None = None
    depth_unit: str | None = None


def load_pts(sources: Sequence[PtsSource]) -> pa.Table:
    """Load PTS rows: ``well_key``, ``pass_name``, ``seq``, ``md_ft``, ``temp_c``,
    ``pressure_mpa`` (absolute), ``flow``, ``time``, ``depth_tied``, ``source``.

    NaN rule: blank or unparsable numbers are NaN; a row with neither a finite
    temperature nor a finite pressure is dropped. No sources: empty table.
    """
    cols: dict[str, list[object]] = {f.name: [] for f in _BODY}
    for source in sources:
        if (source.well is None) == (source.key_col is None):
            raise ConfigError(f"{source_name(source.path)}: set exactly one of well or key_col")
        for sheet, rows in _sheets(source):
            _read_sheet(source, sheet, rows, cols)
    return array_table(_BODY, cols)


def _sheets(source: PtsSource) -> list[tuple[str, list[list[object]]]]:
    path = Path(source.path)
    name = path.name
    if not path.is_file():
        raise NotFoundError(f"{name}: not found")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        header, rows = read_rows(path, None, source.delimiter)
        return [(path.stem, [list(header), *rows])]
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            from python_calamine import CalamineWorkbook
        except ImportError:
            if suffix == ".xls":
                raise ConfigError(
                    f"{name}: .xls needs the python-calamine package; install it or save as .xlsx/.csv"
                ) from None
            return [
                (sheet, [list(h), *r])
                for sheet in _xlsx_sheet_names(path, source.sheets)
                for h, r in [read_rows(path, sheet, ",")]
            ]
        book = CalamineWorkbook.from_path(str(path))
        names = [s for s in book.sheet_names if not source.sheets or s in source.sheets]
        missing = sorted(set(source.sheets) - set(book.sheet_names))
        if missing:
            raise ConfigError(f"{name}: no sheet {missing[0]!r}; sheets are {book.sheet_names}")
        return [(s, [list(r) for r in book.get_sheet_by_name(s).to_python()]) for s in names]
    raise FormatError(f"{name}: unsupported table type {suffix or '(none)'}")


def _xlsx_sheet_names(path: Path, wanted: tuple[str, ...]) -> list[str]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True)
    try:
        return [s for s in book.sheetnames if not wanted or s in wanted]
    finally:
        book.close()


def _unit_in(name: str, table: dict[str, str]) -> str | None:
    """Unit named in a column name: ``(F)``, ``, F``, ``_degC``, ``[psia]``."""
    tokens = re.findall(r"[A-Za-z]+", name.replace("°", "deg"))
    for token in reversed(tokens[1:] if len(tokens) > 1 else []):
        unit = table.get(token.casefold())
        if unit is not None:
            return unit
    return None


def _find(names: list[str], *needles: str) -> int | None:
    folded = [n.casefold() for n in names]
    for needle in needles:
        for i, n in enumerate(folded):
            if needle in n:
                return i
    return None


def _num(value: object) -> float:
    if value is None or isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return math.nan


def _read_sheet(
    source: PtsSource, sheet: str, rows: list[list[object]], cols: dict[str, list[object]]
) -> None:
    where = f"{source_name(source.path)} [{sheet}]"
    head_i = next(
        (i for i, r in enumerate(rows) if any("temp" in str(c).casefold() for c in r if c is not None)),
        None,
    )
    if head_i is None:
        warnings.warn(f"{where}: no temperature column; sheet skipped", UserWarning, stacklevel=3)
        return
    names = ["" if c is None else str(c).strip() for c in rows[head_i]]
    t_i = _find(names, "temp")
    p_i = _find(names, "press")
    d_i = _find(names, "depth", "md")
    f_i = _find(names, "flow", "spinner", "rpm")
    time_i = _find(names, "time")
    k_i = None
    if source.key_col is not None:
        k_i = _find(names, source.key_col.casefold())
        if k_i is None:
            raise FormatError(f"{where}: no {source.key_col!r} column; columns are {names}")
    assert t_i is not None
    t_unit = _norm(source.temp_unit, _TEMP_UNITS, "temp_unit", where) or _unit_in(names[t_i], _TEMP_UNITS)
    if t_unit is None:
        raise ConfigError(f"{where}: unit of {names[t_i]!r} unknown; set temp_unit (degC, degF or K)")
    p_unit = None
    if p_i is not None:
        p_unit = _norm(source.pressure_unit, _PRES_UNITS, "pressure_unit", where) or _unit_in(
            names[p_i], _PRES_UNITS
        )
        if p_unit is None:
            raise ConfigError(
                f"{where}: unit of {names[p_i]!r} unknown; set pressure_unit (psia, psig, kPa, MPa, bar)"
            )
    d_unit = None
    if d_i is not None:
        d_unit = _norm(source.depth_unit, _DEPTH_UNITS, "depth_unit", where) or _unit_in(
            names[d_i], _DEPTH_UNITS
        )
        if d_unit is None:
            raise ConfigError(f"{where}: unit of {names[d_i]!r} unknown; set depth_unit (ft or m)")
    else:
        warnings.warn(
            f"{where}: no depth column, so these readings have no depth tie-in; kept with NaN md_ft",
            UserWarning,
            stacklevel=3,
        )
    seq = 0
    for r in rows[head_i + 1 :]:
        cell = lambda i: r[i] if i is not None and i < len(r) else None  # noqa: E731
        t = _num(cell(t_i))
        p = _num(cell(p_i))
        if not (math.isfinite(t) or math.isfinite(p)):
            continue
        key = source.well if k_i is None else cell(k_i)
        if key is None or str(key).strip() == "":
            continue
        d = _num(cell(d_i))
        cols["well_key"].append(str(key).strip())
        cols["pass_name"].append(sheet)
        cols["seq"].append(seq)
        cols["md_ft"].append(d / 0.3048 if d_unit == "m" else d)
        cols["temp_c"].append(_to_c(t, t_unit))
        cols["pressure_mpa"].append(_to_mpa(p, p_unit))
        cols["flow"].append(_num(cell(f_i)))
        tv = cell(time_i)
        cols["time"].append("" if tv is None else str(tv))
        cols["depth_tied"].append(d_i is not None and math.isfinite(d))
        cols["source"].append(f"{source_name(source.path)}:{sheet}")
        seq += 1


def _norm(value: str | None, table: dict[str, str], key: str, where: str) -> str | None:
    if value is None:
        return None
    unit = table.get(value.casefold())
    if unit is None:
        raise ConfigError(f"{where}: {key} {value!r} not one of {sorted(set(table.values()))}")
    return unit


def _to_c(t: float, unit: str) -> float:
    if unit == "degF":
        return (t - 32.0) / 1.8
    if unit == "K":
        return t - 273.15
    return t


def _to_mpa(p: float, unit: str | None) -> float:
    if unit is None or not math.isfinite(p):
        return math.nan
    return {
        "psia": p * _PSI_TO_MPA,
        "psig": p * _PSI_TO_MPA + _ATM_MPA,
        "kPa": p / 1000.0,
        "MPa": p,
        "bar": p / 10.0,
    }[unit]
