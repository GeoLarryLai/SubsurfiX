# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence
from pathlib import Path

from subsurfix.errors import ConfigError, FormatError, MissingDependencyError, NotFoundError


def source_name(path: str | os.PathLike[str]) -> str:
    """File name used in errors and provenance columns."""
    return Path(path).name


def init_source(obj: object, tuple_fields: Sequence[str]) -> None:
    """Store ``path`` as ``str`` and sequence fields as ``tuple[str, ...]``."""
    raw_path = getattr(obj, "path")
    try:
        path_text = os.fspath(raw_path)
    except TypeError as exc:
        raise ConfigError("path must be a file path") from exc
    if not isinstance(path_text, str):
        path_text = os.fsdecode(path_text)
    object.__setattr__(obj, "path", path_text)
    name = source_name(path_text)
    for field in tuple_fields:
        value = getattr(obj, field)
        if value is None:
            continue
        try:
            frozen = tuple(value)
        except TypeError as exc:
            raise ConfigError(f"{name}: {field} must be a sequence of strings") from exc
        if any(not isinstance(item, str) for item in frozen):
            raise ConfigError(f"{name}: {field} must be a sequence of strings")
        object.__setattr__(obj, field, frozen)


def soft_warning(source: str, kind: str, examples: Sequence[object]) -> str | None:
    """One warning message: count and the first five examples."""
    if not examples:
        return None
    shown = list(examples[:5])
    return f"{source}: {kind}: {len(examples)}; first {shown}"


def read_rows(
    path: str | os.PathLike[str],
    sheet: str | None,
    delimiter: str,
) -> tuple[list[str], list[list[object]]]:
    """Read a CSV/TXT/TSV or XLSX table.

    NaN rule: not applicable (cells stay as text or Excel values).
    Blank data rows are skipped. A missing file is ``NotFoundError``.
    """
    file_path = Path(path)
    name = file_path.name
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ConfigError(f"{name}: delimiter must be a single character")
    if not file_path.is_file():
        raise NotFoundError(f"{name}: not found")
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return _read_delimited(file_path, delimiter, name)
    if suffix in {".xlsx", ".xlsm"}:
        return _read_xlsx(file_path, sheet, name)
    raise FormatError(f"{name}: unsupported table type {suffix or '(none)'}")


def find_column(
    names: Sequence[str],
    candidates: Sequence[str],
    match: str | None = None,
    *,
    required: bool,
    what: str,
    source: str,
) -> int | None:
    """Return the first matching column.

    Exact case-insensitive equality wins over a substring ``match``.
    NaN rule: not applicable.
    """
    folded = [name.casefold() for name in names]
    for candidate in candidates:
        target = candidate.casefold()
        for index, name in enumerate(folded):
            if name == target:
                return index
    if match is not None:
        needle = match.casefold()
        for index, name in enumerate(folded):
            if needle in name:
                return index
    if not required:
        return None
    raise FormatError(
        f"{source}: no {what} column; tried {candidates} and names containing {match!r}; "
        f"columns are {list(names)}"
    )


def cell_to_key(value: object) -> str | None:
    """Parse a well key as text.

    NaN rule: not applicable. Blank is ``None``. Non-text values raise
    ``FormatError`` (callers add the row number and file name).
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool):
        raise FormatError(f"well key {value!r} is not text")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        raise FormatError(f"well key {value!r} is not text")
    raise FormatError(f"well key {value!r} is not text")


def cell_to_float(value: object) -> float:
    """Parse a number.

    NaN rule: blank, non-finite, and unparsable values are NaN (propagate as NaN).
    """
    number, _unparsable = _parse_number(value)
    return number


def cell_to_text(value: object) -> str:
    """Parse a text cell. ``None`` becomes an empty string.

    NaN rule: not applicable.
    """
    if value is None:
        return ""
    return str(value).strip()


def _parse_number(value: object) -> tuple[float, bool]:
    """Return ``(number, unparsable)``. Blank is NaN and not unparsable."""
    if value is None:
        return math.nan, False
    if isinstance(value, bool):
        return math.nan, True
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number, False
        return math.nan, True
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return math.nan, False
        try:
            number = float(text)
        except ValueError:
            return math.nan, True
        if math.isfinite(number):
            return number, False
        return math.nan, True
    return math.nan, True


def _header_name(cell: object) -> str:
    if cell is None:
        return ""
    return str(cell).strip()


def _is_blank_cell(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _read_delimited(
    path: Path,
    delimiter: str,
    name: str,
) -> tuple[list[str], list[list[object]]]:
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise NotFoundError(f"{name}: {exc}") from exc
    with handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header: list[str] | None = None
        rows: list[list[object]] = []
        for line_no, row in enumerate(reader, start=1):
            if header is None:
                header = [_header_name(cell) for cell in row]
                continue
            if len(row) > len(header):
                raise FormatError(
                    f"{name}: line {line_no} has {len(row)} values but the header has {len(header)}"
                )
            if len(row) < len(header):
                padded: list[object] = list(row)
                padded.extend([None] * (len(header) - len(row)))
                row = padded
            if all(_is_blank_cell(cell) for cell in row):
                continue
            rows.append(list(row))
    if header is None:
        raise FormatError(f"{name}: no header row")
    return header, rows


def _read_xlsx(
    path: Path,
    sheet: str | None,
    name: str,
) -> tuple[list[str], list[list[object]]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise MissingDependencyError(
            "openpyxl is needed to read .xlsx tables; it is a regular subsurfix dependency, so run `pip install openpyxl` or reinstall subsurfix"
        ) from exc
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except MissingDependencyError:
        raise
    except Exception as exc:
        raise FormatError(f"{name}: {exc}") from exc
    try:
        if sheet is None:
            worksheet = workbook.worksheets[0]
        elif sheet not in workbook.sheetnames:
            raise NotFoundError(
                f"{name}: sheet {sheet!r} not found; sheets are {list(workbook.sheetnames)}"
            )
        else:
            worksheet = workbook[sheet]
        header: list[str] | None = None
        rows: list[list[object]] = []
        for line_no, raw in enumerate(worksheet.iter_rows(values_only=True), start=1):
            row = list(raw)
            if header is None:
                header = [_header_name(cell) for cell in row]
                continue
            if len(row) > len(header):
                raise FormatError(
                    f"{name}: line {line_no} has {len(row)} values but the header has {len(header)}"
                )
            if len(row) < len(header):
                row.extend([None] * (len(header) - len(row)))
            if all(_is_blank_cell(cell) for cell in row):
                continue
            rows.append(row)
    finally:
        workbook.close()
    if header is None:
        raise FormatError(f"{name}: no header row")
    return header, rows
