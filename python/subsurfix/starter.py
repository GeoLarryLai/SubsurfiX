# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Project starters: a project-file template and a small synthetic demo project.

``template_toml()`` returns the commented project file with every key at its
default. ``init_project()`` writes it into a folder with your CRS and LAS
folder filled in. ``write_demo_project()`` writes a SYNTHETIC project (made-up
wells, logs and tops; not real data) so the whole pipeline can be tried with
no downloads.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from subsurfix import _native

__all__ = ["template_toml", "init_project", "write_demo_project"]


def template_toml() -> str:
    """The commented project-file template; every key shown has its default."""
    return _native.template_toml()


def init_project(
    folder: str | os.PathLike[str],
    crs: str = "EPSG:32614",
    las_dirs: tuple[str, ...] | list[str] = ("las",),
    overwrite: bool = False,
) -> Path:
    """Write ``folder/project.toml`` from the template and return its path.

    ``crs`` is the project coordinate reference system (CRS); ``las_dirs`` are
    LAS folders relative to ``folder``. The ``[[edges.*]]`` sections stay
    commented: point them at your header, tops and survey tables before
    running ``prepare_edges``. Raises ``FileExistsError`` unless ``overwrite``.
    """
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "project.toml"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out}: exists (pass overwrite=True)")
    text = template_toml()
    text = text.replace('crs = "EPSG:32614"', f"crs = {_q(crs)}", 1)
    dirs = ", ".join(_q(x) for x in las_dirs)
    text, n = re.subn(r"(?m)^las_dirs = \[\]", f"las_dirs = [{dirs}]", text, count=1)
    if n != 1:
        raise RuntimeError("template has no `las_dirs = []` line")
    out.write_text(text, encoding="utf-8")
    for x in las_dirs:
        (d / x).mkdir(parents=True, exist_ok=True)
    return out


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


_TOPS = ("SYN_TOP_A", "SYN_TOP_B", "SYN_TOP_C")
# Layer properties above A, A-B, B-C, below C: GR (gAPI), RHOB (g/cm3), NPHI (v/v), DT (us/ft), RESD (ohm.m).
_LAYERS = (
    (95.0, 2.45, 0.30, 95.0, 3.0),
    (35.0, 2.62, 0.08, 58.0, 60.0),
    (110.0, 2.50, 0.27, 88.0, 4.0),
    (25.0, 2.68, 0.05, 52.0, 150.0),
)


def write_demo_project(
    folder: str | os.PathLike[str],
    n_wells: int = 9,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Write a SYNTHETIC demo project and return the path of its project file.

    ``n_wells`` vertical wells (at least 6) on a 500 m grid, LAS files with
    GR, RHOB, NPHI, DT and RESD over 2000-4000 ft, three gently dipping tops
    (``SYN_TOP_A/B/C``) and a header table. One well has no RHOB below
    3000 ft so the fill stage has a gap to fill. All values are made up.
    """
    if n_wells < 6:
        raise ValueError("n_wells must be at least 6")
    d = Path(folder)
    toml = init_project(d, crs="EPSG:32614", las_dirs=("las",), overwrite=overwrite)
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_wells)))
    md = np.arange(2000.0, 4000.0 + 0.25, 0.5)
    heads = ["UWI,Name,X,Y,KB_ft"]
    tops = ["UWI,Top,MD"]
    for i in range(n_wells):
        x = 500000.0 + 500.0 * (i % side)
        y = 4100000.0 + 500.0 * (i // side)
        uwi = f"99{i + 1:012d}"
        kb = 1500.0 + 5.0 * (i % 3)
        dip = 0.02 * (x - 500000.0) / 0.3048 + 0.01 * (y - 4100000.0) / 0.3048
        tz = np.array([2600.0, 3000.0, 3400.0]) + dip + rng.normal(0.0, 5.0, 3) + np.array([0.0, 10.0, 20.0]) * (i % 2)
        layer = np.searchsorted(tz, md)
        props = np.array(_LAYERS)[layer]
        wig = np.sin(md / 37.0 + i) * 0.5 + rng.normal(0.0, 1.0, (md.size, 1)).ravel() * 0.5
        gr = props[:, 0] * (1.0 + 0.12 * wig)
        rhob = props[:, 1] + 0.03 * wig
        nphi = np.clip(props[:, 2] + 0.02 * wig, 0.0, None)
        dt = props[:, 3] * (1.0 + 0.04 * wig)
        resd = props[:, 4] * np.exp(0.3 * wig)
        if i == 1:
            rhob = np.where(md > 3000.0, np.nan, rhob)
        cols = [md, gr, rhob, nphi, dt, resd]
        name = f"SYN-{i + 1:02d}"
        _write_las(d / "las" / f"{uwi}_{name}.las", uwi, name, cols)
        heads.append(f"{uwi},{name},{x:.1f},{y:.1f},{kb:.1f}")
        for t, z in zip(_TOPS, tz):
            tops.append(f"{uwi},{t},{z:.1f}")
    (d / "headers.csv").write_text("\n".join(heads) + "\n", encoding="utf-8")
    (d / "tops.csv").write_text("\n".join(tops) + "\n", encoding="utf-8")
    text = toml.read_text(encoding="utf-8")
    edges = (
        "\n# SYNTHETIC demo inputs written by subsurfix.write_demo_project (made-up data).\n"
        "[[edges.headers]]\n"
        'path = "headers.csv"\nkb_unit = "ft"\nxy_unit = "m"\n'
        'key_col = ["UWI"]\nx_col = ["X"]\ny_col = ["Y"]\nlabel_col = ["Name"]\nkb_col = ["KB_ft"]\n'
        "[[edges.tops]]\n"
        'path = "tops.csv"\ndepth_ref = "md"\ndepth_unit = "ft"\n'
        'key_col = ["UWI"]\ntop_col = ["Top"]\ndepth_col = ["MD"]\n'
    )
    text = _insert_edges(text, edges)
    toml.write_text(text, encoding="utf-8")
    return toml


def _insert_edges(text: str, edges: str) -> str:
    """Append the array-of-tables right after the ``[edges]`` table's keys."""
    m = re.search(r"(?m)^\[edges\]\n(?:(?!\[).*\n)*", text)
    if m is None:
        return text + edges
    return text[: m.end()] + edges + "\n" + text[m.end():]


def _write_las(path: Path, uwi: str, name: str, cols: list[np.ndarray]) -> None:
    md = cols[0]
    lines = [
        "~VERSION INFORMATION",
        " VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
        " WRAP.   NO  : ONE LINE PER DEPTH STEP",
        "~WELL INFORMATION",
        f" STRT.FT {md[0]:.1f} : START DEPTH",
        f" STOP.FT {md[-1]:.1f} : STOP DEPTH",
        f" STEP.FT {md[1] - md[0]:.1f} : STEP",
        " NULL.   -999.25 : NULL VALUE",
        f" WELL.   {name} : WELL (SYNTHETIC)",
        f" UWI .   {uwi} : UNIQUE WELL ID",
        "~CURVE INFORMATION",
        " DEPT.FT     : Measured depth",
        " GR  .GAPI   : Gamma ray (synthetic)",
        " RHOB.G/CC   : Bulk density (synthetic)",
        " NPHI.V/V    : Neutron porosity (synthetic)",
        " DT  .US/F   : Compressional slowness (synthetic)",
        " RESD.OHMM   : Deep resistivity (synthetic)",
        "~A",
    ]
    data = np.column_stack(cols)
    data = np.where(np.isnan(data), -999.25, data)
    lines += [" ".join(f"{v:.4f}" for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
