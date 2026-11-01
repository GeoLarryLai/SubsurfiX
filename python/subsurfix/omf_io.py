# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""OMF (Open Mining Format) import and export, for moving data to and from Leapfrog.

**Experimental.** The API may change.

Uses the ``omf`` 1.x package (OMF v1 file format), which is what Leapfrog's OMF
import and export read and write.

Plain-numpy structures are used in both directions:

* grid: ``{"origin": (x, y, z), "cell_size": (dx, dy, dz) or "tensors": (tu, tv, tw),
  "dims": (nx, ny, nz), "properties": {name: array of nx*ny*nz}}``. Cell order is
  i (x) fastest, then j, then k, the same as SubsurfiX's regular grids.
  A ``CornerPointGrid`` is also accepted; OMF v1 has no corner-point element, so it is
  written as a point set of cell centres carrying the properties.
* wells: list of ``{"name": str, "xyz": (n, 3) array, "logs": {name: array of n}}``,
  written as a line set (the trajectory) with log values on the vertices.
* surfaces: ``{name: {"x": (nx,), "y": (ny,), "z": (ny, nx)}}`` for gridded horizons,
  or ``{name: {"vertices": (n, 3), "triangles": (m, 3)}}`` for triangulated ones.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["export_omf", "import_omf"]

_EXPERIMENTAL = "subsurfix.omf_io is experimental; its API may change."


def _omf():
    import omf

    return omf


def _scalar(omf: Any, name: str, values: Any, location: str, n: int) -> Any:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(
            f"OMF data '{name}' has {arr.size} values; expected {n} ({location})"
        )
    return omf.ScalarData(name=str(name), array=arr, location=location)


def _grid_element(omf: Any, grid: Any, name: str) -> Any:
    if not isinstance(grid, dict):  # CornerPointGrid or anything with centers()
        centers = np.asarray(grid.centers(), dtype=float).reshape(-1, 3)
        props = getattr(grid, "omf_properties", None) or {}
        return omf.PointSetElement(
            name=name,
            description="SubsurfiX corner-point grid cell centres",
            geometry=omf.PointSetGeometry(vertices=centers),
            data=[
                _scalar(omf, k, v, "vertices", len(centers)) for k, v in props.items()
            ],
        )
    nx, ny, nz = (int(d) for d in grid["dims"])
    if "tensors" in grid:
        tu, tv, tw = (np.asarray(t, dtype=float) for t in grid["tensors"])
    else:
        dx, dy, dz = (float(c) for c in grid["cell_size"])
        tu, tv, tw = np.full(nx, dx), np.full(ny, dy), np.full(nz, dz)
    if (tu.size, tv.size, tw.size) != (nx, ny, nz):
        raise ValueError("grid tensors do not match dims")
    n = nx * ny * nz
    geom = omf.VolumeGridGeometry(
        origin=np.asarray(grid.get("origin", (0.0, 0.0, 0.0)), dtype=float),
        tensor_u=tu,
        tensor_v=tv,
        tensor_w=tw,
    )
    data = [
        _scalar(omf, k, v, "cells", n)
        for k, v in (grid.get("properties") or {}).items()
    ]
    return omf.VolumeElement(
        name=name, description="SubsurfiX property grid", geometry=geom, data=data
    )


def _well_element(omf: Any, well: dict) -> Any:
    xyz = np.asarray(well["xyz"], dtype=float).reshape(-1, 3)
    if len(xyz) < 2:
        raise ValueError(
            f"well '{well.get('name')}' needs at least 2 trajectory points"
        )
    seg = np.column_stack([np.arange(len(xyz) - 1), np.arange(1, len(xyz))]).astype(int)
    data = [
        _scalar(omf, k, v, "vertices", len(xyz))
        for k, v in (well.get("logs") or {}).items()
    ]
    return omf.LineSetElement(
        name=str(well.get("name", "well")),
        subtype="line",
        geometry=omf.LineSetGeometry(vertices=xyz, segments=seg),
        data=data,
    )


def _surface_element(omf: Any, name: str, s: dict) -> Any:
    if "triangles" in s:
        geom = omf.SurfaceGeometry(
            vertices=np.asarray(s["vertices"], dtype=float),
            triangles=np.asarray(s["triangles"], dtype=int),
        )
    else:
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        z = np.asarray(s["z"], dtype=float)
        if z.shape != (y.size, x.size):
            raise ValueError(f"surface '{name}': z must have shape (ny, nx)")
        if (
            x.size < 2
            or y.size < 2
            or np.any(np.diff(x) <= 0)
            or np.any(np.diff(y) <= 0)
        ):
            raise ValueError(
                f"surface '{name}': x and y must be increasing, length >= 2"
            )
        geom = omf.SurfaceGridGeometry(
            origin=[x[0], y[0], 0.0],
            tensor_u=np.diff(x),
            tensor_v=np.diff(y),
            offset_w=z.ravel(),  # x fastest
        )
    return omf.SurfaceElement(name=str(name), geometry=geom)


def export_omf(
    path: str | Path,
    grid: Any = None,
    wells: list[dict] | None = None,
    surfaces: dict[str, dict] | None = None,
    name: str = "SubsurfiX export",
) -> Path:
    """Write a grid, well trajectories with logs and horizon surfaces to an OMF v1 file.

    Experimental. Any of ``grid``, ``wells`` and ``surfaces`` may be omitted.
    Returns the written path.
    """
    warnings.warn(_EXPERIMENTAL, FutureWarning, stacklevel=2)
    omf = _omf()
    elements = []
    if grid is not None:
        elements.append(_grid_element(omf, grid, "grid"))
    elements += [_well_element(omf, w) for w in (wells or [])]
    elements += [_surface_element(omf, k, v) for k, v in (surfaces or {}).items()]
    if not elements:
        raise ValueError("export_omf: nothing to write")
    proj = omf.Project(name=name, elements=elements)
    proj.validate()
    path = Path(path)
    omf.OMFWriter(proj, str(path))
    return path


def _data(el: Any) -> dict[str, np.ndarray]:
    out = {}
    for d in el.data:
        arr = getattr(d.array, "array", d.array)
        try:
            out[d.name] = np.asarray(arr, dtype=float)
        except (TypeError, ValueError):
            out[d.name] = np.asarray(arr)
    return out


def import_omf(path: str | Path) -> dict[str, list[dict]]:
    """Read an OMF v1 file into plain numpy structures.

    Experimental. Returns ``{"block_models": [...], "surfaces": [...], "wells": [...],
    "points": [...]}``. Block models have ``origin``, ``tensors``, ``dims``, ``axes``
    and ``properties`` (i fastest); gridded surfaces have ``x``, ``y``, ``z`` (ny, nx);
    triangulated surfaces have ``vertices`` and ``triangles``; line sets (wells,
    polylines) have ``xyz``, ``segments`` and ``logs``. Surfaces can serve as a
    structural framework, e.g. for ``CornerPointGrid.from_surfaces``.
    """
    warnings.warn(_EXPERIMENTAL, FutureWarning, stacklevel=2)
    omf = _omf()
    proj = omf.OMFReader(str(path)).get_project()
    origin0 = np.asarray(proj.origin, dtype=float)
    out: dict[str, list[dict]] = {
        "block_models": [],
        "surfaces": [],
        "wells": [],
        "points": [],
    }
    for el in proj.elements:
        g = el.geometry
        o = origin0 + np.asarray(g.origin, dtype=float)
        meta = {"name": el.name, "description": el.description}
        if isinstance(el, omf.VolumeElement):
            t = [
                np.asarray(x, dtype=float) for x in (g.tensor_u, g.tensor_v, g.tensor_w)
            ]
            out["block_models"].append(
                {
                    **meta,
                    "origin": o,
                    "tensors": tuple(t),
                    "dims": tuple(x.size for x in t),
                    "axes": np.array([g.axis_u, g.axis_v, g.axis_w], dtype=float),
                    "properties": _data(el),
                }
            )
        elif isinstance(el, omf.SurfaceElement):
            if isinstance(g, omf.SurfaceGridGeometry):
                x = o[0] + np.concatenate([[0.0], np.cumsum(g.tensor_u)])
                y = o[1] + np.concatenate([[0.0], np.cumsum(g.tensor_v)])
                w = getattr(g.offset_w, "array", g.offset_w)
                z = o[2] + (
                    np.zeros(x.size * y.size) if w is None else np.asarray(w, float)
                )
                if not (
                    np.allclose(g.axis_u, [1, 0, 0])
                    and np.allclose(g.axis_v, [0, 1, 0])
                ):
                    meta["warning"] = "rotated surface grid; x/y ignore axis_u/axis_v"
                out["surfaces"].append(
                    {
                        **meta,
                        "x": x,
                        "y": y,
                        "z": z.reshape(y.size, x.size),
                        "properties": _data(el),
                    }
                )
            else:
                out["surfaces"].append(
                    {
                        **meta,
                        "vertices": np.asarray(g.vertices.array, float) + o,
                        "triangles": np.asarray(g.triangles.array, int),
                        "properties": _data(el),
                    }
                )
        elif isinstance(el, omf.LineSetElement):
            out["wells"].append(
                {
                    **meta,
                    "xyz": np.asarray(g.vertices.array, float) + o,
                    "segments": np.asarray(g.segments.array, int),
                    "logs": _data(el),
                }
            )
        elif isinstance(el, omf.PointSetElement):
            out["points"].append(
                {
                    **meta,
                    "xyz": np.asarray(g.vertices.array, float) + o,
                    "properties": _data(el),
                }
            )
    return out
