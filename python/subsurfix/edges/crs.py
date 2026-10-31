# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import warnings

import numpy as np

from subsurfix.errors import ConfigError, FormatError, MissingDependencyError


def _pyproj(task: str):
    try:
        import pyproj
    except ImportError as exc:
        raise MissingDependencyError(
            f"pyproj is needed to {task}; install subsurfix[gis]"
        ) from exc
    return pyproj


def project_crs(spec: str):
    """Return a projected metre CRS.

    NaN rule: not applicable. An unusable spec is ``ConfigError``.
    """
    pyproj = _pyproj("read a project CRS")
    try:
        crs = pyproj.CRS.from_user_input(spec)
    except Exception as exc:
        raise ConfigError(f"invalid crs {spec!r}: {exc}") from exc
    units_ok = crs.is_projected and all(
        axis.unit_name in {"metre", "meter"} for axis in crs.axis_info
    )
    if not units_ok:
        raise ConfigError(f"project crs {spec!r} must be a projected CRS in metres")
    return crs


def make_transformer(
    src: str,
    dst: str,
    *,
    allow_network: bool = False,
    allow_ballpark: bool = False,
):
    """Return the most accurate available PROJ transformer.

    NaN rule: not applicable. ``allow_network`` is left as a process-wide PROJ
    setting. A missing datum grid is ``ConfigError`` unless ``allow_ballpark``.
    """
    pyproj = _pyproj("reproject coordinates")
    try:
        src_crs = pyproj.CRS.from_user_input(src)
        dst_crs = pyproj.CRS.from_user_input(dst)
    except Exception as exc:
        raise ConfigError(f"invalid crs {src!r} or {dst!r}: {exc}") from exc
    pyproj.network.set_network_enabled(allow_network)
    # PROJ warns on its own when the best grid is missing. The error below is
    # the contract; do not also surface that warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        group = pyproj.transformer.TransformerGroup(src_crs, dst_crs, always_xy=True)
    if not group.transformers:
        raise ConfigError(f"no transformation from {src} to {dst}")
    # PROJ lists the most accurate available transform first. Without the
    # datum grid, best_available is false and the fallback is ballpark.
    if not group.best_available and not allow_ballpark:
        missing: list[str] = []
        seen: set[str] = set()
        for operation in group.unavailable_operations:
            for grid in getattr(operation, "grids", ()) or ():
                if grid.available:
                    continue
                short = grid.short_name
                if not short or short in seen:
                    continue
                seen.add(short)
                missing.append(short)
        grids = ", ".join(missing) if missing else "(none listed)"
        raise ConfigError(
            f"no accurate transformation from {src} to {dst}; missing grid files: {grids}. "
            "set edges.allow_network = true, or install the grids with projsync, "
            "or set edges.allow_ballpark = true to accept a less accurate transformation"
        )
    return group.transformers[0]


def reproject_xy(
    x: np.ndarray,
    y: np.ndarray,
    src: str,
    dst: str,
    *,
    allow_network: bool = False,
    allow_ballpark: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject coordinates from ``src`` to ``dst``.

    NaN rule: non-finite inputs become NaN. A finite input that transforms to
    a non-finite value is ``FormatError``. Returns new float64 arrays.
    """
    try:
        x_arr = np.array(x, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError("x and y must be float64 arrays of equal shape") from exc
    if x_arr.shape != y_arr.shape:
        raise ConfigError(
            f"x and y must be float64 arrays of equal shape, got {x_arr.shape} and {y_arr.shape}"
        )
    out_x = np.full(x_arr.shape, np.nan, dtype=np.float64)
    out_y = np.full(y_arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    transformer = make_transformer(
        src,
        dst,
        allow_network=allow_network,
        allow_ballpark=allow_ballpark,
    )
    if not np.any(finite):
        return out_x, out_y
    transformed_x, transformed_y = transformer.transform(x_arr[finite], y_arr[finite])
    tx = np.asarray(transformed_x, dtype=np.float64)
    ty = np.asarray(transformed_y, dtype=np.float64)
    bad = ~np.isfinite(tx) | ~np.isfinite(ty)
    if np.any(bad):
        bad_index = int(np.flatnonzero(bad)[0])
        src_x = x_arr[finite].reshape(-1)
        src_y = y_arr[finite].reshape(-1)
        raise FormatError(
            f"{int(np.count_nonzero(bad))} finite coordinates reprojected to non-finite values; "
            f"first input pair ({float(src_x[bad_index])}, {float(src_y[bad_index])})"
        )
    out_x[finite] = tx
    out_y[finite] = ty
    return out_x, out_y


def crs_equals(src: str, dst: str, *, source: str) -> bool:
    """True when ``src`` and ``dst`` are the same CRS.

    NaN rule: not applicable.
    """
    pyproj = _pyproj("compare coordinate reference systems")
    try:
        return bool(pyproj.CRS(src).equals(pyproj.CRS(dst)))
    except Exception as exc:
        raise ConfigError(f"{source}: invalid crs: {exc}") from exc


def parse_wkt(wkt: str, *, source: str) -> str:
    """Validate ``wkt`` and return it.

    NaN rule: not applicable.
    """
    pyproj = _pyproj("read a .prj file")
    try:
        pyproj.CRS.from_wkt(wkt)
    except Exception as exc:
        raise ConfigError(f"{source}: invalid .prj: {exc}") from exc
    return wkt
