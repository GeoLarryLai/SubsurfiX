# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""SubsurfiX Python face."""

try:
    from subsurfix import _native
except ImportError as e:
    raise ImportError(
        "subsurfix: the native extension subsurfix._native is missing or broken; "
        "build it with `maturin develop -r`. There is no pure-Python fallback."
    ) from e

from subsurfix import errors
from subsurfix.errors import SubsurfixError
from subsurfix.plugins import register_model, clear_models, registered_models
from subsurfix.products import (
    read_product,
    read_wells,
    read_logs,
    read_geometry,
    read_tops,
    read_suspects,
    read_layer_time,
    read_subhorizons,
)
from subsurfix.project import Project
from subsurfix.starter import template_toml, init_project, write_demo_project


def prepare_edges(project_toml):  # noqa: ANN001, ANN201
    """Write the edge tables (wells, headers, tops, surveys, selection); see ``subsurfix.edges.prepare_edges``."""
    from subsurfix.edges import prepare_edges as _prepare

    return _prepare(project_toml)


FixedPressure = _native.FixedPressure
FlowModel = _native.FlowModel
__version__: str = _native.version()


def enable_logging(level: str = "info") -> bool:
    """Print engine log events at ``level`` (``error``, ``warn``, ``info``, ``debug``, ``trace``) to stderr.

    Returns ``True`` if installed now, ``False`` if logging was already set up.
    Raises ``InvalidInputError`` for another level. Never called on import.
    """
    return _native.init_logging(level)


__all__ = [
    "_native",
    "errors",
    "SubsurfixError",
    "register_model",
    "clear_models",
    "registered_models",
    "Project",
    "read_product",
    "read_wells",
    "read_logs",
    "read_geometry",
    "read_tops",
    "read_suspects",
    "read_layer_time",
    "read_subhorizons",
    "enable_logging",
    "template_toml",
    "init_project",
    "write_demo_project",
    "prepare_edges",
    "FixedPressure",
    "FlowModel",
    "__version__",
]
