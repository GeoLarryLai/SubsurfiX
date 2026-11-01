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

from subsurfix import diagnostics, errors
from subsurfix.errors import SubsurfixError
from subsurfix.plugins import clear_models, register_model, registered_models
from subsurfix.products import (
    read_geometry,
    read_layer_time,
    read_logs,
    read_product,
    read_subhorizons,
    read_suspects,
    read_tops,
    read_wells,
)
from subsurfix.project import Project
from subsurfix.starter import init_project, template_toml, write_demo_project


def prepare_edges(project_toml):
    """Write the edge tables (wells, headers, tops, surveys, selection); see ``subsurfix.edges.prepare_edges``."""
    from subsurfix.edges import prepare_edges as _prepare

    return _prepare(project_toml)


from subsurfix.cornerpoint import (
    flow_model_from_grdecl,
    read_cornerpoint_grdecl,
    transmissibilities,
)

FixedPressure = _native.FixedPressure
CornerPointGrid = _native.CornerPointGrid
VeGrid = _native.VeGrid
run_co2_ve = _native.run_co2_ve
FlowModel = _native.FlowModel
__version__: str = _native.version()


def enable_logging(level: str = "info") -> bool:
    """Print engine log events at ``level`` (``error``, ``warn``, ``info``, ``debug``, ``trace``) to stderr.

    Returns ``True`` if installed now, ``False`` if logging was already set up.
    Raises ``InvalidInputError`` for another level. Never called on import.
    """
    return _native.init_logging(level)


__all__ = [
    "CornerPointGrid",
    "FixedPressure",
    "FlowModel",
    "Project",
    "SubsurfixError",
    "VeGrid",
    "__version__",
    "_native",
    "clear_models",
    "diagnostics",
    "enable_logging",
    "errors",
    "flow_model_from_grdecl",
    "init_project",
    "prepare_edges",
    "read_cornerpoint_grdecl",
    "read_geometry",
    "read_layer_time",
    "read_logs",
    "read_product",
    "read_subhorizons",
    "read_suspects",
    "read_tops",
    "read_wells",
    "register_model",
    "registered_models",
    "run_co2_ve",
    "template_toml",
    "transmissibilities",
    "write_demo_project",
]
