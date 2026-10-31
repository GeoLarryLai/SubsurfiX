# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations


class SubsurfixError(Exception):
    """Every fallible engine function returns this error."""


class InvalidInputError(SubsurfixError, ValueError):
    """A value is outside its allowed range or violates a stated rule."""


class EmptyInputError(SubsurfixError, ValueError):
    """A required input has no elements."""


class ShapeError(SubsurfixError, ValueError):
    """Array lengths or matrix shapes do not agree."""


class NotFoundError(SubsurfixError, LookupError):
    """A well, log, top, or file that was asked for does not exist."""


class ConfigError(SubsurfixError):
    """The project configuration is invalid."""


class IoError(SubsurfixError, OSError):
    """A file could not be read or written."""


class FormatError(SubsurfixError, ValueError):
    """A file was read but its content does not follow the expected format."""


class NumericalError(SubsurfixError, ArithmeticError):
    """A numerical routine failed (singular system, no convergence, overflow)."""


class ModelError(SubsurfixError):
    """The gradient-boosted tree library or a plug-in model failed."""


class InfeasibleError(SubsurfixError):
    """The problem has no solution under the given constraints (for example a DTW band that admits no path)."""


class MissingDependencyError(SubsurfixError, ImportError):
    """Raised when a required dependency (for example pyproj or shapely) cannot be imported; reinstall subsurfix."""


VARIANT_TO_CLASS: dict[str, type[SubsurfixError]] = {
    "InvalidInput": InvalidInputError,
    "EmptyInput": EmptyInputError,
    "Shape": ShapeError,
    "NotFound": NotFoundError,
    "Config": ConfigError,
    "Io": IoError,
    "Format": FormatError,
    "Numerical": NumericalError,
    "Model": ModelError,
    "Infeasible": InfeasibleError,
}

__all__ = [
    "SubsurfixError",
    "InvalidInputError",
    "EmptyInputError",
    "ShapeError",
    "NotFoundError",
    "ConfigError",
    "IoError",
    "FormatError",
    "NumericalError",
    "ModelError",
    "InfeasibleError",
    "MissingDependencyError",
    "VARIANT_TO_CLASS",
]
