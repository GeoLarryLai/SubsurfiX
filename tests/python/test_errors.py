# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import pytest

from subsurfix import _native
from subsurfix.errors import (
    VARIANT_TO_CLASS,
    InvalidInputError,
    IoError,
    MissingDependencyError,
    NotFoundError,
    SubsurfixError,
)

_DISPLAY = {
    "InvalidInput": "invalid input: boom",
    "EmptyInput": "empty input: boom",
    "Shape": "shape mismatch: boom",
    "NotFound": "not found: boom",
    "Config": "config error: boom",
    "Io": "io error at test-path: boom",
    "Format": "format error: boom",
    "Numerical": "numerical failure: boom",
    "Model": "model error: boom",
    "Infeasible": "infeasible: boom",
}


@pytest.mark.parametrize("variant,cls", list(VARIANT_TO_CLASS.items()))
def test_raise_sx_error_maps_variant(variant: str, cls: type[SubsurfixError]) -> None:
    with pytest.raises(cls) as caught:
        _native._raise_sx_error(variant, "boom")
    exc = caught.value
    assert type(exc) is cls
    assert isinstance(exc, SubsurfixError)
    assert str(exc) == _DISPLAY[variant]


def test_builtin_bases() -> None:
    assert issubclass(InvalidInputError, ValueError)
    assert issubclass(NotFoundError, LookupError)
    assert issubclass(IoError, OSError)
    assert issubclass(MissingDependencyError, ImportError)


def test_unknown_variant_is_plain_value_error() -> None:
    with pytest.raises(ValueError) as caught:
        _native._raise_sx_error("nope", "boom")
    assert not isinstance(caught.value, SubsurfixError)
