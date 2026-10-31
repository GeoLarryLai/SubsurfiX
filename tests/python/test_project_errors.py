# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import pytest

import subsurfix
from subsurfix.errors import IoError


def test_missing_config_is_io_error(tmp_path) -> None:
    with pytest.raises(IoError) as caught:
        subsurfix.Project(tmp_path / "missing.toml")
    assert isinstance(caught.value, subsurfix.SubsurfixError)
    assert "missing.toml" in str(caught.value)
