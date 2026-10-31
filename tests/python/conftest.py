# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import os
import pathlib

import pytest

import subsurfix
from subsurfix import _native


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory) -> subsurfix.Project:
    if not hasattr(_native, "_make_synthetic_project"):
        pytest.fail("rebuild the extension: maturin develop -r --features testing")
    root = tmp_path_factory.mktemp("synthetic_project")
    config = _native._make_synthetic_project(root, n_wells=6, seed=0)
    return subsurfix.Project(config)


@pytest.fixture(scope="session")
def data_fixtures_dir() -> pathlib.Path:
    repo = pathlib.Path(__file__).resolve().parents[2]
    default = repo / ".." / "data" / "fixtures"
    path = pathlib.Path(os.environ.get("SX_DATA_FIXTURES") or default)
    if not path.is_dir():
        pytest.fail(
            f"data fixture {path} not found. Set SX_DATA_FIXTURES or use the default {default}. "
            "Tests that need data fixtures are #[ignore]; run them with: "
            "cargo test --workspace -- --ignored data_"
        )
    return path
