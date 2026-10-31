# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""OMF (Leapfrog) export/import round trips. Experimental module."""

import numpy as np
import pytest

import omf  # noqa: F401  (a regular dependency)
from subsurfix import omf_io

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _case():
    nx, ny, nz = 4, 3, 2
    n = nx * ny * nz
    rng = np.random.default_rng(3)
    grid = {
        "origin": (500.0, 1000.0, -2000.0),
        "cell_size": (50.0, 50.0, 10.0),
        "dims": (nx, ny, nz),
        "properties": {
            "PORO": rng.uniform(0.05, 0.3, n),
            "PERM": rng.lognormal(3, 1, n),
            "PORO_P10": np.full(n, 0.1),
            "PORO_P90": np.full(n, 0.25),
        },
    }
    md = np.linspace(0, 100, 11)
    wells = [
        {
            "name": "W-1",
            "xyz": np.column_stack(
                [np.full(11, 600.0), np.full(11, 1050.0), -1990.0 - md]
            ),
            "logs": {"GR": np.linspace(30, 120, 11)},
        }
    ]
    x, y = np.linspace(500, 700, 5), np.linspace(1000, 1150, 4)
    surfaces = {
        "TopA": {"x": x, "y": y, "z": -2000.0 + 0.01 * x[None, :] + 0 * y[:, None]},
        "Fault": {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, -1.0]]),
            "triangles": np.array([[0, 1, 2]]),
        },
    }
    return grid, wells, surfaces


def test_roundtrip(tmp_path):
    grid, wells, surfaces = _case()
    p = omf_io.export_omf(tmp_path / "m.omf", grid=grid, wells=wells, surfaces=surfaces)
    out = omf_io.import_omf(p)
    bm = out["block_models"][0]
    assert bm["dims"] == (4, 3, 2)
    np.testing.assert_allclose(bm["origin"], grid["origin"])
    np.testing.assert_allclose(bm["tensors"][2], [10.0, 10.0])
    for k, v in grid["properties"].items():
        np.testing.assert_allclose(bm["properties"][k], v)
    w = out["wells"][0]
    assert w["name"] == "W-1" and w["segments"].shape == (10, 2)
    np.testing.assert_allclose(w["xyz"], wells[0]["xyz"])
    np.testing.assert_allclose(w["logs"]["GR"], wells[0]["logs"]["GR"])
    s = {d["name"]: d for d in out["surfaces"]}
    np.testing.assert_allclose(s["TopA"]["x"], surfaces["TopA"]["x"])
    np.testing.assert_allclose(s["TopA"]["y"], surfaces["TopA"]["y"])
    np.testing.assert_allclose(s["TopA"]["z"], surfaces["TopA"]["z"])
    np.testing.assert_array_equal(s["Fault"]["triangles"], [[0, 1, 2]])


def test_partial_and_errors(tmp_path):
    grid, wells, _ = _case()
    out = omf_io.import_omf(omf_io.export_omf(tmp_path / "w.omf", wells=wells))
    assert not out["block_models"] and len(out["wells"]) == 1
    with pytest.raises(ValueError):
        omf_io.export_omf(tmp_path / "e.omf")
    bad = dict(grid, properties={"PORO": np.zeros(5)})
    with pytest.raises(ValueError):
        omf_io.export_omf(tmp_path / "b.omf", grid=bad)


def test_experimental_warning(tmp_path):
    _, wells, _ = _case()
    with pytest.warns(FutureWarning, match="experimental"):
        omf_io.export_omf(tmp_path / "x.omf", wells=wells)
