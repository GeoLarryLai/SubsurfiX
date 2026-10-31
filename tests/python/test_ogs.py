# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""OpenGeoSys (OGS) mesh export through the Python API."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from subsurfix import _native as nat


def _arrays(path):
    root = ET.parse(path).getroot()
    piece = root.find("UnstructuredGrid/Piece")
    out = {a.get("Name"): np.array(a.text.split(), dtype=float) for a in piece.find("CellData")}
    return piece, out


def test_export_ogs_regular(tmp_path):
    n = 6
    m = nat.FlowModel.from_regular([0, 0, 500], [10, 10, 5], [3, 2, 1], np.full((n, 3), 1e-13), np.full(n, 0.2))
    temp = np.linspace(20, 70, n)
    path = tmp_path / "mesh.vtu"
    info = m.export_ogs(str(path), cell_data={"temperature": temp}, material_ids=[0, 0, 1, 1, 2, 2])
    assert info == {"n_nodes": 4 * 3 * 2, "n_cells": n}
    piece, a = _arrays(path)
    assert piece.get("NumberOfCells") == "6"
    np.testing.assert_array_equal(a["MaterialIDs"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(a["temperature"], temp)
    np.testing.assert_allclose(a["porosity"], 0.2)
    np.testing.assert_allclose(a["permeability"], 1e-13)
    z = np.array(piece.find("Points/DataArray").text.split(), dtype=float)[2::3]
    assert z.min() == pytest.approx(-505.0) and z.max() == pytest.approx(-500.0)
    with pytest.raises(ValueError, match="cell_data"):
        m.export_ogs(str(path), cell_data={"t": np.zeros(n + 1)})
    with pytest.raises(ValueError, match="twice"):
        m.export_ogs(str(path), cell_data={"porosity": np.zeros(n)})
