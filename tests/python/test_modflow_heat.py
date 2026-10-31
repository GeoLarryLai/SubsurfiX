# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pytest

import flopy
from subsurfix import _native as nat
from subsurfix.modflow_heat import (
    HeatDoublet,
    find_mf6,
    from_layered,
    regular_grid_to_dis,
    run_gwf_gwe,
    to_layered,
)


def test_grid_conversion_roundtrip():
    dims = (4, 3, 2)
    a = np.arange(24.0)
    assert np.array_equal(from_layered(to_layered(a, dims), dims), a)
    d = regular_grid_to_dis((0, 0, 1000), (10, 20, 5), dims)
    assert (d["nlay"], d["nrow"], d["ncol"]) == (2, 3, 4)
    assert d["top"] == -1000 and d["botm"][1, 0, 0] == -1010
    assert d["index"](1 + 4 * 0 + 12) == (1, 2, 1)
    assert to_layered(a, dims)[d["index"](17)] == 17


@pytest.mark.skipif(find_mf6() is None, reason="mf6 not on PATH")
def test_doublet_matches_run_heat(tmp_path):
    nx, ny = 12, 8
    n = nx * ny
    perm = np.full(n, 300 * 9.869e-16)
    wells = [{"name": "I", "kind": "injector", "cells": [1 + nx * 4], "rate": 0.01, "radius": 0.1},
             {"name": "P", "kind": "producer", "cells": [10 + nx * 4], "bhp": 2e7, "radius": 0.1}]
    yr = 3.156e7
    times = [2 * yr, 5 * yr, 10 * yr]
    case = HeatDoublet((0, 0, 2000), (25, 25, 20), (nx, ny, 1), perm, np.full(n, 0.2), np.full(n, 80.0),
                       wells, [30.0, 0.0], times, dt_max=0.1 * yr)
    mf = run_gwf_gwe(case, tmp_path)
    m = nat.FlowModel.from_regular([0, 0, 2000], [25, 25, 20], [nx, ny, 1], np.column_stack([perm] * 3), np.full(n, 0.2))
    sx = np.asarray(m.run_heat(np.full(n, 2e3), wells, [30.0, 0.0], np.full(n, 80.0), times,
                               props=case.props, dt_max=0.1 * yr)["temperature"])
    assert np.allclose(mf["times"], times)
    assert mf["temperature"][-1].min() < 60  # the cold front has moved
    assert np.abs(mf["temperature"] - sx).max() < 0.05


@pytest.mark.skipif(find_mf6() is None, reason="mf6 not on PATH")
def test_multilayer_wells_match_run_heat(tmp_path):
    """Wells open in three layers of different permeability: MODFLOW 6 MAW flow split, then GWE with WEL/CHD."""
    nx, ny, nz = 10, 6, 3
    n = nx * ny * nz
    md = 9.869e-16
    k_layer = np.repeat([100.0, 400.0, 200.0], nx * ny) * md
    perm = np.column_stack([k_layer, k_layer, 0.1 * k_layer])  # anisotropic, kz = kx / 10
    col_i, col_p = 1 + nx * 3, 8 + nx * 3
    inj = [col_i + nx * ny * k for k in range(nz)]
    prd = [col_p + nx * ny * k for k in range(nz)]
    rate = 0.002
    wells = [{"name": "I", "kind": "injector", "cells": inj, "rate": rate, "radius": 0.1},
             {"name": "P", "kind": "producer", "cells": prd, "bhp": 2e7, "radius": 0.1}]
    yr = 3.156e7
    times = [1 * yr, 2 * yr, 4 * yr]
    case = HeatDoublet((0, 0, 2000), (25, 25, 10), (nx, ny, nz), perm, np.full(n, 0.2), np.full(n, 80.0),
                       wells, [30.0, 0.0], times, dt_max=0.1 * yr)
    mf = run_gwf_gwe(case, tmp_path)

    # step 1 (MAW flow split): per-layer injector rates add up to the well rate and follow layer permeability
    rec = flopy.utils.CellBudgetFile(tmp_path / "maw_split" / "maw.cbc").get_data(text="GWF")[-1]
    q = {int(nd) - 1: -float(v) for nd, v in zip(rec["node2"], rec["q"])}
    idx = regular_grid_to_dis(case.origin, case.cell, case.dims)["index"]

    def node(c):  # MODFLOW 0-based node of SubsurfiX cell c
        k, r, i = idx(c)
        return (k * ny + r) * nx + i

    q_inj = np.array([q[node(c)] for c in inj])
    assert np.isclose(q_inj.sum(), rate, rtol=1e-6)
    assert q_inj[1] > q_inj[2] > q_inj[0] > 0
    q_prd = np.array([q[node(c)] for c in prd])
    assert np.isclose(-q_prd.sum(), rate, rtol=1e-4)  # steady state: what goes in comes out

    # step 2 (GWE) against SubsurfiX's own heat solver with the same multi-layer wells
    m = nat.FlowModel.from_regular([0, 0, 2000], [25, 25, 10], [nx, ny, nz], perm, np.full(n, 0.2))
    sx = np.asarray(m.run_heat(np.full(n, 2e3), wells, [30.0, 0.0], np.full(n, 80.0), times,
                               props=case.props, dt_max=0.1 * yr)["temperature"])
    assert np.allclose(mf["times"], times)
    assert mf["temperature"][-1].min() < 60  # the cold front has moved
    # the high-permeability middle layer cools furthest from the injector
    t_end = mf["temperature"][-1].reshape(nz, ny, nx)
    assert t_end[1].mean() < t_end[2].mean() < t_end[0].mean()
    assert np.abs(mf["temperature"] - sx).max() < 0.05
