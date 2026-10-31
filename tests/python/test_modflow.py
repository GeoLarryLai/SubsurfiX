# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""MODFLOW 6 bridge: SubsurfiX TPFA pressure vs MODFLOW 6 on the same grid and wells."""
import numpy as np
import pytest

import flopy  # noqa: F401  (a regular dependency)
from subsurfix import _native as nat
from subsurfix import modflow

pytestmark = pytest.mark.skipif(not modflow.mf6_available(), reason="mf6 binary not found (get-modflow :flopy)")
MU = 1e-3


def _case(tmp_path, k):
    nx, ny, nz = 6, 5, 3
    n = nx * ny * nz
    perm = np.column_stack([k, k, 0.1 * k])
    m = nat.FlowModel.from_regular((0, 0, 0), (20.0, 20.0, 5.0), (nx, ny, nz), perm, np.full(n, 0.2))
    cid = lambda i, j, kk: i + nx * (j + ny * kk)
    wells = [
        {"name": "INJ", "kind": "injector", "rate": 1e-3, "cells": [cid(0, 0, kk) for kk in range(nz)]},
        {"name": "PROD", "kind": "producer", "bhp": 1.5e7, "cells": [cid(5, 4, kk) for kk in range(nz)]},
    ]
    sol = m.solve_pressure(np.full(n, 1 / MU), wells)
    sim = modflow.to_modflow6(tmp_path, (0, 0, 0), (20.0, 20.0, 5.0), (nx, ny, nz), perm, np.full(n, 0.2), wells, mu=MU)
    modflow.run_modflow6(sim)
    return sol, modflow.read_modflow6(sim), nx * ny * nz


def test_homogeneous_matches_subsurfix(tmp_path):
    sol, res, n = _case(tmp_path, np.full(90, 1e-13))
    assert res["p"].shape == (1, n)
    dp_ssx = np.asarray(sol["p"]) - 1.5e7
    dp_mf = res["p"][0] - 1.5e7
    assert np.max(np.abs(dp_mf - dp_ssx)) / np.max(np.abs(dp_ssx)) < 1e-3
    q = {w["name"]: w["rate"] for w in res["wells"]}
    assert q["inj"] == pytest.approx(1e-3, rel=1e-6)
    assert q["prod"] == pytest.approx(-1e-3, rel=1e-4)


def test_heterogeneous_matches_subsurfix(tmp_path):
    rng = np.random.default_rng(3)
    sol, res, _ = _case(tmp_path, 10 ** rng.uniform(-14, -12, 90))
    dp_ssx = np.asarray(sol["p"]) - 1.5e7
    assert np.max(np.abs(res["p"][0] - 1.5e7 - dp_ssx)) / np.max(np.abs(dp_ssx)) < 1e-3


def test_transient_runs(tmp_path):
    n = 4 * 4 * 2
    perm = np.full((n, 3), 1e-13)
    w = [{"name": "I", "kind": "injector", "rate": 1e-4, "cells": [0, 16]},
         {"name": "P", "kind": "producer", "bhp": 1e7, "cells": [15, 31]}]
    sim = modflow.to_modflow6(tmp_path, (0, 0, 0), (10, 10, 5), (4, 4, 2), perm, np.full(n, 0.2), w,
                              c_rock=5e-10, c_w=4e-10, p0=1e7, times=[1e5, 1e6], nstp=4)
    modflow.run_modflow6(sim)
    res = modflow.read_modflow6(sim)
    assert len(res["times"]) == 8 and res["p"][-1].max() > res["p"][0].max() - 1.0
