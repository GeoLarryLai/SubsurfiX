# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Thread-count and report decimation/streaming options of the flow solvers."""

import numpy as np

from subsurfix import _native as nat

K = 1e-12


def _model(n=40):
    m = nat.FlowModel.from_regular([0, 0, 0], [2, 1, 1], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2))
    wells = [
        {"name": "I", "kind": "injector", "cells": [0], "rate": 1e-5, "radius": 0.1},
        {"name": "P", "kind": "producer", "cells": [n - 1], "bhp": 1e7, "radius": 0.1},
    ]
    return m, wells, n


def test_threads_and_heat_decimation(tmp_path):
    m, wells, n = _model()
    assert nat.flow_threads() >= 1
    mob = np.full(n, 1e3)
    a = m.solve_pressure(mob, wells, n_threads=1)
    b = m.solve_pressure(mob, wells, n_threads=3)
    np.testing.assert_array_equal(a["p"], b["p"])
    times = [1e5 * i for i in range(1, 6)]
    full = m.run_heat(mob, wells, [20.0, 0.0], np.full(n, 80.0), times, dt_max=1e5)
    assert "report_index" not in full
    dec = m.run_heat(mob, wells, [20.0, 0.0], np.full(n, 80.0), times, dt_max=1e5, report_every=2)
    assert dec["report_index"] == [0, 2, 4]
    np.testing.assert_array_equal(dec["temperature"], full["temperature"][[0, 2, 4]])
    d = tmp_path / "snap"
    s = m.run_heat(mob, wells, [20.0, 0.0], np.full(n, 80.0), times, dt_max=1e5, snapshot_dir=str(d), n_threads=2)
    assert s["temperature"].size == 0
    for i in range(5):
        np.testing.assert_array_equal(np.load(d / f"temperature_{i:05d}.npy"), full["temperature"][i])


def test_two_phase_snapshot_dir(tmp_path):
    m, wells, n = _model()
    times = [1e4, 2e4, 3e4]
    full = m.run_two_phase(wells, np.full(n, 0.2), np.full(n, 1e7), times)
    s = m.run_two_phase(wells, np.full(n, 0.2), np.full(n, 1e7), times, report_every=2, snapshot_dir=str(tmp_path))
    assert s["report_index"] == [0, 2]
    for i in (0, 2):
        np.testing.assert_array_equal(np.load(tmp_path / f"sw_{i:05d}.npy"), full["sw"][i])
        np.testing.assert_array_equal(np.load(tmp_path / f"p_{i:05d}.npy"), full["p"][i])
    np.testing.assert_array_equal(s["bhp"], full["bhp"])
