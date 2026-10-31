# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Grid bindings: upscaling, variogram fitting, IDW and ordinary kriging."""
import math

import numpy as np
import pytest

from subsurfix import _native as nat
from subsurfix import errors

O, C, D = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 1, 1)


def test_upscale_means_and_facies():
    p = np.array([[0.2, 0.5, 0.5], [0.7, 0.5, 0.5], [1.5, 0.5, 0.5]])
    cell, val, n = nat.upscale_continuous(O, C, D, p, np.array([1.0, 4.0, 9.0]), "harmonic")
    assert list(cell) == [0, 1] and list(n) == [2, 1]
    assert val[0] == pytest.approx(2 / (1 + 0.25))
    _, g, _ = nat.upscale_continuous(O, C, D, p, np.array([1.0, 4.0, 9.0]), "geometric")
    assert g[0] == pytest.approx(2.0)
    cell, code, _ = nat.upscale_facies(O, C, D, p, np.array([3, 3, 1], dtype=np.int32))
    assert list(code) == [3, 1]
    assert nat.grid_centers(O, C, D).tolist() == [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]


def test_bad_inputs_raise():
    p = np.zeros((1, 3))
    with pytest.raises(errors.SubsurfixError):
        nat.upscale_continuous(O, C, D, p, np.array([-1.0]), "harmonic")
    with pytest.raises(ValueError):
        nat.idw(np.zeros((2, 2)), np.ones(2), np.zeros((1, 3)))


def test_variogram_idw_kriging_roundtrip():
    rng = np.random.default_rng(7)
    p = np.column_stack([rng.uniform(0, 1000, 120), rng.uniform(0, 1000, 120), np.zeros(120)])
    v = 0.01 * p[:, 0] + np.sin(p[:, 1] / 150.0)
    lag, gam, pairs = nat.experimental_variogram(p, v, 50.0, 12)
    m = nat.fit_variogram(lag, gam, pairs, "gaussian")
    assert m["range"] > 0 and m["psill"] > 0
    t = np.array([[500.0, 500.0, 0.0], *p[:1]])
    est, var = nat.ordinary_kriging(p, v, t, m["kind"], m["nugget"], m["psill"], m["range"])
    assert est[1] == v[0] and var[1] == 0.0
    truth = 5.0 + math.sin(500 / 150.0)
    assert abs(est[0] - truth) < 0.3
    assert abs(nat.idw(p, v, t)[0] - truth) < 0.6
