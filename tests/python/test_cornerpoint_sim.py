# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Corner-point grid, universal kriging, SGS, SIS and ml_propagation bindings."""
import numpy as np
import pytest

from subsurfix import _native as nat


def test_cornerpoint_roundtrip_and_upscale():
    xs, ys = np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0])
    surf = np.array([[100, 105, 110] * 2, [104, 107, 110] * 2, [120.0] * 6], dtype=float)
    g = nat.CornerPointGrid.from_surfaces(xs, ys, surf)
    assert g.dims == (2, 1, 2)
    assert g.volumes()[0] == pytest.approx(300.0)
    assert list(g.locate(np.array([[5.0, 5.0, 103.0], [5.0, 5.0, 999.0]]))) == [0, -1]
    text = g.to_grdecl({"PORO": [0.1, 0.2, 0.3, 0.4]})
    h, props = nat.CornerPointGrid.from_grdecl(text)
    assert np.allclose(h.zcorn, g.zcorn) and list(props["PORO"]) == [0.1, 0.2, 0.3, 0.4]
    cell, val, n = g.upscale_continuous(np.array([[5.0, 5.0, 103.0], [5.0, 5.0, 105.0]]), np.array([0.1, 0.3]))
    assert list(cell) == [0] and val[0] == pytest.approx(0.2)
    r = nat.CornerPointGrid.from_regular((0, 0, 0), (1, 1, 1), (2, 2, 2))
    assert r.centers().shape == (8, 3) and np.allclose(r.volumes(), 1.0)


def test_universal_kriging_plane_and_trend():
    p = np.array([[x, y, 0.0] for x in range(0, 600, 100) for y in range(0, 600, 100)], dtype=float)
    v = 1.0 + 0.01 * p[:, 0] + 0.02 * p[:, 1]
    t = np.array([[250.0, 330.0, 0.0]])
    est, _ = nat.universal_kriging(p, v, t, "gaussian", 0.0, 1.0, 300.0, "linear")
    assert est[0] == pytest.approx(1.0 + 2.5 + 6.6, abs=1e-6)
    assert nat.trend_surface(p, v, t)[0] == pytest.approx(10.1)


def test_sgs_sis_and_ml_propagation():
    rng = np.random.default_rng(0)
    p = np.column_stack([rng.uniform(0, 500, 200), rng.uniform(0, 500, 200), np.zeros(200)])
    v = 0.1 + 0.05 * np.sin(p[:, 0] / 80.0)
    t = np.column_stack([rng.uniform(0, 500, 100), rng.uniform(0, 500, 100), np.zeros(100)])
    ns = nat.normal_score(v)
    assert abs(ns.mean()) < 0.05
    r = nat.sgs(p, v, t, "spherical", 0.0, 1.0, 150.0, n_realizations=4, seed=1)
    assert r.shape == (4, 100) and r.min() >= v.min() - 1e-12 and r.max() <= v.max() + 1e-12
    codes = (p[:, 0] > 250).astype(np.int32)
    f = nat.sis(p, codes, t, "spherical", 0.0, 0.25, 150.0, n_realizations=3)
    far = np.abs(t[:, 0] - 250) > 60
    assert ((f[:, far] == (t[far, 0] > 250)).mean()) > 0.9
    x = p[:, :2]
    for model in ("lightgbm", "qrf"):
        q = nat.ml_propagation(x, v, t[:, :2], model=model, n_trees=50)
        assert q.shape == (100, 3) and (q[:, 0] <= q[:, 2]).all()
