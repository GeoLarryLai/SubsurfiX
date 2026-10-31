# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pytest

from subsurfix import _native


def _wells(n_wells=30, seed=2):
    rng = np.random.default_rng(seed)
    xyz, y, g = [], [], []
    for w in range(n_wells):
        x0, y0 = rng.uniform(0, 1000, 2)
        for k in range(15):
            xyz.append((x0, y0, float(k)))
            y.append(np.sin(x0 / 200) + np.cos(y0 / 300) + 0.05 * k + 0.2 * rng.normal())
            g.append(w)
    return np.array(xyz), np.array(y), np.array(g, dtype=np.int32)


def test_ember_returns_envelope_and_realizations() -> None:
    xyz, y, g = _wells()
    t = xyz[::5].copy()
    r = _native.ember(xyz, y, t, g, n_trees=50, anisotropy=(1, 1, 0.1), n_realizations=3)
    assert r["quantiles"].shape == (len(t), 3)
    assert r["realizations"].shape == (3, len(t))
    assert r["embedded"].shape == (len(t), 2)
    assert np.all(r["std"] >= 0)
    # Targets on data points reproduce the data.
    np.testing.assert_array_equal(r["realizations"][0], y[::5])
    assert abs(r["sampling_variogram"]["nugget"] + r["sampling_variogram"]["psill"] - 1) < 1e-9


def test_ml_propagation_ember_method_keeps_shape() -> None:
    xyz, y, g = _wells()
    x = xyz[:, 2:3].copy()
    q_ml = _native.ml_propagation(x, y, x[:20], n_trees=30)
    q_em = _native.ml_propagation(
        x, y, x[:20], n_trees=30, method="ember", train_xyz=xyz, pred_xyz=xyz[:20], train_groups=g
    )
    assert q_ml.shape == q_em.shape == (20, 3)
    with pytest.raises(ValueError):
        _native.ml_propagation(x, y, x[:20], method="ember")


def test_ember_cv_beats_plain_forest_on_smooth_field() -> None:
    xyz, y, g = _wells(40)
    cv = {r["method"]: r for r in _native.ember_cv(xyz, y, g, n_folds=4, n_trees=50, anisotropy=(1, 1, 0.1), n_realizations=5, embed_holdout="group", random_split=False)}
    assert set(cv) == {"ember", "ember_realizations", "ml_propagation"}
    assert cv["ember"]["mae"] < cv["ml_propagation"]["mae"]


def test_ember_options() -> None:
    xyz, y, g = _wells()
    t = xyz[::7].copy()
    for kw in [dict(embed_holdout="sample"), dict(embed=False), dict(spread="auto"), dict(spread=None), dict(spread=1.5), dict(random_split=False, calibrate=True, gibbs_sweeps=10)]:
        r = _native.ember(xyz, y, t, g, n_trees=30, anisotropy=(1, 1, 0.1), n_realizations=2, **kw)
        assert np.all(np.isfinite(r["mean"]))
    with pytest.raises(ValueError):
        _native.ember(xyz, y, t, g, n_trees=10, embed_holdout="bad")
    for bad in ["wide", -1.0]:
        with pytest.raises(ValueError):
            _native.ember(xyz, y, t, g, n_trees=10, spread=bad)
