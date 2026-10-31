# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pytest

from subsurfix import facies


def _data(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = np.arange(n) % 4
    a = np.array([0.0, 6.0, 12.0, 12.0])[c] + rng.normal(size=n)
    b = rng.normal(size=n)
    return np.column_stack([a, b]), (c * 10).astype(np.int32), (np.arange(n) // 50).astype(np.int32)


def test_labelled_separability_recommends_three() -> None:
    x, y, g = _data()
    r = facies.separability(x, ["A", "B"], g, y)
    assert r["mode"] == "labels"
    assert r["recommended_k"] == 3
    assert r["best_logs"] == [0]
    assert [b["k"] for b in facies.best_scores(r)] == [2, 3, 4]


def test_cluster_separability_recommends_three() -> None:
    x, _, g = _data()
    r = facies.separability(x, ["A", "B"], g, k_max=5)
    assert r["mode"] == "clusters"
    assert r["recommended_k"] == 3


def test_plot_separability(tmp_path) -> None:  # noqa: ANN001
    pytest.importorskip("matplotlib")
    x, y, g = _data(1500)
    r = facies.separability(x, ["A", "B"], g, y, min_rows_class=100)
    facies.plot_separability(r, tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 0


def test_bad_classifier_raises() -> None:
    x, y, g = _data(400)
    with pytest.raises(Exception):
        facies.separability(x, ["A", "B"], g, y, classifier="svm")
