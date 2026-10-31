# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pytest

from subsurfix import _native
from subsurfix.errors import ModelError


class MeanModel:
    seen_means: list = []  # class attribute: shared by deep copies

    def fit(self, X, y, sample_weight=None):
        self.mean_ = float(np.average(y, weights=sample_weight))
        MeanModel.seen_means.append(self.mean_)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_, dtype=np.float32)


class LinearModel:
    def fit(self, X, y, sample_weight=None):
        A = np.c_[np.nan_to_num(X.astype(np.float64)), np.ones(len(X))]
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, np.float64)
        s = np.sqrt(w)
        self.coef_ = np.linalg.lstsq(A * s[:, None], y * s, rcond=None)[0]
        return self

    def predict(self, X):
        A = np.c_[np.nan_to_num(X.astype(np.float64)), np.ones(len(X))]
        return A @ self.coef_  # float64; the engine casts to float32


class BoomModel:
    def fit(self, X, y, sample_weight=None):
        raise ValueError("boom")

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=np.float32)


def _f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


X_FIT = _f32([[0.0], [1.0], [2.0], [3.0]])


def test_mean_model() -> None:
    model = MeanModel()
    y = _f32([1, 2, 3, 6])
    x_pred = _f32([[0.0], [5.0], [9.0]])
    got = _native._plugin_roundtrip("GR", model, X_FIT, ["a"], y, x_pred)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, [3.0, 3.0, 3.0], rtol=0, atol=1e-5)
    got = _native._plugin_roundtrip("GR", model, X_FIT, ["a"], y, x_pred, weight=_f32([1, 1, 1, 3]))
    np.testing.assert_allclose(got, [4.0, 4.0, 4.0], rtol=0, atol=1e-5)
    assert not hasattr(model, "mean_")


def test_linear_model() -> None:
    y = _f32([1, 3, 5, 7])
    x_pred = _f32([[4.0], [10.0]])
    got = _native._plugin_roundtrip("GR", LinearModel(), X_FIT, ["a"], y, x_pred)
    np.testing.assert_allclose(got, [9.0, 21.0], rtol=0, atol=1e-5)


def test_fit_exception_is_model_error() -> None:
    with pytest.raises(ModelError) as caught:
        _native._plugin_roundtrip("GR", BoomModel(), X_FIT, ["a"], _f32([1, 2, 3, 4]), X_FIT)
    assert type(caught.value) is ModelError
    assert "ValueError: boom" in str(caught.value)
