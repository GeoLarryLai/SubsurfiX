# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Plug-in fill models.

Any Python object with scikit-learn style ``fit(X, y, sample_weight=None)``
and ``predict(X)`` can replace the built-in gradient-boosted model for one log.

Registration

- One process-wide registry maps a log to a model object.
  ``register_model(log, model)`` takes the log name in any case (unknown name:
  ``NotFoundError``), requires callable ``fit`` and ``predict`` (else
  ``InvalidInputError``), and tries ``copy.deepcopy(model)`` once (failure:
  ``InvalidInputError`` with the Python message). The original object is
  stored; registering the same log again replaces the entry.
- ``clear_models()`` empties the registry; ``clear_models("GR")`` removes one
  entry (unknown name: ``NotFoundError``; no entry: no-op).
- ``registered_models()`` returns ``{log: descriptor}`` in suite order, with
  descriptor ``"py:" + type(model).__module__ + "." + type(model).__qualname__``.
- A run takes one snapshot of the registry at its start; changes during a run
  do not affect it. Each fill run fits its own ``copy.deepcopy`` of the
  registered object; the registered object itself is never fitted.

Model contract

- ``X`` is a C-contiguous ``float32`` array of shape ``(n_rows, n_cols)``; NaN
  marks a missing feature and is passed through (a model that rejects NaN must
  handle it, for example with an imputer step). ``y`` and ``sample_weight``
  are 1-D ``float32``.
- If the object has a callable ``set_feature_names``, it is called with the
  column names (``list[str]``) before every ``fit`` and ``predict``.
- ``predict`` must return ``n_rows`` finite values, shape ``(n_rows,)`` or
  ``(n_rows, 1)``; they are cast to ``float32``. Wrong shape: ``ShapeError``;
  NaN or inf: ``ModelError``.
- An exception raised by the model surfaces as ``ModelError`` carrying the
  Python message and traceback.

Determinism, speed, saving

- Output is deterministic only if the model is: seed it (``random_state=...``)
  and avoid global random state. Bit-identical reruns are not promised for
  plug-ins.
- A registered object's state cannot be hashed, so while any plug-in is
  registered the pipeline treats ``fill`` and every later stage as stale on
  every run, until ``clear_models()``.
- Plug-in models are never written to disk; refresh uses the in-memory models
  of the same run.
- Slower than the built-in model: one copy of ``X`` per call, calls serialized
  by the GIL, plus the model's own time.
"""

from __future__ import annotations

from subsurfix import _native


def register_model(log: str, model: object) -> None:
    """Use ``model`` as the fill model of ``log`` (for example ``"GR"``)."""
    _native.register_model(log, model)


def clear_models(log: str | None = None) -> None:
    """Remove every plug-in, or only the one of ``log``."""
    _native.clear_models(log)


def registered_models() -> dict[str, str]:
    """``{log: descriptor}`` of every registered plug-in, e.g. ``{"GR": "py:mypkg.MeanModel"}``."""
    return _native.registered_models()


__all__ = ["register_model", "clear_models", "registered_models"]
