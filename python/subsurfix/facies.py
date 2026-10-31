# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Lithofacies: log separability, roll-up classes and the five-facies interpretation layer."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from subsurfix import _native

__all__ = [
    "separability",
    "best_scores",
    "plot_separability",
    "interpret",
    "default_cutoffs",
    "rollup_classify",
    "FACIES",
    "ROLLUP",
]

FACIES = ["sandstone", "shale", "limestone", "dolomite", "cherty_carbonate", "carbonate"]
ROLLUP = ["sandstone", "mudstone", "carbonate"]


def _f64(a) -> np.ndarray:  # noqa: ANN001
    return np.ascontiguousarray(a, dtype=np.float64)


def interpret(gr, rhob, nphi, pef=None, method: str = "cutoffs", cutoffs_csv: str | None = None, **kwargs) -> dict[str, Any]:  # noqa: ANN001
    """Five facies and roll-up from GR, RHOB, NPHI (limestone units) and PEF.

    ``method`` is "cutoffs" (rules table, see :func:`default_cutoffs`) or
    "mixing" (non-negative quartz/calcite/dolomite/clay/porosity solve).
    Rows without PEF get a coarser result, flagged in ``coarse``.
    ``cutoffs_csv`` may be CSV text or a path to a CSV file.
    ``nphi_shift`` (keyword, v/v, default 0) is subtracted from NPHI first.
    """
    rhob = _f64(rhob)
    pef = np.full_like(rhob, np.nan) if pef is None else _f64(pef)
    if cutoffs_csv is not None and os.path.isfile(cutoffs_csv):
        with open(cutoffs_csv) as fh:
            cutoffs_csv = fh.read()
    return _native.facies_interpret(_f64(gr), rhob, _f64(nphi), pef, method=method, cutoffs_csv=cutoffs_csv, **kwargs)


def default_cutoffs(path: str | os.PathLike[str] | None = None) -> str:
    """Default rules table as CSV text; also written to ``path`` when given."""
    text = _native.facies_default_cutoffs()
    if path is not None:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def rollup_classify(x_train, y_train, x_pred, shrink: float = 0.1) -> np.ndarray:  # noqa: ANN001
    """Gaussian classifier for the roll-up classes (labels 0..2); -1 where a log is missing."""
    return _native.facies_rollup_classify(_f64(x_train), np.ascontiguousarray(y_train, dtype=np.int32), _f64(x_pred), shrink)


def separability(logs, names, groups, labels=None, **kwargs) -> dict[str, Any]:  # noqa: ANN001
    """Run ``_native.separability`` on array-likes (see its docstring for settings)."""
    x = np.ascontiguousarray(logs, dtype=np.float64)
    g = np.ascontiguousarray(groups, dtype=np.int32)
    lab = None if labels is None else np.ascontiguousarray(labels, dtype=np.int32)
    return _native.separability(x, list(names), g, lab, **kwargs)


def _score(row: dict[str, Any], mode: str) -> float:
    return row["balanced_accuracy"] if mode == "labels" else row["prediction_strength"]


def best_scores(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Best log combination per ``k``: ``k``, logs (names), score."""
    mode = report["mode"]
    best: dict[int, dict[str, Any]] = {}
    for r in report["scores"]:
        if r["k"] not in best or _score(r, mode) > _score(best[r["k"]], mode):
            best[r["k"]] = r
    return [
        {"k": k, "logs": [report["names"][i] for i in r["logs"]], "score": _score(r, mode)}
        for k, r in sorted(best.items())
    ]


def plot_separability(report: dict[str, Any], path: str | os.PathLike[str] | None = None, top: int = 6):  # noqa: ANN201
    """Score vs number of classes per log combination (needs matplotlib).

    Draws the ``top`` combinations with the best mean score, marks the
    recommended ``k``, and shows log importance. Saves to ``path`` if given;
    returns the figure.
    """
    import matplotlib.pyplot as plt

    mode, names = report["mode"], report["names"]
    combos: dict[tuple[int, ...], dict[int, float]] = {}
    for r in report["scores"]:
        combos.setdefault(tuple(r["logs"]), {})[r["k"]] = _score(r, mode)
    ranked = sorted(combos, key=lambda c: -float(np.mean(list(combos[c].values()))))[:top]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    for c in ranked:
        ks = sorted(combos[c])
        ax.plot(ks, [combos[c][k] for k in ks], marker="o", label="+".join(names[i] for i in c))
    k = report["recommended_k"]
    if k:
        ax.axvline(k, color="k", ls="--", lw=1, label=f"recommended k = {k}")
    ax.set_xlabel("number of classes")
    ax.set_ylabel("balanced accuracy" if mode == "labels" else "prediction strength")
    ax.legend(fontsize=7)
    imp = report["importance"]
    bx.barh(list(imp), list(imp.values()))
    bx.set_xlabel("score drop without log")
    best = "+".join(names[i] for i in report["best_logs"])
    fig.suptitle(f"Log separability ({mode}); best logs: {best or '-'}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
