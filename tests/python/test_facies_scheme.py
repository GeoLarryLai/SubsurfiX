# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np

from subsurfix import facies


def test_interpret_pure_rocks_and_coarse_rows() -> None:
    gr = np.array([30.0, 30.0, 120.0, 30.0])
    rhob = np.array([2.65 * 0.85 + 0.15, 2.71 * 0.9 + 0.1, 2.5, 2.62])
    nphi = np.array([-0.02 * 0.85 + 0.15, 0.1, 0.3, 0.05])
    pef = np.array([1.9, 4.9, 3.4, np.nan])
    for method in ["cutoffs", "mixing"]:
        r = facies.interpret(gr, rhob, nphi, pef, method=method)
        names = [facies.FACIES[i] for i in r["facies"]]
        assert names == ["sandstone", "limestone", "shale", "carbonate"], (method, names)
        assert list(r["rollup"]) == [0, 2, 1, 2]
        assert list(r["coarse"]) == [False, False, False, True]
    assert "v_quartz" in facies.interpret(gr, rhob, nphi, pef, method="mixing")


def test_cutoffs_csv_round_trip(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "cutoffs.csv"
    text = facies.default_cutoffs(path)
    assert text.splitlines()[0] == "order,facies,variable,min,max,coarse"
    r = facies.interpret([120.0], [2.5], [0.3], [3.4], cutoffs_csv=str(path))
    assert r["facies"][0] == 1


def test_rollup_classify() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 100).astype(np.int32)
    x = np.column_stack([y * 5.0 + rng.normal(size=300), rng.normal(size=300)])
    assert (facies.rollup_classify(x, y, x) == y).mean() > 0.95
