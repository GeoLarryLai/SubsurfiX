# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Generate the NumPy percentile fixture for sx-core util tests."""

import json
from pathlib import Path

import numpy as np


def as_json(x: float):
    value = float(x)
    if np.isnan(value):
        return None
    return value


def main() -> None:
    rng = np.random.default_rng(1)
    cases = []
    for _ in range(50):
        n = int(rng.integers(1, 60))
        values = rng.normal(size=n) * rng.uniform(0.1, 1000)
        n_nan = int(rng.integers(0, n // 4 + 1))
        if n_nan:
            positions = rng.choice(n, size=n_nan, replace=False)
            values = values.copy()
            values[positions] = np.nan
        ps = np.concatenate(
            [rng.uniform(0, 100, size=5), np.array([0.0, 50.0, 100.0])]
        )
        expected = np.atleast_1d(np.nanpercentile(values, ps)).astype(float)
        if np.all(np.isnan(expected)):
            expected_json = None
        else:
            expected_json = [as_json(x) for x in expected]
        cases.append(
            {
                "values": [as_json(x) for x in values],
                "ps": [float(p) for p in ps],
                "expected": expected_json,
            }
        )
    out = Path(__file__).resolve().parents[1] / "util_percentile_numpy.json"
    text = json.dumps({"cases": cases}, indent=2, allow_nan=False)
    out.write_text(text + "\n", encoding="utf-8")
    size = out.stat().st_size
    if size >= 200 * 1024:
        raise SystemExit(f"fixture is {size} bytes, 200 KB limit")
    print(f"wrote {out} ({size} bytes)")


if __name__ == "__main__":
    main()
