# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pyarrow as pa
import pytest

RUN_COLUMNS = ["stage", "action", "seconds", "n_wells", "stage_key"]


@pytest.fixture(scope="module")
def first_run(synthetic_project) -> pa.Table:
    return synthetic_project.run()


def test_run_status_and_reads(synthetic_project, first_run) -> None:
    assert first_run.column_names == RUN_COLUMNS
    assert first_run.num_rows == 10
    assert set(first_run.column("action").to_pylist()) == {"ran"}

    status = synthetic_project.status()
    assert status.num_rows == 10
    assert set(status.column("state").to_pylist()) == {"up_to_date"}

    wells = synthetic_project.wells()
    assert wells.num_rows == 6
    well = wells.column("well_id")[0].as_py()
    n_samples = synthetic_project.geometry(well).num_rows
    assert n_samples > 0

    logs = synthetic_project.logs(well)
    assert logs.num_rows == n_samples
    assert set(logs.column("well_id").to_pylist()) == {well}

    tops = synthetic_project.tops()
    assert tops.num_rows > 0
    assert {"well_id", "top_name", "md_ft", "suspect", "rank"} <= set(tops.column_names)

    tau = synthetic_project.layer_time(well)
    assert isinstance(tau, np.ndarray)
    assert tau.dtype == np.float64 and tau.ndim == 1
    assert len(tau) == n_samples
    assert np.isfinite(tau).any()


def test_second_run_skips(synthetic_project, first_run) -> None:
    again = synthetic_project.run()
    assert set(again.column("action").to_pylist()) == {"skipped"}
