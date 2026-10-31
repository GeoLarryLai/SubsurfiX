# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""``Project``: run the pipeline of one project folder and read its products."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from subsurfix import _native, products

RUN_SCHEMA = pa.schema(
    [
        pa.field("stage", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("seconds", pa.float64(), nullable=False),
        pa.field("n_wells", pa.int64(), nullable=False),
        pa.field("stage_key", pa.string()),
    ]
)

STATUS_SCHEMA = pa.schema(
    [
        pa.field("stage", pa.string(), nullable=False),
        pa.field("version", pa.int32(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("recorded_key", pa.string()),
        pa.field("finished_utc", pa.string()),
    ]
)


class Project:
    """One project folder, given by its ``project.toml``.

    The file is parsed once here, so a bad config fails at construction
    (``IoError``, ``ConfigError``). ``run`` and ``status`` re-read the file;
    ``config`` shows the file as it was when the ``Project`` was made.
    """

    def __init__(self, config_path: str | os.PathLike[str]) -> None:
        self.config_path: Path = Path(config_path).resolve()
        self.root: Path = self.config_path.parent
        self._config: dict[str, object] = json.loads(_native.config_json(self.config_path))

    def __repr__(self) -> str:
        return f"Project('{self.config_path}')"

    @property
    def config(self) -> dict[str, object]:
        """Deep copy of the parsed config, every default filled in."""
        return copy.deepcopy(self._config)

    def run(
        self,
        to: str | None = None,
        from_: str | None = None,
        force: bool = False,
        threads: int | None = None,
    ) -> pa.Table:
        """Run stages ``from_`` .. ``to`` (default: all) with the GIL released.

        Up-to-date stages are skipped unless ``force``. Returns one row per
        stage: ``stage``, ``action`` (``ran``, ``skipped``, ``checked``,
        ``out_of_range``), ``seconds``, ``n_wells``, ``stage_key``.
        """
        rows = _native.run_pipeline(self.config_path, from_, to, force, threads)
        return pa.Table.from_pylist(rows, schema=RUN_SCHEMA)

    def status(self) -> pa.Table:
        """Per stage: ``stage``, ``version``, ``state`` (``up_to_date``, ``stale``, ``never_run``), ``recorded_key``, ``finished_utc``."""
        rows = _native.pipeline_status(self.config_path)
        return pa.Table.from_pylist(rows, schema=STATUS_SCHEMA)

    def wells(self) -> pa.Table:
        """One row per well, sorted by ``well_id``."""
        return products.read_wells(self.root)

    def logs(self, well: int | str, stage: str = "final") -> pa.Table:
        """Curves of one well after ``stage`` (``raw``, ``qc``, ``norm``, ``filled``, ``final``)."""
        return products.read_logs(self.root, well, stage)

    def geometry(self, well: int | str) -> pa.Table:
        """Per-sample geometry of one well."""
        return products.read_geometry(self.root, well)

    def tops(self) -> pa.Table:
        """Consistent tops of every well."""
        return products.read_tops(self.root)

    def suspects(self) -> pa.Table:
        """Suspect tops, sorted by ``rank``."""
        return products.read_suspects(self.root)

    def layer_time(self, well: int | str) -> np.ndarray:
        """Layer time of one well: 1-D ``float64``, possibly read-only (``.copy()`` to modify)."""
        return products.read_layer_time(self.root, well)

    def subhorizons(self) -> pa.Table:
        """Sub-horizon picks of every well."""
        return products.read_subhorizons(self.root)

    def score_holdout(self, threads: int | None = None) -> dict[str, pa.Table]:
        """Score the fill on held-out data (the project must have run through ``fill``).

        A few wells spread over the area get intervals masked, the fill is rerun
        without them, and the predictions are compared with the measured values.
        Returns ``{"scores": per well and log, "summary": per log}``; each row has
        ``scope`` ``model`` (model rows only) or ``final`` (every filled row) and
        ``r2``, ``rmse``, ``bias`` among other columns. Also written to
        ``stages/fill/holdout/``. Slow: it reruns normalize, neighbors and fill.
        """
        scores, summary = _native.score_holdout(self.config_path, threads)
        return {"scores": _read_csv(scores), "summary": _read_csv(summary)}

    def evaluate_nphi_offset(self, plot: bool = True) -> dict[str, object]:
        """Estimate the field-wide NPHI offset against limestone density porosity.

        Uses measured logs after QC (bad hole masked), before normalization and fill;
        the project must have run through ``qc``. d = NPHI - DPHI_ls on clean, tight
        limestone: P10/P50/P90 per well and field-wide, field offset (median of well
        medians) with a bootstrap interval, and a robust fit of d against
        neutron-density porosity. ``status`` is ``ok`` only for a constant offset
        (interval excludes 0, small slope, enough wells); then the next ``run``
        subtracts ``shift`` from NPHI in normalize unless ``[nphi_offset] apply = false``
        or ``[normalize] nphi_shift`` is set. Writes ``stages/nphi_offset/`` (result.json,
        wells.csv, samples.csv and, with ``plot``, crossplot.png).
        """
        r: dict[str, object] = json.loads(_native.evaluate_nphi_offset(self.config_path))
        if plot:
            _plot_nphi_offset(self.root / "stages/nphi_offset", r)
        return r

    def nphi_shift(self) -> tuple[float, str]:
        """NPHI shift the next ``run`` applies and its source: ``manual``, ``evaluated`` or ``none``."""
        return _native.nphi_shift(self.config_path)

    def fill_summary(self) -> pa.Table:
        """Per well and log: measured, filled and unfilled samples, ``predicted_share``
        and ``mostly_predicted`` (share above ``[fill] mostly_predicted_share``).

        When :meth:`score_holdout` has run, ``holdout_r2`` (``final`` scope, pooled
        per log) is added; otherwise that column is null.
        """
        t = _read_csv(self.root / "stages/fill/fill_summary.csv")
        summary = self.root / "stages/fill/holdout/holdout_summary.csv"
        r2: dict[str, float] = {}
        if summary.exists():
            h = _read_csv(summary).to_pylist()
            r2 = {row["log"]: row["r2"] for row in h if row["scope"] == "final"}
        col = pa.array([r2.get(log) for log in t.column("log").to_pylist()], pa.float64())
        return t.append_column("holdout_r2", col)

    def temperature(self) -> pa.Table:
        """Temperature curves read from the LAS files, in degC on the ingest grid:
        ``well_id``, ``k``, ``md_ft``, ``temp_c`` (rows with a value only).

        These are logging-tool or mud temperatures (TEMP, MTEM, BHT ...), not
        formation temperatures. Empty when no LAS file has one.
        """
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        path = self.root / "stages/ingest/aux_raw.parquet"
        cols = ["well_id", "k", "md_ft", "temp_c"]
        if not path.exists():
            raise FileNotFoundError(f"{path}: run the ingest stage first")
        t = pq.read_table(path)
        if "temp_c" not in t.column_names:
            return t.select(cols[:3]).append_column("temp_c", pa.array([], pa.float32())).slice(0, 0)
        t = t.select(cols)
        return t.filter(pc.is_finite(t.column("temp_c")))

    def pts(self) -> pa.Table:
        """Pressure-temperature-spinner (PTS) survey rows from ``[[edges.pts]]``:
        ``well_id``, ``well_key``, ``pass_name``, ``seq``, ``md_ft``, ``temp_c``,
        ``pressure_mpa`` (absolute), ``flow``, ``time``, ``depth_tied``, ``source``.

        Rows with ``depth_tied`` false have no depth (NaN ``md_ft``). Empty when
        the project has no PTS source.
        """
        import pyarrow.parquet as pq

        from subsurfix.edges.schemas import PTS_SCHEMA

        path = self.root / "edges/pts.parquet"
        if not path.exists():
            return PTS_SCHEMA.empty_table()
        return pq.read_table(path)


def _plot_nphi_offset(folder: Path, r: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    s = _read_csv(folder / "samples.csv").to_pandas()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(s.dphi_ls, s.nphi, c=s.well_id, cmap="tab20", s=2, alpha=0.5)
    x = np.linspace(-0.05, 0.3, 50)
    ax[0].plot(x, x, "k-", lw=1, label="1:1")
    t = r.get("trend_phi")
    if isinstance(t, dict):
        a, b = float(t["intercept"]), float(t["slope"])
        # d = a + b (N + D) / 2  ->  N = (a + D (1 + b / 2)) / (1 - b / 2)
        ax[0].plot(x, (a + x * (1 + b / 2)) / (1 - b / 2), "r-", lw=1.5, label=f"fit: d = {a:.3f} + {b:.2f} PHI_ND")
    f = r.get("field_offset")
    if f is not None:
        ax[0].plot(x, x + float(f), "r--", lw=1, label=f"1:1 + {float(f):.3f}")
    ax[0].set(xlim=(-0.05, 0.3), ylim=(-0.05, 0.4), xlabel="DPHI limestone (v/v)", ylabel="NPHI (v/v)",
              title=f"Clean limestone, colored by well ({r['status']})")
    ax[0].legend(loc="upper left")
    d = (s.nphi - s.dphi_ls)[(s.nphi + s.dphi_ls) / 2 <= 0.05]
    ax[1].hist(d, bins=80, range=(-0.15, 0.3), color="0.5")
    p = r.get("field_pct")
    if isinstance(p, dict):
        for k, st in [("p10", ":"), ("p50", "-"), ("p90", ":")]:
            ax[1].axvline(float(p[k]), color="r", ls=st, label=f"{k.upper()} {float(p[k]):.3f}")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set(xlabel="NPHI - DPHI (v/v), tight samples", ylabel="samples", title="Offset distribution")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "crossplot.png", dpi=110)
    plt.close(fig)


def _read_csv(path: str | os.PathLike[str]) -> pa.Table:
    """CSV product with ``well_name`` kept as text (API numbers keep leading zeros)."""
    opts = pacsv.ConvertOptions(column_types={"well_name": pa.string()})
    return pacsv.read_csv(path, convert_options=opts)


__all__ = ["Project", "RUN_SCHEMA", "STATUS_SCHEMA"]
