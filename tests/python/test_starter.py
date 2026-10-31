# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quick-start helpers: template, init_project, demo project."""
import subprocess
import sys

import pytest

import subsurfix as sx


def test_template_has_core_sections():
    t = sx.template_toml()
    for key in ("[ingest]", "las_dirs", "[project]"):
        assert key in t


def test_init_project_writes_toml(tmp_path):
    p = sx.init_project(tmp_path / "p", crs="EPSG:32614", las_dirs=["las"])
    text = open(p).read()
    assert "EPSG:32614" in text and "las" in text
    with pytest.raises(Exception):
        sx.init_project(tmp_path / "p", crs="EPSG:32614", las_dirs=["las"])


def test_demo_project_runs(tmp_path):
    # Own process: the thread pool is process-wide and other tests size it.
    code = (
        "import subsurfix as sx\n"
        f"t = sx.write_demo_project({str(tmp_path / 'demo')!r}, n_wells=6, seed=1)\n"
        "sx.prepare_edges(t)\n"
        "p = sx.Project(t)\n"
        "p.run(threads=2)\n"
        # A second thread count in the same process runs on its own pool.
        "p.run(threads=3, force=True)\n"
        "f = p.fill_summary()\n"
        "assert f.num_rows > 0 and f.column('holdout_r2').null_count == f.num_rows\n"
        "rows = f.to_pylist()\n"
        "assert any(r['mostly_predicted'] for r in rows), rows\n"
        "h = p.score_holdout()\n"
        "assert set(h) == {'scores', 'summary'} and 'r2' in h['summary'].column_names\n"
        "assert p.fill_summary().column('holdout_r2').null_count < f.num_rows\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
