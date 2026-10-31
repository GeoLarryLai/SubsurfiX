# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Flow bindings: TPFA pressure, Peaceman wells, tracer, IMPES."""
import numpy as np
import pytest

from subsurfix import _native as nat

K, MU = 1e-13, 1e-3


def line(n=100, dx=10.0):
    return nat.FlowModel.from_regular(
        [0, 0, 0], [dx, 10, 10], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2)
    )


def test_linear_pressure():
    m = line()
    fixed = [
        nat.FixedPressure(0, m.boundary_trans(0, 0), 20.0, pressure_unit="MPa"),
        {"cell": 99, "trans": m.boundary_trans(99, 0), "pressure": 1e7},
    ]
    sol = m.solve_pressure(np.full(100, 1 / MU), [], fixed)
    x = (np.arange(100) + 0.5) * 10
    assert np.allclose(sol["p"], 2e7 - 1e7 * x / 1000, rtol=1e-8)
    assert np.allclose(sol["flux"], K * 100 * 1e7 / (MU * 1000))


def test_cornerpoint_model_matches_regular():
    g = nat.CornerPointGrid.from_regular([0, 0, 0], [10, 10, 10], [4, 3, 2])
    a = nat.FlowModel.from_corner_point(g, np.full((24, 3), K), np.full(24, 0.2))
    b = nat.FlowModel.from_regular([0, 0, 0], [10, 10, 10], [4, 3, 2], np.full((24, 3), K), np.full(24, 0.2))
    assert np.allclose(a.connections()[2], b.connections()[2])
    assert a.pore_volume().sum() == pytest.approx(24 * 1000 * 0.2)


def test_wells_tracer_and_impes():
    n = 200
    m = line(n, 1.0)
    wells = [
        {"name": "I", "kind": "injector", "rate": 1e-5, "cells": [0]},
        {"name": "P", "kind": "producer", "bhp": 1e7, "cells": [n - 1]},
    ]
    sol = m.solve_pressure(np.full(n, 1 / MU), wells)
    assert sol["wells"][0]["rate"] == pytest.approx(1e-5)
    assert sol["wells"][1]["rate"] == pytest.approx(-1e-5)
    pv = m.pore_volume().sum()
    t = 0.5 * pv / 1e-5
    tr = m.run_tracer(np.full(n, 1 / MU), wells, [1.0, 0.0], np.zeros(n), [t])
    assert (tr["conc"][0] * m.pore_volume()).sum() == pytest.approx(1e-5 * t)
    corey = {"mu_o": 2e-3}
    r = m.run_impes(wells, np.full(n, 0.2), [0.3 * pv / 1e-5], corey=corey)
    xd = (np.arange(n) + 0.5) / n
    want = np.array([nat.buckley_leverett(x, 0.3, corey) for x in xd])
    assert np.abs(r["sw"][0] - want).mean() < 0.02
    assert r["oil_rate"].shape == (1, 2)


def test_peaceman_and_errors():
    wi = nat.peaceman_wi(K, K, 10, 10, 10, 0.1)
    assert wi == pytest.approx(2 * np.pi * K * 10 / np.log(0.198 * 10 / 0.1), rel=1e-3)
    m = line(5, 1.0)
    with pytest.raises(Exception):
        m.solve_pressure(np.ones(5), [{"kind": "injector", "rate": 1.0, "cells": [0]}])
    with pytest.raises(ValueError):
        m.solve_pressure(np.ones(5), [{"kind": "both", "rate": 1.0, "cells": [0]}])


G = 9.80665


def column(nz=20, dz=2.0, k=1e-12):
    return nat.FlowModel.from_regular(
        [0, 0, 1000], [10, 10, dz], [1, 1, nz], np.full((nz, 3), k), np.full(nz, 0.2)
    )


def test_two_phase_hydrostatic_and_segregation():
    m = column()
    phys = {"gravity": G, "rho_w": 1000.0, "rho_o": 700.0}
    corey0 = {"swc": 0.0, "sor": 0.0}
    r = m.run_two_phase([], np.ones(20), np.full(20, 1e7), [1e4], fixed=[nat.FixedPressure(0, m.boundary_trans(0, 2), 1e7)],
                        corey=corey0, physics=phys)
    z = m.centers()[:, 2]
    assert np.allclose(r["p"][0], 1e7 + 1000 * G * (z - z[0]), rtol=1e-10)
    sw0 = np.where(np.arange(20) < 10, 0.9, 0.1)
    corey = {"swc": 0.1, "sor": 0.1}
    r = m.run_two_phase([], sw0, np.full(20, 1e7), [1e8], corey=corey, physics={**phys, "c_rock": 1e-9})
    assert r["sw"][0][:10].mean() < 0.15 and r["sw"][0][10:].mean() > 0.85
    assert r["water_in_place"][0] == pytest.approx((sw0 * m.pore_volume()).sum(), rel=1e-3)


def test_capillary_curves():
    bc = {"capillary": "brooks_corey", "pd": 1e4, "lambda": 2.0}
    assert nat.capillary_pressure(0.8, {"swc": 0.2, "sor": 0.2}, bc) == pytest.approx(1e4)
    vg = {"capillary": "van_genuchten", "alpha": 1e-4, "n": 2.0}
    assert nat.capillary_pressure(0.8, {"swc": 0.2, "sor": 0.2}, vg) == pytest.approx(0.0, abs=1e-6)
    assert nat.capillary_pressure(0.5, {"swc": 0.2, "sor": 0.2}, vg) > 0
    with pytest.raises(ValueError):
        nat.capillary_pressure(0.5, None, {"capillary": "magic"})


def test_theis_transient_drawdown():
    n, dx, h, phi, ct, q = 41, 10.0, 10.0, 0.2, 1e-9, 1e-3
    m = nat.FlowModel.from_regular([0, 0, 0], [dx, dx, h], [n, n, 1], np.full((n * n, 3), K), np.full(n * n, phi))
    c = n // 2
    well = {"kind": "producer", "rate": q, "cells": [c + n * c]}
    r = m.run_two_phase([well], np.ones(n * n), np.full(n * n, 2e7), [2000.0], corey={"swc": 0.0, "sor": 0.0},
                        physics={"c_rock": ct}, max_dt=50.0)
    got = 2e7 - r["p"][0][c + 5 + n * c]
    want = nat.theis_drawdown(q, MU, K, h, phi, ct, 50.0, 2000.0)
    assert got == pytest.approx(want, rel=0.02)


def test_deviated_well_completions():
    m = nat.FlowModel.from_regular([0, 0, 0], [10, 10, 10], [10, 3, 3], np.full((90, 3), K), np.full(90, 0.2))
    comp = m.trajectory_completions(np.array([[0, 15, 15], [100, 15, 15]], dtype=float))
    assert len(comp["cells"]) == 10 and comp["length"][:, 0].sum() == pytest.approx(100)
    want = nat.peaceman_wi(K, K, 10, 10, 10, 0.1)
    assert np.allclose(comp["wi"], want)
    well = {"kind": "producer", "bhp": 9e6, "cells": comp["cells"], "wi": comp["wi"]}
    sol = m.solve_pressure(np.full(90, 1 / MU), [well], [nat.FixedPressure(0, m.boundary_trans(0, 1), 1e7)])
    assert sol["wells"][0]["wi"] == pytest.approx(list(comp["wi"]))


def test_two_phase_compressible_mass_balance():
    nx = 6
    n = nx * nx * 3
    m = nat.FlowModel.from_regular([0, 0, 1000], [100, 100, 5], [nx, nx, 3], np.full((n, 3), K), np.full(n, 0.2))
    col = lambda i, j: [i + nx * (j + nx * k) for k in range(3)]
    wells = [{"kind": "injector", "rate": 1e-3, "cells": col(0, 0)}, {"kind": "producer", "bhp": 1.5e7, "cells": col(5, 5)}]
    phys = {"gravity": G, "rho_w": 1050.0, "rho_o": 800.0, "capillary": "brooks_corey", "pd": 5e3, "lambda": 2.0,
            "c_rock": 5e-10, "c_w": 4e-10, "c_o": 1e-9}
    p0 = np.full(n, 2e7)
    r = m.run_two_phase(wells, np.full(n, 0.2), p0, [3e7], corey={"swc": 0.2, "sor": 0.2}, physics=phys)
    b = lambda c: 1 + c * (2e7 - 1e7)
    pv = m.pore_volume() * b(5e-10)
    w0, o0 = (pv * 0.2 * b(4e-10)).sum(), (pv * 0.8 * b(1e-9)).sum()
    assert r["water_in_place"][0] - w0 == pytest.approx(r["water_net_in"][0], rel=1e-9)
    assert r["oil_in_place"][0] - o0 == pytest.approx(r["oil_net_in"][0], rel=1e-9)
    assert r["water_clamp_loss"][0] == 0.0


def test_flow_input_validation():
    m = line(10)
    t = m.boundary_trans(0, 0)
    with pytest.raises(ValueError, match=r"tuples are no longer accepted; write FixedPressure\(0, .*, 10000000\.0, pressure_unit='Pa'\)"):
        m.solve_pressure(np.full(10, 1 / MU), [], [(0, t, 1e7)])
    with pytest.raises(ValueError, match=r"use FixedPressure\(cell, trans, pressure"):
        m.solve_pressure(np.full(10, 1 / MU), [], [[0, t]])
    with pytest.raises(Exception, match="trans"):
        nat.FixedPressure(0, -1.0, 1e7)
    with pytest.raises(ValueError, match="pressure unit"):
        nat.FixedPressure(0, t, 1.0, pressure_unit="psig")
    fx = [nat.FixedPressure(9, m.boundary_trans(9, 0), 1e7)]
    bad = {"name": "I", "kind": "injector", "rate": -1e-3, "cells": [0]}
    with pytest.raises(Exception, match="magnitudes"):
        m.solve_pressure(np.full(10, 1 / MU), [bad], fx)
    with pytest.raises(ValueError, match="rate unit"):
        m.solve_pressure(np.full(10, 1 / MU), [dict(bad, rate=1.0, rate_unit="gpm")], fx)
    with pytest.raises(ValueError, match="unknown key"):
        m.solve_pressure(np.full(10, 1 / MU), [dict(bad, rate=1.0, rat=2)], fx)
    a = m.solve_pressure(np.full(10, 1 / MU), [dict(bad, rate=86.4, rate_unit="m3/d")], fx)
    b = m.solve_pressure(np.full(10, 1 / MU), [dict(bad, rate=1e-3)], fx)
    assert np.allclose(a["p"], b["p"])
