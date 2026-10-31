# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Heat transport, corner-point options and upwind reporting through the Python API."""

import numpy as np
import pytest

from subsurfix import _native as nat

K = 1e-12


def test_run_heat_cools_near_injector_and_balances():
    n = 50
    m = nat.FlowModel.from_regular([0, 0, 0], [2, 1, 1], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2))
    wells = [
        {"name": "I", "kind": "injector", "cells": [0], "rate": 1e-5, "radius": 0.1},
        {"name": "P", "kind": "producer", "cells": [n - 1], "bhp": 1e7, "radius": 0.1},
    ]
    r = m.run_heat(np.full(n, 1e3), wells, [20.0, 0.0], np.full(n, 80.0), [1e6, 1e7], dt_max=1e5)
    t = r["temperature"]
    assert t.shape == (2, n)
    assert t[1, 0] < t[0, 0] < 80.0
    assert np.all(t[1] <= 80.0 + 1e-9)
    assert r["energy_change"][-1] == pytest.approx(r["energy_in"][-1], rel=1e-9)
    with pytest.raises(ValueError, match="unknown key"):
        m.run_heat(np.full(n, 1e3), wells, [20.0, 0.0], np.full(n, 80.0), [1e6], props={"bogus": 1.0})
    fx = [{"cell": 0, "conductance": 2.0 * m.boundary_geom(0, 0), "temperature": 10.0}]
    r = m.run_heat(np.full(n, 1e3), wells, [20.0, 0.0], np.full(n, 80.0), [1e6], fixed_t=fx)
    assert r["temperature"][0, 0] < t[0, 0]


def test_corner_point_ntg_mult_and_upwind_info():
    g = nat.CornerPointGrid.from_regular([0, 0, 0], [10, 10, 10], [3, 1, 1])
    perm, poro = np.full((3, 3), K), np.full(3, 0.2)
    base = nat.FlowModel.from_corner_point(g, perm, poro)
    assert base.warnings == []
    m = nat.FlowModel.from_corner_point(g, perm, poro, ntg=np.full(3, 0.5), mult={"multx": [0.1, 1, 1]})
    assert m.connections()[2][0] == pytest.approx(0.05 * base.connections()[2][0])
    assert m.pore_volume()[0] == pytest.approx(0.5 * base.pore_volume()[0])
    with pytest.raises(ValueError, match="unknown multiplier"):
        nat.FlowModel.from_corner_point(g, perm, poro, mult={"MULTQ": [1, 1, 1]})
    sol = base.solve_pressure(
        np.array([1e3, 2e3, 3e3]),
        [{"name": "P", "kind": "producer", "cells": [2], "bhp": 1e7, "radius": 0.1}],
        fixed=[nat.FixedPressure(0, base.boundary_trans(0, 0), 2e7, pressure_unit="Pa")],
    )
    assert sol["warnings"] == [] and sol["upwind_iterations"] >= 1


def test_boundary_inflow_mobility():
    m = nat.FlowModel.from_regular([0, 0, 0], [10, 10, 10], [5, 1, 1], np.full((5, 3), K), np.full(5, 0.2))
    t = m.boundary_trans(0, 0)
    fx = [nat.FixedPressure(0, t, 2e7), nat.FixedPressure(4, t, 1e7)]
    a = m.solve_pressure(np.full(5, 1e3), [], fixed=fx)
    b = m.solve_pressure(np.full(5, 1e3), [], fixed=fx, boundary_mobility=1e4)
    assert b["boundary_flux"][0] > a["boundary_flux"][0] > 0
    assert b["boundary_flux"][0] == pytest.approx(-b["boundary_flux"][1])


def test_adaptive_heat_steps():
    n = 50
    m = nat.FlowModel.from_regular([0, 0, 0], [2, 1, 1], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2))
    wells = [
        {"name": "I", "kind": "injector", "cells": [0], "rate": 1e-5, "radius": 0.1},
        {"name": "P", "kind": "producer", "cells": [n - 1], "bhp": 1e7, "radius": 0.1},
    ]
    args = (np.full(n, 1e3), wells, [20.0, 0.0], np.full(n, 80.0), [1e7])
    r = m.run_heat(*args, dt_max=1e7, dt_init=1e3, dt_change=2.0)
    assert 3 < r["n_steps"] < 1e4 and r["n_rejected"] >= 0
    assert r["energy_change"][-1] == pytest.approx(r["energy_in"][-1], rel=1e-9)
    with pytest.raises(ValueError):
        m.run_heat(*args, dt_change=-1.0)


def test_water_properties_and_heat_props():
    w = nat.water_properties(np.array([3e6, 10e6]), np.array([300.0, 423.15]))
    assert 1.0 / w["density"][0] == pytest.approx(0.100215168e-2, rel=1e-8)
    assert w["cp"][0] == pytest.approx(4173.01218, rel=1e-8)
    assert w["viscosity"][1] == pytest.approx(185.02182e-6, rel=1e-6)
    with pytest.raises(ValueError, match="saturation"):
        nat.water_properties(np.array([1e5]), np.array([400.0]))
    n = 10
    m = nat.FlowModel.from_regular([0, 0, 0], [2, 1, 1], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2))
    wells = [
        {"name": "I", "kind": "injector", "cells": [0], "rate": 1e-6, "radius": 0.1},
        {"name": "P", "kind": "producer", "cells": [n - 1], "bhp": 1e7, "radius": 0.1},
    ]
    args = (np.full(n, 1e3), wells, [20.0, 0.0], np.full(n, 80.0), [1e5])
    r = m.run_heat(*args, props={"water_p": 1e7, "water_t_k": 353.15})
    assert r["energy_change"][-1] == pytest.approx(r["energy_in"][-1], rel=1e-9)
    with pytest.raises(ValueError, match="together"):
        m.run_heat(*args, props={"water_p": 1e7})


def test_bhp_limit_and_crossflow_through_python():
    m = nat.FlowModel.from_regular([0, 0, 0], [10, 10, 5], [10, 1, 1], np.full((10, 3), K), np.full(10, 0.2))
    fx = [nat.FixedPressure(0, m.boundary_trans(0, 0), 2e7)]
    w = [{"name": "P", "kind": "producer", "cells": [9], "rate": 1.0, "bhp_limit": 150.0, "bhp_unit": "bar"}]
    s = m.solve_pressure(np.full(10, 1e3), w, fixed=fx)
    assert s["wells"][0]["bhp_limited"] and s["wells"][0]["bhp"] == pytest.approx(1.5e7)
    with pytest.raises(ValueError, match="rate-controlled"):
        m.solve_pressure(np.full(10, 1e3), [{"kind": "producer", "cells": [9], "bhp": 1e7, "bhp_limit": 1e7}], fixed=fx)
    w = [{"kind": "producer", "cells": [9], "bhp": 1e7, "crossflow": False}]
    assert m.solve_pressure(np.full(10, 1e3), w, fixed=fx)["wells"][0]["bhp_limited"] is False


def test_basal_heat_flux_geotherm():
    n, dx, lam = 20, 50.0, 2.0
    m = nat.FlowModel.from_regular([0, 0, 0], [dx, 1, 1], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2))
    fixed = [{"cell": 0, "trans": m.boundary_trans(0, 0), "pressure": 1e7}]
    props = {"cond_rock": lam, "cond_fluid": lam}
    fx = [{"cell": 0, "conductance": lam * m.boundary_geom(0, 0), "temperature": 15.0}]
    hf = [{"cell": n - 1, "power": 0.06}]
    r = m.run_heat(np.full(n, 1e3), [], [], np.full(n, 15.0), [3e13], fixed=fixed, fixed_temp=[15.0],
                   fixed_t=fx, heat_flux=hf, props=props, dt_max=3e11)
    want = 15.0 + 0.06 * (np.arange(n) + 0.5) * dx / lam
    np.testing.assert_allclose(r["temperature"][0], want, atol=1e-6)
    with pytest.raises(ValueError, match="cell, power"):
        m.run_heat(np.full(n, 1e3), [], [], np.full(n, 15.0), [1.0], fixed=fixed, heat_flux=[{"cell": 0}])
