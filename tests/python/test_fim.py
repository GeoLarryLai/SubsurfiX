# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Experimental fully implicit (OBL) flow: Buckley-Leverett front and mass balance."""
import numpy as np
import pytest

from subsurfix import _native as nat

K = 1e-13
DAY = 86400.0


def line(n, dx):
    return nat.FlowModel.from_regular(
        [0, 0, 0], [dx, dx, dx], [n, 1, 1], np.full((n, 3), K), np.full(n, 0.2)
    )


def test_fim_water_oil_buckley_leverett_front_and_mass_balance():
    n, q = 200, 1e-5
    m = line(n, 1.0)
    pv = m.pore_volume().sum()
    corey = {"mu_o": 2e-3}
    physics = {"rho_w": 1000.0, "rho_o": 1000.0}
    tds = [0.2, 0.4]
    times = [td * pv / q for td in tds]
    wells = [
        {"name": "I", "kind": "injector", "rate": q, "cells": [0]},
        {"name": "P", "kind": "producer", "bhp": 1e7, "cells": [n - 1]},
    ]
    swc = 0.2  # Corey default connate water
    r = nat.run_fim_water_oil(
        m, wells, np.full(n, swc), np.full(n, 1e7), times, corey=corey, physics=physics,
        p_range=(5e6, 1e8), table_points=(16, 501), dt_init=times[0] / 200, dt_max=times[0] / 100,
    )
    assert len(r["phase_names"]) == 2
    xd = (np.arange(n) + 0.5) / n
    xs = np.arange(2000) / 2000
    for k, td in enumerate(tds):
        sw = r["saturation"][k][:, 0]
        want = np.array([nat.buckley_leverett(x, td, corey) for x in xd])
        assert np.abs(sw - want).mean() < 0.03
        a = np.array([nat.buckley_leverett(x, td, corey) for x in xs])
        ia = int(np.argmax(a <= swc + 1e-12))
        half = 0.5 * (swc + a[ia - 1])
        x_num = np.argmax(sw < half) / n
        assert abs(x_num - xs[ia]) < 0.03
        gained = r["mass_in_place"][k][0] - pv * 1000.0 * swc
        inj = r["mass_net_in"][k][0]
        assert gained == pytest.approx(inj, rel=1e-6)
        assert inj == pytest.approx(1000.0 * q * times[k], rel=1e-4)


def test_fim_black_oil_depletion_mass_balance():
    n = 20
    m = line(n, 10.0)
    p = 1e5 + 3e6 * np.arange(11)
    f = (p - 1e5) / 3e7
    pvt = np.column_stack(
        [p, 1 + 150 * f, 1.05 + 0.35 * f, 2e-3 - 1.2e-3 * f, 1 / (1 + 300 * f), 1.2e-5 + 1.3e-5 * f]
    )
    p0 = 1e5 + 2.4e7
    row = pvt[np.argmax(pvt[:, 0] >= p0)]
    rs_mass = row[1] * 0.9 / 850.0
    so_m = 0.8 * (850.0 + row[1] * 0.9) / row[2]
    w_m = 0.2 * 1020.0 / 1.01 * (1 + 4e-10 * (p0 - 1e5))
    zo = so_m / (1 + rs_mass) / (so_m + w_m) * 0.999
    zg = zo * rs_mass * 0.95
    wells = [{"name": "P", "kind": "producer", "bhp": 5e6, "cells": [n - 1]}]
    kw = {
        "pvt": pvt,
        "bw": 1.01,
        "p_range": (1e6, 3e7),
        "table_points": (59, 101, 101),
        "z_gas_max": 0.1,
        "dt_init": 3600.0,
        "dt_max": 5 * DAY,
        "tol": 1e-9,
    }
    args = (m, wells, np.full(n, p0), np.full(n, zg), np.full(n, zo))
    init = nat.run_fim_black_oil(*args, [0.0], **kw)["mass_in_place"][0]
    r = nat.run_fim_black_oil(*args, [30 * DAY, 365 * DAY], **kw)
    total = init.sum()
    for k in range(2):
        d = r["mass_in_place"][k] - init
        assert np.all(np.abs(d - r["mass_net_in"][k]) < 1e-7 * total)
    assert r["saturation"][1][:, 0].max() > 0.01
    assert r["well_mass_rate"][1][0][1] < 0


M_CO2, M_H2O = 44.0095e-3, 18.015268e-3


def test_fim_co2_brine_mass_balance_and_dissolved_co2():
    n, q, p0 = 50, 2e-6, 1.2e7
    m = line(n, 1.0)
    pv = m.pore_volume().sum()
    wells = [
        {"name": "I", "kind": "injector", "rate": q, "cells": [0]},
        {"name": "P", "kind": "producer", "bhp": p0, "cells": [n - 1]},
    ]
    kw = {
        "temperature": 323.15,
        "m_nacl": 1.0,
        "p_range": (8e6, 2e7),
        "table_points": [25, 201],
        "dt_init": 600.0,
        "dt_max": 2 * DAY,
        "tol": 1e-8,
    }
    args = (m, wells, np.full(n, p0), np.zeros(n))
    init = nat.run_fim_co2_brine(*args, [0.0], **kw)["mass_in_place"][0]
    r = nat.run_fim_co2_brine(*args, [0.15 * pv / q, 0.3 * pv / q], **kw)
    assert r["component_names"] == ["CO2", "H2O"]
    assert r["n_fallbacks"] == 0
    for k in range(2):
        d = r["mass_in_place"][k] - init
        assert np.allclose(d, r["mass_net_in"][k], rtol=1e-6, atol=1e-6 * init.max())
    sg = r["saturation"][1][:, 1]
    assert sg.max() > 0.2
    # Brine ahead of the plume holds at most the CO2 solubility of the
    # property module at the cell pressure.
    z = r["z"][1][:, 0]
    ahead = (sg == 0) & (z > 1e-6)
    sol = nat.co2_solubility(r["p"][1], np.full(n, 323.15), np.full(n, 1.0))
    x_mass = sol["x_co2"] * M_CO2 / (sol["x_co2"] * M_CO2 + (1 - sol["x_co2"]) * M_H2O)
    assert np.all(z[ahead] <= x_mass[ahead] * 1.0001)
    # Two-phase cells hold more CO2 than dissolved alone.
    assert np.all(z[sg > 0.05] > x_mass[sg > 0.05])


def test_fim_thermal_water_energy_balance_and_cooling():
    n, q, p0, t0 = 60, 1e-5, 1e7, 350.0
    m = line(n, 1.0)
    wells = [
        {"name": "I", "kind": "injector", "rate": q, "cells": [0]},
        {"name": "P", "kind": "producer", "bhp": p0, "cells": [n - 1]},
    ]
    kw = {
        "injection_temperature": [300.0, t0],
        "p_range": (5e6, 3e7),
        "t_range": (290.0, 360.0),
        "table_points": (26, 141),
        "dt_max": 0.2 * DAY,  # long implicit steps smear the front (numerical diffusion)
    }
    args = (m, wells, np.full(n, p0), np.full(n, t0))
    init = nat.run_fim_thermal_water(*args, [0.0], **kw)["mass_in_place"][0]
    r = nat.run_fim_thermal_water(*args, [10 * DAY, 20 * DAY], **kw)
    for k in range(2):
        d = r["mass_in_place"][k] - init
        assert np.allclose(d, r["mass_net_in"][k], rtol=1e-6)
    t = r["temperature"][1]
    # Far field: untouched by the cold front (compression heating ~0.2 K).
    assert t[0] < 302.0 and t[-1] == pytest.approx(t0, abs=0.5)
    # Monotone cold-to-warm profile, up to the small pressure-work warming ahead of the front.
    assert np.all(np.diff(t) >= -0.01)
