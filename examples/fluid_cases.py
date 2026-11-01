# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""The four fluid cases of example 12 as importable functions.

Same block model, wells, rates and solver settings as the notebook cells, so
``subsurfix.parallel.run_cases`` can run them in separate processes. Each
function returns the solver's result dict; ``linear_solver`` is passed to the
solver ("iterative", the default, or "direct").
"""

from __future__ import annotations

import numpy as np

from block_model import column_wells, layered_block
from subsurfix import _native as nat

YR = 3.15576e7
G = 9.80665
INJ, PROD = (5, 5), (2, 2)


def _setup(years: int = 40):
    bm = layered_block()
    nx, ny, nz = bm["dims"]
    model = nat.FlowModel.from_regular(list(bm["origin"]), list(bm["cell"]), [nx, ny, nz], bm["perm"], bm["phi"])
    times = [0.0] + [y * YR for y in range(2, years + 1, 2)]
    return bm, model, bm["phi"].size, times


def _wells(bm, q_inj, bhp_prod):
    return column_wells(bm, INJ, PROD, rate=q_inj, bhp=bhp_prod)


def water_oil(years: int = 40, linear_solver: str = "iterative") -> dict:
    """Case 1: dead-oil water injection."""
    bm, model, n, times = _setup(years)
    return nat.run_fim_water_oil(
        model, _wells(bm, 0.008, 1.2e7), np.full(n, 0.2), np.full(n, 2e7), times,
        corey={"mu_o": 2e-3}, physics={"gravity": G, "rho_o": 850.0, "c_w": 4e-10, "c_o": 1e-9},
        p_range=(5e6, 5e7), table_points=(31, 201), dt_init=86400.0, dt_max=0.25 * YR, linear_solver=linear_solver)


def oil_gas(years: int = 40, linear_solver: str = "iterative") -> dict:
    """Case 2: gas injection into a black oil."""
    bm, model, n, times = _setup(years)
    p_tab = 1e5 + 3e6 * np.arange(11)
    f = (p_tab - 1e5) / 3e7
    pvt = np.column_stack([p_tab, 1 + 150 * f, 1.05 + 0.35 * f, 2e-3 - 1.2e-3 * f, 1 / (1 + 300 * f), 1.2e-5 + 1.3e-5 * f])
    p0 = 2e7
    rs = np.interp(p0, pvt[:, 0], pvt[:, 1])
    bo = np.interp(p0, pvt[:, 0], pvt[:, 2])
    rho_sc = (0.9, 850.0, 1020.0)
    rs_mass = rs * rho_sc[0] / rho_sc[1]
    oil_mass = 0.8 * (rho_sc[1] + rs * rho_sc[0]) / bo
    water_mass = 0.2 * rho_sc[2] / 1.01 * (1 + 4e-10 * (p0 - 1e5))
    zo0 = oil_mass / (1 + rs_mass) / (oil_mass + water_mass)
    zg0 = 0.999 * zo0 * rs_mass
    return nat.run_fim_black_oil(
        model, _wells(bm, 0.008, 1.0e7), np.full(n, p0), np.full(n, zg0), np.full(n, zo0), times,
        pvt=pvt, bw=1.01, gravity=G, injection=[(1.0, 0.0), (0.0, 0.0)],
        p_range=(5e6, 3e7), table_points=(51, 101, 101), dt_init=86400.0, dt_max=0.25 * YR, linear_solver=linear_solver)


def co2_brine(years: int = 40, linear_solver: str = "iterative") -> dict:
    """Case 3: isothermal CO2 injection into brine."""
    bm, model, n, times = _setup(years)
    return nat.run_fim_co2_brine(
        model, _wells(bm, 0.008, 1.2e7), np.full(n, 2e7), np.zeros(n), times,
        temperature=323.15, m_nacl=1.0, gravity=G, p_range=(5e6, 5e7), table_points=[61, 201],
        dt_init=86400.0, dt_max=0.25 * YR, linear_solver=linear_solver)


def co2_brine_cold(years: int = 40, linear_solver: str = "iterative") -> dict:
    """Case 4: cold CO2 injection with the energy equation."""
    bm, model, n, times = _setup(years)
    return nat.run_fim_co2_brine(
        model, _wells(bm, 0.008, 1.2e7), np.full(n, 2e7), np.zeros(n), times,
        m_nacl=1.0, thermal=True, t0=np.full(n, 333.15), injection_temperature=[298.15, 333.15],
        gravity=G, p_range=(5e6, 5e7), t_range=(290.0, 340.0), table_points=[31, 101, 26],
        dt_init=86400.0, dt_max=0.25 * YR, linear_solver=linear_solver)


CASES = {"water-oil": water_oil, "oil-gas": oil_gas, "CO2-brine": co2_brine, "CO2-brine cold": co2_brine_cold}
