# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Experimental CO2-brine properties (Spycher et al. 2003; Spycher & Pruess 2005;
Ziabakhsh-Ganji & Kooi 2012)."""

import numpy as np
import pytest
from subsurfix import _native
from subsurfix.errors import InvalidInputError


def test_solubility_matches_duan_sun_table() -> None:
    # Duan & Sun (2003) Tables 3 and 4, 333.15 K, 200 bar: 1.2344 (pure), 1.0072 (1 m NaCl).
    out = _native.co2_solubility(
        np.array([200e5, 200e5]), np.array([333.15, 333.15]), np.array([0.0, 1.0])
    )
    np.testing.assert_allclose(out["m_co2"], [1.2344, 1.0072], rtol=0.05)
    assert np.all(out["y_h2o"] > 0)


def test_zgk_matches_published_table_2_6() -> None:
    # Ziabakhsh-Ganji (2015) thesis Table 2.6, 323.15 K, pure water: experimental
    # x_CO2 (%) and the AD % reported for the ZGK model. Ours lies within 1 point.
    p = np.array([101.33e5, 200e5, 405.3e5])
    x_exp = np.array([1.98, 2.30, 2.606])
    ad_paper = np.array([0.425269, 0.528422, 1.094991])
    out = _native.co2_solubility(
        p, np.full(3, 323.15), np.zeros(3), model="ziabakhsh_ganji_kooi_2012"
    )
    ad = np.abs(out["x_co2"] * 100.0 - x_exp) / x_exp * 100.0
    assert np.all(np.abs(ad - ad_paper) < 1.0)


def test_flash_two_phase_and_errors() -> None:
    f = _native.co2_brine_flash(150e5, 323.15, 0.3, 1.0)
    assert 0.0 < f["nu_gas"] < 1.0 and 0.0 < f["s_gas"] < 1.0
    assert f["rho_aq"] > f["rho_gas"] > 0.0
    g = _native.co2_brine_flash(
        150e5, 323.15, 0.3, 1.0, model="ziabakhsh_ganji_kooi_2012"
    )
    assert 0.0 < g["nu_gas"] < 1.0
    with pytest.raises(InvalidInputError):
        _native.co2_brine_flash(700e5, 323.15, 0.3, 1.0)
    with pytest.raises(ValueError):
        _native.co2_brine_flash(150e5, 323.15, 0.3, 1.0, model="nope")
