# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np

from subsurfix import facies


def test_nphi_shift_equals_shifted_input() -> None:
    gr, rh, np_, pe = np.array([20.0, 80.0]), np.array([2.55, 2.45]), np.array([0.12, 0.25]), np.array([5.0, 3.2])
    a = facies.interpret(gr, rh, np_, pe, nphi_shift=0.03)
    b = facies.interpret(gr, rh, np_ - 0.03, pe)
    np.testing.assert_array_equal(a["rhomaa"], b["rhomaa"])
    np.testing.assert_array_equal(a["facies"], b["facies"])
    assert a["rhomaa"][0] < facies.interpret(gr, rh, np_, pe)["rhomaa"][0]
