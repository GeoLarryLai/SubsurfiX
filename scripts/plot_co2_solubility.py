# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Plot CO2 solubility in NaCl brine vs pressure (EXPERIMENTAL model).

Spycher et al. (2003) + Spycher & Pruess (2005), with Duan & Sun (2003)
Table 3-6 values at 333.15 K as markers. Usage: python scripts/plot_co2_solubility.py [out.png]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from subsurfix import _native

DUAN_SUN_333K = {  # bar: mol/kg, Duan & Sun (2003) Tables 3-6
    0.0: {50: 0.6695, 100: 1.0275, 200: 1.2344, 300: 1.3495, 400: 1.4478},
    1.0: {50: 0.5502, 100: 0.8405, 200: 1.0072, 300: 1.1012},
    2.0: {50: 0.4583, 100: 0.6978, 200: 0.8359, 300: 0.9160, 400: 0.9873},
    4.0: {50: 0.3314, 100: 0.5028, 200: 0.6060, 300: 0.6717, 400: 0.7340},
}


def main(out: str = "co2_solubility.png") -> None:
    p_bar = np.linspace(5, 600, 120)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, ref in DUAN_SUN_333K.items():
        s = _native.co2_solubility(
            p_bar * 1e5, np.full_like(p_bar, 333.15), np.full_like(p_bar, m)
        )
        (line,) = ax.plot(p_bar, s["m_co2"], label=f"{m:g} m NaCl")
        ax.plot(list(ref), list(ref.values()), "o", color=line.get_color(), ms=4)
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("CO$_2$ solubility (mol/kg H$_2$O)")
    ax.set_title("60 °C: Spycher & Pruess (2005) lines, Duan & Sun (2003) dots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:2])
