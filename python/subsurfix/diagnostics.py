# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Flow diagnostics (time of flight, tracer partitions, F-Phi, Lorenz, sweep).

The solvers are in Rust (derived from MRST's ``diagnostics`` module, see
``NOTICE``); this module adds the producer tracer response used to compare
the time-stepped tracer (the default) with the TOF prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from subsurfix._native import FlowModel, f_phi, lorenz_coefficient, sweep_efficiency

__all__ = [
    "f_phi",
    "flow_diagnostics",
    "lorenz_coefficient",
    "producer_tracer_response",
    "sweep_efficiency",
    "total_mobility",
]

_COREY_DEFAULT = {"swc": 0.2, "sor": 0.2, "nw": 2.0, "no": 2.0, "krw_max": 1.0, "kro_max": 1.0, "mu_w": 1e-3, "mu_o": 1e-3}


def flow_diagnostics(model: FlowModel, mobility: npt.NDArray[np.float64], wells: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """Run :meth:`FlowModel.flow_diagnostics` and add ``F``, ``Phi``, ``lorenz``, ``Ev`` and ``tD``."""
    d = model.flow_diagnostics(mobility, wells, **kw)
    f, phi = f_phi(model.pore_volume(), d["tof_forward"], d["tof_backward"])
    ev, td = sweep_efficiency(f, phi)
    d.update(F=f, Phi=phi, lorenz=lorenz_coefficient(f, phi), Ev=ev, tD=td)
    return d


def total_mobility(sat_w: npt.NDArray[np.float64], corey: dict[str, float]) -> npt.NDArray[np.float64]:
    """Total mobility ``krw/mu_w + kro/mu_o`` (1/(Pa s)) of a water-oil state with Corey curves.

    ``corey`` keys (defaults as the solvers): swc 0.2, sor 0.2, nw 2, no 2, krw_max 1, kro_max 1,
    mu_w 1e-3, mu_o 1e-3 (Pa s).
    Used to freeze a multiphase state for flow diagnostics.
    """
    corey = {**_COREY_DEFAULT, **corey}
    sw = np.asarray(sat_w, dtype=float)
    se = np.clip((sw - corey["swc"]) / (1.0 - corey["swc"] - corey["sor"]), 0.0, 1.0)
    krw = corey["krw_max"] * se ** corey["nw"]
    kro = corey["kro_max"] * (1.0 - se) ** corey["no"]
    return krw / corey["mu_w"] + kro / corey["mu_o"]


def producer_tracer_response(
    model: FlowModel,
    mobility: npt.NDArray[np.float64],
    wells: list[dict[str, Any]],
    times: list[float] | npt.NDArray[np.float64],
    method: str = "transport",
    cfl: float = 0.9,
) -> dict[str, Any]:
    """Flux-weighted tracer concentration at the producers for unit tracer at the injectors.

    ``method="transport"`` (default): time-stepped explicit upwind tracer
    (:meth:`FlowModel.run_tracer`). ``method="tof"``: steady prediction from
    time of flight, the fraction of produced flux whose total travel time
    ``tau_f + tau_b`` is at most ``t`` (no numerical dispersion).
    Returns ``{"times", "conc", "method"}`` (and ``"n_steps"`` for transport).
    """
    t = np.asarray(times, dtype=float)
    if method == "transport":
        sol = model.solve_pressure(mobility, wells)
        conc_in = [1.0 if w["kind"] == "injector" else 0.0 for w in wells]
        r = model.run_tracer(mobility, wells, conc_in, np.zeros(model.n_cells), list(t), cfl=cfl)
        num = np.zeros(len(t))
        den = 0.0
        for w, ws in zip(wells, sol["wells"]):
            if w["kind"] != "producer":
                continue
            q = -np.asarray(ws["cell_rates"])
            num += r["conc"][:, list(w["cells"])] @ q
            den += q.sum()
        return {"times": t, "conc": num / den, "method": method, "n_steps": r["n_steps"]}
    if method == "tof":
        d = model.flow_diagnostics(mobility, wells)
        tau = d["tof_forward"] + d["tof_backward"]
        pv = model.pore_volume()
        ok = np.isfinite(tau) & (tau > 0) & (tau < d["max_tof"])
        tau, pv = tau[ok], pv[ok]
        flux = pv / tau
        order = np.argsort(tau)
        cf = np.cumsum(flux[order]) / flux.sum()
        idx = np.searchsorted(tau[order], t, side="right")
        conc = np.where(idx > 0, cf[np.maximum(idx - 1, 0)], 0.0)
        return {"times": t, "conc": conc, "method": method}
    raise ValueError(f"unknown method {method!r}; use 'transport' or 'tof'")
