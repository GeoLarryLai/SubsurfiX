# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""MODFLOW 6 heat transport (GWF + GWE) for a SubsurfiX regular grid. **Experimental.**

Builds a coupled groundwater-flow (GWF) and groundwater-energy (GWE) simulation with FloPy, so a SubsurfiX
heat run (``FlowModel.run_heat``) can be cross-checked against MODFLOW 6 (6.5 or newer, Langevin et al. 2024,
https://doi.org/10.5066/P9FL1JCC). Water and rock properties are constant, matching ``run_heat``'s ``props``.

Wells: single-cell wells are a specified-flux well (WEL) with an injection temperature (injector) and a
constant-head cell (CHD, producer). Wells open in several cells are first solved as MODFLOW 6 multi-aquifer
wells (MAW) with Peaceman connection conductances (WI * rho * g / mu) and the producer's BHP / (rho g) as
well head; the simulated per-connection rates then become WEL cells (with the injection temperature) for the
GWE run, with one producer cell held at its simulated head (CHD). This reproduces the MAW flow field
without GWE's MWE package. Flow is steady state, constant density, no gravity
term (pressure = rho g head). ``perm_m2`` is (n,) isotropic or (n, 3) for kx, ky, kz.

The grid conversion lives in :func:`regular_grid_to_dis` only; it will move onto ``subsurfix.modflow`` later.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

G = 9.80665

__all__ = ["HeatDoublet", "build_gwf_gwe", "find_mf6", "regular_grid_to_dis", "run_gwf_gwe"]


@dataclass
class HeatDoublet:
    """A regular-grid heat case. Arrays are in SubsurfiX natural order (i fastest, then j, then k)."""

    origin: tuple[float, float, float]
    cell: tuple[float, float, float]
    dims: tuple[int, int, int]
    perm_m2: np.ndarray
    poro: np.ndarray
    t0: np.ndarray
    wells: list[dict[str, Any]]
    well_temp: list[float]
    times: list[float]
    props: dict[str, float] = field(default_factory=lambda: {
        "rho_fluid": 1000.0, "cp_fluid": 4180.0, "rho_rock": 2650.0, "cp_rock": 900.0, "cond_fluid": 0.6, "cond_rock": 2.5})
    viscosity: float = 5e-4
    dt_max: float = 86400.0


def regular_grid_to_dis(origin, cell, dims) -> dict[str, Any]:
    """SubsurfiX regular grid to MODFLOW 6 DIS keywords. Depth (z down) becomes elevation (top = -z0).

    Returns ``nlay, nrow, ncol, delr, delc, top, botm, xorigin, yorigin`` and ``index``, a function from a
    SubsurfiX cell id to a MODFLOW (layer, row, col). MODFLOW rows run north to south, so row = ny-1-j.
    """
    nx, ny, nz = (int(d) for d in dims)
    dx, dy, dz = (float(c) for c in cell)
    top = -float(origin[2])
    botm = np.array([top - dz * (k + 1) for k in range(nz)])

    def index(c: int) -> tuple[int, int, int]:
        i, j, k = c % nx, (c // nx) % ny, c // (nx * ny)
        return k, ny - 1 - j, i

    return {"nlay": nz, "nrow": ny, "ncol": nx, "delr": dx, "delc": dy, "top": top,
                "botm": botm[:, None, None] * np.ones((nz, ny, nx)), "xorigin": float(origin[0]),
                "yorigin": float(origin[1]), "index": index}


def to_layered(a, dims) -> np.ndarray:
    """SubsurfiX natural-order cell array to a MODFLOW (nlay, nrow, ncol) array."""
    nx, ny, nz = dims
    return np.asarray(a, float).reshape(nz, ny, nx)[:, ::-1, :]


def from_layered(a, dims) -> np.ndarray:
    """Inverse of :func:`to_layered`."""
    return np.asarray(a)[..., ::-1, :].reshape(*np.shape(a)[:-3], -1)


def find_mf6() -> str | None:
    """Path of an ``mf6`` executable on PATH or in FloPy's bin folder (``get-modflow :flopy``), or None."""
    from subsurfix.modflow import _find_exe

    return _find_exe("mf6")


def build_gwf_gwe(case: HeatDoublet, ws: str | Path, exe: str | None = None):
    """Write a coupled GWF+GWE simulation for ``case`` into ``ws``; returns the FloPy simulation."""
    import flopy

    p = case.props
    d = regular_grid_to_dis(case.origin, case.cell, case.dims)
    idx = d.pop("index")
    xo, yo = d.pop("xorigin"), d.pop("yorigin")
    kfac = p["rho_fluid"] * G / case.viscosity
    perm = np.asarray(case.perm_m2, float)
    perm3 = perm.reshape(-1, 3) if perm.ndim == 2 else np.column_stack([perm] * 3)
    k_hyd = to_layered(perm3[:, 0] * kfac, case.dims)
    multi = any(len(w["cells"]) > 1 for w in case.wells)
    edges = [0.0, *case.times]
    perioddata = []
    for a, b in itertools.pairwise(edges):
        n = max(1, int(np.ceil((b - a) / case.dt_max - 1e-9)))
        perioddata.append((b - a, n, 1.0))
    sim = flopy.mf6.MFSimulation(sim_name="sxheat", sim_ws=str(ws), exe_name=exe or find_mf6() or "mf6")
    flopy.mf6.ModflowTdis(sim, nper=len(perioddata), perioddata=perioddata, time_units="seconds")

    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    ims_f = flopy.mf6.ModflowIms(sim, filename="gwf.ims", complexity="moderate", linear_acceleration="bicgstab")
    sim.register_ims_package(ims_f, ["gwf"])
    flopy.mf6.ModflowGwfdis(gwf, length_units="meters", xorigin=xo, yorigin=yo, **d)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k_hyd, k22=to_layered(perm3[:, 1] * kfac, case.dims),
                            k33=to_layered(perm3[:, 2] * kfac, case.dims), save_specific_discharge=True,
                            save_saturation=True)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfsto(gwf, steady_state={0: True}, iconvert=0, ss=0.0, sy=0.0)
    wel, chd = [], []
    if multi:
        wel, chd = _maw_split(case, Path(ws) / "maw_split", exe, d, idx, xo, yo, k_hyd, perm3, kfac)
    for w, tw in zip([] if multi else case.wells, case.well_temp):
        for c in w["cells"]:
            if w["kind"] == "injector":
                wel.append([idx(c), float(w["rate"]) / len(w["cells"]), float(tw)])
            else:
                chd.append([idx(c), 0.0])
    if wel:
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data={0: wel}, auxiliary=["TEMPERATURE"], pname="WEL-1")
    if chd:
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: chd}, pname="CHD-1")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbc",
                           saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")])

    gwe = flopy.mf6.ModflowGwe(sim, modelname="gwe")
    ims_e = flopy.mf6.ModflowIms(sim, filename="gwe.ims", complexity="moderate", linear_acceleration="bicgstab",
                                 inner_dvclose=1e-8, outer_dvclose=1e-8)
    sim.register_ims_package(ims_e, ["gwe"])
    flopy.mf6.ModflowGwedis(gwe, length_units="meters", xorigin=xo, yorigin=yo, **d)
    flopy.mf6.ModflowGweic(gwe, strt=to_layered(case.t0, case.dims))
    flopy.mf6.ModflowGweadv(gwe, scheme="upstream")
    flopy.mf6.ModflowGwecnd(gwe, xt3d_off=True, alh=0.0, ath1=0.0, ktw=p["cond_fluid"], kts=p["cond_rock"])
    flopy.mf6.ModflowGweest(gwe, porosity=to_layered(case.poro, case.dims), heat_capacity_water=p["cp_fluid"],
                            density_water=p["rho_fluid"], heat_capacity_solid=p["cp_rock"],
                            density_solid=p["rho_rock"])
    flopy.mf6.ModflowGwessm(gwe, sources=[("WEL-1", "AUX", "TEMPERATURE")] if wel else None)
    flopy.mf6.ModflowGweoc(gwe, temperature_filerecord="gwe.ucn", saverecord=[("TEMPERATURE", "LAST")])
    flopy.mf6.ModflowGwfgwe(sim, exgtype="GWF6-GWE6", exgmnamea="gwf", exgmnameb="gwe")
    return sim


def _maw_split(case: HeatDoublet, ws: Path, exe, d, idx, xo, yo, k_hyd, perm3, kfac):
    """Steady MODFLOW 6 flow run with multi-aquifer wells; returns WEL rows (per-connection rates, with the
    injection temperature) and one CHD row (a producer cell at its simulated head) for the GWE run."""
    import flopy

    ws.mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name="split", sim_ws=str(ws), exe_name=exe or find_mf6() or "mf6")
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)], time_units="seconds")
    flopy.mf6.ModflowIms(sim, complexity="moderate", linear_acceleration="bicgstab", outer_dvclose=1e-9,
                         inner_dvclose=1e-10, rcloserecord=[1e-10, "RELATIVE_RCLOSE"], outer_maximum=200,
                         inner_maximum=1000)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(gwf, length_units="meters", xorigin=xo, yorigin=yo, **d)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k_hyd, k22=to_layered(perm3[:, 1] * kfac, case.dims),
                            k33=to_layered(perm3[:, 2] * kfac, case.dims))
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    _add_maw(gwf, case, idx, perm3, kfac)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbc",
                           saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")])
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("mf6 MAW flow run failed:\n" + "\n".join(buff[-20:]))
    head = flopy.utils.HeadFile(ws / "gwf.hds").get_data()
    cbc = flopy.utils.CellBudgetFile(ws / "maw.cbc")
    rec = cbc.get_data(text="GWF")[-1]
    q_of = {int(n) - 1: -float(q) for n, q in zip(rec["node2"], rec["q"])}  # +: into the aquifer
    nrow, ncol = d["nrow"], d["ncol"]

    def node(c: int) -> int:
        k, r, i = idx(c)
        return k * nrow * ncol + r * ncol + i

    wel, chd = [], []
    for w, tw in zip(case.wells, case.well_temp):
        cells = [int(c) for c in w["cells"]]
        if w["kind"] == "producer" and not chd:
            c0 = cells[0]
            chd.append([idx(c0), float(head[idx(c0)])])
            cells = cells[1:]
        for c in cells:
            wel.append([idx(c), q_of.get(node(c), 0.0), float(tw)])
    return wel, chd


def _add_maw(gwf, case: HeatDoublet, idx, perm3: np.ndarray, kfac: float) -> None:
    """Multi-aquifer wells with Peaceman connection conductances (SubsurfiX's well model)."""
    import flopy

    from subsurfix import _native as nat

    dx, dy, dz = (float(c) for c in case.cell)
    top = -float(case.origin[2])
    pkg, conn, per = [], [], []
    for iw, w in enumerate(case.wells):
        rw, skin = float(w.get("radius", 0.1)), float(w.get("skin", 0.0))
        cells = [int(c) for c in w["cells"]]
        wi = w.get("wi") or [nat.peaceman_wi(perm3[c, 0], perm3[c, 1], dx, dy, dz, rw, skin) for c in cells]
        kk = [idx(c)[0] for c in cells]
        pkg.append([iw, rw, top - dz * (max(kk) + 1), 0.0, "SPECIFIED", len(cells)])
        for ic, c in enumerate(cells):
            ztop = top - dz * idx(c)[0]
            conn.append([iw, ic, idx(c), ztop, ztop - dz, float(wi[ic]) * kfac, rw])
        if w["kind"] == "injector" or "bhp" not in w:
            q = float(w["rate"])
            per.append([iw, "RATE", q if w["kind"] == "injector" else -abs(q)])
        else:
            per += [[iw, "STATUS", "CONSTANT"], [iw, "WELL_HEAD", float(w["bhp"]) / (case.props["rho_fluid"] * G)]]
    flopy.mf6.ModflowGwfmaw(gwf, no_well_storage=True, budget_filerecord="maw.cbc", nmawwells=len(pkg), packagedata=pkg, connectiondata=conn,
                            perioddata={0: per}, pname="MAW-1")


def run_gwf_gwe(case: HeatDoublet, ws: str | Path, exe: str | None = None) -> dict[str, Any]:
    """Build and run; returns ``times`` (s) and ``temperature`` (n_times, n_cells) in SubsurfiX order."""
    import flopy

    Path(ws).mkdir(parents=True, exist_ok=True)
    sim = build_gwf_gwe(case, ws, exe)
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("mf6 failed:\n" + "\n".join(buff[-20:]))
    ucn = flopy.utils.HeadFile(Path(ws) / "gwe.ucn", text="TEMPERATURE")
    t = np.array(ucn.get_times())
    T = np.array([from_layered(ucn.get_data(totim=x), case.dims) for x in t])
    return {"times": t, "temperature": T}
