# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""MODFLOW 6 bridge (experimental): SubsurfiX property grid -> MODFLOW 6 GWF model via FloPy.

Single-phase, constant-density groundwater flow. Conversions (SI units, metres and seconds):

* hydraulic conductivity  K = k * rho * g / mu          (k in m^2)
* specific storage        Ss = rho * g * (c_rock + phi * c_w)   (1/m)
* well connection conductance  C = WI * rho * g / mu   (WI = Peaceman well index, m^3)
* pressure  p = rho * g * h  (SubsurfiX's ``solve_pressure`` has no gravity term, so the
  MODFLOW head is compared as a potential, p = rho g h, not as p = rho g (h - z)).

Defaults: water at about 20 C, rho = 1000 kg/m3, mu = 1e-3 Pa s, g = 9.80665 m/s2.
Cell ordering follows SubsurfiX: index = i + nx * (j + ny * k), with k = 0 the top layer.
MODFLOW layer k, row j, column i maps to the same cell. Wells use the MAW package with
specified connection conductance so the Peaceman indices match SubsurfiX's.

This module is experimental; the API may change.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["G", "mf6_available", "read_modflow6", "run_modflow6", "to_modflow6"]

G = 9.80665


def mf6_available(exe_name: str = "mf6") -> bool:
    """True when the MODFLOW 6 binary is on PATH or in FloPy's bin folder (``get-modflow :flopy``)."""
    return _find_exe(exe_name) is not None


def _find_exe(exe_name: str) -> str | None:
    p = shutil.which(exe_name)
    if p:
        return p
    fb = Path.home() / ".local" / "share" / "flopy" / "bin" / exe_name
    return str(fb) if fb.exists() else None


def _peaceman(kx: float, ky: float, dx: float, dy: float, dz: float, rw: float, skin: float) -> float:
    from subsurfix import _native as nat

    return float(nat.peaceman_wi(kx, ky, dx, dy, dz, rw, skin))


def to_modflow6(
    ws: str | os.PathLike[str],
    origin: tuple[float, float, float],
    cell: tuple[float, float, float],
    dims: tuple[int, int, int],
    perm: np.ndarray,
    poro: np.ndarray,
    wells: list[dict[str, Any]] | None = None,
    *,
    rho: float = 1000.0,
    mu: float = 1e-3,
    g: float = G,
    c_rock: float = 0.0,
    c_w: float = 0.0,
    p0: float | np.ndarray | None = None,
    times: list[float] | None = None,
    nstp: int = 10,
    name: str = "ssx",
    exe_name: str = "mf6",
):
    """Build (and write) a MODFLOW 6 simulation from a regular SubsurfiX grid.

    ``perm`` is (n, 3) permeability in m^2 (kx, ky, kz); ``poro`` is (n,).
    ``wells`` use the SubsurfiX dict format: ``kind`` 'injector' (``rate`` m3/s, positive in)
    or 'producer' (``bhp`` Pa, or ``rate`` m3/s), ``cells``, optional ``wi``, ``radius`` (0.1 m), ``skin``.
    ``times`` None gives one steady-state period; otherwise transient periods ending at each
    time (s), with ``nstp`` steps each and storage from ``c_rock``/``c_w`` (1/Pa).
    Returns the ``flopy.mf6.MFSimulation`` (already written to ``ws``).
    """
    import flopy

    nx, ny, nz = (int(v) for v in dims)
    n = nx * ny * nz
    perm = np.asarray(perm, float).reshape(n, 3)
    poro = np.asarray(poro, float).reshape(n)
    dx, dy, dz = (float(v) for v in cell)
    kfac = rho * g / mu
    shp = (nz, ny, nx)
    ws = Path(ws)
    ws.mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(ws), exe_name=_find_exe(exe_name) or exe_name)
    steady = times is None
    if steady:
        pdata = [(1.0, 1, 1.0)]
    else:
        ends = np.asarray(times, float)
        lens = np.diff(np.concatenate([[0.0], ends]))
        pdata = [(float(t), int(nstp), 1.2) for t in lens]
    flopy.mf6.ModflowTdis(sim, time_units="seconds", nper=len(pdata), perioddata=pdata)
    flopy.mf6.ModflowIms(sim, complexity="MODERATE", outer_dvclose=1e-8, inner_dvclose=1e-9,
                         outer_maximum=200, inner_maximum=1000,
                         rcloserecord=[1e-10, "RELATIVE_RCLOSE"], linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    top = -float(origin[2])
    botm = top - dz * (np.arange(nz) + 1.0)
    flopy.mf6.ModflowGwfdis(gwf, nlay=nz, nrow=ny, ncol=nx, delr=dx, delc=dy, top=top, botm=botm,
                            xorigin=float(origin[0]), yorigin=float(origin[1]), length_units="meters")
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=(perm[:, 0] * kfac).reshape(shp),
                            k22=(perm[:, 1] * kfac).reshape(shp), k33=(perm[:, 2] * kfac).reshape(shp),
                            save_specific_discharge=True)
    if p0 is None:
        strt = np.zeros(shp)
    else:
        strt = (np.broadcast_to(np.asarray(p0, float), (n,)) / (rho * g)).reshape(shp)
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    if not steady:
        ss = rho * g * (c_rock + poro * c_w)
        flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=ss.reshape(shp), sy=poro.reshape(shp),
                                transient={0: True})

    wells = wells or []
    if wells:
        pkg, conn, per = [], [], []
        for iw, w in enumerate(wells):
            cells = [int(c) for c in w["cells"]]
            rw = float(w.get("radius", 0.1))
            wi = w.get("wi")
            if wi is None:
                wi = [_peaceman(perm[c, 0], perm[c, 1], dx, dy, dz, rw, float(w.get("skin", 0.0))) for c in cells]
            bname = str(w.get("name", f"W{iw + 1}")).lower()
            h0 = float(np.mean([strt.ravel()[c] for c in cells]))
            kmax = max(c // (nx * ny) for c in cells)
            pkg.append([iw, rw, top - dz * (kmax + 1), h0, "SPECIFIED", len(cells), bname])
            for ic, c in enumerate(cells):
                k, rem = divmod(c, nx * ny)
                j, i = divmod(rem, nx)
                ztop = top - dz * k
                conn.append([iw, ic, (k, j, i), ztop, ztop - dz, float(wi[ic]) * kfac, rw])
            if w["kind"] == "producer" and "bhp" in w:
                per += [[iw, "STATUS", "CONSTANT"], [iw, "WELL_HEAD", float(w["bhp"]) / (rho * g)]]
            else:
                q = float(w["rate"])
                per += [[iw, "RATE", q if w["kind"] == "injector" else -abs(q)]]
        maw = flopy.mf6.ModflowGwfmaw(gwf, boundnames=True, no_well_storage=True, nmawwells=len(wells),
                                      packagedata=pkg, connectiondata=conn, perioddata={0: per},
                                      head_filerecord=f"{name}.maw.hds", budget_filerecord=f"{name}.maw.cbc")
        obs = {f"{name}.maw.obs.csv": [(f"{p[6]}_q", "maw", p[6]) for p in pkg]
               + [(f"{p[6]}_h", "head", p[0] + 1) for p in pkg]}
        maw.obs.initialize(continuous=obs)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", budget_filerecord=f"{name}.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim._ssx = {"rho": rho, "g": g, "mu": mu, "n": n, "name": name, "wells": [p[6] for p in pkg] if wells else []}
    sim.write_simulation(silent=True)
    return sim


def run_modflow6(sim, silent: bool = True) -> None:
    """Run the MODFLOW 6 binary on a simulation from :func:`to_modflow6`; raises on failure."""
    ok, buff = sim.run_simulation(silent=silent)
    if not ok:
        raise RuntimeError("MODFLOW 6 did not converge or failed:\n" + "\n".join(buff[-20:]))


def read_modflow6(sim) -> dict[str, Any]:
    """Read results in SubsurfiX order.

    Returns ``times`` (s), ``head`` (n_times, n) in m, ``p`` = rho g head (Pa), and ``wells``:
    a list of dicts with ``name``, ``rate`` (m3/s, positive into the aquifer) and ``bhp`` (Pa)
    at the last time.
    """
    info = sim._ssx
    gwf = sim.get_model(info["name"])
    hf = gwf.output.head()
    times = np.asarray(hf.get_times())
    head = np.array([hf.get_data(totim=t).ravel() for t in times])
    out: dict[str, Any] = {"times": times, "head": head, "p": head * info["rho"] * info["g"], "wells": []}
    obsf = Path(sim.sim_path) / f"{info['name']}.maw.obs.csv"
    if obsf.exists():
        import csv

        with open(obsf) as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        for b in info["wells"]:
            out["wells"].append({"name": b, "rate": -float(last[f"{b.upper()}_Q"]),
                                 "bhp": float(last[f"{b.upper()}_H"]) * info["rho"] * info["g"]})
    return out
