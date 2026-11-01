# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Run independent simulations side by side, one process each.

Cases that do not depend on each other (for example the four fluids of
example 12) can run at the same time in separate Python processes. Each
case is a function plus keyword arguments; the function must be importable
(defined in a module or script file, not typed into a notebook cell) because
it is sent to a fresh process. Its return value must be picklable (the
result dicts of the ``run_fim_*`` solvers are).

``parallel=False`` runs the same cases one after another in this process,
which gives the same results and is the fallback where extra processes are
not wanted.

Example::

    from subsurfix.parallel import run_cases
    out = run_cases({"a": (my_module.case_a, {}), "b": (my_module.case_b, {"rate": 0.01})})
    out.results["a"], out.seconds["a"], out.wall_seconds
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CaseRuns", "run_cases"]

Case = tuple[Callable[..., Any], Mapping[str, Any]]


@dataclass
class CaseRuns:
    """Results of :func:`run_cases`.

    ``results`` and ``seconds`` (run time of each case inside its process)
    are keyed by case name; ``wall_seconds`` is the elapsed time of the whole
    call; ``processes`` is the number of worker processes (0 = serial).
    """

    results: dict[str, Any] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    processes: int = 0


def _init_worker(threads: int) -> None:
    # One solver thread per process unless asked otherwise, so N processes
    # do not each start a pool the size of the machine.
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    from subsurfix import _native

    if hasattr(_native, "set_flow_threads"):
        _native.set_flow_threads(threads)


def _timed(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> tuple[Any, float]:
    t = time.perf_counter()
    r = fn(**kwargs)
    return r, time.perf_counter() - t


def run_cases(
    cases: Mapping[str, Case],
    parallel: bool = True,
    max_workers: int | None = None,
    threads_per_case: int = 1,
) -> CaseRuns:
    """Run independent cases, each in its own process (or serially).

    Args:
        cases: ``{name: (function, kwargs)}``; ``function(**kwargs)`` runs one case.
        parallel: ``False`` runs the cases one after another in this process.
        max_workers: processes to use (default: the number of cases, capped by
            the CPU count).
        threads_per_case: solver threads inside each process (parallel mode).

    Returns:
        :class:`CaseRuns` with the results in the order of ``cases``.
    """
    out = CaseRuns()
    t0 = time.perf_counter()
    if not parallel or len(cases) <= 1:
        for name, (fn, kw) in cases.items():
            out.results[name], out.seconds[name] = _timed(fn, kw)
        out.wall_seconds = time.perf_counter() - t0
        return out
    n = max_workers or min(len(cases), os.cpu_count() or 1)
    # "spawn" works the same on Linux, macOS and Windows and never copies a
    # running solver thread pool into a child process.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker, initargs=(threads_per_case,)) as ex:
        futs = {name: ex.submit(_timed, fn, kw) for name, (fn, kw) in cases.items()}
        for name, f in futs.items():
            out.results[name], out.seconds[name] = f.result()
    out.wall_seconds = time.perf_counter() - t0
    out.processes = n
    return out
