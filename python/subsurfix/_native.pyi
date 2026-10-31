# SPDX-License-Identifier: AGPL-3.0-or-later
"""Type stubs for subsurfix._native. Every native function has a stub here."""

import os

import numpy as np
import numpy.typing as npt
from typing import Any

def _raise_sx_error(variant: str, message: str) -> None: ...
def version() -> str: ...
def template_toml() -> str: ...
def stage_names() -> list[str]: ...
def stage_products() -> dict[str, list[str]]: ...
def config_json(config_path: str | os.PathLike[str]) -> str: ...
def run_pipeline(
    config_path: str | os.PathLike[str], from_stage: str | None = None,
    to_stage: str | None = None, force: bool = False, threads: int | None = None,
) -> list[dict[str, object]]: ...
def pipeline_status(config_path: str | os.PathLike[str]) -> list[dict[str, object]]: ...
def score_holdout(
    config_path: str | os.PathLike[str], threads: int | None = None
) -> tuple[str, str]: ...
def init_logging(level: str = "info") -> bool: ...
def register_model(log: str, model: object) -> None: ...
def clear_models(log: str | None = None) -> None: ...
def registered_models() -> dict[str, str]: ...
def _make_synthetic_project(
    dir: str | os.PathLike[str], n_wells: int = 10, seed: int = 0,
) -> str: ...
def _plugin_roundtrip(
    log: str, model: object, x_fit: npt.NDArray[np.float32], names: list[str],
    y: npt.NDArray[np.float32], x_pred: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.float32]: ...

# --- regular grid, upscaling, variograms, IDW, ordinary kriging ---
# Points are float64 arrays of shape (n, 3). Cell index = i + nx*(j + ny*k).
def upscale_continuous(
    origin: tuple[float, float, float], cell: tuple[float, float, float], dims: tuple[int, int, int],
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64],
    method: str = "arithmetic", min_count: int = 1,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.float64], npt.NDArray[np.uint64]]: ...
def upscale_facies(
    origin: tuple[float, float, float], cell: tuple[float, float, float], dims: tuple[int, int, int],
    points: npt.NDArray[np.float64], codes: npt.NDArray[np.int32], min_count: int = 1,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.uint64]]: ...
def grid_centers(
    origin: tuple[float, float, float], cell: tuple[float, float, float], dims: tuple[int, int, int],
) -> npt.NDArray[np.float64]: ...
def experimental_variogram(
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], lag_width: float, n_lags: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint64]]: ...
def fit_variogram(
    lag: npt.NDArray[np.float64], gamma: npt.NDArray[np.float64], pairs: npt.NDArray[np.uint64],
    kind: str = "spherical",
) -> dict[str, float | str]: ...
def idw(
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], targets: npt.NDArray[np.float64],
    power: float = 2.0, max_neighbors: int = 12, max_distance: float = ...,
) -> npt.NDArray[np.float64]: ...
def ordinary_kriging(
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], targets: npt.NDArray[np.float64],
    kind: str, nugget: float, psill: float, range: float,
    max_neighbors: int = 12, max_distance: float = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def universal_kriging(
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], targets: npt.NDArray[np.float64],
    kind: str, nugget: float, psill: float, range: float, drift: str = "linear",
    max_neighbors: int = 16, max_distance: float = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def trend_surface(
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], targets: npt.NDArray[np.float64],
    drift: str = "linear",
) -> npt.NDArray[np.float64]: ...

class CornerPointGrid:
    """Corner-point grid (Eclipse layout). z grows downward."""
    def __init__(
        self, dims: tuple[int, int, int], coord: npt.NDArray[np.float64], zcorn: npt.NDArray[np.float64],
        actnum: npt.NDArray[np.int32] | None = None,
    ) -> None: ...
    @staticmethod
    def from_surfaces(
        xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64], surfaces: npt.NDArray[np.float64],
    ) -> CornerPointGrid: ...
    @staticmethod
    def from_regular(
        origin: tuple[float, float, float], cell: tuple[float, float, float], dims: tuple[int, int, int],
    ) -> CornerPointGrid: ...
    @staticmethod
    def from_grdecl(text: str) -> tuple[CornerPointGrid, dict[str, npt.NDArray[np.float64]]]: ...
    def to_grdecl(self, props: dict[str, list[float]] | None = None) -> str: ...
    @property
    def dims(self) -> tuple[int, int, int]: ...
    @property
    def coord(self) -> npt.NDArray[np.float64]: ...
    @property
    def zcorn(self) -> npt.NDArray[np.float64]: ...
    @property
    def actnum(self) -> npt.NDArray[np.int32]: ...
    def centers(self) -> npt.NDArray[np.float64]: ...
    def volumes(self) -> npt.NDArray[np.float64]: ...
    def locate(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]: ...
    def upscale_continuous(
        self, points: npt.NDArray[np.float64], values: npt.NDArray[np.float64],
        method: str = "arithmetic", min_count: int = 1,
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.float64], npt.NDArray[np.uint64]]: ...
    def upscale_facies(
        self, points: npt.NDArray[np.float64], codes: npt.NDArray[np.int32], min_count: int = 1,
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.uint64]]: ...

# --- stochastic simulation and ml_propagation ---
def sgs(
    points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], targets: npt.NDArray[np.float64],
    kind: str, nugget: float, psill: float, range: float, n_realizations: int = 10,
    max_neighbors: int = 16, seed: int = 0, max_distance: float = ...,
) -> npt.NDArray[np.float64]: ...
def sis(
    points: npt.NDArray[np.float64], codes: npt.NDArray[np.int32], targets: npt.NDArray[np.float64],
    kind: str, nugget: float, psill: float, range: float, n_realizations: int = 10,
    max_neighbors: int = 16, seed: int = 0, max_distance: float = ...,
) -> npt.NDArray[np.int32]: ...
def normal_score(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def ml_propagation(
    train_x: npt.NDArray[np.float64], train_y: npt.NDArray[np.float64], pred_x: npt.NDArray[np.float64],
    model: str = "qrf", quantiles: list[float] = ..., n_trees: int = 200, min_leaf: int = 10, seed: int = 0,
) -> npt.NDArray[np.float64]: ...

class FixedPressure:
    def __init__(self, cell: int, trans: float, pressure: float, pressure_unit: str = "Pa") -> None: ...
    @property
    def cell(self) -> int: ...
    @property
    def trans(self) -> float: ...
    @property
    def pressure(self) -> float: ...

class FlowModel:
    @staticmethod
    def from_regular(
        origin: tuple[float, float, float] | list[float],
        cell: tuple[float, float, float] | list[float],
        dims: tuple[int, int, int] | list[int],
        perm: npt.NDArray[np.float64],
        poro: npt.NDArray[np.float64],
    ) -> FlowModel: ...
    @staticmethod
    def from_corner_point(
        grid: CornerPointGrid,
        perm: npt.NDArray[np.float64],
        poro: npt.NDArray[np.float64],
        ntg: npt.NDArray[np.float64] | None = None,
        mult: dict[str, npt.NDArray[np.float64] | list[float]] | None = None,
        fault_tol_m: float = 1e-3,
    ) -> FlowModel: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def n_fault_connections(self) -> int: ...
    @property
    def n_pinch_connections(self) -> int: ...
    @property
    def n_cells(self) -> int: ...
    def pore_volume(self) -> npt.NDArray[np.float64]: ...
    def connections(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
    def boundary_trans(self, cell: int, axis: int) -> float: ...
    def boundary_geom(self, cell: int, axis: int) -> float: ...
    def run_heat(
        self,
        mobility: npt.NDArray[np.float64],
        wells: list[dict[str, Any]],
        well_temp: list[float],
        t0: npt.NDArray[np.float64],
        times: list[float],
        fixed: list[FixedPressure | dict[str, Any]] | None = None,
        fixed_temp: list[float] | None = None,
        fixed_t: list[dict[str, float]] | None = None,
        props: dict[str, float] | None = None,
        dt_max: float = 86400.0,
        inj_mobility: float | None = None,
        dt_init: float | None = None,
        dt_change: float | None = None,
        heat_flux: list[dict[str, float]] | None = None,
    ) -> dict[str, Any]: ...
    def export_ogs(
        self,
        path: str | os.PathLike[str],
        cell_data: dict[str, npt.NDArray[np.float64]] | None = None,
        material_ids: list[int] | None = None,
        merge_tol: float = 1e-6,
    ) -> dict[str, int]: ...
    def solve_pressure(
        self,
        mobility: npt.NDArray[np.float64],
        wells: list[dict[str, Any]],
        fixed: list[FixedPressure | dict[str, Any]] | None = None,
        inj_mobility: float | None = None,
        boundary_mobility: float | None = None,
    ) -> dict[str, Any]: ...
    def run_tracer(
        self,
        mobility: npt.NDArray[np.float64],
        wells: list[dict[str, Any]],
        well_conc: list[float],
        c0: npt.NDArray[np.float64],
        times: list[float],
        fixed: list[FixedPressure | dict[str, Any]] | None = None,
        fixed_conc: list[float] | None = None,
        cfl: float = 0.9,
    ) -> dict[str, Any]: ...
    def run_impes(
        self,
        wells: list[dict[str, Any]],
        sw0: npt.NDArray[np.float64],
        times: list[float],
        fixed: list[FixedPressure | dict[str, Any]] | None = None,
        corey: dict[str, float] | None = None,
        cfl: float = 0.5,
        pressure_dt: float = ...,
        boundary_fw: float = 1.0,
    ) -> dict[str, Any]: ...
    def centers(self) -> npt.NDArray[np.float64]: ...
    def active(self) -> npt.NDArray[np.bool_]: ...
    def trajectory_completions(
        self, path: npt.NDArray[np.float64], radius: float = 0.1, skin: float = 0.0
    ) -> dict[str, Any]: ...
    def run_two_phase(
        self,
        wells: list[dict[str, Any]],
        sw0: npt.NDArray[np.float64],
        p0: npt.NDArray[np.float64],
        times: list[float],
        fixed: list[FixedPressure | dict[str, Any]] | None = None,
        corey: dict[str, float] | None = None,
        physics: dict[str, Any] | None = None,
        cfl: float = 0.5,
        max_dt: float = ...,
        boundary_fw: float = 1.0,
        ds_target: float = ...,
        dp_target: float = ...,
    ) -> dict[str, Any]: ...

def peaceman_wi(
    kx: float, ky: float, dx: float, dy: float, dz: float, rw: float, skin: float = 0.0
) -> float: ...
def buckley_leverett(xd: float, td: float, corey: dict[str, float] | None = None) -> float: ...
def capillary_pressure(
    sw: float, corey: dict[str, float] | None = None, physics: dict[str, Any] | None = None
) -> float: ...
def water_properties(
    p_pa: npt.NDArray[np.float64], t_k: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]: ...
def theis_drawdown(
    q: float, mu: float, k: float, h: float, phi: float, ct: float, r: float, t: float
) -> float: ...
