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
def evaluate_nphi_offset(config_path: str | os.PathLike[str]) -> str: ...
def nphi_shift(config_path: str | os.PathLike[str]) -> tuple[float, str]: ...
def init_logging(level: str = "info") -> bool: ...
def register_model(log: str, model: object) -> None: ...
def clear_models(log: str | None = None) -> None: ...
def registered_models() -> dict[str, str]: ...
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
    method: str = "ml", train_xyz: npt.NDArray[np.float64] | None = None,
    pred_xyz: npt.NDArray[np.float64] | None = None, train_groups: npt.NDArray[np.int32] | None = None,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0), max_neighbors: int = 16, embed: bool = True,
    embed_holdout: str = "group", random_split: bool = True, calibrate: bool = False,
    conformal: bool | None = None,
) -> npt.NDArray[np.float64]: ...
def ember(
    train_xyz: npt.NDArray[np.float64], train_y: npt.NDArray[np.float64], pred_xyz: npt.NDArray[np.float64],
    train_groups: npt.NDArray[np.int32] | None = None, train_secondary: npt.NDArray[np.float64] | None = None,
    pred_secondary: npt.NDArray[np.float64] | None = None, model: str = "qrf", quantiles: list[float] = ...,
    n_trees: int = 200, min_leaf: int = 10, seed: int = 0,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0), use_coordinates: bool = True,
    embed_ranges: list[float] | None = None, max_neighbors: int = 16, max_per_group: int = 4,
    n_realizations: int = 0, sampling_variogram: tuple[str, float, float, float] | None = None,
    sim_neighbors: int = 16, embed: bool = True, embed_holdout: str = "group", random_split: bool = True,
    gibbs_sweeps: int = 100, calibrate: bool = False, spread: float | str | None = "inflate",
) -> dict[str, Any]: ...
def ember_cv(
    xyz: npt.NDArray[np.float64], y: npt.NDArray[np.float64], groups: npt.NDArray[np.int32],
    secondary: npt.NDArray[np.float64] | None = None, n_folds: int = 5, block: float = 0.0, model: str = "qrf",
    n_trees: int = 200, min_leaf: int = 10, seed: int = 0,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0), use_coordinates: bool = True,
    embed_ranges: list[float] | None = None, max_neighbors: int = 16, max_per_group: int = 4,
    n_realizations: int = 20, sampling_variogram: tuple[str, float, float, float] | None = None,
    sim_neighbors: int = 16, embed: bool = True, embed_holdout: str = "group", random_split: bool = True,
    gibbs_sweeps: int = 100, calibrate: bool = False, spread: float | str | None = "inflate",
    conformal: bool | None = None,
) -> list[dict[str, Any]]: ...

# --- facies ---
def facies_interpret(
    gr: npt.NDArray[np.float64], rhob: npt.NDArray[np.float64], nphi: npt.NDArray[np.float64],
    pef: npt.NDArray[np.float64], method: str = "cutoffs", cutoffs_csv: str | None = None,
    endpoints: dict[str, tuple[float, float, float]] | None = None, shale_clay: float = 0.5,
    chert_quartz: float = 0.2, nphi_shift: float = 0.0,
) -> dict[str, Any]: ...
def facies_default_cutoffs() -> str: ...
def facies_rollup_classify(
    x_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.int32], x_pred: npt.NDArray[np.float64],
    shrink: float = 0.1,
) -> npt.NDArray[np.int32]: ...
def separability(
    logs: npt.NDArray[np.float64], names: list[str], groups: npt.NDArray[np.int32],
    labels: npt.NDArray[np.int32] | None = None, classifier: str = "gaussian", n_folds: int = 5,
    seed: int = 0, min_rows_class: int = 200, max_rows: int = 5000, max_logs: int = 0,
    min_coverage: float = 0.3, target_accuracy: float = 0.7, min_recall: float = 0.5, tol: float = 0.02,
    k_max: int = 8, ps_threshold: float = 0.8, ps_margin: float = 0.05, valley_max: float = 0.7,
    ps_splits: int = 10,
) -> dict[str, Any]: ...

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
        n_threads: int | None = None,
        report_every: int = 1,
        snapshot_dir: str | None = None,
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
        n_threads: int | None = None,
    ) -> dict[str, Any]: ...
    def flow_diagnostics(
        self,
        mobility: npt.NDArray[np.float64],
        wells: list[dict[str, Any]],
        fixed: list[FixedPressure | dict[str, Any]] | None = None,
        inj_mobility: float | None = None,
        boundary_mobility: float | None = None,
        max_tof: float | None = None,
        n_threads: int | None = None,
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
        n_threads: int | None = None,
        report_every: int = 1,
        snapshot_dir: str | None = None,
    ) -> dict[str, Any]: ...

def peaceman_wi(
    kx: float, ky: float, dx: float, dy: float, dz: float, rw: float, skin: float = 0.0
) -> float: ...
def buckley_leverett(xd: float, td: float, corey: dict[str, float] | None = None) -> float: ...
def capillary_pressure(
    sw: float, corey: dict[str, float] | None = None, physics: dict[str, Any] | None = None
) -> float: ...
def set_flow_threads(n: int) -> None: ...
def f_phi(
    pore_volume: npt.NDArray[np.float64],
    tof_forward: npt.NDArray[np.float64],
    tof_backward: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def lorenz_coefficient(f: npt.NDArray[np.float64], phi: npt.NDArray[np.float64]) -> float: ...
def sweep_efficiency(
    f: npt.NDArray[np.float64], phi: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def flow_threads() -> int: ...
def water_properties(
    p_pa: npt.NDArray[np.float64], t_k: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]: ...
def co2_solubility(
    p_pa: npt.NDArray[np.float64],
    t_k: npt.NDArray[np.float64],
    m_nacl: npt.NDArray[np.float64],
    model: str = "spycher_pruess_2005",
) -> dict[str, npt.NDArray[np.float64]]: ...
def co2_brine_flash(
    p_pa: float, t_k: float, z_co2: float, m_nacl: float, model: str = "spycher_pruess_2005"
) -> dict[str, float]: ...
def theis_drawdown(
    q: float, mu: float, k: float, h: float, phi: float, ct: float, r: float, t: float
) -> float: ...
def run_fim_water_oil(
    model: FlowModel,
    wells: list[dict[str, Any]],
    sw0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
    times: list[float],
    corey: dict[str, float] | None = None,
    physics: dict[str, Any] | None = None,
    injection: list[float] | None = None,
    p_range: tuple[float, float] = ...,
    table_points: tuple[int, int] = ...,
    dt_init: float = ...,
    dt_max: float = ...,
    dt_min: float = ...,
    max_newton: int = 20,
    tol: float = 1e-6,
) -> dict[str, Any]: ...
def run_fim_black_oil(
    model: FlowModel,
    wells: list[dict[str, Any]],
    p0: npt.NDArray[np.float64],
    zg0: npt.NDArray[np.float64],
    zo0: npt.NDArray[np.float64],
    times: list[float],
    pvt: npt.NDArray[np.float64],
    rho_sc: tuple[float, float, float] = ...,
    bw: float = 1.0,
    c_w: float = ...,
    mu_w: float = ...,
    s_res: tuple[float, float, float] = ...,
    n_exp: tuple[float, float, float] = ...,
    kr_max: tuple[float, float, float] = ...,
    c_o_undersat: float = ...,
    c_rock: float = 0.0,
    p_ref: float = ...,
    gravity: float = 0.0,
    injection: list[list[float]] | None = None,
    p_range: tuple[float, float] = ...,
    table_points: tuple[int, int, int] = ...,
    z_gas_max: float = 1.0,
    dt_init: float = ...,
    dt_max: float = ...,
    dt_min: float = ...,
    max_newton: int = 20,
    tol: float = 1e-6,
) -> dict[str, Any]: ...
def fim_water_mass_fraction(p: float, sw: float, physics: dict[str, Any] | None = None) -> float: ...
def run_fim_co2_brine(
    model: FlowModel,
    wells: list[dict[str, Any]],
    p0: npt.NDArray[np.float64],
    z0: npt.NDArray[np.float64],
    times: list[float],
    temperature: float = 323.15,
    m_nacl: float = 0.0,
    solubility_model: str = "spycher_pruess_2005",
    thermal: bool = False,
    t0: npt.NDArray[np.float64] | None = None,
    injection_temperature: list[float] | None = None,
    injection: list[float] | None = None,
    s_res: tuple[float, float] = ...,
    n_exp: tuple[float, float] = ...,
    kr_max: tuple[float, float] = ...,
    c_rock: float = 0.0,
    p_ref: float = ...,
    rock: dict[str, float] | None = None,
    gravity: float = 0.0,
    p_range: tuple[float, float] = ...,
    t_range: tuple[float, float] = ...,
    table_points: list[int] | None = None,
    dt_init: float = ...,
    dt_max: float = ...,
    dt_min: float = ...,
    max_newton: int = 20,
    tol: float = 1e-6,
) -> dict[str, Any]: ...
def run_fim_thermal_water(
    model: FlowModel,
    wells: list[dict[str, Any]],
    p0: npt.NDArray[np.float64],
    t0: npt.NDArray[np.float64],
    times: list[float],
    injection_temperature: list[float] | None = None,
    rock: dict[str, float] | None = None,
    c_rock: float = 0.0,
    p_ref: float = ...,
    gravity: float = 0.0,
    p_range: tuple[float, float] = ...,
    t_range: tuple[float, float] = ...,
    table_points: tuple[int, int] = ...,
    dt_init: float = ...,
    dt_max: float = ...,
    dt_min: float = ...,
    max_newton: int = 20,
    tol: float = 1e-6,
) -> dict[str, Any]: ...

class VeGrid:
    """Top-surface (2D) grid for CO2 vertical-equilibrium runs (experimental)."""
    @staticmethod
    def from_cartesian(
        xs: list[float], ys: list[float], top: list[float], thickness: list[float], poro: list[float], perm: list[float]
    ) -> VeGrid: ...
    @staticmethod
    def from_corner_point(
        grid: CornerPointGrid, perm: npt.NDArray[np.float64], poro: npt.NDArray[np.float64]
    ) -> VeGrid: ...
    @property
    def n_cells(self) -> int: ...
    @property
    def top(self) -> npt.NDArray[np.float64]: ...
    @property
    def thickness(self) -> npt.NDArray[np.float64]: ...
    @property
    def poro(self) -> npt.NDArray[np.float64]: ...
    @property
    def area(self) -> npt.NDArray[np.float64]: ...
    @property
    def column(self) -> npt.NDArray[np.int64]: ...
    def centers(self) -> npt.NDArray[np.float64]: ...
    def trap_analysis(self, tol: float = 1e-3) -> dict[str, Any]: ...

def run_co2_ve(
    grid: VeGrid,
    wells: list[dict[str, Any]],
    times: list[float],
    s_wr: float = 0.1,
    s_gr: float = 0.2,
    p_fluid: float = 1e7,
    t_fluid: float = 313.15,
    m_nacl: float = 0.0,
    fluid: dict[str, float] | None = None,
    gravity: float = 9.80665,
    open_boundary: bool = True,
    p_ref: float | None = None,
    z_ref: float | None = None,
    cfl: float = 0.5,
    dt_max: float = 365.25 * 86400.0,
    trap_tol: float = 1e-3,
) -> dict[str, Any]: ...
def ve_fluid_properties(p: float, t: float, m_nacl: float = 0.0) -> dict[str, float]: ...
def nordbotten_celia_h(r: float, volume: float, thickness: float, phi_eff: float, mobility_ratio: float) -> float: ...
