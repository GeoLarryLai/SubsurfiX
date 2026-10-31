"""3D animation of property grids (temperature, pressure, ...) with wells.

Works with SubsurfiX solver output or any per-step 3D arrays, for example
MODFLOW 6 results saved with :func:`save_frames_npz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["RegularGrid", "Well", "animate_grid", "load_frames_npz", "save_frames_npz"]


@dataclass
class RegularGrid:
    """Regular grid: ``origin`` (x, y, z top), ``spacing`` (dx, dy, dz), ``shape`` (nx, ny, nz).

    Cell order is x fastest, then y, then z (SubsurfiX ``FlowModel.from_regular``); ``z`` grows downward.
    """

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def to_pyvista(self):
        import pyvista as pv

        nx, ny, nz = self.shape
        dx, dy, dz = self.spacing
        x0, y0, z0 = self.origin
        img = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(dx, dy, dz),
            origin=(x0, y0, -z0 - nz * dz),
        )
        return img

    def to_display(self, values) -> np.ndarray:
        """Reorder SubsurfiX cell values (z down) to pyvista cell order (z up)."""
        nx, ny, nz = self.shape
        v = np.asarray(values, dtype=float).reshape(nz, ny, nx)[::-1]
        return v.ravel()


@dataclass
class Well:
    """Vertical well at (x, y) between depths ``top`` and ``bottom`` (positive down)."""

    name: str
    x: float
    y: float
    top: float
    bottom: float
    color: str = "black"
    extra: dict = field(default_factory=dict)


def save_frames_npz(
    path, grid: RegularGrid, frames, times, name: str = "temperature", units: str = ""
) -> None:
    """Save per-step cell arrays (n_steps, n_cells) plus the grid for :func:`animate_grid`."""
    np.savez_compressed(
        path,
        origin=grid.origin,
        spacing=grid.spacing,
        shape=grid.shape,
        frames=np.asarray(frames, float),
        times=np.asarray(times, float),
        name=name,
        units=units,
    )


def load_frames_npz(path):
    """Load ``(grid, frames, times, name, units)``; ``frames`` may also be (n_steps, nz, ny, nx)."""
    d = np.load(path)
    grid = RegularGrid(
        tuple(d["origin"]), tuple(d["spacing"]), tuple(int(s) for s in d["shape"])
    )
    frames = np.asarray(d["frames"], float).reshape(len(d["times"]), -1)
    return grid, frames, d["times"], str(d["name"]), str(d["units"])


def animate_grid(
    grid: RegularGrid,
    frames,
    out,
    *,
    times=None,
    wells=(),
    label="temperature",
    units="°C",
    clim=None,
    cmap="coolwarm",
    fps=6,
    width=800,
    height=560,
    rotate_deg=40.0,
    cutaway=True,
    z_exaggeration=1.0,
    title=None,
    time_label="{:.0f} yr",
    hold_last=6,
    still=None,
    still_index=-1,
    bar_fmt="%.0f",
):
    """Render ``frames`` (n_steps, n_cells) on ``grid`` to a GIF or MP4 at ``out``.

    The grid is shown with one quarter cut away (``cutaway``) so the interior is visible,
    wells as tubes, and the camera turns ``rotate_deg`` degrees over the animation.
    ``still`` writes one PNG of frame ``still_index``. ``bar_fmt`` formats the colour-bar labels.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    frames = np.asarray(frames, float).reshape(len(frames), -1)
    if frames.shape[1] != grid.n_cells:
        raise ValueError(
            f"frames have {frames.shape[1]} cells, grid has {grid.n_cells}"
        )
    times = np.arange(len(frames)) if times is None else np.asarray(times)
    clim = clim or (float(frames.min()), float(frames.max()))
    mesh = grid.to_pyvista()
    mesh.cell_data[label] = grid.to_display(frames[0])
    if z_exaggeration != 1.0:
        mesh = mesh.scale([1, 1, z_exaggeration], inplace=False)
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing

    if cutaway:
        b = mesh.bounds
        box = pv.Box(
            bounds=(
                b[0] + 0.5 * (b[1] - b[0]),
                b[1] + 1,
                b[2] + 0.5 * (b[3] - b[2]),
                b[3] + 1,
                b[4] + 0.5 * (b[5] - b[4]),
                b[5] + 1,
            )
        )  # top quadrant facing the camera
        view = mesh.clip_box(box, invert=True, crinkle=True)
    else:
        view = mesh
    zt = z_exaggeration

    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background("white")
    bar = {
        "title": f"{label} ({units})" if units else label,
        "color": "#222222",
        "n_labels": 5,
        "fmt": bar_fmt,
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.15,
        "height": 0.7,
        "width": 0.05,
        "title_font_size": 16,
        "label_font_size": 13,
    }
    actor = pl.add_mesh(
        view,
        scalars=label,
        cmap=cmap,
        clim=clim,
        show_edges=True,
        edge_color="#555555",
        line_width=0.2,
        opacity=1.0,
        scalar_bar_args=bar,
    )
    pl.add_mesh(mesh.outline(), color="#444444", line_width=1)
    for w in wells:
        top = (-w.top + 0.12 * nz * dz) * zt
        line = pv.Line((w.x, w.y, top), (w.x, w.y, -w.bottom * zt))
        pl.add_mesh(line.tube(radius=0.012 * max(nx * dx, ny * dy)), color=w.color)
        pl.add_point_labels(
            [(w.x, w.y, top)],
            [w.name],
            font_size=14,
            text_color=w.color,
            shape=None,
            show_points=False,
            always_visible=True,
        )
    if title:
        pl.add_text(title, position="upper_left", font_size=12, color="#222222")
    tlab = pl.add_text("", position="lower_left", font_size=12, color="#222222")
    pl.camera_position = "iso"
    pl.camera.elevation = -12
    pl.camera.zoom(1.05)

    out = Path(out)
    if out.suffix.lower() == ".gif":
        pl.open_gif(str(out), fps=fps, loop=0, palettesize=128)
    else:
        pl.open_movie(str(out), framerate=fps)
    steps = list(range(len(frames))) + [len(frames) - 1] * hold_last
    da = rotate_deg / max(len(steps) - 1, 1)
    for n, i in enumerate(steps):
        mesh.cell_data[label] = grid.to_display(frames[i])
        new = mesh.clip_box(box, invert=True, crinkle=True) if cutaway else mesh
        actor.mapper.dataset.copy_from(new)
        tlab.set_text("lower_left", time_label.format(times[i]))
        if n:
            pl.camera.azimuth = pl.camera.azimuth + da
        pl.write_frame()
        if still is not None and n == (
            len(steps) - 1 if still_index == -1 else still_index
        ):
            pl.screenshot(str(still))
    pl.close()
    return out
