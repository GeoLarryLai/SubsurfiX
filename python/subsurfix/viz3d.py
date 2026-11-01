"""3D animation of property grids (temperature, pressure, ...) with wells.

Works with SubsurfiX solver output or any per-step 3D arrays, for example
MODFLOW 6 results saved with :func:`save_frames_npz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _register_colormaps():
    """Two extra matplotlib colormaps used as defaults here (any other matplotlib colormap works too).

    ``porosity_grey``: light, translucent-friendly greys, darker for low values (tight rock), near white for high.
    ``cold``: dark blue at the low end to a clear light blue at the high end, so any cooling reads as cold.
    ``viridis_bright``: viridis without its dark purple end, so small values do not look grey.
    """
    try:
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:  # pragma: no cover
        return
    vir = matplotlib.colormaps["viridis"]
    new = {"porosity_grey": LinearSegmentedColormap.from_list("porosity_grey", ["#8c8c8c", "#f7f7f7"]),
           "cold": LinearSegmentedColormap.from_list("cold", ["#08306b", "#2171b5", "#4292c6", "#9ecae1"]),
           "viridis_bright": LinearSegmentedColormap.from_list("viridis_bright", vir(np.linspace(0.3, 1.0, 64)))}
    for name, cm in new.items():
        if name not in matplotlib.colormaps:
            matplotlib.colormaps.register(cm, name=name)


_register_colormaps()

__all__ = ["LayeredGrid", "RegularGrid", "Well", "animate_grid", "animate_grids", "animate_layered", "load_frames_npz", "save_frames_npz"]


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
class LayeredGrid:
    """Grid whose layers follow a structural top: node coordinates ``xs`` (nx + 1), ``ys`` (ny + 1), node top
    depths ``top`` ((ny + 1, nx + 1), positive down), ``nz`` layers of ``dz`` m, and an optional per-cell
    ``active`` mask (natural order, i fastest, then j, then k from the top). Inactive cells are not drawn.

    Cell values passed to the animation functions are for every cell (``n_cells``); only active ones are shown.
    """

    xs: np.ndarray
    ys: np.ndarray
    top: np.ndarray
    nz: int
    dz: float
    active: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.xs) - 1, len(self.ys) - 1, int(self.nz))

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (float(np.mean(np.diff(self.xs))), float(np.mean(np.diff(self.ys))), float(self.dz))

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def _ids(self) -> np.ndarray:
        if self.active is None:
            return np.arange(self.n_cells)
        return np.flatnonzero(np.asarray(self.active, bool))

    def to_pyvista(self):
        import pyvista as pv

        nx, ny, nz = self.shape
        X, Y = np.meshgrid(np.asarray(self.xs, float), np.asarray(self.ys, float))
        top = np.asarray(self.top, float).reshape(ny + 1, nx + 1)
        z = np.stack([-(top + self.dz * k) for k in range(nz + 1)])  # k from the top, z up
        sg = pv.StructuredGrid(np.broadcast_to(X, z.shape).T, np.broadcast_to(Y, z.shape).T, z.T)
        return sg.extract_cells(self._ids())

    def to_display(self, values) -> np.ndarray:
        """Active-cell values in the order of :meth:`to_pyvista`."""
        return np.asarray(values, dtype=float).ravel()[self._ids()]

    def surface_depth(self, x: float, y: float) -> float:
        """Top depth at (x, y), bilinear between nodes."""
        xs, ys = np.asarray(self.xs, float), np.asarray(self.ys, float)
        top = np.asarray(self.top, float).reshape(len(ys), len(xs))
        i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
        j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
        u, v = (x - xs[i]) / (xs[i + 1] - xs[i]), (y - ys[j]) / (ys[j + 1] - ys[j])
        return float((1 - u) * (1 - v) * top[j, i] + u * (1 - v) * top[j, i + 1] + (1 - u) * v * top[j + 1, i]
                     + u * v * top[j + 1, i + 1])


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
    zoom=1.05,
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
        view = mesh.copy()
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
        top = -w.top * zt + max(0.12 * nz * dz * zt, 0.08 * max(nx * dx, ny * dy))
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
    pl.camera.zoom(zoom)

    out = Path(out)
    if out.suffix.lower() == ".gif":
        pl.open_gif(str(out), fps=fps, loop=0, palettesize=128)
    else:
        pl.open_movie(str(out), framerate=fps)
    steps = list(range(len(frames))) + [len(frames) - 1] * hold_last
    da = rotate_deg / max(len(steps) - 1, 1)
    for n, i in enumerate(steps):
        mesh.cell_data[label] = grid.to_display(frames[i])
        actor.mapper.SetInputData(mesh.clip_box(box, invert=True, crinkle=True) if cutaway else mesh.copy())
        if hasattr(tlab, "set_text"):
            tlab.set_text("lower_left", time_label.format(times[i]))
        else:
            tlab.input = time_label.format(times[i])
        if n:
            pl.camera.azimuth = pl.camera.azimuth + da
        pl.write_frame()
        if still is not None and n == (
            len(steps) - 1 if still_index == -1 else still_index
        ):
            pl.screenshot(str(still))
    pl.close()
    return out


def animate_grids(
    grid: RegularGrid,
    panels,
    out,
    *,
    times,
    wells=(),
    label="saturation",
    units="",
    clim=(0.0, 1.0),
    cmap="viridis",
    shape=None,
    fps=5,
    panel_size=(420, 420),
    rotate_deg=30.0,
    cutaway=True,
    z_exaggeration=1.0,
    title=None,
    time_label="{:.0f} yr",
    hold_last=5,
    bar_fmt="%.1f",
    still=None,
    zoom=1.15,
    colors=128,
):
    """Side-by-side animation of several runs on the same ``grid``.

    ``panels`` is a list of ``(panel_title, frames)`` or ``(panel_title, frames, options)`` with ``frames``
    (n_steps, n_cells), all at the same ``times``. ``options`` may set ``label``, ``units``, ``clim``, ``cmap``,
    ``bar_fmt`` and ``bar`` (draw a colour bar in this panel) per panel; otherwise the shared values are used and
    one colour bar is drawn in the last panel. ``shape`` is (rows, columns), default one row. Every panel uses the
    same camera (linked and turned together), wells and cutaway, with one shared time stamp, so the panels can be
    compared frame by frame. ``colors`` is the GIF palette size (fewer colours give a smaller file).
    ``grid`` may be a :class:`RegularGrid` or a :class:`LayeredGrid` (structural top, inactive cells hidden);
    see :func:`animate_layered` for a figure with header, footer and plume highlighting.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    times = np.asarray(times, float)
    data = []
    for c, item in enumerate(panels):
        name, frames = item[0], item[1]
        o = dict(item[2]) if len(item) > 2 else {}
        f = np.asarray(frames, float).reshape(len(frames), -1)
        if f.shape != (len(times), grid.n_cells):
            raise ValueError(f"panel {name!r}: frames {f.shape}, expected ({len(times)}, {grid.n_cells})")
        o.setdefault("label", label)
        o.setdefault("units", units)
        o.setdefault("clim", clim)
        o.setdefault("cmap", cmap)
        o.setdefault("bar_fmt", bar_fmt)
        o.setdefault("bar", c == len(panels) - 1)
        data.append((name, f, o))
    k = len(data)
    rows, cols = shape if shape is not None else (1, k)
    if rows * cols < k:
        raise ValueError(f"shape {shape} has fewer than {k} panels")
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    zt = z_exaggeration
    base = grid.to_pyvista()
    if zt != 1.0:
        base = base.scale([1, 1, zt], inplace=False)
    b = base.bounds
    box = pv.Box(bounds=(b[0] + 0.5 * (b[1] - b[0]), b[1] + 1, b[2] + 0.5 * (b[3] - b[2]), b[3] + 1,
                         b[4] + 0.5 * (b[5] - b[4]), b[5] + 1))
    pl = pv.Plotter(off_screen=True, shape=(rows, cols), window_size=(panel_size[0] * cols, panel_size[1] * rows),
                    border=False)
    pl.set_background("white")
    meshes, actors = [], []
    for c, (name, f, o) in enumerate(data):
        pl.subplot(c // cols, c % cols)
        mesh = base.copy()
        sname = f"{o['label']} [{c}]"
        mesh.cell_data[sname] = grid.to_display(f[0])
        view = mesh.clip_box(box, invert=True, crinkle=True) if cutaway else mesh.copy()
        bar = {"title": f"{o['label']} ({o['units']})" if o["units"] else o["label"], "color": "#222222",
               "n_labels": 5, "fmt": o["bar_fmt"], "vertical": True, "position_x": 0.84, "position_y": 0.2,
               "height": 0.6, "width": 0.06, "title_font_size": 14, "label_font_size": 12}
        actors.append(pl.add_mesh(view, scalars=sname, cmap=o["cmap"], clim=o["clim"], show_edges=True,
                                  edge_color="#555555", line_width=0.2, show_scalar_bar=bool(o["bar"]),
                                  scalar_bar_args=bar))
        meshes.append((mesh, sname))
        if isinstance(grid, RegularGrid):
            pl.add_mesh(mesh.outline(), color="#444444", line_width=1)
        for w in wells:
            top = -w.top * zt + max(0.12 * nz * dz * zt, 0.08 * max(nx * dx, ny * dy))
            line = pv.Line((w.x, w.y, top), (w.x, w.y, -w.bottom * zt))
            pl.add_mesh(line.tube(radius=0.012 * max(nx * dx, ny * dy)), color=w.color)
            pl.add_point_labels([(w.x, w.y, top)], [w.name], font_size=12, text_color=w.color, shape=None,
                                show_points=False, always_visible=True)
        pl.add_text(name, position="upper_edge", font_size=12, color="#222222")
        pl.camera_position = "iso"
        pl.camera.elevation = -12
        pl.camera.zoom(zoom)
    pl.link_views()
    pl.subplot(rows - 1, cols - 1)
    if title:
        pl.add_text(title, position="lower_right", font_size=9, color="#444444")
    pl.subplot(rows - 1, 0)
    tlab = pl.add_text("", position="lower_left", font_size=13, color="#222222")
    out = Path(out)
    if out.suffix.lower() == ".gif":
        pl.open_gif(str(out), fps=fps, loop=0, palettesize=colors)
    else:
        pl.open_movie(str(out), framerate=fps)
    steps = list(range(len(times))) + [len(times) - 1] * hold_last
    da = rotate_deg / max(len(steps) - 1, 1)
    for n, i in enumerate(steps):
        for (_, f, _o), (mesh, sname), actor in zip(data, meshes, actors):
            mesh.cell_data[sname] = grid.to_display(f[i])
            actor.mapper.SetInputData(mesh.clip_box(box, invert=True, crinkle=True) if cutaway else mesh.copy())
        if hasattr(tlab, "set_text"):
            tlab.set_text("lower_left", time_label.format(times[i]))
        else:
            tlab.input = time_label.format(times[i])
        if n:
            pl.camera.azimuth = pl.camera.azimuth + da
        pl.write_frame()
        if still is not None and n == len(steps) - 1:
            pl.screenshot(str(still))
    pl.close()
    return out


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


ALPHA_LEVELS = (0.55, 0.7, 0.85, 1.0)  # opacity classes for changed cells in 3D
ALPHA_EDGES = (0.62, 0.77, 0.92)


def _change_alpha(d, thr, dmax, relative=False):
    """Opacity for changed cells: 0.55 just above the threshold, 1 at ``dmax`` and beyond."""
    dmax = max(float(dmax or 0.0), (1.0 + 1e-9) * thr + 1e-30 if relative else 2.0 * thr, 1e-30)
    return 0.55 + 0.45 * np.clip((np.asarray(d, float) - thr) / (dmax - thr), 0.0, 1.0)


def _colorbar_image(o, w, h, vertical=False):
    """Matplotlib colour bar for a panel: ticks below (or right), title above, nothing overlapping."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    cax = fig.add_axes((0.06, 0.38, 0.88, 0.22))
    if o.get("log"):  # log-scale bar, ticks at whole decades
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import FuncFormatter

        sm = ScalarMappable(norm=LogNorm(*o["clim"]), cmap=o["cmap"])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        dec = 10.0 ** np.arange(np.ceil(np.log10(o["clim"][0])), np.floor(np.log10(o["clim"][1])) + 1)
        cb.set_ticks(dec)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cb.ax.minorticks_off()
    else:
        sm = ScalarMappable(norm=Normalize(*o["clim"]), cmap=o["cmap"])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", format=o.get("bar_fmt", "%.2f"))
        cb.set_ticks(np.linspace(*o["clim"], 5))
    cb.ax.tick_params(labelsize=7, pad=1, length=2)
    title = f"{o['label']} ({o['units']})" if o.get("units") else o["label"]
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=8, color="#222222")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h))


def _section_image(sec, v, v0, o, w, h, background=None, bg_clim=None, dmax=None, bg_cmap="porosity_grey"):
    """Matplotlib picture of a vertical cross-section (see ``section`` in :func:`animate_layered`)."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from PIL import Image

    dist = np.asarray(sec["distance"], float) / 1000.0
    half = 0.5 * (dist[1] - dist[0]) if len(dist) > 1 else 0.25
    dz = float(sec["dz"])
    polys, vals, chg, bg = [], [], [], []
    for d, top, cells in zip(dist, sec["tops"], sec["columns"]):
        for k, c in enumerate(cells):
            z0 = top + k * dz
            polys.append([(d - half, z0), (d + half, z0), (d + half, z0 + dz), (d - half, z0 + dz)])
            vals.append(v[c])
            chg.append(abs(v[c] - v0[c]))
            bg.append(np.nan if background is None else background[c])
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0.11, 0.2, 0.74, 0.6))
    cax = fig.add_axes((0.875, 0.2, 0.025, 0.6))
    chg = np.asarray(chg)
    if o["plume"] is None:
        pc = PolyCollection(polys, array=np.asarray(vals), cmap=o["cmap"], edgecolors="#888888", linewidths=0.2)
        pc.set_clim(*o["clim"])
        ax.add_collection(pc)
    else:  # faint grey porosity underneath, change drawn in colour with opacity rising with its size
        if background is not None:
            gb = PolyCollection(polys, array=np.asarray(bg), cmap=bg_cmap, edgecolors="#bbbbbb", linewidths=0.2,
                                alpha=0.85)
            gb.set_clim(*(bg_clim or (np.nanmin(bg), np.nanmax(bg))))
            ax.add_collection(gb)
        thr = o.get("_thr", o["plume"])
        sel = np.flatnonzero(chg > thr)
        if sel.size:  # a little more translucent than in 3D, with cell outlines, so the porosity shows through
            pc = PolyCollection([polys[q] for q in sel], array=np.asarray(vals)[sel], cmap=o["cmap"],
                                edgecolors="#9a9a9a", linewidths=0.25)
            pc.set_clim(*o["clim"])
            pc.set_alpha(o.get("section_alpha", 0.6)
                         * _change_alpha(chg[sel], thr, dmax, o.get("plume_above_median") is not None))
            ax.add_collection(pc)
    for d, z0, z1, color, name in sec.get("wells", []):
        ax.plot([d / 1000.0] * 2, [z0, z1], color=color, lw=4, solid_capstyle="butt")
        ax.text(d / 1000.0, z0 - 2, name, color=color, ha="center", va="bottom", fontsize=7, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5})
    ax.set_xlim(dist[0] - half, dist[-1] + half)
    zt, zb = min(sec["tops"]), max(sec["tops"]) + dz * max(len(c) for c in sec["columns"])
    ax.set_ylim(zb + 2, zt - 12)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb = fig.colorbar(ScalarMappable(norm=Normalize(*o["clim"]), cmap=o["cmap"]), cax=cax, format=o.get("bar_fmt", "%.2f"))
    cb.set_ticks(np.linspace(*o["clim"], 3))
    cb.ax.tick_params(labelsize=6, pad=1, length=2)
    ax.tick_params(labelsize=7)
    ends = sec.get("ends")
    if ends:
        for x, name, ha in ((0.01, ends[0], "left"), (0.99, ends[1], "right")):
            ax.text(x, 0.04, name, transform=ax.transAxes, ha=ha, va="bottom", fontsize=8, fontweight="bold",
                    color="#333333")
        ax.set_xlabel(f"distance along the well line, {ends[0]} to {ends[1]} (km)", fontsize=7, labelpad=1)
    else:
        ax.set_xlabel("distance along the injector-producer line (km)", fontsize=7, labelpad=1)
    ax.set_ylabel("depth (m)", fontsize=7)
    ax.set_title(sec.get("title", "cross-section"), fontsize=8, pad=4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h))


def animate_layered(
    grid,
    panels,
    out,
    *,
    times,
    wells=(),
    label="saturation",
    units="",
    clim=(0.0, 1.0),
    cmap="viridis",
    shape=None,
    fps=4,
    panel_size=(460, 380),
    rotate_deg=20.0,
    z_exaggeration=40.0,
    view_dir=(1.0, -0.35, 0.75),
    zoom=1.2,
    header=None,
    footer=None,
    time_label="{:.0f} years",
    hold_last=5,
    bar_fmt="%.2f",
    still=None,
    colors=256,
    plume=None,
    context_opacity=0.25,
    well_radius=None,
    edge_color="#5a5a5a",
    axes=True,
    section=None,
    background=None,
    background_clim=None,
    background_label="porosity",
    background_cmap="porosity_grey",
    background_log=False,
):
    """Side-by-side 3D animation on a :class:`LayeredGrid` (or :class:`RegularGrid`) in a titled figure.

    Like :func:`animate_grids` (same ``panels`` format and per-panel options), drawn the way a geomodel is
    usually shown: real cell geometry, inactive cells hidden, vertical exaggeration ``z_exaggeration``, wells as
    vertical sticks above the top surface, an x/y/z orientation marker (``axes``), and a ``header`` line above and
    ``footer`` line(s) below the panels with the time stamp at the top right. ``view_dir`` is the direction from
    the model towards the camera; the camera turns ``rotate_deg`` degrees over the animation.

    ``plume`` (a number) highlights change: cells whose value differs from the first frame by more than ``plume``
    are drawn opaque and only the outer surface of the rest of the model is drawn, at ``context_opacity``, so a plume or front inside the model is
    visible. Without ``plume`` the model is opaque. ``still`` writes a PNG of the last frame. The changed cells are
    drawn with opacity rising with the size of the change (0.55 at the threshold, 1 at the panel option
    ``change_max``, by default the largest change in the run). Change is measured from the panel option ``ref``
    (a value or one value per cell, e.g. the initial state), by default from the first frame. The panel option
    ``plume_above_median`` (a margin) sets the threshold in each frame to the median change over the model plus
    the margin, so only change above the model-wide background (e.g. near-well pressure build-up) is coloured;
    ``section_alpha`` (default 0.6) scales the overlay opacity in the cross-section.
    ``plume_frac`` sets the threshold as a fraction of the largest change in the run instead of ``plume``.

    ``background`` (one value per cell, e.g. porosity) colours the faint shell in greyscale instead of plain grey,
    and is drawn as grey cells under the change overlay in the cross-sections. ``background_cmap`` sets its colormap
    (default ``"porosity_grey"``: light greys, darker = lower value, i.e. tighter rock). Every colormap here (``cmap``, the per-panel ``cmap`` option, ``background_cmap``) accepts
    any matplotlib colormap name or ``Colormap`` object; continuous maps give smooth gradients.
    ``background_log`` shades the background by log10 of its values (e.g. permeability; ``background_clim`` in
    real units); a shared colour bar titled ``background_label`` (grey base) is drawn under the footer.

    ``section`` adds a vertical cross-section under every panel: a dict with ``columns`` (a list of per-column
    cell-id lists, top to bottom, along a line), ``distance`` (m along the line, one per column), ``tops`` (top
    depth of each column, m), ``dz`` (layer thickness), optional ``wells`` ([(distance, z_open_top,
    z_open_bottom, color, name)]), ``title`` and ``height`` (pixels, default 190).
    """
    import pyvista as pv
    from PIL import Image, ImageDraw

    pv.OFF_SCREEN = True
    times = np.asarray(times, float)
    data = []
    for c, item in enumerate(panels):
        name, frames = item[0], item[1]
        o = dict(item[2]) if len(item) > 2 else {}
        f = np.asarray(frames, float).reshape(len(frames), -1)
        if f.shape != (len(times), grid.n_cells):
            raise ValueError(f"panel {name!r}: frames {f.shape}, expected ({len(times)}, {grid.n_cells})")
        for k, v in (("label", label), ("units", units), ("clim", clim), ("cmap", cmap), ("bar_fmt", bar_fmt),
                     ("bar", c == len(panels) - 1), ("plume", plume)):
            o.setdefault(k, v)
        data.append((name, f, o))
    k = len(data)
    rows, cols = shape if shape is not None else (1, k)
    if rows * cols < k:
        raise ValueError(f"shape {shape} has fewer than {k} panels")
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    zt = z_exaggeration
    base = grid.to_pyvista().scale([1, 1, zt], inplace=False)
    width = max(nx * dx, ny * dy)
    pl = pv.Plotter(off_screen=True, shape=(rows, cols), window_size=(panel_size[0] * cols, panel_size[1] * rows),
                    border=False)
    pl.set_background("white")
    bg_disp = bg_clim = bg_raw = None
    if background is not None:
        bg = np.asarray(background, float).reshape(-1)
        bar_clim = tuple(background_clim) if background_clim else (float(np.nanmin(bg)), float(np.nanmax(bg)))
        if background_log:  # e.g. permeability: shade by log10, label the bar in the real units
            bg = np.log10(np.clip(bg, 1e-30, None))
            bg_clim = tuple(float(np.log10(v)) for v in bar_clim)
        else:
            bg_clim = bar_clim
        bg_disp, bg_raw = grid.to_display(bg), bg
    state = []
    for c, (name, f, o) in enumerate(data):
        pl.subplot(c // cols, c % cols)
        mesh = base.copy()
        sname = f"{o['label']} [{c}]"
        mesh.cell_data[sname] = grid.to_display(f[0])
        if bg_disp is not None:
            mesh.cell_data["_bg"] = bg_disp
        o["_ref"] = (f[0] if o.get("ref") is None
                     else np.broadcast_to(np.asarray(o["ref"], float), f[0].shape).astype(float))
        if o["plume"] is not None and o.get("change_max") is None:
            o["change_max"] = float(np.nanmax(np.abs(f - o["_ref"])))
        if o.get("plume_frac") is not None:  # threshold as a fraction of the largest change in the run
            o["plume"] = float(o["plume_frac"]) * float(np.nanmax(np.abs(f - o["_ref"])))
            o["change_max"] = o.get("change_max") or float(np.nanmax(np.abs(f - o["_ref"])))
        bar = {"title": f"{o['label']} ({o['units']})" if o["units"] else o["label"], "color": "#222222",
               "n_labels": 5, "fmt": o["bar_fmt"], "vertical": False, "position_x": 0.2, "position_y": 0.03,
               "height": 0.07, "width": 0.6, "title_font_size": 13, "label_font_size": 11}
        style = {"scalars": sname, "cmap": o["cmap"], "clim": o["clim"], "show_edges": True, "edge_color": edge_color,
                 "line_width": 0.3}
        a_pl = None
        if o["plume"] is None:
            a_ctx = pl.add_mesh(mesh.copy(), show_scalar_bar=False, **style)
        else:  # faint grey shell and outline of the model, opaque coloured plume cells inside
            shell = mesh.extract_surface(algorithm="dataset_surface")
            if bg_disp is None:
                pl.add_mesh(shell, color="#c8c8c8", opacity=context_opacity, show_edges=False)
            else:
                pl.add_mesh(shell, scalars="_bg", cmap=background_cmap, clim=bg_clim, opacity=context_opacity,
                            show_edges=False, show_scalar_bar=False)
            pl.add_mesh(shell.extract_feature_edges(feature_angle=40, boundary_edges=True, non_manifold_edges=False,
                                                    manifold_edges=False), color="#777777", line_width=1.0)
            a_ctx = None
            a_pl = []  # changed cells in opacity classes, one actor per class
            for q, op in enumerate(ALPHA_LEVELS):
                a_pl.append(pl.add_mesh(mesh.copy(), show_scalar_bar=False,
                                        specular=0.0, ambient=0.25, diffuse=0.8, opacity=op, **style))
        state.append((mesh, sname, a_ctx, a_pl))
        for w in wells:
            top = -w.top * zt + max(0.15 * nz * dz * zt, 0.08 * width)
            line = pv.Line((w.x, w.y, top), (w.x, w.y, -w.bottom * zt))
            rad = well_radius or 0.01 * width
            if "open" in w.extra:  # thin casing, thick bright open (perforated) interval
                pl.add_mesh(line.tube(radius=0.45 * rad), color="#333333")
                z0, z1 = w.extra["open"]
                seg = pv.Line((w.x, w.y, -z0 * zt), (w.x, w.y, -z1 * zt))
                pl.add_mesh(seg.tube(radius=1.6 * rad), color=w.color)
            else:
                pl.add_mesh(line.tube(radius=rad), color=w.color)
            pl.add_point_labels([(w.x, w.y, top)], [w.name], font_size=13, text_color=w.color, shape=None,
                                show_points=False, always_visible=True, bold=True)
        pl.add_text(name, position="upper_edge", font_size=12, color="#222222")
        pl.view_vector(view_dir, viewup=(0, 0, 1))
        pl.camera.zoom(zoom)
        if axes and c == 0:
            pl.add_axes(line_width=2, color="#222222", viewport=(0.0, 0.0, 0.16, 0.22))
    pl.link_views()
    # no depth peeling: with the offscreen renderer it drops translucent scalar-coloured cells

    W, H3 = panel_size[0] * cols, panel_size[1] * rows
    sec_h = int(section.get("height", 190)) if section else 0
    H = H3 + sec_h * rows
    n_foot = len(footer.splitlines()) if footer else 0
    top_band, foot_band = (34 if header else 26), 8 + 17 * n_foot
    f_head, f_time, f_foot = _font(17), _font(17), _font(12)

    bar_imgs = [_colorbar_image(o, int(0.62 * panel_size[0]), 54) for _, _, o in data]
    bg_bar = None
    if bg_clim is not None:  # one shared colour bar for the grey base, in its own strip under the footer
        bg_bar = _colorbar_image({"clim": bar_clim, "cmap": background_cmap, "label": f"{background_label} (grey base)",
                                  "bar_fmt": "%.2f", "log": background_log}, min(360, W - 20), 54)
        foot_band += bg_bar.height + 6

    def compose(img, tstr):
        canvas = Image.new("RGB", (W, H + top_band + foot_band), "white")
        shot = Image.fromarray(img).convert("RGB").resize((W, H3))
        if not section:
            canvas.paste(shot, (0, top_band))
        else:
            for r in range(rows):
                canvas.paste(shot.crop((0, r * panel_size[1], W, (r + 1) * panel_size[1])),
                             (0, top_band + r * (panel_size[1] + sec_h)))
            for c, sim in enumerate(sec_imgs):
                canvas.paste(sim, ((c % cols) * panel_size[0], top_band + (c // cols) * (panel_size[1] + sec_h)
                                   + panel_size[1]))
        for c, bim in enumerate(bar_imgs):  # one matplotlib colour bar per 3D panel, title clear of the ticks
            canvas.paste(bim, ((c % cols) * panel_size[0] + (panel_size[0] - bim.width) // 2,
                               top_band + (c // cols) * (panel_size[1] + sec_h) + panel_size[1] - bim.height - 2))
        d = ImageDraw.Draw(canvas)
        if header:
            d.text((10, 8), header, fill="#111111", font=f_head)
        tw = d.textlength(tstr, font=f_time)
        d.text((W - tw - 12, 8), tstr, fill="#111111", font=f_time)
        for n, line in enumerate(footer.splitlines() if footer else []):
            d.text((10, top_band + H + 4 + 17 * n), line, fill="#333333", font=f_foot)
        if bg_bar is not None:
            canvas.paste(bg_bar, ((W - bg_bar.width) // 2, canvas.height - bg_bar.height - 4))
        return canvas

    steps = list(range(len(times))) + [len(times) - 1] * hold_last
    da = rotate_deg / max(len(steps) - 1, 1)
    images = []
    for n, i in enumerate(steps):
        for (_, f, o), (mesh, sname, a_ctx, a_pl) in zip(data, state):
            mesh.cell_data[sname] = grid.to_display(f[i])
            if a_pl is None:
                a_ctx.mapper.SetInputData(mesh.copy())
            else:
                chg = np.abs(grid.to_display(f[i]) - grid.to_display(o["_ref"]))
                o["_thr"] = o["plume"]
                if o.get("plume_above_median") is not None:  # only cells rising above this frame's model-wide rise
                    o["_thr"] = float(np.nanmedian(np.abs(f[i] - o["_ref"]))) + float(o["plume_above_median"])
                sel = np.flatnonzero(chg > o["_thr"])
                alpha = _change_alpha(chg[sel], o["_thr"], o["change_max"], o.get("plume_above_median") is not None)
                lev = np.searchsorted(ALPHA_EDGES, alpha)
                for q, act in enumerate(a_pl):
                    sq = sel[lev == q]
                    act.SetVisibility(bool(sq.size))
                    if sq.size:  # extract_cells adds vtkOriginalCellIds; colour by the field itself
                        part = mesh.extract_cells(sq)
                        part.set_active_scalars(sname, preference="cell")
                        act.mapper.SetInputData(part)
                        act.mapper.SetScalarModeToUseCellFieldData()
                        act.mapper.SelectColorArray(sname)
                        act.mapper.SetScalarRange(*o["clim"])
        if n:
            pl.camera.azimuth = pl.camera.azimuth + da
        pl.render()
        sec_imgs = ([_section_image(section, f[i], o["_ref"], o, panel_size[0], sec_h, bg_raw, bg_clim,
                                    o.get("change_max"), background_cmap) for _, f, o in data] if section else [])
        images.append(compose(pl.screenshot(return_img=True), time_label.format(times[i])))
    pl.close()
    out = Path(out)
    pick = sorted({0, len(images) // 3, 2 * len(images) // 3, len(images) - 1})  # palette from several frames
    strip = Image.new("RGB", (images[0].width, images[0].height * len(pick)))
    for n, q0 in enumerate(pick):
        strip.paste(images[q0], (0, images[0].height * n))
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
    if still is not None:
        images[-1].save(str(still))
    return out
