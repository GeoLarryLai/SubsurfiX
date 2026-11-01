"""Render the README model picture: the Kansas (KGS) property model on its structural top in 3D, porosity (left)
and permeability on a log scale (right), each with its own colour bar.

    python scripts/make_porosity_image.py --out image/hero_3d_porosity_model.png

``--cmap`` (porosity, default ``copper``) and ``--perm-cmap`` (permeability, default ``cividis``) take any
matplotlib colormap name; ``--porosity-only`` draws the porosity model alone. The wells that constrain the
property model (the grid columns holding upscaled well data, ``idx`` in ``grid_upscaled.npz``) are drawn as short
sticks standing on the top surface; ``--no-wells`` leaves them out and ``--well-height`` sets their height in m.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from kansas_window import DATA, kansas_structural  # noqa: E402

from subsurfix.viz3d import LayeredGrid, _colorbar_image  # noqa: E402


VEX = 40.0  # vertical exaggeration


def well_columns(grid, data_dir=DATA):
    """(x, y) of the grid columns holding upscaled well data, i.e. the wells that constrain the model."""
    g = np.load(Path(data_dir) / "grid_upscaled.npz")
    nx, ny, _ = grid.shape
    cols = np.unique(np.asarray(g["idx"], int) % (nx * ny))
    xs, ys = np.asarray(grid.xs, float), np.asarray(grid.ys, float)
    return [(0.5 * (xs[c % nx] + xs[c % nx + 1]), 0.5 * (ys[c // nx] + ys[c // nx + 1])) for c in cols]


def trim(img, pad):
    """Crop the white border around the drawing and leave ``pad`` pixels on every side."""
    from PIL import Image, ImageChops

    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    out = Image.new("RGB", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), "white")
    out.paste(img.crop(box), (pad, pad))
    return out


def render(grid, values, cmap, clim, size, title, log=False, wells=(), well_height=12.0):
    import pyvista as pv
    from PIL import Image

    w, h = size
    mesh = grid.to_pyvista().scale([1, 1, VEX], inplace=False)
    v = grid.to_display(np.asarray(values, float))
    mesh.cell_data["v"] = np.log10(v) if log else v
    pl = pv.Plotter(off_screen=True, window_size=(w, h))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="v", cmap=cmap, clim=[np.log10(c) for c in clim] if log else clim, show_edges=True,
                edge_color="#3a2a1a", line_width=0.3, specular=0.0, show_scalar_bar=False)
    if len(wells):
        tops = np.array([[x, y, -grid.surface_depth(x, y) * VEX] for x, y in wells])
        heads = tops + [0.0, 0.0, well_height * VEX]
        for t, h in zip(tops, heads):
            pl.add_mesh(pv.Line(t, h), color="#1a1a1a", line_width=2.0)
        pl.add_points(heads, color="#1a1a1a", point_size=6.0, render_points_as_spheres=True)
    pl.view_vector((1.0, -0.35, 0.75), viewup=(0, 0, 1))
    pl.camera.zoom(1.25)
    shot = Image.fromarray(pl.screenshot(return_img=True)).convert("RGB")
    pl.close()
    shot = trim(shot, 10)  # even margins around the model
    bar = _colorbar_image({"clim": clim, "cmap": cmap, "label": title, "bar_fmt": "%.2f", "log": log},
                          280, 54).resize((int(0.5 * w), int(0.5 * w * 54 / 280)))  # title above, ticks below
    wd = max(shot.width, bar.width)
    out = Image.new("RGB", (wd, shot.height + bar.height), "white")
    out.paste(shot, ((wd - shot.width) // 2, 0))
    out.paste(bar, ((wd - bar.width) // 2, shot.height))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hero_3d_porosity_model.png")
    ap.add_argument("--cmap", default="copper", help="porosity colormap (any matplotlib name)")
    ap.add_argument("--perm-cmap", default="cividis", help="permeability colormap (any matplotlib name)")
    ap.add_argument("--clim", default="0.0,0.30", help="porosity colour limits 'lo,hi'")
    ap.add_argument("--perm-clim", default="1,1000", help="permeability colour limits in mD 'lo,hi' (log scale)")
    ap.add_argument("--size", default="900,640", help="size of each panel 'w,h'")
    ap.add_argument("--porosity-only", action="store_true")
    ap.add_argument("--no-wells", action="store_true", help="do not draw the constraining wells")
    ap.add_argument("--well-height", type=float, default=12.0, help="well stick height above the top, m")
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont

    bm = kansas_structural()
    nz = bm["dims"][2]
    grid = LayeredGrid(bm["xs"], bm["ys"], bm["top_nodes"], nz, bm["cell"][2], bm["active"])
    size = tuple(int(v) for v in a.size.split(","))
    wk = {"wells": [] if a.no_wells else well_columns(grid), "well_height": a.well_height}
    panels = [render(grid, bm["phi"], a.cmap, tuple(float(v) for v in a.clim.split(",")), size, "porosity (fraction)",
                     **wk)]
    if not a.porosity_only:
        panels.append(render(grid, bm["k_md"], a.perm_cmap, tuple(float(v) for v in a.perm_clim.split(",")), size,
                             "permeability (mD, log scale)", log=True, **wk))
    head, gap, side = 34, 30, 20
    out = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1) + 2 * side,
                            max(p.height for p in panels) + head), "white")
    x = side
    for p in panels:
        out.paste(p, (x, head))
        x += p.width + gap
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    title = "Kansas KGS property model: 11 x 11 km, 12 layers on the structural top, vertical exaggeration 40x"
    if wk["wells"]:
        title += f"; sticks: the {len(wk['wells'])} well columns that constrain it"
    d.text((side, 8), title,
           fill="#222222", font=font)
    out.save(a.out)
    print(a.out, out.size)


if __name__ == "__main__":
    main()
