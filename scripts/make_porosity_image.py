"""Render the README model picture: the Kansas (KGS) property model on its structural top in 3D, porosity (left)
and permeability on a log scale (right), each with its own colour bar.

    python scripts/make_porosity_image.py --out image/hero_3d_porosity_model.png

``--cmap`` (porosity, default ``copper``) and ``--perm-cmap`` (permeability, default ``cividis``) take any
matplotlib colormap name; ``--porosity-only`` draws the porosity model alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from kansas_window import kansas_structural  # noqa: E402

from subsurfix.viz3d import LayeredGrid, _colorbar_image  # noqa: E402


def render(grid, values, cmap, clim, size, title, log=False):
    import pyvista as pv
    from PIL import Image

    w, h = size
    mesh = grid.to_pyvista().scale([1, 1, 40.0], inplace=False)
    v = grid.to_display(np.asarray(values, float))
    mesh.cell_data["v"] = np.log10(v) if log else v
    pl = pv.Plotter(off_screen=True, window_size=(w, h))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="v", cmap=cmap, clim=[np.log10(c) for c in clim] if log else clim, show_edges=True,
                edge_color="#3a2a1a", line_width=0.3, specular=0.0, show_scalar_bar=False)
    pl.view_vector((1.0, -0.35, 0.75), viewup=(0, 0, 1))
    pl.camera.zoom(1.25)
    shot = Image.fromarray(pl.screenshot(return_img=True)).convert("RGB")
    pl.close()
    shot = shot.crop((0, 0, w, h - 60))  # trim the empty strip under the model
    bar = _colorbar_image({"clim": clim, "cmap": cmap, "label": title, "bar_fmt": "%.2f", "log": log},
                          280, 54).resize((int(0.5 * w), int(0.5 * w * 54 / 280)))  # title above, ticks below
    out = Image.new("RGB", (w, shot.height + bar.height), "white")
    out.paste(shot, (0, 0))
    out.paste(bar, ((w - bar.width) // 2, shot.height))
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
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont

    bm = kansas_structural()
    nz = bm["dims"][2]
    grid = LayeredGrid(bm["xs"], bm["ys"], bm["top_nodes"], nz, bm["cell"][2], bm["active"])
    size = tuple(int(v) for v in a.size.split(","))
    panels = [render(grid, bm["phi"], a.cmap, tuple(float(v) for v in a.clim.split(",")), size, "porosity (fraction)")]
    if not a.porosity_only:
        panels.append(render(grid, bm["k_md"], a.perm_cmap, tuple(float(v) for v in a.perm_clim.split(",")), size,
                             "permeability (mD, log scale)", log=True))
    head = 34
    out = Image.new("RGB", (sum(p.width for p in panels), panels[0].height + head), "white")
    for n, p in enumerate(panels):
        out.paste(p, (n * size[0], head))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    d.text((12, 8), "Kansas KGS property model: 11 x 11 km, 12 layers on the structural top, vertical exaggeration 40x",
           fill="#222222", font=font)
    out.save(a.out)
    print(a.out, out.size)


if __name__ == "__main__":
    main()
