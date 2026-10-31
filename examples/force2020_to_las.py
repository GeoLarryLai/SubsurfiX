# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Convert FORCE 2020 competition wells (CSV, metres) to a SubsurfiX project.

Writes one LAS per well (depth in ft), a header CSV (UTM zone 31N, KB from MD + Z_LOC) and a long tops CSV
whose tops are the first measured depth of each lithostratigraphic group label (GROUP column), skipped when the
group already starts at the first logged sample (log top, not a true top). Deviated "S" wells are treated as
vertical (no survey in the file).

Data: Bormann P., Aursand P., Dilib F., Dischington P., Manral S. (2020), 2020 FORCE Machine Learning Contest,
https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition. Logs NLOD 2.0, labels CC BY 4.0.

usage: python force2020_to_las.py FORCE_CSV OUT_DIR WELL [WELL ...]
"""
import sys
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

FT = 0.3048
CURVES = {"GR": ("GR", "gAPI"), "RHOB": ("RHOB", "g/cm3"), "NPHI": ("NPHI", "v/v"), "DTC": ("DTC", "us/ft"),
          "RDEP": ("RD", "ohm.m"), "PEF": ("PEF", "b/e"), "CALI": ("CALI", "in"), "DRHO": ("DRHO", "g/cm3"),
          "BS": ("BS", "in")}


def key(well: str) -> str:
    return "NO_" + well.replace("/", "-").replace(" ", "")


def convert(csv: Path, out: Path, wells: list[str]) -> None:
    d = pd.read_csv(csv, sep=";")
    (out / "las").mkdir(parents=True, exist_ok=True)
    heads, tops = [], []
    for w in wells:
        g = d[d.WELL == w].sort_values("DEPTH_MD")
        md = g.DEPTH_MD.to_numpy()
        las = lasio.LASFile()
        las.well["WELL"].value = w
        las.well["UWI"].value = key(w)
        las.well["STRT"].value, las.well["STOP"].value = md[0] / FT, md[-1] / FT
        las.well["STRT"].unit = las.well["STOP"].unit = "F"
        las.append_curve("DEPT", md / FT, unit="F")
        for col, (mn, unit) in CURVES.items():
            if g[col].notna().any():
                las.append_curve(mn, g[col].to_numpy(float), unit=unit)
        las.write(str(out / "las" / f"{key(w)}.las"), version=2.0)
        kb_m = float(np.nanmedian(g.DEPTH_MD + g.Z_LOC))  # MD − TVDSS ≈ KB elevation (vertical approximation)
        heads.append({"UWI": key(w), "Name": w, "X": g.X_LOC.median(), "Y": g.Y_LOC.median(), "KB": kb_m})
        first = g.groupby("GROUP", sort=False).DEPTH_MD.min()
        for name, top in first.items():
            if top > md[0] + 1.0:
                tops.append({"UWI": key(w), "Top": name.title().replace(" Gp.", ""), "MD": top})
    pd.DataFrame(heads).to_csv(out / "headers.csv", index=False)
    pd.DataFrame(tops).to_csv(out / "tops.csv", index=False)


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
