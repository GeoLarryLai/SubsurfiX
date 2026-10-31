#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Fetch the open example data (not stored in this repository):
#   kgs/T34S_R11W/  Kansas Geological Survey (KGS) LAS files and formation tops
#                   for township T34S R11W, Barber County, Kansas.
#   "The source of this material is the Kansas Geological Survey website at
#    http://www.kgs.ku.edu/. All Rights Reserved."  Terms of use:
#    https://www.kgs.ku.edu/General/copyright.html (free use, redistribution
#    and modification with this citation).
# Usage: scripts/fetch_open_data.sh [dir]   then: export SX_OPEN_DATA=<dir>
set -e
root=${1:-data_public}; d="$root/kgs"; t="$d/T34S_R11W"; mkdir -p "$t"
curl -sL -o "$d/ks_las_files.zip" https://www.kgs.ku.edu/PRS/Ora_Archive/ks_las_files.zip
curl -sL -o "$d/ks_tops.zip" https://www.kgs.ku.edu/PRS/Ora_Archive/ks_tops.zip
python3 - "$d" "$t" <<'PY'
import csv, io, sys, zipfile
d, t = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(f"{d}/ks_las_files.zip")
rows = [r for r in csv.DictReader(io.TextIOWrapper(z.open("ks_las_files.txt"), encoding="latin-1"))
        if r["Location"].startswith("T34S R11W,")]
with open(f"{t}/index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(rows)
with open(f"{t}/urls.txt", "w") as f:
    f.write("\n".join(r["URL"].replace("https://www.kgs.ku.edu/b_1/", "https://kgsimages.blob.core.windows.net/web/web_1/") for r in rows) + "\n")
ids = {r["KGS_ID"] for r in rows}
z = zipfile.ZipFile(f"{d}/ks_tops.zip")
rd = csv.DictReader(io.TextIOWrapper(z.open("ks_tops.txt"), encoding="latin-1"))
with open(f"{t}/tops.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rd.fieldnames, quoting=csv.QUOTE_ALL); w.writeheader()
    w.writerows(r for r in rd if r["KID"] in ids)
print(len(rows), "LAS files listed")
PY
(cd "$t" && xargs -P8 -n1 curl -sLO -m 120 < urls.txt)
echo "export SX_OPEN_DATA=$root"
