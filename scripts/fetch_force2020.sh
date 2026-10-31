#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Fetch the public FORCE 2020 hidden test wells (NLOD 2.0 logs, CC-BY-4.0 labels).
# Not stored in the repository. Usage: scripts/fetch_force2020.sh [dir]
set -e
d=${1:-data_public/force2020}; mkdir -p "$d"
curl -sL -o "$d/hidden_test.csv" https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/hidden_test.csv
echo "c75e817bfefcabd2bacdefd9e1c74dbd829d24543646bf0b85ef8fbf5c06f069  $d/hidden_test.csv" | sha256sum -c -
echo "export SX_FORCE2020_CSV=$d/hidden_test.csv"
