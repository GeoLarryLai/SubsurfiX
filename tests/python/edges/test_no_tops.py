# SPDX-License-Identifier: AGPL-3.0-or-later
"""Without any tops, every well in the area is selected."""
import pyarrow as pa

from subsurfix.edges.area import select_wells


def test_no_tops_selects_all_wells():
    h = pa.table({"well_id": [1, 2], "well_key": ["A", "B"], "x_m": [0.0, 1.0], "y_m": [0.0, 1.0]})
    sel = select_wells(h, None, None)
    assert sel.column("selected").to_pylist() == [True, True]
    tops = pa.table({"well_id": [1]})
    assert select_wells(h, None, tops).column("selected").to_pylist() == [True, False]
