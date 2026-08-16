"""GDS export round-trip on a synthetic profile."""

import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

from invdx.export.gds import export_profile_gds


def test_export_round_trip(tmp_path):
    # 3 teeth of 200 nm separated by 200 nm gaps, at 100 px/um
    rho = np.zeros(1000)
    for start in (200, 240, 280):
        rho[start:start + 20] = 1
    out = tmp_path / "t.gds"

    res = export_profile_gds(rho, grid_per_um=100, width_um=10.0,
                             min_feature_um=0.130, out=str(out),
                             taper_length_um=5.0)
    assert res["n_teeth"] == 3
    assert res["rule_ok"]  # 200 nm features > 130 nm rule
    assert res["min_solid_um"] == pytest.approx(0.20, abs=1e-9)

    lib = gdstk.read_gds(str(out))
    (cell,) = lib.cells
    # 3 teeth + taper + waveguide = 5 polygons
    assert len(cell.polygons) == 5


def test_rule_violation_reported(tmp_path):
    rho = np.zeros(1000)
    rho[100:120] = 1
    rho[125:200] = 1          # 50 nm void gap violates the 130 nm rule
    res = export_profile_gds(rho, grid_per_um=100, width_um=10.0,
                             min_feature_um=0.130,
                             out=str(tmp_path / "v.gds"), taper_length_um=0.0)
    assert not res["rule_ok"]
