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


def test_taper_is_a_taper_and_meets_the_waveguide(tmp_path):
    """The taper's shape, not just its presence in the polygon count.

    Reading this polygon's BOUNDING BOX gives the grating width at both
    ends, which looks like an abrupt wide-aperture -> narrow-waveguide
    junction. It is not one: the polygon is a trapezoid that is exactly the
    waveguide's width where the two meet.

    The reason that reading survived is here: every other polygon this
    exporter writes is a rectangle, so bounding box equals shape for all of
    them, and the two existing tests assert only how MANY polygons come out.
    The taper is the single polygon whose bbox differs from its shape, and
    nothing had ever asserted one of its vertices.
    """
    rho = np.zeros(400)
    rho[100:120] = 1
    L, W, wg_w = 300.0, 10.0, 0.45
    out = tmp_path / "taper.gds"
    export_profile_gds(rho, grid_per_um=100, width_um=W, min_feature_um=0.02,
                       out=str(out), taper_length_um=L, wg_width_um=wg_w,
                       wg_length_um=20.0)

    (cell,) = gdstk.read_gds(str(out)).cells
    taper = [p for p in cell.polygons
             if p.points[:, 0].min() == pytest.approx(-L, abs=1e-9)
             and p.points[:, 0].max() == pytest.approx(0.0, abs=1e-9)]
    assert len(taper) == 1, "expected exactly one polygon spanning the taper"
    pts = taper[0].points
    assert len(pts) == 4

    def width_at(poly_pts, x):
        ys = sorted(y for (px, y) in poly_pts if abs(px - x) < 1e-9)
        assert len(ys) == 2, f"expected two vertices at x={x}, got {len(ys)}"
        return ys[1] - ys[0]

    # the junction condition: a taper that does not match the waveguide it
    # joins is exactly the abrupt interface a bbox reading suggests
    assert width_at(pts, -L) == pytest.approx(wg_w, abs=1e-9)
    assert width_at(pts, 0.0) == pytest.approx(W, abs=1e-9)

    # and the access waveguide is the same width where it arrives
    wg = [p for p in cell.polygons
          if p.points[:, 0].max() == pytest.approx(-L, abs=1e-9)]
    assert len(wg) == 1
    assert width_at(wg[0].points, -L) == pytest.approx(wg_w, abs=1e-9)

    # stated explicitly: the bounding box does NOT describe this polygon, so
    # a bbox-based check would pass while the taper degenerated to a slab
    bbox_w = pts[:, 1].max() - pts[:, 1].min()
    assert bbox_w == pytest.approx(W, abs=1e-9)
    assert width_at(pts, -L) != pytest.approx(bbox_w, abs=1e-9)


def _simple_rho():
    """One 200 nm tooth at 100 px/um — passes the 130 nm rule."""
    rho = np.zeros(400)
    rho[100:120] = 1
    return rho


def _violating_rho():
    """50 nm void gap — violates the 130 nm rule."""
    rho = np.zeros(1000)
    rho[100:120] = 1
    rho[125:200] = 1
    return rho


def test_taper_half_angle_closed_form(tmp_path):
    """The reported half angle must equal the closed-form geometry
    atan(((W - wg_w)/2) / L) — asserted against an independent computation,
    not against the function's own output."""
    import math

    L, W, wg_w = 300.0, 10.0, 0.45
    res = export_profile_gds(_simple_rho(), grid_per_um=100, width_um=W,
                             min_feature_um=0.130,
                             out=str(tmp_path / "a.gds"),
                             taper_length_um=L, wg_width_um=wg_w)
    expected_deg = math.degrees(math.atan(((W - wg_w) / 2) / L))
    assert res["taper_half_angle_deg"] == pytest.approx(expected_deg,
                                                        rel=1e-12)
    # sanity anchor on the actual number, so a wrong closed form above
    # cannot silently agree with an equally wrong implementation
    assert res["taper_half_angle_deg"] == pytest.approx(0.9119, abs=5e-4)
    # no criterion given -> no judgement
    assert res["taper_adiabatic_margin"] is None

    # with a criterion, margin = criterion / half_angle
    res2 = export_profile_gds(_simple_rho(), grid_per_um=100, width_um=W,
                              min_feature_um=0.130,
                              out=str(tmp_path / "b.gds"),
                              taper_length_um=L, wg_width_um=wg_w,
                              adiabatic_half_angle_deg=1.26)
    assert res2["taper_adiabatic_margin"] == pytest.approx(
        1.26 / expected_deg, rel=1e-12)


def test_taper_half_angle_none_when_no_taper(tmp_path):
    res = export_profile_gds(_simple_rho(), grid_per_um=100, width_um=10.0,
                             min_feature_um=0.130,
                             out=str(tmp_path / "c.gds"),
                             taper_length_um=0.0)
    assert res["taper_half_angle_deg"] is None
    assert res["taper_adiabatic_margin"] is None


def _run_gds_cli(design_npy, out_gds, *extra):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "invdx.export.gds",
         "--design", str(design_npy), "--out", str(out_gds), *extra],
        capture_output=True, text=True, timeout=300)


def test_cli_rule_violation_exits_2(tmp_path):
    """The CLI must not print FAIL and exit 0: an unenforced rule is no
    rule. Violation -> exit code 2 (file still written for inspection)."""
    design = tmp_path / "v.npy"
    np.save(design, _violating_rho())
    out = tmp_path / "v.gds"
    proc = _run_gds_cli(design, out)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout


def test_cli_allow_rule_violation_exits_0(tmp_path):
    design = tmp_path / "v.npy"
    np.save(design, _violating_rho())
    out = tmp_path / "v.gds"
    proc = _run_gds_cli(design, out, "--allow-rule-violation")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout  # still reported, just not fatal
    assert out.exists()


def test_cli_clean_design_exits_0(tmp_path):
    """Control: without the flag, a rule-clean design still exits 0 — the
    gate only fires on violations."""
    design = tmp_path / "ok.npy"
    np.save(design, _simple_rho())
    out = tmp_path / "ok.gds"
    proc = _run_gds_cli(design, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
    assert out.exists()
