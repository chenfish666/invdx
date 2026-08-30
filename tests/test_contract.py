"""Geometry contract: the fingerprint must move when the geometry does.

Written failure-first. The point of this module is to make a silent handoff
corruption loud, so the tests that matter are the ones asserting it FAILS --
a fingerprint that always matches is indistinguishable from no check at all,
and that is the exact shape of error this whole layer exists to catch.
"""

import math

import pytest

gdstk = pytest.importorskip("gdstk")

from invdx.export import contract as C


TRAPEZOID = [(-300.0, -0.225), (0.0, -5.0), (0.0, 5.0), (-300.0, 0.225)]


def _write_ind(path, polys):
    """Minimal .ind with the block shape observed in real converter output."""
    with open(path, "w") as f:
        f.write("background_index = 1\n\n")
        for i, pts in enumerate(polys, 1):
            closed = list(pts) + [pts[0]]          # .ind repeats the first
            f.write(f"polygon {i}\n\tcount = {len(closed)}\n\tpoints =\n")
            for x, y in closed:
                f.write(f"\t\t{x} {y}\n")
            f.write("\tend points\nend polygon\n\n")


def test_closing_vertex_is_normalised(tmp_path):
    """.ind repeats the first vertex, GDS does not; counts must still agree."""
    ind = tmp_path / "t.ind"
    _write_ind(ind, [TRAPEZOID])
    fp = C.fingerprint_ind(ind)
    assert fp["polygons"][0]["n_vertices"] == 4
    assert fp["polygons"][0]["area"] == pytest.approx(
        C.polygon_fingerprint(TRAPEZOID)["area"])


def test_round_trip_of_a_trapezoid_matches(tmp_path):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("T")
    cell.add(gdstk.Polygon(TRAPEZOID, layer=1))
    gds = tmp_path / "t.gds"
    lib.write_gds(str(gds))
    ind = tmp_path / "t.ind"
    _write_ind(ind, [TRAPEZOID])

    res = C.compare(C.fingerprint_gds(gds), C.fingerprint_ind(ind))
    assert res["ok"], res["findings"]


def test_a_filled_hole_is_caught(tmp_path):
    """The failure the vendor's own contour documentation admits to.

    A hole silently filled changes area and nothing else obvious: the outline
    still looks right, the simulation still runs, the efficiency is merely
    wrong. Area is the one descriptor that moves.
    """
    outer = gdstk.rectangle((0, 0), (10, 10))
    holed = gdstk.boolean(outer, gdstk.rectangle((3, 3), (7, 7)), "not")
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("H")
    for p in holed:
        cell.add(p)
    gds = tmp_path / "h.gds"
    lib.write_gds(str(gds))

    ind = tmp_path / "h.ind"                       # hole dropped
    _write_ind(ind, [[(0, 0), (10, 0), (10, 10), (0, 10)]])

    res = C.compare(C.fingerprint_gds(gds), C.fingerprint_ind(ind))
    assert not res["ok"]
    assert any("area" in f for f in res["findings"])


def test_a_decomposed_polygon_is_caught(tmp_path):
    """A trapezoid split into rectangles keeps the area and loses the shape.

    Total area can be preserved exactly by a decomposition, so area alone will
    not catch it -- the polygon count does, and the report says the pairing is
    meaningless rather than printing per-polygon deltas against the wrong
    partner.
    """
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("D")
    cell.add(gdstk.Polygon(TRAPEZOID, layer=1))
    gds = tmp_path / "d.gds"
    lib.write_gds(str(gds))

    # two strips whose areas sum to the trapezoid's
    a = C.polygon_fingerprint(TRAPEZOID)["area"]
    ind = tmp_path / "d.ind"
    _write_ind(ind, [[(-300, -0.225), (-150, -0.225), (-150, 0.225),
                      (-300, 0.225)],
                     [(-150, -a / 300 + 0.225), (0, -5), (0, 5),
                      (-150, 0.225)]])

    res = C.compare(C.fingerprint_gds(gds), C.fingerprint_ind(ind))
    assert not res["ok"]
    assert any("count" in f for f in res["findings"])


def test_declared_count_must_match_the_lines_read(tmp_path):
    """A converter that writes a count it did not honour should not be trusted."""
    ind = tmp_path / "bad.ind"
    ind.write_text("polygon 1\n\tcount = 9\n\tpoints =\n"
                   "\t\t0 0\n\t\t1 0\n\t\t1 1\n\t\t0 0\n"
                   "\tend points\nend polygon\n")
    with pytest.raises(ValueError, match="count=9"):
        C.parse_ind_polygons(ind)


def test_area_is_invariant_under_the_axis_relabelling(tmp_path):
    """GDS calls the in-plane pair x,y; .ind calls it x,z. Area must not care."""
    swapped = [(y, x) for x, y in TRAPEZOID]
    assert C.polygon_fingerprint(swapped)["area"] == pytest.approx(
        C.polygon_fingerprint(TRAPEZOID)["area"])
    assert C.polygon_fingerprint(swapped)["perimeter"] == pytest.approx(
        C.polygon_fingerprint(TRAPEZOID)["perimeter"])


def test_taper_area_matches_the_closed_form():
    """An independent value, so the shoelace implementation is not self-checked."""
    fp = C.polygon_fingerprint(TRAPEZOID)
    # trapezoid: mean of the two parallel widths times the separation
    assert fp["area"] == pytest.approx((0.45 + 10.0) / 2 * 300.0, rel=1e-12)
    # Mass sits toward the WIDE end, not the narrow one: the centroid is at
    # -104.3 while the midpoint is -150. An earlier version of this comment
    # had it backwards, and `< -100` could not tell the two apart -- -299
    # would have passed it too.
    assert -150.0 < fp["centroid"][0] < 0.0
    assert fp["centroid"][1] == pytest.approx(0.0, abs=1e-12)
    assert not math.isnan(fp["perimeter"])


def _sq(x0, y0, w=10.0):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + w), (x0, y0 + w)]


def test_dropped_cell_placement_is_caught(tmp_path):
    """The failure a real converter produces and every other descriptor misses.

    Discarding cell-reference offsets collapses every shape onto the origin.
    Total area, polygon count and total vertex count are all IDENTICAL --
    measured on the real converter, not hypothesised. Only where the shapes
    sit changes, so only a layout-level descriptor can see it.
    """
    shipped = C.fingerprint([_sq(0, 0), _sq(20, 0), _sq(40, 0)], "gds")
    got = C.fingerprint([_sq(0, 0), _sq(0, 0), _sq(0, 0)], "ind")

    assert shipped["total_area"] == pytest.approx(got["total_area"])
    assert shipped["n_polygons"] == got["n_polygons"]
    assert shipped["total_vertices"] == got["total_vertices"]

    res = C.compare(shipped, got)
    assert not res["ok"]
    assert any("extent" in f for f in res["findings"])


def test_legitimate_fracturing_is_not_called_damage():
    """A vertex ceiling splits one polygon into several. Area is conserved;
    polygon count and total vertex count are not -- measured on the real
    converter, 401 vertices came back as 137/137/136, i.e. 410. Keying the
    verdict on either count would cry wolf on every 2D free-form design, and
    a guard that cries wolf gets switched off.
    """
    whole = C.fingerprint([[(0, 0), (10, 0), (10, 10), (0, 10)]], "gds")
    split = C.fingerprint([[(0, 0), (5, 0), (5, 10), (0, 10)],
                           [(5, 0), (10, 0), (10, 10), (5, 10)]], "ind")
    assert whole["total_vertices"] != split["total_vertices"]
    res = C.compare(whole, split)
    assert res["ok"], res["findings"]
    assert any("vertex ceiling" in f for f in res["findings"])


def test_layout_descriptors_survive_a_global_translation():
    """A converter re-origining the design is not damage, and must not report
    as damage -- otherwise the two checks above become false-positive engines.
    """
    a = C.fingerprint([_sq(0, 0), _sq(20, 0)], "gds")
    b = C.fingerprint([_sq(1000, -500), _sq(1020, -500)], "ind")
    res = C.compare(a, b)
    assert res["ok"], res["findings"]
