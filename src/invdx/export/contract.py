"""Geometry contract: fingerprint what this exporter ships, and check what
comes back.

The handoff to an external solver currently goes GDS-II -> a third-party
converter -> the solver's own format. That chain is not auditable: when a
polygon comes out the far end a different shape, nothing raises, the
simulation runs, and the number is merely wrong.

Two failures of exactly that kind already happened. A trapezoidal taper was
read as a rectangle because a bounding box was measured instead of a shape --
caught only because someone happened to look. And an index contrast of 0.01
sat in a converted file describing silicon, which is a materials failure this
module does not address but which shows what silent looks like.

So: a fingerprint of every polygon shipped, and a reader for what comes back,
so the comparison is arithmetic rather than eyeballing. The quantities are
chosen to be invariant under the axis relabelling the formats disagree about
(GDS calls the in-plane pair x,y; the .ind files call it x,z) while still
being sensitive to the failures that matter:

    vertex count   a trapezoid decomposed into rectangles changes it
    area           a filled hole changes it, and nothing else will
    perimeter      a coarsened outline changes it while area barely moves
    bbox, centroid placement and mirroring

What this module does judge: whether the polygon COUNT changed, which is a
structural fact rather than a tolerance. What it does not judge well is how
much area difference is acceptable -- quantisation to a database unit moves
area a little, a filled hole moves it a lot, and where the line sits depends
on the design. DEFAULT_AREA_RTOL is a starting point, not a standard.
"""

from __future__ import annotations

import json
import math
import re


DEFAULT_AREA_RTOL = 1e-3


def _close(pts):
    """Drop a duplicated closing vertex if present.

    GDS polygons carry N vertices implicitly closed; .ind writes N+1 with the
    first repeated. Comparing counts without normalising this reports a
    difference of one on every single polygon.
    """
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-12 \
            and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        return pts[:-1]
    return pts


def polygon_fingerprint(pts):
    """Shape descriptors for one polygon, from its vertices."""
    p = _close([(float(a), float(b)) for a, b in pts])
    n = len(p)
    area2 = 0.0
    perim = 0.0
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
        perim += math.hypot(x1 - x0, y1 - y0)
    area = abs(area2) / 2.0
    if abs(area2) > 1e-18:
        cx /= 3.0 * area2
        cy /= 3.0 * area2
    else:                      # degenerate: fall back to the vertex mean
        cx = sum(x for x, _ in p) / n
        cy = sum(y for _, y in p) / n
    xs = [x for x, _ in p]
    ys = [y for _, y in p]
    return {"n_vertices": n,
            "area": area,
            "perimeter": perim,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "centroid": [cx, cy]}


def fingerprint(polys, source):
    """Fingerprint a whole layout, sorted so ordering differences do not show.

    Sorted by (bbox min x, bbox min y): the converter is under no obligation to
    preserve the exporter's polygon order, and an ordering difference is not a
    geometry difference.
    """
    fps = [polygon_fingerprint(p) for p in polys]
    # bbox corner alone collides for shapes sharing one -- e.g. a square and a
    # triangle on the same corner -- and a collision pairs the wrong polygons,
    # inventing differences that are not there.
    fps.sort(key=lambda f: (round(f["bbox"][0], 9), round(f["bbox"][1], 9),
                            round(f["area"], 9), f["n_vertices"],
                            round(f["perimeter"], 9)))
    total_area = sum(f["area"] for f in fps)
    # Layout-level descriptors. Per-polygon numbers cannot see two failures a
    # real converter produces:
    #
    #   * fracturing -- one 401-vertex polygon came back as 137/137/136, so
    #     polygon count AND total vertex count both move (401 -> 410) while
    #     area is conserved to 0.0013%. Anything keyed on those two counts
    #     false-positives on every legitimate fracture.
    #   * dropped placement -- cell references whose offsets are discarded
    #     collapse every shape onto the origin. Count, vertices and total area
    #     are all IDENTICAL; only where the shapes sit changes.
    #
    # So: the extent of the whole layout and the arrangement within it, both
    # measured relative to the layout's own corner so a converter re-origining
    # the design is not reported as damage.
    if fps:
        x0 = min(f["bbox"][0] for f in fps)
        y0 = min(f["bbox"][1] for f in fps)
        x1 = max(f["bbox"][2] for f in fps)
        y1 = max(f["bbox"][3] for f in fps)
        layout_bbox = [x0, y0, x1, y1]
        extent = [x1 - x0, y1 - y0]
        if total_area > 0:
            cx = sum(f["centroid"][0] * f["area"] for f in fps) / total_area
            cy = sum(f["centroid"][1] * f["area"] for f in fps) / total_area
        else:
            cx = cy = 0.0
        rel = sorted([round(f["centroid"][0] - x0, 9),
                      round(f["centroid"][1] - y0, 9)] for f in fps)
    else:
        layout_bbox, extent, cx, cy, rel = [0.0] * 4, [0.0, 0.0], 0.0, 0.0, []
    return {"source": source,
            "n_polygons": len(fps),
            "total_area": total_area,
            "total_vertices": sum(f["n_vertices"] for f in fps),
            "layout_bbox": layout_bbox,
            "layout_extent": extent,
            "layout_centroid_rel": [round(cx - layout_bbox[0], 9),
                                    round(cy - layout_bbox[1], 9)],
            "centroids_rel": rel,
            "polygons": fps}


def fingerprint_gds(path, layer=None):
    """Fingerprint every polygon in a GDS file (all cells)."""
    import gdstk

    lib = gdstk.read_gds(str(path))
    # top_level() + get_polygons() flattens references. Reading cell.polygons
    # only sees shapes placed literally in that cell, so a layout built from
    # references reports a fraction of its real area with every bbox sitting
    # at the origin -- and this function exists to check what came back from
    # someone else, where hierarchy is normal.
    polys = []
    for cell in lib.top_level():
        for p in cell.get_polygons():
            if layer is None or p.layer == layer:
                polys.append([(x, y) for x, y in p.points])
    fp = fingerprint(polys, f"gds:{path}")
    # Scale matters for any cross-format comparison and the file states it.
    fp["unit"] = float(lib.unit)
    fp["precision"] = float(lib.precision)
    return fp


_POLY_RE = re.compile(r"^\s*polygon\b", re.I)
_END_RE = re.compile(r"^\s*end\s+polygon\b", re.I)
_PTS_RE = re.compile(r"^\s*points\s*=", re.I)
_ENDPTS_RE = re.compile(r"^\s*end\s+points\b", re.I)
_NUM_RE = re.compile(r"^\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*$")
# startswith("count") also matched counter_foo; this wants the assignment.
_COUNT_RE = re.compile(r"^\s*count\s*=\s*(\d+)", re.I)


def parse_ind_polygons(path):
    """Vertex lists from an RSoft .ind file.

    The format is not publicly specified; this reads the block shape observed
    in real files -- `polygon N` / `count = M` / `points =` ... `end points` /
    `end polygon`, with coordinates as two whitespace-separated numbers per
    line. `count` is the number of point LINES, which includes the repeated
    closing vertex.

    Raises when a block's declared count disagrees with the lines read: a
    converter that writes a count it did not honour is exactly the kind of
    thing worth failing on rather than quietly trusting.
    """
    polys, cur, in_pts, declared = [], None, False, None
    for lineno, line in enumerate(open(path, encoding="utf-8",
                                      errors="replace"), 1):
        if _POLY_RE.match(line):
            if cur is not None:
                raise ValueError(
                    f"{path}:{lineno}: a polygon block opened while the "
                    f"previous one was never closed -- the earlier block "
                    f"would have been dropped silently")
            cur, declared = [], None
        elif cur is not None and _COUNT_RE.match(line):
            declared = int(_COUNT_RE.match(line).group(1))
        elif _PTS_RE.match(line):
            in_pts = True
        elif _ENDPTS_RE.match(line):
            in_pts = False
        elif _END_RE.match(line):
            if cur is not None:
                if declared is not None and declared != len(cur):
                    raise ValueError(
                        f"{path}:{lineno}: polygon declares count={declared} "
                        f"but {len(cur)} point lines were read")
                polys.append(cur)
            cur, in_pts, declared = None, False, None
        elif in_pts and cur is not None:
            m = _NUM_RE.match(line)
            if m:
                cur.append((float(m.group(1)), float(m.group(2))))
    if cur is not None:
        raise ValueError(f"{path}: file ends inside an unclosed polygon block")
    return polys


def fingerprint_ind(path):
    return fingerprint(parse_ind_polygons(path), f"ind:{path}")


def compare(a, b, area_rtol=DEFAULT_AREA_RTOL, perim_rtol=1e-2):
    """Pair polygons between two fingerprints and report every difference.

    Pairing is by sorted order, which is why fingerprint() sorts. If the counts
    differ the pairing is meaningless and that is reported instead -- a
    converter that split one polygon into several is the headline, not a
    per-polygon delta.
    """
    out = {"n_polygons": [a["n_polygons"], b["n_polygons"]],
           "total_area": [a["total_area"], b["total_area"]],
           "total_vertices": [a["total_vertices"], b["total_vertices"]],
           "ok": True, "findings": []}

    # Scale first: if the two sides disagree about what a unit is, every
    # number below compares apples to metres and the report is confidently
    # wrong rather than silent.
    ua, ub = a.get("unit"), b.get("unit")
    if ua and ub and abs(ua - ub) / ua > 1e-9:
        out["ok"] = False
        out["findings"].append(
            f"unit {ua:g} vs {ub:g}: the two files disagree about scale, so "
            f"every comparison below is meaningless until that is resolved")
        return out

    ta, tb = a["total_area"], b["total_area"]
    if ta == 0.0 and tb != 0.0:
        out["ok"] = False
        out["findings"].append(
            f"total area 0 -> {tb:.6g}: the source fingerprint has no area at "
            f"all, so every relative check below is undefined")
    area_moved = ta > 0 and abs(tb - ta) / ta > area_rtol
    split = a["n_polygons"] != b["n_polygons"]

    # Area is checked even when the counts differ. A converter that fills a
    # hole AND splits the shape moves both, and reporting only the split
    # discards the one descriptor that sees the hole -- the failure this
    # module was written for.
    if area_moved:
        out["ok"] = False
        out["findings"].append(
            f"total area {ta:.6g} -> {tb:.6g} "
            f"({100 * (tb - ta) / ta:+.3f}%): candidates include a filled "
            f"hole, a dropped shape, or a scale mismatch -- this is a "
            f"hypothesis, not a diagnosis")
    # Layout extent and arrangement: translation-invariant, and the only
    # descriptors that see placement being dropped.
    ea, eb = a.get("layout_extent"), b.get("layout_extent")
    if ea and eb:
        for k, axis in enumerate("xy"):
            if ea[k] > 0 and abs(eb[k] - ea[k]) / ea[k] > area_rtol:
                out["ok"] = False
                out["findings"].append(
                    f"layout {axis} extent {ea[k]:.6g} -> {eb[k]:.6g} "
                    f"({100 * (eb[k] - ea[k]) / ea[k]:+.3f}%): the shapes no "
                    f"longer occupy the same span. A converter that discards "
                    f"cell-reference offsets collapses everything onto the "
                    f"origin and leaves area, counts and vertices untouched")
    ca, cb = a.get("centroids_rel"), b.get("centroids_rel")
    if ca and cb and len(ca) == len(cb):
        moved = sum(1 for p, q in zip(ca, cb)
                    if abs(p[0] - q[0]) > 1e-6 or abs(p[1] - q[1]) > 1e-6)
        if moved:
            out["ok"] = False
            out["findings"].append(
                f"{moved} of {len(ca)} polygons sit somewhere else relative to "
                f"the layout corner: same shapes, different arrangement")

    if split:
        # NOT a failure by itself. A converter with a vertex ceiling fractures
        # legitimately -- measured: 401 vertices came back as 137/137/136,
        # area conserved to 0.0013%. Reporting that as damage would make this
        # tool cry wolf on every 2D free-form design, and a guard that cries
        # wolf gets switched off.
        if not area_moved:
            out["findings"].append(
                f"polygon count {a['n_polygons']} -> {b['n_polygons']} and "
                f"vertices {a['total_vertices']} -> {b['total_vertices']}, "
                f"while area and layout held: consistent with the converter's "
                f"vertex ceiling re-fracturing the same geometry. Not flagged "
                f"as damage. Per-polygon pairing is skipped -- it would "
                f"compare unrelated shapes.")
            return out
        out["ok"] = False
        out["findings"].append(
            f"polygon count {a['n_polygons']} -> {b['n_polygons']}: the "
            f"shapes were split or merged. "
            + ("Total area moved too, so the geometry changed, not just its "
               "partitioning." if area_moved else
               "Total area held, so this is most likely a re-fracturing of "
               "the same geometry.")
            + " Per-polygon pairing is skipped: it would compare unrelated "
              "shapes.")
        return out

    for i, (pa, pb) in enumerate(zip(a["polygons"], b["polygons"])):
        if pa["n_vertices"] != pb["n_vertices"]:
            out["ok"] = False
            out["findings"].append(
                f"polygon {i}: {pa['n_vertices']} -> {pb['n_vertices']} "
                f"vertices")
        if pa["area"] == 0.0 and pb["area"] > 0.0:
            # `pa["area"] > 0` as the only entry condition let a degenerate
            # source -- collinear vertices, a zero-area sliver -- skip the
            # comparison entirely, so a real shape on the other side raised
            # nothing. Reproduced: collinear vs a unit square gave ok=True
            # with an empty findings list while total_area read [0.0, 1.0].
            out["ok"] = False
            out["findings"].append(
                f"polygon {i}: source has zero area (degenerate -- collinear "
                f"or a sliver) but the other side has {pb['area']:.6g}; the "
                f"per-polygon area check cannot run on a zero denominator")
        elif pa["area"] > 0 and abs(pb["area"] - pa["area"]) / pa["area"] \
                > area_rtol:
            out["ok"] = False
            out["findings"].append(
                f"polygon {i}: area {pa['area']:.6g} -> {pb['area']:.6g} "
                f"({100 * (pb['area'] - pa['area']) / pa['area']:+.3f}%)")
        # Perimeter was computed and never compared. It is the descriptor
        # that sees an outline coarsened by a converter -- area barely moves,
        # perimeter does -- and it is invariant under the translation,
        # mirroring and axis relabelling the two formats disagree about, so
        # comparing it costs no false positives.
        if pa["perimeter"] > 0 and abs(pb["perimeter"] - pa["perimeter"]) \
                / pa["perimeter"] > perim_rtol:
            out["ok"] = False
            out["findings"].append(
                f"polygon {i}: perimeter {pa['perimeter']:.6g} -> "
                f"{pb['perimeter']:.6g} "
                f"({100 * (pb['perimeter'] - pa['perimeter']) / pa['perimeter']:+.2f}%)"
                f" while area held -- an outline that was simplified")
    return out


def read_fingerprint(path):
    """Counterpart to write_fingerprint, so a stored contract can be checked."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_fingerprint(path, fp):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=1)
