#!/usr/bin/env python
"""Freeze a 1D design onto the 2D (nx, ny) design grid.

scripts/20 evaluates a frozen design ONLY at shape design_shape_2d(cfg) with
the design pixel an integer multiple of spacing_um. A 1D design (e.g. a
(500,) vector at 50 px/um = 0.02 um pixels) satisfies neither at 0.05 um —
0.02/0.05 is not an integer, and the guard leg is non-relaxable for good
reason. This script closes that gap OUTSIDE the measured driver, which stays
bit-identical:

  1. resample the binary 1D vector from its native grid onto a coarser target
     grid by exact area-weighted averaging (integer interval-overlap
     arithmetic, no float grid accumulation), threshold mean >= 0.5 -> 1
     (tie goes to solid — the rule is fixed here, not chosen per run);
  2. extrude the result uniformly along y (the 1D semantic: every tooth
     spans the full waveguide width);
  3. write the (nx, ny) .npy scripts/20 accepts, plus a provenance sidecar
     JSON (sha256 of source and output, the exact rule, and the round-trip
     disagreement count).

The resampling is a declared approximation, not a bit-exact copy: a 0.02 um
pixelization cannot land on a 0.05 um grid without moving tooth edges by up
to dst_pixel/2 = 0.025 um. The sidecar's n_src_px_changed says how many
source pixels a nearest-neighbour read-back of the output disagrees on, so a
report can quote the distortion instead of hiding it.

  # a frozen 1D design -> 0.05 um pixels, 10 um y extrusion
  python scripts/21_extrude_1d_design.py \
      --rho runs/<1d-opt-run>/design_rho.npy \
      --src-config runs/<1d-opt-run>/config.json \
      --dst-grid 20 --l-design-y 10.0 --out runs/design_rho_2d_g20.npy
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def resample_binary_1d(rho, src_grid, dst_grid):
    """Area-weighted resample of a strictly binary 1D density from src_grid
    to dst_grid pixels/um over the same physical length.

    Returns (rho_dst, mean_dst): the thresholded binary vector
    (mean >= 0.5 -> 1.0) and the raw area-weighted means. Overlaps are
    computed in integer units of 1/(src_grid*dst_grid) um — exact, no float
    grid accumulation error. Requires n_src*dst_grid % src_grid == 0 so the
    physical length is representable on both grids.
    """
    rho = np.asarray(rho)
    if rho.ndim != 1:
        raise ValueError(f"need a 1D vector, got shape {rho.shape}")
    if not np.all(np.isin(np.unique(rho), (0.0, 1.0))):
        raise ValueError("need a strictly binary design")
    n_src = rho.size
    if (n_src * dst_grid) % src_grid != 0:
        raise ValueError(
            f"{n_src} px at {src_grid}/um is {n_src / src_grid} um, not "
            f"representable as whole pixels at {dst_grid}/um")
    n_dst = (n_src * dst_grid) // src_grid
    # integer unit u = 1/(src*dst) um: src px j = [j*dst, (j+1)*dst) u,
    # dst px i = [i*src, (i+1)*src) u; dst px width = src units of u
    mean = np.empty(n_dst)
    for i in range(n_dst):
        lo, hi = i * src_grid, (i + 1) * src_grid
        j0, j1 = lo // dst_grid, (hi - 1) // dst_grid
        acc = 0
        for j in range(j0, j1 + 1):
            ov = min(hi, (j + 1) * dst_grid) - max(lo, j * dst_grid)
            acc += ov * int(rho[j])
        mean[i] = acc / src_grid
    rho_dst = (mean >= 0.5).astype(float)   # tie -> solid (fixed rule)
    return rho_dst, mean


def roundtrip_changed(rho_src, rho_dst, src_grid, dst_grid):
    """How many source pixels disagree with a nearest-centre read-back of
    the resampled design — the distortion number the sidecar reports."""
    n_src = rho_src.size
    centers = (np.arange(n_src) + 0.5) / src_grid          # um
    idx = np.minimum((centers * dst_grid).astype(int), rho_dst.size - 1)
    return int(np.sum(rho_src != rho_dst[idx]))


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--rho", required=True, metavar="NPY",
                   help="strictly binary 1D design vector (the design_rho.npy "
                        "of a 1D-path optimization run)")
    p.add_argument("--src-config", required=True, metavar="CONFIG_JSON",
                   help="config.json of the run that produced --rho; "
                        "provides design_grid_per_um and L_design (the "
                        "vector length is validated against them)")
    p.add_argument("--dst-grid", type=int, required=True, metavar="PX_PER_UM",
                   help="target design grid (e.g. 20 -> 0.05 um pixels)")
    p.add_argument("--l-design-y", type=float, required=True, metavar="UM",
                   help="y extrusion length (um); ny = L_design_y * dst_grid")
    p.add_argument("--out", required=True, metavar="NPY",
                   help="output (nx, ny) .npy; provenance sidecar lands at "
                        "<out>.provenance.json")
    args = p.parse_args()

    with open(args.src_config) as f:
        src_cfg = json.load(f)
    src_grid = int(src_cfg["design_grid_per_um"])
    l_design = float(src_cfg["L_design"])

    rho = np.load(args.rho)
    n_expect = round(l_design * src_grid)
    if rho.ndim != 1 or rho.size != n_expect:
        raise SystemExit(
            f"--rho {args.rho} has shape {rho.shape}; --src-config says "
            f"L_design={l_design} um at {src_grid} px/um = ({n_expect},)")
    if not np.all(np.isin(np.unique(rho), (0.0, 1.0))):
        raise SystemExit(f"--rho must be strictly binary, found values "
                         f"{np.unique(rho)}")
    if args.dst_grid > src_grid:
        raise SystemExit(
            f"--dst-grid {args.dst_grid} > source grid {src_grid}: "
            f"upsampling would invent sub-pixel structure the source design "
            f"never had")

    rho_dst, mean = resample_binary_1d(rho, src_grid, args.dst_grid)
    n_changed = roundtrip_changed(rho, rho_dst, src_grid, args.dst_grid)

    ny = round(args.l_design_y * args.dst_grid)
    rho_2d = np.tile(rho_dst[:, None], (1, ny))
    np.save(args.out, rho_2d)

    prov = {
        "src_rho": os.path.abspath(args.rho),
        "src_rho_sha256": _sha(args.rho),
        "src_config": os.path.abspath(args.src_config),
        "src_grid_per_um": src_grid,
        "L_design_um": l_design,
        "dst_grid_per_um": args.dst_grid,
        "L_design_y_um": args.l_design_y,
        "rule": "area-weighted mean over exact interval overlaps, "
                "threshold mean >= 0.5 -> 1 (tie -> solid), then uniform "
                "y extrusion",
        "out_shape": list(rho_2d.shape),
        "fill_src": float(rho.mean()),
        "fill_dst": float(rho_dst.mean()),
        "n_src_px_changed_roundtrip": n_changed,
        "n_src_px": int(rho.size),
        "out_sha256": _sha(args.out),
    }
    with open(args.out + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[extrude] ({rho.size},)@{src_grid}/um -> {rho_2d.shape}@"
          f"{args.dst_grid}/um  fill {prov['fill_src']:.4f} -> "
          f"{prov['fill_dst']:.4f}  roundtrip disagreement "
          f"{n_changed}/{rho.size} src px")
    print(f"[extrude] out sha256 {prov['out_sha256']}")
    print(f"[done] {args.out} + .provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
