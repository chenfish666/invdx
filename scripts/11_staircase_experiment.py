#!/usr/bin/env python
"""Staircase experiment: measure what binary voxel fill does to a
continuously swept geometry, using both of this toolbox's engines.

Sweep the PhC rod radius r continuously and track the band-gap's upper
edge (where bulk transmission recovers through -20 dB). Two engines, same
nominal geometry:

  toy  — binary rasterization (same discretization class as released
         fdtdx's explicit geometry): the edge should move in STAIRS,
         jumping only when the rod boundary crosses a grid cell
  Meep — subpixel smoothing: the edge should move SMOOTHLY

This turns a claim that can otherwise only be argued from reading source
("binary fill breaks continuous cross-verification") into a measured figure,
and is the justification for the engine division of labor: grid-snapped
geometry -> fdtdx; continuous sweeps -> a subpixel-smoothing solver.

CPU-only (toy + meep -np 4).  ~15 min detached.

  python scripts/11_staircase_experiment.py --tag staircase
"""

import os

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import phc_bend
from invdx import runio


def gap_upper_edge(freqs, T_db, thresh=-20.0):
    """Highest-frequency -20 dB crossing (gap upper edge), interpolated."""
    f = np.asarray(freqs)
    t = np.asarray(T_db)
    idx = np.where(t < thresh)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[-1]
    if i + 1 >= len(f):
        return float(f[-1])
    # linear interp between last in-gap sample and the next one
    f0, f1, t0, t1 = f[i], f[i + 1], t[i], t[i + 1]
    return float(f0 + (thresh - t0) / (t1 - t0) * (f1 - f0))


def main():
    p = base_parser(__doc__)
    p.add_argument("--r-lo", type=float, default=0.210)
    p.add_argument("--r-hi", type=float, default=0.240)
    p.add_argument("--n-r", type=int, default=13)
    args = p.parse_args()
    cfg = apply_overrides(
        phc_bend.PhCBendConfig(n_side=11, res_per_a=10, toy_steps=3000,
                               n_freq=41, f_min=0.30, f_max=0.50),
        args)
    d = start_run(cfg, args, "staircase")

    radii = np.linspace(args.r_lo, args.r_hi, args.n_r)
    # toy grid cell = 1/res_per_a = 0.1a: radius quantum visible over a
    # 0.03a sweep. Meep at the same 10/a resolution but with subpixel
    # smoothing should glide.
    rows = []
    for r in radii:
        cfg.r_rod = float(r)
        bulk = phc_bend.toy_bulk_transmission(cfg)
        Tdb = 10 * np.log10(np.abs(bulk["T"]) + 1e-12)
        edge_toy = gap_upper_edge(bulk["freqs"], Tdb)

        mres = phc_bend.meep_bend_transmission(cfg, n_ranks=4)
        Tk = 10 * np.log10(np.abs(np.asarray(mres["T_bulk"])) + 1e-12)
        edge_meep = gap_upper_edge(mres["freqs"], Tk)

        rows.append({"r_rod": float(r), "edge_toy": edge_toy,
                     "edge_meep": edge_meep})
        print(f"[stair] r={r:.4f}a  edge(toy binary)={edge_toy:.4f}  "
              f"edge(meep subpixel)={edge_meep:.4f}")

    et = np.array([x["edge_toy"] for x in rows])
    em = np.array([x["edge_meep"] for x in rows])
    # staircase metric: how many sweep steps produce ZERO edge movement
    flat_toy = int(np.sum(np.abs(np.diff(et)) < 1e-4))
    flat_meep = int(np.sum(np.abs(np.diff(em)) < 1e-4))
    print(f"\n[stair] zero-motion steps out of {len(radii)-1}: "
          f"toy {flat_toy} vs meep {flat_meep}")
    print("[stair] expectation: toy >> meep (plateaus between cell "
          "crossings) — binary fill quantizes geometry")

    runio.save_json(os.path.join(d, "results.json"), {
        "radii_a": radii.tolist(), "rows": rows,
        "flat_steps_toy": flat_toy, "flat_steps_meep": flat_meep,
        "grid_cell_a": 1.0 / cfg.res_per_a})
    print(f"[done] {d}/results.json")


if __name__ == "__main__":
    main()
