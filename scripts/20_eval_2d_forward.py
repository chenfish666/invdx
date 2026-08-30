#!/usr/bin/env python
"""Forward-only CE of a FROZEN 2D design at a chosen grid.

Loads a strictly binary design_rho_2d.npy, rebuilds the differentiable 3D
scene of scripts/15's --design-2d path (build_scene_design_3d) around it with
with_transforms=False (prescribed-profile placement: the parameters ARE the
physical density, no filter, no projection), and runs ONE forward FDTD plus
the empty-cell P_in run. No optimizer, no gradient is ever evaluated — the
checkpointed GradientConfig the scene builder installs is inert for a
value-only call (jax.checkpoint changes nothing in pure forward mode).

The config comes from the run directory of the optimization that produced the
design (--config <run_dir>/config.json), so every geometry field is the one
the design was optimized in; --set then overrides ONLY the field under test
(spacing_um for the grid-halving control). P_in is re-measured at the same
spacing as the design run — the two runs of one CE ratio must never mix grids.

  # CPU self-test (shrunk scene, frozen self-test design):
  python scripts/20_eval_2d_forward.py \
      --config runs/<selftest>/config.json --rho runs/<selftest>/design_rho_2d.npy \
      --wg-width 2.0 --allow-t-si-snap --tag a-cpu

  # production-scale design, re-evaluated on a coarser grid:
  python scripts/20_eval_2d_forward.py \
      --config runs/<2d-opt-run>/config.json \
      --rho runs/<2d-opt-run>/design_rho_2d.npy \
      --wg-width 10.0 --allow-t-si-snap --set spacing_um=0.05 --tag eval-0p05
"""

import hashlib
import json
import os
import sys
import time

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import grating_coupler


def main():
    sys.stdout.reconfigure(line_buffering=True)

    p = base_parser(__doc__)
    p.add_argument("--config", required=True, metavar="CONFIG_JSON",
                   help="config.json of the run that produced the design; "
                        "every field is inherited, --set overrides on top")
    p.add_argument("--rho", required=True, metavar="NPY",
                   help="frozen binary design, shape design_shape_2d(cfg)")
    p.add_argument("--wg-width", type=float, default=10.0,
                   help="waveguide width W (um) of the 3D scene (a CLI arg of "
                        "scripts/15, not a config field — pass the same value)")
    p.add_argument("--allow-t-si-snap", action="store_true",
                   help="accept a t_si that snaps on the grid (same meaning "
                        "as in scripts/15)")
    args = p.parse_args()

    with open(args.config) as f:
        stored = json.load(f)
    cfg = grating_coupler.GratingCouplerConfig(**stored)
    cfg.beta_schedule = tuple(cfg.beta_schedule)  # JSON round-trip, as in 15
    cfg = apply_overrides(cfg, args)
    grating_coupler.assert_design_grid_snaps_2d(cfg, allow_t_si_snap=args.allow_t_si_snap)

    nx, ny = grating_coupler.design_shape_2d(cfg)
    rho = np.load(args.rho)
    if rho.shape != (nx, ny):
        raise SystemExit(f"--rho {args.rho} has shape {rho.shape}; the config "
                         f"needs ({nx}, {ny})")
    vals = np.unique(rho)
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise SystemExit(f"--rho must be strictly binary, found values {vals}"
                         f" — this driver evaluates the DELIVERED design, not "
                         f"a latent")
    with open(args.rho, "rb") as f:
        rho_sha = hashlib.sha256(f.read()).hexdigest()

    d = start_run(cfg, args, "eval2d")

    # the actual silicon thickness this grid realizes (fdtdx round() snap) —
    # both grids of a CE ratio must report it, or the ratio hides a confound
    t_si_cells = round(cfg.t_si / cfg.spacing_um)
    t_si_snapped = t_si_cells * cfg.spacing_um
    print(f"[grid] spacing {cfg.spacing_um} um: t_si {cfg.t_si} um -> "
          f"{t_si_cells} cells = {t_si_snapped:.6g} um on the grid")

    lams = [cfg.lam_c]
    t0 = time.time()
    p_in = grating_coupler.beam_power_3d(cfg, lams, wg_width_um=args.wg_width)
    print(f"[beam] 3D P_in = {['%.4g' % v for v in p_in]} at {lams} um "
          f"({time.time() - t0:.0f}s)")

    # with_transforms=False: prescribed-profile placement; beta is unused
    # (no TanhProjection in the chain) but the loss signature still takes it
    _, _, _, _, device, value_fn = grating_coupler.make_ce_value_and_grad_3d(
        cfg, p_in, lams=lams, with_transforms=False,
        wg_width_um=args.wg_width, allow_t_si_snap=args.allow_t_si_snap)

    t0 = time.time()
    # jax.device_get first: under multi-device sharding (CUDA_VISIBLE_DEVICES=0,1
    # -> fdtdx's built-in X-axis split) the result is not fully replicated and
    # __float__ raises. device_get gathers; a no-op on a single device.
    import jax  # deferred import: this script sets env before touching jax elsewhere
    loss = float(np.asarray(jax.device_get(
        value_fn(rho.reshape(nx, ny, 1).astype(np.float32), 128.0))))
    ce = -loss
    wall = time.time() - t0
    ce_db = float(10 * np.log10(max(ce, 1e-15)))
    print(f"[eval] CE = {ce:.6g} ({ce_db:.3f} dB) at spacing "
          f"{cfg.spacing_um} um  ({wall:.0f}s)")

    res = {
        "CE": ce,
        "CE_dB": ce_db,
        "spacing_um": cfg.spacing_um,
        "t_si_config_um": cfg.t_si,
        "t_si_cells": t_si_cells,
        "t_si_snapped_um": t_si_snapped,
        "P_in": p_in,
        "lams_um": lams,
        "wg_width_um": args.wg_width,
        "rho_file": os.path.abspath(args.rho),
        "rho_sha256": rho_sha,
        "rho_shape": [nx, ny],
        "config_source": os.path.abspath(args.config),
        "wall_s_forward": wall,
    }
    from invdx import runio
    runio.save_json(os.path.join(d, "results.json"), res)
    print(f"[done] {d}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
