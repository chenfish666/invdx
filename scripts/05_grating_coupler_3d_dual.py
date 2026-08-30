#!/usr/bin/env python
"""Dual-GPU 3D grating_coupler validation: the empty-cell normalization run and the
grating run execute SIMULTANEOUSLY on GPU 0 and GPU 1 (one process per
device), then the phasor planes are combined. Task-level parallelism —
first brick of the orchestration layer: ~2x wall time vs scripts/04, and
each card hosts exactly one ArrayContainer (no dual-container OOM).

  python scripts/05_grating_coupler_3d_dual.py --tag dual3d          # orchestrator
  (worker mode --role empty|grating is invoked internally)
"""

import os
import subprocess
import sys
import time

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import grating_coupler
from invdx import runio

SPACING_575_16 = 0.575 / 16
LAMS = list(np.linspace(1.28, 1.36, 9))


def make_cfg(args):
    return apply_overrides(
        grating_coupler.GratingCouplerConfig(spacing_um=SPACING_575_16, sim_time_s=0.8e-12,
                        theta_deg=10.0), args)


def worker(args):
    cfg = make_cfg(args)
    teeth = (None if args.role == "empty" else
             grating_coupler.uniform_grating_teeth(cfg, args.period, args.duty))
    planes = grating_coupler.run_3d_planes(cfg, teeth, args.width,
                                with_chip=(args.role == "grating"),
                                lams_um=LAMS)
    np.savez(args.out, **planes)
    print(f"[worker:{args.role}] planes -> {args.out}")


def orchestrate(args, argv):
    d = start_run(make_cfg(args), args, "coupler-3d-dual")
    t0 = time.time()
    procs = []
    for gpu, role in ((1, "empty"), (0, "grating")):
        out = os.path.join(d, f"planes_{role}.npz")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")
        cmd = [sys.executable, os.path.abspath(__file__),
               "--role", role, "--out", out] + argv
        procs.append((role, out, subprocess.Popen(
            cmd, env=env,
            stdout=open(os.path.join(d, f"{role}.log"), "w"),
            stderr=subprocess.STDOUT)))
    outs = {}
    for role, out, proc in procs:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"worker {role} failed (rc={rc}) — see {d}/{role}.log")
        outs[role] = np.load(out)
    wall = time.time() - t0

    cfg = make_cfg(args)
    res = grating_coupler.combine_3d(outs["empty"]["fiber"], outs["grating"]["wg"],
                          cfg, LAMS, wg_width_um=args.width)
    res["wall_s"] = round(wall, 1)
    for r in res["spectrum"]:
        print(f"[3d]   {r['lam_um']:.3f} um: {r['CE_dB']:7.2f} dB")
    pk = res["peak"]
    print(f"[3d] ridge peak: {pk['CE_dB']:.2f} dB @ {pk['lam_um']:.3f} um  "
          f"[dual-GPU wall {res['wall_s']}s]")
    print(f"[3d] {res['note']}")
    runio.save_json(os.path.join(d, "results.json"), res)
    print(f"[done] {d}/results.json")


def main():
    p = base_parser(__doc__)
    p.add_argument("--period", type=float, default=0.575)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--width", type=float, default=10.0)
    p.add_argument("--role", choices=("empty", "grating"), default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.role:
        worker(args)
    else:
        # pass through user-facing args to workers (not --role/--out)
        argv = []
        for kv in args.set:
            argv += ["--set", kv]
        argv += ["--period", str(args.period), "--duty", str(args.duty),
                 "--width", str(args.width)]
        orchestrate(args, argv)


if __name__ == "__main__":
    main()
