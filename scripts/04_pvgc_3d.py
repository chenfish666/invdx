#!/usr/bin/env python
"""First 3D validation of the PVGC problem on GPU (pvgc validate_3d
equivalent): extrude the uniform grating to width W, radial Gaussian beam,
CE into the approximate fundamental waveguide mode.

Reference points: 2D quasi-result at theta=10 is ~-9.8 dB (25nm); 3D CE is
expected LOWER (lateral diffraction loss). The waveguide mode is a separable
TE0(z)*cos(y) approximation — stated in the output.

Long run: launch detached (setsid nohup) — see Makefile target or:
  setsid nohup env CUDA_VISIBLE_DEVICES=0 <python> scripts/04_pvgc_3d.py \
      --tag first3d > runs/3d_launch.log 2>&1 &
"""

import os
import time

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import pvgc
from invdx import runio

# default spacing 575/16 nm: period = 16 cells and tooth = 8 cells EXACTLY,
# so the effective duty is 0.500 with no grid snapping (conventions lesson 6);
# 62.7M cells fits a single RTX 6000 Ada (25nm/187M cells OOMs at 45GB)
SPACING_575_16 = 0.575 / 16


def main():
    p = base_parser(__doc__)
    p.add_argument("--period", type=float, default=0.575)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--width", type=float, default=10.0)
    args = p.parse_args()
    cfg = apply_overrides(
        pvgc.PVGCConfig(spacing_um=SPACING_575_16, sim_time_s=0.8e-12,
                        theta_deg=10.0),
        args)
    d = start_run(cfg, args, "pvgc-3d")

    teeth = pvgc.uniform_grating_teeth(cfg, period=args.period, duty=args.duty)
    lams = list(np.linspace(1.28, 1.36, 9))
    t0 = time.time()
    res = pvgc.characterize_3d(cfg, teeth, wg_width_um=args.width,
                               lams_um=lams)
    res["wall_s"] = round(time.time() - t0, 1)
    for r in res["spectrum"]:
        print(f"[3d]   {r['lam_um']:.3f} um: {r['CE_dB']:7.2f} dB")
    pk = res["peak"]
    print(f"[3d] ridge peak: {pk['CE_dB']:.2f} dB @ {pk['lam_um']:.3f} um "
          f"(2D quasi ridge ~-8.9 dB; expect lower in 3D)  "
          f"[{res['wall_s']}s]")
    print(f"[3d] {res['note']}")
    runio.save_json(os.path.join(d, "results.json"), res)
    print(f"[done] {d}/results.json")


if __name__ == "__main__":
    main()
