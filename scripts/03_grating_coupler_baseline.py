#!/usr/bin/env python
"""grating_coupler cross-engine acceptance baselines: fdtdx against an independent
Meep reference computed from a fine period sweep.

Uniform fully-etched grating P=0.575um duty=0.5 on 220nm SOI, fiber-side
excitation (Gaussian beam), CE into the -x slab TE0 mode at 1310nm, at two
incidence angles:

  theta=10deg : off-vertical incidence — the coupled case, checked against
                the Meep reference peak below
  theta=0deg  : a symmetric grating at vertical incidence is physics-limited
                by second-order Bragg, so CE collapses

Agreement between the two independent engines validates both the problem
layer and the second-order-Bragg explanation of the theta=0 collapse.

  python scripts/03_grating_coupler_baseline.py                          # ~minutes
  python scripts/03_grating_coupler_baseline.py --set spacing_um=0.025   # quick look
"""

import os
import time

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import grating_coupler
from invdx import runio

# Cross-engine acceptance is SPECTRAL (conventions.py lesson 6): engines
# discretize the same nominal geometry into slightly different effective
# linewidth, and the ridge wavelength moves measurably with tooth width — so
# acceptance is on the ridge peak (within ~2 dB of the reference peak below)
# and the ridge wavelength is reported, never on CE at exactly 1310 nm.
REFERENCE_PEAK_DB = -10.1


def main():
    p = base_parser(__doc__)
    p.add_argument("--period", type=float, default=0.575)
    p.add_argument("--duty", type=float, default=0.5)
    args = p.parse_args()
    cfg = apply_overrides(grating_coupler.GratingCouplerConfig(), args)
    d = start_run(cfg, args, "coupler-baseline")

    teeth = grating_coupler.uniform_grating_teeth(cfg, period=args.period, duty=args.duty)
    results = {}

    # theta=10: dense spectrum, ridge-based comparison
    cfg.theta_deg = 10.0
    t0 = time.time()
    lams = list(np.linspace(1.26, 1.36, 11))
    spec = grating_coupler.characterize_spectrum(cfg, teeth, lams)
    ce_db = [r["CE_dB"] for r in spec["spectrum"]]
    k = int(np.argmax(ce_db))
    results["theta_10_spectrum"] = spec
    results["theta_10_peak"] = {"lam_um": lams[k], "CE_dB": ce_db[k],
                                "wall_s": round(time.time() - t0, 1)}
    print(f"[baseline] theta=10 ridge: {ce_db[k]:.2f} dB @ {lams[k]:.3f} um "
          f"(reference peak {REFERENCE_PEAK_DB} dB @ 1.310; ridge position "
          f"shifts with effective duty — see conventions lesson 6)")

    # theta=0: single-wavelength suppression check (broadband-deep, robust)
    cfg.theta_deg = 0.0
    t0 = time.time()
    res = grating_coupler.characterize(cfg, teeth)
    res["wall_s"] = round(time.time() - t0, 1)
    results["theta_0"] = res
    print(f"[baseline] theta=0: CE = {res['CE_dB']:.2f} dB (expect < -20 dB, "
          f"symmetric vertical suppression)  [{res['wall_s']}s]")

    runio.save_json(os.path.join(d, "results.json"), results)
    print(f"[done] {d}/results.json")


if __name__ == "__main__":
    main()
