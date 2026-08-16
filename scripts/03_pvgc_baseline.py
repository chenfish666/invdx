#!/usr/bin/env python
"""PVGC cross-engine acceptance baselines (fdtdx vs pvgc/Meep references).

Uniform fully-etched grating P=0.575um duty=0.5 on 220nm SOI, Config B
(fiber-side Gaussian beam), CE into the -x slab TE0 mode at 1310nm:

  theta=10deg : pvgc res-80 reference  -10.1 dB (Meep, fine period sweep)
  theta=0deg  : pvgc v1 diagnosis      ~ -27 dB (symmetric grating at
                vertical incidence is physics-limited by 2nd-order Bragg)

Agreement between the two independent engines validates BOTH the new
problem layer and the v1 root-cause diagnosis.

  python scripts/03_pvgc_baseline.py                      # res-80, ~minutes
  python scripts/03_pvgc_baseline.py --set spacing_um=0.025   # quick look
"""

import os
import time

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import pvgc
from invdx import runio

# Cross-engine acceptance is SPECTRAL (conventions.py lesson 6): engines
# discretize the same nominal geometry into slightly different effective
# linewidth, and the ridge moves ~2.4 nm/nm of tooth width — so we accept on
# ridge peak (within ~2 dB of pvgc's -10.1) and report the ridge wavelength,
# never on CE at exactly 1310 nm.
PVGC_PEAK_REF_DB = -10.1


def main():
    p = base_parser(__doc__)
    p.add_argument("--period", type=float, default=0.575)
    p.add_argument("--duty", type=float, default=0.5)
    args = p.parse_args()
    cfg = apply_overrides(pvgc.PVGCConfig(), args)
    d = start_run(cfg, args, "pvgc-baseline")

    teeth = pvgc.uniform_grating_teeth(cfg, period=args.period, duty=args.duty)
    results = {}

    # theta=10: dense spectrum, ridge-based comparison
    cfg.theta_deg = 10.0
    t0 = time.time()
    lams = list(np.linspace(1.26, 1.36, 11))
    spec = pvgc.characterize_spectrum(cfg, teeth, lams)
    ce_db = [r["CE_dB"] for r in spec["spectrum"]]
    k = int(np.argmax(ce_db))
    results["theta_10_spectrum"] = spec
    results["theta_10_peak"] = {"lam_um": lams[k], "CE_dB": ce_db[k],
                                "wall_s": round(time.time() - t0, 1)}
    print(f"[baseline] theta=10 ridge: {ce_db[k]:.2f} dB @ {lams[k]:.3f} um "
          f"(pvgc peak ref {PVGC_PEAK_REF_DB} dB @ 1.310; ridge position "
          f"shifts with effective duty — see conventions lesson 6)")

    # theta=0: single-wavelength suppression check (broadband-deep, robust)
    cfg.theta_deg = 0.0
    t0 = time.time()
    res = pvgc.characterize(cfg, teeth)
    res["wall_s"] = round(time.time() - t0, 1)
    results["theta_0"] = res
    print(f"[baseline] theta=0: CE = {res['CE_dB']:.2f} dB (expect < -20 dB, "
          f"symmetric vertical suppression)  [{res['wall_s']}s]")

    runio.save_json(os.path.join(d, "results.json"), results)
    print(f"[done] {d}/results.json")


if __name__ == "__main__":
    main()
