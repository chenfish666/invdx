#!/usr/bin/env python
"""Tiny forward-only fdtdx vacuum simulation on GPU, run through the full
config / --set / run-directory plumbing. Proves the Layer A GPU path works
end-to-end before anything physics-critical is built on it.

  python scripts/01_smoke_fdtdx.py --tag smoke --set spacing_um=0.05
"""

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.config import BaseConfig
from invdx.engines import fdtdx_engine
from invdx import runio

import os


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = apply_overrides(BaseConfig(spacing_um=0.05), args)
    d = start_run(cfg, args, "smoke-fdtdx")

    platform = fdtdx_engine.gpu_platform()
    print(f"[smoke] jax platform: {platform}")

    config, objs, cons, det = fdtdx_engine.vacuum_flux_scene(cfg)
    arrays = fdtdx_engine.run_forward(config, objs, cons, seed=cfg.seed)
    flux = fdtdx_engine.steady_flux(arrays, det)
    print(f"[smoke] steady vacuum flux through top plane: {flux:.6g}")

    runio.save_json(os.path.join(d, "result.json"),
                    {"platform": platform, "flux": flux})
    print(f"[done] results in {d}")


if __name__ == "__main__":
    main()
