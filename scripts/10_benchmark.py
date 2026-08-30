#!/usr/bin/env python
"""Cross-engine throughput benchmark on the SAME synthetic case.

Case: 3D box (default 8x8x8 um at res 20/um -> 4.1M cells; --big uses
16x16x16 at res 40 -> 262M-cell class via --size/--res) with a dielectric
slab, fixed step count, no detectors — pure field-update throughput in
Mcell-steps/s. This replaces guesswork ("fdtdx ~10x Meep?") with measured
numbers on THIS machine; run it when the GPUs are otherwise idle or the
numbers are garbage.

  python scripts/10_benchmark.py                    # fdtdx GPU + Meep MPI
  python scripts/10_benchmark.py --np 16            # Meep rank count
  python scripts/10_benchmark.py --skip-meep        # GPU only
  python scripts/10_benchmark.py --size 12 --res 32 # bigger case

Third-solver column: run the exported case in whatever other solver you want
to compare against and append its wall time by hand — the case definition
(size/res/steps) is in the output JSON so the comparison stays
apples-to-apples.
"""

import argparse
import json
import os
import time

import numpy as np


def bench_fdtdx(size_um, res_per_um, steps, gpu="0"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    import jax
    import jax.numpy as jnp

    import fdtdx

    spacing = 1.0 / res_per_um * 1e-6
    # sim_time chosen to give ~steps steps at courant 0.99
    dt_est = 0.99 * spacing / (3e8 * np.sqrt(3))
    sim_config = fdtdx.SimulationConfig(
        time=steps * dt_est, resolution=spacing, dtype=jnp.float32,
        courant_factor=0.99)
    objs, cons = [], []
    vol = fdtdx.SimulationVolume(
        partial_real_shape=tuple(s * 1e-6 for s in size_um))
    objs.append(vol)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=int(round(1.0 * res_per_um)))
    bd, cl = fdtdx.boundary_objects_from_config(bound_cfg, vol)
    cons.extend(cl)
    objs.extend(bd.values())
    slab = fdtdx.UniformMaterialObject(
        name="slab", material=fdtdx.Material(permittivity=4.0),
        partial_real_shape=(size_um[0] * 1e-6, size_um[1] * 1e-6,
                            size_um[2] / 4 * 1e-6))
    cons.append(slab.place_at_center(vol, axes=(0, 1, 2)))
    objs.append(slab)
    src = fdtdx.UniformPlaneSource(
        name="src", partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1.31e-6),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=fdtdx.WaveCharacter(wavelength=1.31e-6),
            spectral_width=fdtdx.WaveCharacter(frequency=0.2 * 2.3e14)),
        direction="-", fixed_E_polarization_vector=(0.0, 1.0, 0.0))
    cons.append(src.place_relative_to(
        vol, axes=(2,), own_positions=(1,), other_positions=(1,),
        margins=(-1.2e-6,)))
    cons.append(src.same_size(vol, axes=(0, 1)))
    objs.append(src)

    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, k2)
    n_steps = sim_config.time_steps_total
    cells = int(np.prod([int(round(s * res_per_um)) for s in size_um]))

    t0 = time.time()
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects,
                               config=sim_config, key=key)
    jax.block_until_ready(arrays.fields.E)
    wall = time.time() - t0
    dev = jax.devices()[0]
    return {"engine": f"fdtdx ({dev.platform}:{dev.device_kind})",
            "cells": cells, "steps": int(n_steps), "wall_s": wall,
            "mcell_steps_per_s": cells * n_steps / wall / 1e6,
            "note": "includes JIT compile (single call); re-run for hot rate"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=float, default=8.0)
    p.add_argument("--res", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--np", type=int, default=8, dest="n_ranks")
    p.add_argument("--gpu", default="0")
    p.add_argument("--skip-meep", action="store_true")
    p.add_argument("--skip-fdtdx", action="store_true")
    p.add_argument("--out", default="runs/benchmark.json")
    args = p.parse_args()
    size = [args.size] * 3

    rows = []
    if not args.skip_fdtdx:
        rows.append(bench_fdtdx(size, args.res, args.steps, gpu=args.gpu))
        print(f"[bench] {rows[-1]['engine']}: "
              f"{rows[-1]['mcell_steps_per_s']:.0f} Mcell-steps/s "
              f"({rows[-1]['cells']/1e6:.1f}M cells, {rows[-1]['steps']} "
              f"steps, {rows[-1]['wall_s']:.1f}s)")
    if not args.skip_meep:
        from invdx.engines import meep_bridge
        r = meep_bridge.run_job(
            "benchmark",
            {"resolution": args.res, "size_um": size, "steps": args.steps},
            n_ranks=args.n_ranks, timeout=3600)
        rows.append(r)
        print(f"[bench] meep ({r['processes']} ranks): "
              f"{r['mcell_steps_per_s']:.0f} Mcell-steps/s "
              f"({r['cells']/1e6:.1f}M cells, {r['steps']} steps, "
              f"{r['wall_s']:.1f}s)")

    out = {"case": {"size_um": size, "res_per_um": args.res,
                    "steps": args.steps}, "results": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
