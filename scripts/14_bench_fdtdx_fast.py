#!/usr/bin/env python
"""Vanilla fdtdx.run_fdtd vs invdx run_fdtd_fast on the 10_benchmark case.

Same synthetic case as scripts/10_benchmark.py (cubic cell, dielectric slab,
uniform plane source, PML all around, no detectors): pure field-update
throughput. Runs each engine twice — the first call pays trace+compile, the
second is the hot rate that matters for optimization sweeps — and also
compares the final E/H fields of the two engines bitwise (the GPU backend
preserves bitwise equality at default XLA flags; the CPU gate for that claim
is tests/test_fdtdx_perf.py).

  python scripts/14_bench_fdtdx_fast.py --size 8 --res 25 --steps 500 --gpu 0
"""

import argparse
import json
import os
import time


def build_case(size_um, res_per_um, steps):
    import jax
    import jax.numpy as jnp
    import numpy as np

    import fdtdx

    spacing = 1.0 / res_per_um * 1e-6
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
    cells = int(np.prod(arrays.fields.E.shape[1:]))
    return arrays, objects, sim_config, key, cells


def bench(run_fn, arrays, objects, sim_config, key):
    import jax

    t0 = time.time()
    _, out = run_fn(arrays=arrays, objects=objects, config=sim_config, key=key)
    jax.block_until_ready(out.fields.E)
    cold = time.time() - t0
    t0 = time.time()
    _, out = run_fn(arrays=arrays, objects=objects, config=sim_config, key=key)
    jax.block_until_ready(out.fields.E)
    hot = time.time() - t0
    return cold, hot, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=float, default=8.0)
    p.add_argument("--res", type=int, default=25)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", default="runs/benchmark_fast.json")
    args = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import functools

    import jax
    import jax.numpy as jnp

    import fdtdx
    from invdx.engines.fdtdx_perf import run_fdtd_fast

    arrays, objects, sim_config, key, cells = build_case(
        [args.size] * 3, args.res, args.steps)
    n_steps = sim_config.time_steps_total
    dev = jax.devices()[0]
    print(f"[case] {cells/1e6:.1f}M cells, {n_steps} steps, "
          f"{dev.platform}:{dev.device_kind}")

    vanilla = functools.partial(fdtdx.run_fdtd, show_progress=False)
    rows = {}
    outs = {}
    for name, fn in (("vanilla", vanilla), ("fast", run_fdtd_fast)):
        cold, hot, out = bench(fn, arrays, objects, sim_config, key)
        rate = cells * n_steps / hot / 1e6
        rows[name] = {"cold_s": cold, "hot_s": hot,
                      "hot_mcell_steps_per_s": rate}
        outs[name] = out
        print(f"[bench] {name:8s} cold {cold:7.2f}s  hot {hot:7.2f}s  "
              f"{rate:8.0f} Mcell-steps/s")

    dE = float(jnp.max(jnp.abs(outs["fast"].fields.E - outs["vanilla"].fields.E)))
    dH = float(jnp.max(jnp.abs(outs["fast"].fields.H - outs["vanilla"].fields.H)))
    speedup = rows["vanilla"]["hot_s"] / rows["fast"]["hot_s"]
    print(f"[equiv] max|dE| = {dE:.3e}   max|dH| = {dH:.3e}")
    print(f"[result] hot-rate speedup: {speedup:.2f}x")

    out_doc = {"case": {"size_um": [args.size] * 3, "res_per_um": args.res,
                        "steps": int(n_steps), "cells": cells,
                        "device": f"{dev.platform}:{dev.device_kind}"},
               "results": rows,
               "speedup_hot": speedup,
               "max_abs_dE": dE, "max_abs_dH": dH}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
