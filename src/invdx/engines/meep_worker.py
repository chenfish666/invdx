"""Child side of the cross-env Meep protocol — runs INSIDE the Meep
environment (the spack view; `conda` was the earlier host and the pixi
fallback still is), usually under mpirun. Never import jax or fdtdx here.

Which invdx modules are actually importable there is narrower than it looks,
and was measured rather than assumed: `runio`, `fab.measure` and
`engines.conventions` are numpy-pure and import fine; `modes` and
`fab.filters_np` need autograd, which the spack view does not carry, so they
raise. `meep.adjoint` needs it too and is likewise unavailable — forward runs
are unaffected, adjoint ones are not. An earlier version of this docstring
listed `modes` and `fab.filters_np` as usable.

Usage:  mpirun -np N python meep_worker.py <jobdir>
Reads <jobdir>/job.json {"task": ..., "payload": {...}} (+ optional .npy
inputs), writes <jobdir>/result.json (master rank only).
"""

import json
import os
import sys

import numpy as np


def task_ping(payload, jobdir):
    import meep as mp
    return {
        "meep_version": getattr(mp, "__version__", "?"),
        "processes": mp.count_processors(),
        "python": sys.version.split()[0],
    }


def _flux_through_cell(mp, resolution, decay_tol, geometry=()):
    """2D cell, +y Gaussian pulse plane source at the bottom, flux line at the
    top; returns the transmitted flux. Shared by vacuum/slab tasks so
    normalization cancels in ratios."""
    fcen = 1.0 / 1.55
    cell = mp.Vector3(4, 6)
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=[mp.PML(1.0)],
        geometry=list(geometry),
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                           component=mp.Ez,
                           center=mp.Vector3(0, -2.0),
                           size=mp.Vector3(4, 0))],
        k_point=mp.Vector3(),          # periodic sides: clean plane wave
    )
    fr = sim.add_flux(fcen, 0, 1,
                      mp.FluxRegion(center=mp.Vector3(0, 2.0),
                                    size=mp.Vector3(4, 0)))
    # decay_by passed explicitly — the 1e-11 default is ~3.4x slower (lesson 4)
    sim.run(until_after_sources=mp.stop_when_dft_decayed(decay_tol, 0, None))
    return mp.get_fluxes(fr)[0]


def task_vacuum_flux(payload, jobdir):
    import meep as mp
    resolution = int(payload.get("resolution", 20))
    decay_tol = float(payload.get("dft_decay_tol", 1e-6))
    flux = _flux_through_cell(mp, resolution, decay_tol)
    return {"flux": float(flux), "resolution": resolution}


def task_slab_transmission(payload, jobdir):
    """Transmission of a lossless dielectric slab at normal incidence,
    normalized by an empty-cell run (ratio cancels power conventions)."""
    import meep as mp
    resolution = int(payload.get("resolution", 20))
    decay_tol = float(payload.get("dft_decay_tol", 1e-6))
    n_slab = float(payload.get("n_slab", 2.0))
    t_slab = float(payload.get("t_slab", 0.5))
    slab = mp.Block(center=mp.Vector3(0, 0),
                    size=mp.Vector3(mp.inf, t_slab, mp.inf),
                    material=mp.Medium(index=n_slab))
    empty = _flux_through_cell(mp, resolution, decay_tol)
    through = _flux_through_cell(mp, resolution, decay_tol, geometry=[slab])
    return {"transmission": float(through / empty),
            "flux_empty": float(empty), "flux_slab": float(through),
            "n_slab": n_slab, "t_slab": t_slab, "resolution": resolution}


def task_phc_bend(payload, jobdir):
    """PhC 90-bend transmission spectra (problems/phc_bend geometry, defined
    once and imported here so both engines simulate the SAME rod list).

    Four runs -> two ratio spectra:
      T_bend(f) = P(bend out) / P(straight out)      bend transmission
      T_bulk(f) = P(through crystal) / P(vacuum)     band-gap location
    """
    import types

    import meep as mp

    from invdx.problems import phc_bend  # numpy-pure, safe in the meep env

    cfg = types.SimpleNamespace(
        n_side=int(payload["n_side"]), r_rod=float(payload["r_rod"]),
        eps_rod=float(payload["eps_rod"]), pad_a=float(payload["pad_a"]))
    res = int(payload.get("resolution", 20))
    decay_tol = float(payload.get("dft_decay_tol", 1e-6))
    f_min, f_max = float(payload["f_min"]), float(payload["f_max"])
    n_freq = int(payload["n_freq"])
    defect = payload.get("defect")
    defect = tuple(defect) if defect else None

    L = cfg.n_side + 2 * cfg.pad_a
    c = cfg.n_side // 2
    fcen, df = 0.5 * (f_min + f_max), (f_max - f_min)

    def a2m(x, y):
        return mp.Vector3(x - L / 2, y - L / 2)

    src_pos = a2m(cfg.pad_a + 1.0, cfg.pad_a + c + 0.5)
    x_out = cfg.pad_a + cfg.n_side - 1.0
    straight_fr = dict(center=a2m(x_out, cfg.pad_a + c + 0.5),
                       size=mp.Vector3(0, 3.0))
    bend_fr = dict(center=a2m(cfg.pad_a + c + 0.5, x_out),
                   size=mp.Vector3(3.0, 0))

    def run_flux(layout, dfct, fr):
        rods = [] if layout == "bulk_empty" else [
            mp.Cylinder(radius=cfg.r_rod, height=mp.inf,
                        center=a2m(cx, cy),
                        material=mp.Medium(epsilon=cfg.eps_rod))
            for cx, cy in phc_bend.rod_centers_a(cfg, layout, dfct)]
        sim = mp.Simulation(
            cell_size=mp.Vector3(L, L),
            resolution=res,
            boundary_layers=[mp.PML(2.0)],
            geometry=rods,
            sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df),
                               component=mp.Ez, center=src_pos)],
            force_complex_fields=False,
        )
        flux = sim.add_flux(fcen, df, n_freq, mp.FluxRegion(**fr))
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            decay_tol, 0, None))
        out = np.asarray(mp.get_fluxes(flux))
        freqs = np.asarray(mp.get_flux_freqs(flux))
        sim.reset_meep()
        return freqs, out

    freqs, p_straight = run_flux("straight", None, straight_fr)
    _, p_bend = run_flux("bend", defect, bend_fr)
    _, p_empty = run_flux("bulk_empty", None, straight_fr)
    _, p_bulk = run_flux("bulk", None, straight_fr)

    return {"freqs": freqs.tolist(),
            "T_bend": (p_bend / p_straight).tolist(),
            "T_bulk": (p_bulk / p_empty).tolist(),
            "P_straight": p_straight.tolist(), "P_bend": p_bend.tolist(),
            "resolution": res, "defect": list(defect) if defect else None}


def task_benchmark(payload, jobdir):
    """Raw throughput benchmark: a 3D box with a dielectric slab, fixed
    number of time steps, no detectors (pure field-update rate). Returns
    cells, steps and wall time so Mcell-steps/s is comparable across
    engines on the SAME synthetic case."""
    import time

    import meep as mp

    res = int(payload.get("resolution", 20))
    size = payload.get("size_um", [8.0, 8.0, 8.0])
    steps = int(payload.get("steps", 500))
    cell = mp.Vector3(*size)
    sim = mp.Simulation(
        cell_size=cell, resolution=res,
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(center=mp.Vector3(), material=mp.Medium(index=2.0),
                           size=mp.Vector3(size[0], size[1], size[2] / 4))],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                           component=mp.Ez, center=mp.Vector3())],
    )
    sim.init_sim()
    dt = 0.5 / res                      # Meep courant 0.5 default
    t0 = time.time()
    sim.run(until=steps * dt)
    wall = time.time() - t0
    nx, ny, nz = (int(round(s * res)) for s in size)
    cells = nx * ny * nz
    return {"engine": "meep", "cells": cells, "steps": steps,
            "wall_s": wall,
            "mcell_steps_per_s": cells * steps / wall / 1e6,
            "processes": mp.count_processors(), "resolution": res}


TASKS = {
    "ping": task_ping,
    "vacuum_flux": task_vacuum_flux,
    "slab_transmission": task_slab_transmission,
    "phc_bend": task_phc_bend,
    "benchmark": task_benchmark,
}


def main(jobdir):
    from invdx import runio

    with open(os.path.join(jobdir, "job.json")) as f:
        job = json.load(f)
    result = TASKS[job["task"]](job.get("payload", {}), jobdir)
    runio.save_json(os.path.join(jobdir, "result.json"), result)
    if runio.am_master():
        print(f"[worker] {job['task']} done -> result.json")


if __name__ == "__main__":
    main(sys.argv[1])
