"""Child side of the cross-env Meep protocol — runs INSIDE the Meep
environment (the spack view; `conda` was the earlier host and the pixi
fallback still is), usually under mpirun. Never import jax or fdtdx here.

Which invdx modules are actually importable there is narrower than it looks,
and was measured rather than assumed: `runio`, `fab.measure` and
`engines.conventions` are numpy-pure and import fine; `modes` and
`fab.filters_np` need autograd, which the spack view does not carry, so they
raise. `meep.adjoint` needs it too and is likewise unavailable — forward runs
are unaffected, adjoint ones are not. An earlier version of this docstring
listed `modes` and `fab.filters_np` as usable. `problems.phc_bend` and
`problems.grating_coupler_np` are numpy-pure by design; the second is held to
it by tests/test_grating_coupler_np.py, which imports it (and this module)
with jax, fdtdx, autograd and meep made unimportable. `problems.grating_coupler`
itself imports jax and is never imported here.

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


# grating_coupler: tolerance of the lesson-1 check below. Meep's own flux and
# the power summed here from the DFT arrays of the same line should agree to
# well under 1%; what this catches is a factor of 2 (the convention) or of
# the resolution (a wrong dl), both silent otherwise.
GC_FLUX_CHECK_TOL = 0.02


def _beam_points(gc, beam):
    """The beam's source line as point sources one pixel apart: (x,
    amplitude, start_time). A line source has one time profile for all its
    points and the tilt IS a per-point delay
    (grating_coupler_np.fiber_beam_profile), so the line is sampled into
    point sources, each launched at its own time; the earliest starts at 0.
    """
    n = int(round(beam["width"] / beam["pitch"]))
    xs = beam["x0"] + (np.arange(n) - (n - 1) / 2) * beam["pitch"]
    amp, delay = gc.fiber_beam_profile(xs, beam["x0"], beam["w0"],
                                       beam["theta_deg"])
    return xs, amp, delay - delay.min()


def _line_dft(sim, dft, comps, nfreq, monitor, shift, res, name):
    """Complex DFT arrays of `comps` on a line monitor, one row per
    frequency, and the sample coordinates along the line (problem frame).

    The coordinates come from Meep's own metadata for this DFT object and are
    checked against where the monitor was asked to be: the analytic mode is
    sampled on them, and an offset here would put the mode beside the slab
    without raising anything.
    """
    along = 0 if monitor["size"][0] > 0 else 1
    tics = sim.get_array_metadata(dft_cell=dft)
    coords = np.asarray(tics[along], dtype=float).ravel() + shift[along]
    across = np.asarray(tics[1 - along], dtype=float).ravel() + shift[1 - along]
    fields = {c: np.array([np.asarray(sim.get_dft_array(dft, comp, i)).ravel()
                           for i in range(nfreq)])
              for c, comp in comps}
    lo = monitor["center"][along] - monitor["size"][along] / 2
    hi = lo + monitor["size"][along]
    off = max(abs(coords.min() - lo), abs(coords.max() - hi),
              float(np.max(np.abs(across - monitor["center"][1 - along]))))
    if off > 1.5 / res:
        raise RuntimeError(
            f"{name}: DFT samples span [{coords.min():.4f}, {coords.max():.4f}]"
            f" at {across.mean():.4f}, asked for [{lo:.4f}, {hi:.4f}] at "
            f"{monitor['center'][1 - along]:.4f} (problem frame): Meep's "
            f"array metadata is not in the frame this task assumes, and the "
            f"mode would be sampled in the wrong place. No CE reported.")
    for c, arr in fields.items():
        if arr.shape[1] != coords.size:
            raise RuntimeError(
                f"{name}: {c} has {arr.shape[1]} DFT samples per frequency "
                f"but the metadata lists {coords.size} coordinates. No CE "
                f"reported.")
    return coords, fields


def _gc_measure(payload, xs_f, empty, flux_in, ys_w, grating):
    """CE(lambda) rows and the two field checks, from what the two Meep runs
    of task_grating_coupler recorded; numpy only, so the test suite runs it
    on fields built from Maxwell's equations without Meep.

    xs_f, empty["Ez"/"Hx"], flux_in -- fiber_mon samples (x, problem frame),
        their DFT arrays (one row per wavelength) and Meep's own flux, from
        the empty run
    ys_w, grating["Ez"/"Hy"] -- wg_mon samples (y) and DFT arrays, from the
        grating run
    """
    from invdx.engines import conventions
    from invdx.problems import grating_coupler_np as gc

    beam, mons = payload["beam"], payload["monitors"]
    lams = [float(lam) for lam in payload["lams_um"]]
    dl = 1.0 / int(payload["resolution"])
    rows = []
    for i, lam in enumerate(lams):
        ez_f = empty["Ez"][i]
        row = gc.ce_sample(lam, ez_f, empty["Hx"][i], grating["Ez"][i],
                           -grating["Hy"][i], ys_w, dl, **payload["mode"])
        intensity = np.abs(ez_f) ** 2
        row["flux_over_2P_in"] = float(
            abs(flux_in[i]) / conventions.meep_to_physical_power(row["P_in"]))
        row["beam_centroid_um"] = float(np.sum(xs_f * intensity)
                                        / np.sum(intensity))
        rows.append(row)

    worst = max(abs(r["flux_over_2P_in"] - 1.0) for r in rows)
    if worst > GC_FLUX_CHECK_TOL:
        raise RuntimeError(
            f"Meep's flux through fiber_mon and 2 x P_in from its DFT arrays "
            f"disagree by up to {worst:.1%} (allowed {GC_FLUX_CHECK_TOL:.0%}):"
            f" the arrays are not read the way the CE chain assumes "
            f"(conventions lesson 1). No CE reported.")
    walk = (beam["y"] - mons["fiber_mon"]["center"][1]) \
        * np.tan(np.deg2rad(beam["theta_deg"]))
    expected = beam["x0"] + walk
    centroid_tol = max(0.25 * abs(walk), 2.0 * dl)
    off = [r for r in rows
           if abs(r["beam_centroid_um"] - expected) > centroid_tol]
    if off:
        raise RuntimeError(
            f"beam centroid on fiber_mon at {off[0]['beam_centroid_um']:+.3f}"
            f" um ({off[0]['lam_um']} um), expected {expected:+.3f} +- "
            f"{centroid_tol:.3f}: the beam does not leave at theta = "
            f"{beam['theta_deg']} toward +x the way the fdtdx scene's does. "
            f"No CE reported.")

    return rows, {"flux_over_2P_in_max_dev": float(worst),
                  "flux_check_tol": GC_FLUX_CHECK_TOL,
                  "beam_centroid_expected_um": float(expected),
                  "beam_centroid_tol_um": float(centroid_tol)}


def task_grating_coupler(payload, jobdir):
    """grating_coupler CE(lambda) on Meep: the fiber-side measurement chain of
    problems/grating_coupler, on the device `payload` describes
    (grating_coupler_np.meep_payload, built from the blocks and planes the
    fdtdx scene uses).

    Two runs and one source, as on the fdtdx side:
      empty    no chip: the incident beam on fiber_mon -> P_in(lambda)
      grating  the chip: fields on wg_mon -> P_mode(lambda), the -x TE0 power
    CE = P_mode / P_in comes from grating_coupler_np.ce_sample, the function
    `characterize_spectrum` calls too.

    Meep-side conventions, each checked here rather than assumed:
      - polarization: the problem's Ez (fdtdx's Ey). Meep's
        Sx = -1/2 Re(Ez conj(Hy)), so -Hy is the H ce_sample's wg line wants;
        a -x traveling mode (Hy = +n_eff Ez) is then accepted and a +x one
        rejected.
      - power: P_in and P_mode both carry the physical 1/2, so Meep's missing
        1/2 (conventions lesson 1) cancels in CE. It is checked anyway: Meep's
        own flux through fiber_mon must equal meep_to_physical_power(P_in),
        or the DFT arrays are not being read the way this task assumes and
        no CE is reported.
      - tilt: for theta > 0 the beam must walk toward +x
        (grating_coupler_np.fiber_beam_profile); its centroid on fiber_mon is
        checked against x0 + (y_src - y_mon) tan(theta) at every wavelength.
      - decay: stop_when_dft_decayed(payload["dft_decay_tol"]), passed
        explicitly (lesson 4); a run that has not decayed by
        payload["max_run_time"] raises instead of reporting.
    """
    import time

    import meep as mp

    from invdx.problems import grating_coupler_np as gc

    res = int(payload["resolution"])
    cell = payload["cell"]
    shift = np.asarray(cell["center"], dtype=float)    # problem -> Meep frame
    beam, mons = payload["beam"], payload["monitors"]
    lams = [float(lam) for lam in payload["lams_um"]]
    freqs = [1.0 / lam for lam in lams]
    tol = float(payload["dft_decay_tol"])
    t_cap = float(payload["max_run_time"])
    xs_src, amp, t_start = _beam_points(gc, beam)

    def v3(x, y):
        return mp.Vector3(float(x - shift[0]), float(y - shift[1]))

    def run(with_chip, name, comps):
        geometry = [mp.Block(center=v3(*b["center"]),
                             size=mp.Vector3(b["size"][0], b["size"][1],
                                             mp.inf),
                             material=mp.Medium(index=b["index"]))
                    for b in payload["blocks"]] if with_chip else []
        sources = [mp.Source(mp.GaussianSource(beam["fcen"],
                                               fwidth=beam["fwidth"],
                                               start_time=float(t)),
                             component=mp.Ez, center=v3(x, beam["y"]),
                             amplitude=float(a))
                   for x, a, t in zip(xs_src, amp, t_start)]
        sim = mp.Simulation(cell_size=mp.Vector3(*cell["size"]),
                            resolution=res,
                            boundary_layers=[mp.PML(cell["dpml"])],
                            geometry=geometry, sources=sources)
        mon = mons[name]
        where = dict(center=v3(*mon["center"]), size=mp.Vector3(*mon["size"]))
        dft = sim.add_dft_fields([comp for _, comp in comps], freqs, **where)
        flux = sim.add_flux(freqs, mp.FluxRegion(**where))
        decayed = mp.stop_when_dft_decayed(tol, 0, None)
        t0 = time.time()
        sim.run(until_after_sources=lambda s: decayed(s)
                or s.meep_time() >= t_cap)
        info = {"meep_time": float(sim.meep_time()),
                "wall_s": round(time.time() - t0, 1)}
        if info["meep_time"] >= t_cap:
            raise RuntimeError(
                f"{name} run ({'grating' if with_chip else 'empty'}): DFT "
                f"fields had not decayed to {tol:g} by t = {t_cap:g} (um/c). "
                f"No CE reported.")
        coords, fields = _line_dft(sim, dft, comps, len(freqs), mon, shift,
                                   res, name)
        F = np.asarray(mp.get_fluxes(flux), dtype=float)
        sim.reset_meep()
        return coords, fields, F, info

    xs_f, empty, flux_in, info_e = run(False, "fiber_mon",
                                       [("Ez", mp.Ez), ("Hx", mp.Hx)])
    ys_w, grating, _, info_g = run(True, "wg_mon",
                                   [("Ez", mp.Ez), ("Hy", mp.Hy)])

    rows, checks = _gc_measure(payload, xs_f, empty, flux_in, ys_w, grating)

    return {"engine": "meep", "meep_version": getattr(mp, "__version__", "?"),
            "processes": mp.count_processors(), "resolution": res,
            "n_sources": int(xs_src.size), "spectrum": rows,
            "checks": checks,
            "runs": {"empty": info_e, "grating": info_g},
            "payload": payload}


TASKS = {
    "ping": task_ping,
    "vacuum_flux": task_vacuum_flux,
    "slab_transmission": task_slab_transmission,
    "phc_bend": task_phc_bend,
    "grating_coupler": task_grating_coupler,
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
