#!/usr/bin/env python
"""Checkpoint count C vs wall-clock time and memory, on the pvgc FOM.

The existing memory cost model has three anchors (C=10/20/28 @ 1.944M cells
-> 5.93/11.25/15.42 GB), and exactly ONE timing point (C=20 -> ~20x forward,
docs/m1-optimize.md), all measured at the M1 recipe (spacing_um=0.020,
design_grid_per_um=50, sim_time_s=0.8e-12, theta_deg=10 -> ~1.944M cells,
~21k steps). The unknown that decides whether a 3D adjoint round is 8 hours
or two weeks is the LOW-C behaviour: does C=1 cost ~20x a forward run, or
~100x? This script's job is that low-C time curve; the memory numbers are
collected along the way because they are nearly free to read off the same
runs, not because they are the point.

This script's DEFAULTS reproduce the M1 recipe above on purpose: a sweep run
with no `--set` overrides lands at the same ~1.944M cells / ~21k steps as
the three existing memory anchors, so its C=10 and C=20 rows are a direct
external check on 5.93 GB and 11.25 GB, not just new numbers next to them.
Any `--set` that changes spacing_um / design_grid_per_um / sim_time_s /
theta_deg breaks that comparability — this script does not refuse such a
run (a CPU smoke test needs a tiny grid), but it stamps
`anchor_comparable: false` plus an explicit warning into results.json so a
tiny-grid run can never be mistaken for a real check on the existing anchors.

  uv run python scripts/18_checkpoint_sweep.py --tag c-sweep \\
      --checkpoints 1,2,5,10,20,28 --repeat 3     # GPU, real numbers,
                                                   # anchor-comparable defaults
  JAX_PLATFORMS=cpu uv run python scripts/18_checkpoint_sweep.py \\
      --tag cpu-smoke --checkpoints 1,2 --set spacing_um=0.05 \\
      --set sim_time_s=0.05e-12 --set L_design=6.0 ...    # CPU smoke, seconds
                                                   # (NOT anchor-comparable)

For each requested C this builds `pvgc.make_ce_value_and_grad(cfg,
num_checkpoints=C)` fresh (num_checkpoints is baked into the traced
GradientConfig, so it cannot be swapped inside one compilation) and times
two things off the SAME compiled scene: `value_fn` (forward only — no
value_and_grad, no backward pass) as the baseline, and `vg_fn`
(value_and_grad — forward AND checkpointed backward) as the number that
actually depends on C. `ratio = vg_s / forward_s` isolates the backward
cost's growth with C from any per-C jitter in the forward baseline itself.

SCOPE — read before trusting a number out of this script:

* Quasi-2D only. `cfg.n_y_cells` (default 4) makes every scene here a thin
  periodic-y slab, exactly like the rest of this repo's pvgc problem. A real
  3D scene's bytes/cell and time/checkpoint are NOT validated here — this
  measures the pvgc problem this repo actually runs, not a 3D projection of
  it.
* The linear fit (`fit_bytes_per_cell`) is only valid inside the measured C
  range. Extrapolating it outside that range may be used to RANK candidate
  checkpoint counts against each other; it must never be used to rule a
  checkpoint count OUT (e.g. "C=64 will OOM") — that claim needs a
  measurement at or near C=64, not a line drawn through low-C points.
* `peak_bytes_in_use` is a high-water-mark counter for the whole process,
  and jax 0.11.0 exposes no public API to reset it (`jax.clear_caches()` is
  called before every C anyway, per the spec this script implements, but it
  only drops the compilation cache — the XLA allocator's peak counter is
  untouched by it; verified empirically: `jax.local_devices()[0].memory_stats()`
  has no reset method, and neither does the `jax` module). Consequence: in
  one process, `peak_bytes` for a given row is "at least the true peak for
  this C, possibly inflated by a larger C measured earlier in the same
  sweep" — a lower bound that over-reports, not under-reports (the safe
  direction per this project's measurement discipline: an inflated peak
  makes you provision more memory than needed, not less). Sweeping C in
  increasing order (the default) keeps this close to exact; a descending or
  mixed `--checkpoints` order will visibly contaminate the smaller-C rows'
  peak_bytes with the larger-C runs that preceded them. For an isolated,
  uncontaminated peak reading for one C, invoke this script with a single
  value in `--checkpoints` per process.
* On CPU, `jax.local_devices()[0].memory_stats()` returns `None` (confirmed
  empirically on jax 0.11.0 / jaxlib CPU backend) — `peak_bytes` and
  `bytes_limit` are recorded as `null` in results.json and left blank in
  sweep.csv, never as 0 or a guess. `--set device nvidia-smi` numbers are
  deliberately never used here: nvidia-smi reports the allocator's reserved
  pool, which jumps in 2^k-ish buckets and is not the working-set number the
  memory cost model is built on.
* An OOM (or any other build/measurement failure) is recorded as its own
  row — `error_type` / `error_stage` set, the timing fields for the failed
  stage left blank — never silently dropped from sweep.csv.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import pvgc

CSV_FIELDS = ["C", "forward_s", "vg_s", "ratio", "peak_bytes", "bytes_limit",
             "n_cells", "steps"]

# The exact condition the three existing memory anchors (C=10/20/28 ->
# 5.93/11.25/15.42 GB) were measured at (docs/m1-optimize.md). This script's
# own defaults reproduce it (see make_config); anchor_comparable() below
# checks whether a given --set-modified cfg still matches it.
ANCHOR_CONDITION = dict(spacing_um=0.020, design_grid_per_um=50,
                        sim_time_s=0.8e-12, theta_deg=10.0)


def anchor_comparable(cfg):
    """True iff cfg matches the exact condition the existing memory anchors
    (C=10/20/28 -> 5.93/11.25/15.42 GB, docs/m1-optimize.md) were measured
    at, so this sweep's own C=10/20 rows can be checked against them.
    Any other cfg produces real numbers, just not ones comparable to those
    anchors — callers must not assume a smoke-test grid's peak_bytes says
    anything about the 1.944M-cell numbers already on record."""
    for k, v in ANCHOR_CONDITION.items():
        cur = getattr(cfg, k)
        if isinstance(v, float):
            if not math.isclose(cur, v, rel_tol=1e-9):
                return False
        elif cur != v:
            return False
    return True


def make_config(args):
    """M1-recipe defaults (spacing_um=0.020, design_grid_per_um=50,
    sim_time_s=0.8e-12, theta_deg=10), so a bare invocation lands at the
    same ~1.944M cells / ~21k steps the existing memory anchors were
    measured at. --set overrides any of them (needed for the CPU smoke
    test, which trades comparability for a grid tiny enough for seconds)."""
    cfg = pvgc.PVGCConfig(**ANCHOR_CONDITION)
    cfg = apply_overrides(cfg, args)
    pvgc.assert_design_grid_snaps(cfg)
    return cfg


def n_cells(cfg):
    """Total Yee-cell count of the scene this sweep measures (quasi-2D: the
    thin y axis is only `cfg.n_y_cells` cells)."""
    nx = int(round(cfg.cell_x / cfg.spacing_um))
    nz = int(round(cfg.cell_z / cfg.spacing_um))
    return nx * int(cfg.n_y_cells) * nz


def n_steps(cfg):
    """Time-step count fdtdx will run for `cfg.sim_time_s` — independent of
    num_checkpoints (checkpointing changes what the backward pass recomputes
    and stores, not how many forward steps there are)."""
    from invdx.engines.fdtdx_engine import make_sim_config
    return make_sim_config(cfg, time_s=cfg.sim_time_s).time_steps_total


def fit_bytes_per_cell(checkpoints, peak_bytes, cells):
    """Least-squares peak_bytes = intercept + slope * C, then divide by the
    (constant, across one sweep) cell count to get bytes/cell.

    Returns None if fewer than 2 (C, peak_bytes) pairs are available (memory
    unavailable on this platform, or every row OOM'd/errored before it could
    read a peak) — an honest "not enough data", never a guessed fit.
    Residuals are `measured - predicted` at each fitted point, in bytes, so
    the fit's quality is visible next to its coefficients, not hidden behind
    an R^2 summary.
    """
    xs = [float(c) for c, p in zip(checkpoints, peak_bytes) if p is not None]
    ys = [float(p) for p in peak_bytes if p is not None]
    if len(xs) < 2:
        return None
    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    A = np.vstack([np.ones_like(xs_a), xs_a]).T
    (intercept, slope), *_ = np.linalg.lstsq(A, ys_a, rcond=None)
    fitted = intercept + slope * xs_a
    residuals = (ys_a - fitted).tolist()
    return {
        "intercept_bytes": float(intercept),
        "slope_bytes_per_checkpoint": float(slope),
        "base_bytes_per_cell": float(intercept) / cells,
        "bytes_per_cell_per_checkpoint": float(slope) / cells,
        "fit_C_values": xs,
        "fit_peak_bytes": ys,
        "residuals_bytes": residuals,
        "n_points": len(xs),
        "note": ("valid only across fit_C_values (" + str(min(xs)) + "-"
                + str(max(xs)) + "); extrapolation outside that range may "
                "rank candidates but must not be used to rule one out"),
    }


def measure_one(cfg, p_in, C, p0, beta, repeat, warmup):
    """One row: build the C-specific scene, time forward and value_and_grad,
    read the memory high-water mark. Never raises — failures land in the
    returned dict's error_type/error_stage instead of aborting the sweep."""
    import jax

    row = {"C": int(C), "n_cells": n_cells(cfg), "steps": n_steps(cfg),
           "forward_s": None, "vg_s": None, "ratio": None,
           "peak_bytes": None, "bytes_limit": None,
           "forward_s_all": None, "vg_s_all": None,
           "error_type": None, "error_stage": None, "error_msg": None}

    jax.clear_caches()   # see module docstring: drops compile cache only,
                         # does NOT reset the allocator's peak-bytes counter
    try:
        vg_fn, _objects, _arrays, _params0, _device, value_fn = \
            pvgc.make_ce_value_and_grad(cfg, p_in, num_checkpoints=C)
    except Exception as e:
        row["error_type"], row["error_stage"] = type(e).__name__, "build"
        row["error_msg"] = str(e)[:500]
        return row

    try:
        if warmup:
            jax.block_until_ready(value_fn(p0, beta))
        times = []
        for _ in range(repeat):
            t0 = time.time()
            out = value_fn(p0, beta)
            jax.block_until_ready(out)
            times.append(time.time() - t0)
        row["forward_s_all"] = times
        row["forward_s"] = float(np.mean(times))
    except Exception as e:
        row["error_type"], row["error_stage"] = type(e).__name__, "forward"
        row["error_msg"] = str(e)[:500]
        return row

    try:
        if warmup:
            jax.block_until_ready(vg_fn(p0, beta))
        times = []
        for _ in range(repeat):
            t0 = time.time()
            out = vg_fn(p0, beta)
            jax.block_until_ready(out)
            times.append(time.time() - t0)
        row["vg_s_all"] = times
        row["vg_s"] = float(np.mean(times))
        row["ratio"] = (row["vg_s"] / row["forward_s"]
                       if row["forward_s"] is not None else None)
    except Exception as e:
        row["error_type"], row["error_stage"] = type(e).__name__, "vg"
        row["error_msg"] = str(e)[:500]
        # forward succeeded, vg did not: keep the forward numbers, leave
        # vg_s/ratio blank rather than dropping the whole row

    dev = jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
    except Exception:
        stats = None
    if stats:
        row["peak_bytes"] = stats.get("peak_bytes_in_use")
        row["bytes_limit"] = stats.get("bytes_limit")
    return row


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in rows:
            w.writerow([r[k] if r[k] is not None else "" for k in CSV_FIELDS])


def main():
    sys.stdout.reconfigure(line_buffering=True)

    p = base_parser(__doc__)
    p.add_argument("--checkpoints", default="1,2,5,10,20",
                   metavar="C1,C2,...",
                   help="fdtdx num_checkpoints values to sweep, low-C first")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeats per C, after warmup (mean goes in the CSV; "
                        "every repeat's raw time is kept in results.json)")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="run value_fn/vg_fn once and discard before timing, "
                        "to exclude JIT trace+compile from the wall-clock")
    args = p.parse_args()

    checkpoints = [int(c) for c in args.checkpoints.split(",")]

    cfg = make_config(args)
    comparable = anchor_comparable(cfg)
    if not comparable:
        print("[sweep] WARNING: cfg does not match the anchor condition "
              f"{ANCHOR_CONDITION} — this run's peak_bytes numbers are NOT "
              "comparable to the existing 5.93/11.25/15.42 GB anchors "
              "(docs/m1-optimize.md), see results.json anchor_comparable",
              flush=True)

    d = start_run(cfg, args, "checkpoint-sweep")

    import jax
    import jax.numpy as jnp

    p_in_c, _azimuth_sign, _slope = pvgc.calibrated_beam(cfg)
    n_vox = pvgc.n_design_voxels(cfg)
    p0 = jnp.full((n_vox, 1, 1), 0.5, dtype=jnp.float32)
    beta = jnp.asarray(float(cfg.beta_schedule[0]), dtype=jnp.float32)

    rows = []
    for C in checkpoints:
        print(f"[sweep] C={C} ...", flush=True)
        row = measure_one(cfg, [p_in_c], C, p0, beta, args.repeat, args.warmup)
        if row["error_type"]:
            print(f"[sweep] C={C}: {row['error_stage']} failed — "
                  f"{row['error_type']}: {row['error_msg']}", flush=True)
        else:
            print(f"[sweep] C={C}: forward {row['forward_s']:.4g}s  "
                  f"vg {row['vg_s']:.4g}s  ratio {row['ratio']:.3g}  "
                  f"peak_bytes {row['peak_bytes']}", flush=True)
        rows.append(row)

    write_csv(os.path.join(d, "sweep.csv"), rows)

    cells = n_cells(cfg)
    fit = fit_bytes_per_cell(checkpoints, [r["peak_bytes"] for r in rows],
                             cells)
    memory_available = any(r["peak_bytes"] is not None for r in rows)

    results = {
        "cfg": asdict(cfg),
        "checkpoints": checkpoints,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "n_cells": cells,
        "steps": n_steps(cfg),
        "anchor_condition": ANCHOR_CONDITION,
        "anchor_comparable": comparable,
        "anchor_note": (
            "cfg matches the exact condition (spacing_um=0.020, "
            "design_grid_per_um=50, sim_time_s=0.8e-12, theta_deg=10) the "
            "existing memory anchors (C=10/20/28 -> 5.93/11.25/15.42 GB) "
            "were measured at — this run's C=10/C=20 peak_bytes rows are a "
            "direct external check on those numbers." if comparable else
            "cfg does NOT match the anchor condition "
            f"{ANCHOR_CONDITION} (this run's cfg: " +
            ", ".join(f"{k}={getattr(cfg, k)}" for k in ANCHOR_CONDITION) +
            ") — none of this run's numbers may be compared against the "
            "existing 5.93/11.25/15.42 GB anchors; they are only valid "
            "against each other and other runs at this same (non-anchor) "
            "condition."),
        "rows": rows,
        "fit": fit,
        "memory_stats_available": memory_available,
        "memory_stats_note": (
            "peak_bytes/bytes_limit come from "
            "jax.local_devices()[0].memory_stats(); this is None on the CPU "
            "backend (confirmed empirically, jax 0.11.0), so those fields "
            "are null here, not 0 or a guess. See module docstring for the "
            "high-water-mark caveat when sweeping multiple C in one process."
            if not memory_available else
            "peak_bytes is a process-wide high-water mark, not reset "
            "between C values (see module docstring)."),
        "quasi_2d_only": ("cfg.n_y_cells = " + str(cfg.n_y_cells) + ": every "
                          "number here is from a quasi-2D (thin periodic-y) "
                          "scene; true 3D bytes/cell and time/checkpoint are "
                          "not validated by this script."),
    }
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"[sweep] done -> {d}")


if __name__ == "__main__":
    main()
