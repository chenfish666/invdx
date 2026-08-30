#!/usr/bin/env python
"""Real memory peak of method="reversible" + LinearReconstructEveryK
(+ optional fp16 recorder storage), against the checkpointed model that
scripts/18_checkpoint_sweep.py measures.

Every acceptance criterion this script applies -- the alpha bands, the P1/P2
endpoint difference tests, the too-good-to-be-true veto -- was written down
BEFORE the sweep was run and is hard-coded below. A number that lands in band
is therefore a prediction that held, not a band drawn around the result.

Structure, CLI and sweep.csv columns deliberately mirror
scripts/18_checkpoint_sweep.py so the two CSVs can be laid side by side:
shared columns keep the same names/semantics; the sweep variable is
(K, store_dtype) instead of C, and recorder-specific columns are appended.

  # GPU, real numbers, anchor-comparable defaults (MUST be a single visible
  # GPU: with >1 device init_sharded_dict pads the latent axis and the store
  # prediction is off — see the multi-device warning the sweep prints)
  uv run python scripts/19_reversible_sweep.py --tag rev-sweep \\
      --k-values 1,8,16,24 --dtypes float32,float16 --repeat 1

  # CPU smoke (NOT anchor-comparable; peak_bytes is null on CPU)
  JAX_PLATFORMS=cpu uv run python scripts/19_reversible_sweep.py \\
      --tag cpu-smoke --k-values 1,8 --dtypes float32,float16 \\
      --set spacing_um=0.05 --set design_grid_per_um=20 --set t_si=0.2 \\
      --set L_design=6.0 --set sim_time_s=0.01e-12 --set theta_deg=0

  # negative control: build the CHECKPOINTED path and demand that the
  # recorder effect-check FAILs on it (exit 0 iff it does fail)
  JAX_PLATFORMS=cpu uv run python scripts/19_reversible_sweep.py \\
      --negative-control --set spacing_um=0.05 ... (same tiny overrides)

SCOPE — read before trusting a number out of this script:

* Reversible autodiff needs a Recorder; this script builds
  Recorder(modules=[LinearReconstructEveryK(k=K)] (+ DtypeConversion(dtype)
  for non-float32 storage)) and hands it to grating_coupler.make_ce_value_and_grad via
  the gradient_config parameter switch (grating_coupler.build_scene_design). The
  simulation dtype stays float32 ALWAYS — fp16 exists only inside the
  recorder storage (the initialization.py:335 complex trap).
* EFFECT CHECK (not the measured quantity): before any timing, the placed
  arrays.recording_state is checked against an INDEPENDENT reconstruction of
  what the recorder must allocate — key count, latent length (re-derived
  from T and K by this script, mirroring time_filter.py:170-174), stored
  dtype, and byte-exact total size predicted from cfg geometry alone. A row
  whose check fails records error_type=RecorderCheckFAIL and is never timed.
  --negative-control runs the same check on a checkpointed build and exits 0
  only if the check correctly FAILs there.
* peak_bytes is jax's peak_bytes_in_use — the same instrument scripts/18
  uses, with the same high-water-mark caveat (docstring there): within one
  process a later row can only inherit-or-exceed earlier peaks. Rows are
  therefore swept in ascending predicted-store order by default, and each row
  gets a clean_peak flag (first-to-reach-a-new-high rule). CPU runs record
  peak_bytes as null, never 0. nvidia-smi is never used: it reports the
  allocator's reserved pool, not the working set the cost model is built on.
* Quasi-2D only BY DEFAULT: that scene has 4 PML interfaces (y is periodic
  — no y faces); the 6-face store formula is for full 3D and does NOT apply
  there. The per-step store is
      2(E,H) x 3 components x (2*ny*nz + 2*nx*ny) x bytes_per_scalar.
* --scene 3d swaps in the full-3D wiring, reused verbatim
  (grating_coupler.make_ce_value_and_grad_3d / build_scene_design_3d — no parallel
  builder). Full 3D has 6 PML interfaces; the per-step store is
      2(E,H) x 3 components x 2*(ny*nz + nx*nz + nx*ny) x bytes_per_scalar
  and the effect check expects 12 interface arrays (6 x E,H). The quasi-2D
  default path is byte-identical to before this switch existed.

  Process hygiene on the 3D path (the high-water mark is per-process and not
  resettable):
    1. P_in process:   --scene 3d --emit-p-in-3d          (writes p_in_3d.json)
    2. sweep process:  --scene 3d --p-in-3d <value> ...   (6 reversible rows)
    3. control:        --scene 3d --checkpointed-control 20 --p-in-3d <value>
  The checkpointed control row's predicted peak lands mid-sweep, so running
  it inside the sweep process would guarantee a contaminated reading; the
  P_in empty-cell forward would likewise sit under the smallest row. Omitting
  --p-in-3d computes P_in in-process (fine on CPU where peak is null; loud
  warning on GPU).
* An OOM or build failure is recorded as its own row (error_type/
  error_stage set), never silently dropped.
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
from invdx.problems import grating_coupler

CSV_FIELDS = ["K", "store_dtype", "forward_s", "vg_s", "ratio", "peak_bytes",
              "bytes_limit", "n_cells", "steps", "latent_steps",
              "store_bytes_pred", "store_bytes_actual", "recorder_check",
              "clean_peak", "grad_l2"]

# Same anchor condition as scripts/18: a bare invocation lands on the
# identical ~1.944M-cell / 1311-step scene the checkpointed model
# peak = N x (300.19 + 120.03*C) was measured at, so the two sweeps compare
# like for like. (Duplicated from scripts/18 because scripts are not an
# importable package; keep in sync manually.)
ANCHOR_CONDITION = dict(spacing_um=0.020, design_grid_per_um=50,
                        sim_time_s=0.05e-12, theta_deg=10.0)

# --scene 3d: the same full-3D scene the 2D free-form optimization driver
# (scripts/15 --design-2d) runs on, which is the whole point of the
# side-by-side checkpointed control. wg_width is a CLI arg (10.0 default).
ANCHOR_CONDITION_3D = dict(spacing_um=0.10, design_grid_per_um=10,
                           sim_time_s=0.8e-12, theta_deg=10.0,
                           L_design_y=10.0)

# Acceptance bands, fixed before the sweep ran and recorded next to the
# computed numbers in results.json so the mechanical pass/fail is visible;
# the final judgment still belongs to a write-up read against these bands.
ALPHA_BAND = (0.9, 2.2)          # quasi-2D scene: wide prior
ALPHA_BAND_3D = (1.8, 2.5)       # full 3D: transfer test around alpha ~ 2.11
BASE_FLOOR_BYTES_PER_CELL = 300.0   # veto trip line (both scenes)


def alpha_band(scene):
    return ALPHA_BAND_3D if scene == "3d" else ALPHA_BAND


def anchor_comparable(cfg, scene="quasi2d"):
    """True iff cfg matches the scene's anchor condition exactly."""
    anchor = ANCHOR_CONDITION_3D if scene == "3d" else ANCHOR_CONDITION
    for k, v in anchor.items():
        cur = getattr(cfg, k)
        if isinstance(v, float):
            if not math.isclose(cur, v, rel_tol=1e-9):
                return False
        elif cur != v:
            return False
    return True


def make_config(args):
    scene = getattr(args, "scene", "quasi2d")
    anchor = ANCHOR_CONDITION_3D if scene == "3d" else ANCHOR_CONDITION
    cfg = grating_coupler.GratingCouplerConfig(**anchor)
    cfg = apply_overrides(cfg, args)
    if scene == "3d":
        grating_coupler.assert_design_grid_snaps_2d(
            cfg, allow_t_si_snap=args.allow_t_si_snap)
    else:
        grating_coupler.assert_design_grid_snaps(cfg)
    return cfg


def grid_dims(cfg, scene="quasi2d", wg_width=10.0):
    """(nx, ny, nz) of the scene — same rounding as grating_coupler's own n_cells.

    3d: y is real; cell_y mirrors build_scene_3d's `wg_width + 3.0 + 2*dpml`
    (a structural probe of the placed scene confirms (200, 150, 97) at the
    anchor condition)."""
    nx = int(round(cfg.cell_x / cfg.spacing_um))
    nz = int(round(cfg.cell_z / cfg.spacing_um))
    if scene == "3d":
        cell_y = wg_width + 3.0 + 2 * cfg.dpml
        return nx, int(round(cell_y / cfg.spacing_um)), nz
    return nx, int(cfg.n_y_cells), nz


def n_cells(cfg, scene="quasi2d", wg_width=10.0):
    nx, ny, nz = grid_dims(cfg, scene, wg_width)
    return nx * ny * nz


def n_steps(cfg):
    from invdx.engines.fdtdx_engine import make_sim_config
    return make_sim_config(cfg, time_s=cfg.sim_time_s).time_steps_total


def expected_latent_steps(T, K):
    """Independent re-derivation of LinearReconstructEveryK's latent array
    length (time_filter.py:170-174): every K-th step from 0, plus the final
    step T-1 if the stride missed it. The effect check compares the placed
    recorder against THIS number, not against the recorder's own field."""
    steps = list(range(0, T, K))
    if steps[-1] != T - 1:
        steps.append(T - 1)
    return len(steps)


def per_step_store_bytes(cfg, store_dtype_name, scene="quasi2d",
                         wg_width=10.0):
    """Recorder bytes per stored latent step, from cfg geometry alone.

    quasi2d — 4 PML interfaces (x-, x+, z-, z+; y is periodic, verified
    structurally against the placed arrays):
        2 fields x 3 components x (2*ny*nz + 2*nx*ny) x bytes_per_scalar
    3d — 6 PML interfaces, each a 1-cell-thick full cross-section slice
    (also verified structurally against the placed arrays):
        2 fields x 3 components x 2*(ny*nz + nx*nz + nx*ny) x bytes_per_scalar
    """
    import jax.numpy as jnp
    nx, ny, nz = grid_dims(cfg, scene, wg_width)
    if scene == "3d":
        scalars = 2 * 3 * 2 * (ny * nz + nx * nz + nx * ny)
    else:
        scalars = 2 * 3 * (2 * ny * nz + 2 * nx * ny)
    return scalars * jnp.dtype(store_dtype_name).itemsize


def store_bytes_pred(cfg, T, K, store_dtype_name, scene="quasi2d",
                     wg_width=10.0):
    return expected_latent_steps(T, K) * per_step_store_bytes(
        cfg, store_dtype_name, scene, wg_width)


def make_reversible_gc(K, store_dtype_name):
    """GradientConfig(method="reversible") with the recorder stack this sweep
    measures: LinearReconstructEveryK(k=K), then DtypeConversion for non-fp32
    storage
    (modules run in order at compress time, so the conversion is what lands
    in the store; at decompress time the reversed order converts back to
    fp32 BEFORE the linear interpolation — interpolation stays fp32)."""
    import jax.numpy as jnp
    import fdtdx

    modules = [fdtdx.LinearReconstructEveryK(k=K)]
    if jnp.dtype(store_dtype_name) != jnp.dtype(jnp.float32):
        modules.append(fdtdx.DtypeConversion(dtype=jnp.dtype(store_dtype_name)))
    return fdtdx.GradientConfig(method="reversible",
                                recorder=fdtdx.Recorder(modules=modules))


def recorder_effect_check(arrays, cfg, T, K, store_dtype_name,
                          scene="quasi2d", wg_width=10.0):
    """Is the reversible recorder REALLY attached, with the predicted store?

    Checks the placed ArrayContainer's recording_state (created — or not —
    inside fdtdx place_objects, initialization.py:768-786) against numbers
    this script derives independently from cfg and (T, K, dtype). This is
    deliberately NOT the measured quantity (peak_bytes): it is a structural
    witness that the thing whose memory we claim to measure exists.

    Returns (ok, details) — details lists every failure explicitly and, on
    success, the actual store layout. A checkpointed build must FAIL here
    (recording_state is None); --negative-control asserts exactly that.
    """
    import jax.numpy as jnp

    details = {"failures": [], "data_arrays": {}}
    rs = arrays.recording_state
    if rs is None:
        details["failures"].append(
            "recording_state is None (no Recorder attached — this is the "
            "checkpointed/no-gradient layout)")
        return False, details

    exp_latent = expected_latent_steps(T, K)
    exp_dtype = jnp.dtype(store_dtype_name)
    exp_total = int(store_bytes_pred(cfg, T, K, store_dtype_name,
                                     scene, wg_width))
    n_faces = 6 if scene == "3d" else 4
    face_note = ("6 PML x E,H; full 3D" if scene == "3d"
                 else "4 PML x E,H; y is periodic")

    keys = sorted(rs.data.keys())
    e_keys = [k for k in keys if k.endswith("_E")]
    h_keys = [k for k in keys if k.endswith("_H")]
    if not (len(keys) == 2 * n_faces and len(e_keys) == n_faces
            and len(h_keys) == n_faces):
        details["failures"].append(
            f"expected {2 * n_faces} interface arrays ({face_note}), got "
            f"{len(keys)}: {keys}")

    total = 0
    for k in keys:
        v = rs.data[k]
        details["data_arrays"][k] = {"shape": list(v.shape),
                                     "dtype": str(v.dtype),
                                     "nbytes": int(v.nbytes)}
        total += int(v.nbytes)
        if v.shape[0] != exp_latent:
            details["failures"].append(
                f"{k}: latent axis {v.shape[0]} != expected {exp_latent} "
                f"(T={T}, K={K}; a value above the expectation also catches "
                f"multi-device padding)")
        if jnp.dtype(v.dtype) != exp_dtype:
            details["failures"].append(
                f"{k}: stored dtype {v.dtype} != requested {exp_dtype.name}")
    details["store_bytes_actual"] = total
    details["store_bytes_pred"] = exp_total
    details["latent_steps_expected"] = exp_latent
    if total != exp_total:
        details["failures"].append(
            f"total store {total} B != predicted {exp_total} B "
            f"(byte-exact match required)")
    return len(details["failures"]) == 0, details


def measure_one(cfg, p_in, K, store_dtype_name, p0, beta, repeat, warmup,
                scene="quasi2d", wg_width=10.0, allow_t_si_snap=False):
    """One row: build the (K, dtype) reversible scene, run the effect check,
    time forward and value_and_grad, read the memory high-water mark. Never
    raises — failures land in error_type/error_stage."""
    import jax
    import jax.numpy as jnp

    T = n_steps(cfg)
    row = {"K": int(K), "store_dtype": store_dtype_name,
           "n_cells": n_cells(cfg, scene, wg_width), "steps": T,
           "latent_steps": expected_latent_steps(T, K),
           "store_bytes_pred": int(store_bytes_pred(cfg, T, K,
                                                    store_dtype_name,
                                                    scene, wg_width)),
           "store_bytes_actual": None, "recorder_check": None,
           "forward_s": None, "vg_s": None, "ratio": None,
           "peak_bytes": None, "bytes_limit": None, "clean_peak": None,
           "grad_l2": None, "forward_s_all": None, "vg_s_all": None,
           "check_details": None,
           "error_type": None, "error_stage": None, "error_msg": None}

    jax.clear_caches()   # drops compile cache only; the allocator's
                         # peak-bytes counter is NOT reset (scripts/18 note)
    try:
        gc = make_reversible_gc(K, store_dtype_name)
        if scene == "3d":
            vg_fn, _objects, arrays, _params0, _device, value_fn = \
                grating_coupler.make_ce_value_and_grad_3d(
                    cfg, p_in, gradient_config=gc, wg_width_um=wg_width,
                    allow_t_si_snap=allow_t_si_snap)
        else:
            vg_fn, _objects, arrays, _params0, _device, value_fn = \
                grating_coupler.make_ce_value_and_grad(cfg, p_in, gradient_config=gc)
    except Exception as e:
        row["error_type"], row["error_stage"] = type(e).__name__, "build"
        row["error_msg"] = str(e)[:500]
        return row

    ok, details = recorder_effect_check(arrays, cfg, T, K, store_dtype_name,
                                        scene, wg_width)
    row["recorder_check"] = "PASS" if ok else "FAIL"
    row["check_details"] = details
    row["store_bytes_actual"] = details.get("store_bytes_actual")
    if not ok:
        row["error_type"] = "RecorderCheckFAIL"
        row["error_stage"] = "effect-check"
        row["error_msg"] = "; ".join(details["failures"])[:500]
        return row   # never time a scene whose recorder is not proven live

    return _time_and_read_peak(row, value_fn, vg_fn, p0, beta, repeat,
                               warmup)


def _time_and_read_peak(row, value_fn, vg_fn, p0, beta, repeat, warmup):
    """Timing + peak reading shared by the sweep rows and the checkpointed
    control row (identical instrument, identical order of operations)."""
    import jax
    import jax.numpy as jnp

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
        out = None
        for _ in range(repeat):
            t0 = time.time()
            out = vg_fn(p0, beta)
            jax.block_until_ready(out)
            times.append(time.time() - t0)
        row["vg_s_all"] = times
        row["vg_s"] = float(np.mean(times))
        row["ratio"] = (row["vg_s"] / row["forward_s"]
                        if row["forward_s"] else None)
        grad = out[1]
        row["grad_l2"] = float(jnp.linalg.norm(grad))
    except Exception as e:
        row["error_type"], row["error_stage"] = type(e).__name__, "vg"
        row["error_msg"] = str(e)[:500]
        # keep the forward numbers, leave vg blank

    dev = jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
    except Exception:
        stats = None
    if stats:
        row["peak_bytes"] = stats.get("peak_bytes_in_use")
        row["bytes_limit"] = stats.get("bytes_limit")
    return row


def mark_clean_peaks(rows):
    """First-to-reach-a-new-high rule: a row's
    peak reading is clean iff it strictly exceeds every earlier reading in
    this process by more than 1 MB. Rows without a reading get None."""
    high = -1.0
    for r in rows:
        p = r.get("peak_bytes")
        if p is None:
            r["clean_peak"] = None
            continue
        r["clean_peak"] = bool(p > high + 1e6)
        high = max(high, float(p))


def fit_alpha(rows, cells, band=ALPHA_BAND):
    """Least-squares peak = a + alpha * store_bytes_pred over CLEAN rows.

    Returns None with a reason if fewer than 3 clean (store, peak) pairs
    exist — an honest "not enough data", never a guessed fit. B_rev = a / N
    is the reversible base term this experiment exists to learn."""
    pts = [(float(r["store_bytes_pred"]), float(r["peak_bytes"]))
           for r in rows if r.get("clean_peak") and r["peak_bytes"] is not None]
    if len(pts) < 3:
        return {"ok": False, "reason": f"only {len(pts)} clean rows (<3)",
                "n_points": len(pts)}
    xs = np.asarray([p[0] for p in pts])
    ys = np.asarray([p[1] for p in pts])
    A = np.vstack([np.ones_like(xs), xs]).T
    (a, alpha), *_ = np.linalg.lstsq(A, ys, rcond=None)
    fitted = a + alpha * xs
    return {"ok": True, "intercept_bytes": float(a), "alpha": float(alpha),
            "base_bytes_per_cell": float(a) / cells,
            "alpha_band": list(band),
            "alpha_in_band": bool(band[0] <= alpha <= band[1]),
            "fit_store_bytes": xs.tolist(), "fit_peak_bytes": ys.tolist(),
            "residuals_bytes": (ys - fitted).tolist(), "n_points": len(pts)}


def _delta(hi, lo, band):
    d = hi["peak_bytes"] - lo["peak_bytes"]
    ds = hi["store_bytes_pred"] - lo["store_bytes_pred"]
    return {"delta_peak_bytes": int(d), "delta_store_pred_bytes": int(ds),
            "band_bytes": [band[0] * ds, band[1] * ds],
            "in_band": bool(band[0] * ds <= d <= band[1] * ds)}


def _find_row(rows, K, dt):
    for r in rows:
        if r["K"] == K and r["store_dtype"] == dt and \
                r["peak_bytes"] is not None:
            return r
    return None


def endpoint_deltas(rows):
    """P1/P2: the zero-free-parameter difference tests between the
    well-separated endpoint configs, as fixed before the sweep ran. Uses
    whatever of the three endpoint rows exist and have peak readings; never
    invents a number."""
    out = {}
    k1f32, k1f16, k24f16 = _find_row(rows, 1, "float32"), \
        _find_row(rows, 1, "float16"), _find_row(rows, 24, "float16")
    if k1f32 and k24f16:
        out["P1"] = _delta(k1f32, k24f16, ALPHA_BAND)
    if k1f32 and k1f16:
        out["P2"] = _delta(k1f32, k1f16, ALPHA_BAND)
    return out


def endpoint_deltas_3d(rows):
    """The 3D scene's P1/P2. P1 pairs the largest-store config THAT HAS A
    READING against K24-fp16, preference order K1-fp32 > K1-fp16 > K8-fp32
    (fixed a priori: availability is decided by OOM, never by the values).
    P2 is the same-K dtype differential at K=8, chosen because both configs
    sit far from OOM."""
    out = {}
    k24f16 = _find_row(rows, 24, "float16")
    if k24f16:
        for K, dt, name in ((1, "float32", "K1f32-K24f16"),
                            (1, "float16", "K1f16-K24f16"),
                            (8, "float32", "K8f32-K24f16")):
            hi = _find_row(rows, K, dt)
            if hi:
                out["P1"] = dict(_delta(hi, k24f16, ALPHA_BAND_3D),
                                 pair=name)
                break
    k8f32, k8f16 = _find_row(rows, 8, "float32"), _find_row(rows, 8,
                                                            "float16")
    if k8f32 and k8f16:
        out["P2"] = dict(_delta(k8f32, k8f16, ALPHA_BAND_3D),
                         pair="K8f32-K8f16")
    return out


def veto_rows(rows, cells):
    """Too-good-to-be-true veto: any measured peak below
    N x 300 B is suspicious (below the checkpointed MEASURED base term —
    prime suspect is a recorder store that never touched the measured
    device). Returns the offending rows' (K, dtype)."""
    floor = BASE_FLOOR_BYTES_PER_CELL * cells
    return [{"K": r["K"], "store_dtype": r["store_dtype"],
             "peak_bytes": r["peak_bytes"], "floor_bytes": floor}
            for r in rows if r["peak_bytes"] is not None
            and r["peak_bytes"] < floor]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in rows:
            w.writerow([r[k] if r.get(k) is not None else ""
                        for k in CSV_FIELDS])


def negative_control(cfg, d, scene="quasi2d", wg_width=10.0,
                     allow_t_si_snap=False):
    """Build the CHECKPOINTED scene (gradient_config=None → the untouched
    legacy path, through the same place_objects that would create a
    recording_state if one were configured) and demand the effect check
    FAILs on it. Exit 0 iff it does; a PASS here means the check itself is
    broken and no sweep may be trusted until it is fixed."""
    import jax
    import fdtdx

    T = n_steps(cfg)
    if scene == "3d":
        sim_config, objs, cons, _tmpl = grating_coupler.build_scene_design_3d(
            cfg, num_checkpoints=2, wg_width_um=wg_width,
            allow_t_si_snap=allow_t_si_snap)  # gradient_config omitted
    else:
        sim_config, objs, cons, _tmpl = grating_coupler.build_scene_design(
            cfg, num_checkpoints=2)   # gradient_config omitted on purpose
    key = jax.random.PRNGKey(cfg.seed)
    key, k1 = jax.random.split(key)
    _objects, arrays, _params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    ok, details = recorder_effect_check(arrays, cfg, T, K=1,
                                        store_dtype_name="float32",
                                        scene=scene, wg_width=wg_width)
    result = {
        "mode": "negative-control",
        "built": "checkpointed (gradient_config=None, num_checkpoints=2)",
        "gradient_method_in_config": sim_config.gradient_config.method,
        "check_ok": ok,
        "check_details": details,
        "expected": "check must FAIL (recording_state is None)",
        "verdict": ("NEGATIVE CONTROL OK — effect check correctly FAILED "
                    "on a recorder-less build" if not ok else
                    "NEGATIVE CONTROL BROKEN — effect check PASSED on a "
                    "recorder-less build; fix the check before measuring"),
    }
    with open(os.path.join(d, "negative_control.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[negctl] {result['verdict']}")
    for fail in details["failures"]:
        print(f"[negctl] check failure (expected): {fail}")
    return 0 if not ok else 1


def checkpointed_control(cfg, p_in, C, p0, beta, repeat, warmup,
                         scene="quasi2d", wg_width=10.0,
                         allow_t_si_snap=False):
    """The single checkpointed-C row on the SAME scene as the reversible
    sweep — always in its OWN process (its predicted peak lands mid-sweep;
    in-process it would be a contaminated reading by construction).

    Structural witness (this row's substitute for the recorder check):
    a fresh build with these exact arguments must carry
    gradient_config.method == "checkpointed" and num_checkpoints == C, and
    the MEASURED build's arrays.recording_state must be None — no recorder
    may participate in the peak this row attributes to checkpointing.
    """
    import jax

    T = n_steps(cfg)
    row = {"C": int(C), "n_cells": n_cells(cfg, scene, wg_width), "steps": T,
           "control_check": None, "check_failures": None,
           "forward_s": None, "vg_s": None, "ratio": None,
           "peak_bytes": None, "bytes_limit": None, "grad_l2": None,
           "forward_s_all": None, "vg_s_all": None,
           "error_type": None, "error_stage": None, "error_msg": None}

    failures = []
    try:
        if scene == "3d":
            sim_config, *_ = grating_coupler.build_scene_design_3d(
                cfg, num_checkpoints=C, wg_width_um=wg_width,
                allow_t_si_snap=allow_t_si_snap)
        else:
            sim_config, *_ = grating_coupler.build_scene_design(cfg, num_checkpoints=C)
        gcfg = sim_config.gradient_config
        if gcfg.method != "checkpointed":
            failures.append(f"gradient method {gcfg.method!r} != "
                            f"'checkpointed'")
        elif int(gcfg.num_checkpoints) != int(C):
            failures.append(f"num_checkpoints {gcfg.num_checkpoints} != {C}")

        if scene == "3d":
            vg_fn, _o, arrays, _p, _dev, value_fn = \
                grating_coupler.make_ce_value_and_grad_3d(
                    cfg, p_in, num_checkpoints=C, wg_width_um=wg_width,
                    allow_t_si_snap=allow_t_si_snap)
        else:
            vg_fn, _o, arrays, _p, _dev, value_fn = \
                grating_coupler.make_ce_value_and_grad(cfg, p_in, num_checkpoints=C)
    except Exception as e:
        row["error_type"], row["error_stage"] = type(e).__name__, "build"
        row["error_msg"] = str(e)[:500]
        return row

    if arrays.recording_state is not None:
        failures.append("recording_state is not None — a recorder is "
                        "attached to the supposedly checkpointed build")
    row["control_check"] = "PASS" if not failures else "FAIL"
    row["check_failures"] = failures
    if failures:
        row["error_type"] = "ControlCheckFAIL"
        row["error_stage"] = "effect-check"
        row["error_msg"] = "; ".join(failures)[:500]
        return row

    jax.clear_caches()
    return _time_and_read_peak(row, value_fn, vg_fn, p0, beta, repeat,
                               warmup)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    p = base_parser(__doc__)
    p.add_argument("--k-values", default="1,8,16,24", metavar="K1,K2,...",
                   help="LinearReconstructEveryK strides to sweep")
    p.add_argument("--dtypes", default="float32,float16",
                   metavar="D1,D2,...",
                   help="recorder STORAGE dtypes (simulation stays float32)")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeats per config after warmup (mean in the CSV, "
                        "raw times in results.json)")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="run value_fn/vg_fn once, discarded, before timing")
    p.add_argument("--order", choices=["ascending", "given"],
                   default="ascending",
                   help="ascending = sweep by predicted store size (keeps "
                        "the process-wide high-water mark honest); "
                        "given = exactly the CLI order")
    p.add_argument("--negative-control", action="store_true",
                   help="build the checkpointed path and require the "
                        "recorder effect-check to FAIL on it; exit 0 iff "
                        "it does (no sweep is run)")
    p.add_argument("--scene", choices=["quasi2d", "3d"], default="quasi2d",
                   help="quasi2d = the quasi-2D anchor scene (default, "
                        "unchanged); 3d = the full-3D "
                        "build_scene_design_3d scene")
    p.add_argument("--wg-width", type=float, default=10.0,
                   help="3d only: waveguide width, sets cell_y = w+3+2*dpml")
    p.add_argument("--allow-t-si-snap", action="store_true",
                   help="3d only: accept fdtdx snapping t_si to the grid "
                        "(0.10 um chain-verification grids need it)")
    p.add_argument("--p-in-3d", type=float, default=None,
                   help="3d only: static P_in from a previous "
                        "--emit-p-in-3d process; omitting it computes P_in "
                        "in-process, which pollutes the first row's peak "
                        "on GPU (fine on CPU)")
    p.add_argument("--emit-p-in-3d", action="store_true",
                   help="3d only: run the empty-cell beam_power_3d once, "
                        "write p_in_3d.json, exit (keeps P_in out of the "
                        "sweep process's peak)")
    p.add_argument("--checkpointed-control", type=int, default=None,
                   metavar="C",
                   help="measure the single checkpointed-C row on the same "
                        "scene instead of the sweep — ALWAYS its own "
                        "process, or the reading is contaminated")
    args = p.parse_args()

    cfg = make_config(args)
    scene, wg_width = args.scene, args.wg_width
    comparable = anchor_comparable(cfg, scene)
    if not comparable:
        anchor = ANCHOR_CONDITION_3D if scene == "3d" else ANCHOR_CONDITION
        ref = ("the 3D prediction table" if scene == "3d"
               else "the checkpointed model peak = N x (300.19 + 120.03*C) "
                    "or to the prediction table")
        print(f"[sweep] WARNING: cfg does not match the {scene} anchor "
              f"condition {anchor} — peak_bytes here is NOT comparable to "
              f"{ref}", flush=True)

    suffix = "3d" if scene == "3d" else ""
    if args.negative_control:
        name = "reversible-negctl" + suffix
    elif args.emit_p_in_3d:
        name = "reversible-pin" + suffix
    elif args.checkpointed_control is not None:
        name = "chk-control" + suffix
    else:
        name = "reversible-sweep" + suffix
    d = start_run(cfg, args, name)

    if args.negative_control:
        sys.exit(negative_control(cfg, d, scene, wg_width,
                                  args.allow_t_si_snap))

    if args.emit_p_in_3d:
        if scene != "3d":
            sys.exit("--emit-p-in-3d only makes sense with --scene 3d")
        p_in = grating_coupler.beam_power_3d(cfg, None, wg_width_um=wg_width)[0]
        out = {"lam_um": cfg.lam_c, "p_in": float(p_in),
               "wg_width_um": wg_width,
               "note": "static empty-cell P_in (beam_power_3d), measured "
                       "in its own process so the sweep's first-row peak "
                       "stays clean"}
        with open(os.path.join(d, "p_in_3d.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[p_in] {p_in:.6e} -> {os.path.join(d, 'p_in_3d.json')}")
        return

    import jax
    import jax.numpy as jnp

    if len(jax.local_devices()) > 1:
        print(f"[sweep] WARNING: {len(jax.local_devices())} devices visible "
              "— init_sharded_dict pads the latent axis to a device-count "
              "multiple; the effect check will (correctly) FAIL every row. "
              "Run with a single visible GPU.", flush=True)

    T = n_steps(cfg)
    configs = [(int(k), dt) for dt in args.dtypes.split(",")
               for k in args.k_values.split(",")]
    if args.order == "ascending":
        configs.sort(key=lambda c: store_bytes_pred(cfg, T, c[0], c[1],
                                                    scene, wg_width))

    if scene == "3d":
        if args.p_in_3d is not None:
            p_in = float(args.p_in_3d)
            p_in_source = "cli (--p-in-3d, own-process measurement)"
        else:
            print("[sweep] WARNING: no --p-in-3d given — computing P_in "
                  "in-process; on GPU this empty-cell forward sits under "
                  "the smallest rows' peaks. Fine on CPU.",
                  flush=True)
            p_in = grating_coupler.beam_power_3d(cfg, None, wg_width_um=wg_width)[0]
            p_in_source = "in-process beam_power_3d (peak-contaminating)"
        nxd, nyd = grating_coupler.design_shape_2d(cfg)
        p0 = jnp.full((nxd, nyd, 1), 0.5, dtype=jnp.float32)
    else:
        p_in_c, _azimuth_sign, _slope = grating_coupler.calibrated_beam(cfg)
        p_in = [p_in_c]
        p_in_source = "calibrated_beam (quasi-2D, unchanged)"
        n_vox = grating_coupler.n_design_voxels(cfg)
        p0 = jnp.full((n_vox, 1, 1), 0.5, dtype=jnp.float32)
    beta = jnp.asarray(float(cfg.beta_schedule[0]), dtype=jnp.float32)

    if args.checkpointed_control is not None:
        C = args.checkpointed_control
        print(f"[control] checkpointed C={C} on the {scene} scene ...",
              flush=True)
        row = checkpointed_control(cfg, p_in, C, p0, beta, args.repeat,
                                   args.warmup, scene, wg_width,
                                   args.allow_t_si_snap)
        if row["error_type"]:
            print(f"[control] {row['error_stage']} failed — "
                  f"{row['error_type']}: {row['error_msg']}", flush=True)
        else:
            print(f"[control] check={row['control_check']}  "
                  f"forward {row['forward_s']:.4g}s  vg {row['vg_s']:.4g}s  "
                  f"ratio {row['ratio']:.3g}  peak_bytes {row['peak_bytes']}",
                  flush=True)
        results = {
            "mode": "checkpointed-control", "scene": scene,
            "wg_width_um": wg_width, "cfg": asdict(cfg),
            "n_cells": n_cells(cfg, scene, wg_width), "steps": T,
            "grid_dims": list(grid_dims(cfg, scene, wg_width)),
            "anchor_comparable": comparable,
            "p_in": p_in, "p_in_source": p_in_source,
            "row": row,
            "acceptance_criteria": "acceptance criteria fixed before this control ran; "
                      "see the module docstring",
        }
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"[control] done -> {d}")
        return

    rows = []
    for K, dt in configs:
        print(f"[sweep] K={K} store_dtype={dt} "
              f"(store_pred="
              f"{int(store_bytes_pred(cfg, T, K, dt, scene, wg_width)):,} "
              f"B) ...", flush=True)
        row = measure_one(cfg, p_in, K, dt, p0, beta,
                          args.repeat, args.warmup, scene, wg_width,
                          args.allow_t_si_snap)
        if row["error_type"]:
            print(f"[sweep] K={K}/{dt}: {row['error_stage']} failed — "
                  f"{row['error_type']}: {row['error_msg']}", flush=True)
        else:
            print(f"[sweep] K={K}/{dt}: check={row['recorder_check']}  "
                  f"forward {row['forward_s']:.4g}s  vg {row['vg_s']:.4g}s  "
                  f"ratio {row['ratio']:.3g}  peak_bytes {row['peak_bytes']}",
                  flush=True)
        rows.append(row)
        mark_clean_peaks(rows)
        write_csv(os.path.join(d, "sweep.csv"), rows)

    cells = n_cells(cfg, scene, wg_width)
    fit = fit_alpha(rows, cells, band=alpha_band(scene))
    deltas = (endpoint_deltas_3d(rows) if scene == "3d"
              else endpoint_deltas(rows))
    vetoes = veto_rows(rows, cells)
    memory_available = any(r["peak_bytes"] is not None for r in rows)

    if scene == "3d":
        chk_ref = (
            "the checkpointed C=20 reference on THIS scene is a separate "
            "--checkpointed-control 20 process, not the quasi-2D model."
            if comparable else
            "NOT anchor-comparable — do not compare against the 3D "
            "prediction table")
    else:
        chk_ref = ("checkpointed model on the same scene: peak = N "
                   "x (300.19 + 120.03*C) bytes"
                   if comparable else
                   "NOT anchor-comparable — do not compare "
                   "against the checkpointed model or the "
                   "prediction table")

    results = {
        "cfg": asdict(cfg),
        "scene": scene,
        "wg_width_um": wg_width,
        "p_in": p_in,
        "p_in_source": p_in_source,
        "configs": [[k, dt] for k, dt in configs],
        "order": args.order,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "n_cells": cells,
        "steps": T,
        "grid_dims": list(grid_dims(cfg, scene, wg_width)),
        "anchor_condition": (ANCHOR_CONDITION_3D if scene == "3d"
                             else ANCHOR_CONDITION),
        "anchor_comparable": comparable,
        "checkpointed_reference": chk_ref,
        "rows": rows,
        "fit": fit,
        "endpoint_deltas_P1_P2": deltas,
        "veto_rows_below_base_floor": vetoes,
        "memory_stats_available": memory_available,
        "memory_stats_note": (
            "peak_bytes/bytes_limit from jax.local_devices()[0]"
            ".memory_stats(); None on the CPU backend — recorded as null, "
            "never 0. High-water-mark caveat: see scripts/18 docstring; "
            "clean_peak marks first-to-reach-a-new-high rows."
            if memory_available else
            "CPU backend: memory_stats() is None; peak_bytes/bytes_limit "
            "are null. This run can validate mechanics and the effect "
            "check, NEVER memory numbers."),
        "acceptance_criteria": ("3D acceptance criteria, fixed before this sweep ran; "
                   "see the module docstring" if scene == "3d"
                   else "quasi-2D acceptance criteria, fixed before this "
                        "sweep ran; see the module docstring"),
    }
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"[sweep] done -> {d}")


if __name__ == "__main__":
    main()
