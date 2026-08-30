#!/usr/bin/env python
"""Inverse-design driver: one full optimization round on the grating_coupler coupler.

One `fdtdx.Device` over the design window replaces the run-length-encoded
teeth, the CE of `grating_coupler.characterize` is rebuilt as a traced jnp expression,
and Adam walks the latent density through the beta annealing schedule. The
run directory it leaves behind is exactly what script 07 already knows how to
re-measure (design_rho.npy + config.json), so the optimization and its
independent verification stay separate programs.

  python scripts/15_grating_coupler_optimize.py --tag opt --gradcheck       # ~13 h GPU
  python scripts/15_grating_coupler_optimize.py --tag smoke --iters 4 \
      --set sim_time_s=0.3e-12                                   # ~10 min
  python scripts/15_grating_coupler_optimize.py --resume runs/<dir>         # after a kill

Defaults carry the reference recipe (20 nm grid / 20 nm design pixels /
0.8 ps / theta = 10 deg): the grid must divide t_si and the design pixel
exactly or the Device silently snaps to a different geometry than the
measurement chain, and theta = 10 is the only grating_coupler configuration with a
cross-validated baseline to falsify the result against. Every value is still
`--set`-overridable.

Cost model (measured on a 24 GB workstation GPU): one checkpointed
value_and_grad is ~20x a forward run, so ~19 min/iteration at the defaults.
Launch detached:
  setsid nohup make coupler-opt GPU=0 > runs/coupler-opt.log 2>&1 &
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

from invdx import optimize, runio
from invdx.cli import base_parser, apply_overrides, start_run
from invdx.fab import measure
from invdx.problems import grating_coupler
from invdx.richardson_fd import richardson_fd_check

# Reference-recipe defaults (rationale in this module's docstring); --set overrides any of them
RECIPE = dict(spacing_um=0.020, sim_time_s=0.8e-12, theta_deg=10.0)
DESIGN_GRID_PER_UM = 50

GRADCHECK_H = 0.05
GRADCHECK_TOL = 0.05
GRADCHECK_K = 3
# Only voxels whose gradient is at least this fraction of the largest are
# eligible for the finite-difference comparison — see `gradcheck`.
GRADCHECK_MIN_REL_GRAD = 0.05


# --------------------------------------------------------------------------
# provenance: a run must be re-runnable from its own directory
# --------------------------------------------------------------------------


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip()
    except Exception as e:                      # provenance never kills a run
        return f"<unavailable: {e}>"


def write_provenance(run_dir):
    """Append cmd.txt (the exact command) and an env.txt section (git HEAD,
    GPUs, uv.lock digest). Appended, not overwritten: a resumed run is a new
    invocation of the same experiment and both commands matter."""
    if not runio.am_master():
        return
    root = _repo_root()
    argv = list(sys.argv)
    argv[0] = os.path.relpath(os.path.abspath(argv[0]), root)
    cmd = "uv run python " + shlex.join(argv)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={gpus} " + cmd
    with open(os.path.join(run_dir, "cmd.txt"), "a") as f:
        f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} (cwd: repo root)\n")
        f.write(cmd + "\n")

    lock = os.path.join(root, "uv.lock")
    digest = "<missing>"
    if os.path.exists(lock):
        with open(lock, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(run_dir, "env.txt"), "a") as f:
        f.write(f"\n---- coupler-opt provenance {time.strftime('%Y-%m-%d %H:%M:%S')} ----\n")
        f.write(f"git HEAD: {_sh(['git', '-C', root, 'rev-parse', 'HEAD'])}\n")
        f.write(f"git dirty: {bool(_sh(['git', '-C', root, 'status', '--porcelain']))}\n")
        f.write(f"uv.lock sha256: {digest}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {gpus}\n")
        f.write(_sh(["nvidia-smi", "-L"]) + "\n")


# --------------------------------------------------------------------------


def make_config(args):
    """Fresh reference-recipe config, or the exact config of the run being
    resumed."""
    if args.resume:
        with open(os.path.join(args.resume, "config.json")) as f:
            stored = json.load(f)
        cfg = grating_coupler.GratingCouplerConfig(**stored)
        # asdict() -> JSON turns the tuple schedule into a list; beta_for_iter
        # accepts both, but the config must round-trip to its own type
        cfg.beta_schedule = tuple(cfg.beta_schedule)
    else:
        cfg = grating_coupler.GratingCouplerConfig(**RECIPE)
        cfg.design_grid_per_um = DESIGN_GRID_PER_UM
    cfg = apply_overrides(cfg, args)
    if getattr(args, "design_2d", False):
        grating_coupler.assert_design_grid_snaps_2d(
            cfg, allow_t_si_snap=args.allow_t_si_snap)
    else:
        grating_coupler.assert_design_grid_snaps(cfg)
    return cfg


def initial_density(cfg, args):
    """(rho0, description) — the starting design profile."""
    n = grating_coupler.n_design_voxels(cfg)
    if args.init == "grating":
        teeth = grating_coupler.uniform_grating_teeth(cfg, args.init_period,
                                           args.init_duty)
        rho = grating_coupler.rasterize_teeth(cfg, teeth)
        desc = (f"uniform grating P={args.init_period} um duty={args.init_duty}"
                f" rasterized onto {n} design pixels ({len(teeth)} teeth)")
    elif args.init == "uniform":
        rho = np.full(n, float(args.init_value))
        desc = f"uniform density {args.init_value}"
    else:
        if not args.init_file:
            raise SystemExit("--init file needs --init-file PATH")
        rho = np.load(args.init_file).astype(float).ravel()
        desc = f"file {args.init_file}"
    if rho.size != n:
        raise SystemExit(f"initial design has {rho.size} pixels, the config "
                         f"needs {n} (L_design={cfg.L_design}, "
                         f"design_grid_per_um={cfg.design_grid_per_um})")
    return rho, desc


def initial_density_2d(cfg, args):
    """(rho0 (nx, ny), description) — the 2D twin of `initial_density`
    (--design-2d path only; the 1D function stays untouched).

    --init grating rasterizes the 1D cross-validated grating and extrudes it
    uniformly along y (np.tile) — a y-uniform START inside a y-capable scene,
    so iteration 0 is comparable against the 1D arms while the optimiser is
    free to break y-uniformity from iteration 1 on. --init file accepts a 2D
    (nx, ny) array, a flat nx*ny vector, or a 1D (nx,) design to extrude
    (warm start from a 1D arm).
    """
    nx, ny = grating_coupler.design_shape_2d(cfg)
    if args.init == "grating":
        teeth = grating_coupler.uniform_grating_teeth(cfg, args.init_period,
                                           args.init_duty)
        rho1d = grating_coupler.rasterize_teeth(cfg, teeth)
        rho = np.tile(rho1d[:, None], (1, ny))
        desc = (f"uniform grating P={args.init_period} um duty="
                f"{args.init_duty} rasterized onto {nx} design pixels "
                f"({len(teeth)} teeth), extruded uniformly onto {ny} y "
                f"pixels")
    elif args.init == "uniform":
        rho = np.full((nx, ny), float(args.init_value))
        desc = f"uniform density {args.init_value} on ({nx}, {ny})"
    else:
        if not args.init_file:
            raise SystemExit("--init file needs --init-file PATH")
        raw = np.load(args.init_file).astype(float)
        if raw.shape == (nx, ny):
            rho = raw
        elif raw.size == nx * ny:
            rho = raw.reshape(nx, ny)
        elif raw.ndim == 1 and raw.size == nx:
            rho = np.tile(raw[:, None], (1, ny))
        else:
            raise SystemExit(
                f"--init-file {args.init_file} has shape {raw.shape}; the "
                f"2D config needs ({nx}, {ny}), a flat {nx * ny} vector, or "
                f"a 1D ({nx},) design to extrude (L_design={cfg.L_design}, "
                f"L_design_y={cfg.L_design_y}, "
                f"design_grid_per_um={cfg.design_grid_per_um})")
        desc = f"file {args.init_file} -> ({nx}, {ny})"
    return rho, desc


def gradcheck(vg_fn, value_fn, p, beta, seed=0):
    """Richardson-extrapolated central finite differences on a few voxels
    before spending 13 GPU-hours.

    Three things this deliberately does NOT do naively:

    * It evaluates at clip(p, h, 1-h), h = GRADCHECK_H (the larger of the two
      step sizes below). A binary initial design sits exactly on the box
      edges, where a central difference silently becomes one-sided and
      reports a spurious factor-2 error. The shifted point is a legitimate
      place to check the chain rule, and it is the point all four evaluations
      below use.
    * It only samples voxels whose gradient is within a factor 1/0.05 of the
      largest. Deep inside a wide tooth the tanh projection saturates and the
      derivative is genuinely ~1e-3 of the peak; the finite difference there
      is a subtraction of two nearly equal float32 numbers, so its own noise
      floor (~1e-7 x |CE| / 2h) swamps the signal and the "relative error"
      measures the noise, not the adjoint. That mechanism is specific to
      voxels BELOW the eligibility floor. For an eligible voxel, the error a
      single h=0.05 central difference reports is dominated by truncation,
      not float32 cancellation: diagnosed 2026-08-17 on a production-scale
      (theta=10, sim_time=0.8e-12) run, halving h cut the discrepancy ~4x —
      the O(h^2) signature of a central difference — and 3 of 8 eligible
      voxels failed the 5% tolerance on this truncation error alone while the
      Richardson extrapolate below agreed with the adjoint to 0.01%.
    * It Richardson-extrapolates instead of trusting a single h: FD(h) and
      FD(h/2) are both computed and combined as
      FD_R = (4*FD(h/2) - FD(h)) / 3, which cancels the leading O(h^2) term
      and leaves O(h^4). `rel_err` compares FD_R — not the raw FD(h) — against
      the adjoint. `fd_consistency` = |FD(h) - FD(h/2)| / |FD_R| is reported
      alongside it: a large fd_consistency with FD_R still matching the
      adjoint means the single-h finite difference had not converged, not
      that the adjoint is wrong.
    """
    import jax.numpy as jnp

    base = np.clip(np.asarray(p, dtype=float), GRADCHECK_H, 1 - GRADCHECK_H)
    beta_j = jnp.asarray(beta, dtype=jnp.float32)
    f0, grad = vg_fn(jnp.asarray(base, dtype=jnp.float32), beta_j)
    if isinstance(f0, tuple):        # w_s11 > 0: vg_fn returns (loss, aux)
        f0 = f0[0]
    g = np.asarray(grad)

    mag = np.abs(g).ravel()
    eligible = np.flatnonzero(mag >= GRADCHECK_MIN_REL_GRAD * mag.max())
    if eligible.size < GRADCHECK_K:      # degenerate gradient: take the top K
        eligible = np.argsort(-mag)[:GRADCHECK_K]
    print(f"[gradcheck] {eligible.size}/{mag.size} voxels carry a gradient "
          f"above {GRADCHECK_MIN_REL_GRAD:.0%} of the peak ({mag.max():.3e}); "
          f"sampling {GRADCHECK_K} of them")

    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(eligible, size=GRADCHECK_K, replace=False)
    checks, worst = [], 0.0
    for fi in flat_idx:
        idx = np.unravel_index(fi, base.shape)

        def evaluate(sign, hh, idx=idx):
            pert = base.copy()
            pert[idx] += sign * hh
            return float(value_fn(jnp.asarray(pert, dtype=jnp.float32), beta_j))

        rc = richardson_fd_check(evaluate, GRADCHECK_H, g[idx])
        fd, fd_h, fd_h2 = rc["fd"], rc["fd_h"], rc["fd_h2"]
        rel, fd_consistency = rc["rel_err"], rc["fd_consistency"]
        worst = max(worst, rel)
        checks.append({"idx": [int(i) for i in idx], "adjoint": float(g[idx]),
                       "fd": float(fd), "fd_h": float(fd_h),
                       "fd_h2": float(fd_h2), "rel_err": float(rel),
                       "fd_consistency": float(fd_consistency)})
        print(f"[gradcheck] voxel {idx[0]:4d}: adjoint {float(g[idx]):+.6e}  "
              f"FD_R {fd:+.6e}  rel err {rel:.2%}  "
              f"fd_consistency {fd_consistency:.2%}")
    return {"f0": float(f0), "worst_rel_err": worst, "checks": checks,
            "beta": float(beta), "h": GRADCHECK_H,
            "grad_max": float(mag.max()), "n_eligible": int(eligible.size),
            "n_voxels": int(mag.size)}


# --------------------------------------------------------------------------
# reversible gradient, behind --rev-k: the same recorder instrument that the
# memory sweep in scripts/19 measures. The four helpers below are
# byte-identical in construction to their scripts/19 originals; they are
# duplicated here on purpose so that finished sweep's code stays untouched.
# --------------------------------------------------------------------------


def n_time_steps(cfg):
    """Total FDTD steps of this config — same derivation as scripts/19."""
    from invdx.engines.fdtdx_engine import make_sim_config
    return make_sim_config(cfg, time_s=cfg.sim_time_s).time_steps_total


def k_nyquist(cfg, T):
    """floor(optical period / dt / 2): the largest reconstruction stride K
    that still samples the optical carrier at Nyquist (worked example:
    K_Nyq = 11 at 0.10 um spacing, T = 4196 steps). dt scales with the grid
    (CFL), so K_Nyq doubles when the spacing halves."""
    dt = cfg.sim_time_s / T
    period_s = cfg.lam_c * 1e-6 / 299792458.0
    return int(period_s / dt / 2)


def expected_latent_steps(T, K):
    """LinearReconstructEveryK's latent length, re-derived independently
    (scripts/19; fdtdx time_filter.py:170-174): every K-th step from 0 plus
    the final step T-1 if the stride missed it."""
    steps = list(range(0, T, K))
    if steps[-1] != T - 1:
        steps.append(T - 1)
    return len(steps)


def make_reversible_gc(K, store_dtype_name):
    """GradientConfig(method="reversible") with the same recorder stack as
    scripts/19's make_reversible_gc: LinearReconstructEveryK(k=K), then
    DtypeConversion for non-fp32 storage — the conversion is what lands in
    the store; decompression converts back to fp32 BEFORE interpolation."""
    import jax.numpy as jnp
    import fdtdx

    modules = [fdtdx.LinearReconstructEveryK(k=K)]
    if jnp.dtype(store_dtype_name) != jnp.dtype(jnp.float32):
        modules.append(fdtdx.DtypeConversion(dtype=jnp.dtype(store_dtype_name)))
    return fdtdx.GradientConfig(method="reversible",
                                recorder=fdtdx.Recorder(modules=modules))


def reversible_witness(arrays, cfg, T, K, store_dtype_name, wg_width_um):
    """Structural witness that the reversible recorder is REALLY attached
    with the requested (K, dtype) — scripts/19's recorder_effect_check for
    the 3d scene, applied to the scene this script actually optimizes.

    Byte-exact store prediction (matched on GPU for every configuration the
    memory sweep measured): 6 PML interfaces, each a 1-cell-thick full
    cross-section slice, E and H, 3 components; ny mirrors build_scene_3d's
    `wg_width + 3.0 + 2*dpml`. A checkpointed build must FAIL this witness
    (recording_state is None) — that negative control runs in the CPU
    self-test and in tests/test_reversible_optimize.py.

    Returns (ok, details); the caller aborts the run on failure.
    """
    import jax.numpy as jnp

    details = {"failures": [], "data_arrays": {}}
    rs = arrays.recording_state
    if rs is None:
        details["failures"].append(
            "recording_state is None (no Recorder attached — this is the "
            "checkpointed/no-gradient layout)")
        return False, details

    nx = int(round(cfg.cell_x / cfg.spacing_um))
    nz = int(round(cfg.cell_z / cfg.spacing_um))
    ny = int(round((wg_width_um + 3.0 + 2 * cfg.dpml) / cfg.spacing_um))
    exp_latent = expected_latent_steps(T, K)
    exp_dtype = jnp.dtype(store_dtype_name)
    per_step = 2 * 3 * 2 * (ny * nz + nx * nz + nx * ny) * exp_dtype.itemsize
    exp_total = exp_latent * per_step

    keys = sorted(rs.data.keys())
    e_keys = [k for k in keys if k.endswith("_E")]
    h_keys = [k for k in keys if k.endswith("_H")]
    if not (len(keys) == 12 and len(e_keys) == 6 and len(h_keys) == 6):
        details["failures"].append(
            f"expected 12 interface arrays (6 PML x E,H; full 3D), got "
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
                f"(T={T}, K={K})")
        if jnp.dtype(v.dtype) != exp_dtype:
            details["failures"].append(
                f"{k}: stored dtype {v.dtype} != requested {exp_dtype.name}")
    details["store_bytes_actual"] = total
    details["store_bytes_pred"] = int(exp_total)
    details["latent_steps_expected"] = int(exp_latent)
    if total != exp_total:
        details["failures"].append(
            f"total store {total} B != predicted {exp_total} B "
            f"(byte-exact match required)")
    return len(details["failures"]) == 0, details


def main():
    # a 13-hour detached run whose log only appears at the end is a run you
    # cannot supervise; block buffering is the default once stdout is a file
    sys.stdout.reconfigure(line_buffering=True)

    p = base_parser(__doc__)
    p.add_argument("--iters", type=int, default=40,
                   help="total iterations (also the beta-schedule denominator)")
    p.add_argument("--lr", type=float, default=0.05, help="Adam learning rate")
    p.add_argument("--checkpoints", type=int, default=20,
                   help="fdtdx gradient checkpoints (measured sweet spot: 20 "
                        "= 11.3 GB peak, 20x forward; 40 OOMs a 24 GB card)")
    p.add_argument("--init", choices=("grating", "uniform", "file"),
                   default="grating")
    p.add_argument("--init-period", type=float, default=0.575,
                   help="--init grating: period (um)")
    p.add_argument("--init-duty", type=float, default=0.5,
                   help="--init grating: duty cycle")
    p.add_argument("--init-value", type=float, default=0.5,
                   help="--init uniform: constant latent density")
    p.add_argument("--init-file", default=None,
                   help="--init file: .npy design vector")
    p.add_argument("--lams", default=None, metavar="L1,L2,...",
                   help="FOM wavelengths in um (default: cfg.lam_c alone); "
                        "several are aggregated with the smooth minimum")
    p.add_argument("--gradcheck", action="store_true",
                   help="finite-difference the adjoint before optimizing and "
                        "abort if it deviates by more than 5%%")
    p.add_argument("--resume", default=None, metavar="RUNDIR",
                   help="continue the optimization in an existing run dir")
    p.add_argument("--finalize-only", action="store_true",
                   help="run no optimizer iterations: load the checkpoint's "
                        "state (its true on-disk beta, not a recomputed one) "
                        "and just redo the finalization tail (design_rho*.npy, "
                        "results.json). Requires --resume; recovers a design "
                        "from an interrupted run without perturbing it.")
    p.add_argument("--run-dir", default=None, metavar="DIR",
                   help="use this exact run dir instead of a timestamped one "
                        "(requeue-safe batch jobs)")
    p.add_argument("--time-budget-h", type=float, default=None,
                   help="stop before an iteration that would exceed this "
                        "wall-clock budget")
    p.add_argument("--design-2d", action="store_true",
                   help="2D free-form path: xi(x,y) Device in the "
                        "REAL 3D scene (build_scene_design_3d), ConicFilter2D"
                        ", plane-overlap CE. Default off = the 1D path, "
                        "byte-identical to before this flag existed. A "
                        "--resume of a 2D run must pass this flag again.")
    p.add_argument("--allow-t-si-snap", action="store_true",
                   help="--design-2d only: accept a t_si that snaps on the "
                        "grid (coarse-grid chain verification; see "
                        "assert_design_grid_snaps_2d)")
    p.add_argument("--wg-width", type=float, default=10.0,
                   help="--design-2d only: waveguide width W (um) of the 3D "
                        "scene")
    p.add_argument("--rev-k", type=int, default=None, metavar="K",
                   help="--design-2d only: use the reversible gradient "
                        "instead of checkpointing, storing every K-th "
                        "boundary step "
                        "(--checkpoints becomes inert). Refused if "
                        "K > K_Nyquist (aliased gradient); loud warning if "
                        "K > K_Nyquist/2. Default off = the checkpointed "
                        "path, byte-identical to before this flag existed. "
                        "A --resume of a --rev-k run must pass this flag "
                        "again.")
    p.add_argument("--rev-dtype", choices=("float16", "float32"),
                   default="float16",
                   help="--rev-k boundary-store dtype (measured with fp16: "
                        "B_rev = 358.1 B/cell on the 3D scene)")
    p.add_argument("--y-uniform", action="store_true",
                   help="--design-2d only: freeze the y degree of freedom "
                        "— the loss sees the y-mean of the "
                        "latent broadcast back over y, and the gradient is "
                        "its exact chain-rule pullback (the y-averaged "
                        "gradient). Everything else runs the unmodified 2D "
                        "path. A --resume of a --y-uniform run must pass "
                        "this flag again.")
    p.add_argument("--verify-lams", default=None, metavar="LO,HI,N",
                   help="after optimizing, measure a dense CE spectrum of the "
                        "binarized design through the ordinary measurement "
                        "chain (two extra runs)")
    p.add_argument("--no-final-check", action="store_true",
                   help="skip the binarized-design re-measurement")
    args = p.parse_args()

    if args.design_2d and args.verify_lams:
        raise SystemExit(
            "--design-2d cannot take --verify-lams: the dense-spectrum "
            "re-measurement is teeth-based (characterize_spectrum) and a 2D "
            "pattern has no teeth representation. Drop --verify-lams.")
    if args.y_uniform and not args.design_2d:
        raise SystemExit("--y-uniform is a constraint on the 2D latent and "
                         "needs --design-2d")
    if args.rev_k is not None and not args.design_2d:
        raise SystemExit("--rev-k (reversible gradient) is wired for the "
                         "--design-2d path only; the 1D path keeps its "
                         "measured checkpointed configuration")
    if args.rev_k is not None and args.rev_k < 1:
        raise SystemExit(f"--rev-k must be >= 1, got {args.rev_k}")

    cfg = make_config(args)
    if args.resume:
        d = args.resume
        if not os.path.exists(os.path.join(d, optimize.STATE_FILE)):
            raise SystemExit(f"{d} has no {optimize.STATE_FILE} to resume from")
        # the config actually in force wins, so --set on a resume is recorded
        runio.save_json(os.path.join(d, "config.json"), asdict(cfg))
        print(f"[run] resuming -> {d}")
    else:
        d = start_run(cfg, args, "coupler-opt", run_dir=args.run_dir)
    write_provenance(d)

    lams = ([float(v) for v in args.lams.split(",")] if args.lams
            else [cfg.lam_c])

    # ---- incident beam power: design-independent, measured once ----
    t0 = time.time()
    if args.design_2d:
        # 3D empty-cell run; the 3D chain has no tilt calibration
        # (build_scene_3d's azimuth_sign stays at its default, exactly as in
        # characterize_3d), so P_in and the design scene share one geometry
        p_in_c, azimuth_sign = None, 1.0
        p_in = grating_coupler.beam_power_3d(cfg, lams, wg_width_um=args.wg_width)
        print(f"[beam] 3D P_in = {['%.4g' % v for v in p_in]} at "
              f"{lams} um  ({time.time() - t0:.0f}s)")
    else:
        p_in_c, azimuth_sign, slope = grating_coupler.calibrated_beam(cfg)
        if len(lams) == 1 and abs(lams[0] - cfg.lam_c) < 1e-12:
            p_in = [p_in_c]
        else:
            p_in = grating_coupler.beam_power_spectrum(cfg, lams,
                                            azimuth_sign=azimuth_sign)
        print(f"[beam] P_in = {['%.4g' % v for v in p_in]} at "
              f"{lams} um  (tilt slope {slope:+.3f} rad/um, "
              f"{time.time() - t0:.0f}s)")

    # ---- differentiable FOM ----
    rev_info = None
    if args.design_2d:
        rev_gc = None
        if args.rev_k is not None:
            T = n_time_steps(cfg)
            k_nyq = k_nyquist(cfg, T)
            if args.rev_k > k_nyq:
                raise SystemExit(
                    f"--rev-k {args.rev_k} > K_Nyquist = {k_nyq} at this "
                    f"grid (T = {T} steps): the boundary store cannot "
                    f"represent the optical carrier and the reconstructed "
                    f"gradient is aliased. Refusing to optimize.")
            if args.rev_k > k_nyq // 2:
                print(f"[reversible] WARNING: K = {args.rev_k} > K_Nyq/2 = "
                      f"{k_nyq // 2} — inside the Nyquist limit but without "
                      f"margin; the recommended operating point is "
                      f"K <= K_Nyq/2")
            rev_gc = make_reversible_gc(args.rev_k, args.rev_dtype)
        vg_fn, objects, arrays, params0, device, value_fn = \
            grating_coupler.make_ce_value_and_grad_3d(
                cfg, p_in, num_checkpoints=args.checkpoints, lams=lams,
                wg_width_um=args.wg_width, gradient_config=rev_gc,
                allow_t_si_snap=args.allow_t_si_snap)
        if args.rev_k is not None:
            ok, details = reversible_witness(arrays, cfg, T, args.rev_k,
                                             args.rev_dtype, args.wg_width)
            if not ok:
                raise SystemExit("[reversible] witness FAIL: "
                                 + "; ".join(details["failures"]))
            print(f"[reversible] witness PASS: K={args.rev_k} "
                  f"dtype={args.rev_dtype} T={T} K_Nyq={k_nyq} "
                  f"latent_steps={details['latent_steps_expected']} "
                  f"store={details['store_bytes_actual']:,} B (byte-exact)")
            rev_info = {"K": args.rev_k, "dtype": args.rev_dtype,
                        "T": int(T), "K_nyquist": int(k_nyq),
                        "latent_steps": details["latent_steps_expected"],
                        "store_bytes": details["store_bytes_actual"],
                        "note": "num_checkpoints is inert on this run "
                                "(gradient_config overrides it)"}
        nx, ny = grating_coupler.design_shape_2d(cfg)
        n_vox = nx * ny
        grad_desc = (f"reversible K={args.rev_k} {args.rev_dtype}"
                     if args.rev_k is not None else
                     f"{args.checkpoints} checkpoints")
        print(f"[design] 2D free-form ({nx}, {ny}) = {n_vox} voxels, latent "
              f"shape {tuple(params0[device.name].shape)}, {grad_desc}")
        rho0, init_desc = initial_density_2d(cfg, args)
        p0 = rho0.reshape(nx, ny, 1)
        if args.y_uniform:
            # freeze the y degree of freedom. Enforced as a linear
            # reparametrization IN THE FORWARD PASS — the physics
            # only ever sees S(p), S = broadcast(mean over y): a hard
            # constraint, like the 1D path where the scene is built from a
            # y-extruded design. S is symmetric idempotent, so the chain rule
            # gives grad = S(grad at S(p)) — the y-averaged gradient: the two
            # candidate implementations (latent broadcast / gradient
            # averaging) coincide. p0 is projected too, so the latent stays
            # bit-exactly y-uniform for any --init.
            import jax.numpy as jnp

            def _y_mean(a):
                return jnp.broadcast_to(a.mean(axis=1, keepdims=True),
                                        a.shape)

            base_vg, base_value = vg_fn, value_fn

            def vg_fn(p, beta):
                v, g = base_vg(_y_mean(p), beta)
                return v, _y_mean(g)

            def value_fn(p, beta):
                return base_value(_y_mean(p), beta)

            p0 = np.broadcast_to(p0.mean(axis=1, keepdims=True),
                                 p0.shape).copy()
            print(f"[design] --y-uniform: {nx} effective variables "
                  f"(y frozen; latent stays ({nx}, {ny}, 1))")
    else:
        vg_fn, objects, arrays, params0, device, value_fn = \
            grating_coupler.make_ce_value_and_grad(cfg, p_in,
                                        num_checkpoints=args.checkpoints,
                                        lams=lams)
        n_vox = grating_coupler.n_design_voxels(cfg)
        print(f"[design] {n_vox} voxels, latent shape "
              f"{tuple(params0[device.name].shape)}, "
              f"{args.checkpoints} checkpoints")

        rho0, init_desc = initial_density(cfg, args)
        p0 = rho0.reshape(n_vox, 1, 1)

    gc_res = None
    if args.finalize_only:
        if not args.resume:
            raise SystemExit("--finalize-only requires --resume RUNDIR")
        state = optimize.load_state(d)
        state.stop_reason = "finalize-only"
        print(f"[opt] finalize-only: using checkpointed iteration "
              f"{state.iteration} beta={state.beta} (no iterations run)")
    else:
        if args.gradcheck:
            gc_res = gradcheck(vg_fn, value_fn, p0,
                               optimize.beta_for_iter(cfg, 0, args.iters),
                               seed=cfg.seed)
            if gc_res["worst_rel_err"] > GRADCHECK_TOL:
                runio.save_json(os.path.join(d, "results.json"),
                                {"gradcheck": gc_res, "aborted": "gradcheck"})
                raise SystemExit(
                    f"[gradcheck] FAIL: worst rel err "
                    f"{gc_res['worst_rel_err']:.2%} > {GRADCHECK_TOL:.0%} — the "
                    f"adjoint is not trustworthy, do not optimize (check "
                    f"spacing_um vs design_grid_per_um first)")
            print(f"[gradcheck] PASS (worst {gc_res['worst_rel_err']:.2%} < "
                  f"{GRADCHECK_TOL:.0%})")

        # ---- the loop ----
        def on_iter(row):
            extra = ""
            if not np.isnan(row["s11_dB"]):
                extra = (f"S11 {row['s11_dB']:.2f} dB  "
                         f"FOM {row['fom']:.4g}  ")
            print(f"[opt] iter {row['iter']:3d}  beta {row['beta']:5.0f}  "
                  f"CE {row['CE_dB']:7.3f} dB  {extra}"
                  f"|g| {row['grad_norm']:.3e}  "
                  f"{row['wall_s']:.0f}s", flush=True)

        state = optimize.run_loop(
            vg_fn, p0, cfg, n_iters=args.iters, lr=args.lr, run_dir=d,
            resume=bool(args.resume), on_iter=on_iter,
            time_budget_h=args.time_budget_h)
        print(f"[opt] stopped after iteration {state.iteration} "
              f"({state.stop_reason})")

    # ---- designs on disk (design_rho.npy is what script 07 reads;
    #      design_rho_2d.npy is the 2D twin, which script 07 CANNOT
    #      re-measure: that would need a 3D verifier, which does not
    #      exist yet) ----
    if args.design_2d:
        p_final = state.p
        if args.y_uniform:
            lat = np.asarray(p_final, dtype=float)
            y_dev = float(np.max(np.abs(lat - lat.mean(axis=1,
                                                       keepdims=True))))
            y_std_max = float(lat.std(axis=1).max())
            print(f"[y-uniform] latent max |p - mean_y(p)| = {y_dev:.3e}, "
                  f"max std over y = {y_std_max:.3e} (must be 0)")
            # write the design the loss actually saw: S(p), not p
            p_final = np.broadcast_to(lat.mean(axis=1, keepdims=True),
                                      lat.shape)
        rho_cont = grating_coupler.rho_from_params_2d(device, p_final, state.beta)
        rho_bin = (rho_cont > 0.5).astype(float)
        np.save(os.path.join(d, "design_rho_2d_cont.npy"), rho_cont)
        np.save(os.path.join(d, "design_rho_2d.npy"), rho_bin)
    else:
        rho_cont = grating_coupler.rho_from_params(device, state.p, state.beta)
        rho_bin = (rho_cont > 0.5).astype(float)
        np.save(os.path.join(d, "design_rho_cont.npy"), rho_cont)
        np.save(os.path.join(d, "design_rho.npy"), rho_bin)

    hist = []
    last_s11 = last_fom = None
    csv_path = os.path.join(d, optimize.HISTORY_FILE)
    if os.path.exists(csv_path):
        rows = np.genfromtxt(csv_path, delimiter=",", names=True)
        hist = np.atleast_1d(rows["CE"]).tolist()
        if rows.dtype.names and "s11_dB" in rows.dtype.names:
            v = float(np.atleast_1d(rows["s11_dB"])[-1])
            last_s11 = None if np.isnan(v) else v
            last_fom = float(np.atleast_1d(rows["fom"])[-1])

    if args.design_2d:
        # fab/measure.py is exact for 1D designs ONLY (its own docstring);
        # a 2D min-feature measure (imageruler / disk morphology) is not
        # implemented. Record the gap, don't fake a number.
        solid = void = None
        teeth = []
    else:
        solid, void = measure.min_feature_1d(rho_bin, cfg.design_grid_per_um)
        teeth = grating_coupler.profile_teeth(cfg, rho_bin)
    res = {
        "history": hist,
        "CE_dB": (float(10 * np.log10(max(hist[-1], 1e-15))) if hist
                  else None),
        "S11_dB": last_s11,
        "fom": last_fom,
        "w_s11": cfg.w_s11,
        "iters_done": state.iteration + 1,
        "n_iters": args.iters,
        "stop_reason": state.stop_reason,
        "beta_final": state.beta,
        "lams_um": lams,
        "P_in": p_in,
        "init": init_desc,
        "lr": args.lr,
        "num_checkpoints": args.checkpoints,
        "linewidth": ({"min_solid_um": solid, "min_void_um": void,
                       "n_teeth": len(teeth)} if not args.design_2d else
                      {"note": "2D min-feature measurement not implemented "
                               "(fab/measure.py is 1D-only)"}),
    }
    if args.design_2d:
        res["design_2d"] = {
            "shape": list(grating_coupler.design_shape_2d(cfg)),
            "L_design_y": cfg.L_design_y,
            "wg_width_um": args.wg_width,
            "allow_t_si_snap": bool(args.allow_t_si_snap),
        }
        if args.y_uniform:
            # key only present when the constraint is on: a run WITHOUT the
            # flag keeps its results.json byte-identical to before
            res["design_2d"]["y_uniform"] = {
                "latent_max_dev_from_y_mean": y_dev,
                "latent_max_std_y": y_std_max,
            }
        if rev_info is not None:
            # same contract: key only present when --rev-k is on
            res["reversible"] = rev_info
    if gc_res is not None:
        res["gradcheck"] = gc_res
    elif os.path.exists(os.path.join(d, "results.json")):
        # a resumed segment rewrites results.json; the gradient check that
        # licensed this run happened in the first segment and must survive
        with open(os.path.join(d, "results.json")) as f:
            prev = json.load(f)
        if "gradcheck" in prev:
            res["gradcheck"] = prev["gradcheck"]

    # ---- binarization gap through the ORDINARY measurement chain ----
    if args.design_2d:
        if not args.no_final_check:
            print("[final] 2D design: the teeth-based re-measurement cannot "
                  "represent a free-form pattern — skipped (independent "
                  "re-measurement would need a 3D Device-placement verifier, "
                  "which does not exist here)")
        res["binarization_gap_note"] = (
            "not measured: no independent re-measurement path for a 2D "
            "design")
    elif not args.no_final_check:
        meas = grating_coupler.characterize(cfg, teeth, p_in=p_in_c,
                                 azimuth_sign=azimuth_sign)
        res["binarized"] = meas
        if res["CE_dB"] is not None:
            res["binarization_gap_dB"] = float(res["CE_dB"] - meas["CE_dB"])
            if cfg.w_s11 > 0:
                res["binarization_gap_note"] = (
                    "continuous CE is wg-reciprocal; gap embeds ~0.06 dB "
                    "reciprocity mismatch")
            print(f"[final] continuous {res['CE_dB']:.3f} dB -> binarized "
                  f"{meas['CE_dB']:.3f} dB "
                  f"(gap {res['binarization_gap_dB']:.3f} dB)")
        else:
            print(f"[final] binarized design: {meas['CE_dB']:.3f} dB")

    if args.verify_lams:
        lo, hi, n = args.verify_lams.split(",")
        vlams = list(np.linspace(float(lo), float(hi), int(n)))
        spec = grating_coupler.characterize_spectrum(cfg, teeth, vlams,
                                          azimuth_sign=azimuth_sign)
        res["spectrum"] = spec["spectrum"]
        res["peak"] = max(spec["spectrum"], key=lambda r: r["CE_dB"])
        bw, lam_lo, lam_hi, note = grating_coupler.bandwidth_3db(spec["spectrum"])
        res["bandwidth_3db"] = {"bw_nm": bw * 1e3, "lam_lo_um": lam_lo,
                                "lam_hi_um": lam_hi, "note": note}
        print(f"[verify] ridge peak {res['peak']['CE_dB']:.2f} dB @ "
              f"{res['peak']['lam_um']:.3f} um, 3 dB bandwidth "
              f"{bw * 1e3:.1f} nm")

    runio.save_json(os.path.join(d, "results.json"), res)
    print(f"[done] {d}/results.json")
    if args.design_2d:
        print("[next] no independent verifier for a 2D design exists here "
              "(script 07 is teeth-based/quasi-2D)")
    else:
        print(f"[next] uv run python scripts/07_grating_coupler_verify_design.py --run {d} "
              f"--s11 --lam-lo 1.26 --lam-hi 1.36 --n-lam 11")
    return 0


if __name__ == "__main__":
    sys.exit(main())
