#!/usr/bin/env python
"""M1 — the first real inverse-design round on the PVGC coupler.

One `fdtdx.Device` over the design window replaces the run-length-encoded
teeth, the CE of `pvgc.characterize` is rebuilt as a traced jnp expression,
and Adam walks the latent density through the beta annealing schedule. The
run directory it leaves behind is exactly what script 07 already knows how to
re-measure (design_rho.npy + config.json), so the optimization and its
independent verification stay separate programs.

  python scripts/15_pvgc_optimize.py --tag m1 --gradcheck        # ~13 h GPU
  python scripts/15_pvgc_optimize.py --tag smoke --iters 4 \
      --set sim_time_s=0.3e-12                                   # ~10 min
  python scripts/15_pvgc_optimize.py --resume runs/<dir>         # after a kill

Defaults carry the M1 recipe (20 nm grid / 20 nm design pixels / 0.8 ps /
theta = 10 deg): the grid must divide t_si and the design pixel exactly or the
Device silently snaps to a different geometry than the measurement chain, and
theta = 10 is the only pvgc configuration with a cross-validated baseline to
falsify the result against. Every value is still `--set`-overridable.

Cost model (measured, Quadro RTX 6000): one checkpointed value_and_grad is
~20x a forward run, so ~19 min/iteration at the defaults. Launch detached:
  setsid nohup make pvgc-opt GPU=0 > runs/pvgc-opt.log 2>&1 &
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
from invdx.problems import pvgc

# M1 recipe defaults (rationale in this module's docstring); --set overrides any of them
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
        f.write(f"\n---- pvgc-opt provenance {time.strftime('%Y-%m-%d %H:%M:%S')} ----\n")
        f.write(f"git HEAD: {_sh(['git', '-C', root, 'rev-parse', 'HEAD'])}\n")
        f.write(f"git dirty: {bool(_sh(['git', '-C', root, 'status', '--porcelain']))}\n")
        f.write(f"uv.lock sha256: {digest}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {gpus}\n")
        f.write(_sh(["nvidia-smi", "-L"]) + "\n")


# --------------------------------------------------------------------------


def make_config(args):
    """Fresh M1 config, or the exact config of the run being resumed."""
    if args.resume:
        with open(os.path.join(args.resume, "config.json")) as f:
            stored = json.load(f)
        cfg = pvgc.PVGCConfig(**stored)
        # asdict() -> JSON turns the tuple schedule into a list; beta_for_iter
        # accepts both, but the config must round-trip to its own type
        cfg.beta_schedule = tuple(cfg.beta_schedule)
    else:
        cfg = pvgc.PVGCConfig(**RECIPE)
        cfg.design_grid_per_um = DESIGN_GRID_PER_UM
    cfg = apply_overrides(cfg, args)
    pvgc.assert_design_grid_snaps(cfg)
    return cfg


def initial_density(cfg, args):
    """(rho0, description) — the starting design profile."""
    n = pvgc.n_design_voxels(cfg)
    if args.init == "grating":
        teeth = pvgc.uniform_grating_teeth(cfg, args.init_period,
                                           args.init_duty)
        rho = pvgc.rasterize_teeth(cfg, teeth)
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
        fd_by_h = {}
        for h in (GRADCHECK_H, GRADCHECK_H / 2):
            vals = {}
            for sign in (+1, -1):
                pert = base.copy()
                pert[idx] += sign * h
                vals[sign] = float(value_fn(
                    jnp.asarray(pert, dtype=jnp.float32), beta_j))
            fd_by_h[h] = (vals[+1] - vals[-1]) / (2 * h)
        fd_h, fd_h2 = fd_by_h[GRADCHECK_H], fd_by_h[GRADCHECK_H / 2]
        fd = (4 * fd_h2 - fd_h) / 3          # Richardson extrapolation, O(h^4)
        rel = abs(fd - g[idx]) / (abs(fd) + 1e-12)
        fd_consistency = abs(fd_h - fd_h2) / (abs(fd) + 1e-12)
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
    p.add_argument("--verify-lams", default=None, metavar="LO,HI,N",
                   help="after optimizing, measure a dense CE spectrum of the "
                        "binarized design through the ordinary measurement "
                        "chain (two extra runs)")
    p.add_argument("--no-final-check", action="store_true",
                   help="skip the binarized-design re-measurement")
    args = p.parse_args()

    cfg = make_config(args)
    if args.resume:
        d = args.resume
        if not os.path.exists(os.path.join(d, optimize.STATE_FILE)):
            raise SystemExit(f"{d} has no {optimize.STATE_FILE} to resume from")
        # the config actually in force wins, so --set on a resume is recorded
        runio.save_json(os.path.join(d, "config.json"), asdict(cfg))
        print(f"[run] resuming -> {d}")
    else:
        d = start_run(cfg, args, "pvgc-opt", run_dir=args.run_dir)
    write_provenance(d)

    lams = ([float(v) for v in args.lams.split(",")] if args.lams
            else [cfg.lam_c])

    # ---- incident beam power: design-independent, measured once ----
    t0 = time.time()
    p_in_c, azimuth_sign, slope = pvgc.calibrated_beam(cfg)
    if len(lams) == 1 and abs(lams[0] - cfg.lam_c) < 1e-12:
        p_in = [p_in_c]
    else:
        p_in = pvgc.beam_power_spectrum(cfg, lams, azimuth_sign=azimuth_sign)
    print(f"[beam] P_in = {['%.4g' % v for v in p_in]} at "
          f"{lams} um  (tilt slope {slope:+.3f} rad/um, {time.time() - t0:.0f}s)")

    # ---- differentiable FOM ----
    vg_fn, objects, arrays, params0, device, value_fn = \
        pvgc.make_ce_value_and_grad(cfg, p_in, num_checkpoints=args.checkpoints,
                                    lams=lams)
    n_vox = pvgc.n_design_voxels(cfg)
    print(f"[design] {n_vox} voxels, latent shape "
          f"{tuple(params0[device.name].shape)}, {args.checkpoints} checkpoints")

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
            print(f"[opt] iter {row['iter']:3d}  beta {row['beta']:5.0f}  "
                  f"CE {row['CE_dB']:7.3f} dB  |g| {row['grad_norm']:.3e}  "
                  f"{row['wall_s']:.0f}s", flush=True)

        state = optimize.run_loop(
            vg_fn, p0, cfg, n_iters=args.iters, lr=args.lr, run_dir=d,
            resume=bool(args.resume), on_iter=on_iter,
            time_budget_h=args.time_budget_h)
        print(f"[opt] stopped after iteration {state.iteration} "
              f"({state.stop_reason})")

    # ---- designs on disk (design_rho.npy is what script 07 reads) ----
    rho_cont = pvgc.rho_from_params(device, state.p, state.beta)
    rho_bin = (rho_cont > 0.5).astype(float)
    np.save(os.path.join(d, "design_rho_cont.npy"), rho_cont)
    np.save(os.path.join(d, "design_rho.npy"), rho_bin)

    hist = []
    csv_path = os.path.join(d, optimize.HISTORY_FILE)
    if os.path.exists(csv_path):
        rows = np.genfromtxt(csv_path, delimiter=",", names=True)
        hist = np.atleast_1d(rows["CE"]).tolist()

    solid, void = measure.min_feature_1d(rho_bin, cfg.design_grid_per_um)
    teeth = pvgc.profile_teeth(cfg, rho_bin)
    res = {
        "history": hist,
        "CE_dB": (float(10 * np.log10(max(hist[-1], 1e-15))) if hist
                  else None),
        "iters_done": state.iteration + 1,
        "n_iters": args.iters,
        "stop_reason": state.stop_reason,
        "beta_final": state.beta,
        "lams_um": lams,
        "P_in": p_in,
        "init": init_desc,
        "lr": args.lr,
        "num_checkpoints": args.checkpoints,
        "linewidth": {"min_solid_um": solid, "min_void_um": void,
                      "n_teeth": len(teeth)},
    }
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
    if not args.no_final_check:
        meas = pvgc.characterize(cfg, teeth, p_in=p_in_c,
                                 azimuth_sign=azimuth_sign)
        res["binarized"] = meas
        if res["CE_dB"] is not None:
            res["binarization_gap_dB"] = float(res["CE_dB"] - meas["CE_dB"])
            print(f"[final] continuous {res['CE_dB']:.3f} dB -> binarized "
                  f"{meas['CE_dB']:.3f} dB "
                  f"(gap {res['binarization_gap_dB']:.3f} dB)")
        else:
            print(f"[final] binarized design: {meas['CE_dB']:.3f} dB")

    if args.verify_lams:
        lo, hi, n = args.verify_lams.split(",")
        vlams = list(np.linspace(float(lo), float(hi), int(n)))
        spec = pvgc.characterize_spectrum(cfg, teeth, vlams,
                                          azimuth_sign=azimuth_sign)
        res["spectrum"] = spec["spectrum"]
        res["peak"] = max(spec["spectrum"], key=lambda r: r["CE_dB"])
        bw, lam_lo, lam_hi, note = pvgc.bandwidth_3db(spec["spectrum"])
        res["bandwidth_3db"] = {"bw_nm": bw * 1e3, "lam_lo_um": lam_lo,
                                "lam_hi_um": lam_hi, "note": note}
        print(f"[verify] ridge peak {res['peak']['CE_dB']:.2f} dB @ "
              f"{res['peak']['lam_um']:.3f} um, 3 dB bandwidth "
              f"{bw * 1e3:.1f} nm")

    runio.save_json(os.path.join(d, "results.json"), res)
    print(f"[done] {d}/results.json")
    print(f"[next] uv run python scripts/07_pvgc_verify_design.py --run {d} "
          f"--s11 --lam-lo 1.26 --lam-hi 1.36 --n-lam 11")
    return 0


if __name__ == "__main__":
    sys.exit(main())
