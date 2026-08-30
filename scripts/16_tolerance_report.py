#!/usr/bin/env python
"""Tolerance report for a finished grating_coupler optimization run (the procedure
docs/tolerance.md specifies): a per-voxel sensitivity map and a three-corner
robust-design evaluation. No re-optimization; reads only run_dir/config.json
and run_dir/opt_state.npz (the checkpoint format optimize.py already writes)
and writes into run_dir/tolerance/.

  python scripts/16_tolerance_report.py runs/<coupler-opt-dir>
  python scripts/16_tolerance_report.py runs/<dir> --lams 1.27,1.35,9

Sensitivity map (tolerance.md part 1): one backward pass of the existing
`grating_coupler.make_ce_value_and_grad` FOM, evaluated at the run's own final (p, beta)
-- cost is one empty-cell calibration run plus one checkpointed value-and-
gradient call, no re-optimization. Writes sensitivity.csv (per design voxel)
and sensitivity.png (rho profile + |dCE/drho|).

Corner evaluation (tolerance.md part 2): the SAME latent p, re-projected at
the three robust-design corners -- eroded (cfg.eta_e), nominal (cfg.eta_i,
default 0.5), dilated (cfg.eta_d), tolerance.md's own three-field definition
-- through the authoritative numpy conic-filter + tanh-projection chain
(fab.filters_np; fab.transforms.ConicFilter1D wraps the same math for the
differentiable path). Each corner is binarized (> 0.5, the same rule
scripts/15 uses for design_rho.npy) and measured with grating_coupler.characterize.
Corner table columns are exactly tolerance.md's: corner, CE_dB, bw_3db_nm,
ridge_lam_um. Bandwidth and ridge wavelength need a spectrum: with the
default single wavelength (cfg.lam_c) those two columns are left blank;
pass --lams LO,HI,N to fill them from a real characterize_spectrum sweep.

Yield_90% = Pr(CE_corner >= 0.9 * CE_nominal) is printed per tolerance.md's
definition, evaluated on the 3-corner sample -- n=3 is a fragility screen,
explicitly NOT a statistical yield estimate (also per tolerance.md).

CPU/GPU-neutral: platform is whatever JAX_PLATFORMS/CUDA_VISIBLE_DEVICES say
at process start, same as every other script here.
"""

import csv
import json
import os
import sys

import numpy as np

from invdx.cli import base_parser, apply_overrides
from invdx import optimize
from invdx.fab.filters_np import conic_filter_matrix, make_mapping
from invdx.problems import grating_coupler

OUT_SUBDIR = "tolerance"
CORNER_DEFS = (("eroded", "eta_e"), ("nominal", "eta_i"), ("dilated", "eta_d"))


def load_run(run_dir):
    """(cfg, p_flat, beta, iteration) from an existing coupler-opt run dir."""
    with open(os.path.join(run_dir, "config.json")) as f:
        stored = json.load(f)
    cfg = grating_coupler.GratingCouplerConfig(**stored)
    cfg.beta_schedule = tuple(cfg.beta_schedule)
    if not os.path.exists(os.path.join(run_dir, optimize.STATE_FILE)):
        raise SystemExit(f"{run_dir} has no {optimize.STATE_FILE} -- run "
                         f"scripts/15_grating_coupler_optimize.py first")
    state = optimize.load_state(run_dir)      # opt_state pytree not needed here
    p_flat = np.asarray(state.p, dtype=float).reshape(-1)
    return cfg, p_flat, float(state.beta), int(state.iteration)


def _design_x_um(cfg, n_vox):
    x0 = -cfg.L_design / 2
    dx = 1.0 / cfg.design_grid_per_um
    return x0 + (np.arange(n_vox) + 0.5) * dx


# --------------------------------------------------------------------------
# 1. sensitivity map
# --------------------------------------------------------------------------


def sensitivity_map(cfg, p_flat, beta, p_in_c, num_checkpoints=20):
    """One backward pass -> dCE/dp per design voxel, plus the nominal rho
    the same latent maps to (what design_rho_cont.npy would hold)."""
    import jax.numpy as jnp

    n_vox = grating_coupler.n_design_voxels(cfg)
    p3 = jnp.asarray(p_flat.reshape(n_vox, 1, 1), dtype=jnp.float32)
    beta_j = jnp.asarray(beta, dtype=jnp.float32)

    vg_fn, objects, arrays, params0, device, value_fn = \
        grating_coupler.make_ce_value_and_grad(cfg, p_in_c, num_checkpoints=num_checkpoints)
    loss, grad = vg_fn(p3, beta_j)
    if isinstance(loss, tuple):      # w_s11 > 0: vg_fn returns (loss, aux);
        loss = loss[0]               # "CE"/dCE below are then FOM/dFOM
    dCE_dp = -np.asarray(grad, dtype=float).reshape(-1)   # loss = -CE (module doc)
    rho = grating_coupler.rho_from_params(device, p3, beta)
    return {"x_um": _design_x_um(cfg, n_vox), "rho": rho, "dCE_drho": dCE_dp,
            "CE": float(-loss)}


def write_sensitivity_csv(path, sens):
    grad_abs = np.abs(sens["dCE_drho"])
    peak = grad_abs.max()
    grad_abs_norm = grad_abs / peak if peak > 0 else grad_abs
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["voxel_idx", "x_um", "rho", "dCE_drho", "abs_grad_norm"])
        for i in range(sens["x_um"].size):
            w.writerow([i, f"{sens['x_um'][i]:.6f}", f"{sens['rho'][i]:.6f}",
                       f"{sens['dCE_drho'][i]:.6e}", f"{grad_abs_norm[i]:.6f}"])


def plot_sensitivity(sens, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from invdx.viz.plots import C1, C2, INK, INK2, GRID

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), dpi=150,
                                   sharex=True)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)

    ax1.set_title("Per-voxel sensitivity", color=INK, fontsize=12, loc="left")
    ax1.plot(sens["x_um"], sens["rho"], color=C1, linewidth=1.5)
    ax1.set_ylabel("rho (design density)", color=INK2, fontsize=10)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(sens["x_um"], np.abs(sens["dCE_drho"]), color=C2, linewidth=1.5)
    ax2.set_xlabel("x (um)", color=INK2, fontsize=10)
    ax2.set_ylabel("|dCE/drho|", color=INK2, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------
# 2. corner evaluation
# --------------------------------------------------------------------------


def project_corner(cfg, p_flat, eta, beta):
    """Latent p -> binary rho at projection threshold `eta` (same conic
    filter + tanh projection fab.transforms.ConicFilter1D wraps for the
    differentiable path, evaluated here host-side via fab.filters_np)."""
    n_vox = p_flat.size
    W = conic_filter_matrix(n_vox, cfg.filter_radius, cfg.design_grid_per_um)
    rho_cont = np.asarray(make_mapping(W)(p_flat, eta, beta))
    return (rho_cont > 0.5).astype(float)


def corner_table(cfg, p_flat, beta, p_in_c, azimuth_sign, lams_spec=None,
                 seed=0):
    rows = []
    for name, eta_attr in CORNER_DEFS:
        eta = getattr(cfg, eta_attr)
        rho_bin = project_corner(cfg, p_flat, eta, beta)
        teeth = grating_coupler.profile_teeth(cfg, rho_bin)
        row = {"corner": name}
        if lams_spec:
            spec = grating_coupler.characterize_spectrum(cfg, teeth, lams_spec,
                                              azimuth_sign=azimuth_sign,
                                              seed=seed)["spectrum"]
            peak = max(spec, key=lambda r: r["CE_dB"])
            bw, lam_lo, lam_hi, note = grating_coupler.bandwidth_3db(spec)
            row["CE_dB"] = peak["CE_dB"]
            row["bw_3db_nm"] = bw * 1e3
            row["ridge_lam_um"] = peak["lam_um"]
            if note:
                print(f"[tolerance] corner {name}: {note}")
        else:
            meas = grating_coupler.characterize(cfg, teeth, p_in=p_in_c,
                                     azimuth_sign=azimuth_sign, seed=seed)
            row["CE_dB"] = meas["CE_dB"]
            row["bw_3db_nm"] = None
            row["ridge_lam_um"] = None
        rows.append(row)
    return rows


def write_corners_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corner", "CE_dB", "bw_3db_nm", "ridge_lam_um"])
        for r in rows:
            w.writerow([r["corner"], f"{r['CE_dB']:.4f}",
                       "" if r["bw_3db_nm"] is None else f"{r['bw_3db_nm']:.4f}",
                       "" if r["ridge_lam_um"] is None else f"{r['ridge_lam_um']:.6f}"])


def yield_90(rows):
    """Pr(CE_corner >= 0.9 * CE_nominal) over the corner sample
    (docs/tolerance.md Reporting conventions)."""
    ce_nominal = 10 ** (next(r["CE_dB"] for r in rows
                             if r["corner"] == "nominal") / 10.0)
    thr = 0.9 * ce_nominal
    passes = sum(1 for r in rows if 10 ** (r["CE_dB"] / 10.0) >= thr)
    return passes / len(rows), passes, len(rows)


# --------------------------------------------------------------------------


def main():
    sys.stdout.reconfigure(line_buffering=True)

    p = base_parser(__doc__)
    p.add_argument("run_dir",
                   help="finished coupler-opt run directory (read-only, "
                        "output goes to run_dir/tolerance/)")
    p.add_argument("--checkpoints", type=int, default=20,
                   help="fdtdx gradient checkpoints for the sensitivity "
                        "backward pass (see scripts/15 --checkpoints)")
    p.add_argument("--lams", default=None, metavar="LO,HI,N",
                   help="corner CE(lambda) spectrum: LO/HI wavelengths (um) "
                        "and N samples; fills bw_3db_nm/ridge_lam_um in "
                        "corners.csv. Default: single wavelength at "
                        "cfg.lam_c, those two columns left blank.")
    args = p.parse_args()

    cfg, p_latent, beta, iteration = load_run(args.run_dir)
    cfg = apply_overrides(cfg, args)

    out_dir = os.path.join(args.run_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[tolerance] run_dir={args.run_dir} iteration={iteration} "
          f"beta={beta:g} n_voxels={p_latent.size}")

    p_in_c, azimuth_sign, _ = grating_coupler.calibrated_beam(cfg)

    # ---- 1. sensitivity map ----
    sens = sensitivity_map(cfg, p_latent, beta, p_in_c,
                           num_checkpoints=args.checkpoints)
    sens_csv = os.path.join(out_dir, "sensitivity.csv")
    sens_png = os.path.join(out_dir, "sensitivity.png")
    write_sensitivity_csv(sens_csv, sens)
    plot_sensitivity(sens, sens_png)
    ce_db = 10 * np.log10(max(sens["CE"], 1e-15))
    print(f"[tolerance] CE at report point: {sens['CE']:.6g} ({ce_db:.3f} dB)")
    print(f"[tolerance] wrote {sens_csv} ({sens['x_um'].size} voxels)")
    print(f"[tolerance] wrote {sens_png}")

    # ---- 2. corner evaluation ----
    lams_spec = None
    if args.lams:
        lo, hi, n = args.lams.split(",")
        lams_spec = list(np.linspace(float(lo), float(hi), int(n)))

    rows = corner_table(cfg, p_latent, beta, p_in_c, azimuth_sign,
                        lams_spec=lams_spec, seed=cfg.seed)
    corners_csv = os.path.join(out_dir, "corners.csv")
    write_corners_csv(corners_csv, rows)
    print(f"[tolerance] wrote {corners_csv}")
    for r in rows:
        bw = "" if r["bw_3db_nm"] is None else f"  bw {r['bw_3db_nm']:.1f} nm"
        print(f"[tolerance] corner {r['corner']:>8s}: CE {r['CE_dB']:.3f} dB{bw}")
    if lams_spec is None:
        print("[tolerance] bw_3db_nm / ridge_lam_um left blank: single-"
              "wavelength corners only (pass --lams LO,HI,N for a spectrum)")

    frac, passes, n = yield_90(rows)
    print(f"[tolerance] Yield_90% = {frac:.0%} ({passes}/{n} corners with "
          f"CE_corner >= 0.9*CE_nominal) -- n={n} corner screen, NOT a "
          f"statistical yield estimate (docs/tolerance.md)")
    print(f"[done] {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
