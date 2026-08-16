"""Gate 2 — adjoint/autodiff gradients vs central finite differences.

A wrong gradient silently corrupts every optimization while forward physics
looks fine, so this runs before any physics baseline (pvgc tier-3 pattern;
pvgc measured 5-8% systematic error from under-resolution this way).

Part A: ConicFilter1D chain rule vs the authoritative numpy+autograd mapping.
Part B: jax.value_and_grad through a tiny fdtdx device sim vs central FD on
        k=3 random design voxels, rel. err < 5%.
Part C: the same finite-difference check on the REAL pvgc design path
        (filter -> projection -> Device -> full coupler scene -> mode
        overlap), because Part B's toy cell shares no code with it beyond
        fdtdx itself. Cheap settings, same 5% tolerance.
"""

import os

import numpy as np

from .runner import GateResult

NAME = "gradcheck"
ORDER = 2
REQUIRES = ("gpu",)

FD_H = 0.05
REL_TOL = 0.05
K_SAMPLES = 3
# Part C only: a central difference cannot resolve a derivative whose signal
# sits under its own float32 cancellation noise, so sample voxels carrying at
# least this fraction of the peak gradient (see _part_c_pvgc_device).
MIN_REL_GRAD = 0.05


def _part_a_filter_chain():
    import autograd
    import autograd.numpy as npa
    import jax
    import jax.numpy as jnp

    from invdx.fab import filters_np
    from invdx.fab.transforms import ConicFilter1D

    NX, GRID, RADIUS = 200, 100, 0.13
    rng = np.random.default_rng(1)
    x = np.clip(0.5 + 0.2 * rng.standard_normal(NX), 0, 1)

    W = filters_np.conic_filter_matrix(NX, RADIUS, GRID)
    g_ref = autograd.grad(lambda z: npa.sum(npa.dot(W, z) ** 2))(x)

    tr = ConicFilter1D(radius_um=RADIUS, axis=0)
    tr = tr.aset("_single_voxel_size", (1e-6 / GRID, 1e-6, 1e-6),
                 create_new_ok=True)

    def f(z):
        return (tr({"p": z.reshape(NX, 1, 1)})["p"] ** 2).sum()

    g_jax = np.asarray(jax.grad(f)(jnp.asarray(x))).ravel()
    err = np.max(np.abs(g_jax - g_ref)) / (np.max(np.abs(g_ref)) + 1e-30)
    return float(err)


def _part_b_fdtdx_fd(cfg):
    import jax
    import fdtdx

    from invdx.engines import fdtdx_engine

    config, objs, cons, det, device = fdtdx_engine.device_slab_scene(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=key)

    def loss(p):
        a, o, _ = fdtdx.apply_params(arrays, objects, p, key)
        _, a = fdtdx.run_fdtd(arrays=a, objects=o, config=config, key=key)
        return a.detector_states[det.name]["poynting_flux"].sum()

    val_and_grad = jax.jit(jax.value_and_grad(loss))
    val_only = jax.jit(loss)

    # random mid-range design so the gradient is informative
    rng = np.random.default_rng(cfg.seed)
    name = device.name
    base = np.clip(0.5 + 0.1 * rng.standard_normal(params[name].shape), 0, 1)
    params = dict(params)
    params[name] = jax.numpy.asarray(base, dtype=params[name].dtype)

    f0, grads = val_and_grad(params)
    g = np.asarray(grads[name])

    flat_idx = rng.choice(base.size, size=K_SAMPLES, replace=False)
    checks, n_bad = [], 0
    for fi in flat_idx:
        idx = np.unravel_index(fi, base.shape)
        for sign, store in ((+1, "fp"), (-1, "fm")):
            pert = base.copy()
            pert[idx] = np.clip(pert[idx] + sign * FD_H, 0, 1)
            p = dict(params)
            p[name] = jax.numpy.asarray(pert, dtype=params[name].dtype)
            val = float(val_only(p))
            if store == "fp":
                fp = val
            else:
                fm = val
        fd = (fp - fm) / (2 * FD_H)
        rel = abs(fd - g[idx]) / (abs(fd) + 1e-12)
        n_bad += rel >= REL_TOL
        checks.append({"idx": [int(i) for i in idx],
                       "adjoint": float(g[idx]), "fd": float(fd),
                       "rel_err": float(rel)})
    return float(f0), checks, n_bad


def _part_c_pvgc_device():
    """Finite-difference the production pvgc inverse-design gradient.

    Cheap-but-real settings: the full coupler scene at the 20 nm grid (the
    grid the M1 recipe uses, and the only one that divides t_si exactly), a
    0.15 ps run and 10 checkpoints. The FOM is the unnormalized mode power
    (P_in = 1): the incident-beam normalization is a design-independent
    constant, so leaving it out saves an empty-cell run without weakening the
    check by anything.

    The base design is the uniform grating softened to 0.1/0.9 — a mid-grey
    slab has almost no gradient signal to check (it is not a grating), and a
    hard 0/1 profile sits on the clip boundary where the central difference
    would degenerate into a one-sided one.

    Sampling is restricted to voxels carrying at least MIN_REL_GRAD of the
    peak gradient. Deep inside a wide tooth the tanh projection saturates and
    the true derivative is ~1e-3 of the peak; there the finite difference is
    a subtraction of two nearly equal float32 numbers and its own noise floor
    exceeds the signal, so its "relative error" measures rounding, not the
    adjoint. (Measured in script 15: such a voxel reported 7.7% while its
    neighbours agreed to 0.02-0.2%, the two derivatives differing by 1.3e-8
    in absolute terms.) That float32-cancellation mechanism only applies to
    voxels BELOW the eligibility floor. For an eligible voxel, the FD error at
    a single h is dominated by truncation instead (diagnosed 2026-08-17 on a
    production-scale run: halving h cut the discrepancy ~4x, the O(h^2)
    signature of a central difference) — which is exactly what the Richardson
    extrapolation below cancels. Do not "fix" a future failure here by
    raising REL_TOL — that would hide a real one just as effectively.

    sim_time_s defaults to 0.15e-12 so this gate stays fast; that is well
    short of the 0.8e-12 production scale, so it cannot by itself catch a
    truncation-error failure that only shows up at production settings.
    Production-scale gradcheck validation runs via scripts/15's own
    `--gradcheck` on the actual recipe (see script 15's `gradcheck()`), not
    here. Set INVDX_G2_SIM_TIME_S to override this default for local
    debugging.
    """
    import jax.numpy as jnp

    from invdx.problems import pvgc

    sim_time_s = float(os.environ.get("INVDX_G2_SIM_TIME_S", 0.15e-12))
    pcfg = pvgc.PVGCConfig(spacing_um=0.020, sim_time_s=sim_time_s,
                           theta_deg=10.0)
    pcfg.design_grid_per_um = 50
    vg_fn, _, _, params, device, value_fn = pvgc.make_ce_value_and_grad(
        pcfg, p_in=1.0, num_checkpoints=10)

    rho = pvgc.rasterize_teeth(pcfg, pvgc.uniform_grating_teeth(
        pcfg, period=0.575, duty=0.5))
    base = (0.1 + 0.8 * rho).reshape(params[device.name].shape)
    beta = jnp.asarray(float(pcfg.beta_schedule[0]), dtype=jnp.float32)

    f0, grads = vg_fn(jnp.asarray(base, dtype=jnp.float32), beta)
    g = np.asarray(grads)

    mag = np.abs(g).ravel()
    eligible = np.flatnonzero(mag >= MIN_REL_GRAD * mag.max())
    if eligible.size < K_SAMPLES:
        eligible = np.argsort(-mag)[:K_SAMPLES]
    rng = np.random.default_rng(pcfg.seed)
    flat_idx = rng.choice(eligible, size=K_SAMPLES, replace=False)
    checks, n_bad = [], 0
    for fi in flat_idx:
        idx = np.unravel_index(fi, base.shape)
        fd_by_h = {}
        for h in (FD_H, FD_H / 2):
            vals = {}
            for sign in (+1, -1):
                pert = base.copy()
                pert[idx] += sign * h
                vals[sign] = float(value_fn(
                    jnp.asarray(pert, dtype=jnp.float32), beta))
            fd_by_h[h] = (vals[+1] - vals[-1]) / (2 * h)
        fd_h, fd_h2 = fd_by_h[FD_H], fd_by_h[FD_H / 2]
        fd = (4 * fd_h2 - fd_h) / 3          # Richardson extrapolation, O(h^4)
        rel = abs(fd - g[idx]) / (abs(fd) + 1e-12)
        fd_consistency = abs(fd_h - fd_h2) / (abs(fd) + 1e-12)
        n_bad += rel >= REL_TOL
        checks.append({"idx": [int(i) for i in idx],
                       "adjoint": float(g[idx]), "fd": float(fd),
                       "fd_h": float(fd_h), "fd_h2": float(fd_h2),
                       "rel_err": float(rel),
                       "fd_consistency": float(fd_consistency)})
    return float(f0), checks, n_bad, {"grad_max": float(mag.max()),
                                      "n_eligible": int(eligible.size),
                                      "n_voxels": int(mag.size)}


def run(cfg, args):
    err_a = _part_a_filter_chain()
    if err_a > 1e-4:
        return GateResult(NAME, "fail",
                          {"reason": f"ConicFilter1D chain rule deviates from "
                                     f"autograd reference: rel err {err_a:.2e}"})

    f0, checks, n_bad = _part_b_fdtdx_fd(cfg)
    details = {"filter_chain_rel_err": err_a, "f0": f0, "fd_checks": checks}
    if n_bad:
        return GateResult(NAME, "fail", {
            "reason": f"{n_bad}/{len(checks)} sampled fdtdx gradients exceed "
                      f"{REL_TOL:.0%} rel. err — do not trust any optimization "
                      f"until resolved (check resolution vs design grid first)",
            **details})

    f0_c, checks_c, n_bad_c, info_c = _part_c_pvgc_device()
    details["pvgc_f0"] = f0_c
    details["pvgc_fd_checks"] = checks_c
    details["pvgc_sampling"] = info_c
    if n_bad_c:
        return GateResult(NAME, "fail", {
            "reason": f"{n_bad_c}/{len(checks_c)} sampled pvgc design "
                      f"gradients exceed {REL_TOL:.0%} rel. err — the "
                      f"inverse-design path is not trustworthy (check "
                      f"spacing_um vs design_grid_per_um first)",
            **details})
    return GateResult(NAME, "ok", details)
