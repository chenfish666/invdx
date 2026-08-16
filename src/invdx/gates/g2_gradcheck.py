"""Gate 2 — adjoint/autodiff gradients vs central finite differences.

A wrong gradient silently corrupts every optimization while forward physics
looks fine, so this runs before any physics baseline (pvgc tier-3 pattern;
pvgc measured 5-8% systematic error from under-resolution this way).

Part A: ConicFilter1D chain rule vs the authoritative numpy+autograd mapping.
Part B: jax.value_and_grad through a tiny fdtdx device sim vs central FD on
        k=3 random design voxels, rel. err < 5%.
"""

import numpy as np

from .runner import GateResult

NAME = "gradcheck"
ORDER = 2
REQUIRES = ("gpu",)

FD_H = 0.05
REL_TOL = 0.05
K_SAMPLES = 3


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
    return GateResult(NAME, "ok", details)
