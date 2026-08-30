"""Gate 2 — adjoint/autodiff gradients vs central finite differences.

A wrong gradient silently corrupts every optimization while forward physics
looks fine, so this runs before any physics baseline. Under-resolution is the
classic culprit this catches: the forward fields still look right while the
adjoint gradients come back systematically too small.

Part A: ConicFilter1D chain rule vs the authoritative numpy+autograd mapping.
Part B: jax.value_and_grad through a tiny fdtdx device sim vs central FD on
        k=3 random design voxels, rel. err < 5%.
Part C: the same finite-difference check on a REAL problem's design path
        (filter -> projection -> Device -> full scene -> mode overlap),
        because Part B's toy cell shares no code with it beyond fdtdx
        itself. Cheap settings, same 5% tolerance.

Parts A and B are problem-independent and always run. Part C measures the
problem named by `--problem` (default `invdx.problems.DEFAULT`), through that
problem's `gradcheck_case()`. The split of responsibility is deliberate: the
problem owns the settings, the starting design and the dtype conventions of
its own callables; this gate owns the check — the eligibility floor, the
voxel sampling, the Richardson extrapolation and REL_TOL. That is what lets a
problem with no fdtdx and no jax be gradient-checked by this same code.

A problem may declare `gradcheck_case=Unsupported("why not")`. Parts A and B
still run and are still real coverage, so the gate reports [part] rather than
[ok] or [n/a], with the problem's reason on the same line. A problem that
declares nothing at all is an import error, which the runner turns into
[FAIL] — silence is never an answer here.
"""

import numpy as np

from invdx import problems
from invdx.richardson_fd import richardson_fd_check

from .runner import GateResult, PARTIAL

NAME = "gradcheck"
ORDER = 2
REQUIRES = ("gpu",)

FD_H = 0.05
REL_TOL = 0.05
K_SAMPLES = 3
# Part C only: a central difference cannot resolve a derivative whose signal
# sits under its own float32 cancellation noise, so sample voxels carrying at
# least this fraction of the peak gradient (see _part_c_problem_device).
MIN_REL_GRAD = 0.05

# Part C's sampling report is a merge of two authors: whatever the problem put
# in `GradcheckCase.info`, plus these three, which the gate measures itself.
# A name in both is refused rather than resolved. Picking a winner would mean
# writing one author's number under the other author's name -- the report still
# parses, every key is present, and the value is simply someone else's. Raising
# turns that into an import-time-loud failure with the colliding name in it.
_GATE_OWNED_INFO_KEYS = frozenset({"grad_max", "n_eligible", "n_voxels"})


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


def _part_c_problem_device(spec):
    """Finite-difference a problem's production inverse-design gradient.

    The settings, the starting design and the callables come from
    `spec.gradcheck_case()` — that is the problem's business, and for
    `grating_coupler` its docstring is where the choice of grid, run length
    and 0.1/0.9 softening is argued. Everything below is the check, and is
    the same for every problem.

    Sampling is restricted to voxels carrying at least MIN_REL_GRAD of the
    peak gradient. Deep inside a wide tooth the tanh projection saturates and
    the true derivative is ~1e-3 of the peak; there the finite difference is
    a subtraction of two nearly equal float32 numbers and its own noise floor
    exceeds the signal, so its "relative error" measures rounding, not the
    adjoint. (Measured in script 15: such a voxel reported 7.7% while its
    neighbours agreed to 0.02-0.2%, the two derivatives differing by 1.3e-8
    in absolute terms.) That float32-cancellation mechanism only applies to
    voxels BELOW the eligibility floor. For an eligible voxel, the FD error at
    a single h is dominated by truncation instead (diagnosed on a
    production-scale run: halving h cut the discrepancy ~4x, the O(h^2)
    signature of a central difference) — which is exactly what the Richardson
    extrapolation below cancels. Do not "fix" a future failure here by
    raising REL_TOL — that would hide a real one just as effectively.
    """
    case = spec.gradcheck_case()
    base = np.asarray(case.base)
    beta = case.beta

    f0, grads = case.vg_fn(base, beta)
    g = np.asarray(grads)

    mag = np.abs(g).ravel()
    eligible = np.flatnonzero(mag >= MIN_REL_GRAD * mag.max())
    if eligible.size < K_SAMPLES:
        eligible = np.argsort(-mag)[:K_SAMPLES]
    rng = np.random.default_rng(case.seed)
    flat_idx = rng.choice(eligible, size=K_SAMPLES, replace=False)
    checks, n_bad = [], 0
    for fi in flat_idx:
        idx = np.unravel_index(fi, base.shape)

        def evaluate(sign, hh, idx=idx):
            pert = base.copy()
            pert[idx] += sign * hh
            return case.value_fn(pert, beta)

        rc = richardson_fd_check(evaluate, FD_H, g[idx])
        fd, fd_h, fd_h2 = rc["fd"], rc["fd_h"], rc["fd_h2"]
        rel, fd_consistency = rc["rel_err"], rc["fd_consistency"]
        n_bad += rel >= REL_TOL
        checks.append({"idx": [int(i) for i in idx],
                       "adjoint": float(g[idx]), "fd": float(fd),
                       "fd_h": float(fd_h), "fd_h2": float(fd_h2),
                       "rel_err": float(rel),
                       "fd_consistency": float(fd_consistency)})
    info = dict(case.info)
    clash = sorted(_GATE_OWNED_INFO_KEYS.intersection(info))
    if clash:
        raise ValueError(
            f"{spec.name}.gradcheck_case() put gate-owned key(s) {clash} in "
            f"GradcheckCase.info; the gate measures these itself. Rename them "
            f"in the problem -- reserved: {sorted(_GATE_OWNED_INFO_KEYS)}.")
    info.update({"grad_max": float(mag.max()),
                 "n_eligible": int(eligible.size),
                 "n_voxels": int(mag.size)})
    return float(f0), checks, n_bad, info


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

    spec = problems.from_args(args)
    details["problem"] = spec.name
    slot = spec.gradcheck_case
    if isinstance(slot, problems.Unsupported):
        # Parts A and B passed and are real coverage, so this is not [n/a];
        # Part C did not run, so it is not [ok] either. The reason rides on
        # the same console line as the status, which is the whole reason the
        # status is worth having.
        details["reason"] = (f"parts A+B passed; part C not applicable to "
                             f"{spec.name}: {slot.reason}")
        return GateResult(NAME, PARTIAL, details)

    f0_c, checks_c, n_bad_c, info_c = _part_c_problem_device(spec)
    details[f"{spec.name}_f0"] = f0_c
    details[f"{spec.name}_fd_checks"] = checks_c
    details[f"{spec.name}_sampling"] = info_c
    if n_bad_c:
        return GateResult(NAME, "fail", {
            "reason": f"{n_bad_c}/{len(checks_c)} sampled {spec.name} design "
                      f"gradients exceed {REL_TOL:.0%} rel. err — the "
                      f"inverse-design path is not trustworthy (check "
                      f"spacing_um vs design_grid_per_um first)",
            **details})
    return GateResult(NAME, "ok", details)
