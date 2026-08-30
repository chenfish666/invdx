"""Equivalence gate for the vendored checkpointed_fdtd (engines/
fdtdx_checkpoint_buffers).

run_fdtd_buffers must be BITWISE-identical to fdtdx.run_fdtd (method=
"checkpointed") on a small differentiable scene: both the forward detector
readout and jax.value_and_grad must match exactly (max abs diff == 0.0).
The vendored module only changes what the checkpointed while_loop stores
(alpha/kappa/sigma/inv_permittivities move to eqxi.while_loop's `buffers=`
instead of the per-checkpoint residual store) — it must not change what is
computed. Any nonzero diff here means the materialize/restore wrapper (see
fdtdx_checkpoint_buffers.py docstring) broke something — most likely one of
the two "dangers" flagged there: an in-place write to one of those four
arrays somewhere in the call graph, or a donation dependency. Fix the root
cause, never loosen these asserts.

A second test runs TWO optimizer-style gradient steps (not just one) to
catch silent cross-iteration corruption: if alpha/kappa/sigma/inv_permittivities
were ever written to inside the loop, the materialize/restore wrapper would
freeze that field at its initial value every iteration, which a single
forward+grad comparison cannot detect — only a multi-iteration parameter
update surfaces it.

Runs on CPU (JAX_PLATFORMS=cpu) so it works on any machine, using the same
--xla_backend_optimization_level=0 convention as test_fdtdx_perf.py (rules
out backend-specific FMA contraction as a confound; not expected to matter
here since both paths run the identical upstream forward() op-for-op, but
keeping the same discipline as the other vendor-equivalence test).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=0"
).strip()

import jax
import jax.numpy as jnp
import optax
import pytest

import fdtdx

from invdx.engines import fdtdx_engine
from invdx.engines.fdtdx_checkpoint_buffers import run_fdtd_buffers
from invdx.problems.grating_coupler import GratingCouplerConfig


def _tiny_gradcheck_scene():
    """A small differentiable slab scene (fdtdx_engine.device_slab_scene),
    coarsened for test speed. Already has a checkpointed GradientConfig."""
    cfg = GratingCouplerConfig(spacing_um=0.05)
    config, object_list, constraints, detector, device = fdtdx_engine.device_slab_scene(
        cfg, wavelength_um=1.55, cell_um=(1.5, 1.5, 1.5), time_s=15e-15, n_voxels_x=5)
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=k1)
    p0 = jnp.asarray(params[device.name])
    return config, objects, arrays, params, device, detector, key, k2, p0


def _loss_fn(run_fdtd_fn, config, objects, arrays, params, device, detector, key, k2):
    def loss(p):
        prm = dict(params)
        prm[device.name] = p
        a, o, _ = fdtdx.apply_params(arrays, objects, prm, k2)
        _, a = run_fdtd_fn(arrays=a, objects=o, config=config, key=key,
                            show_progress=False)
        flux = a.detector_states[detector.name]["poynting_flux"]
        return jnp.sum(flux[:, 0])
    return loss


def test_forward_and_grad_bit_exact_vs_upstream():
    config, objects, arrays, params, device, detector, key, k2, p0 = _tiny_gradcheck_scene()

    loss_new = _loss_fn(run_fdtd_buffers, config, objects, arrays, params,
                        device, detector, key, k2)
    loss_old = _loss_fn(fdtdx.run_fdtd, config, objects, arrays, params,
                        device, detector, key, k2)

    val_new, grad_new = jax.value_and_grad(loss_new)(p0)
    val_old, grad_old = jax.value_and_grad(loss_old)(p0)

    assert float(jnp.max(jnp.abs(val_new - val_old))) == 0.0, (
        f"forward value differs: new={val_new!r} old={val_old!r} — the "
        "buffers= change was supposed to be value-neutral (see GATE.md "
        "Class A); a nonzero diff here means it changed the computed graph, "
        "not just what gets checkpointed."
    )
    assert float(jnp.max(jnp.abs(grad_new - grad_old))) == 0.0, (
        f"gradient differs: max|dgrad|="
        f"{float(jnp.max(jnp.abs(grad_new - grad_old)))!r} — see module "
        "docstring's 'silent corruption' danger (an in-place write to "
        "alpha/kappa/sigma/inv_permittivities would show up here even "
        "though the forward value matched)."
    )


def test_two_iteration_optimizer_bit_exact_vs_upstream():
    """V3-style check: run TWO gradient-descent steps end-to-end and compare
    every intermediate (param, loss, grad) triple bit-exactly. A materialize/
    restore bug that silently freezes a coefficient array at its initial
    value would still pass a single-step check (the frozen value equals the
    true value on iteration 0) but diverge by iteration 1."""
    config, objects, arrays, params, device, detector, key, k2, p0 = _tiny_gradcheck_scene()

    loss_new = _loss_fn(run_fdtd_buffers, config, objects, arrays, params,
                        device, detector, key, k2)
    loss_old = _loss_fn(fdtdx.run_fdtd, config, objects, arrays, params,
                        device, detector, key, k2)

    opt = optax.sgd(learning_rate=1e-3)

    def run(loss_fn):
        p = p0
        opt_state = opt.init(p)
        history = []
        for _ in range(2):
            val, grad = jax.value_and_grad(loss_fn)(p)
            updates, opt_state = opt.update(grad, opt_state)
            p = optax.apply_updates(p, updates)
            history.append((val, grad, p))
        return history

    hist_new = run(loss_new)
    hist_old = run(loss_old)

    for it, ((val_n, grad_n, p_n), (val_o, grad_o, p_o)) in enumerate(
        zip(hist_new, hist_old)
    ):
        assert float(jnp.max(jnp.abs(val_n - val_o))) == 0.0, f"iter {it}: loss differs"
        assert float(jnp.max(jnp.abs(grad_n - grad_o))) == 0.0, f"iter {it}: grad differs"
        assert float(jnp.max(jnp.abs(p_n - p_o))) == 0.0, f"iter {it}: updated params differ"
