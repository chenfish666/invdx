"""Vendors ``checkpointed_fdtd`` from fdtdx (c) Yannik Mahlau et al., MIT
License (github.com/ymahlau/fdtdx) (the sanctioned "vendor one module when
blocked" escape hatch — see README Layer A notes, and fdtdx_perf.py /
fdtdx_fixes.py for the same pattern).

Upstream file/line (fdtdx 0.6.2, pinned): ``fdtdx/fdtd/fdtd.py:474-485``,
function ``checkpointed_fdtd``. It calls::

    state = eqxi.while_loop(
        max_steps=..., cond_fun=..., body_fun=_forward_body_with_progress,
        init_val=state, kind="lax" if config.only_forward is None else
        "checkpointed", checkpoints=...,
    )

WITHOUT ``buffers=``. ``equinox.internal.while_loop``'s ``buffers=`` kwarg
(see ``equinox/internal/_loop/loop.py`` docstring and
``equinox/internal/_loop/checkpointed.py``) exists precisely to let leaves of
the loop carry skip the per-checkpoint residual store: on the forward pass
(``checkpointed.py::_checkpointed_while_loop_fwd``, ~line 464) buffer leaves
are replaced with ``None`` before being written into the ``checkpoints``-many
residual copies, and are instead threaded through the *outer* ``lax.while_loop``
carry as a single value; on the backward pass they are restored from
``filled_buffers`` — the value at the END of the forward loop.

Measured on the anchor scene (1.944M cells, C in {2,4,8,16}, via
``compiled.memory_analysis()``): unpatched per-checkpoint slope is
268.033 B/cell/ckpt, made up of one full
carry copy (E,H,psi_E,psi_H,alpha,kappa,sigma,inv_permittivities = 37
channels, 148.033 B/cell) plus a second copy of psi_E+psi_H+alpha+kappa+sigma
(30 channels, 120.000 B/cell) needed for the backward-pass VJP. alpha, kappa,
sigma and inv_permittivities are PML/material coefficients: confirmed
time-invariant by grep across the whole installed fdtdx package — no
``.aset("alpha"|"kappa"|"sigma"|"inv_permittivities", ...)`` call exists
anywhere in ``fdtdx/`` (0.6.2), so nothing ever writes to these carry leaves
inside the loop. Marking them ``buffers=`` should drop the slope to ~120
B/cell/ckpt (both copies lose those channels); see
``tests/test_fdtdx_checkpoint_buffers.py``, which re-measures the slope
before and after rather than trusting this estimate.

WHY THIS FILE EXISTS AT ALL (not just a one-line ``buffers=`` addition at the
call site): ``buffers=``-marked leaves are NOT plain arrays inside the loop
body — they become ``equinox.internal._loop.common._Buffer`` proxies that
support ONLY ``[]`` (``__getitem__``, forwarding to the underlying array) and
``.at[].set()``. They have no ``__add__``/``__mul__``/``__truediv__``/etc.
fdtdx's own physics code (``fdtdx/fdtd/update.py`` curl/CPML coefficient math)
reads ``arrays.alpha`` / ``.kappa`` / ``.sigma`` / ``.inv_permittivities`` and
uses them directly in elementwise arithmetic (e.g. ``sigma / kappa + alpha``,
``1 - c*sigma_E*eta0*inv_eps/2``) — passing a ``_Buffer`` into that would
raise at trace time. Rewriting every arithmetic use site across
update.py/curl.py/tfsf.py/detector.py would turn this into a multi-file vendor
and risk drifting from upstream math. Instead this module wraps the loop body
once: before calling the UNMODIFIED upstream ``forward()``, it materializes
plain-array copies of the four coefficient leaves via ``buf[...]`` (a
value-preserving full-slice read — bitwise identical to the original array);
``forward()`` then runs completely unchanged and produces a next-state whose
alpha/kappa/sigma/inv_permittivities slots are simply discarded and replaced
with the SAME (untouched) ``_Buffer`` objects that came in, which is exactly
what equinox's internal bookkeeping requires (it asserts, in
``common.py::new_body_fun``, that a buffer-tagged input leaf maps to a
buffer-tagged output leaf with the same ``_tag``) and is safe/correct
*because* those four arrays are provably never written inside the loop (see
above). If that invariant is ever violated (a future fdtdx update adds a
write to one of these), this file's restore-the-original-buffer step would
silently freeze that field at its initial value forever — the exact "silent
corruption, only visible in multi-iteration gradients" failure mode this
scheme was designed to rule out. tests/test_fdtdx_checkpoint_buffers.py
covers this with a bit-exact multi-iteration gradient check on top of the
single-forward bit-exactness check — a frozen coefficient array can survive
one forward pass unnoticed and only diverge once gradients accumulate.

DELETE THIS FILE once either:
  (a) upstream fdtdx exposes ``buffers=`` (or an equivalent no-checkpoint-copy
      marking) on ``checkpointed_fdtd``/``run_fdtd`` directly, or
  (b) upstream/equinox makes buffer leaves support elementwise arithmetic
      directly (removing the need for the materialize/restore wrapper).
Re-check the upstream line numbers above on any fdtdx re-pin (mirrored-from
map style, see fdtdx_perf.py's own note).
"""

from collections.abc import Callable

import equinox.internal as eqxi
import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.progress import _make_pbar, _wrap_body_with_progress
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.fdtd.stop_conditions import StoppingCondition, TimeStepCondition


def _buffers_of(state: SimulationState):
    """The four time-invariant coefficient leaves to exclude from checkpoint
    storage. Whole-array leaves of ``ArrayContainer`` (see
    ``fdtdx/fdtd/container.py``), never written inside the fdtd loop (see
    module docstring)."""
    _, arr = state
    return (arr.alpha, arr.kappa, arr.sigma, arr.inv_permittivities)


def _materialize_and_restore(body_fun):
    """Wrap a ``(time_step, ArrayContainer) -> (time_step, ArrayContainer)``
    body function so it is safe to call with ``eqxi.while_loop(buffers=...)``
    marking alpha/kappa/sigma/inv_permittivities.

    On entry those four leaves are ``_Buffer`` proxies (only ``[]`` and
    ``.at[].set()`` work). The wrapper materializes plain-array copies (bitwise
    identical values, via a full-slice read) for the UNMODIFIED upstream
    ``body_fun`` to compute with, then restore the original (untouched)
    buffer objects at those four positions in the output — required by
    equinox's internal bookkeeping, and safe only because those leaves are
    provably never written inside the loop (see module docstring).
    """

    def wrapped(state):
        step, arr = state
        plain_arr = (
            arr.aset("alpha", arr.alpha[...])
            .aset("kappa", arr.kappa[...])
            .aset("sigma", arr.sigma[...])
            .aset("inv_permittivities", arr.inv_permittivities[...])
        )
        next_step, next_arr = body_fun((step, plain_arr))
        next_arr = (
            next_arr.aset("alpha", arr.alpha)
            .aset("kappa", arr.kappa)
            .aset("sigma", arr.sigma)
            .aset("inv_permittivities", arr.inv_permittivities)
        )
        return next_step, next_arr

    return wrapped


def checkpointed_fdtd_buffers(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Vendored ``fdtdx.fdtd.fdtd.checkpointed_fdtd`` (0.6.2) with
    ``buffers=`` wired in for alpha/kappa/sigma/inv_permittivities. See module
    docstring for the why/how and the upstream line reference. Behaviorally a
    drop-in replacement: same signature, same return type, and — per
    tests/test_fdtdx_checkpoint_buffers.py — bitwise identical output and
    gradients to ``fdtdx.fdtd.fdtd.checkpointed_fdtd`` on the grating_coupler scene
    subset. Only the memory footprint of the checkpoint store differs.
    """
    import jax.numpy as jnp

    arrays = arrays.reset()
    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    if stopping_condition is not None:
        stopping_condition = stopping_condition.setup(state, config, objects)
    else:
        stopping_condition = TimeStepCondition().setup(state, config, objects)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (checkpointed, buffers)",
        progress_callback=progress_callback,
    )

    from fdtdx.fdtd.forward import forward
    from functools import partial

    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=config.invertible_optimization,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)
    _buffered_body = _materialize_and_restore(_forward_body_with_progress)

    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=partial(
            stopping_condition,
            config=config,
            objects=objects,
        ),
        body_fun=_buffered_body,
        init_val=state,
        buffers=_buffers_of,
        kind="lax" if config.only_forward is None else "checkpointed",
        checkpoints=(None if config.gradient_config is None else config.gradient_config.num_checkpoints),
    )
    _close_pbar()

    return state


def run_fdtd_buffers(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Drop-in for ``fdtdx.run_fdtd`` (mirrors ``fdtdx/fdtd/wrapper.py``
    exactly) that routes the ``method="checkpointed"`` gradient path through
    ``checkpointed_fdtd_buffers`` instead of upstream's
    ``checkpointed_fdtd``. All other branches (``reversible``, or
    ``gradient_config is None``) delegate to upstream unchanged — this file
    only touches the one call site relevant to grating_coupler's gradient path.
    """
    if stopping_condition is not None:
        if config.gradient_config is not None:
            raise NotImplementedError(
                "Custom stopping conditions are not yet compatible with gradient computation. "
                "Set config.gradient_config to None or use default time-based stopping by "
                "setting stopping_condition=None."
            )

    if config.gradient_config is None:
        return checkpointed_fdtd_buffers(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    if config.gradient_config.method == "reversible":
        return reversible_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    elif config.gradient_config.method == "checkpointed":
        return checkpointed_fdtd_buffers(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    else:
        raise Exception(f"Unknown gradient computation method: {config.gradient_config.method}")
