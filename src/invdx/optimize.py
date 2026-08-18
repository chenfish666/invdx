"""Layer B — the inverse-design loop, deliberately engine-agnostic.

Everything problem-specific lives behind two arguments: a value-and-gradient
callable `vg_fn(p, beta) -> (loss, dloss/dp)` and the config that owns the
binarization schedule. The loop itself only knows how to

  * walk the beta schedule (continuous -> binary annealing),
  * take Adam steps on a latent vector confined to [0, 1],
  * checkpoint every single iteration, atomically, and resume from it,
  * stop on iteration count, wall-clock budget, or convergence.

Checkpointing is not optional polish here: one pvgc inverse-design round is
~13 GPU-hours, longer than any interactive session and longer than most queue
slots, so an interrupted run that cannot resume is a lost day. Keeping this
module free of fdtdx/pvgc imports is what makes the resume path unit-testable
without a GPU (tests/test_pvgc_optimize.py).

Sign convention: `vg_fn` returns a LOSS to minimize, and every problem in this
repo maximizes a figure of merit, so loss = -FOM and history.csv records
CE = -loss. Any FOM plugged in here must follow that convention. vg_fn may
also return ((loss, aux_dict), grad) with the true `ce` and the linear `s11`
in aux (a penalized FOM); the `CE` column then carries the TRUE CE and the
`fom` column always carries -loss.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import runio

STATE_FILE = "opt_state.npz"
HISTORY_FILE = "history.csv"
HISTORY_HEADER = ["iter", "beta", "CE", "CE_dB", "grad_norm", "wall_s",
                  "s11_dB", "fom"]


@dataclass
class OptState:
    """One resumable optimizer position (plus this session's bookkeeping)."""

    p: Any                              # latent design array
    opt_state: Any                      # optax state pytree (or its flat form)
    iteration: int                      # index of the LAST completed iteration
    beta: float                         # projection steepness used for it
    history: list = field(default_factory=list)
    stop_reason: str = ""


def beta_for_iter(cfg, it, n_iters):
    """Projection steepness for iteration `it` out of `n_iters`.

    cfg.beta_schedule stages are spread evenly over the run (5 stages x 8
    iterations for the nominal 40-iteration round). Accepts a tuple or a list:
    a run dir's config.json round-trips the tuple as a JSON list, and
    reconstructing the config from it must not change the schedule.
    """
    sched = tuple(cfg.beta_schedule)
    if not sched:
        raise ValueError("cfg.beta_schedule is empty — nothing to anneal")
    n = max(int(n_iters), 1)
    stage = min(len(sched) - 1, int(it) * len(sched) // n)
    return float(sched[stage])


def _flat_opt_state(opt_state):
    from jax.flatten_util import ravel_pytree

    flat, _ = ravel_pytree(opt_state)
    return np.asarray(flat)


def save_state(run_dir, state):
    """Write opt_state.npz atomically (.tmp.npz + os.replace).

    A checkpoint is written after every iteration; a crash mid-write must
    never leave a half-file where the previous good one used to be.
    """
    path = os.path.join(run_dir, STATE_FILE)
    tmp = path + ".tmp.npz"          # np.savez appends .npz unless present
    np.savez(tmp,
             p=np.asarray(state.p),
             opt_flat=_flat_opt_state(state.opt_state),
             iteration=np.asarray(int(state.iteration)),
             beta=np.asarray(float(state.beta)))
    os.replace(tmp, path)
    return path


def load_state(run_dir, unravel=None):
    """Read opt_state.npz back.

    unravel — the second return value of `jax.flatten_util.ravel_pytree` on a
              freshly initialized optimizer state of the SAME shape; without
              it the returned `opt_state` is the flat array (the optax pytree
              structure is not stored, it is rebuilt from opt.init).
    """
    with np.load(os.path.join(run_dir, STATE_FILE)) as z:
        flat = z["opt_flat"]
        return OptState(p=z["p"],
                        opt_state=flat if unravel is None else unravel(flat),
                        iteration=int(z["iteration"]),
                        beta=float(z["beta"]))


def run_loop(vg_fn, p0, cfg, n_iters, lr, run_dir, resume=False, on_iter=None,
             time_budget_h=None, stop_rel_tol=0.005, stop_patience=5):
    """Adam on a [0,1]-boxed latent vector with per-iteration checkpointing.

    vg_fn(p, beta) -> (loss, grad); loss = -FOM (see module docstring)
    p0             — initial latent array (ignored when resume=True)
    n_iters        — total iterations of the WHOLE run, resumed or not, and
                     the denominator of the beta schedule: keep it fixed
                     across a resume or the annealing changes underneath you
    resume         — continue from run_dir/opt_state.npz
    on_iter(row)   — optional callback with the history row just written
    time_budget_h  — stop before an iteration that would overrun the budget
    stop_rel_tol / stop_patience — stop after `patience` consecutive
                     iterations whose relative FOM improvement stays below
                     `rel_tol`, but only inside the final beta stage (earlier
                     stages plateau on purpose, that is what annealing does)

    Returns the final OptState, with `history` holding this session's rows and
    `stop_reason` in {"iters", "time-budget", "converged"}.
    """
    import jax.numpy as jnp
    import optax
    from jax.flatten_util import ravel_pytree

    p = jnp.asarray(p0, dtype=jnp.float32)
    opt = optax.adam(lr)
    opt_state = opt.init(p)
    _, unravel = ravel_pytree(opt_state)

    it0 = 0
    beta0 = beta_for_iter(cfg, 0, n_iters)
    if resume:
        prev = load_state(run_dir, unravel=unravel)
        p = jnp.asarray(prev.p, dtype=jnp.float32)
        opt_state = prev.opt_state
        it0 = prev.iteration + 1
        # the checkpointed beta is ground truth for the schedule stage
        # actually used to produce `prev.p` — recomputing it from
        # beta_for_iter(cfg, iteration, n_iters) is wrong whenever n_iters
        # differs from the run that wrote the checkpoint (e.g. --resume
        # --iters=<iters_done> for a finalize-only resume), because it
        # rescales the schedule denominator and silently jumps stages.
        beta0 = prev.beta

    csv_path = os.path.join(run_dir, HISTORY_FILE)
    # a resumed pre-s11 run keeps its own (6-column) header: append_csv only
    # writes a header on file creation, so rows must match what is there
    hdr = HISTORY_HEADER
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            first = f.readline().strip()
        if first:
            hdr = first.split(",")
    beta_final = beta_for_iter(cfg, n_iters - 1, n_iters)
    budget_s = None if time_budget_h is None else float(time_budget_h) * 3600.0
    t_start = time.time()

    state = OptState(p=p, opt_state=opt_state, iteration=it0 - 1,
                     beta=beta0, stop_reason="iters")
    prev_fom, stall, last_wall = None, 0, 0.0

    for it in range(it0, n_iters):
        if budget_s is not None and \
                time.time() - t_start + last_wall > budget_s:
            state.stop_reason = "time-budget"
            break

        beta = beta_for_iter(cfg, it, n_iters)
        t0 = time.time()
        out, grad = vg_fn(p, jnp.asarray(beta, dtype=jnp.float32))
        val, aux = out if isinstance(out, tuple) else (out, None)
        loss = float(val)
        updates, opt_state = opt.update(grad, opt_state)
        p = jnp.clip(optax.apply_updates(p, updates), 0.0, 1.0)
        last_wall = time.time() - t0

        fom = -loss
        ce = float(aux["ce"]) if aux else fom
        row = {"iter": it, "beta": beta, "CE": ce,
               "CE_dB": float(10 * np.log10(max(ce, 1e-15))),
               "grad_norm": float(jnp.linalg.norm(grad)),
               "wall_s": last_wall,
               "s11_dB": (float(10 * np.log10(max(float(aux["s11"]), 1e-15)))
                          if aux else float("nan")),
               "fom": fom}
        runio.append_csv(csv_path, hdr, [row[k] for k in hdr])
        state.p, state.opt_state = p, opt_state
        state.iteration, state.beta = it, beta
        save_state(run_dir, state)
        state.history.append(row)
        if on_iter is not None:
            on_iter(row)

        if beta >= beta_final and prev_fom is not None:
            improved = (fom - prev_fom) / max(abs(prev_fom), 1e-30)
            stall = stall + 1 if improved < stop_rel_tol else 0
            if stall >= stop_patience:
                state.stop_reason = "converged"
                break
        prev_fom = fom

    return state
