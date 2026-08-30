#!/usr/bin/env python
"""Lesson 2 mainline: your first adjoint gradient -- healing a defect in the
textbook photonic-crystal 90-degree bend benchmark.

Storyline:
  1. Take the phc_bend bend, knock out the Layer-I rod whose loss hurts most
     (transmission drops hard)
  2. Draw a 2a x 2a design region around the defect; the material there
     becomes a continuous parameter
     eps = 1 + (eps_rod - 1) * sigmoid(theta)
  3. jax.grad differentiates through the whole FDTD time evolution, returning
     the gradient at every design-region pixel in one pass
     (that is the adjoint method: one forward + one backward = all gradients)
  4. Verify the gradient by finite differences first (the invdx G2 rule: an
     unverified gradient does not ship)
  5. Adam iterations -- watch inverse design heal the bend by itself

  python scripts/09_toy_adjoint.py                  # full run (~3 min, CPU)
  python scripts/09_toy_adjoint.py --gradcheck-only # gradient check only (~1 min)
"""

import argparse
import os
import sys
import time

# Determinism first: the lesson and the gradient check both run on CPU +
# float64 (the GPU is for scaling up later)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import phc_bend
from invdx.toy import fdtd2d, fdtd2d_jax
from invdx import runio

DEFECT = (1, 0)                     # benchmark Layer-I horizontal defect (hurts most)
FSTARS = (0.31, 0.34, 0.37)         # three in-gap frequencies at this size


def build_case(cfg):
    """Measurement setup + straight-waveguide normalization (run once in
    numpy, then treated as a constant)."""
    ports = phc_bend._toy_ports(cfg)
    fcen = 0.5 * (cfg.f_min + cfg.f_max)
    spread = 1.0 / (np.pi * (cfg.f_max - cfg.f_min) / 2)
    n = int(round((cfg.n_side + 2 * cfg.pad_a) * cfg.res_per_a))
    dx = 1.0 / cfg.res_per_a
    kw = dict(nx=n, ny=n, dx=dx, steps=cfg.toy_steps,
              source={**ports["src"], "t0": 4 * spread, "spread": spread,
                      "fcen": fcen},
              courant=cfg.toy_courant)
    ref = fdtd2d.run(**kw, eps=phc_bend.epsilon_grid(cfg, "straight"),
                     line_probes={"out": ports["straight_out"]})
    dt = cfg.toy_courant * dx
    p_ref = fdtd2d.line_flux_spectrum(ref["lines"]["out"], FSTARS, dt, dx,
                                      sign=-1.0)
    return kw, ports["bend_out"], dt, dx, p_ref


def design_box(cfg):
    """Grid slice of the 2a x 2a box around the defect (lattice coordinates
    ix in {c+1,c+2}, iy in {c-1,c})."""
    res, c, pad = cfg.res_per_a, cfg.center, cfg.pad_a
    gx = slice(int((pad + c + 1) * res), int((pad + c + 3) * res))
    gy = slice(int((pad + c - 1) * res), int((pad + c + 1) * res))
    return gx, gy


def main():
    p = base_parser(__doc__)
    p.add_argument("--gradcheck-only", action="store_true")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.2)
    args = p.parse_args()
    cfg = apply_overrides(
        phc_bend.PhCBendConfig(n_side=9, res_per_a=10, toy_steps=2500),
        args)
    d = start_run(cfg, args, "toy-adjoint")

    kw, bend_out, dt, dx, p_ref = build_case(cfg)
    eps_damaged = phc_bend.epsilon_grid(cfg, "bend", defect=DEFECT)
    eps_intact = phc_bend.epsilon_grid(cfg, "bend")
    gx, gy = design_box(cfg)
    p_ref_j = jnp.asarray(p_ref)
    base = jnp.asarray(eps_damaged)

    def eps_of(theta):
        box = 1.0 + (cfg.eps_rod - 1.0) * jax.nn.sigmoid(theta)
        return base.at[gx, gy].set(box)

    def mean_T(eps):
        _, ys = fdtd2d_jax.simulate(**kw, eps=eps,
                                    line_probes={"out": bend_out})
        E, H = ys["out"]
        P = fdtd2d_jax.line_flux_spectrum_jnp(E, H, FSTARS, dt, dx,
                                              sign=+1.0)
        return jnp.mean(P / p_ref_j)

    objective = jax.jit(lambda th: mean_T(eps_of(th)))
    vg = jax.jit(jax.value_and_grad(lambda th: mean_T(eps_of(th))))

    # start from the damaged structure itself (theta inverted from damaged eps)
    p0 = (np.asarray(eps_damaged[gx, gy]) - 1.0) / (cfg.eps_rod - 1.0)
    p0 = np.clip(p0, 1e-3, 1 - 1e-3)
    theta = jnp.asarray(np.log(p0 / (1 - p0)))

    T_damaged = float(objective(theta))
    T_intact = float(mean_T(jnp.asarray(eps_intact)))
    print(f"[base] intact bend   mean T = {T_intact:.3f}")
    print(f"[base] damaged bend  mean T = {T_damaged:.3f}   "
          f"(defect {DEFECT}, benchmark Layer-I)")

    # ---- gradient check (G2 rule: an unverified gradient does not ship) ----
    t0 = time.time()
    val, g = vg(theta)
    t_grad = time.time() - t0
    print(f"[adjoint] one backward pass = gradients for all {g.size} "
          f"design-region parameters  ({t_grad:.1f}s, incl. compile)")
    rng = np.random.default_rng(0)
    idx = [tuple(rng.integers(0, s) for s in g.shape) for _ in range(3)]
    h = 1e-4
    worst = 0.0
    for ij in idx:
        e = jnp.zeros_like(theta).at[ij].set(h)
        fd = (float(objective(theta + e)) - float(objective(theta - e))) \
            / (2 * h)
        ad = float(g[ij])
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
        worst = max(worst, rel)
        print(f"[gradcheck] pixel {ij}: adjoint {ad:+.6e}  "
              f"FD {fd:+.6e}  rel err {rel:.2e}")
    if worst > 1e-5:
        print("[gradcheck] FAIL -- gradient not trustworthy, stopping")
        return 1
    print(f"[gradcheck] PASS (worst {worst:.2e} < 1e-5)")
    if args.gradcheck_only:
        runio.save_json(os.path.join(d, "results.json"),
                        {"gradcheck_worst_rel": worst})
        return 0

    # ---- inverse design: let the gradient heal the bend on its own ----
    import optax

    opt = optax.adam(args.lr)
    state = opt.init(theta)
    hist = []
    for it in range(args.iters):
        val, g = vg(theta)
        upd, state = opt.update(-g, state)     # maximizing -> feed -grad to the minimizer
        theta = optax.apply_updates(theta, upd)
        hist.append(float(val))
        if it % 5 == 0 or it == args.iters - 1:
            print(f"[opt] iter {it:3d}  mean T = {float(val):.3f}")

    T_healed = float(objective(theta))
    print(f"\n[result] damaged {T_damaged:.3f} -> healed {T_healed:.3f} "
          f"(intact {T_intact:.3f})")

    # what the design region ended up as (mean density per lattice cell,
    # 0 = vacuum, 1 = solid rod)
    dens = np.asarray(jax.nn.sigmoid(theta))
    r = cfg.res_per_a
    print("[design] healed-region density (2x2 lattice, one entry = one a x a cell):")
    for jy in range(1, -1, -1):
        row = " ".join(
            f"{dens[ix * r:(ix + 1) * r, jy * r:(jy + 1) * r].mean():.2f}"
            for ix in range(2))
        print("         " + row)

    runio.save_json(os.path.join(d, "results.json"), {
        "T_intact": T_intact, "T_damaged": T_damaged, "T_healed": T_healed,
        "gradcheck_worst_rel": worst, "history": hist,
        "fstars": list(FSTARS), "defect": list(DEFECT)})
    np.savez(os.path.join(d, "design.npz"), theta=np.asarray(theta),
             eps=np.asarray(eps_of(theta)))
    print(f"[done] {d}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
