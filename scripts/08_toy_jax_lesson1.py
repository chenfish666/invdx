#!/usr/bin/env python
"""Lesson 1 checker: your JAX engine vs the numpy engine, same photonic-
crystal gap measurement.

  python scripts/08_toy_jax_lesson1.py                # check the library engine
  python scripts/08_toy_jax_lesson1.py --scan-demo    # three-line lax.scan primer
  python scripts/08_toy_jax_lesson1.py --gpu          # check, then run the same code on GPU
  python scripts/08_toy_jax_lesson1.py \
      --file tutorials/01-jax-port/fdtd2d_jax_skeleton.py
                                  # check the skeleton you filled in (mainline untouched)

Pass criterion: pointwise difference < 1e-9 at float64 (in practice ~1e-15,
machine precision).
"""

import argparse
import os
import sys
import time

# The lesson checks on CPU by default (deterministic, no queueing for a GPU);
# --gpu opts in. The platform choice must take effect before `import jax`,
# otherwise the CUDA plugin probes the machine first and prints warnings.
if "--gpu" not in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The float64 switch must be set before any jax array exists -- this is JAX's
# most classic trap: it defaults to float32 (GPU-friendly) while numpy is
# float64. Without it your engine "looks right" but differs from numpy at
# 1e-7, and you cannot tell a physics bug from a precision gap.
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


def scan_demo():
    """lax.scan is a loop that carries state: cumsum in three lines."""
    import jax.numpy as jnp

    def step(carry, x):          # carry in, carry out, plus one output per step
        carry = carry + x
        return carry, carry      # (new state, what to record for this step)

    total, history = jax.lax.scan(step, 0.0, jnp.arange(5.0))
    print("total   =", total)          # 10.0
    print("history =", history)        # [0 1 3 6 10]
    print()
    print("FDTD mapping: carry = (Ez, Hx, Hy) field state; x = source amplitude per step;")
    print("per-step output = probe reading. scan compiles the whole time loop into one")
    print("XLA program, so jax.grad can then differentiate it w.r.t. any input (e.g. eps).")


def build_case():
    from invdx.problems import phc_bend

    cfg = phc_bend.PhCBendConfig(n_side=9, res_per_a=10, toy_steps=2000,
                                 n_freq=9)
    eps = phc_bend.epsilon_grid(cfg, "bulk")
    ports = phc_bend._toy_ports(cfg)
    fcen = 0.5 * (cfg.f_min + cfg.f_max)
    spread = 1.0 / (np.pi * (cfg.f_max - cfg.f_min) / 2)
    n = eps.shape[0]
    return dict(nx=n, ny=n, dx=1.0 / cfg.res_per_a, steps=cfg.toy_steps,
                source={**ports["bulk_src"], "t0": 4 * spread,
                        "spread": spread, "fcen": fcen},
                eps=eps, line_probes={"out": ports["bulk_out"]})


def load_engine(path):
    """Load a student-filled skeleton from a file path (tutorial mode)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("student_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(engine_file=None):
    from invdx.toy import fdtd2d

    if engine_file:
        fdtd2d_jax = load_engine(engine_file)
        print(f"[mode] checking tutorial skeleton: {engine_file}")
    else:
        from invdx.toy import fdtd2d_jax

    kw = build_case()
    print(f"[case] photonic-crystal bulk gap measurement, {kw['nx']}^2 grid, "
          f"{kw['steps']} steps (small, runs in seconds)")

    t0 = time.time()
    ref = fdtd2d.run(**kw)
    t_np = time.time() - t0

    try:
        t0 = time.time()
        mine = fdtd2d_jax.run(**kw)
        t_j1 = time.time() - t0
    except NotImplementedError as e:
        print(f"\n[todo] {e}")
        print("[todo] a blank is still empty -- work through "
              "tutorials/01-jax-port/README.md one blank at a time; if you get "
              "stuck, check your answer against src/invdx/toy/fdtd2d_jax.py.")
        return 1

    t0 = time.time()
    mine = fdtd2d_jax.run(**kw)
    t_j2 = time.time() - t0

    E0, H0 = ref["lines"]["out"]["E"], ref["lines"]["out"]["H"]
    E1, H1 = mine["lines"]["out"]["E"], mine["lines"]["out"]["H"]
    scale = float(np.max(np.abs(E0)))
    dE = float(np.max(np.abs(E0 - E1)))
    dH = float(np.max(np.abs(H0 - H1)))
    print(f"[diff] max|dE| = {dE:.3e}, max|dH| = {dH:.3e} "
          f"(field scale {scale:.3e})")
    print(f"[time] numpy {t_np:.2f}s | jax first run {t_j1:.2f}s (with "
          f"compile) | jax rerun {t_j2:.2f}s")

    if dE < 1e-9 * scale and dH < 1e-9 * scale:
        print("\n[PASS] both engines agree to the last bit -- your first JAX FDTD works.")
        print("       Next lesson (scripts/09): make eps a parameter, push jax.grad")
        print("       through the whole time evolution, then check it against finite differences.")
        return 0
    print("\n[FAIL] engines differ -- check: (1) blank order (H, then E, source, Mur)")
    print("       (2) is Ez_old snapshotted BEFORE the E interior update")
    print("       (3) is eps divided in the E update (not in the H update)")
    return 1


def gpu_check():
    from invdx.toy import fdtd2d_jax

    dev = jax.devices()[0]
    print(f"[gpu] jax default device: {dev.platform} ({dev.device_kind})")
    kw = build_case()
    t0 = time.time()
    fdtd2d_jax.run(**kw)
    t1 = time.time() - t0
    t0 = time.time()
    fdtd2d_jax.run(**kw)
    t2 = time.time() - t0
    print(f"[gpu] same code, zero edits, on {dev.platform}: first run "
          f"{t1:.2f}s, rerun {t2:.2f}s")
    print("[gpu] that is the port's second payoff: the numpy version is CPU-only, forever.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-demo", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--file", default=None,
                   help="path to a student-filled skeleton to check instead "
                        "of the mainline engine")
    args = p.parse_args()
    if args.scan_demo:
        scan_demo()
        return 0
    rc = check(args.file)
    if rc == 0 and args.gpu:
        gpu_check()
    return rc


if __name__ == "__main__":
    sys.exit(main())
