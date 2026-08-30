"""Lesson 1: port the toy engine to JAX (your workbook).

This file is the **blanked-out skeleton**: the scaffolding (state init, the
lax.scan loop, output packing) is already written; the three blanks A/B/C are
the physics itself and are yours to fill in -- each maps to the identically
named section of fdtd2d.py (the numpy version). Lesson text, concepts, hints
and checkpoints live in README.md in this folder; the checker is:

    python scripts/08_toy_jax_lesson1.py \\
        --file tutorials/01-jax-port/fdtd2d_jax_skeleton.py

Once filled in, it uses the phc_bend gap measurement to verify your version is
bit-for-bit identical to the numpy one (the difference should be ~1e-15 at
float64).

The one conceptual difference vs numpy: JAX arrays are immutable (functional
style).
    numpy:  Hx -= ...        Ez[1:-1,1:-1] += ...
    JAX:    Hx = Hx - ...    Ez = Ez.at[1:-1,1:-1].add(...)
The time loop goes to jax.lax.scan -- it compiles the whole loop into one XLA
program, which is exactly what later lets jax.grad differentiate "through the
entire time evolution".
"""

import jax
import jax.numpy as jnp
import numpy as np


def gaussian_pulse(t, t0, spread, fcen=None):
    """Same as the numpy version, but in jnp (t may be the whole time axis)."""
    env = jnp.exp(-(((t - t0) / spread) ** 2))
    if fcen is None:
        return env
    return jnp.sin(2 * jnp.pi * fcen * (t - t0)) * env


def run(nx, ny, dx, steps, source, probes=(), courant=0.5, eps=None,
        line_probes=None):
    """Exactly the interface of invdx.toy.fdtd2d.run (new engine, same API)."""
    dt = courant * dx
    if eps is None:
        eps = jnp.ones((nx, ny))
    else:
        eps_np = np.asarray(eps, dtype=float)
        edge = np.concatenate([eps_np[0], eps_np[-1], eps_np[:, 0],
                               eps_np[:, -1]])
        if not np.allclose(edge, 1.0):
            raise ValueError("eps must be 1.0 on the outermost cells")
        eps = jnp.asarray(eps_np)
    mur = (dt - dx) / (dt + dx)

    # Precompute the whole source waveform; scan eats one value per step
    # (cheaper and cleaner than recomputing it inside the loop)
    t_axis = np.arange(steps) * dt
    amps = gaussian_pulse(jnp.asarray(t_axis), source["t0"], source["spread"],
                          source.get("fcen"))

    line_probes = line_probes or {}
    probes = tuple(tuple(p) for p in probes)

    def inject(Ez, a):
        if "j0" in source:
            return Ez.at[source["i"], source["j0"]:source["j1"]].add(a)
        return Ez.at[source["i"], source["j"]].add(a)

    def step(state, a):
        """One time step: state in, state out (the scan contract)."""
        Ez, Hx, Hy = state

        # ============== BLANK A: H update (Faraday's law) ================
        # Mirror the two "H from curl E" lines in fdtd2d.py.
        # Remember: JAX is immutable -> Hx = Hx - ... (not Hx -= ...)
        raise NotImplementedError(
            "BLANK A: update Hx and Hy from curl E -- see this lesson's "
            "README.md")
        # Hx = ...
        # Hy = ...
        # ================================================================

        # Mur needs "Ez one step ago" -- snapshot it before the interior
        # update overwrites it
        Ez_old = Ez

        # ====== BLANK B: E interior update (Ampere's law, material here) ==
        # Mirror the "E interior from curl H" block in fdtd2d.py.
        # Interior slice assignment: Ez = Ez.at[1:-1, 1:-1].add(...)
        # Do not forget to divide by eps[1:-1, 1:-1] -- the one and only
        # place the material enters.
        raise NotImplementedError(
            "BLANK B: update the Ez interior from curl H, dividing by eps -- "
            "see this lesson's README.md")
        # curl = ...
        # Ez = ...
        # ================================================================

        Ez = inject(Ez, a)

        # ======= BLANK C: first-order Mur absorbing boundary (4 edges) ====
        # Mirror the four Mur lines in fdtd2d.py. For each edge:
        #   Ez = Ez.at[edge].set(Ez_old[inner]
        #                        + mur * (Ez[inner] - Ez_old[edge]))
        raise NotImplementedError(
            "BLANK C: set the four Mur boundary edges -- see this lesson's "
            "README.md")
        # Ez = ...
        # Ez = ...
        # Ez = ...
        # Ez = ...
        # ================================================================

        # Per-step observables (scan stacks them along the time axis for you)
        out = {"probes": jnp.stack([Ez[p] for p in probes]) if probes
               else jnp.zeros((0,)),
               "energy": 0.5 * (jnp.sum(eps * Ez ** 2) + jnp.sum(Hx ** 2)
                                + jnp.sum(Hy ** 2)) * dx * dx}
        for name, (axis, k, lo, hi) in line_probes.items():
            if axis == "x":
                out[name] = (Ez[k, lo:hi], Hy[min(k, nx - 2), lo:hi])
            else:
                out[name] = (Ez[lo:hi, k], Hx[lo:hi, min(k, ny - 2)])
        return (Ez, Hx, Hy), out

    init = (jnp.zeros((nx, ny)), jnp.zeros((nx, ny - 1)),
            jnp.zeros((nx - 1, ny)))
    (Ez, _, _), ys = jax.lax.scan(step, init, amps)

    # Pack into exactly the numpy version's output shape (downstream code
    # never needs to know the engine changed)
    return {"Ez": np.asarray(Ez), "t": t_axis,
            "probes": {p: np.asarray(ys["probes"][:, i])
                       for i, p in enumerate(probes)},
            "energy": np.asarray(ys["energy"]),
            "lines": {name: {"E": np.asarray(ys[name][0]),
                             "H": np.asarray(ys[name][1])}
                      for name in line_probes}}
