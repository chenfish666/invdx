"""Adjoint-gradient regression for the JAX toy engine: jax.grad through the
full time evolution must match central finite differences (the toy-scale
version of gate G2). Tiny case so the whole test stays ~seconds on CPU.
"""

import numpy as np
import pytest


def test_adjoint_matches_finite_difference():
    jax = pytest.importorskip("jax")
    if not jax.config.jax_enable_x64:
        try:
            jax.config.update("jax_enable_x64", True)
        except RuntimeError:
            pytest.skip("x64 must be enabled before jax arrays exist")
    import jax.numpy as jnp

    from invdx.toy import fdtd2d_jax

    nx = ny = 80
    dx, steps, fstar = 0.02, 500, 2.0
    kw = dict(nx=nx, ny=ny, dx=dx, steps=steps,
              source={"i": 15, "j": 40, "t0": 0.3, "spread": 0.08,
                      "fcen": fstar},
              line_probes={"out": ("x", 65, 20, 60)})
    base = jnp.ones((nx, ny))
    gx, gy = slice(35, 45), slice(30, 50)
    dt = 0.5 * dx

    def objective(theta):
        eps = base.at[gx, gy].set(1.0 + 3.0 * jax.nn.sigmoid(theta))
        _, ys = fdtd2d_jax.simulate(**kw, eps=eps)
        E, H = ys["out"]
        P = fdtd2d_jax.line_flux_spectrum_jnp(E, H, (fstar,), dt, dx,
                                              sign=1.0)
        return P[0]

    theta0 = jnp.zeros((10, 20))
    f = jax.jit(objective)
    g = jax.jit(jax.grad(objective))(theta0)

    h = 1e-4
    for ij in ((3, 7), (8, 15)):
        e = jnp.zeros_like(theta0).at[ij].set(h)
        fd = (float(f(theta0 + e)) - float(f(theta0 - e))) / (2 * h)
        ad = float(g[ij])
        assert abs(fd - ad) <= 1e-5 * max(abs(fd), abs(ad)), (ij, fd, ad)
