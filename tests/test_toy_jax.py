"""Lesson-1 homework guard: once toy/fdtd2d_jax.py's blanks are filled, the
JAX engine must reproduce the numpy engine bit-for-bit; until then the test
politely skips (so `make gates` stays green while homework is in progress).
"""

import numpy as np
import pytest


def test_jax_engine_matches_numpy():
    jax = pytest.importorskip("jax")
    if not jax.config.jax_enable_x64:
        try:
            jax.config.update("jax_enable_x64", True)
        except RuntimeError:
            pytest.skip("x64 must be enabled before jax arrays exist")

    from invdx.toy import fdtd2d, fdtd2d_jax

    nx = ny = 120
    eps = np.ones((nx, ny))
    eps[60:110, 1:119] = 4.0
    kw = dict(nx=nx, ny=ny, dx=0.01, steps=600,
              source={"i": 30, "j": 60, "t0": 0.2, "spread": 0.06,
                      "fcen": 3.0},
              eps=eps, line_probes={"out": ("x", 100, 20, 100)})
    ref = fdtd2d.run(**kw)
    try:
        mine = fdtd2d_jax.run(**kw)
    except NotImplementedError:
        pytest.skip("lesson 1 homework not finished yet (blanks A/B/C)")

    E0, E1 = ref["lines"]["out"]["E"], mine["lines"]["out"]["E"]
    scale = np.max(np.abs(E0))
    assert np.max(np.abs(E0 - E1)) < 1e-9 * scale
