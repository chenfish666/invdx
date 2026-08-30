"""filters_np (autograd, authoritative) vs filters_jax must agree — values
AND gradients. This is what lets the jax twins enter the fdtdx differentiable
path without re-earning trust from scratch.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from invdx.fab import filters_np, filters_jax

RADIUS, GRID = 0.13, 100
NX = 200


def _x():
    rng = np.random.default_rng(0)
    return np.clip(0.5 + 0.2 * rng.standard_normal(NX), 0, 1)


def test_mapping_values_match():
    W = filters_np.conic_filter_matrix(NX, RADIUS, GRID)
    x = _x()
    y_np = filters_np.make_mapping(W)(x, 0.5, 8.0)
    y_jax = np.asarray(filters_jax.make_mapping(W)(
        jax.numpy.asarray(x, dtype=jax.numpy.float64)
        if jax.config.jax_enable_x64 else jax.numpy.asarray(x), 0.5, 8.0))
    # float32 trace unless x64 enabled; tolerance chosen accordingly
    tol = 1e-12 if jax.config.jax_enable_x64 else 5e-6
    assert np.max(np.abs(y_np - y_jax)) < tol


def test_mapping_gradients_match():
    import autograd
    import autograd.numpy as npa

    W = filters_np.conic_filter_matrix(NX, RADIUS, GRID)
    x = _x()
    g_np = autograd.grad(
        lambda z: npa.sum(filters_np.make_mapping(W)(z, 0.5, 8.0) ** 2))(x)
    g_jax = np.asarray(jax.grad(
        lambda z: (filters_jax.make_mapping(W)(z, 0.5, 8.0) ** 2).sum())(
            jax.numpy.asarray(x)))
    tol = 1e-12 if jax.config.jax_enable_x64 else 5e-5
    assert np.max(np.abs(g_np - g_jax)) < tol


def test_softmin_matches():
    v = np.array([0.3, 0.9, 0.5])
    s_np = filters_np.softmin(v, 30.0)
    s_jax = float(filters_jax.softmin(v, 30.0))
    assert abs(s_np - s_jax) < 1e-6


def test_conic_filter_transform_matches_reference():
    """ConicFilter1D (through the fdtdx transform machinery, minus init) must
    reduce to W @ x along the chosen axis."""
    from invdx.fab.transforms import ConicFilter1D

    W = filters_np.conic_filter_matrix(NX, RADIUS, GRID)
    x = _x()
    tr = ConicFilter1D(radius_um=RADIUS, axis=0)
    # bypass init_module: inject the single field the __call__ path reads
    tr = tr.aset("_single_voxel_size", (1e-6 / GRID, 1e-6, 1e-6),
                 create_new_ok=True)
    out = tr({"p": jax.numpy.asarray(x.reshape(NX, 1, 1))})["p"]
    ref = (W @ x).reshape(NX, 1, 1)
    assert np.max(np.abs(np.asarray(out) - ref)) < 5e-6
