"""Unit tests for invdx.fab: conic filter, tanh projection, softmin, and the
minimum-feature / corner-dilation measurements."""

import numpy as np
import pytest

from invdx.config import BaseConfig
from invdx.fab import filters_np, measure

CFG = BaseConfig()


def test_conic_filter_rows_normalized():
    W = filters_np.conic_filter_matrix(200, CFG.filter_radius, CFG.design_grid_per_um)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_tanh_projection_step_limit():
    x = np.linspace(0, 1, 11)
    p = filters_np.tanh_projection(x, 1e3, 0.5)
    assert np.all(p[x < 0.45] < 1e-3)
    assert np.all(p[x > 0.55] > 1 - 1e-3)


def test_softmin_behaves_like_min():
    v = np.array([0.3, 0.9, 0.5])
    sm = filters_np.softmin(v, 200.0)
    assert sm <= v.min() + 1e-9
    assert abs(sm - v.min()) < 1e-2


def test_softmin_weights_sum_to_one_and_favor_min():
    v = np.array([0.3, 0.9, 0.5])
    w = filters_np.softmin_weights(v, 30.0)
    assert abs(w.sum() - 1.0) < 1e-12
    assert w.argmax() == v.argmin()


def test_min_feature_run_length_exact():
    rho = np.zeros(1000)
    rho[100:120] = 1   # 20 px = 200 nm solid
    rho[125:1000] = 1  # 5 px  = 50 nm void gap between the runs
    ms, mv = measure.min_feature_1d(rho, CFG.design_grid_per_um)
    assert ms == pytest.approx(0.20, abs=1e-9)
    assert mv == pytest.approx(0.05, abs=1e-9)


def test_cd_corner_dilation_shrinks_void():
    rho = np.zeros(1000)
    rho[100:120] = 1
    rho[125:1000] = 1
    _, mv = measure.min_feature_1d(rho, CFG.design_grid_per_um)
    b = measure.erode_dilate_1d(rho, +0.02, CFG.design_grid_per_um)
    _, mv2 = measure.min_feature_1d(b, CFG.design_grid_per_um)
    assert mv2 < mv


def test_mapping_chain_autograd_differentiable():
    import autograd
    import autograd.numpy as npa

    W = filters_np.conic_filter_matrix(200, CFG.filter_radius, CFG.design_grid_per_um)
    mapping = filters_np.make_mapping(W)
    g = autograd.grad(lambda z: npa.sum(mapping(z, 0.5, 8.0) ** 2))(
        np.full(200, 0.5))
    assert np.all(np.isfinite(g))
