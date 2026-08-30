"""Unit tests for invdx.modes — the power-convention invariants.

test_overlap_self_normalization is the invariant that caught grating_coupler's 2x CE
bug; if it ever fails, suspect a convention change before anything else.
"""

import numpy as np

from invdx import modes
from invdx.engines import conventions

W0 = 4.6  # standard single-mode fiber waist radius (um)


def _mode():
    xs = np.linspace(-7, 7, 561)
    return modes.gaussian_mode_1d(xs, 0.0, W0) + (xs,)


def test_overlap_self_normalization():
    Eg, Hg, Pg, dx, _ = _mode()
    T = modes.overlap_power(Eg, Hg, Eg, Hg, Pg, dx)
    assert abs(T / Pg - 1.0) < 1e-9


def test_overlap_discriminates_mismatched_fields():
    Eg, Hg, Pg, dx, xs = _mode()
    E2 = Eg * np.cos(2 * np.pi * xs / 1.0)
    T2 = modes.overlap_power(E2, E2, Eg, Hg, Pg, dx)
    assert T2 / Pg < 0.05


def test_meep_bridge_factor_is_two():
    assert conventions.meep_to_physical_power(1.0) == 2.0
    assert conventions.MEEP_POWER_OMITS_HALF


def test_multifreq_gradient_collapse():
    dJ = np.ones((100, 3))
    g = conventions.collapse_multifreq_gradient(dJ)
    assert g.shape == (100,)
    assert np.allclose(g, 3.0)
    # single-frequency gradients pass through untouched
    g1 = conventions.collapse_multifreq_gradient(np.ones(100))
    assert g1.shape == (100,)


def test_resolution_guard():
    import pytest
    from invdx.config import BaseConfig

    cfg = BaseConfig(resolution=40, design_grid_per_um=100)
    with pytest.raises(ValueError):
        conventions.assert_resolution_covers_design_grid(cfg)
    conventions.assert_resolution_covers_design_grid(
        BaseConfig(resolution=100, design_grid_per_um=100))
