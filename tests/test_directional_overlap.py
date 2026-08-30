"""Directional-overlap sign conventions — the class of bug that produced a
-65 dB phantom (upward target written with the downward H sign) and grating_coupler's
2x CE error. Pure math, no simulation.
"""

import numpy as np
import pytest

from invdx.problems import grating_coupler

LAM, W0, TH = 1.31, 4.6, 10.0


def _upward_field(xs, kx_sign):
    """Synthetic tilted upward-traveling wave sampled on a horizontal line:
    E along y, Hx = -cos(theta)*Ey (upward), lateral phase kx_sign*k0*sin."""
    k0 = 2 * np.pi / LAM
    th = np.deg2rad(TH)
    env = np.exp(-((xs / W0) ** 2))
    E = env * np.exp(1j * kx_sign * k0 * np.sin(th) * xs)
    H = -np.cos(th) * E
    return E, H


def test_upward_target_accepts_upward_wave():
    xs = np.linspace(-8, 8, 1281)
    dx = xs[1] - xs[0]
    E, H = _upward_field(xs, kx_sign=-1.0)
    Eg, Hg = grating_coupler.gaussian_mode_tilted(xs, 0.0, W0, LAM, TH, kx_sign=-1.0)
    P = grating_coupler.overlap_power_directional(E, H, Eg, Hg, dx)
    Pm = abs(0.5 * np.real(np.sum(Eg * np.conj(Hg))) * dx)
    assert P == pytest.approx(Pm, rel=1e-10)   # perfect match -> full power


def test_upward_target_rejects_downward_wave():
    xs = np.linspace(-8, 8, 1281)
    dx = xs[1] - xs[0]
    E, H = _upward_field(xs, kx_sign=-1.0)
    P_up = grating_coupler.overlap_power_directional(
        E, H, *grating_coupler.gaussian_mode_tilted(xs, 0.0, W0, LAM, TH, -1.0), dx)
    # downward wave: flip H sign
    P_down = grating_coupler.overlap_power_directional(
        E, -H, *grating_coupler.gaussian_mode_tilted(xs, 0.0, W0, LAM, TH, -1.0), dx)
    assert P_down < 1e-18 * P_up


def test_wrong_kx_sign_discriminated():
    xs = np.linspace(-8, 8, 1281)
    dx = xs[1] - xs[0]
    E, H = _upward_field(xs, kx_sign=-1.0)
    P_match = grating_coupler.overlap_power_directional(
        E, H, *grating_coupler.gaussian_mode_tilted(xs, 0.0, W0, LAM, TH, -1.0), dx)
    P_wrong = grating_coupler.overlap_power_directional(
        E, H, *grating_coupler.gaussian_mode_tilted(xs, 0.0, W0, LAM, TH, +1.0), dx)
    # 2*k0*sin(10deg) lateral phase mismatch across a 4.6um waist
    assert P_wrong < 0.05 * P_match
