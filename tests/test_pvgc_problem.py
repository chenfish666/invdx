"""Pure-math tests for the PVGC problem layer (no simulation, no GPU)."""

import numpy as np
import pytest

from invdx.problems import pvgc

CFG = pvgc.PVGCConfig()


def test_te0_neff_bracketed_and_consistent():
    neff = pvgc.slab_te0_neff(CFG.lam_c, CFG.t_si, CFG.n_si, CFG.n_sio2, 1.0)
    assert CFG.n_sio2 < neff < CFG.n_si
    # dispersion relation residual ~ 0 at the root
    k0 = 2 * np.pi / CFG.lam_c
    kap = k0 * np.sqrt(CFG.n_si ** 2 - neff ** 2)
    gs = k0 * np.sqrt(neff ** 2 - CFG.n_sio2 ** 2)
    gc = k0 * np.sqrt(neff ** 2 - 1.0)
    resid = kap * CFG.t_si - np.arctan(gs / kap) - np.arctan(gc / kap)
    assert abs(resid) < 1e-9


def test_mode_field_continuous_and_decaying():
    zs = np.linspace(-1.5, 1.5, 3001)
    Ey, Hz, neff = pvgc.slab_te0_mode(zs, 0.0, CFG)
    # continuity at both interfaces (grid straddles them)
    i0 = np.argmin(np.abs(zs - 0.0))
    i1 = np.argmin(np.abs(zs - CFG.t_si))
    assert abs(Ey[i0 + 1] - Ey[i0 - 1]) < 0.02
    assert abs(Ey[i1 + 1] - Ey[i1 - 1]) < 0.02
    # evanescent tails decay
    assert abs(Ey[0]) < 0.01 * np.max(np.abs(Ey))
    assert abs(Ey[-1]) < 0.05 * np.max(np.abs(Ey))
    assert np.allclose(Hz, neff * Ey)


def test_directional_overlap_self_normalization():
    zs = np.linspace(-1.25, 1.25, 2001)
    dz = zs[1] - zs[0]
    Em, Hm, _ = pvgc.slab_te0_mode(zs, 0.0, CFG)
    # forward mode measured against itself -> exactly the mode power
    Pm = 0.5 * np.real(np.sum(Em * np.conj(Hm))) * dz
    P = pvgc.overlap_power_directional(Em, Hm, Em, Hm, dz)
    assert P == pytest.approx(Pm, rel=1e-12)
    # forward fields measured against the backward mode -> ~0
    P_back = pvgc.overlap_power_directional(Em, Hm, Em, -Hm, dz)
    assert P_back < 1e-20 * Pm


def test_teeth_builders():
    teeth = pvgc.uniform_grating_teeth(CFG, period=0.575, duty=0.5)
    assert len(teeth) == 17            # int(10 // 0.575)
    x0, w = teeth[0]
    assert x0 == pytest.approx(-5.0)
    assert w == pytest.approx(0.2875)

    rho = np.zeros(1000)
    rho[0:50] = 1                       # one 0.5 um tooth at the window start
    t2 = pvgc.profile_teeth(CFG, rho)
    assert len(t2) == 1
    assert t2[0][0] == pytest.approx(-5.0)
    assert t2[0][1] == pytest.approx(0.5)
