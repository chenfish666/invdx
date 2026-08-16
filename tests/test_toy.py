"""Physics sanity tests for the toy 2D FDTD (Layer C, milestone M-toy-1)."""

import numpy as np

from invdx.toy import fdtd2d


def _sim(steps=500, nx=240, ny=240, probes=((150, 120), (200, 120))):
    return fdtd2d.run(
        nx=nx, ny=ny, dx=0.01, steps=steps,
        source={"i": 120, "j": 120, "t0": 0.20, "spread": 0.06},
        probes=probes,
    )


def test_pulse_travels_at_c():
    # differential timing between two probes cancels the systematic shift of
    # the 2D radiated waveform (a soft point source in 2D is not a clean
    # Gaussian at the probe) — only the propagation speed remains
    out = _sim()
    t = out["t"]
    t1 = t[np.argmax(np.abs(out["probes"][(150, 120)]))]
    t2 = t[np.argmax(np.abs(out["probes"][(200, 120)]))]
    v = 0.50 / (t2 - t1)                    # probes are 0.5 apart, c = 1
    assert abs(v - 1.0) < 0.02


def test_energy_bounded_and_absorbed():
    out = _sim(steps=1200)
    e = out["energy"]
    i_peak = int(np.argmax(e))
    # energy injection ends with the pulse; nothing may grow afterwards
    assert np.all(e[i_peak:] <= e[i_peak] * (1 + 1e-9))
    # Mur boundaries absorb the outgoing wave (first-order: imperfect but real)
    assert e[-1] < 0.2 * e[i_peak]


def test_field_stays_finite():
    out = _sim(steps=300)
    assert np.all(np.isfinite(out["Ez"]))


def test_pulse_slows_in_dielectric():
    # same differential-timing trick as test_pulse_travels_at_c, but both
    # probes sit inside an eps=4 block: the measured speed must be c/n = 0.5
    nx = ny = 240
    eps = np.ones((nx, ny))
    eps[135:239, 1:239] = 4.0           # edges stay vacuum (Mur requirement)
    out = fdtd2d.run(
        nx=nx, ny=ny, dx=0.01, steps=900,
        source={"i": 120, "j": 120, "t0": 0.20, "spread": 0.06},
        probes=((150, 120), (200, 120)), eps=eps)
    t = out["t"]
    t1 = t[np.argmax(np.abs(out["probes"][(150, 120)]))]
    t2 = t[np.argmax(np.abs(out["probes"][(200, 120)]))]
    v = 0.50 / (t2 - t1)
    assert abs(v - 0.5) < 0.02


def test_eps_must_be_vacuum_on_edges():
    import pytest
    eps = np.full((60, 60), 4.0)
    with pytest.raises(ValueError):
        fdtd2d.run(nx=60, ny=60, dx=0.01, steps=10,
                   source={"i": 30, "j": 30, "t0": 0.1, "spread": 0.03},
                   eps=eps)
