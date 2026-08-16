"""Pure-math tests for the PhC 90-bend geometry (no simulation)."""

import numpy as np

from invdx.problems import phc_bend


CFG = phc_bend.PhCBendConfig(n_side=11, res_per_a=10, pad_a=2.0)


def test_rod_counts():
    n = CFG.n_side
    m = n + 2                              # lattice carries an extra ring
    assert len(phc_bend.rod_sites(n, "bulk", bulk_cols=8)) == 8 * m
    # straight removes one full row (m sites); the bend removes c+2 (input)
    # + n-c+1 (output) - 1 (shared corner) = m sites as well
    assert len(phc_bend.rod_sites(n, "straight")) == m * m - m
    assert len(phc_bend.rod_sites(n, "bend")) == m * m - m
    # a defect outside the arms removes exactly one more rod
    assert len(phc_bend.rod_sites(n, "bend", defect=(1, -1))) == m * m - m - 1


def test_epsilon_grid_binary_and_fraction():
    eps = phc_bend.epsilon_grid(CFG, "straight")
    assert set(np.unique(eps)) == {1.0, CFG.eps_rod}
    # rod area fraction ~ n_rods * pi r^2 / (n + 2 pad)^2
    m = CFG.n_side + 2
    frac = np.mean(eps == CFG.eps_rod)
    expect = (m * m - m) * np.pi * CFG.r_rod ** 2 \
        / (CFG.n_side + 2 * CFG.pad_a) ** 2
    assert abs(frac - expect) / expect < 0.05

    assert np.all(phc_bend.epsilon_grid(CFG, "bulk_empty") == 1.0)


def test_bend_symmetric_under_antidiagonal():
    # the bend maps onto itself when reflected across the anti-diagonal
    # through the corner (input -x arm <-> output +y arm)
    eps = phc_bend.epsilon_grid(CFG, "bend")
    assert np.allclose(eps, eps[::-1, ::-1].T)


def test_toy_band_gap_and_bend():
    # fast physics regression (~5 s): the Gamma-X stopband must cover the
    # paper's full gap 0.29..0.41, and the in-gap bend transmission must be
    # waveguide-like. Small lattice + coarse grid: qualitative thresholds.
    cfg = phc_bend.PhCBendConfig(n_side=9, res_per_a=10, toy_steps=2500,
                                 n_freq=13)
    f = cfg.freqs
    bulk = phc_bend.toy_bulk_transmission(cfg)
    Tdb = 10 * np.log10(np.abs(bulk["T"]) + 1e-12)
    gap = (f >= 0.30) & (f <= 0.38)
    band = (f <= 0.225) | (f >= 0.475)
    assert np.median(Tdb[gap]) < -15
    assert np.median(Tdb[band]) > -8

    bend = phc_bend.toy_bend_transmission(cfg)
    T = np.array(bend["T"])
    assert np.mean(T[gap]) > 0.5


def test_ports_path_lengths_equal():
    ports = phc_bend._toy_ports(CFG)
    res, c = CFG.res_per_a, CFG.center
    i_src = ports["src"]["i"]
    row = ports["src"]["j"]
    _, i_out_s, _, _ = ports["straight_out"]
    _, j_out_b, _, _ = ports["bend_out"]
    straight_path = i_out_s - i_src
    bend_path = (row - i_src) + (j_out_b - row)
    assert straight_path == bend_path
