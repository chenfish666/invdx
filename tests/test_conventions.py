"""Guards in invdx.engines.conventions, plus their grating_coupler FOM-factory call
sites: FOM spectral sampling coarse enough to step over a resonance leaves
the optimizer free to collapse the response into the gaps between samples.

The factory tests only exercise the guard, which fires BEFORE any fdtdx
scene is built, so they stay CPU-cheap; they still need fdtdx/jax importable
because importing problems.grating_coupler pulls both in.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from invdx.engines.conventions import assert_fom_sampling_covers_band


def _max_adjacent_gap_nm(lams_um):
    """Same spacing rule as the grating_coupler FOM factories: the WIDEST adjacent gap
    decides, computed here independently so the test does not import the
    factory's own arithmetic."""
    s = sorted(lams_um)
    return max(b - a for a, b in zip(s[:-1], s[1:])) * 1e3


def test_guard_rejects_coarse_sampling():
    with pytest.raises(ValueError, match="coarser"):
        assert_fom_sampling_covers_band(20.0, 23.0)  # 20 > 23/2


def test_guard_accepts_dense_sampling():
    assert_fom_sampling_covers_band(10.0, 23.0)   # 10 <= 23/2
    assert_fom_sampling_covers_band(11.5, 23.0)   # boundary: exactly half


def test_sparse_edge_case_is_caught_by_max_gap():
    """[1.3000, 1.3005, 1.4000] um: two samples 0.5 nm apart plus a 99.5 nm
    hole at the band edge. With min() this list passed (measured); the
    widest-gap rule must block it."""
    lams = [1.3000, 1.3005, 1.4000]
    gap_nm = _max_adjacent_gap_nm(lams)
    assert gap_nm == pytest.approx(99.5, abs=1e-9)

    with pytest.raises(ValueError, match="coarser"):
        assert_fom_sampling_covers_band(gap_nm, 23.0)

    # and this is exactly why min() was a bug: the NARROWEST gap of the same
    # list (0.5 nm) sails through the guard, so a min()-based caller lets an
    # arbitrarily large hole into the band
    min_gap_nm = min(b - a for a, b in zip(sorted(lams)[:-1],
                                           sorted(lams)[1:])) * 1e3
    assert min_gap_nm == pytest.approx(0.5, abs=1e-9)
    assert_fom_sampling_covers_band(min_gap_nm, 23.0)  # must NOT raise


# ---------------------------------------------------------------------------
# Call sites: both grating_coupler FOM factories must run the guard on the widest gap,
# and must do so before building any scene (these calls return in seconds on
# CPU precisely because the ValueError preempts scene construction).
# ---------------------------------------------------------------------------

fdtdx = pytest.importorskip("fdtdx")

from invdx.problems import grating_coupler  # noqa: E402  (needs fdtdx importable)

SPARSE_EDGE_LAMS = [1.3000, 1.3005, 1.4000]


def test_fom_factory_2d_blocks_sparse_edge_list():
    cfg = grating_coupler.GratingCouplerConfig()
    with pytest.raises(ValueError, match="coarser"):
        grating_coupler.make_ce_value_and_grad(cfg, p_in=1.0, lams=SPARSE_EDGE_LAMS)


def test_fom_factory_3d_blocks_sparse_edge_list():
    cfg = grating_coupler.GratingCouplerConfig()
    with pytest.raises(ValueError, match="coarser"):
        grating_coupler.make_ce_value_and_grad_3d(cfg, p_in=[1.0] * len(SPARSE_EDGE_LAMS),
                                       lams=SPARSE_EDGE_LAMS)
