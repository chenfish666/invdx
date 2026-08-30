"""Tests for grating_coupler.energy_budget — the box-face power accounting that turns a
one-off manual "where did the power go" tally into a first-class, checked
measurement (see grating_coupler.py's "Energy budget" section for the full history).

All CPU. Four independent things are pinned here, matching the four
judgments that are practically testable without a full GPU-scale run
(judgment #1 — PhasorDetector not PoyntingFluxDetector — is a code-review
fact enforced by build_scene's implementation, not something a unit test
can observe from the outside; judgment #5 — no transmitted-past-grating
channel — is a docstring fact about wg_slab's extent, not a numeric one):

  * judgment #2 (signed vs abs()): a synthetic pure +x/-x traveling TE0
    mode, read by BOTH the mode-overlap formula and the new raw signed-flux
    formula, must agree to ~1e-6 or better — the positive control that
    keeps the two formulas honest about each other.
  * judgment #3 (abstention): `_box_bounds` must RAISE, not return a
    plausible-looking box, when the box would clip the PML or enclose the
    fiber-side source.
  * judgment #4 (dual threshold): `check_energy_closure` must flag BOTH a
    residual that's too big (books don't balance) and one that's
    suspiciously too small (implausible for a finite pulsed run).
  * end-to-end wiring: one real (tiny, coarse) fdtdx run through
    `energy_budget` produces the exact interface contract's keys, with
    port_face_net_in > 0 as asserted by the interface.

The tiny scene config below is the SAME coarse-grid/short-run recipe
tests/test_datasets.py, tests/test_tolerance_report.py and
tests/test_s11_fom.py already use and keep CPU-fast; it is NOT fine enough
for closure_residual_frac_of_input to land inside the "healthy" 0.1%-0.5%
band (a hand check at 2x finer grid / 6x longer run time confirmed the
residual shrinks from -2019% to -5.7% and the box's x-faces converge
towards port_face_net_in as expected — i.e. the formulas are correct and
the discrepancy is grid/run-time resolution, not a bug — but that
convergence run costs ~3 minutes on CPU and is deliberately NOT part of
this fast suite). The end-to-end test below therefore checks structure and
sign, not the closure numbers landing in the healthy band.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("fdtdx")

from invdx.problems import grating_coupler
from invdx.problems.grating_coupler import GratingCouplerConfig


def _tiny_cfg(**overrides):
    """Same tiny quasi-2D grating_coupler scene as test_datasets.py::_tiny_cfg (coarse
    grid, short run — keeps this fast); see this module's docstring for why
    the closure residual is not asserted to land in the healthy band."""
    cfg = GratingCouplerConfig(
        spacing_um=0.05,
        sim_time_s=0.05e-12,
        L_design=6.0,
        pad_x=2.0,
        dpml=0.6,
        t_box=1.5,
        t_sub=0.8,
        air_above=2.0,
        x_mon_wg=-4.0,
        x_src_wg=-4.5,
        design_grid_per_um=20,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------
# judgment #2 — signed flux vs mode overlap on a synthetic pure mode
# --------------------------------------------------------------------------


def test_signed_flux_matches_mode_overlap_for_pure_traveling_mode():
    """Positive control: a field that IS EXACTLY a pure traveling TE0 mode
    must give the SAME absolute power whether read by the existing
    mode-overlap formula (`overlap_power_directional`) or the new raw
    signed-flux formula (`signed_poynting_flux_x`) — both reduce
    algebraically to 0.5*Re(sum(Em*conj(Hm)))*dl for E=Em,H=Hm. This is
    what proves the two "machines" (mode-projected CE numerators and
    box-face energy accounting) share one absolute power scale, so an
    energy budget may safely mix them. float32 + an arbitrary global phase
    to match what a real PhasorDetector actually returns (fdtdx hardcodes
    float32; a real field is complex with an arbitrary phase reference)."""
    cfg = GratingCouplerConfig()
    zs = np.linspace(-1.25, 1.25, 4001)
    dl = float(zs[1] - zs[0])
    Em, Hm_fwd, _ = grating_coupler.slab_te0_mode(zs, 0.0, cfg)
    phase = np.exp(1j * 0.37)   # arbitrary — must cancel out of both formulas
    Em32 = (Em * phase).astype(np.complex64)
    Hm32 = (Hm_fwd * phase).astype(np.complex64)

    for Hm, direction in ((Hm32, "+x"), (-Hm32, "-x")):
        p_overlap = grating_coupler.overlap_power_directional(Em32, Hm, Em32, Hm, dl)
        p_flux = grating_coupler.signed_poynting_flux_x(Em32, Hm, dl)
        assert p_overlap > 0, direction
        rel_diff = abs(p_overlap - abs(p_flux)) / p_overlap
        assert rel_diff < 1e-6, (direction, p_overlap, p_flux, rel_diff)
        # sign: +x travel -> positive flux, -x travel -> negative flux
        expected_sign = 1.0 if direction == "+x" else -1.0
        assert p_flux * expected_sign > 0, (direction, p_flux)


# --------------------------------------------------------------------------
# judgment #4 — dual-threshold closure gate
# --------------------------------------------------------------------------


def test_check_energy_closure_dual_threshold():
    healthy = grating_coupler.check_energy_closure(0.002)
    assert healthy["status"] == "ok" and healthy["ok"] is True

    too_big = grating_coupler.check_energy_closure(0.02)
    assert too_big["status"] == "fail_high" and too_big["ok"] is False
    assert "balance" in too_big["message"].lower()

    # sign-independent (abs of residual)
    neg_big = grating_coupler.check_energy_closure(-0.02)
    assert neg_big["status"] == "fail_high"

    # exactly at hi is still "ok" (strict >)
    at_hi = grating_coupler.check_energy_closure(0.005, hi=0.005)
    assert at_hi["status"] == "ok"


def test_lower_gate_does_not_fire_on_a_converged_residual():
    """The lower gate must not punish a fine grid.

    A grid-refinement study shows this residual converging as the cells
    shrink: 36.55%,
    3.95%, 0.72%, 0.1505% at spacing 0.080/0.040/0.020/0.010 um, order 2.26.
    The gate's earlier fixed 0.1% floor would have failed the 0.005 um run
    (~0.031% by that order) purely for being better resolved. Every value
    on that measured curve, and the finer ones it implies, must pass.
    """
    for residual in (0.3655, 0.0395, 0.0072, 0.001505, 3.1e-4, 6.6e-5, 1e-6):
        got = grating_coupler.check_energy_closure(residual, hi=0.5)
        assert got["ok"] is True, (residual, got["status"], got["message"])


def test_lower_gate_fires_on_algebraic_cancellation():
    """Exact zero, and ulp-level residuals, are the bug this gate names."""
    assert grating_coupler.check_energy_closure(0.0)["status"] == "fail_low"

    # With the face scale supplied, the floor is a few ulp of the largest
    # term: a residual there cannot come from resolution.
    faces = (1.0e-3, -4.0e-4, 2.0e-4, -8.0e-4)
    port = 1.0e-3
    eps = float(np.finfo(np.float32).eps)
    below = 2.0 * eps          # well under 8 ulp of max|face| / port == 8*eps
    above = 100.0 * eps
    got_lo = grating_coupler.check_energy_closure(below, face_terms=faces,
                                       port_face_net_in=port)
    assert got_lo["status"] == "fail_low", got_lo
    got_ok = grating_coupler.check_energy_closure(above, face_terms=faces,
                                       port_face_net_in=port)
    assert got_ok["ok"] is True, got_ok


def test_closure_accounting_is_sensitive_to_a_broken_face():
    """Negative control: the books must NOTICE a face that is wrong.

    The lower gate is silent by design, so on its own it cannot tell a
    correct implementation from one whose sum is dead. Perturb one face by
    a known amount and require the residual to move by that amount and the
    high gate to fire — without this, "closure ok" only means the number
    was small, not that the check can see anything.
    """
    # These must NOT sum to zero: an exactly-cancelling set is the bug the
    # lower gate exists to catch, so it would fail this test for the right
    # reason and hide whether the perturbation is noticed at all. The first
    # draft here was [1e-3, -4e-4, 2e-4, -8e-4] and did sum to zero, which
    # is not a coincidence -- round numbers are what a person reaches for,
    # and round numbers cancel. Any gate whose criterion IS cancellation
    # will therefore fire on hand-picked test data more often than on real
    # data, so give it an untidy mantissa (or take the fluxes from an
    # actual run) and it is testing the gate rather than the habit.
    faces = [1.0e-3, -4.0e-4, 2.0e-4, -7.98e-4]
    port = 1.0e-3
    residual = sum(faces) / port
    assert abs(residual) == pytest.approx(2.0e-3, rel=1e-9)
    assert grating_coupler.check_energy_closure(residual, face_terms=faces,
                                     port_face_net_in=port)["ok"] is True

    for i in range(len(faces)):
        broken = list(faces)
        broken[i] += 0.02 * port          # 2% of the input power, one face
        moved = sum(broken) / port
        assert abs(moved - residual) == pytest.approx(0.02, rel=1e-9)
        got = grating_coupler.check_energy_closure(moved, face_terms=broken,
                                        port_face_net_in=port)
        assert got["status"] == "fail_high", (i, got)


# --------------------------------------------------------------------------
# judgment #3 — abstention: box must exclude PML and the source, or raise
# --------------------------------------------------------------------------


def test_box_bounds_ok_for_default_config():
    x_lo, x_hi, z_lo, z_hi = grating_coupler._box_bounds(GratingCouplerConfig())
    assert x_lo < 0 < x_hi
    assert z_lo < 0 < z_hi


def test_box_bounds_raises_when_top_face_would_sit_in_pml():
    """air_above too thin: the box's top face (fiber_line_y - 0.4) no
    longer clears the top PML boundary (t_si + air_above)."""
    cfg = GratingCouplerConfig(air_above=0.1)
    with pytest.raises(ValueError, match="PML"):
        grating_coupler._box_bounds(cfg)


def test_box_bounds_raises_when_box_would_enclose_the_source():
    """src_beam_y too low: the box's top face would sit ABOVE the fiber
    source plane, i.e. the source would be inside the box — Poynting's
    theorem then no longer implies zero net outward flux for a lossless
    interior (a source contributes real injected power)."""
    cfg = GratingCouplerConfig(src_beam_y=0.5)
    with pytest.raises(ValueError, match="source"):
        grating_coupler._box_bounds(cfg)


def test_box_bounds_raises_when_geometry_is_degenerate():
    """L_design/pad_x too small: the fixed 0.25 um inset from the PML
    boundary no longer fits — x_lo would land to the right of x_hi."""
    cfg = GratingCouplerConfig(L_design=0.1, pad_x=0.05)
    with pytest.raises(ValueError, match="degenerate"):
        grating_coupler._box_bounds(cfg)


# --------------------------------------------------------------------------
# port_face_net_in > 0 assertion (pure-numpy, no fdtdx run)
# --------------------------------------------------------------------------


def test_energy_budget_raises_when_port_net_flux_is_not_positive():
    """The interface contract asserts port_face_net_in > 0. Feed
    `_energy_budget_from_fields` a synthetic port field that is purely the
    BACKWARD (+x, into-the-grating) mode — net flow at the port is then
    the wrong way — and confirm it raises instead of returning a budget."""
    cfg = GratingCouplerConfig()
    zs = np.linspace(-1.25, 1.25, 501)
    dl = float(zs[1] - zs[0])
    Em, Hm_fwd, _ = grating_coupler.slab_te0_mode(zs, 0.0, cfg)
    Em, Hm_fwd = Em.astype(np.complex64), Hm_fwd.astype(np.complex64)
    zero = np.zeros_like(Em)
    box_bounds = grating_coupler._box_bounds(cfg)

    with pytest.raises(RuntimeError, match="port_face_net_in"):
        grating_coupler._energy_budget_from_fields(
            cfg, zs, Em, Hm_fwd,           # port field = pure +x (backward)
            zero, zero, zero, zero, zero, zero, zero, zero,
            p_in=1.0, box_bounds=box_bounds)


# --------------------------------------------------------------------------
# end-to-end: one real (tiny) fdtdx run through the full detector wiring
# --------------------------------------------------------------------------


def test_energy_budget_end_to_end_structure_and_signs():
    cfg = _tiny_cfg()
    teeth = grating_coupler.uniform_grating_teeth(cfg, period=0.575, duty=0.5)

    eb = grating_coupler.energy_budget(cfg, teeth)

    # exact interface contract (do not invent/rename fields)
    required = {
        "denominator", "P_fwd", "P_back", "n_eff", "port_face_net_in",
        "face_out_xlo", "face_out_xhi", "face_out_zlo", "face_out_zhi",
        "closure_sum_outward", "closure_residual_frac_of_input",
        "injection_purity_check",
    }
    assert required.issubset(eb.keys())
    assert eb["denominator"] == "P_fwd = forward slab-TE0 overlap at wg_mon"

    for key in ("P_fwd", "P_back", "n_eff", "port_face_net_in",
               "face_out_xlo", "face_out_xhi", "face_out_zlo",
               "face_out_zhi", "closure_sum_outward",
               "closure_residual_frac_of_input", "injection_purity_check"):
        assert np.isfinite(eb[key]), key

    # interface asserts port_face_net_in > 0 — energy_budget() would have
    # raised RuntimeError otherwise, so this also holds trivially, but make
    # the contract visible here too
    assert eb["port_face_net_in"] > 0

    # closure_check ran automatically and is self-consistent with the
    # standalone function on the same residual
    assert eb["closure_check"] == grating_coupler.check_energy_closure(
        eb["closure_residual_frac_of_input"])

    # box bounds match what _box_bounds computes independently
    x_lo, x_hi, z_lo, z_hi = grating_coupler._box_bounds(cfg)
    assert eb["box_bounds_um"] == {
        "x_lo": x_lo, "x_hi": x_hi, "z_lo": z_lo, "z_hi": z_hi}

    # n_eff must be a guided-mode index (between cladding and core)
    assert cfg.n_sio2 < eb["n_eff"] < cfg.n_si

    # 0 < CE <= 1: some power reaches the target mode, not more than P_in
    assert 0.0 < eb["CE"] < 1.0
