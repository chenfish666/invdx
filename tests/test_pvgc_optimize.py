"""Pure-math tests for the M1 inverse-design driver (no simulation, no GPU).

Three things get pinned here, all of them silent-corruption risks:
  * the design vector <-> teeth mapping (a shifted rasterization would make
    the differentiable Device and the measurement chain describe different
    devices while both look plausible),
  * the jnp twin of the directional mode overlap against the authoritative
    numpy version (the fdtdx FOM is only as trustworthy as this parity),
  * checkpoint/resume, because a 13-hour round that cannot resume is a lost
    day and the failure only shows up when it is too late to fix.
"""

import os

import numpy as np
import pytest

from invdx import optimize
from invdx.problems import pvgc

jax = pytest.importorskip("jax")

CFG = pvgc.PVGCConfig(spacing_um=0.020)
CFG.design_grid_per_um = 50


# --------------------------------------------------------------------------
# design vector <-> teeth
# --------------------------------------------------------------------------


def test_rasterize_profile_roundtrip_is_exact_on_grid_aligned_teeth():
    """profile_teeth . rasterize_teeth == identity for grid-aligned teeth."""
    rho = np.zeros(pvgc.n_design_voxels(CFG))
    rho[10:35] = 1.0        # 0.5 um tooth
    rho[100:113] = 1.0      # 0.26 um tooth
    rho[499] = 1.0          # tooth touching the window edge
    teeth = pvgc.profile_teeth(CFG, rho)
    assert np.array_equal(pvgc.rasterize_teeth(CFG, teeth), rho)
    assert pvgc.profile_teeth(CFG, pvgc.rasterize_teeth(CFG, teeth)) == teeth


def test_rasterize_places_teeth_at_the_right_pixels():
    """A tooth is silicon exactly where the pixel CENTRE is inside it."""
    rho = pvgc.rasterize_teeth(CFG, [(-5.0, 0.5)])
    assert rho[:25].all() and not rho[25:].any()
    # an off-grid tooth snaps to whole pixels, it never smears into grey
    rho2 = pvgc.rasterize_teeth(CFG, [(-5.0 + 0.011, 0.5)])
    assert set(np.unique(rho2)) <= {0.0, 1.0}
    assert rho2.sum() == 25
    assert rho2[0] == 0.0 and rho2[1] == 1.0 and rho2[25] == 1.0


def test_rasterized_grating_keeps_every_tooth_the_same_width():
    """The width must not jitter: a 0.2875 um tooth on 0.02 um pixels rounds
    to 14 pixels for EVERY tooth. Letting it alternate 14/15 (which a
    pixel-centre rule does) costs 13 dB — the rasterization is part of the
    physics, not a formatting detail."""
    teeth = pvgc.uniform_grating_teeth(CFG, period=0.575, duty=0.5)
    rho = pvgc.rasterize_teeth(CFG, teeth)
    rendered = pvgc.profile_teeth(CFG, rho)
    assert len(rendered) == len(teeth) == 17
    assert {round(w * CFG.design_grid_per_um) for _, w in rendered} == {14}
    assert rho.sum() == 17 * 14


def test_n_design_voxels_matches_script_07_expectation():
    """Script 07 rebuilds PVGCConfig() with its default L_design, so the
    design vector length is fixed by design_grid_per_um alone."""
    assert pvgc.n_design_voxels(CFG) == int(CFG.L_design * 50) == 500


# --------------------------------------------------------------------------
# grid-snapping guard
# --------------------------------------------------------------------------


def test_design_grid_guard_rejects_the_default_spacing():
    """0.0125 um snaps t_si (0.220) to 0.225 — the Device would silently be a
    different device from the one the measurement chain describes."""
    bad = pvgc.PVGCConfig()          # spacing_um = 0.0125
    with pytest.raises(ValueError, match="t_si"):
        pvgc.assert_design_grid_snaps(bad)


def test_design_grid_guard_accepts_the_clean_grids():
    for spacing, grid in ((0.020, 50), (0.020, 25), (0.010, 100), (0.010, 50)):
        cfg = pvgc.PVGCConfig(spacing_um=spacing)
        cfg.design_grid_per_um = grid
        pvgc.assert_design_grid_snaps(cfg)


def test_design_grid_guard_rejects_design_grid_finer_than_the_mesh():
    cfg = pvgc.PVGCConfig(spacing_um=0.020)
    cfg.design_grid_per_um = 100          # 10 nm pixels on a 20 nm mesh
    with pytest.raises(ValueError):
        pvgc.assert_design_grid_snaps(cfg)


# --------------------------------------------------------------------------
# jnp twin of the measurement chain
# --------------------------------------------------------------------------


def _monitor_fields(n=125, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    H = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return E.astype(np.complex64), H.astype(np.complex64)


def test_overlap_jnp_matches_numpy_on_random_fields():
    E, H = _monitor_fields()
    Em, Hm_back, Pm, _ = pvgc.te0_target_on_monitor(CFG, E.size)
    ref = pvgc.overlap_power_directional(E, H, Em, Hm_back, CFG.spacing_um)
    got = float(pvgc.overlap_power_directional_jnp(
        jax.numpy.asarray(E), jax.numpy.asarray(H), Em, Hm_back,
        CFG.spacing_um, Pm))
    assert got == pytest.approx(ref, rel=2e-5)


def test_overlap_jnp_self_normalizes_and_rejects_the_wrong_direction():
    """The mode measured against itself carries exactly the mode power; the
    forward mode measured against the backward target carries ~none."""
    Em, Hm_back, Pm, _ = pvgc.te0_target_on_monitor(CFG, 125)
    E = jax.numpy.asarray(Em.astype(np.complex64))
    H = jax.numpy.asarray(Hm_back.astype(np.complex64))
    P = float(pvgc.overlap_power_directional_jnp(E, H, Em, Hm_back,
                                                 CFG.spacing_um, Pm))
    assert P == pytest.approx(Pm, rel=1e-5)
    P_wrong = float(pvgc.overlap_power_directional_jnp(
        E, -H, Em, Hm_back, CFG.spacing_um, Pm))
    assert P_wrong < 1e-10 * Pm


def test_te0_target_pm_matches_the_numpy_chain():
    """Pm precomputed here must equal the one the numpy overlap derives."""
    Em, Hm_back, Pm, neff = pvgc.te0_target_on_monitor(CFG, 125)
    assert Pm == pytest.approx(
        abs(0.5 * np.real(np.sum(Em * np.conj(Hm_back))) * CFG.spacing_um))
    assert CFG.n_sio2 < neff < CFG.n_si


def test_te0_target_tracks_wavelength_dispersion():
    _, _, _, neff_c = pvgc.te0_target_on_monitor(CFG, 125)
    _, _, _, neff_hi = pvgc.te0_target_on_monitor(CFG, 125, lam_um=1.36)
    assert neff_hi < neff_c          # longer wavelength -> less confined


# --------------------------------------------------------------------------
# beta schedule
# --------------------------------------------------------------------------


def test_beta_schedule_spreads_stages_evenly():
    betas = [optimize.beta_for_iter(CFG, it, 40) for it in range(40)]
    assert betas[:8] == [8.0] * 8
    assert betas[8:16] == [16.0] * 8
    assert betas[-8:] == [128.0] * 8
    assert betas[-1] == float(CFG.beta_schedule[-1])


def test_beta_schedule_survives_the_json_roundtrip():
    """asdict() -> JSON turns the tuple into a list; a resumed config must
    still produce the same schedule (and the same types)."""
    import json
    from dataclasses import asdict

    revived = pvgc.PVGCConfig(**json.loads(json.dumps(asdict(CFG))))
    assert isinstance(revived.beta_schedule, list)      # the trap
    assert [optimize.beta_for_iter(revived, it, 40) for it in range(40)] == \
           [optimize.beta_for_iter(CFG, it, 40) for it in range(40)]
    revived.beta_schedule = tuple(revived.beta_schedule)
    assert revived.beta_schedule == CFG.beta_schedule


def test_beta_schedule_handles_short_and_long_runs():
    assert optimize.beta_for_iter(CFG, 0, 1) == 8.0
    assert optimize.beta_for_iter(CFG, 0, 0) == 8.0        # no ZeroDivision
    # never index past the last stage, whatever the iteration count
    assert optimize.beta_for_iter(CFG, 999, 3) == 128.0


# --------------------------------------------------------------------------
# checkpoint / resume
# --------------------------------------------------------------------------


def test_checkpoint_roundtrip_restores_the_optimizer_exactly(tmp_path):
    optax = pytest.importorskip("optax")
    from jax.flatten_util import ravel_pytree

    p = jax.numpy.asarray(np.linspace(0, 1, 500).reshape(500, 1, 1),
                          dtype=jax.numpy.float32)
    opt = optax.adam(0.05)
    state = opt.init(p)
    # take one real step so the Adam moments are not all zeros
    g = jax.numpy.asarray(np.random.default_rng(0).standard_normal(p.shape),
                          dtype=jax.numpy.float32)
    updates, state = opt.update(g, state)
    p = jax.numpy.clip(optax.apply_updates(p, updates), 0.0, 1.0)

    written = optimize.save_state(
        tmp_path, optimize.OptState(p=p, opt_state=state, iteration=7,
                                    beta=32.0))
    assert os.path.basename(written) == optimize.STATE_FILE
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp.npz")]

    _, unravel = ravel_pytree(opt.init(p))
    back = optimize.load_state(tmp_path, unravel=unravel)
    assert back.iteration == 7 and back.beta == 32.0
    assert np.array_equal(np.asarray(back.p), np.asarray(p))
    for a, b in zip(jax.tree.leaves(back.opt_state), jax.tree.leaves(state)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_checkpoint_write_is_atomic(tmp_path):
    """A failed write must leave the previous good checkpoint in place."""
    optax = pytest.importorskip("optax")

    p = jax.numpy.zeros((4, 1, 1))
    opt = optax.adam(0.05)
    good = optimize.OptState(p=p, opt_state=opt.init(p), iteration=1, beta=8.0)
    optimize.save_state(tmp_path, good)

    broken = optimize.OptState(p=p, opt_state=object(),   # unflattenable
                               iteration=2, beta=16.0)
    with pytest.raises(Exception):
        optimize.save_state(tmp_path, broken)
    assert optimize.load_state(tmp_path).iteration == 1


def test_run_loop_resumes_with_continuous_iteration_numbers(tmp_path):
    """The whole resume contract, exercised on a trivial quadratic FOM: the
    history is contiguous, the iteration counter does not restart, and the
    latent vector picks up where it stopped."""
    pytest.importorskip("optax")
    import jax.numpy as jnp

    target = jnp.asarray(np.linspace(0, 1, 12).reshape(12, 1, 1),
                         dtype=jnp.float32)

    def vg_fn(p, beta):
        return jax.value_and_grad(lambda q: ((q - target) ** 2).sum())(p)

    p0 = np.full((12, 1, 1), 0.5)
    first = optimize.run_loop(vg_fn, p0, CFG, n_iters=2, lr=0.1,
                              run_dir=str(tmp_path))
    assert first.iteration == 1 and len(first.history) == 2

    second = optimize.run_loop(vg_fn, p0, CFG, n_iters=3, lr=0.1,
                               run_dir=str(tmp_path), resume=True)
    assert second.iteration == 2 and len(second.history) == 1

    rows = np.genfromtxt(os.path.join(tmp_path, optimize.HISTORY_FILE),
                         delimiter=",", names=True)
    assert [int(v) for v in rows["iter"]] == [0, 1, 2]
    # the resumed step continued from the checkpointed latent, it did not
    # restart from p0 (the loss must keep improving; loss = -CE column)
    assert rows["CE"][2] > rows["CE"][1] > rows["CE"][0]


def test_run_loop_stops_on_a_converged_final_stage(tmp_path):
    """A FOM that stops improving inside the last beta stage ends the run
    early instead of burning the remaining GPU-hours."""
    pytest.importorskip("optax")
    import jax.numpy as jnp

    def vg_fn(p, beta):
        return jnp.asarray(-1.0), jnp.zeros_like(p)   # flat: zero improvement

    state = optimize.run_loop(vg_fn, np.zeros((4, 1, 1)), CFG, n_iters=40,
                              lr=0.1, run_dir=str(tmp_path),
                              stop_patience=2)
    assert state.stop_reason == "converged"
    assert state.iteration < 39
    assert optimize.beta_for_iter(CFG, state.iteration, 40) == 128.0
