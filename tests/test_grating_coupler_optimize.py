"""Pure-math tests for the inverse-design driver (no simulation, no GPU).

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
import sys

import numpy as np
import pytest

from invdx import optimize
from invdx.problems import grating_coupler

jax = pytest.importorskip("jax")

CFG = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
CFG.design_grid_per_um = 50


# --------------------------------------------------------------------------
# design vector <-> teeth
# --------------------------------------------------------------------------


def test_rasterize_profile_roundtrip_is_exact_on_grid_aligned_teeth():
    """profile_teeth . rasterize_teeth == identity for grid-aligned teeth."""
    rho = np.zeros(grating_coupler.n_design_voxels(CFG))
    rho[10:35] = 1.0        # 0.5 um tooth
    rho[100:113] = 1.0      # 0.26 um tooth
    rho[499] = 1.0          # tooth touching the window edge
    teeth = grating_coupler.profile_teeth(CFG, rho)
    assert np.array_equal(grating_coupler.rasterize_teeth(CFG, teeth), rho)
    assert grating_coupler.profile_teeth(CFG, grating_coupler.rasterize_teeth(CFG, teeth)) == teeth


def test_rasterize_places_teeth_at_the_right_pixels():
    """A tooth is silicon exactly where the pixel CENTRE is inside it."""
    rho = grating_coupler.rasterize_teeth(CFG, [(-5.0, 0.5)])
    assert rho[:25].all() and not rho[25:].any()
    # an off-grid tooth snaps to whole pixels, it never smears into grey
    rho2 = grating_coupler.rasterize_teeth(CFG, [(-5.0 + 0.011, 0.5)])
    assert set(np.unique(rho2)) <= {0.0, 1.0}
    assert rho2.sum() == 25
    assert rho2[0] == 0.0 and rho2[1] == 1.0 and rho2[25] == 1.0


def test_rasterized_grating_keeps_every_tooth_the_same_width():
    """The width must not jitter: a 0.2875 um tooth on 0.02 um pixels rounds
    to 14 pixels for EVERY tooth. Letting it alternate 14/15 (which a
    pixel-centre rule does) costs 13 dB — the rasterization is part of the
    physics, not a formatting detail."""
    teeth = grating_coupler.uniform_grating_teeth(CFG, period=0.575, duty=0.5)
    rho = grating_coupler.rasterize_teeth(CFG, teeth)
    rendered = grating_coupler.profile_teeth(CFG, rho)
    assert len(rendered) == len(teeth) == 17
    assert {round(w * CFG.design_grid_per_um) for _, w in rendered} == {14}
    assert rho.sum() == 17 * 14


def test_n_design_voxels_matches_script_07_expectation():
    """Script 07 rebuilds GratingCouplerConfig() with its default L_design, so the
    design vector length is fixed by design_grid_per_um alone."""
    assert grating_coupler.n_design_voxels(CFG) == int(CFG.L_design * 50) == 500


# --------------------------------------------------------------------------
# grid-snapping guard
# --------------------------------------------------------------------------


def test_design_grid_guard_rejects_the_default_spacing():
    """0.0125 um snaps t_si (0.220) to 0.225 — the Device would silently be a
    different device from the one the measurement chain describes."""
    bad = grating_coupler.GratingCouplerConfig()          # spacing_um = 0.0125
    with pytest.raises(ValueError, match="t_si"):
        grating_coupler.assert_design_grid_snaps(bad)


def test_design_grid_guard_accepts_the_clean_grids():
    for spacing, grid in ((0.020, 50), (0.020, 25), (0.010, 100), (0.010, 50)):
        cfg = grating_coupler.GratingCouplerConfig(spacing_um=spacing)
        cfg.design_grid_per_um = grid
        grating_coupler.assert_design_grid_snaps(cfg)


def test_design_grid_guard_rejects_design_grid_finer_than_the_mesh():
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    cfg.design_grid_per_um = 100          # 10 nm pixels on a 20 nm mesh
    with pytest.raises(ValueError):
        grating_coupler.assert_design_grid_snaps(cfg)


@pytest.mark.parametrize("bad", [-50, 0, 50.5, 50.0000001])
def test_design_grid_guard_rejects_invalid_design_grid_per_um(bad):
    """design_grid_per_um is annotated `int` but that is not enforced by the
    dataclass itself — a negative value in particular used to pass the
    divisibility check below by accident (it only compares magnitudes)."""
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    cfg.design_grid_per_um = bad
    with pytest.raises(ValueError, match="design_grid_per_um"):
        grating_coupler.assert_design_grid_snaps(cfg)


# --------------------------------------------------------------------------
# dtype: cfg.dtype is otherwise a dead parameter on this path
# --------------------------------------------------------------------------


def test_build_scene_rejects_non_float32_dtype():
    """build_scene/build_scene_3d hardcode float32; a cfg.dtype override must
    fail loudly instead of being silently ignored (only
    engines.fdtdx_engine.make_sim_config actually reads cfg.dtype)."""
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020, dtype="float64")
    with pytest.raises(NotImplementedError, match="dtype"):
        grating_coupler.build_scene(cfg)


def test_build_scene_accepts_the_default_float32_dtype():
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    sim_config, object_list, constraints = grating_coupler.build_scene(cfg)
    assert object_list and constraints


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
    Em, Hm_back, Pm, _ = grating_coupler.te0_target_on_monitor(CFG, E.size)
    ref = grating_coupler.overlap_power_directional(E, H, Em, Hm_back, CFG.spacing_um)
    got = float(grating_coupler.overlap_power_directional_jnp(
        jax.numpy.asarray(E), jax.numpy.asarray(H), Em, Hm_back,
        CFG.spacing_um, Pm))
    assert got == pytest.approx(ref, rel=2e-5)


def test_overlap_jnp_self_normalizes_and_rejects_the_wrong_direction():
    """The mode measured against itself carries exactly the mode power; the
    forward mode measured against the backward target carries ~none."""
    Em, Hm_back, Pm, _ = grating_coupler.te0_target_on_monitor(CFG, 125)
    E = jax.numpy.asarray(Em.astype(np.complex64))
    H = jax.numpy.asarray(Hm_back.astype(np.complex64))
    P = float(grating_coupler.overlap_power_directional_jnp(E, H, Em, Hm_back,
                                                 CFG.spacing_um, Pm))
    assert P == pytest.approx(Pm, rel=1e-5)
    P_wrong = float(grating_coupler.overlap_power_directional_jnp(
        E, -H, Em, Hm_back, CFG.spacing_um, Pm))
    assert P_wrong < 1e-10 * Pm


def test_te0_target_pm_matches_the_numpy_chain():
    """Pm precomputed here must equal the one the numpy overlap derives."""
    Em, Hm_back, Pm, neff = grating_coupler.te0_target_on_monitor(CFG, 125)
    assert Pm == pytest.approx(
        abs(0.5 * np.real(np.sum(Em * np.conj(Hm_back))) * CFG.spacing_um))
    assert CFG.n_sio2 < neff < CFG.n_si


def test_te0_target_tracks_wavelength_dispersion():
    _, _, _, neff_c = grating_coupler.te0_target_on_monitor(CFG, 125)
    _, _, _, neff_hi = grating_coupler.te0_target_on_monitor(CFG, 125, lam_um=1.36)
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

    revived = grating_coupler.GratingCouplerConfig(**json.loads(json.dumps(asdict(CFG))))
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


def test_run_loop_resume_keeps_checkpointed_beta_not_recomputed(tmp_path):
    """`--resume --iters=<iters_done>` (0 extra iterations, the finalize-only
    shape) must not silently reinterpret the beta schedule under a smaller
    denominator. Regression for the 2026-08-17 bug: runs/coupler-opt-154 and
    -156 both really stopped at beta=64 (iteration 25 of a 40-iteration
    schedule); recomputing via beta_for_iter(cfg, 25, 26) — the old
    behaviour — silently jumps to beta=128, the schedule's LAST stage, which
    would bake a sharper-than-actual TanhProjection into the saved design."""
    optax = pytest.importorskip("optax")

    p = jax.numpy.zeros((4, 1, 1))
    opt = optax.adam(0.05)
    optimize.save_state(tmp_path, optimize.OptState(
        p=p, opt_state=opt.init(p), iteration=25, beta=64.0))

    # confirm the scenario actually reproduces the bug's arithmetic before
    # trusting the assertion below to mean anything
    assert optimize.beta_for_iter(CFG, 25, 26) == 128.0
    assert optimize.beta_for_iter(CFG, 25, 40) == 64.0

    def vg_fn(p, beta):
        return jax.numpy.asarray(0.0), jax.numpy.zeros_like(p)

    state = optimize.run_loop(vg_fn, p, CFG, n_iters=26, lr=0.1,
                              run_dir=str(tmp_path), resume=True)
    assert state.iteration == 25          # 0 extra iterations ran
    assert state.beta == 64.0             # the checkpointed value, unchanged


# --------------------------------------------------------------------------
# Richardson extrapolation (scripts/15's gradcheck FD)
# --------------------------------------------------------------------------


def _load_grating_coupler_optimize_script():
    """scripts/15_grating_coupler_optimize.py by path — its filename can't be a normal
    `import` target (leading digit), and importing it costs nothing extra: it
    only pulls in invdx submodules already exercised elsewhere in this file,
    no GPU or fdtdx call happens until gradcheck() actually runs."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", "15_grating_coupler_optimize.py")
    spec = importlib.util.spec_from_file_location("_grating_coupler_optimize_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gradcheck_richardson_beats_single_h_on_a_cubic():
    """A central difference on f(x) = x**3 has an EXACT, closed-form
    truncation term: FD(h) = f'(x) + h**2 (no higher-order remainder, because
    f is a cubic — see the derivation this pins). At x=0.1, h=GRADCHECK_H the
    term (h**2 = 0.0025) is ~8% of f'(x) = 0.03, enough on its own to fail
    GRADCHECK_TOL — the same shape as the 2026-08-17 production gradcheck
    failure. Richardson extrapolation cancels that h**2 term algebraically,
    so FD_R must recover the analytic gradient (handed back as `vg_fn`'s
    exact `grad`, no autodiff involved) to float32 precision, while the raw
    single-h estimate and fd_consistency both still see the truncation."""
    import jax.numpy as jnp

    mod = _load_grating_coupler_optimize_script()

    def vg_fn(p, beta):
        return jnp.sum(p ** 3), 3 * p ** 2       # exact value & gradient

    def value_fn(p, beta):
        return jnp.sum(p ** 3)

    p0 = np.full((6, 1, 1), 0.1)
    res = mod.gradcheck(vg_fn, value_fn, p0, beta=1.0, seed=0)

    assert res["checks"], "gradcheck sampled no voxels"
    for c in res["checks"]:
        raw_rel = abs(c["fd_h"] - c["adjoint"]) / abs(c["adjoint"])
        assert raw_rel > mod.GRADCHECK_TOL, (
            "test setup should reproduce a single-h FD failing the gate")
        assert c["fd_consistency"] > mod.GRADCHECK_TOL, (
            "fd_consistency should flag the same unconverged single-h FD")
        assert c["rel_err"] < 1e-3, (
            "Richardson-extrapolated FD must match the analytic gradient")
    assert res["worst_rel_err"] < mod.GRADCHECK_TOL


def test_gradcheck_richardson_agrees_with_single_h_when_fd_is_exact():
    """A central difference on f(x) = x**2 has NO truncation term at all (the
    odd-order remainder in the Taylor expansion vanishes identically), so
    FD(h) == FD(h/2) == f'(x) up to float32 rounding alone for ANY h.
    Richardson must not fabricate a disagreement where none exists:
    fd_consistency and rel_err both stay at the rounding floor."""
    import jax.numpy as jnp

    mod = _load_grating_coupler_optimize_script()

    def vg_fn(p, beta):
        return jnp.sum(p ** 2), 2 * p            # exact value & gradient

    def value_fn(p, beta):
        return jnp.sum(p ** 2)

    p0 = np.full((6, 1, 1), 0.5)
    res = mod.gradcheck(vg_fn, value_fn, p0, beta=1.0, seed=0)

    assert res["checks"], "gradcheck sampled no voxels"
    for c in res["checks"]:
        assert c["rel_err"] < 1e-4
        assert c["fd_consistency"] < 1e-4


# --------------------------------------------------------------------------
# --finalize-only (scripts/15's checkpoint-recovery path)
# --------------------------------------------------------------------------


def test_finalize_only_writes_designs_from_checkpoint_beta_no_extra_history(
        tmp_path, monkeypatch):
    """`--finalize-only` must: (1) write the three finalization files using
    the CHECKPOINTED beta (not a recomputed one — that's the same bug as
    above, this time exercised through the CLI path), and (2) run no
    optimizer iterations, so history.csv keeps exactly its pre-existing rows.

    grating_coupler.calibrated_beam / make_ce_value_and_grad / rho_from_params are
    stubbed: they front a real fdtdx simulation (GPU-gated everywhere else in
    this repo, see src/invdx/gates/g2_gradcheck.py's REQUIRES = ("gpu",)),
    which this CPU/no-GPU test suite never invokes. Stubbing them keeps the
    test exercising scripts/15's control flow — not fdtdx — and lets
    rho_from_params report which beta it was actually called with."""
    import json
    from dataclasses import asdict

    optax = pytest.importorskip("optax")
    from invdx import runio

    mod = _load_grating_coupler_optimize_script()

    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    cfg.design_grid_per_um = 50
    n_vox = grating_coupler.n_design_voxels(cfg)
    run_dir = str(tmp_path)
    runio.save_json(os.path.join(run_dir, "config.json"), asdict(cfg))

    csv_path = os.path.join(run_dir, mod.optimize.HISTORY_FILE)
    for it, beta, ce in ((0, 8.0, 0.01), (1, 8.0, 0.02)):
        runio.append_csv(csv_path, mod.optimize.HISTORY_HEADER,
                         [it, beta, ce, -20.0, 0.0, 0.1, float("nan"), ce])
    rows_before = np.genfromtxt(csv_path, delimiter=",", names=True)

    p = jax.numpy.zeros((n_vox, 1, 1))
    opt = optax.adam(0.05)
    mod.optimize.save_state(run_dir, mod.optimize.OptState(
        p=p, opt_state=opt.init(p), iteration=1, beta=64.0))

    seen_betas = []

    def fake_rho_from_params(device, params, beta):
        seen_betas.append(float(beta))
        return np.full(n_vox, 1.0 if beta == 64.0 else 0.0)

    def fake_make_ce_value_and_grad(cfg, p_in, num_checkpoints=20, lams=None,
                                    **kw):
        dev = type("Dev", (), {"name": "dev"})()
        params0 = {"dev": np.zeros((n_vox, 1, 1))}   # main() prints its shape
        return None, None, None, params0, dev, None

    monkeypatch.setattr(mod.grating_coupler, "calibrated_beam",
                        lambda cfg, seed=0: (1.0, 1.0, 0.0))
    monkeypatch.setattr(mod.grating_coupler, "make_ce_value_and_grad",
                        fake_make_ce_value_and_grad)
    monkeypatch.setattr(mod.grating_coupler, "rho_from_params", fake_rho_from_params)
    monkeypatch.setattr(
        sys, "argv",
        ["scripts/15_grating_coupler_optimize.py", "--resume", run_dir,
         "--finalize-only", "--no-final-check"])

    assert mod.main() == 0

    assert seen_betas == [64.0]           # not beta_for_iter(cfg, 1, 2)=16.0

    cont = np.load(os.path.join(run_dir, "design_rho_cont.npy"))
    binr = np.load(os.path.join(run_dir, "design_rho.npy"))
    assert (cont == 1.0).all() and (binr == 1.0).all()
    assert os.path.exists(os.path.join(run_dir, "results.json"))
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    assert res["iters_done"] == 2

    rows_after = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert len(rows_after) == len(rows_before) == 2


def test_a_degrading_fom_is_not_reported_as_converged(tmp_path):
    """A run whose FOM is getting worse has not converged.

    The stall counter tests |relative change| < tol: a plateau is a SMALL
    change, in either direction. A one-sided test also counted a FOM that was
    falling, so a diverging run stopped and labelled itself "converged" -- a
    false claim about the optimizer's state, and one real runs can trigger
    (a late-stage step has been seen to drop the FOM by 9 dB before Adam
    recovered).
    """
    pytest.importorskip("optax")
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    cfg.design_grid_per_um = 50
    cfg.beta_schedule = (8.0,)          # one stage: beta_final from step 0

    # run_loop takes loss = -FOM, so a RISING loss is a degrading FOM.
    rising_loss = iter([-32.0, -16.0, -8.0, -4.0, -2.0, -1.0])

    def vg_fn(p, beta):
        return jax.numpy.asarray(next(rising_loss)), jax.numpy.zeros_like(p)

    p0 = jax.numpy.full((4, 1, 1), 0.5, dtype=jax.numpy.float32)
    state = optimize.run_loop(vg_fn, p0, cfg, n_iters=6, lr=0.05,
                              run_dir=str(tmp_path),
                              stop_rel_tol=0.005, stop_patience=2)

    assert state.stop_reason != "converged", (
        "a monotonically worsening FOM was reported as "
        f"{state.stop_reason!r}")
    assert state.iteration == 5, "the run should have used its full budget"
