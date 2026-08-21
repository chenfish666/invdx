"""The w_s11 reflection-penalized FOM.

Contracts pinned here:
  * w_s11 is a float 0.0 default (cli._cast_like casts --set overrides to the
    default's type; an int default would make --set w_s11=0.3 raise),
  * w_s11 = 0.0 keeps the legacy fiber-excited scalar-loss path (bit-identity
    vs the pre-change code was a one-off pre/post check, not a permanent
    test; the SHAPE of the return value is the permanent tell),
  * w_s11 > 0 runs ONE wg-side excitation and returns
    ((loss, {"ce", "s11"}), grad) with -loss == ce - w_s11*s11 exactly,
  * the wg-path gradient back-propagates through BOTH monitors (the
    fiber_mon adjoint is the open risk — gated by a finite-difference check
    on TWO directions plus a Richardson-extrapolated FD-vs-adjoint gate; see
    test_wg_gradient_matches_finite_difference's docstring for why this is
    Richardson and not a raw h-refinement no-growth check),
  * run_loop records the TRUE CE, the s11_dB and the penalized fom as
    separate history.csv columns, and a resumed pre-s11 run keeps its own
    6-column layout,
  * the traced aux {"ce", "s11"} agrees with the INDEPENDENT numpy
    measurement chain (pvgc.wg_side_characterize) on the same grid-aligned
    binary grating — the external ground truth the t3 identity cannot give.

CPU, the same tiny quasi-2D scene as tests/test_tolerance_report.py
(t_si=0.20 is mandatory: the differentiable Device path requires t_si to be
an integer multiple of spacing_um). num_checkpoints=4 everywhere — the
checkpointed backward graph at the production 20 costs >3 min of XLA compile
on CPU for zero extra coverage.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
pytest.importorskip("fdtdx")
optax = pytest.importorskip("optax")

import jax.numpy as jnp

from invdx import optimize, runio
from invdx.cli import apply_overrides
from invdx.problems import pvgc
from invdx.problems.pvgc import PVGCConfig, n_design_voxels
from invdx.richardson_fd import richardson_fd_check

W_S11 = 0.7
BETA = 8.0


def _tiny_cfg(**overrides):
    """Same tiny quasi-2D pvgc scene as test_tolerance_report.py::_tiny_cfg
    (see that module's docstring for why t_si=0.20)."""
    cfg = PVGCConfig(
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
        t_si=0.20,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def p0():
    n = n_design_voxels(_tiny_cfg())
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(0.2, 0.8, size=(n, 1, 1)),
                       dtype=jnp.float32)


@pytest.fixture(scope="module")
def legacy(p0):
    """w_s11=0 build evaluated once: (out, grad) at (p0, BETA)."""
    vg_fn, _, _, _, _, value_fn = pvgc.make_ce_value_and_grad(
        _tiny_cfg(), 1.0, num_checkpoints=4)
    out, grad = vg_fn(p0, jnp.asarray(BETA, dtype=jnp.float32))
    return SimpleNamespace(out=out, grad=grad, value_fn=value_fn)


@pytest.fixture(scope="module")
def wg(p0):
    """w_s11>0 build evaluated once: ((loss, aux), grad) at (p0, BETA)."""
    vg_fn, _, _, _, _, value_fn = pvgc.make_ce_value_and_grad(
        _tiny_cfg(w_s11=W_S11), 1.0, num_checkpoints=4)
    out, grad = vg_fn(p0, jnp.asarray(BETA, dtype=jnp.float32))
    return SimpleNamespace(out=out, grad=grad, value_fn=value_fn)


# --------------------------------------------------------------------------
# t1 — config field and CLI override (no simulation)
# --------------------------------------------------------------------------


def test_w_s11_defaults_to_float_zero_and_casts_from_cli():
    assert type(PVGCConfig().w_s11) is float
    assert PVGCConfig().w_s11 == 0.0
    cfg = apply_overrides(PVGCConfig(),
                          SimpleNamespace(set=["w_s11=0.3"], resolution=None))
    assert type(cfg.w_s11) is float and cfg.w_s11 == 0.3


# --------------------------------------------------------------------------
# t2 — w_s11=0 keeps the legacy scalar-loss path
# --------------------------------------------------------------------------


def test_legacy_path_returns_scalar_loss_with_finite_grad(legacy):
    """DEGRADED check: the wg branch returns a (loss, aux) tuple, so a scalar
    proves the legacy branch was taken. True bit-identity against the
    pre-change code was verified once pre/post change (loss repr + grad
    sha256 identical), it cannot be re-checked permanently from here."""
    assert not isinstance(legacy.out, tuple)
    assert np.isfinite(float(legacy.out))
    g = np.asarray(legacy.grad)
    assert np.isfinite(g).all() and np.linalg.norm(g) > 0


# --------------------------------------------------------------------------
# t3 — wg path: exact intra-run identity and ranges
# --------------------------------------------------------------------------


def test_wg_path_fom_identity_and_ranges(wg):
    assert isinstance(wg.out, tuple)
    loss, aux = wg.out
    ce, r11 = float(aux["ce"]), float(aux["s11"])
    # loss = -(ce - w*r11) by construction — exact up to float re-association;
    # the abs term keeps this robust when ce and w*r11 nearly cancel (float32)
    assert -float(loss) == pytest.approx(
        ce - W_S11 * r11, rel=1e-5, abs=1e-7 * max(ce, W_S11 * r11))
    # the 0.05 ps tiny scene is transient-dominated; r11 stays a power ratio
    assert 0.0 < r11 <= 1.0
    assert 0.0 < ce < 1.0
    print(f"[t3] wg-reciprocal CE {10 * np.log10(ce):.2f} dB, "
          f"R11 {10 * np.log10(r11):.2f} dB")


def test_wg_ce_vs_legacy_ce_logged_not_asserted(wg, legacy):
    """The wg CE term relies on reciprocity (measured mismatch 0.059 dB on
    the production scene); on this transient tiny scene the delta is
    uncontrolled, so it is logged for the record, never asserted."""
    ce_wg = float(wg.out[1]["ce"])
    ce_legacy = -float(legacy.out)
    print(f"[t3-log] CE wg-reciprocal {10 * np.log10(abs(ce_wg) + 1e-30):.2f} "
          f"dB vs fiber {10 * np.log10(abs(ce_legacy) + 1e-30):.2f} dB")


# --------------------------------------------------------------------------
# t4 — the fiber_mon-adjoint gate: FD directional check on the wg path
# --------------------------------------------------------------------------


def test_wg_gradient_is_finite_nonzero_and_differs_from_legacy(wg, legacy):
    g_wg = np.asarray(wg.grad)
    assert np.isfinite(g_wg).all() and np.linalg.norm(g_wg) > 0
    assert not np.allclose(g_wg, np.asarray(legacy.grad))


def test_wg_gradient_matches_finite_difference(wg, p0):
    """If this fails, the fiber_mon phasor adjoint is NOT propagated by the
    checkpointed backward pass (the flagged UNKNOWN) — surface it, do not
    tune the tolerance.

    Review round-01 strengthening: TWO independent random directions (one
    direction can hide a wrong gradient component by chance).

    2026-08-21 gate history — READ BEFORE "fixing" a future failure here:
    the original round-01 gate additionally required an h-refinement
    no-growth check (halving h must never GROW the FD-vs-adjoint error,
    since central-difference truncation is O(h^2)). That gate FAILED on an
    RTX 6000 Ada GPU (err(h) 2.343e-07 -> err(h/2) 4.439e-07, error grew) while
    PASSING on a Quadro RTX 6000 GPU and on CPU, same commit, same config.
    Cross-architecture evidence ruled out a gradient bug: history.csv's
    grad_norm was byte-for-byte identical across the two GPUs (0.24972 /
    0.197567, matching through the next optimizer iteration too) — the
    adjoint is not hardware-dependent. What IS hardware-dependent is the FD
    probe: it re-evaluates the loss twice per h, and Ada's different
    reduction order / FMA scheduling perturbs those evaluations at the
    ~1e-7 level — exactly the scale of both err(h) and err(h/2) above. That
    check was comparing two numbers sitting on the same float32 noise floor,
    not a real truncation trend.
    The fix is NOT a looser tolerance (this docstring used to say so in so
    many words: do not tune it) — it is a better ESTIMATOR. Richardson
    extrapolation combines FD(h) and FD(h/2) as
    FD_R = (4*FD(h/2) - FD(h)) / 3, algebraically cancelling the leading
    O(h^2) truncation term instead of trying to observe its sign from two
    noisy raw quotients. FD_R vs the adjoint is what is gated below;
    err(h)/err(h/2)/their ratio are still computed and printed but are
    INFORMATIONAL ONLY — real signal on a Turing GPU, but not
    architecture-stable enough to gate on.
    6 value_fn evaluations = 6 tiny sims (~20 s CPU)."""
    beta_j = jnp.asarray(BETA, dtype=jnp.float32)
    g = np.asarray(wg.grad)

    def central_fd(v, h):
        vj = jnp.asarray(v, dtype=jnp.float32)
        f_plus = float(wg.value_fn(p0 + h * vj, beta_j))
        f_minus = float(wg.value_fn(p0 - h * vj, beta_j))
        return (f_plus - f_minus) / (2 * h), max(abs(f_plus), abs(f_minus))

    h = 0.02
    per_seed = {}
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(g.shape)
        v /= np.linalg.norm(v)
        gdotv = float(np.sum(g * v))
        fd, _ = central_fd(v, h)
        assert fd == pytest.approx(gdotv, rel=0.25), \
            f"seed {seed}: directional FD {fd:.6e} vs adjoint {gdotv:.6e}"
        per_seed[seed] = (v, gdotv, fd)

    v1, gdotv1, _ = per_seed[1]
    v1j = jnp.asarray(v1, dtype=jnp.float32)

    def evaluate(sign, hh):
        return float(wg.value_fn(p0 + sign * hh * v1j, beta_j))

    rc = richardson_fd_check(evaluate, h, gdotv1)

    # informational only, NOT gated (see docstring: this signal is real on
    # Turing but sits under Ada's own FD-probe rounding noise)
    err_h = abs(rc["fd_h"] - gdotv1)
    err_half = abs(rc["fd_h2"] - gdotv1)
    print(f"[t4][informational, not gated] err(h={h}) {err_h:.3e} -> "
          f"err(h/2) {err_half:.3e} (ratio {err_half / (err_h + 1e-30):.2f}x)")

    print(f"[t4] Richardson FD_R {rc['fd']:.6e} vs adjoint {gdotv1:.6e}  "
          f"rel_err {rc['rel_err']:.2e}  fd_consistency {rc['fd_consistency']:.2e}")
    # XCHECK_RTOL (defined below, t6) is this file's own established
    # convention for "how tight can a cross-check be before conceptual bugs
    # get lost in rounding noise" — reused here rather than inventing a new
    # number. Measured on this scene: rel_err ~5e-6 (seed 1), ~2e-4 (seed 2),
    # 4-200x inside this margin.
    assert rc["rel_err"] < XCHECK_RTOL, \
        (f"Richardson-extrapolated FD {rc['fd']:.6e} vs adjoint {gdotv1:.6e}: "
         f"rel_err {rc['rel_err']:.2e} >= {XCHECK_RTOL:.0e}")


def test_richardson_gate_negative_control_rejects_perturbed_adjoint(wg, p0):
    """Negative control for the Richardson gate above — PLUS a companion
    near-threshold positive control, because the two only mean something
    together.

    A gate that never fires is worse than no gate: it looks like coverage
    while providing none (this project hit exactly this "zero-sensitivity
    check" shape more than once during this investigation). But a negative
    control is only valid if its perturbation is actually LARGER than the
    gate's tolerance — corrupting the adjoint by less than XCHECK_RTOL and
    finding the gate still passes would prove nothing about sensitivity,
    it would just be correct behaviour. So both perturbation sizes below are
    DERIVED from the real XCHECK_RTOL the gate above uses (read, not
    guessed):
      * NEG_FACTOR corrupts the adjoint by 10x XCHECK_RTOL — far enough
        past the tolerance that "still green" can only mean the gate is
        blind, not that the corruption was too small to matter.
      * POS_FACTOR corrupts it by 0.3x XCHECK_RTOL — safely inside the
        tolerance, so this direction proves the gate does NOT cry wolf on a
        deviation smaller than its own stated threshold.
    Both share the cached FD(h)/FD(h/2) evaluations from the true adjoint
    (only 4 sims total for this whole test, same direction as t4's seed 1).
    """
    beta_j = jnp.asarray(BETA, dtype=jnp.float32)
    g = np.asarray(wg.grad)
    h = 0.02
    NEG_FACTOR = 10 * XCHECK_RTOL     # >> tolerance -> must FAIL the gate
    POS_FACTOR = 0.3 * XCHECK_RTOL    # << tolerance -> must still PASS

    rng = np.random.default_rng(1)          # same direction as t4's seed 1
    v = rng.standard_normal(g.shape)
    v /= np.linalg.norm(v)
    gdotv = float(np.sum(g * v))
    vj = jnp.asarray(v, dtype=jnp.float32)

    cache = {}

    def evaluate(sign, hh):
        key = (sign, hh)
        if key not in cache:
            cache[key] = float(wg.value_fn(p0 + sign * hh * vj, beta_j))
        return cache[key]

    rc_true = richardson_fd_check(evaluate, h, gdotv)
    print(f"[negctl] true adjoint:            rel_err {rc_true['rel_err']:.2e}  "
          f"gate={'PASS' if rc_true['rel_err'] < XCHECK_RTOL else 'FAIL'}")
    assert rc_true["rel_err"] < XCHECK_RTOL, (
        "sanity precondition failed: the unperturbed gate must pass before "
        "either control below means anything")

    corrupted_bad = gdotv * (1.0 + NEG_FACTOR)
    rc_bad = richardson_fd_check(evaluate, h, corrupted_bad)   # cache hits
    print(f"[negctl] adjoint*(1+{NEG_FACTOR:.1e}) [{NEG_FACTOR / XCHECK_RTOL:.0f}x "
          f"tol]: rel_err {rc_bad['rel_err']:.2e}  "
          f"gate={'PASS' if rc_bad['rel_err'] < XCHECK_RTOL else 'FAIL'}")
    assert rc_bad["rel_err"] >= XCHECK_RTOL, (
        f"NEGATIVE CONTROL FAILED TO FAIL: an adjoint corrupted {NEG_FACTOR / XCHECK_RTOL:.0f}x "
        f"past tolerance still passes the Richardson gate (rel_err "
        f"{rc_bad['rel_err']:.2e} < {XCHECK_RTOL:.0e}) -- the gate has no "
        f"sensitivity, exactly the failure mode this control exists to catch")

    corrupted_ok = gdotv * (1.0 + POS_FACTOR)
    rc_ok = richardson_fd_check(evaluate, h, corrupted_ok)     # cache hits
    print(f"[negctl] adjoint*(1+{POS_FACTOR:.1e}) [{POS_FACTOR / XCHECK_RTOL:.1f}x "
          f"tol]: rel_err {rc_ok['rel_err']:.2e}  "
          f"gate={'PASS' if rc_ok['rel_err'] < XCHECK_RTOL else 'FAIL'}")
    assert rc_ok["rel_err"] < XCHECK_RTOL, (
        f"COMPANION POSITIVE CONTROL FAILED: a deviation smaller than "
        f"XCHECK_RTOL ({POS_FACTOR / XCHECK_RTOL:.1f}x tol) still trips the "
        f"gate (rel_err {rc_ok['rel_err']:.2e} >= {XCHECK_RTOL:.0e}) -- the "
        f"gate is crying wolf on a legitimately small deviation")


# --------------------------------------------------------------------------
# t5 — run_loop history contract (no simulation)
# --------------------------------------------------------------------------


def test_run_loop_tuple_vg_fn_records_true_ce_s11_fom(tmp_path):
    def vg_fn(p, beta):
        return (jnp.asarray(-0.5), {"ce": jnp.asarray(0.6),
                                    "s11": jnp.asarray(0.1)}), \
            jnp.zeros_like(p)

    optimize.run_loop(vg_fn, np.zeros((4, 1, 1)), _tiny_cfg(),
                      n_iters=2, lr=0.1, run_dir=str(tmp_path))
    csv_path = os.path.join(tmp_path, optimize.HISTORY_FILE)
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    assert header == optimize.HISTORY_HEADER
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert float(np.atleast_1d(rows["CE"])[-1]) == pytest.approx(0.6)
    assert float(np.atleast_1d(rows["fom"])[-1]) == pytest.approx(0.5)
    assert float(np.atleast_1d(rows["s11_dB"])[-1]) == pytest.approx(-10.0,
                                                                     abs=1e-3)


def test_run_loop_scalar_vg_fn_records_nan_s11_and_fom_equals_ce(tmp_path):
    def vg_fn(p, beta):
        return jnp.asarray(-0.25), jnp.zeros_like(p)

    optimize.run_loop(vg_fn, np.zeros((4, 1, 1)), _tiny_cfg(),
                      n_iters=2, lr=0.1, run_dir=str(tmp_path))
    rows = np.genfromtxt(os.path.join(tmp_path, optimize.HISTORY_FILE),
                         delimiter=",", names=True)
    assert float(np.atleast_1d(rows["CE"])[-1]) == pytest.approx(0.25)
    assert float(np.atleast_1d(rows["fom"])[-1]) == pytest.approx(0.25)
    assert np.isnan(float(np.atleast_1d(rows["s11_dB"])[-1]))


def test_run_loop_resume_keeps_old_6_column_layout(tmp_path):
    """A pre-s11 run resumed under the new code must keep appending rows the
    old header describes — 6 fields, not 8."""
    old_header = ["iter", "beta", "CE", "CE_dB", "grad_norm", "wall_s"]
    csv_path = os.path.join(tmp_path, optimize.HISTORY_FILE)
    runio.append_csv(csv_path, old_header, [0, 8.0, 0.01, -20.0, 0.0, 0.1])

    p = jnp.zeros((4, 1, 1))
    optimize.save_state(str(tmp_path), optimize.OptState(
        p=p, opt_state=optax.adam(0.1).init(p), iteration=0, beta=8.0))

    def vg_fn(p, beta):
        return jnp.asarray(-0.25), jnp.zeros_like(p)

    optimize.run_loop(vg_fn, p, _tiny_cfg(), n_iters=3, lr=0.1,
                      run_dir=str(tmp_path), resume=True)
    with open(csv_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[0].split(",") == old_header
    assert all(len(ln.split(",")) == 6 for ln in lines[1:])
    assert len(lines) == 1 + 3            # 1 pre-existing + 2 appended rows


# --------------------------------------------------------------------------
# t6 — independent numpy ground truth for the traced S11 and CE (review
#      round-01 GAP 1: the t3 identity is true by construction; only a
#      chain that never touches the traced code can certify what R11 IS)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xcheck():
    """One grid-aligned binary grating measured through BOTH wg-excited
    chains: pvgc.wg_side_characterize (its own run, its own numpy overlap
    code) vs the traced aux of make_ce_value_and_grad.

    Apples-to-apples construction — rho (0.6 um period, 50% duty, 60/120
    pixels Si) is the source of truth:
      * traced chain: rho as the design Device with with_transforms=False
        (design_device's documented acceptance-check path — the latent p IS
        the physical density, so a 0/1 vector places exact air/Si voxels;
        beta is then inert),
      * numpy chain: profile_teeth(rho) as static UniformMaterialObject
        teeth (teeth are DERIVED from rho so both describe the same pixels;
        rasterize_teeth . profile_teeth == identity is asserted below and
        pinned in tests/test_pvgc_optimize.py). The design pixel (1/20 um)
        equals spacing_um exactly, and Device and teeth anchor with the
        same (X0 + x)*UM margins, so both render to the same cells.
    Every other scene ingredient — source, PML, monitors, layer stack, sim
    length — comes from the SAME build_scene(excitation="wg",
    with_chip=True) call in both chains, and INVDX_FAST=0 pins the numpy
    run to the vanilla fdtdx.run_fdtd the traced loss calls.
    """
    cfg = _tiny_cfg(w_s11=W_S11)
    rho = pvgc.rasterize_teeth(
        cfg, pvgc.uniform_grating_teeth(cfg, period=0.6, duty=0.5))
    teeth = pvgc.profile_teeth(cfg, rho)
    assert np.array_equal(pvgc.rasterize_teeth(cfg, teeth), rho)

    prev = os.environ.get("INVDX_FAST")
    os.environ["INVDX_FAST"] = "0"
    try:
        num = pvgc.wg_side_characterize(cfg, teeth)
    finally:
        if prev is None:
            os.environ.pop("INVDX_FAST", None)
        else:
            os.environ["INVDX_FAST"] = prev

    vg_fn, _, _, _, _, _ = pvgc.make_ce_value_and_grad(
        cfg, 1.0, num_checkpoints=4, with_transforms=False)
    p_bin = jnp.asarray(rho.reshape(-1, 1, 1), dtype=jnp.float32)
    (_, aux), _ = vg_fn(p_bin, jnp.asarray(BETA, dtype=jnp.float32))
    return SimpleNamespace(num=num, aux=aux)


# Tolerance for both cross-checks. After the fixture's apples-to-apples
# construction the ONLY differences left are rounding: two separately
# compiled float32 runs (XLA fusion/FMA clustering perturbs fields at the
# ~1e-7 level — tests/test_fdtdx_perf.py module docstring) and float64
# numpy vs float32 jnp overlap postprocessing; both quantities are ratios,
# so common scale factors cancel. Measured agreement: s11 1.3e-7 rel,
# ce 1.3e-6 rel. rel=1e-3 sits ~3 decades above that rounding floor and
# many decades below every conceptual failure mode (a direction-sign error
# suppresses the overlap by ~6 decades; a missing P_in normalization
# shifts it by ~4). Do NOT widen on failure — a disagreement means the
# traced quantity measures something DIFFERENT from the validated numpy
# chain; investigate instead.
XCHECK_RTOL = 1e-3


def test_traced_s11_matches_independent_numpy_chain(xcheck):
    s11_tr, s11_np = float(xcheck.aux["s11"]), xcheck.num["S11"]
    print(f"[t6] S11 traced {s11_tr:.6e} vs numpy {s11_np:.6e} "
          f"(rel {abs(s11_tr - s11_np) / s11_np:.1e})")
    assert s11_tr == pytest.approx(s11_np, rel=XCHECK_RTOL)


def test_traced_ce_matches_independent_numpy_chain(xcheck):
    ce_tr, ce_np = float(xcheck.aux["ce"]), xcheck.num["CE_fwd"]
    print(f"[t6] CE traced {ce_tr:.6e} vs numpy {ce_np:.6e} "
          f"(rel {abs(ce_tr - ce_np) / ce_np:.1e})")
    assert ce_tr == pytest.approx(ce_np, rel=XCHECK_RTOL)
