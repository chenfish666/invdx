"""The w_s11 reflection-penalized FOM (campaign/s11-fom).

Contracts pinned here:
  * w_s11 is a float 0.0 default (cli._cast_like casts --set overrides to the
    default's type; an int default would make --set w_s11=0.3 raise),
  * w_s11 = 0.0 keeps the legacy fiber-excited scalar-loss path (bit-identity
    vs the pre-change code was a one-off pre/post check, not a permanent
    test; the SHAPE of the return value is the permanent tell),
  * w_s11 > 0 runs ONE wg-side excitation and returns
    ((loss, {"ce", "s11"}), grad) with -loss == ce - w_s11*s11 exactly,
  * the wg-path gradient back-propagates through BOTH monitors (the
    fiber_mon adjoint is the open risk — gated by a finite-difference check),
  * run_loop records the TRUE CE, the s11_dB and the penalized fom as
    separate history.csv columns, and a resumed pre-s11 run keeps its own
    6-column layout.

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
    tune the tolerance."""
    h = 0.02
    rng = np.random.default_rng(1)
    v = rng.standard_normal(np.asarray(p0).shape)
    v /= np.linalg.norm(v)
    beta_j = jnp.asarray(BETA, dtype=jnp.float32)
    vj = jnp.asarray(v, dtype=jnp.float32)
    f_plus = float(wg.value_fn(p0 + h * vj, beta_j))
    f_minus = float(wg.value_fn(p0 - h * vj, beta_j))
    fd = (f_plus - f_minus) / (2 * h)
    gdotv = float(np.sum(np.asarray(wg.grad) * v))
    assert fd == pytest.approx(gdotv, rel=0.25), \
        f"directional FD {fd:.6e} vs adjoint {gdotv:.6e}"


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
