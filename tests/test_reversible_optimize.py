"""Pure-math tests for the reversible-recording path (no GPU, no simulation).

Pins the pieces that would corrupt silently if left unpinned:
  * expected_latent_steps / k_nyquist against known-good anchors
    (T = 4196 at 0.10 um: K_Nyq = 11, K=24 -> 176 latent steps) and their
    grid-halved twins (T = 8392: K_Nyq = 22, K=11 -> 764),
  * make_reversible_gc really builds method="reversible" with the
    LinearReconstructEveryK + DtypeConversion stack (fp16) or without the
    conversion (fp32),
  * reversible_witness: a fabricated correct recording_state PASSES
    byte-exactly, and each single corruption (missing recorder = the
    checkpointed negative control, wrong dtype, wrong latent axis) FAILS
    with the failure named,
  * scripts/21 resample_binary_1d: hand-computed area weights on the
    non-integer 50 -> 20 px/um ratio, the tie -> solid rule, identity at
    equal grids, and the binary/length guards.
"""

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script(fname, modname):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", fname)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


opt = _load_script("15_grating_coupler_optimize.py", "_grating_coupler_optimize_script_rev")
ext = _load_script("21_extrude_1d_design.py", "_extrude_1d_script")


# --------------------------------------------------------------------------
# latent steps + Nyquist against known-good anchors
# --------------------------------------------------------------------------


def test_expected_latent_steps_anchor():
    # T=4196, K=24: range(0, 4196, 24) has 175 entries, last is 4176 != 4195
    # -> 176. A K=24/float16 store holds exactly 176 latent steps.
    assert opt.expected_latent_steps(4196, 24) == 176
    assert opt.expected_latent_steps(4196, 1) == 4196
    assert opt.expected_latent_steps(4196, 11) == 383
    # the reversible path's operating point
    assert opt.expected_latent_steps(8392, 11) == 764


def test_k_nyquist_doubles_when_dt_halves():
    cfg = SimpleNamespace(sim_time_s=0.8e-12, lam_c=1.31)
    # dt = 0.8e-12/4196 -> period/dt = 22.9 -> 11
    assert opt.k_nyquist(cfg, 4196) == 11
    # spacing halved -> T doubles -> K_Nyq = 22 (the finer grid)
    assert opt.k_nyquist(cfg, 8392) == 22


def test_make_reversible_gc_stack():
    fdtdx = pytest.importorskip("fdtdx")
    gc16 = opt.make_reversible_gc(11, "float16")
    assert gc16.method == "reversible"
    mods = gc16.recorder.modules
    assert len(mods) == 2
    assert isinstance(mods[0], fdtdx.LinearReconstructEveryK)
    assert mods[0].k == 11
    assert isinstance(mods[1], fdtdx.DtypeConversion)
    gc32 = opt.make_reversible_gc(11, "float32")
    assert len(gc32.recorder.modules) == 1


# --------------------------------------------------------------------------
# reversible_witness: fabricated positive + three corruptions
# --------------------------------------------------------------------------

# tiny fake scene: spacing 1.0 um, cell_x=4, cell_z=3, dpml=0.5, wg_width=2
# -> nx=4, nz=3, ny=round(2+3+1)=6; T=10, K=3 -> latent=(0,3,6,9)=4 steps
_CFG = SimpleNamespace(cell_x=4.0, cell_z=3.0, spacing_um=1.0, dpml=0.5)
_T, _K, _WG = 10, 3, 2.0
_NX, _NY, _NZ, _L = 4, 6, 3, 4


def _fake_arrays(dtype="float16", latent=_L):
    data = {}
    for axis, (a, b) in (("x", (_NY, _NZ)), ("y", (_NX, _NZ)),
                         ("z", (_NX, _NY))):
        for side in ("minus", "plus"):
            for field in ("E", "H"):
                data[f"pml_{axis}_{side}_{field}"] = np.zeros(
                    (latent, 3, a, b), dtype=dtype)
    return SimpleNamespace(recording_state=SimpleNamespace(data=data))


def test_witness_passes_byte_exact():
    ok, details = opt.reversible_witness(_fake_arrays(), _CFG, _T, _K,
                                         "float16", _WG)
    assert ok, details["failures"]
    # 2 fields x 3 comps x 2 faces x (6*3+4*3+4*6) x 2 B x 4 latent steps
    assert details["store_bytes_pred"] == 12 * 54 * 2 * 4
    assert details["store_bytes_actual"] == details["store_bytes_pred"]
    assert details["latent_steps_expected"] == _L


def test_witness_negative_control_checkpointed():
    # a checkpointed build has recording_state None — the negative control:
    # the witness MUST fail on it
    arrays = SimpleNamespace(recording_state=None)
    ok, details = opt.reversible_witness(arrays, _CFG, _T, _K,
                                         "float16", _WG)
    assert not ok
    assert "recording_state is None" in details["failures"][0]


def test_witness_rejects_wrong_dtype():
    ok, details = opt.reversible_witness(_fake_arrays(dtype="float32"),
                                         _CFG, _T, _K, "float16", _WG)
    assert not ok
    assert any("dtype" in f for f in details["failures"])


def test_witness_rejects_wrong_latent_axis():
    ok, details = opt.reversible_witness(_fake_arrays(latent=_L + 1),
                                         _CFG, _T, _K, "float16", _WG)
    assert not ok
    assert any("latent axis" in f for f in details["failures"])


# --------------------------------------------------------------------------
# scripts/21 resampling: hand-computed weights, tie rule, guards
# --------------------------------------------------------------------------


def test_resample_50_to_20_hand_computed():
    # a non-integer ratio: 50 -> 20 px/um, 5 src px per 2 dst px, weights 20/20/10
    # and 10/20/20 in units of 1/1000 um (hand-computed)
    rho = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    out, mean = ext.resample_binary_1d(rho, 50, 20)
    assert mean == pytest.approx([0.6, 0.6])
    assert out.tolist() == [1.0, 1.0]
    rho = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    out, mean = ext.resample_binary_1d(rho, 50, 20)
    assert mean == pytest.approx([0.6, 0.2])
    assert out.tolist() == [1.0, 0.0]


def test_resample_tie_goes_solid():
    # 4 -> 2 px/um: one dst px averages two src px with equal weight;
    # [1, 0] -> mean exactly 0.5 -> the tie rule says solid
    out, mean = ext.resample_binary_1d(np.array([1.0, 0.0]), 4, 2)
    assert mean == pytest.approx([0.5])
    assert out.tolist() == [1.0]


def test_resample_identity_at_equal_grids():
    rho = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    out, mean = ext.resample_binary_1d(rho, 10, 10)
    assert out.tolist() == rho.tolist()
    assert ext.roundtrip_changed(rho, out, 10, 10) == 0


def test_resample_guards():
    with pytest.raises(ValueError, match="binary"):
        ext.resample_binary_1d(np.array([0.5, 1.0]), 4, 2)
    with pytest.raises(ValueError, match="not.*representable|representable"):
        ext.resample_binary_1d(np.array([1.0, 0.0, 1.0]), 50, 20)
    with pytest.raises(ValueError, match="1D"):
        ext.resample_binary_1d(np.ones((2, 2)), 4, 2)
