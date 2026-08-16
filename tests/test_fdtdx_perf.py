"""Equivalence gate for the vendored fast forward loop (engines/fdtdx_perf).

run_fdtd_fast must be BITWISE-identical to fdtdx.run_fdtd on the supported
scene subset: max abs diff of final E/H/psi fields exactly 0.0 and every
PhasorDetector phasor exactly equal. The component-tuple layout and the
hoisted PML coefficients are value-preserving transformations (verified in
scripts/13_curl_microbench.py); any nonzero diff here means an op-order
regression in fdtdx_perf.py — fix the op order, never loosen these asserts.

Runs on CPU (JAX_PLATFORMS=cpu, float32 as in production) so it works on any
machine.

XLA_FLAGS note: the CPU backend's LLVM codegen contracts mul+add pairs into
FMAs inside fused kernels, and WHICH pairs get contracted depends on fusion
clustering — so two graphs that are per-op identical (vanilla's stacked
state vs the fast path's component tuples) produce ~1e-8-level diffs at the
default optimization level, and even vanilla-compiled differs from
vanilla-eager. --xla_backend_optimization_level=0 disables that codegen
contraction (both engines run under the same flag), leaving exactly the
per-op IEEE semantics that the asserts below check at 0.0. The GPU backend
does not have this issue: scripts/13_curl_microbench.py measured bitwise
identity at default flags on Turing. Both env vars must be set before the
first JAX computation of the process to take effect.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=0"
).strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

from invdx.engines.fdtdx_perf import run_fdtd_fast
from invdx.problems.pvgc import PVGCConfig, build_scene, uniform_grating_teeth


def _tiny_cfg(**overrides) -> PVGCConfig:
    """Small quasi-2D pvgc scene: coarse grid, short run, keeps CI fast."""
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
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _place(cfg, teeth, seed=0):
    sim_config, objs, cons = build_scene(cfg, teeth=teeth, with_chip=True)
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, k2)
    return arrays, objects, sim_config, key


def _assert_bitwise_equal(cfg, teeth):
    arrays, objects, sim_config, key = _place(cfg, teeth)

    _, ref = fdtdx.run_fdtd(
        arrays=arrays, objects=objects, config=sim_config, key=key,
        show_progress=False)
    step_fast, fast = run_fdtd_fast(
        arrays=arrays, objects=objects, config=sim_config, key=key)

    assert int(step_fast) == sim_config.time_steps_total

    for name in ("E", "H", "psi_E", "psi_H"):
        r = getattr(ref.fields, name)
        f = getattr(fast.fields, name)
        assert f.shape == r.shape and f.dtype == r.dtype
        max_diff = float(jnp.max(jnp.abs(f - r)))
        # bitwise gate: 1e-7-level diffs mean an op-order change, not noise
        assert max_diff == 0.0, f"{name}: max abs diff {max_diff} != 0.0"

    assert set(fast.detector_states) == set(ref.detector_states)
    for det_name, ref_state in ref.detector_states.items():
        for k, ref_arr in ref_state.items():
            fast_arr = fast.detector_states[det_name][k]
            assert np.array_equal(np.asarray(fast_arr), np.asarray(ref_arr)), (
                f"detector {det_name}[{k}] not exactly equal")
        # sanity: the run actually excited the detectors (guards against a
        # trivially-passing all-zeros comparison)
    total = sum(
        float(np.abs(np.asarray(v)).sum())
        for s in ref.detector_states.values() for v in s.values())
    assert total > 0.0, "vanilla run produced all-zero detector states"


def test_bitwise_equivalence_pvgc_quasi2d():
    """Vertical beam (azimuth 0), PML x/z + periodic y, grating teeth."""
    cfg = _tiny_cfg()
    teeth = uniform_grating_teeth(cfg, period=0.6, duty=0.5, n_periods=4)
    _assert_bitwise_equal(cfg, teeth)


def test_bitwise_equivalence_tilted_source():
    """Tilted beam (theta=8deg -> azimuth=-8) over periodic-y boundaries."""
    cfg = _tiny_cfg(theta_deg=8.0)
    teeth = uniform_grating_teeth(cfg, period=0.55, duty=0.45, n_periods=4)
    _assert_bitwise_equal(cfg, teeth)


def test_rejects_gradient_config():
    """Forward-only guard must trip loudly, not silently mis-simulate."""
    cfg = _tiny_cfg()
    arrays, objects, sim_config, key = _place(cfg, teeth=None)
    sim_config = sim_config.aset(
        "gradient_config", fdtdx.GradientConfig(method="checkpointed", num_checkpoints=2))
    with pytest.raises(NotImplementedError, match="FORWARD-ONLY"):
        run_fdtd_fast(arrays=arrays, objects=objects, config=sim_config, key=key)
