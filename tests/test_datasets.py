"""End-to-end test of the batch forward-dataset mechanism (invdx.datasets).

CPU, a tiny quasi-2D pvgc scene (the same coarse-grid/short-sim-time recipe
tests/test_fdtdx_perf.py uses to keep the forward loop itself fast), and a
handful of designs. This pins the MECHANISM's promises — shard/manifest
writing, sha256, resume-by-skip, the optional spectrum column — not pvgc
physics (that is characterize()'s own test coverage) and not any sampling
distribution (sample_designs' generic kinds are a working default, not a
claim about a good training set).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=0"
).strip()

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("fdtdx")

from invdx import datasets
from invdx.problems.pvgc import PVGCConfig, n_design_voxels


def _tiny_cfg(**overrides):
    """Same tiny quasi-2D pvgc scene as test_fdtdx_perf._tiny_cfg (coarse
    grid, short run — keeps CI fast), plus a small design grid."""
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
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------
# pure-math: sampling determinism, no simulation
# --------------------------------------------------------------------------


def test_sample_designs_is_deterministic_in_kind_seed_n_cfg():
    """Resume-by-skip relies on this: re-sampling with identical arguments
    must reproduce the identical design sequence."""
    cfg = _tiny_cfg()
    a = list(datasets.sample_designs("uniform-grating", 5, 0, cfg))
    b = list(datasets.sample_designs("uniform-grating", 5, 0, cfg))
    assert len(a) == 5
    for ra, rb in zip(a, b):
        assert np.array_equal(ra, rb)
    # a different seed must (almost certainly) sample a different design
    c = list(datasets.sample_designs("uniform-grating", 5, 1, cfg))
    assert not all(np.array_equal(ra, rc) for ra, rc in zip(a, c))


def test_sample_designs_random_rho_shape_and_range():
    cfg = _tiny_cfg()
    rhos = list(datasets.sample_designs("random-rho", 3, 0, cfg))
    assert len(rhos) == 3
    n = n_design_voxels(cfg)
    for rho in rhos:
        assert rho.shape == (n,)
        assert np.all(rho >= 0.0) and np.all(rho <= 1.0)


def test_sample_designs_rejects_unknown_kind():
    cfg = _tiny_cfg()
    with pytest.raises(ValueError, match="unknown sample kind"):
        list(datasets.sample_designs("not-a-kind", 1, 0, cfg))


# --------------------------------------------------------------------------
# shard helpers, no simulation
# --------------------------------------------------------------------------


def test_shard_write_is_atomic_and_hash_matches(tmp_path):
    path = os.path.join(tmp_path, "shard_00000.npz")
    datasets._write_shard(path, np.ones((2, 4)), np.array([0.1, 0.2]))
    assert os.path.exists(path)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp.npz")]
    assert datasets._sha256_file(path) == datasets._sha256_file(path)
    assert datasets._shard_is_valid(path, 2)
    assert not datasets._shard_is_valid(path, 3)
    assert not datasets._shard_is_valid(os.path.join(tmp_path, "missing.npz"), 2)


# --------------------------------------------------------------------------
# end-to-end: forward-only batch, real (tiny) fdtdx simulations
# --------------------------------------------------------------------------


def test_generate_dataset_uniform_grating_shards_manifest_and_resume(tmp_path):
    cfg = _tiny_cfg()
    run_dir = str(tmp_path / "ds-grating")
    os.makedirs(run_dir)

    calls = []
    manifest = datasets.generate_dataset(
        cfg, kind="uniform-grating", n=3, run_dir=run_dir, shard_size=2,
        on_shard=calls.append)

    # ---- manifest shape ----
    assert manifest["schema_version"] == datasets.SCHEMA_VERSION
    assert manifest["complete"] is True
    assert manifest["n_samples_written"] == 3
    s = manifest["sampling"]
    assert (s["kind"], s["seed"], s["n"], s["shard_size"]) == \
           ("uniform-grating", cfg.seed, 3, 2)
    assert set(s["kind_params"]) == {"period_range_um", "duty_range"}
    assert manifest["invdx_git_hash"] is None or len(manifest["invdx_git_hash"]) == 40
    assert manifest["n_design_voxels"] == n_design_voxels(cfg)
    assert set(manifest["units"]) >= {"rho", "label_ce", "spectrum"}
    assert manifest["axis_order"]["rho"] == ["sample", "design_voxel"]
    assert len(manifest["shards"]) == 2                    # sizes 2, 1
    assert [e["n_samples"] for e in manifest["shards"]] == [2, 1]
    assert [c["skipped_existing"] for c in calls] == [False, False]

    manifest_path = os.path.join(run_dir, datasets.MANIFEST_FILE)
    assert os.path.exists(manifest_path)
    assert os.path.exists(os.path.join(run_dir, datasets.README_FILE))

    n = n_design_voxels(cfg)
    for entry in manifest["shards"]:
        path = os.path.join(run_dir, entry["file"])
        assert os.path.exists(path)
        assert datasets._sha256_file(path) == entry["sha256"]
        with np.load(path) as z:
            assert z["rho"].shape == (entry["n_samples"], n)
            assert z["label_ce"].shape == (entry["n_samples"],)
            assert set(np.unique(z["rho"])) <= {0.0, 1.0}   # binary grating
            assert np.all(z["label_ce"] >= 0.0)              # CE is a power ratio
            assert "spectrum" not in z                       # --lams not used

    # ---- resume: identical args against the same dir must skip every shard,
    # not touch the files on disk, and reproduce the identical manifest ----
    mtimes_before = {e["file"]: os.path.getmtime(os.path.join(run_dir, e["file"]))
                     for e in manifest["shards"]}
    calls2 = []
    manifest2 = datasets.generate_dataset(
        cfg, kind="uniform-grating", n=3, run_dir=run_dir, shard_size=2,
        on_shard=calls2.append)

    assert [c["skipped_existing"] for c in calls2] == [True, True]
    # identical content (file/n_samples/sha256) — only skipped_existing differs
    for e1, e2 in zip(manifest["shards"], manifest2["shards"]):
        assert e2["skipped_existing"] is True
        for k in ("file", "n_samples", "sha256", "index_start"):
            assert e1[k] == e2[k]
    for e in manifest2["shards"]:
        mtime_after = os.path.getmtime(os.path.join(run_dir, e["file"]))
        assert mtime_after == mtimes_before[e["file"]], (
            "a resumed run re-wrote a shard that should have been skipped")


def test_generate_dataset_random_rho_with_optional_spectrum(tmp_path):
    cfg = _tiny_cfg()
    run_dir = str(tmp_path / "ds-random")
    os.makedirs(run_dir)
    lams = [cfg.lam_c, cfg.lam_c + 0.02]

    manifest = datasets.generate_dataset(
        cfg, kind="random-rho", n=1, run_dir=run_dir, shard_size=1,
        lams_um=lams)

    assert manifest["lams_um"] == lams
    assert manifest["sampling"]["kind"] == "random-rho"
    entry = manifest["shards"][0]
    with np.load(os.path.join(run_dir, entry["file"])) as z:
        assert "spectrum" in z
        assert z["spectrum"].shape == (1, len(lams))
        assert z["rho"].shape == (1, n_design_voxels(cfg))
