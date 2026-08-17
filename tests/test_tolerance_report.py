"""End-to-end tests for scripts/16_tolerance_report.py (M1+ tolerance
report): per-voxel sensitivity map + three-corner robust-design evaluation.

CPU, a tiny quasi-2D pvgc scene -- the same coarse-grid/short-sim-time
recipe tests/test_datasets.py::_tiny_cfg uses, plus a t_si override to
0.20 um: the differentiable Device path this script's sensitivity map needs
(pvgc.assert_design_grid_snaps) requires t_si to be an integer multiple of
spacing_um, which test_datasets.py's recipe never needed because it only
ever exercises the forward-only characterize() path, never
make_ce_value_and_grad.

The run dir is constructed directly (config.json + opt_state.npz), not via
scripts/15_pvgc_optimize.py: that checkpoint pair is the entire contract
scripts/16 reads, and building it by hand keeps this test to one
optimization-free simulation budget.
"""

import csv
import importlib.util
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytest.importorskip("jax")
pytest.importorskip("fdtdx")
optax = pytest.importorskip("optax")

from invdx import optimize, runio
from invdx.problems.pvgc import PVGCConfig, n_design_voxels


def _tiny_cfg(**overrides):
    """Same tiny quasi-2D pvgc scene as test_datasets.py::_tiny_cfg, plus
    t_si=0.20 (see module docstring): 0.220 (the PVGCConfig default) is not
    a multiple of spacing_um=0.05 (0.220/0.05 = 4.4), which trips
    assert_design_grid_snaps the moment make_ce_value_and_grad builds the
    differentiable Device -- a path the original recipe never exercised."""
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


def _load_tolerance_script():
    """scripts/16_tolerance_report.py by path (leading digit blocks a normal
    `import`), same trick tests/test_pvgc_optimize.py uses for script 15."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", "16_tolerance_report.py")
    spec = importlib.util.spec_from_file_location(
        "_tolerance_report_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_run_dir(tmp_path, cfg):
    run_dir = str(tmp_path / "pvgc-opt-tiny")
    os.makedirs(run_dir, exist_ok=True)
    runio.save_json(os.path.join(run_dir, "config.json"), asdict(cfg))

    import jax.numpy as jnp

    n = n_design_voxels(cfg)
    rng = np.random.default_rng(0)
    p0 = jnp.asarray(rng.uniform(0.2, 0.8, size=(n, 1, 1)), dtype=jnp.float32)
    opt_state = optax.adam(0.05).init(p0)
    optimize.save_state(run_dir, optimize.OptState(
        p=p0, opt_state=opt_state, iteration=0, beta=8.0))
    return run_dir


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tolerance-report")
    return _make_run_dir(tmp_path, _tiny_cfg())


def _run_main(argv_tail):
    """--checkpoints 4 (script default is 20): fdtdx's checkpointed backward
    graph is dramatically slower to XLA-compile on CPU as checkpoint count
    grows, even for this tiny scene (~430 timesteps) -- measured >3 min wall
    clock at the production default 20 vs a few seconds at 4. Checkpoint
    count only trades compile/memory cost for a re-run of forward physics
    already covered elsewhere (test_pvgc_optimize.py); it does not change
    what this test is pinning (CSV/PNG shape and content), so it is safe to
    shrink here to stay inside the CI time budget."""
    mod = _load_tolerance_script()
    argv = sys.argv
    sys.argv = ["16_tolerance_report.py"] + list(argv_tail) + \
        (["--checkpoints", "4"] if "--checkpoints" not in argv_tail else [])
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    return rc


# --------------------------------------------------------------------------
# pure-math: checkpoint read-back
# --------------------------------------------------------------------------


def test_load_run_reads_back_the_checkpoint(run_dir):
    mod = _load_tolerance_script()
    cfg = _tiny_cfg()
    cfg2, p_flat, beta, iteration = mod.load_run(run_dir)
    assert cfg2.design_grid_per_um == cfg.design_grid_per_um
    assert cfg2.t_si == pytest.approx(cfg.t_si)
    assert p_flat.shape == (n_design_voxels(cfg),)
    assert np.all(p_flat >= 0.0) and np.all(p_flat <= 1.0)
    assert beta == 8.0
    assert iteration == 0


def test_load_run_fails_loudly_without_a_checkpoint(tmp_path):
    mod = _load_tolerance_script()
    run_dir = str(tmp_path / "no-checkpoint")
    os.makedirs(run_dir)
    runio.save_json(os.path.join(run_dir, "config.json"), asdict(_tiny_cfg()))
    with pytest.raises(SystemExit, match="opt_state"):
        mod.load_run(run_dir)


# --------------------------------------------------------------------------
# end-to-end: real (tiny) fdtdx simulations, CPU
# --------------------------------------------------------------------------


def test_tolerance_report_end_to_end_default(run_dir):
    """No --lams: sensitivity.csv/.png + corners.csv with bw/ridge blank."""
    assert _run_main([run_dir]) == 0

    n = n_design_voxels(_tiny_cfg())
    out_dir = os.path.join(run_dir, "tolerance")
    sens_csv = os.path.join(out_dir, "sensitivity.csv")
    sens_png = os.path.join(out_dir, "sensitivity.png")
    corners_csv = os.path.join(out_dir, "corners.csv")
    assert os.path.exists(sens_csv)
    assert os.path.exists(sens_png) and os.path.getsize(sens_png) > 0
    assert os.path.exists(corners_csv)

    with open(sens_csv) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == \
           ["voxel_idx", "x_um", "rho", "dCE_drho", "abs_grad_norm"]
    assert len(rows) == n                              # sensitivity length == design voxels
    assert [r["voxel_idx"] for r in rows] == [str(i) for i in range(n)]
    x_um = [float(r["x_um"]) for r in rows]
    assert x_um == sorted(x_um)                         # monotonic voxel positions
    for r in rows:
        rho = float(r["rho"])
        assert -1e-6 <= rho <= 1 + 1e-6
        float(r["dCE_drho"])
        norm = float(r["abs_grad_norm"])
        assert -1e-9 <= norm <= 1 + 1e-9
    assert max(float(r["abs_grad_norm"]) for r in rows) == pytest.approx(1.0)

    with open(corners_csv) as f:
        crows = list(csv.DictReader(f))
    assert list(crows[0].keys()) == \
           ["corner", "CE_dB", "bw_3db_nm", "ridge_lam_um"]   # tolerance.md, fixed
    assert [r["corner"] for r in crows] == ["eroded", "nominal", "dilated"]
    for r in crows:
        float(r["CE_dB"])
        assert r["bw_3db_nm"] == "" and r["ridge_lam_um"] == ""  # single-lambda: blank


def test_corner_table_with_lams_fills_bandwidth_and_ridge(run_dir):
    """Exercises corner_table()'s --lams path directly (not through main()):
    it does not depend on sensitivity_map() at all, and going through the
    full CLI a second time would re-pay the ~90s checkpointed-gradient
    compile that test_tolerance_report_end_to_end_default already paid for
    the exact same (cfg, p, beta) -- pure waste on a shared CPU box where
    that compile is the dominant, load-sensitive cost (measured 5-90s
    depending on contention)."""
    mod = _load_tolerance_script()
    cfg, p_flat, beta, _ = mod.load_run(run_dir)
    p_in_c, azimuth_sign, _ = mod.pvgc.calibrated_beam(cfg)
    lams_spec = list(np.linspace(1.28, 1.34, 3))

    rows = mod.corner_table(cfg, p_flat, beta, p_in_c, azimuth_sign,
                            lams_spec=lams_spec)
    assert [r["corner"] for r in rows] == ["eroded", "nominal", "dilated"]
    for r in rows:
        float(r["CE_dB"])
        assert 1.28 <= r["ridge_lam_um"] <= 1.34
        assert r["bw_3db_nm"] >= 0.0

    corners_csv = os.path.join(run_dir, "tolerance", "corners_lams.csv")
    mod.write_corners_csv(corners_csv, rows)
    with open(corners_csv) as f:
        crows = list(csv.DictReader(f))
    assert list(crows[0].keys()) == \
           ["corner", "CE_dB", "bw_3db_nm", "ridge_lam_um"]
    for r in crows:
        float(r["CE_dB"]); float(r["bw_3db_nm"]); float(r["ridge_lam_um"])


def test_project_corner_orders_solid_fraction_with_eta(run_dir):
    """Pure-math sanity check on the projection direction: a lower
    threshold (eta_d) must never produce LESS solid than a higher one
    (eta_e), for the same latent and beta -- the definition of erosion vs
    dilation. Uses the checkpoint's own latent/beta, no simulation."""
    mod = _load_tolerance_script()
    cfg, p_flat, beta, _ = mod.load_run(run_dir)
    rho_e = mod.project_corner(cfg, p_flat, cfg.eta_e, beta)
    rho_i = mod.project_corner(cfg, p_flat, cfg.eta_i, beta)
    rho_d = mod.project_corner(cfg, p_flat, cfg.eta_d, beta)
    assert rho_e.sum() <= rho_i.sum() <= rho_d.sum()


def test_yield_90_counts_the_nominal_corner_as_a_pass():
    mod = _load_tolerance_script()
    rows = [{"corner": "eroded", "CE_dB": -30.0},
           {"corner": "nominal", "CE_dB": -20.0},
           {"corner": "dilated", "CE_dB": -20.1}]
    frac, passes, n = mod.yield_90(rows)
    assert n == 3 and passes == 2                       # nominal + dilated pass, eroded fails
    assert frac == pytest.approx(2 / 3)
