"""Pure-math tests for scripts/18_checkpoint_sweep.py (no GPU, no simulation).

Pins three things that are silent-corruption risks if left unpinned:
  * the linear fit (peak_bytes = intercept + slope*C) recovers known
    coefficients and reports residuals, and honestly returns None instead
    of guessing when fewer than two memory points are available,
  * a failed row (OOM or otherwise) keeps its C/n_cells/steps but leaves
    the failed stage's timing fields None, and sweep.csv renders that as a
    blank field, not "None" or 0,
  * anchor_comparable() agrees with the module's own ANCHOR_CONDITION.
"""

import csv
import importlib.util
import io
import os

import numpy as np
import pytest

from invdx.problems.pvgc import PVGCConfig


def _load_sweep_script():
    """scripts/18_checkpoint_sweep.py by path (leading digit blocks a normal
    `import`), same trick tests/test_pvgc_optimize.py uses for script 15."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", "18_checkpoint_sweep.py")
    spec = importlib.util.spec_from_file_location("_checkpoint_sweep_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep_script()


# --------------------------------------------------------------------------
# fit_bytes_per_cell
# --------------------------------------------------------------------------


def test_fit_recovers_known_linear_coefficients():
    cells = 1_000_000
    base_bytes_per_cell = 200.0
    per_ckpt_bytes_per_cell = 15.0
    checkpoints = [1, 2, 5, 10, 20]
    peak = [cells * (base_bytes_per_cell + per_ckpt_bytes_per_cell * c)
            for c in checkpoints]

    fit = sweep.fit_bytes_per_cell(checkpoints, peak, cells)

    assert fit is not None
    assert fit["base_bytes_per_cell"] == pytest.approx(base_bytes_per_cell, rel=1e-9)
    assert fit["bytes_per_cell_per_checkpoint"] == pytest.approx(per_ckpt_bytes_per_cell, rel=1e-9)
    assert fit["n_points"] == len(checkpoints)
    # exact synthetic line -> residuals must vanish
    assert all(abs(r) < 1e-6 for r in fit["residuals_bytes"])


def test_fit_reports_nonzero_residuals_on_noisy_data():
    cells = 1_000_000
    checkpoints = [1, 2, 5, 10, 20]
    # not a perfect line: the fit must still return coefficients, but with
    # visibly nonzero residuals rather than silently claiming a perfect fit
    peak = [1.0e8, 2.3e8, 5.0e8, 9.5e8, 2.05e9]

    fit = sweep.fit_bytes_per_cell(checkpoints, peak, cells)

    assert fit is not None
    assert len(fit["residuals_bytes"]) == len(checkpoints)
    assert max(abs(r) for r in fit["residuals_bytes"]) > 0


def test_fit_ignores_none_entries_and_needs_at_least_two_points():
    checkpoints = [1, 2, 5]
    # only one real memory reading (the others OOM'd / unavailable)
    peak = [None, 4.0e8, None]
    assert sweep.fit_bytes_per_cell(checkpoints, peak, 1_000_000) is None

    # zero readings
    assert sweep.fit_bytes_per_cell(checkpoints, [None, None, None], 1_000_000) is None

    # exactly two -> a fit is returned (not silently refused)
    fit = sweep.fit_bytes_per_cell([1, 2], [2.0e8, 3.0e8], 1_000_000)
    assert fit is not None
    assert fit["n_points"] == 2


# --------------------------------------------------------------------------
# anchor_comparable / ANCHOR_CONDITION
# --------------------------------------------------------------------------


def test_default_recipe_is_anchor_comparable():
    cfg = PVGCConfig(**sweep.ANCHOR_CONDITION)
    assert sweep.anchor_comparable(cfg)


def test_tiny_smoke_cfg_is_not_anchor_comparable():
    cfg = PVGCConfig(spacing_um=0.05, design_grid_per_um=20,
                     sim_time_s=0.05e-12, theta_deg=0.0)
    assert not sweep.anchor_comparable(cfg)


def test_anchor_comparable_is_exact_not_approximate_for_design_grid():
    cfg = PVGCConfig(**sweep.ANCHOR_CONDITION)
    cfg.design_grid_per_um = sweep.ANCHOR_CONDITION["design_grid_per_um"] + 1
    assert not sweep.anchor_comparable(cfg)


# --------------------------------------------------------------------------
# n_cells
# --------------------------------------------------------------------------


def test_n_cells_matches_hand_computed_grid():
    cfg = PVGCConfig(spacing_um=0.05, design_grid_per_um=20,
                     sim_time_s=0.05e-12, L_design=6.0, pad_x=2.0, dpml=0.6,
                     t_box=1.5, t_sub=0.8, air_above=2.0, t_si=0.20)
    nx = round(cfg.cell_x / cfg.spacing_um)
    nz = round(cfg.cell_z / cfg.spacing_um)
    expected = nx * cfg.n_y_cells * nz
    assert sweep.n_cells(cfg) == expected
    assert sweep.n_cells(cfg) > 0


# --------------------------------------------------------------------------
# error rows: OOM / any build failure is a row, not a silent skip
# --------------------------------------------------------------------------


def test_measure_one_records_build_failure_without_raising(monkeypatch):
    cfg = PVGCConfig(spacing_um=0.05, design_grid_per_um=20,
                     sim_time_s=0.05e-12, L_design=6.0, pad_x=2.0, dpml=0.6,
                     t_box=1.5, t_sub=0.8, air_above=2.0, t_si=0.20)

    def _boom(*a, **k):
        raise RuntimeError("simulated OOM")

    monkeypatch.setattr(sweep.pvgc, "make_ce_value_and_grad", _boom)

    row = sweep.measure_one(cfg, [1.0], 5, p0=None, beta=None, repeat=1,
                            warmup=True)

    assert row["C"] == 5
    assert row["error_type"] == "RuntimeError"
    assert row["error_stage"] == "build"
    assert "simulated OOM" in row["error_msg"]
    assert row["forward_s"] is None
    assert row["vg_s"] is None
    assert row["ratio"] is None
    # n_cells/steps are config-derived, not simulation output: they must
    # still be populated even though the build failed
    assert row["n_cells"] > 0
    assert row["steps"] > 0


def test_measure_one_keeps_forward_row_when_only_vg_fails(monkeypatch):
    """forward succeeding while backward OOMs is the whole point of this
    sweep (backward cost grows with C, forward does not) — that row must
    keep its forward number and blank only vg_s/ratio."""
    cfg = PVGCConfig(spacing_um=0.05, design_grid_per_um=20,
                     sim_time_s=0.05e-12, L_design=6.0, pad_x=2.0, dpml=0.6,
                     t_box=1.5, t_sub=0.8, air_above=2.0, t_si=0.20)

    def _fake_build(cfg, p_in, num_checkpoints=20, **kw):
        def value_fn(p, beta):
            return 0.42     # jax.block_until_ready passes plain values through

        def vg_fn(p, beta):
            raise RuntimeError("simulated backward OOM")

        return vg_fn, None, None, None, None, value_fn

    monkeypatch.setattr(sweep.pvgc, "make_ce_value_and_grad", _fake_build)

    row = sweep.measure_one(cfg, [1.0], 5, p0=None, beta=None, repeat=1,
                            warmup=False)

    assert row["forward_s"] is not None
    assert row["error_stage"] == "vg"
    assert row["vg_s"] is None
    assert row["ratio"] is None


# --------------------------------------------------------------------------
# CSV rendering: None -> blank field, not "None" or 0
# --------------------------------------------------------------------------


def test_write_csv_blanks_none_fields(tmp_path):
    rows = [
        {"C": 1, "forward_s": 0.5, "vg_s": None, "ratio": None,
         "peak_bytes": None, "bytes_limit": None, "n_cells": 1000,
         "steps": 500},
        {"C": 2, "forward_s": 0.5, "vg_s": 12.0, "ratio": 24.0,
         "peak_bytes": 123456, "bytes_limit": 999999, "n_cells": 1000,
         "steps": 500},
    ]
    path = str(tmp_path / "sweep.csv")
    sweep.write_csv(path, rows)

    with open(path, newline="") as f:
        r = list(csv.DictReader(f))

    assert r[0]["C"] == "1"
    assert r[0]["vg_s"] == ""       # OOM row: blank, not "None"/"0"
    assert r[0]["ratio"] == ""
    assert r[0]["peak_bytes"] == ""
    assert r[1]["vg_s"] == "12.0"
    assert r[1]["peak_bytes"] == "123456"
