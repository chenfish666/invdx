"""Unit tests for the reporting toolkit: 3dB bandwidth math, viz rendering,
report table generation. Pure post-processing — no simulation."""

import json
import os

import numpy as np
import pytest

from invdx.problems import grating_coupler


def _spec(lams, ces):
    return [{"lam_um": l, "CE_dB": c} for l, c in zip(lams, ces)]


def test_bandwidth_3db_interpolated():
    # triangle peaked at 1.31, slope 1 dB per 10 nm -> 3 dB down at +/-30 nm
    lams = np.arange(1.27, 1.351, 0.01)
    ces = -10 - np.abs(lams - 1.31) * 100
    bw, lo, hi, note = grating_coupler.bandwidth_3db(_spec(lams, ces))
    assert note is None
    assert bw == pytest.approx(0.060, abs=1e-9)
    assert lo == pytest.approx(1.28)
    assert hi == pytest.approx(1.34)


def test_bandwidth_3db_clipped_is_honest():
    # flat spectrum: never drops 3 dB -> clamped to range, note set
    lams = np.arange(1.27, 1.351, 0.01)
    bw, lo, hi, note = grating_coupler.bandwidth_3db(_spec(lams, np.full(len(lams), -10.0)))
    assert note is not None and "lower bound" in note
    assert bw == pytest.approx(lams[-1] - lams[0])


def test_viz_and_report_from_synthetic_run(tmp_path):
    from invdx import report
    from invdx.viz import render_run

    res = {
        "spectrum": _spec([1.29, 1.31, 1.33], [-25.0, -20.0, -24.0]),
        "peak": {"lam_um": 1.31, "CE_dB": -20.0},
        "linewidth": {"min_solid_um": 0.17, "min_void_um": 0.35,
                      "n_teeth": 13},
        "bandwidth_3db": {"bw_nm": 25.0, "lam_lo_um": 1.30,
                          "lam_hi_um": 1.325, "note": None},
        "corners": {"erode_10nm": {
            "spectrum": _spec([1.29, 1.31, 1.33], [-26.0, -21.0, -25.0]),
            "peak": {"lam_um": 1.31, "CE_dB": -21.0}}},
        "source_run": "/x/robust-test",
    }
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "results.json", "w") as f:
        json.dump(res, f)

    made = render_run(str(d), pdf=True)
    assert any(m.endswith("spectrum.png") for m in made)
    for m in made:
        assert os.path.getsize(m) > 5000
        assert os.path.exists(m[:-4] + ".pdf")

    table = report.markdown_table([report._row(res)])
    assert "robust-test" in table and "-20.00" in table and "170/350" in table
