"""Integration test for the generic handoff exporter (invdx.export.handoff):
runs it (CPU only) against the real smoke3a pvgc-opt run dir and checks the
resulting package is internally consistent.

Skips if that run dir isn't present: runs/ is gitignored (every machine has
its own local runs), so this is a local/CPU integration check rather than a
pure-math unit test.
"""

import csv
import hashlib
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from invdx.export.handoff import export_handoff

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO_ROOT, "runs", "20260817-023418-pvgc-opt-smoke3a")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.skipif(not os.path.isdir(RUN_DIR),
                    reason=f"local-only run dir not present: {RUN_DIR}")
def test_handoff_export_smoke3a(tmp_path):
    out_dir, manifest = export_handoff(RUN_DIR, str(tmp_path / "handoff"))

    for name in ("eps.npy", "design_rho.npy", "spectrum.csv",
                "manifest.json", "README.txt"):
        assert os.path.exists(os.path.join(out_dir, name)), name
    assert manifest["notes"] == []  # smoke3a has everything: nothing skipped

    # manifest sha256 matches the actual bytes on disk, for every listed file
    for name, meta in manifest["files"].items():
        assert _sha256(os.path.join(out_dir, name)) == meta["sha256"], name

    # eps grid shape/spacing/theta match the run's own config.json
    with open(os.path.join(RUN_DIR, "config.json")) as f:
        cfg = json.load(f)
    spacing = cfg["spacing_um"]
    X0 = cfg["L_design"] / 2 + cfg["pad_x"] + cfg["dpml"]
    Z0 = cfg["t_box"] + cfg["t_sub"] + cfg["dpml"]
    cell_x = 2 * X0
    cell_z = Z0 + cfg["t_si"] + cfg["air_above"] + cfg["dpml"]
    nx = int(round(cell_x / spacing))
    nz = int(round(cell_z / spacing))

    eps = np.load(os.path.join(out_dir, "eps.npy"))
    assert eps.shape == (nx, nz)
    assert manifest["eps_grid"]["shape"] == [nx, nz]
    assert manifest["eps_grid"]["spacing_um"] == spacing
    assert manifest["theta_deg"] == cfg["theta_deg"]

    # design_rho.npy is an exact copy of the run's design vector
    rho_orig = np.load(os.path.join(RUN_DIR, "design_rho.npy"))
    rho_copy = np.load(os.path.join(out_dir, "design_rho.npy"))
    assert np.array_equal(rho_orig, rho_copy)
    assert manifest["design_grid"]["n_design_voxels"] == rho_orig.shape[0]

    # spectrum.csv parses and matches results.json's binarized re-measurement
    # (smoke3a has no dense spectrum, so this exercises the fallback path)
    with open(os.path.join(RUN_DIR, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(out_dir, "spectrum.csv")) as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"lam_um", "CE", "CE_dB"}
    assert len(rows) == 1
    assert float(rows[0]["lam_um"]) == pytest.approx(results["lams_um"][0])
    assert float(rows[0]["CE_dB"]) == pytest.approx(
        results["binarized"]["CE_dB"])
    assert "binarized" in manifest["spectrum"]["source"]
