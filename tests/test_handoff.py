"""Integration test for the generic handoff exporter (invdx.export.handoff):
runs it (CPU only) against a real run directory and checks the resulting
package is internally consistent.

Skips when no suitable run is present: runs/ is gitignored (every machine has
its own local runs), so this is a local/CPU integration check rather than a
pure-math unit test.

The run is DISCOVERED rather than named. A hardcoded directory name would be
one rename away from turning this test into a permanent silent skip -- it
would still report green while checking nothing, which is the failure mode
this suite exists to catch elsewhere.
"""

import csv
import glob
import hashlib
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from invdx.export.handoff import export_handoff

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_run_dir():
    """Newest run that exercises the single-point spectrum fallback path.

    The two conditions below are what the assertions downstream actually
    depend on, spelled out rather than assumed of one known directory: the
    export must skip nothing (`notes == []`), and the spectrum must come from
    the binarized re-measurement rather than a dense sweep (one wavelength).

    The glob is restricted to `20*` on purpose. Run directories are named
    `%Y%m%d-%H%M%S-<tag>`, so within that convention a reverse lexical sort
    IS newest-first -- but a hand-copied directory sorts by its first letter
    instead, and would quietly win the "newest" contest against every real
    run. Matching the convention keeps the name sort meaning what it says.
    """
    for d in sorted(glob.glob(os.path.join(REPO_ROOT, "runs", "20*")),
                    reverse=True):
        try:
            if not all(os.path.isfile(os.path.join(d, n)) for n in
                       ("config.json", "design_rho.npy", "results.json")):
                continue
            with open(os.path.join(d, "results.json")) as f:
                r = json.load(f)
            if "binarized" in r and len(r.get("lams_um", [])) == 1:
                return d
        except (OSError, ValueError):
            continue
    return None


RUN_DIR = _find_run_dir()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.skipif(RUN_DIR is None,
                    reason="no local run dir with a single-point spectrum")
def test_handoff_export_is_internally_consistent(tmp_path):
    out_dir, manifest = export_handoff(RUN_DIR, str(tmp_path / "handoff"))

    for name in ("eps.npy", "design_rho.npy", "spectrum.csv",
                "manifest.json", "README.txt"):
        assert os.path.exists(os.path.join(out_dir, name)), name
    assert manifest["notes"] == []  # this run has everything: nothing skipped

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
    # (the selected run has no dense spectrum, so this exercises the fallback)
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


def test_handoff_rejects_rank2_design(tmp_path):
    """A rank-2 design_rho.npy must be refused, not exported as its
    x-projection: this exporter is the 1D chain (teeth run-length ->
    extruded rectangles), and rho.shape[0] would silently become
    n_design_voxels while the y structure vanished."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(run_dir / "design_rho.npy", np.zeros((8, 8)))
    with open(run_dir / "config.json", "w") as f:
        json.dump({}, f)

    with pytest.raises(ValueError, match="1D chain"):
        export_handoff(str(run_dir), str(tmp_path / "handoff"))
