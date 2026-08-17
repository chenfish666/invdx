"""Generic handoff export — package a run directory's design/results into a
tool-neutral bundle for cross-checking with any external solver (commercial
or in-house), so a result never depends on invdx to be re-verified.

    python -m invdx.export.handoff <run_dir> [--out <dir>]

Output (progressive: a file is only written when its source data exists in
the run dir; anything skipped is recorded in manifest.json["notes"]):

    eps.npy         rasterized relative-permittivity grid of the design,
                    built from design_rho.npy via the existing
                    rho -> teeth -> permittivity-raster path
                    (problems.pvgc.profile_teeth + eps_grid_xz)
    design_rho.npy  verbatim copy of the run's design density vector
    spectrum.csv    lam_um,CE,CE_dB coupling-efficiency spectrum, read from
                    results.json (dense spectrum if present, else the
                    single-point binarized-design re-measurement, else the
                    raw top-level CE/CE_dB as a last resort)
    manifest.json   units, grid/axis definitions, provenance, and a
                    description + sha256 for every other file
    README.txt      short self-description of the package

Runs entirely on CPU: eps rasterization is plain numpy (no simulation), and
importing the pvgc problem module only pulls in fdtdx/jax for its dataclass
and constant definitions. JAX_PLATFORMS is pinned to "cpu" below (before
fdtdx/jax are imported) so this never touches a GPU.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from ..problems import pvgc

_PVGC_FIELDS = {f.name for f in dataclasses.fields(pvgc.PVGCConfig)}
_PVGC_MARKERS = ("lam_c", "L_design", "design_grid_per_um", "t_si")


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _git_hash(repo_root):
    if repo_root is None:
        return None
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Config / design loading
# --------------------------------------------------------------------------


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _is_pvgc_config(cfg_dict):
    return bool(cfg_dict) and all(k in cfg_dict for k in _PVGC_MARKERS)


def _pvgc_config_from_dict(cfg_dict):
    kwargs = {k: v for k, v in cfg_dict.items() if k in _PVGC_FIELDS}
    if "beta_schedule" in kwargs:
        kwargs["beta_schedule"] = tuple(kwargs["beta_schedule"])
    return pvgc.PVGCConfig(**kwargs)


def _rasterize_eps(cfg, rho_binary):
    """design_rho -> teeth -> permittivity raster, via the existing pvgc
    functions (profile_teeth, eps_grid_xz); no new rasterization logic."""
    nx = int(round(cfg.cell_x / cfg.spacing_um))
    nz = int(round(cfg.cell_z / cfg.spacing_um))
    teeth = pvgc.profile_teeth(cfg, rho_binary)
    eps = pvgc.eps_grid_xz(cfg, teeth, nx, nz)
    return eps


# --------------------------------------------------------------------------
# Spectrum extraction (progressive fallback across what a run dir may have)
# --------------------------------------------------------------------------


def _spectrum_rows(results):
    """(rows, source_note) with rows = [(lam_um, CE, CE_dB), ...], or
    (None, reason) when no CE data is found."""
    if not results:
        return None, "no results.json found in the run dir"

    spec = results.get("spectrum")
    if isinstance(spec, list) and spec:
        rows = [(r["lam_um"], r["CE"], r["CE_dB"]) for r in spec]
        return rows, ("results.json['spectrum']: dense CE(lambda) spectrum "
                      "from a multi-wavelength FDTD measurement")

    binarized = results.get("binarized")
    lams = results.get("lams_um") or []
    if isinstance(binarized, dict) and "CE" in binarized and lams:
        rows = [(lams[0], binarized["CE"], binarized["CE_dB"])]
        return rows, ("results.json['binarized']: single-wavelength FDTD "
                      "re-measurement of the binarized design_rho.npy "
                      "(the actual measured device, not an optimizer trace "
                      "value)")

    if "CE_dB" in results and lams:
        ce_db = results["CE_dB"]
        ce = results.get("CE", 10 ** (ce_db / 10))
        rows = [(lams[0], ce, ce_db)]
        return rows, ("results.json top-level CE/CE_dB: raw value from the "
                      "run, may be a continuous-design optimizer trace value "
                      "rather than an independent re-measurement (see the "
                      "project README's binarization-gap note)")

    return None, "no CE/spectrum data found in results.json"


def _write_spectrum_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam_um", "CE", "CE_dB"])
        for lam_um, ce, ce_db in rows:
            w.writerow([lam_um, ce, ce_db])


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------

_README = """\
INVDX HANDOFF PACKAGE
======================

A self-contained, tool-neutral snapshot of one invdx design/verification
run, for cross-checking with any external solver (commercial or in-house).
Reading it requires only numpy/a CSV reader/a JSON reader -- no invdx
dependency.

Files (see manifest.json for exact units, axes, and per-file checksums)
-------------------------------------------------------------------------
- manifest.json   Machine-readable index: units, grid/axis definitions,
                  provenance (source run, invdx git hash, generation
                  time), and a one-line description + sha256 checksum for
                  every other file in this package.
- eps.npy         Rasterized relative permittivity (dielectric constant)
                  grid of the design. Shape/axis order/spacing/extent are
                  in manifest.json["eps_grid"].
- design_rho.npy  Verbatim copy of the source run's design density vector
                  (one value per design-grid pixel).
- spectrum.csv    Coupling-efficiency spectrum: lam_um, CE, CE_dB. See
                  manifest.json["spectrum"]["source"] for how these
                  numbers were obtained.
- README.txt      This file.

Reading the .npy files (numpy)
-------------------------------
    import numpy as np
    eps = np.load("eps.npy")           # 2D permittivity grid
    rho = np.load("design_rho.npy")    # 1D design density vector

All lengths are in micrometers (um) unless a field says otherwise.
"""


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def export_handoff(run_dir, out_dir=None):
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"not a directory: {run_dir}")
    out_dir = os.path.abspath(out_dir) if out_dir else os.path.join(
        run_dir, "handoff")
    os.makedirs(out_dir, exist_ok=True)

    repo_root = _repo_root()
    cfg_dict = _load_json(os.path.join(run_dir, "config.json")) or {}
    results = _load_json(os.path.join(run_dir, "results.json"))

    notes = []
    files = {}  # name -> {"description": ..., "sha256": ...}
    eps_info = None
    design_info = None

    rho_path = os.path.join(run_dir, "design_rho.npy")
    if os.path.exists(rho_path):
        dst = os.path.join(out_dir, "design_rho.npy")
        shutil.copy2(rho_path, dst)
        files["design_rho.npy"] = {
            "description": "Verbatim copy of the source run's binarized "
                           "design density vector (one value per design "
                           "grid pixel).",
            "sha256": _sha256_file(dst)}
        rho = np.load(rho_path)
        design_info = {"n_design_voxels": int(rho.shape[0]),
                       "design_grid_per_um": cfg_dict.get("design_grid_per_um"),
                       "min_feature_um": cfg_dict.get("min_feature")}

        if _is_pvgc_config(cfg_dict):
            cfg = _pvgc_config_from_dict(cfg_dict)
            rho_binary = (np.asarray(rho) > 0.5).astype(float)
            eps = _rasterize_eps(cfg, rho_binary)
            eps_path = os.path.join(out_dir, "eps.npy")
            np.save(eps_path, eps)
            files["eps.npy"] = {
                "description": "Rasterized relative permittivity "
                               "(dielectric constant) grid of the design, "
                               "crisp cell fill (no subpixel averaging), "
                               "including the PML padding region.",
                "sha256": _sha256_file(eps_path)}
            eps_info = {
                "shape": list(eps.shape),
                "axis_order": ["x_um", "y_um"],
                "axis_note": ("pvgc coordinates: x = propagation direction "
                              "(grating/waveguide axis), y = vertical "
                              "(y=0 at the bottom of the silicon device "
                              "layer); both axes span the full simulation "
                              "cell, PML included"),
                "spacing_um": cfg.spacing_um,
                "extent_um": {"x": [-cfg.X0, cfg.X0],
                             "y": [-cfg.Z0, cfg.cell_z - cfg.Z0]},
                "sample_positions": ("pixel centers: "
                                     "x_i = -X0 + (i+0.5)*spacing_um, "
                                     "y_j = -Z0 + (j+0.5)*spacing_um"),
                "is_binarized": bool(np.all((rho == 0) | (rho == 1))),
                "unique_permittivity_values":
                    [float(v) for v in np.unique(eps)],
            }
        else:
            notes.append(
                "eps.npy not generated: config.json does not look like a "
                "pvgc design (missing one of " + ", ".join(_PVGC_MARKERS) +
                "); automatic rasterization is only implemented for the "
                "pvgc problem so far.")
    else:
        notes.append("design_rho.npy not found in run dir; eps.npy and "
                     "design_rho.npy were not exported.")

    spectrum_info = None
    rows, spec_note = _spectrum_rows(results)
    if rows:
        csv_path = os.path.join(out_dir, "spectrum.csv")
        _write_spectrum_csv(rows, csv_path)
        files["spectrum.csv"] = {
            "description": "Coupling-efficiency spectrum: free-space "
                           "wavelength (um), linear coupling efficiency "
                           "CE, and CE in dB.",
            "sha256": _sha256_file(csv_path)}
        spectrum_info = {"source": spec_note, "n_points": len(rows),
                         "columns": ["lam_um", "CE", "CE_dB"]}
    else:
        notes.append(f"spectrum.csv not generated: {spec_note}")

    readme_path = os.path.join(out_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(_README)
    files["README.txt"] = {
        "description": "Human-readable guide to this handoff package.",
        "sha256": _sha256_file(readme_path)}

    source_run_dir = (os.path.relpath(run_dir, repo_root)
                      if repo_root and run_dir.startswith(repo_root)
                      else os.path.basename(run_dir))
    manifest = {
        "generator": "invdx.export.handoff",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "invdx_git_hash": _git_hash(repo_root),
        "source_run_dir": source_run_dir,
        "units": "micrometers (um) for all lengths; degrees for angles",
        "theta_deg": cfg_dict.get("theta_deg"),
        "eps_grid": eps_info,
        "design_grid": design_info,
        "spectrum": spectrum_info,
        "files": files,
        "notes": notes,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("-o", "--out", default=None,
                   help="output dir (default: <run_dir>/handoff)")
    args = p.parse_args()

    out_dir, manifest = export_handoff(args.run_dir, args.out)
    for name in manifest["files"]:
        print(f"[handoff] {os.path.join(out_dir, name)}")
    for note in manifest["notes"]:
        print(f"[handoff] note: {note}")
    print(f"[done] {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
