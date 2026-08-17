"""Batch forward-simulation dataset generation — a minimal, general mechanism.

Scope, deliberately narrow: this module answers ONE question — "how do I turn
a batch of sampled geometries into a self-describing, resumable dataset of
(rho, FOM) pairs a downstream ML framework can train a surrogate on?" — and
nothing else. It does NOT decide what a good sampling distribution is, what a
good label is beyond the existing pvgc CE figure of merit, or how the
resulting dataset should be split/normalized/consumed for training. Those are
research decisions that belong to the study using this module, made through
`kind`/`cfg`, not baked in here.

Three moving parts:

  sample_designs(kind, n, seed, cfg)  -- pluggable geometry sampler; yields
      physical density arrays rho (values in [0, 1]). Two generic kinds ship
      here (`uniform-grating`, `random-rho`); add a new one by adding an
      entry to SAMPLE_KINDS, no caller changes needed.

  generate_dataset(cfg, kind, n, run_dir, ...) -- runs one forward-only pvgc
      simulation per sampled design (through the same INVDX_FAST-gated
      forward loop `problems.pvgc.characterize` already uses — see
      `problems/pvgc.py`'s `_fdtd_forward`) and writes the result as npz
      shards + a JSON manifest, both written incrementally so a crash loses
      at most the shard in progress and a re-run of the same command skips
      whatever shards already exist on disk.

  the on-disk format: dataset_manifest.json (schema version, cfg snapshot,
      sampling kind/seed/n, per-shard sample count + sha256, units, axis
      order, invdx git hash) plus shard_NNNNN.npz files, each holding
      `rho` (N, n_design_voxels), `label_ce` (N,), and optionally `spectrum`
      (N, n_wavelengths). No new dependency: npz is numpy's own format, and
      nothing here imports torch or jax at module scope — fdtdx/jax are only
      touched inside the functions that actually run a simulation, so a pure
      numpy consumer can load `import invdx.datasets` to read the manifest
      schema without installing the GPU stack.

Currently wired to the pvgc problem only (`problems.pvgc`); a second problem
would plug in the same way `uniform-grating`/`random-rho` do — through a
`kind` (or, if the FOM itself needs to change, a `problem` argument threaded
the same way sample_designs threads `kind`).
"""

import hashlib
import itertools
import json
import os
import subprocess
import time
from dataclasses import asdict

import numpy as np

from . import runio

SCHEMA_VERSION = 1

MANIFEST_FILE = "dataset_manifest.json"
README_FILE = "README.txt"

# Defaults for the two generic sampling kinds below. These are NOT a claim
# about a good training distribution — just enough of a working default that
# `sample_designs` runs out of the box. A real study overrides them via the
# cfg._period_range_um / cfg._duty_range / cfg._rho_beta / cfg._rho_eta
# attributes (same "extra attribute on cfg" convention `problems.pvgc` uses
# for cfg._lams_um) or, more simply, by writing its own `kind`.
_DEFAULT_PERIOD_RANGE_UM = (0.4, 0.8)
_DEFAULT_DUTY_RANGE = (0.3, 0.7)
_DEFAULT_RHO_BETA = 64.0


# --------------------------------------------------------------------------
# geometry sampling
# --------------------------------------------------------------------------


def _sample_uniform_grating(n, seed, cfg):
    """Random (period, duty) uniform gratings, rasterized onto the design
    grid with the same `rasterize_teeth` the M1 optimizer's `--init grating`
    uses (see pvgc.rasterize_teeth for why pixel-rounding is not cosmetic)."""
    from .problems import pvgc

    rng = np.random.default_rng(seed)
    lo, hi = getattr(cfg, "_period_range_um", _DEFAULT_PERIOD_RANGE_UM)
    dlo, dhi = getattr(cfg, "_duty_range", _DEFAULT_DUTY_RANGE)
    for _ in range(n):
        period = float(rng.uniform(lo, hi))
        duty = float(rng.uniform(dlo, dhi))
        teeth = pvgc.uniform_grating_teeth(cfg, period, duty)
        yield pvgc.rasterize_teeth(cfg, teeth)


def _sample_random_rho(n, seed, cfg):
    """Random latent density, conic-filtered and tanh-projected — the same
    filter/projection chain `fab.transforms.ConicFilter1D` +
    `fdtdx.TanhProjection` apply inside the differentiable Device, run here
    numpy-side (fab.filters_np, the authoritative reference implementation)
    so no jax/fdtdx import is needed just to sample. The filter radius is
    cfg.filter_radius (== cfg.min_feature), so a high projection beta yields
    a near-binary density whose guaranteed feature size matches the same
    fabrication rule the optimizer enforces — "manufacturable" in the same
    sense the rest of this repo means it, not a stronger claim."""
    from .fab.filters_np import conic_filter_matrix, tanh_projection
    from .problems.pvgc import n_design_voxels

    rng = np.random.default_rng(seed)
    nx = n_design_voxels(cfg)
    W = conic_filter_matrix(nx, cfg.filter_radius, cfg.design_grid_per_um)
    beta = getattr(cfg, "_rho_beta", _DEFAULT_RHO_BETA)
    eta = getattr(cfg, "_rho_eta", cfg.eta_i)
    for _ in range(n):
        x = rng.uniform(0.0, 1.0, size=nx)
        yield np.asarray(tanh_projection(W @ x, beta, eta), dtype=float)


#: kind name -> generator(n, seed, cfg). Add a new sampling strategy by
#: adding an entry here; `sample_designs` needs no changes.
SAMPLE_KINDS = {
    "uniform-grating": _sample_uniform_grating,
    "random-rho": _sample_random_rho,
}


def sample_designs(kind, n, seed, cfg):
    """Yield `n` physical density arrays (each shape (n_design_voxels,),
    values in [0, 1]) for the requested `kind`. Deterministic in
    (kind, n, seed, cfg): re-calling with the same arguments reproduces the
    same sequence, which is what makes shard-level resume safe."""
    try:
        fn = SAMPLE_KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown sample kind {kind!r}; know: "
                         f"{sorted(SAMPLE_KINDS)}") from None
    yield from fn(n, seed, cfg)


# --------------------------------------------------------------------------
# forward measurement (one design -> one FOM), forward-only / INVDX_FAST path
# --------------------------------------------------------------------------


def _measure_one(cfg, rho, p_in_c, azimuth_sign, seed, lams_um=None):
    """rho -> {"ce": float, "spectrum": [float, ...] or absent}.

    Uses `problems.pvgc.characterize`, NOT the differentiable `ce_from_arrays`
    path: `characterize` runs through `_run`/`_fdtd_forward`, which is what
    honors INVDX_FAST (the vendored fast forward loop); the differentiable
    path calls `fdtdx.run_fdtd` directly to support checkpointed gradients
    and bypasses it. A dataset has no backward pass to support, so the cheap
    forward-only path is strictly the right one here.
    """
    from .problems import pvgc

    teeth = pvgc.profile_teeth(cfg, rho)
    meas = pvgc.characterize(cfg, teeth, p_in=p_in_c, azimuth_sign=azimuth_sign,
                             seed=seed)
    out = {"ce": float(meas["CE"])}
    if lams_um:
        spec = pvgc.characterize_spectrum(cfg, teeth, lams_um,
                                          azimuth_sign=azimuth_sign, seed=seed)
        out["spectrum"] = [float(r["CE"]) for r in spec["spectrum"]]
    return out


# --------------------------------------------------------------------------
# shard I/O (npz, atomic writes, sha256)
# --------------------------------------------------------------------------


def _shard_path(run_dir, idx):
    return os.path.join(run_dir, f"shard_{idx:05d}.npz")


def _shard_is_valid(path, expected_n):
    """A shard on disk counts as done only if it loads and has exactly the
    sample count this run expects — a mismatch (e.g. --n or --shard-size
    changed between invocations) forces a redo instead of silently mixing
    dataset generations."""
    try:
        with np.load(path) as z:
            if "rho" not in z or "label_ce" not in z:
                return False
            return (z["rho"].shape[0] == expected_n and
                   z["label_ce"].shape[0] == expected_n)
    except Exception:
        return False


def _write_shard(path, rho, label_ce, spectrum=None):
    """Atomic (.tmp.npz + os.replace), same pattern as optimize.save_state:
    a crash mid-write must never leave a half-file under the final name."""
    tmp = path + ".tmp.npz"          # np.savez appends .npz unless present
    kwargs = dict(rho=np.asarray(rho, dtype=float),
                  label_ce=np.asarray(label_ce, dtype=float))
    if spectrum is not None:
        kwargs["spectrum"] = np.asarray(spectrum, dtype=float)
    np.savez(tmp, **kwargs)
    os.replace(tmp, path)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_hash():
    try:
        r = subprocess.run(["git", "-C", _repo_root(), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr).strip() or None
    except Exception:
        return None


def _kind_params(kind, cfg):
    if kind == "uniform-grating":
        return {"period_range_um": list(getattr(cfg, "_period_range_um",
                                                _DEFAULT_PERIOD_RANGE_UM)),
                "duty_range": list(getattr(cfg, "_duty_range",
                                           _DEFAULT_DUTY_RANGE))}
    if kind == "random-rho":
        return {"beta": float(getattr(cfg, "_rho_beta", _DEFAULT_RHO_BETA)),
                "eta": float(getattr(cfg, "_rho_eta", cfg.eta_i))}
    return {}


def _readme_text(manifest):
    n = manifest["n_samples_written"]
    nx = manifest["n_design_voxels"]
    lams = manifest["lams_um"]
    s = manifest["sampling"]
    lines = [
        "invdx pvgc forward-simulation dataset",
        "======================================",
        "",
        f"schema_version: {manifest['schema_version']}",
        f"invdx git hash: {manifest['invdx_git_hash']}",
        f"sampling: kind={s['kind']} seed={s['seed']} n={s['n']} "
        f"shard_size={s['shard_size']}",
        f"kind_params: {s['kind_params']}",
        f"samples written: {n} across {len(manifest['shards'])} shard(s)",
        f"design vector length (n_design_voxels): {nx}",
    ]
    if lams:
        lines.append(f"spectrum wavelengths (um): {lams}")
    lines += [
        "",
        "Files",
        "-----",
        f"{MANIFEST_FILE}   - full schema, cfg snapshot, per-shard sha256",
        "shard_NNNNN.npz         - one shard: rho (N, n_design_voxels), "
        "label_ce (N,)" + (", spectrum (N, n_wavelengths)" if lams else ""),
        "",
        "Sampling strategy and label design are research decisions left to",
        "the caller (see invdx.datasets.sample_designs / SAMPLE_KINDS); this",
        "dataset only records the mechanism's own defaults for the `kind`",
        "used to generate it (see dataset_manifest.json -> sampling.kind_params).",
        "",
        "Read with numpy",
        "----------------",
        ">>> import numpy as np",
        ">>> z = np.load('shard_00000.npz')",
        ">>> rho, label_ce = z['rho'], z['label_ce']",
        "",
        "Read with torch (illustrative only — torch is not a dependency of",
        "this repo or this dataset)",
        "--------------------------------------------------------------",
        ">>> import torch, numpy as np",
        ">>> z = np.load('shard_00000.npz')",
        ">>> rho = torch.from_numpy(z['rho']).float()",
        ">>> label_ce = torch.from_numpy(z['label_ce']).float()",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


def generate_dataset(cfg, kind, n, run_dir, shard_size=8, lams_um=None,
                     on_shard=None):
    """Sample `n` designs of `kind`, measure each forward-only, write
    npz shards + dataset_manifest.json into `run_dir` (an existing run
    directory, e.g. from `cli.start_run`).

    Resumable: re-calling with the identical (kind, n, seed via cfg.seed,
    shard_size, and any kind-specific cfg._* sampling params) against the
    same run_dir skips every shard already valid on disk and only computes
    the remainder — safe because `sample_designs` is deterministic in those
    arguments, so re-generating the design sequence from scratch reproduces
    the same shard grouping. Changing any of them and reusing run_dir mixes
    generations; the CLI should use a fresh run_dir for a new configuration.

    on_shard(entry) — optional callback invoked after each shard (skipped or
    freshly computed) with its manifest entry, for progress reporting.

    Returns the final manifest dict.
    """
    from .problems import pvgc

    if n <= 0:
        raise ValueError("n must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    lams = [float(v) for v in lams_um] if lams_um else None
    seed = int(cfg.seed)

    manifest_path = os.path.join(run_dir, MANIFEST_FILE)

    # design-independent: measured once for the whole dataset, exactly like
    # the M1 optimizer measures it once for a whole optimization run.
    p_in_c, azimuth_sign, slope = pvgc.calibrated_beam(cfg, seed=seed)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invdx_git_hash": _git_hash(),
        "problem": "pvgc",
        "sampling": {"kind": kind, "seed": seed, "n": int(n),
                     "shard_size": int(shard_size),
                     "kind_params": _kind_params(kind, cfg)},
        "cfg": asdict(cfg),
        "n_design_voxels": pvgc.n_design_voxels(cfg),
        "beam_calibration": {"p_in": p_in_c, "azimuth_sign": azimuth_sign,
                             "tilt_slope_rad_per_um": slope,
                             "lam_c_um": cfg.lam_c},
        "lams_um": lams,
        "units": {
            "lengths_in_cfg": "micrometers (um), per invdx.config.BaseConfig",
            "rho": "material density in [0, 1] (binarized at 0.5 before "
                  "simulation, same convention as pvgc.profile_teeth)",
            "label_ce": "linear coupling efficiency at cfg.lam_c: "
                       "P_mode / P_in (dimensionless power ratio, not dB)",
            "spectrum": "linear coupling efficiency at each wavelength in "
                       "lams_um (dimensionless); present only when lams_um "
                       "is set",
        },
        "axis_order": {
            "rho": ["sample", "design_voxel"],
            "label_ce": ["sample"],
            "spectrum": ["sample", "wavelength"],
        },
        "shards": [],
        "n_samples_written": 0,
        "complete": False,
    }
    runio.save_json(manifest_path, manifest)

    gen = sample_designs(kind, n, seed, cfg)
    offset = 0
    while offset < n:
        idx = len(manifest["shards"])
        count = min(shard_size, n - offset)
        batch = [np.asarray(r, dtype=float)
                for r in itertools.islice(gen, count)]
        if len(batch) != count:
            raise RuntimeError(
                f"sample_designs({kind!r}) produced {len(batch)} designs, "
                f"expected {count} — sampler is not honoring `n`")

        path = _shard_path(run_dir, idx)
        skipped = os.path.exists(path) and _shard_is_valid(path, count)
        if not skipped:
            ces = []
            specs = [] if lams else None
            for rho in batch:
                m = _measure_one(cfg, rho, p_in_c, azimuth_sign, seed,
                                 lams_um=lams)
                ces.append(m["ce"])
                if lams:
                    specs.append(m["spectrum"])
            _write_shard(path, np.stack(batch), np.asarray(ces),
                        np.asarray(specs) if lams else None)

        entry = {"file": os.path.basename(path), "n_samples": count,
                 "sha256": _sha256_file(path), "index_start": offset,
                 "skipped_existing": skipped}
        manifest["shards"].append(entry)
        manifest["n_samples_written"] += count
        runio.save_json(manifest_path, manifest)   # write-per-shard: crash-safe
        if on_shard:
            on_shard(entry)
        offset += count

    manifest["complete"] = True
    runio.save_json(manifest_path, manifest)
    with open(os.path.join(run_dir, README_FILE), "w") as f:
        f.write(_readme_text(manifest))
    return manifest
