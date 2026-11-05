> **English** · [繁體中文](journal.zh-TW.md)

[← back to docs index](README.md)

# Journal (append-only)

Working log of what actually happened, in order. Numbers cite their source
(log file, report path, or the file or test that carries the result).
Entries are appended, never rewritten; retractions go to `RETRACTIONS.md`
instead of editing history here.

One exception, made on 2026-11-05: the entries were edited to drop the
details of the example devices they were measured on, so that this log
describes the toolbox only. The unedited log is in the git history.

Sources given as `runs/…` or `spack/env/install.log` are local run artifacts,
not committed (see `.gitignore`): they identify which run a number came from,
they are not links a reader can follow. Sources given as paths under `src/`,
`scripts/`, `tests/`, or `spack/` are files in this repository — those are the
citations a reader can open.

---

## 2026-08-17 — Environment reproducibility (L1 uv, L2 spack)

**L1: uv takes over the Python/GPU layer**

- `pyproject.toml` pins `jax[cuda12]==0.11.0` + `fdtdx==0.6.2`; `uv.lock` freezes
  148 packages. Existing conda envs untouched (fallback).
- Acceptance: all gates G0–G5 green (G5 `[ok]`, not skipped); benchmark parity
  vs `runs/benchmark_fast_v6.json` — `fast` (this repo's accelerated forward
  loop) hot, i.e. timed after the compile run, 1250.832 vs 1250.799
  Mcell-steps/s; `vanilla` (stock `fdtdx.run_fdtd`) 636.748 vs 638.033;
  `peak_mem_gb` identical at 6.015903; `max|dE|=max|dH|=0.0` (bitwise
  contract: the two fields must be exactly equal, no drift allowed).
  Source: `runs/bench_uv_verify.json`.

**L2: meep built from source via spack**

- spack pinned two layers deep: tool `v1.2.0` + packages repo `v2026.06.0`
  (`spack/env/spack.yaml`). 106-package chain, `reuse: false`, system gcc 11.4,
  hardlink view at `spack/env/.spack-env/view` (= `meep_bridge.py` default path).
- Build incident: first install failed at meep itself — SWIG 4.4.1-generated
  bindings hit the `structmember.h` `READONLY` macro vs `meep.hpp` enum clash
  (known SWIG ≥4.1 conflict, fixed upstream after 1.29). Fix: pin `swig @=4.0.2`
  (`@4.0.2` without `=` matches the swig-fortran fork — trap). Second install:
  meep@1.29.0 built in 2m51s. Source: `spack/env/install.log`.
- Acceptance: bare `view/bin/python` imports meep 1.29.0 with no activation;
  `mpirun -np 2` sees 2 ranks; G5 cross-engine agreement
  `T_analytic=0.73978, T_fdtdx=0.74412, T_meep=0.74230` — fdtdx vs meep 0.24%
  (tol 10%). Clean-clone drill: fresh clone + `spack install` (1.3s, view only)
  + `make smoke-meep` → 1.29.0. Source: `runs/20260817-022058-gates/gates_report.json`.
- Known measurement artifact: a full `make gates` run OOMed in G2 Part C while
  a concurrent development run held GPU memory on both cards (JAX
  preallocation). Not a spack issue; gates to be re-run serially on clean
  GPUs. Source: `runs/20260817-021820-gates/gates_report.json`.

## 2026-08-17 — First production inverse-design driver

- New differentiable path: one `fdtdx.Device` over the design window, one
  voxel per design pixel, replacing a block-geometry route that binarizes the
  design and was never differentiable. FOM = a jnp twin of the measurement
  chain's mode overlap; the V3 consistency check (the V-numbers are this
  milestone's verification items, a separate series from the G0–G5 gates)
  between the differentiable FOM and the block-geometry chain on the same
  binary design: the two routes agree to |Δ| = 2.5e-6 dB (threshold 0.05).
- G2 gained Part C (3-voxel FD on a real design path): rel err
  0.0008% / 0.177% / 0.067% vs 5% tolerance, with a signal-floor sampler
  (278/500 voxels above 5% of peak gradient are eligible — FD on
  tanh-saturated voxels is float32 cancellation noise, so tolerance was NOT
  loosened). All six gates verified at HEAD across independent runs
  (`runs/20260817-032448-gates` for G0–G4 serial on a clean GPU;
  G5 re-run post-fix, 99.73 s).
- Two findings from implementation: (1) rasterizing a starting design by
  pixel-centre inclusion jittered its feature widths by ±3.5% on the 20 nm
  grid and moved the FOM by more than an order of magnitude — the rasterizer
  now rounds edges the way fdtdx places blocks; (2) `VAR= cmd` sets an env var
  to empty string, which the bridge treated as a path — `meep_bridge.py` now
  falls back to the default view when the variable is set but empty.
- Loop smoke (0.15 ps, 3 iters incl. one resume): the FOM improved
  monotonically across the resume boundary and the binarization gap stayed
  negligible at the starting beta.
- Iteration timings measured under CPU/GPU contention are recorded but not
  quotable as performance figures (measurement discipline).

## 2026-08-17 (night) — Slurm launch; a full-scale run stopped by its own safety gate

- Slurm path validated on the local cluster: `gres` assigns
  `CUDA_VISIBLE_DEVICES` 0/1 correctly to concurrent jobs, and the sbatch
  script derives its run directory from the job ID with no changes. An sbatch
  smoke run and a scancel-mid-iteration requeue drill both passed —
  `history.csv` iterations 0,1,2,3 continuous across the kill/resume
  boundary, no duplicate rows.
- The first formal full-scale run was stopped by the pre-run gradcheck:
  voxel 213, rel err 6.29% vs the 5% tolerance, deterministic under the fixed
  seed, so it was recorded as `gradcheck_failed` rather than silently
  retried. A second configuration passed gradcheck (1.93%) and ran overnight.
- Root cause, from an h-scan on GPU (sweeping the finite-difference step h):
  **FD truncation, not an adjoint defect**
  — the residual scales as h³ (×0.126 per h-halving), Richardson
  extrapolation from the same runs agrees with the adjoint to 0.0106%, the
  float32 noise floor is 200× below the observed residual, and the failing
  voxels are not low-signal. Longer runs (0.8 ps vs 0.15 ps) inflate FOM
  curvature in design space, which is why smoke-scale checks passed. The
  earlier "float32 cancellation" attribution for this failure mode is
  retracted — see `RETRACTIONS.md`. Fix: a two-h Richardson gradcheck plus an
  FD self-consistency indicator (`gradcheck()` in the optimization driver,
  mirrored in `src/invdx/gates/g2_gradcheck.py` Part C); tolerances and the
  signal floor are unchanged.

**L2 stage two: meep 1.34.0 via project-owned spack package repo**

- `spack/spack_repo/invdx/` carries a full copy (not a subclass — spack
  constraints can only tighten under inheritance) of the upstream meep recipe
  with three changes: `version("1.34.0", ...)`, python gated `@:3.11` for
  `@:1.31` / `3.11:3.13` for `@1.32:`, `py-numpy@2:` for `@1.32:`.
- sha256 finding: the same v1.34.0 tag has two official artifacts. The GitHub
  release dist tarball (`3c9284…60bc6`, the conda-forge one) lacks
  `python/numpy.i` and fails at `No rule to make target 'numpy.i'`; the git-tag
  archive (`1fa6dd…78ea4`) + autoreconf builds. Both hashes verified directly
  against github.com. Recipe uses the git-tag archive.
- SWIG experiment: the `@=4.0.2` pin (needed for 1.29's READONLY clash) was
  removed and meep 1.34.0 built cleanly with swig 4.4.1 in 3m16s — upstream fix
  confirmed. Chain now: python 3.13.13, numpy 2.4.6 (conda-baseline era).
- Acceptance: bare view import 1.34.0; 2 MPI ranks; `make smoke-meep` via
  default path reports 1.34.0; G5 re-run twice from scratch (once on GPU 1, then an
  independent re-run) — `T_meep=0.7423000529144463`, bit-identical to the
  1.29 result on this case; lock reproducible after `concretize --force`.
  Source: `runs/20260817-025817-gates/gates_report.json`, `spack/env/install.log`.

## 2026-08-17 — Bug: resume recomputed beta instead of reading the checkpoint

Recovering the designs from two deliberately stopped runs (both really at
beta=64 on disk) surfaced a resume bug: `run_loop`'s pre-loop `state.beta`
was `beta_for_iter(cfg, iteration, n_iters)` — recomputed from whatever
`n_iters` the resuming invocation passed, not read from `opt_state.npz`. The
natural way to finalize a killed run, `--resume <dir> --iters <iters_done>`
(0 extra iterations, just wanting to write the final design out), shrinks the
schedule denominator and silently jumps a stage: both runs recomputed
beta=128 instead of the true beta=64, which would have baked a
sharper-than-actual TanhProjection into `design_rho.npy`/`design_rho_cont.npy`.
Worked around at the time with a throwaway script (not committed) that read
`state.beta` off disk directly instead of going through `--resume`.

Fix: `run_loop` now seeds `state.beta` from the checkpoint's own `beta`
field on resume, and only from `beta_for_iter` on a fresh start; the
schedule still advances normally as new iterations run, since each
iteration inside the loop keeps recomputing `beta` from the current
`it`/`n_iters` regardless. The driver gained `--finalize-only`, which loads a
checkpoint and reruns the finalization tail (`design_rho*.npy`,
`results.json`) with zero optimizer iterations — the throwaway script's
functionality, promoted to a real flag, after which the throwaway script was
deleted. Verified against the real runs: `--finalize-only` reproduced the
throwaway script's `design_rho.npy` byte-for-byte (sha256 match). Regression
tests are under `tests/`.

## 2026-11-05 — README rewrite, checked against what the code does

Re-ran what runs without a GPU, on a 4-core Intel Xeon cloud VM (CPU only,
no Meep).

- `pytest tests/` (what G0 wraps): 320 passed, 1 skipped (a test that needs a
  real run directory), 9 min on 4 cores.
- The toy-engine adjoint script failed its own gradcheck: one pixel at a
  relative error of 2.28e-5 against the 1e-5 bar, with every adjoint value
  equal to the recorded transcript. An h-scan put the failure on float64
  rounding: 1e-6 to 2e-5 at h = 1e-4, 1.1e-7 to 1.9e-7 at h = 1e-3, 1.6e-5 at
  h = 1e-2 (where truncation takes over). h is now 1e-3; the rerun passes
  (worst 1.92e-7) and every optimization line matches the recorded
  transcript to the printed digit.
- `docs/dependencies.md` credited time-reversible backpropagation with making
  the optimization fit on one GPU. The production path is checkpointed, and
  reversible mode had already been measured as worse on these scenes.
- The README and the docs now describe the toolbox only; pages written
  around specific example devices were removed.

Where the numbers the README quotes come from: G5 and G2 Part C, the
2026-08-17 entries above; G4, a forward/reverse mismatch of 0.076 dB against
the 0.5 dB gate, bitwise identical before and after the provenance change in
commit `1b326e5`; the fast forward path and the single-card capacity, the
module docstring of `src/invdx/engines/fdtdx_perf.py` (measured on a Quadro
RTX 6000); the adjoint memory model, `RETRACTIONS.md`, 2026-08-22.
