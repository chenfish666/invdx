[← back to docs index](README.md)

# Journal (append-only)

Working log of what actually happened, in order. Numbers cite their source
(commit, log file, or report path). Entries are appended, never rewritten;
retractions go to `RETRACTIONS.md` instead of editing history here.

Sources given as `runs/…` or `spack/env/install.log` are local run artifacts,
not committed (see `.gitignore`): they identify which run a number came from,
they are not links a reader can follow. Commit hashes are the citations that
resolve inside this repository.

---

## 2026-08-17 — Environment reproducibility (L1 uv, L2 spack)

**L1: uv takes over the Python/GPU layer** (commit `61e539b`)

- `pyproject.toml` pins `jax[cuda12]==0.11.0` + `fdtdx==0.6.2`; `uv.lock` freezes
  148 packages. Existing conda envs untouched (fallback).
- Acceptance: all gates G0–G5 green (G5 `[ok]`, not skipped); benchmark parity
  vs `runs/benchmark_fast_v6.json` — fast hot 1250.832 vs 1250.799 Mcell-steps/s,
  vanilla 636.748 vs 638.033, `peak_mem_gb` identical at 6.015903,
  `max|dE|=max|dH|=0.0` (bitwise contract). Source: `runs/bench_uv_verify.json`.

**L2: meep built from source via spack** (commit `8e3dffb`)

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
  preallocation). Not a spack issue; gates to be re-run serially on clean GPUs
  when M1 lands. Source: `runs/20260817-021820-gates/gates_report.json`.

## 2026-08-17 — M1: first production inverse-design driver (pvgc)

(commits `c243ac7` + `70d23cc`)

- New differentiable path: `fdtdx.Device` over the real pvgc scene (the
  legacy `profile_teeth` route binarizes inside `pvgc.profile_teeth` and was never
  differentiable). FOM = jnp twin of the TE0 overlap; V3 consistency between
  the differentiable FOM and the legacy `characterize` chain on the same
  binary design: |Δ| = 2.5e-6 dB (threshold 0.05).
- G2 gained Part C (3-voxel FD on the real pvgc Device scene): rel err
  0.0008% / 0.177% / 0.067% vs 5% tolerance, with a signal-floor sampler
  (278/500 voxels above 5% of peak gradient are eligible — FD on
  tanh-saturated voxels is float32 cancellation noise, so tolerance was NOT
  loosened). All six gates verified at HEAD across independent runs
  (`runs/20260817-032448-gates` for G0–G4 serial on a clean GPU;
  G5 re-run post-fix, 99.73 s).
- Two physics findings from implementation: (1) center-rule rasterization of
  the seed grating loses 13 dB (−13.501 → −26.427 dB) via ±3.5% duty jitter
  on the 20 nm grid — the rasterizer now rounds edges the way fdtdx places
  blocks; (2) `VAR= cmd` sets an env var to empty string, which the bridge
  treated as a path — fixed in `70d23cc`.
- Loop smoke (0.15 ps, 3 iters incl. one resume): CE −24.73 → −16.20 dB,
  binarization gap −0.07 dB. Source: `runs/20260817-023418-pvgc-opt-smoke3a/`.
- Iteration timings measured under CPU/GPU contention are recorded but not
  quotable as performance figures (measurement discipline).

## 2026-08-17 — Verification chain exercised end to end

Both optimization rounds were stopped deliberately part-way (θ=10 at
iteration 25/40, θ=0 at 31/40, both still at β=64 with the binarization
schedule unfinished) and their checkpoints finalized into designs. The point
was to exercise the verification chain on real, full-scale designs, not to
produce a good coupler — the quality numbers below are of a half-finished
design and are recorded as such.

Chain result on the θ=10 design: `scripts/07` re-measured it independently on
the finer default grid across 1.26–1.36 µm, giving a ridge peak of −17.05 dB
at 1.300 µm and a 3 dB bandwidth of 37.6 nm; reciprocity mismatch **0.14 dB**
(S11 −3.15 dB), inside the 0.5 dB gate. The binarization gap was 2.81 dB
(θ=0: 3.84 dB), well past the 1.0 dB figure pre-registered in
`tolerance.md` — expected at β=64, and recorded rather than re-thresholded.
`scripts/16` produced the sensitivity map plus the corner table: eroded
−19.84 dB, nominal −17.53 dB, dilated −17.81 dB, i.e. this immature design is
an order more sensitive to over-etch than to under-etch. Yield_90% = 67%
(2/3), labeled in the output itself as an n=3 corner screen and not a
statistical yield.

What the exercise confirms: every stage runs on production-scale inputs, the
thresholds fire when the design does not meet them, and the reports carry
their own caveats (blank bandwidth columns when only single-wavelength
corners were run, the n=3 disclaimer on the yield line). Sources:
`runs/20260817-212907-pvgc-verify/results.json`,
`runs/pvgc-opt-156/tolerance/`.

## 2026-08-17 (night) — Slurm production launch; θ=10 stopped by its own safety gate

- Slurm path validated on the local cluster: `gres` assigns
  `CUDA_VISIBLE_DEVICES` 0/1 correctly to concurrent jobs, and the sbatch
  script derives its run directory from the job ID with no changes. An sbatch
  smoke run and a scancel-mid-iteration requeue drill both passed —
  `history.csv` iterations 0,1,2,3 continuous across the kill/resume
  boundary, no duplicate rows.
- The formal θ=10 round was stopped by the pre-run gradcheck: voxel 213,
  rel err 6.29% vs the 5% tolerance, deterministic under the fixed seed, so
  it was recorded as `gradcheck_failed` rather than silently retried. The
  exploratory θ=0 round passed gradcheck (1.93%) and ran overnight.
- Root cause, from an h-scan on GPU: **FD truncation, not an adjoint defect**
  — the residual scales as h³ (×0.126 per h-halving), Richardson
  extrapolation from the same runs agrees with the adjoint to 0.0106%, the
  float32 noise floor is 200× below the observed residual, and the failing
  voxels are not low-signal. Longer runs (0.8 ps vs 0.15 ps) inflate FOM
  curvature in design space, which is why smoke-scale checks passed. The
  earlier "float32 cancellation" attribution for this failure mode is
  retracted — see `RETRACTIONS.md`. Fix: a two-h Richardson gradcheck plus an
  FD self-consistency indicator (`gradcheck()` in
  `scripts/15_pvgc_optimize.py`, mirrored in `gates/g2_gradcheck.py` Part C);
  tolerances and the signal floor are unchanged.

**L2 stage two: meep 1.34.0 via project-owned spack package repo** (commit `614f57b`)

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
