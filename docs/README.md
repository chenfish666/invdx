> **English** · [繁體中文](README.zh-TW.md)

# Docs index

One-line pointer to each doc; start at the top-level [`README.md`](../README.md)
for the project overview.
**Languages:** English · [繁體中文](../README.zh-TW.md).

Every page listed below has a Traditional Chinese twin next to it
(`X.zh-TW.md`), and each half links to the other at the top of the page.
The Export section lists commands rather than pages.

## Tutorials

- [`../tutorials/01-jax-port/`](../tutorials/01-jax-port/) — porting a
  2D FDTD to JAX, with a deliberately gapped skeleton to fill in yourself.
  **Languages:** [English](../tutorials/01-jax-port/README.md) ·
  [繁體中文](../tutorials/01-jax-port/README.zh-TW.md) — lesson page and its
  `RESULTS` reference output both.

## Environment & reproduction

- [`env.md`](env.md) — the uv/spack environment split, architecture diagram,
  clean-clone reproduction steps, and a primer for newcomers on each half:
  which layer a new dependency belongs in, what `uv.lock` and `spack.lock`
  each pin, the offline/no-network path and what was actually measured about
  it, the drift checks, and the pits both halves have fallen into.
  **Languages:** English · [繁體中文](env.zh-TW.md).
- `bash scripts/bootstrap.sh` — layer L1 (uv, JAX, fdtdx): installs, gates on
  the GPU driver version, and verifies by importing. `bash spack/bootstrap.sh`
  is its L2 (Meep) counterpart. Both are idempotent; `make env-drift` checks
  that the committed lockfiles still match the committed intent.

## How-to

- [`new-problem.md`](new-problem.md) — add a problem of your own: what a
  problem module declares, how to look at the geometry before paying for a
  simulation, and the conventions that give a wrong answer quietly rather
  than raising. The two gates that measure a concrete problem (G2 Part C,
  G4) are inherited by declaring one `ProblemSpec` — or declined in writing,
  with the reason printed by the gate; forgetting to decide is an import
  error rather than a silent loss of coverage.
  **Languages:** English · [繁體中文](new-problem.zh-TW.md).

## Export

- `python -m invdx.export.handoff <run-dir>` (or `make handoff RUN=…`) — a
  tool-neutral package: permittivity grid, spectra, design vector, manifest.
- `python -m invdx.export.gds --design <run-dir>` — GDS-II layout with a
  minimum-feature self-check; also writes `<out>.gds.fingerprint.json`, the
  export-time half of the geometry contract.
- `invdx.export.contract` — fingerprints the exported polygons and reads back
  what a converter produced, so a handoff can be checked rather than trusted.
