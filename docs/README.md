# Docs index

One-line pointer to each doc; start at the top-level [`README.md`](../README.md)
for the project overview.

## Tutorials

- [`phc-bend-walkthrough.md`](phc-bend-walkthrough.md) — hands-on, one-command-per-step
  reproduction of the classic PhC 90° bend benchmark on both engines (in Chinese).
- [`../tutorials/01-jax-port/`](../tutorials/01-jax-port/) — lesson one: porting a
  2D FDTD to JAX, with a deliberately gapped skeleton to fill in yourself.
- [`../tutorials/02-first-adjoint/`](../tutorials/02-first-adjoint/) — lesson two:
  your first adjoint gradient, checked against finite differences.

## Environment & reproduction

- [`env.md`](env.md) — the uv/spack environment split, architecture diagram,
  clean-clone reproduction steps, and a spack primer for newcomers.
- [`dependencies.md`](dependencies.md) — what the toolbox stands on: who
  maintains each package, what the licenses add up to (including how the GPL
  engine stays isolated), and what would break if one disappeared.

## Method notes

- [`optimize.md`](optimize.md) — the inverse-design loop: the
  differentiable Device path, FOM, Richardson gradcheck, checkpoint/resume,
  and Slurm usage.
- [`tolerance.md`](tolerance.md) — design-for-tolerance method notes and
  reporting conventions (sensitivity maps, corner evaluation).

## Export

- `python -m invdx.export.handoff <run-dir>` (or `make handoff RUN=…`) — a
  tool-neutral package: permittivity grid, spectra, design vector, manifest.
- `python -m invdx.export.gds --design <run-dir>` — GDS-II layout with a
  minimum-feature self-check; also writes `<out>.gds.fingerprint.json`, the
  export-time half of the geometry contract.
- `invdx.export.contract` — fingerprints the exported polygons and reads back
  what a converter produced, so a handoff can be checked rather than trusted.

## Honest record

- [`journal.md`](journal.md) — append-only working log; every reported
  number cites the run, commit, or report file it came from.
- [`RETRACTIONS.md`](RETRACTIONS.md) — conclusions this project published
  and later found wrong, corrected in place with a pointer here rather than
  silently edited away.
