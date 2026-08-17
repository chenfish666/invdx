# Docs index

One-line pointer to each doc; start at the top-level [`README.md`](../README.md)
for the project overview.

## Tutorials

- [`phc-bend-walkthrough.md`](phc-bend-walkthrough.md) — hands-on, one-command-per-step
  reproduction of the classic PhC 90° bend benchmark on both engines (in Chinese).

## Environment & reproduction

- [`env.md`](env.md) — the uv/spack environment split, architecture diagram,
  clean-clone reproduction steps, and a spack primer for newcomers.

## Method notes

- [`m1-optimize.md`](m1-optimize.md) — the M1 inverse-design loop: the
  differentiable Device path, FOM, Richardson gradcheck, checkpoint/resume,
  and Slurm usage.
- [`tolerance.md`](tolerance.md) — design-for-tolerance method notes and
  reporting conventions (sensitivity maps, corner evaluation).

## Honest record

- [`journal.md`](journal.md) — append-only working log; every reported
  number cites the run, commit, or report file it came from.
- [`RETRACTIONS.md`](RETRACTIONS.md) — conclusions this project published
  and later found wrong, corrected in place with a pointer here rather than
  silently edited away.
