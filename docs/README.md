> **English** · [繁體中文](README.zh-TW.md)

# Docs index

One-line pointer to each doc; start at the top-level [`README.md`](../README.md)
for the project overview.
**Languages:** English · [繁體中文](../README.zh-TW.md).

Every page listed below ends with the languages it is available in. A bilingual
page is a pair — English `X.md` next to Traditional Chinese `X.zh-TW.md` — and
each half links to the other at the top of the page. (The Export section lists
commands rather than pages, so it carries no language tag.)

`make bilingual` checks the pairs mechanically — matching code blocks, links,
heading levels and cross-links — and then enforces the rulings in
[`glossary.zh-TW.md`](glossary.zh-TW.md): every filename and section name it
cites must resolve, no rendering it bans may survive anywhere in the Chinese
tree, and a Chinese form it binds to an English term may not appear where that
term is missing. Each check prints what it looked at, so a shrinking
denominator is visible. It does not check prose; only a cold read does that.

A rule this page and the top-level `README.md` are held to for the same
reason — stated as the rule it is, not as a description of the whole tree,
because one page named below does not yet meet it: **a precise count does not
go into prose that nobody regenerates.** How many unit tests there are, how
many bilingual pairs, how many lines a module runs to — each of those was
written out by hand once and was wrong within weeks, and a stale number reads
exactly like a fresh one. So a count on these two pages either comes from the
command that computes it (`--problem`'s help line lists the registered
problems; `make bilingual` prints the pairs it found and fails when a floor
drops; `make runs` lists the run directories) or it is replaced by the
qualitative claim it was standing in for. The counts that do survive in prose
are the ones a reader can falsify without leaving the page: "six gates" stays
because all six are named, one per row, in the table under the sentence and
again in the workflow diagram, so a seventh gate makes the word visibly wrong.
"178 tests" did not stay, because nothing on the page could ever contradict it.

The known exception is [`dependencies.md`](dependencies.md), which still
carries three hand-written counts — the Python environment's package total,
the native environment's, and the number of `nvidia-*` wheels. Nothing
regenerates them: each is a `grep -c` away from `uv.lock` or
`spack/env/spack.lock`, and until that is wired in they are only as fresh as
the last person who re-derived them. Bringing that page under the rule is a
known to-do, not an oversight this paragraph is unaware of.

## Terminology

- [`glossary.md`](glossary.md) — the terms the Chinese docs bind, and why:
  one concept that acquired two Chinese names, collisions where the obvious
  Chinese word already means something else, terms deliberately left in
  English, and the symbols that must be defined at first use. Grown from
  repeated cold reads, not from foresight. It holds rulings only — claims
  about what any file currently says are recomputed by `make bilingual`
  instead, because this page causes the revisions that would falsify them.
  **Languages:** English · [繁體中文](glossary.zh-TW.md).

## Tutorials

- [`phc-bend-walkthrough.md`](phc-bend-walkthrough.md) — hands-on, one-command-per-step
  reproduction of the classic PhC 90° bend benchmark on both engines.
  **Languages:** English · [繁體中文](phc-bend-walkthrough.zh-TW.md).
- [`../tutorials/01-jax-port/`](../tutorials/01-jax-port/) — lesson one: porting a
  2D FDTD to JAX, with a deliberately gapped skeleton to fill in yourself.
  **Languages:** [English](../tutorials/01-jax-port/README.md) ·
  [繁體中文](../tutorials/01-jax-port/README.zh-TW.md) — lesson page and its
  `RESULTS` reference output both.
- [`../tutorials/02-first-adjoint/`](../tutorials/02-first-adjoint/) — lesson two:
  your first adjoint gradient, checked against finite differences.
  **Languages:** [English](../tutorials/02-first-adjoint/README.md) ·
  [繁體中文](../tutorials/02-first-adjoint/README.zh-TW.md) — lesson page and its
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
- [`dependencies.md`](dependencies.md) — what the toolbox stands on: who
  maintains each package, what the licenses add up to (including how the GPL
  engine stays isolated), and what would break if one disappeared.
  **Languages:** English · [繁體中文](dependencies.zh-TW.md).

## How-to

- [`new-problem.md`](new-problem.md) — add your own device to the toolbox:
  what a problem module has to provide, which file to copy from, how to look
  at the geometry before paying for a simulation, and the convention
  contracts that give a wrong answer quietly rather than raising. The two
  gates that measure a concrete device (G2 Part C, G4) are inherited by
  declaring one `ProblemSpec` — or declined in writing, with the reason
  printed by the gate; forgetting to decide is an import error rather than a
  silent loss of coverage.
  **Languages:** English · [繁體中文](new-problem.zh-TW.md).

## Method notes

- [`optimize.md`](optimize.md) — the inverse-design loop: the
  differentiable Device path, FOM, Richardson gradcheck, checkpoint/resume,
  and Slurm usage.
  **Languages:** English · [繁體中文](optimize.zh-TW.md).
- [`tolerance.md`](tolerance.md) — design-for-tolerance method notes and
  reporting conventions (sensitivity maps, corner evaluation).
  **Languages:** English · [繁體中文](tolerance.zh-TW.md).

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
  **Languages:** English · [繁體中文](journal.zh-TW.md).
- [`RETRACTIONS.md`](RETRACTIONS.md) — conclusions this project published
  and later found wrong, corrected in place with a pointer here rather than
  silently edited away.
  **Languages:** English · [繁體中文](RETRACTIONS.zh-TW.md).
