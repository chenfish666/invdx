# invdx — photonic inverse-design toolbox

A fast GPU FDTD engine cross-validated against an independent reference
engine, behind one config-driven, validation-gated workflow.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

## Table of contents

- [What is invdx](#what-is-invdx)
- [Quickstart](#quickstart)
- [Workflow](#workflow)
- [Validation gates](#validation-gates--all-six-real-zero-skips)
- [Measured performance doctrine](#measured-performance-doctrine-2026-08-turing-class-gpus)
- [Hard-won guardrails](#hard-won-guardrails-encoded-in-enginesconventionspy)
- [Inverse design (M1)](#inverse-design-m1-the-pvgc-coupler)
- [Problems](#problems-srcinvdxproblems)
- [Scripts](#scripts)
- [Honest record](#honest-record)
- [Paper toolkit](#paper-toolkit)
- [Engine licenses](#engine-licenses)
- [Citing](#citing)
- [Extension points](#extension-points-deliberately-not-built-yet)
- [Docs](#docs)

## What is invdx

invdx combines a fast GPU engine with an independent, authoritative
cross-validation engine behind one config-driven, validation-gated
methodology layer. Three layers, each with one job: unmodified pinned
engines ([FDTDX](https://github.com/ymahlau/fdtdx) for design, Meep as the
cross-validation anchor, reached only through a subprocess bridge); a
methodology layer that is this project's own identity (run-directory
provenance, a six-gate validation framework, fabrication-robustness
utilities, explicit cross-engine convention alignment); and a small,
self-written 2D FDTD kept deliberately independent of both, used to learn —
and independently check — the physics. Full architecture and environment
details: [`docs/env.md`](docs/env.md).

## Quickstart

```bash
uv sync --extra gpu --extra dev
make test       # pure-python unit tests (~10 s)
make gates      # all six validation gates, needs GPU (~3 min)
make phc-bend   # lab-lineage PhC benchmark, toy engine, CPU (~2 min)
```

Every script runs through `cli.start_run`: each invocation writes
`runs/<timestamp>-<name>[-tag]/` with `config.json` (including any `--set`
overrides), `cmdline.txt` and `env.txt`, so any figure is reconstructible
months later. Unknown `--set` keys are rejected, not ignored.

## Workflow

```mermaid
flowchart TD
    Config["config.py"] --> Overrides["--set overrides"]
    Overrides --> RunDir["cli.start_run (run dir)"]
    RunDir --> Script{"optimize or measure script"}
    Script --> Gates["gates, in order:<br/>G0 unit, G1 api, G2 gradcheck,<br/>G3 physics, G4 reciprocity,<br/>G5 cross-engine (Meep)"]

    Gates -.->|"any gate fails"| Stop["stop the line"]

    Gates --> Verify["scripts/07 independent re-verify"]
    Verify --> Viz["viz"]
    Verify --> Report["report"]
    Verify --> Export["export.handoff"]

    style Stop fill:#e05555,stroke:#900,color:#fff
```

Long jobs are launched detached (SSH-disconnect safe):
`setsid nohup <cmd> > runs/<log> 2>&1 &`.

## Validation gates — all six real, zero skips

`make check` (G0 only) / `make gates` (all). Ordering is load-bearing — each
gate assumes everything before it. Runner: `gates/runner.py`; machine-readable
`gates_report.json` per run.

| # | gate | what it pins down |
|---|---|---|
| G0 | unit | pure-math invariants (63 tests: overlaps, filters, toy physics, geometries, design-vector round-trips, checkpoint/resume) |
| G1 | api | released-fdtdx API surface, GPU visible, meep bridge ping |
| G2 | gradcheck | fdtdx value_and_grad vs finite differences, on a toy cell **and on the real pvgc design path**; filter chain rule |
| G3 | physics | vacuum flux conservation |
| G4 | reciprocity | forward vs reciprocal CE on the PVGC problem (measured 0.076 dB; the gate class that caught pvgc's 2× bug) |
| G5 | cross-engine | fdtdx vs Meep transmission after convention alignment |

Treat gate failures as stop-the-line events.

## Measured performance doctrine (2026-08, Turing-class GPUs)

The performance work here follows a simple discipline: measure before
believing, microbenchmark before surgery, and record dead ends so nobody
re-walks them. Upstream engines are excellent physics codes; these notes are
about squeezing a specific machine's hardware, and anything generally useful gets
offered upstream.

- Meep MPI rank counts: **fewer beats more** — FDTD is memory-bandwidth
  bound. Measured here: 16-core Zen4 peaks at np=8 (np=16 is
  26-35% slower, np=32/SMT is 4x slower); dual-socket Xeon peaks at np=16
  with NUMA binding (np=64/full SMT is 5x slower). Sweep YOUR machine with
  `scripts/12_cpu_tuning.py` before trusting any default.
- GPU sharing is a measured dead end for fdtdx (CUDA MPS: -58% aggregate;
  plain time-slicing: zero gain — one sim ~80% saturates a card at 0.5M
  cells). Batch small sims with jax.vmap inside one process instead.
- Cross-node domain decomposition over 1GbE is disqualified by arithmetic
  (halo exchange ~64 ms/step vs 5-15 ms/step compute); ship whole
  independent jobs to the other node instead (~1.5 s protocol overhead).

## Hard-won guardrails (encoded in `engines/conventions.py`)

1. Meep DFT fields / |alpha|² / fluxes omit the physical ½ power factor —
   any self-written overlap must bridge conventions explicitly (pvgc lost a
   factor 2 in CE to this; the reciprocity gate caught it).
2. `meep.adjoint` multi-frequency gradients arrive as `(Nx, nf)` — `sum(axis=1)`.
3. Simulation resolution must be ≥ design-grid density, or adjoint gradients
   are systematically small (pvgc: 5–8% at res 40, exact at res 80).
4. Meep's `decay_by` default (1e-11) is ~3.4× slower than 1e-6 at equal
   accuracy — the worker always passes it explicitly.
5. Config is the single source of truth; scripts never hardcode numbers.
6. Cross-engine comparisons of resonant structures compare **spectral
   ridges, never single-wavelength values**: engines discretize the same
   nominal geometry into different effective linewidths (crisp cell fill vs
   subpixel averaging, up to half a cell per edge), and e.g. the PVGC ridge
   moves ~2.4 nm per nm of tooth width — 20 dB apart at a fixed wavelength
   while the ridge peaks agree.

## Inverse design (M1: the PVGC coupler)

A single `fdtdx.Device` over the design window (one voxel per design pixel,
conic filter → tanh projection) plus a jnp twin of the CE chain lets one
backward pass reach all 500 design variables — the differentiable
counterpart to the run-length-encoded, non-differentiable `profile_teeth`
route used for measurement.

```bash
uv run python scripts/15_pvgc_optimize.py --tag m1 --gradcheck   # ~13 h, 1 GPU
uv run python scripts/15_pvgc_optimize.py --resume runs/<dir>    # after a kill
sbatch -p <partition> -t 14:00:00 slurm/pvgc_opt.sbatch          # requeue-safe
```

```mermaid
flowchart TD
    Start{"init or resume<br/>opt_state.npz"}
    Start --> Forward["forward: fdtdx.Device"]
    Forward --> Grad["gradcheck: Richardson two-point h<br/>once, upfront only"]
    Grad --> Backward["backward: checkpointed"]
    Backward --> Write["atomic write:<br/>opt_state.npz + history.csv"]
    Write --> Forward

    Backward -.->|"Slurm requeue / SIGTERM"| Kill["job interrupted"]
    Kill -.->|"--resume"| Start

    Write -.->|"outside the loop"| Verify["scripts/07: independent re-measure<br/>optimizer readout is not final"]
```

Implementation details, the gradcheck story, and checkpoint/resume semantics:
[`docs/m1-optimize.md`](docs/m1-optimize.md).

## Problems (`src/invdx/problems/`)

- **`pvgc`** — O-band perfectly-vertical grating coupler (iSiPP50G rules) on
  the fdtdx engine. Quasi-2D and full-3D CE measurement chains
  (fiber-side and waveguide-side excitation, analytic slab TE0 target,
  directional overlaps, dense CE(λ) spectra from single runs); 2D ridge
  cross-validated three ways against the pvgc/Meep reference (−8.8…−10.1 dB
  consistent); 3D ridge matches quasi-2D at W=10 µm; dual-GPU task
  parallelism for the two independent runs of a 3D measurement (1.97×,
  bit-identical); inverse design in `scripts/15` (see above). Scripts
  03–05, 07, 15.
- **`phc_bend`** — PhC 90° bend from the lab's own paper (square lattice,
  a=1 µm, R=0.225a, ε=10): Γ-X stopband 0.27–0.41 covering the paper's full
  gap 0.29–0.41, in-gap bend transmission T≈0.85–1.1, paper's point-defect
  conclusion reproduced (horizontal/vertical ≫ slant) — on the toy engine,
  cross-checked against Meep. Hands-on tutorial:
  `docs/phc-bend-walkthrough.md`. Script 06.

## Scripts

| script | purpose |
|---|---|
| `00_check.py` | gate runner (`--only`, `--through`) |
| `01_smoke_fdtdx.py` | tiny forward fdtdx sim through config/cli/runio |
| `02_smoke_meep_bridge.py` | meep-env subprocess round-trip |
| `03_pvgc_baseline.py` | PVGC cross-engine acceptance baselines (θ=10 ridge, θ=0 suppression) |
| `04_pvgc_3d.py` | single-GPU 3D PVGC validation |
| `05_pvgc_3d_dual.py` | dual-GPU task-parallel 3D (empty/grating on separate cards) |
| `06_phc_bend.py` | PhC bend benchmark, stage by stage (`--stage eps\|gap\|bend\|meep\|compare\|defect`) |
| `07_pvgc_verify_design.py` | independent-engine verification of a pvgc design run (linewidth + CE spectrum + CD corners) |
| `15_pvgc_optimize.py` | PVGC inverse design: adjoint optimization of the grating profile (`--gradcheck`, `--resume`) |
| `16_tolerance_report.py` | design-for-tolerance report on a finished optimization run: per-voxel sensitivity map + three-corner (eta_e/eta_i/eta_d) robust-design evaluation (`--lams` for a corner CE spectrum) |
| `17_generate_dataset.py` | batch forward-simulation dataset generation (`--kind uniform-grating\|random-rho`, npz shards + manifest, resumable) |

(08–14 are the toy-engine lessons and the performance benchmarks; `make help`
lists the Makefile targets that wrap the common invocations.)

## Honest record

Simulation tooling makes it easy to quietly re-run until a number looks
good, then report only that run — selective reporting is the easiest way a
simulation project can mislead itself and everyone reading it. Two files
exist to make that harder here:
[`docs/journal.md`](docs/journal.md) is an append-only working log where
every reported number cites the run, commit, or report file it came from;
[`docs/RETRACTIONS.md`](docs/RETRACTIONS.md) is where a conclusion this
project published and later found wrong gets written down in place, rather
than silently edited away. Treat both as part of the results, not as
bookkeeping.

## Paper toolkit

- `python -m invdx.viz <run-dir> [--pdf]` — renders every known figure from a
  run's JSON/npy snapshots (CE spectra with CD corners, band-gap and bend
  transmission, optimization traces, permittivity maps). Figures are DERIVED
  artifacts: always re-renderable from the run dir alone. `--pdf` adds
  vector output for LaTeX.
- `python -m invdx.report <run-dir> [...]` — machine-generated Markdown table
  of the paper numbers (peak CE, 3 dB bandwidth, linewidth vs rule, corner
  peaks, S11/reciprocity); several run dirs at once give the
  efficiency/linewidth/robustness comparison table directly.
- `pvgc.bandwidth_3db` — interpolated 3 dB bandwidth with an honest
  "clipped by sampled range = lower bound" note; wired into script 07
  together with optional `--s11` (waveguide-side back-reflection +
  reciprocity check on the final design).
- `python -m invdx.export.handoff <run-dir> [--out <dir>]` — packages a run's
  design and results into a tool-neutral bundle (rasterized permittivity
  grid, design vector, CE spectrum, manifest with units/axes/checksums) for
  cross-checking with any external solver, independent of invdx.

## Engine licenses

invdx itself is MIT. fdtdx is MIT (one vendored subclass, attributed in
`engines/fdtdx_fixes.py`). Meep is GPL-2.0+ and is never linked or vendored:
it runs as a separate program in its own environment via the subprocess
bridge, so invdx carries no GPL obligations. RSoft FullWAVE is commercial
and out-of-repo entirely (we only emit exchange files for it).

## Citing

If this toolbox is useful, cite it via [`CITATION.cff`](CITATION.cff) (or
GitHub's "Cite this repository" button). The engines have their own
citations: FDTDX (Mahlau et al., JOSS 2026) and Meep (Oskooi et al.,
Computer Physics Communications 2010) — cite them directly for the physics
engines themselves; see Engine licenses above for how each is used here.

## Extension points (deliberately not built yet)

- Field-map snapshots (|E|^2 of the coupling region) from fdtdx runs.
- PreFab process-prediction integration (2D patterns only; cloud token).
- Robust three-field optimization for the problem modules; ProcessSpec-driven
  design-rule import.
- More lab-lineage benchmarks (periodically segmented waveguides from prior
  lab theses, and the shift-or-shrink bent waveguide [Liao & Lu, JMOe 2019]
  as an inverse-design comparison target).
- `toy/` milestones: JAX port → PML → adjoint via `jax.grad` → registered as a
  third gradient reference in the gradcheck/cross-engine gates.

## Docs

Full index, grouped by tutorials / environment / method notes / honest
record: [`docs/README.md`](docs/README.md).
