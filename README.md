# invdx — photonic inverse-design toolbox

Personal research toolbox combining a fast GPU engine with an authoritative
cross-validation engine behind one config-driven, validation-gated workflow.

## Three-layer architecture

- **Layer A — engines (unmodified, pinned, citable).**
  [FDTDX](https://github.com/ymahlau/fdtdx) (JAX GPU FDTD, Mahlau et al.,
  JOSS 2026, `fdtdx==0.6.2`) for design/production; [Meep](https://meep.readthedocs.io)
  (Oskooi et al. 2010, conda env `meep`) as the cross-validation anchor,
  reached only via a subprocess bridge (`invdx.engines.meep_bridge`).
  Upstream code is never modified. **Vendored exceptions (recorded per the
  layer's own rule):** `engines/fdtdx_fixes.py` — a `GaussianBeamSource`
  subclass overriding one method of the released `GaussianPlaneSource`,
  whose profile builder NaNs on rectangular source planes (documented in
  the module docstring).
- **Layer B — methodology (this package's identity).**
  Config + `--set` override + run-directory snapshots (`config.py`, `cli.py`,
  `runio.py`); the validation-gate framework (`gates/`); fabrication-robustness
  utilities (`fab/`: conic filter, tanh projection, linewidth measurement,
  CD-corner erosion/dilation); mode-overlap with an explicit power-convention
  contract (`modes.py`); cross-engine convention alignment (`engines/conventions.py`);
  GDS export (`export/gds.py`). Ported/adapted from the predecessor PVGC study (to be published).
- **Layer C — learning core (`toy/`).**
  A minimal, self-written 2D FDTD kept deliberately independent of fdtdx:
  a vehicle for learning Yee/PML/adjoint internals, and a third independent
  implementation used as a physics cross-reference. Current state
  (M-toy-1b): dielectrics, carrier-modulated pulses, line probes + spectral
  flux — real spectroscopy, drives the `phc_bend` benchmark. Roadmap in
  `toy/__init__.py`.

## Environment

```bash
mamba create -y -n invdx python=3.12 pip
mamba run -n invdx pip install "jax[cuda12]" "fdtdx==0.6.2" \
    optax autograd scipy matplotlib gdstk pytest
mamba run -n invdx pip install -e .
```

Point the Meep bridge at your Meep env with `INVDX_MEEP_ENV`
(default: `/root/miniforge3/envs/meep`).

The Meep side lives in the pre-existing conda env `meep` (MPI pymeep) and is
never imported into this env; `engines/meep_bridge.py` spawns
`mpirun -np N <meep-env-python> engines/meep_worker.py <jobdir>` and exchanges
`.npy`/`.json` files. The worker imports only numpy-pure invdx modules.

Note the API trap: the PyPI `fdtdx==0.6.2` and a dev checkout that also
calls itself 0.6.2 differ (the release configures via
`SimulationConfig(resolution=<meters>)`; dev-only names like `UniformGrid`
don't exist here). The G1 gate pins the released API surface.

## Quickstart

```bash
make test       # pure-python unit tests (~10 s)
make gates      # all six validation gates, needs GPU (~3 min)
make phc-bend   # lab-lineage PhC benchmark, toy engine, CPU (~2 min)
```

Every script goes through `cli.start_run`: each invocation writes
`runs/<timestamp>-<name>[-tag]/` with `config.json` (including your `--set`
overrides), `cmdline.txt` and `env.txt` — any figure is reconstructible
months later. Unknown `--set` keys are rejected, not ignored.

Long jobs are launched detached (SSH-disconnect safe):
`setsid nohup <cmd> > runs/<log> 2>&1 &`.

## Validation gates — all six real, zero skips

`make check` (G0 only) / `make gates` (all). Ordering is load-bearing — each
gate assumes everything before it. Runner: `gates/runner.py`; machine-readable
`gates_report.json` per run.

| # | gate | what it pins down |
|---|---|---|
| G0 | unit | pure-math invariants (35 tests: overlaps, filters, toy physics, geometries) |
| G1 | api | released-fdtdx API surface, GPU visible, meep bridge ping |
| G2 | gradcheck | fdtdx value_and_grad vs finite differences; filter chain rule |
| G3 | physics | vacuum flux conservation |
| G4 | reciprocity | forward vs reciprocal CE on the PVGC problem (measured 0.076 dB; the gate class that caught pvgc's 2× bug) |
| G5 | cross-engine | fdtdx vs Meep transmission after convention alignment |

Treat gate failures as stop-the-line events.

## Measured performance doctrine (2026-08, this fleet)

The performance work here follows the discipline the author picked up
competing in HPC competitions: measure before believing, microbenchmark
before surgery, and record dead ends so nobody re-walks them. Upstream
engines are excellent physics codes; these notes are about squeezing THIS
fleet's hardware, and anything generally useful gets offered upstream.

- Meep MPI rank counts: **fewer beats more** — FDTD is memory-bandwidth
  bound. Measured on the dev fleet: 16-core Zen4 peaks at np=8 (np=16 is
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

## Problems (`src/invdx/problems/`)

- **`pvgc`** — O-band perfectly-vertical grating coupler (iSiPP50G rules) on
  the fdtdx engine. Quasi-2D and full-3D CE measurement chains
  (fiber-side and waveguide-side excitation, analytic slab TE0 target,
  directional overlaps, dense CE(λ) spectra from single runs); 2D ridge
  cross-validated three ways against the pvgc/Meep reference (−8.8…−10.1 dB
  consistent); 3D ridge matches quasi-2D at W=10 µm; dual-GPU task
  parallelism for the two independent runs of a 3D measurement (1.97×,
  bit-identical). Scripts 03–05, 07.
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

## Docs

- `docs/phc-bend-walkthrough.md` — hands-on reproduction of the lab's PhC
  paper, one command per step (in Chinese).

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

## Engine licenses

invdx itself is MIT. fdtdx is MIT (one vendored subclass, attributed in
`engines/fdtdx_fixes.py`). Meep is GPL-2.0+ and is never linked or vendored:
it runs as a separate program in its own environment via the subprocess
bridge, so invdx carries no GPL obligations. RSoft FullWAVE is commercial
and out-of-repo entirely (we only emit exchange files for it).

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
