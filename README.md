> **English** · [繁體中文](README.zh-TW.md)

# invdx

**A GPU FDTD toolbox for passive photonic devices, built so its numbers can be checked.**

invdx simulates on [fdtdx](https://github.com/ymahlau/fdtdx) (JAX, GPU,
differentiable) and checks that engine against
[Meep](https://github.com/NanoComp/meep), an independent FDTD code reached
through a subprocess bridge. Six verification gates, from unit tests to
fdtdx-vs-Meep agreement, have to pass before results are trusted, and each
run directory keeps its config, command line, environment and hardware. On
that base sit adjoint gradients and an inverse-design loop. Inverse design is
one feature among these, not the whole toolbox.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

## What it does

| | |
|---|---|
| **Simulate on the GPU** | quasi-2D and full-3D scenes on fdtdx; a whole spectrum from one pulsed run; a forward-only fast path about 2× faster than stock fdtdx, with bit-identical fields |
| **Cross-check on a second engine** | Meep 1.34 (CPU, MPI), built from source and called only as a separate program; every convention difference between the two engines written down in one module |
| **Verify the toolchain** | six gates G0–G5 (table below); a missing prerequisite makes a gate fail, never skip |
| **Adjoint gradients and inverse design** | gradients through the whole FDTD run, checked against finite differences; an Adam loop for any differentiable parameterization; density → conic filter → tanh projection with a β schedule; resume after a crash, Slurm requeue |
| **Reproduce a run** | locked environments (uv for JAX/fdtdx, spack for Meep); every run directory records its config, command line, environment and a hardware probe |

## Results so far

| Measured | Result |
|---|---|
| fdtdx vs Meep vs the analytic Airy formula (G5: dielectric slab, normal incidence) | T = 0.7441 / 0.7423 / 0.7398; fdtdx and Meep 0.24% apart |
| adjoint vs finite differences through a full design path, filter → projection → device → mode overlap (G2) | 0.0008%, 0.18% and 0.07% relative error on three voxels (gate: 5%) |
| forward vs reverse excitation (G4) | 0.08 dB apart (gate: 0.5 dB) |
| fast forward path vs stock fdtdx | 640 → 1245 Mcell-steps/s (1.94×) at 8.0M cells; 653 → 1529 (2.34×) at 16.8M cells; fields bit-identical |
| largest forward run on one 24 GB card | 15.6M cells with stock fdtdx (out of memory at 27M); 64M cells with the fast path and memory reclaim |
| adjoint memory | peak ≈ (372.18 + 268.0329·C) bytes per cell, C = gradient checkpoints; the slope is the same on Turing and Ada |

Everything here is simulation; nothing has been compared against a fabricated
device.

### Hardware these results come from

- GPU: NVIDIA Quadro RTX 6000, NVIDIA RTX 6000 Ada Generation
- CPU: AMD Ryzen 9 7950X, dual-socket Intel Xeon Gold

## What fits, and what does not

All three engines solve FDTD on a uniform Cartesian grid, and every adapter
gives each material one real, constant permittivity. That decides the class
of problem.

**Fits:** passive, linear, lossless dielectric structures whose figure of
merit is a ratio of two runs (the device run divided by a reference run).

**Needs new engine work:** dispersive or lossy materials (so metals and
plasmonics), nonlinear optics, gain and active media, anisotropic or
magneto-optic media, permittivity that varies in time, thermal or electrical
co-simulation, non-uniform meshes, band-structure (eigenmode) calculations.

Know these before you start:

- **The Meep cross-check stops at 2D.** The bridge only has 2D measurement
  tasks, so a 3D structure can be simulated and optimized on fdtdx but not
  yet checked against Meep. That is a missing bridge task, not a physics
  limit.
- **Ports need a mode you can write down.** There is no numerical mode solver
  in the chain, so a port mode without a closed form has to be supplied (MPB
  can compute one).
- **One run has to fit on one GPU.** With the memory model above (GB = 10⁹
  bytes) and a 16 GB budget (the largest adjoint run known to fit on a 24 GB
  card peaked at 15.4 GB), C = 10 leaves room for about 5.2M cells: a 3.5 µm
  cube at 20 nm spacing. fdtdx has no true 2D, so a "2D" scene keeps a thin
  periodic third axis a few cells thick, and costs that many times its
  in-plane cell count.
- **Inverse design needs a differentiable figure of merit and continuous
  variables.** The optimizer accepts any parameterization that returns a loss
  and its gradient. A density on a pixel grid works as shipped; a geometric
  parameterization needs its own smooth rasterizer; discrete choices are out.
- **The toy engine is for learning and cross-checks, not design**: it uses a
  first-order Mur boundary rather than a PML.

## Quickstart

```bash
bash scripts/bootstrap.sh   # install and verify layer L1 (uv, JAX, fdtdx)
make test                   # unit tests, CPU only
make gates                  # all six gates: G1-G4 need a GPU, G5 also needs Meep
```

Meep is built from source with spack: [docs/env.md](docs/env.md).

## Verification gates

The gates run in order, each assuming the ones before it passed, and every
run writes a machine-readable `gates_report.json`.

| Gate | What it checks | Needs |
|---|---|---|
| G0 unit | the whole `tests/` suite: overlaps, filters, toy physics, geometry, design-vector round trips, checkpoint and resume | CPU |
| G1 api | the fdtdx API this repo uses, a GPU visible to JAX, and a ping through the Meep bridge when Meep is installed | GPU |
| G2 gradcheck | adjoint gradients against finite differences: the filter chain rule, a toy cell, and the design path of the problem under test (5%) | GPU |
| G3 physics | flux conservation in an empty cell (5%) | GPU |
| G4 reciprocity | forward against reverse excitation of the problem under test (0.5 dB); this is the gate that caught a factor-2 normalization bug here | GPU |
| G5 cross-engine | fdtdx against Meep and the analytic Airy formula on a dielectric slab (10%) | GPU, Meep |

Treat a red gate as a stop-the-line event. In the verification-and-validation
sense all six are *verification*: they ask whether the code solves its
equations. None is *validation* against a measured device.

Gate reports also record which problem module produced their numbers. That
catches accidents, such as a copied module that kept its old name or a gate
that reports numbers without saying where they came from. It is not a
security boundary: a problem module runs in-process and could write the
report itself.

## How it is put together

| Layer | What it holds |
|---|---|
| A: engines | fdtdx (GPU, design) and Meep (CPU, reference), version-pinned and used unmodified. Meep runs as a separate program behind the subprocess bridge and is never imported into the invdx process. When a pinned release blocks the work, the fix is a vendored module with attribution rather than a patched package |
| B: methodology | config and run directories, the gates, design filters and projection, mode overlaps, cross-engine conventions, problem definitions |
| C: toy | a self-written 2D FDTD, independent of both engines, for learning and as a third opinion |

## Hard-won guardrails (encoded in `engines/conventions.py`)

1. Meep's DFT fields, |alpha|² and fluxes omit the physical ½ power factor.
   A self-written overlap has to bridge that explicitly; skipping it costs a
   factor 2 in any power ratio built from them, and the reciprocity gate is
   what catches it.
2. `meep.adjoint` multi-frequency gradients arrive as `(Nx, nf)`; sum over
   axis 1.
3. Simulation resolution must be at least the design-grid density, or adjoint
   gradients come back systematically small (measured: 5–8% low at
   resolution 40, exact at 80).
4. Meep's `decay_by` default (1e-11) is about 3.4× slower than 1e-6 at equal
   accuracy, so the worker always passes it explicitly.
5. The config is the single source of truth; scripts never hardcode numbers.
6. Compare resonant structures across engines by **spectral ridge, never by
   a single wavelength**. The engines discretize the same nominal geometry
   into slightly different effective linewidths, and a resonance moves in
   wavelength faster than the edge error itself, so two engines can disagree
   by tens of dB at one wavelength while their ridge peaks agree.

## Docs

- [docs/README.md](docs/README.md): the full index.
- [docs/new-problem.md](docs/new-problem.md): adding a problem of your own.
- [docs/env.md](docs/env.md): the uv and spack environments, and how to
  rebuild them from a clean clone.
- [tutorials/01-jax-port/](tutorials/01-jax-port/): porting the toy engine to
  JAX, as a lesson.

## Engine licenses

invdx is MIT. fdtdx is MIT; the few fdtdx modules vendored here keep its
attribution. Meep is GPL-2.0-or-later and is never linked, vendored or
imported into the invdx process: it runs as a separate program behind the
subprocess bridge. The Python dependency tree has three copyleft packages,
all used unmodified: `tidy3d` (LGPL-2.1-or-later, pulled in by fdtdx and
running in the same process) and `certifi` and `tqdm` (MPL-2.0).

## Citing

Cite invdx through [`CITATION.cff`](CITATION.cff) (or GitHub's "Cite this
repository" button), and cite the engines for the physics: FDTDX (Mahlau et
al., Journal of Open Source Software, 2026) and Meep (Oskooi et al., Computer
Physics Communications, 2010).
