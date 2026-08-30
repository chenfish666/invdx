# Dependencies

What this toolbox stands on, who maintains each piece, what the licenses add
up to, and what would break if any one of them went away.

Two numbers frame the rest of this page: the Python environment resolves to
**148 packages** (`uv.lock`), and the native environment to **122 packages**
(`spack/env/spack.lock`). Almost none of those are chosen directly — the
tables below cover what the code actually imports, plus the transitive
arrivals that carry a license or security consequence worth naming.

## Two layers, two failure modes

The project deliberately runs on two environments that never share a process.

**The Python layer** (`uv.lock`, Python 3.12) holds the differentiable design
path: JAX, the FDTDX engine, the optimizer, and the export tooling. It
installs from wheels in seconds and fails in the way wheel stacks fail —
resolver conflicts, ABI mismatches between JAX and its CUDA plugins, a
transitive package drifting behind upstream.

**The native layer** (`spack/env/spack.lock`, Python 3.13) holds Meep and its
MPI/HDF5/GSL stack — the independent cross-validation engine. It is compiled
from source, takes hours, and fails in the way source builds fail: a compiler
or MPI mismatch, a missing system header, a build that succeeds but produces
a subtly different numerical result.

The two Python interpreters are genuinely different builds (3.12.14 and
3.13.13), which is the honest reason the split exists rather than a stylistic
preference: JAX/CUDA and a Meep build cannot be resolved into one environment.
They communicate only through files. `engines/meep_bridge.py` writes
`job.json` plus `.npy` arrays into a job directory, spawns
`mpirun -np N <spack-view>/bin/python meep_worker.py <jobdir>`, blocks, and
reads back `result.json`. NumPy's `.npy` format is the entire shared ABI
between the two worlds.

## How to read the "maintained by" column

The distinction that matters for "can this be depended on" is not popularity
but who is on the hook when it breaks:

- **Foundation / company** — institutional funding, several paid maintainers,
  deprecation cycles and a security process. Breakage is announced.
- **Academic lab** — research software from a funded group. Usually good code
  with a published paper behind it, but staffed by a small number of people
  whose funding and careers move on. API stability is best-effort.
- **Single maintainer** — one person's project, however excellent. The risk is
  not code quality; it is that there is no second person.

The rule of thumb this project follows: institutional packages may be
depended on freely, academic and single-maintainer packages are pinned
exactly and kept behind an interface that could be replaced.

## Python layer — declared and directly imported

| Package | Locked | Where it is used | Maintained by | License |
|---|---|---|---|---|
| `numpy` | 2.4.6 | Everywhere. The array type, and the cross-environment exchange format for the Meep bridge | Foundation (NumFOCUS) | BSD-3-Clause (with vendored 0BSD / MIT / Zlib / CC0 fragments) |
| `scipy` | 1.18.0 | One import: `scipy.ndimage` dilation/erosion in `fab/measure.py`, for minimum-linewidth measurement. Never on a differentiable path | Foundation (NumFOCUS) | BSD-3-Clause |
| `autograd` | 1.9.1 | The numpy-side autodiff: `fab/filters_np.py`, `modes.py`, and — most importantly — `gates/g2_gradcheck.py`, where it is the independent reference gradient that the JAX path is checked against | Individual volunteers (originated in an academic lab; the founding authors no longer commit) | MIT |
| `gdstk` | 1.0.1 | GDSII layout write and read-back verification: `export/gds.py`, `export/contract.py` | Single maintainer | BSL-1.0 |
| `pyevtk` | 1.7.0 | One function, `imageToVTK`, in `export/vtk.py`, for ParaView output | Single maintainer, small collaboration org | MIT |
| `jax` | 0.11.0 (pinned `==`) | The GPU differentiable core. Imported by 30 files: `optimize.py`, `fab/filters_jax.py`, `toy/fdtd2d_jax.py`, the whole `engines/fdtdx_*` family | Company (Google) | Apache-2.0 |
| `fdtdx` | 0.6.2 (pinned `==`) | The GPU FDTD engine, imported by 19 files. Chosen for `reversible_fdtd` — time-reversible backpropagation, O(1) memory in timesteps | Academic lab (published in JOSS, peer-reviewed) | MIT |
| `optax` | 0.2.8 | Four lines: `optax.adam` and `optax.apply_updates` in `optimize.py` and one toy script | Company (Google DeepMind) | Apache-2.0 |
| `equinox` | 0.13.8 | **Not declared anywhere in `pyproject.toml`.** `engines/fdtdx_checkpoint_buffers.py` imports `equinox.internal` directly for the `buffers=` kwarg. Arrives only as a transitive dependency of fdtdx | Single maintainer | Apache-2.0 |
| `matplotlib` | 3.11.1 | Declared under `[dev]`, but used by main-line entry points: `viz/plots.py` (`make viz`) and the tolerance report (`make tolerance`) | Foundation (NumFOCUS) | PSF-based permissive (non-copyleft) |
| `pytest` | 9.1.1 | Test suite only | Foundation-adjacent, multi-maintainer | MIT |

Only two things are pinned exactly — `jax==0.11.0` and `fdtdx==0.6.2`. The
fdtdx pin is load-bearing rather than cautious: `engines/fdtdx_fixes.py`
carries a vendored subclass that repairs an axis-order bug in the 0.6.2
Gaussian source, and that patch is valid for 0.6.2 only. Everything else
floats in `pyproject.toml` and is held still by `uv.lock`.

Two declaration problems are worth fixing rather than documenting forever.
`equinox` is imported directly but never declared, so the build depends on
fdtdx continuing to pull it in; worse, the import reaches into
`equinox.internal`, a private API with no compatibility promise, which makes
a quiet minor-version break more likely than the package disappearing. And
`matplotlib` sits in `[dev]` while two `make` targets need it, so a
production-only install is missing a main-line dependency.

## Python layer — arrives with something else

| Package | Locked | Why it is present | Maintained by | License |
|---|---|---|---|---|
| `jaxlib`, `jax-cuda12-plugin`, `jax-cuda12-pjrt` | 0.11.0 | JAX's XLA backend and CUDA execution plugins. Never imported directly, but without them JAX is CPU-only. Versions must track `jax` exactly | Company (Google) | Apache-2.0 |
| 13 × `nvidia-*-cu12` | cuBLAS 12.9.2.10, cuDNN 9.24.0.43, NCCL 2.31.2, and 10 others | Pulled in by `jax[cuda12]`; loaded dynamically by jaxlib. No project code touches them | Company (NVIDIA) | **`LicenseRef-NVIDIA-Proprietary`** — not open source |
| `tidy3d` | 2.12.0 | Zero mentions in project code, but a hard dependency of fdtdx, which uses it for mode solving. Importing `fdtdx` loads it into the same process | Company (Flexcompute) | **LGPL-2.1-or-later** |
| `pillow` | 11.3.0 | Via matplotlib, imageio, and moviepy. Only reached when matplotlib writes a PNG | Community org (multi-maintainer) | MIT-CMU |
| `certifi` | 2026.7.22 | Transitive, via the tidy3d chain | Company-backed | **MPL-2.0** |
| `tqdm` | 4.70.0 | Transitive, via the tidy3d chain | Community | **MPL-2.0 AND MIT** |

The tidy3d dependency deserves a plain statement because it is invisible from
the project's own source. It is a commercial cloud FDTD vendor's client SDK,
and it brings its whole client stack into the environment: AWS SDK
(`boto3`/`botocore`), HTTP and auth libraries, an MCP server stack, and OS
keyring bindings. A load test confirms none of the network or auth modules
are imported when `fdtdx` is imported — only the data-layer packages
(`dask`, `xarray`, `pandas`, `shapely`, `h5py`) come along. So there is no
phone-home behaviour. But the packages are installed, they are attack surface,
and an offline HPC project has no use for them.

One security item is open. Scanning all 148 locked packages against OSV
returns exactly one hit: `pillow` 11.3.0, with 36 advisories. All are image
*decoding* vulnerabilities (for example a JPEG2000 tile-decoding buffer
accumulation leading to memory exhaustion, introduced in 8.2.0, fixed in
12.3.0), and this project only ever writes PNGs through matplotlib — it never
parses untrusted images, so real exposure is low.

It is also not fixable from here, which is worth stating precisely because
the obvious remedy looks like it should work and does not. `uv lock
--upgrade-package pillow` runs, reports 148 packages resolved, and leaves
`pillow` at 11.3.0 — no error, no warning. Asking for the fixed version by
name is what surfaces the reason:

    uv lock --upgrade-package "pillow==12.3.0"
    # ... moviepy 2.2.1 depends on pillow<12.0,>=9.2.0
    # ... fdtdx>=0.6.2 depends on moviepy>=2.1.1
    # ... your project's requirements are unsatisfiable

So the ceiling is structural — `fdtdx` → `moviepy` → `pillow<12.0` — and it
closes when `moviepy` relaxes its bound or `fdtdx` stops depending on it, not
when anything in this repository changes. The native layer is not subject to
that chain and already carries `py-pillow` 12.2.0, which is why the two
layers have drifted apart.

## Native layer — the Meep environment

Built from source by Spack. The lockfile pins two levels: the package specs
themselves, and the Spack package repository at tag `v2026.06.0` — without
that second pin, nothing is actually reproducible.

| Component | Version | Role |
|---|---|---|
| `meep` | 1.34.0 | The FDTD cross-validation engine. Built from a package recipe carried in this repo, with `+python +mpi +hdf5 +libctl +harminv +mpb +gsl +openmp` |
| `mpich` | 5.0.1 | MPI, chosen to match the reference conda Meep build and reduce cross-engine variables |
| `hdf5` | 1.14.6 | Meep's field output format |
| `libctl`, `harminv`, `mpb` | 4.5.1, 1.4.2, 1.11.1 | Meep's own supporting libraries — control language, harmonic inversion, mode solver |
| `fftw`, `gsl`, `openblas` | 3.3.11, 2.8, 0.3.33 | Numerics underneath Meep |
| `py-numpy`, `py-scipy`, `py-mpi4py`, `py-matplotlib` | 2.4.6, 1.17.1, 4.1.1, 3.11.0 | The worker-side Python stack. `runio.py` uses `mpi4py` for rank-aware output |

`spack.lock` records no license field for any package, so the native licenses
below are the upstream projects' own declarations rather than something read
off the installed artifacts. Meep is GPL-2.0-or-later; GSL is GPL-3.0-or-later;
FFTW is GPL-2.0-or-later. This is the most license-encumbered part of the
whole system, and it is also the most thoroughly isolated — see below.

Note that the two layers pin different versions of the same libraries on
purpose (`py-scipy` 1.17.1 here versus 1.18.0 in the Python layer). They are
separate environments; agreement between them is a coincidence, not a
constraint.

## What the licenses add up to

The project itself is MIT. Nothing in the dependency tree changes that, but
three things need to be stated rather than assumed.

**GPL is real here, and it is isolated by process boundary.** Meep and its
GPL numerics are never linked into the project and never vendored. They are
reached only by spawning a separate program — a different Python interpreter,
in a different environment, with a disjoint dependency tree — and exchanging
JSON and `.npy` files through a job directory. No GPL code is imported into
any process that also holds project code, and the project distributes none of
it. The subprocess bridge is often described as a practical workaround for an
unsolvable environment conflict; it is also what keeps the GPL obligation from
propagating.

**There are three copyleft packages in the Python tree, not one.** `tidy3d`
is LGPL-2.1-or-later (the package declares "v2 or later"; the bare form
`LGPL-2.1` understates it). `certifi` is MPL-2.0 and `tqdm` is
`MPL-2.0 AND MIT` — MPL is file-level weak copyleft, not a permissive
license. All three are used unmodified and none is redistributed, so
obligations stay with those files and nothing reaches project code. Unlike
Meep, though, tidy3d runs *in the same process*, so the boundary here is the
license's own weakness, not an architectural separation. The top-level
README's "Engine licenses" section currently covers fdtdx and Meep but not
this, and should say so.

**The one genuinely closed-source block is NVIDIA's.** The 13 CUDA runtime
wheels are `LicenseRef-NVIDIA-Proprietary`. This is the only place where a
claim of "fully open-source and reproducible" does not hold, and it is worth
naming precisely because the source-level story is misleading: NCCL's source
is BSD-3-Clause, but the `nvidia-nccl-cu12` wheel that actually gets installed
declares proprietary terms. What is recorded here is what the installed
artifact declares.

One caveat on the security scan: OSV coverage of proprietary binary packages
is inherently incomplete, so "no advisories" for the NVIDIA wheels means "no
records found", not "no vulnerabilities". The vendor's own security bulletins
are the authority there, and were not consulted for this page.

## If one of them disappeared

**Project over.** `numpy` and `jax` are not dependencies so much as the
substrate — numpy is additionally the only shared format between the two
environments. `jaxlib` and the CUDA wheels are welded to JAX by version.

**The expensive one.** `fdtdx` is the costliest dependency in the system.
Time-reversible FDTD backpropagation — O(1) memory in timesteps rather than
O(T) — is the reason a 3D optimization fits on a single 23 GB GPU at all.
The alternatives are all worse in a specific way: Meep's adjoint solver is
CPU-bound and one to two orders of magnitude slower, Tidy3D is commercial and
cloud-hosted, and the in-repo `toy/fdtd2d_jax.py` is a 2D teaching
implementation, nowhere near production. There is one deliberate mitigation:
`export/handoff.py` emits a tool-neutral package (permittivity grid, design
vector, coupling-efficiency spectra, manifest), so design *results* are not
locked to the engine that produced them.

**Cheap to replace.** `scipy` contributes two morphological functions used
only for measurement; hand-writing them against a 3×3 structuring element is
roughly an hour's work and cannot change any physical result. `pyevtk`
contributes one function against a published file format — 30 to 50 lines, or
switch to `meshio`. `optax` contributes Adam and `apply_updates`, about 20
lines of JAX; it stays only because fdtdx installs it anyway.

**Replaceable, at a cost.** `gdstk` would mean rewriting the GDS export and
its read-back check, but GDSII is a standard format so no data is trapped;
`gdsfactory` or the KLayout API could take over. `autograd`'s loss would be
methodological rather than functional: it exists to be a *non-JAX* second
opinion on gradients in gate G2, and replacing it with JAX itself would
collapse the cross-framework independence that makes the check meaningful.
The existing Richardson finite-difference path could cover it at lower
precision.

**Ask a different question.** `equinox` is unlikely to disappear, and if the
`buffers=` optimization were lost the code would fall back to fdtdx's
un-buffered path — more memory, same results. The realistic failure is that
the private `equinox.internal` API shifts under a routine upgrade. That is a
monitoring problem, not a replacement problem, and the first step is to
declare the package with a compatible version range instead of relying on
fdtdx to supply it.

## Known gaps in this inventory

Stated rather than hidden, so the next pass knows where to start:

- The build layer is not covered. `setuptools>=68` is a `[build-system]`
  requirement and appears nowhere in `uv.lock`, because uv does not lock build
  backends.
- Native licenses are upstream declarations, not verified from installed files.
- The operational harness lives outside version control and is not part of the
  distributed project, so its own imports (`pyyaml`, `nvidia-ml-py`, `psutil`)
  are deliberately excluded here. They are guarded by `try/except` with
  explicit install instructions, and two of the three are absent from the
  environment entirely — a local tooling concern, not a project dependency.
- Package counts, licenses, and advisories are point-in-time. They can be
  re-derived from `uv.lock` and `spack/env/spack.lock`, which are the
  authoritative records; this page is a reading of them, not a substitute.
