> **English** · [繁體中文](new-problem.zh-TW.md)

[← back to docs index](README.md)

# How to add a new problem

You want to simulate a device this repo has never seen. This page is the path
from "the environment works" to "my device is built, measured, anchored to a
known answer, gated, and — if you want it — optimizable".

It is a how-to, not an API reference. Every step ends with a command that
tells you whether the step worked. The worked example is a real problem
module, short enough to be printed in full below, that ends up agreeing with a
closed-form answer to 1% in half a minute of CPU time.

Prerequisite: `make check` passes. If it does not, start at
[`env.md`](env.md) instead.

---

## The contract: what a problem module actually is

There is no abstract base class to inherit and no entry-point autodiscovery:
nothing scans for problems, so a module becomes reachable either by one line in
the registry dict in `src/invdx/problems/__init__.py`, or by handing `--problem`
a dotted module path, which needs no edit here at all. Beyond that, scripts and
tests import your module by name — that is still the wiring for code written
*for* one problem, which most of `scripts/` is.

There is exactly one thing you must declare, and it exists because of the gap
this page used to end on: two of the six gates measure a concrete problem, and
a new problem used to get neither of them, silently, by default. So a problem
module ends with

```python
PROBLEM = ProblemSpec(
    config_cls=<YourName>Config,
    gradcheck_case=...,      # a factory, or Unsupported("why not")
    reciprocity_case=...,    # a factory, or Unsupported("why not")
)
```

Neither gate slot has a default, so "I forgot" is an import error rather than
a quiet loss of coverage. `problems.load("<your_problem>")` reads that declaration
after you add your module to the registry dict in
`src/invdx/problems/__init__.py`; `load` also accepts a dotted module path, so
a problem living outside this repo can be gated without being vendored in.
Details and the exact types: [`src/invdx/problems/contract.py`](../src/invdx/problems/contract.py).

**Your problem does not name itself, and you do not need to know anyone
else's name to write one.** Whatever `load` was asked for *is* the name — the
registry key, or the last segment of the dotted path
(`yourpkg.problems.spiral` is named `spiral`) — and `load` stamps it onto the
spec it hands back. `<your_problem>` on this page is a stand-in for whatever
you pick, never for anybody else's problem: no step below needs the name of a
problem you did not write. Writing it out a third time, next to the module path and
the registry key, would be a copy with no derivation and nothing comparing
it. If you declare a `name=` anyway and it disagrees with the name you loaded
under, `load` raises rather than quietly correcting you; and a dotted path
whose last segment is a *registered* problem's name —
`yourpkg.problems.grating_coupler` — is refused before it is imported,
whatever the module does or does not declare. Both refusals protect the same
thing: the name is the key the gate reports are filed under
(`<your_problem>_f0`, `<your_problem>_fd_checks`,
`<your_problem>_sampling`, `details["problem"]`), so a wrong one
puts your numbers under a shipped problem's label in `gates_report.json`.
Neither rule touches a problem with a name of its own — `spiral`, `mmi`,
`tmm_stack` all load from anywhere.

`load` also stamps the import path it resolved, and the gates write it as
`details["problem_module"]`, so the report says where its numbers came from
rather than only what they are called. The name is assigned by the loader;
the path is the one field that can be checked against a tree, and it stays in
`gates_report.json` after the surrounding run directory (and its
`cmdline.txt`) is gone.

Neither `problem` nor `problem_module` is yours to write. Both are stamped by
the gate from the spec `load` returned, and a `ReciprocityCase.extra` or
`GradcheckCase.info` carrying either name is refused with the colliding key
named — as is any key the gate measures itself (`CE_fwd_dB`, `grad_max`, …)
or the runner writes (`seconds`, `reason`, `exception`). Put your own numbers
under your own names. The rule exists because a silent merge would file your
value under the gate's name: the report still parses, every expected key is
there, and no reader can tell whose number it is. The same goes for the two
fields on the spec itself: they must be exactly `str` and nothing str-like,
because every question the loader asks about them (`str(...).strip()`, `==`)
is asked *of* the value, and a `str` subclass answers on its own behalf. The
same rule reaches the report: every key in `details`, at any depth, and the
two identity values a gate stamps, must be exactly `str` — a subclass that
overrides `__hash__` is not the key it spells as far as any guard is
concerned, and is still written out under that name by `json.dump`.

For the identity keys there is then a second check that does not depend on
the gate remembering any of this — before writing the report, the runner
works out what `--problem` asked for, derives the identity from that request
alone without loading anything, and fails the gate if the result disagrees
(on any status) or carries no identity at all (unless the result is already a
`[FAIL]`, whose own diagnosis must not be buried under a provenance
complaint). That second check is on by default for **every** gate, including
one written next year by someone who never read this page; a gate that
measures no problem is excused only by saying, in its own module, what it
measures instead:

```python
from invdx.gates import NoProblem      # inside the package: from .runner import NoProblem

MEASURES_PROBLEM = NoProblem(
    "G3 checks flux conservation in an EMPTY cell: ... there is no device "
    "in it to attribute `flux_in`, `flux_out` or their ratio to")
```

Nothing you write in a problem module can turn that off, and nothing you
*fail* to write in one can either.

The command that tells you the declaration is well formed — it constructs,
and an empty reason does not:

```bash
uv run python -c "
from invdx.gates import NoProblem
print(NoProblem('G3 checks flux conservation in an EMPTY cell'))
try:
    NoProblem('')
except ValueError as e:
    print('empty reason refused:', str(e).split(':')[0])
"
```

The reason is mandatory, and an empty one raises the same way an empty
`Unsupported(...)` does. The excuse used to be the bare constant
`MEASURES_PROBLEM = False`, which is where this rule comes from: an audit
copied G3's declaration — comment block and all — into a new gate that really
did measure a device, and the gate reported its numbers, stamped no identity
and printed `[ok]`. Three characters that are correct in four modules cannot
be wrong in a fifth. A sentence about an empty cell can, and a reviewer sees
it sitting next to the coupling efficiencies. That is the whole of what this
buys: the copy becomes **visible**, not impossible — the same boundary as the
section just below. The one thing it does enforce is that `False` no longer
parses, so nobody carries the old spelling forward by accident.

`--problem` is the usual source of that truth but not the only one: a gate
that always measures one particular problem, whatever was asked for, declares
`MEASURES_PROBLEM = '<name>'`, and the runner resolves that name the same
request-side way instead of reading it off the loaded problem. So the
declaration cannot name one problem while the report names another. It does
not verify that the gate imported the module it named — a gate that says one
name and loads a different problem to measure is a gate lying about its own
work, which is the other side of the boundary described just below.

### What that buys, and what it does not

Say this plainly, because it changes how you should read a report that is not
your own: **these rules are record-keeping, not a security boundary.** Loading
your problem imports it, and an imported module runs in the same process as
the gates. It can reach into `invdx.gates` directly, replace the runner's
functions, or write `gates_report.json` without simulating anything. A module
that is *trying* to lie can produce a report byte-identical to an honest one —
an audit of this repo built one, a pure-CPU stand-in with a `time.sleep` where
a 91-second GPU run should have been — and no check running inside the same
process will catch it. If a problem module deliberately lies, the report is
not evidence, and nothing here changes that.

What these rules catch is the whole space of things that go wrong *without
anyone meaning them to*, which is the space you are actually in while writing
a new problem: forgetting to declare a gate, copying a module and keeping the
name it came with, naming your file after a shipped problem, letting your
`extra` dict land on top of the gate's keys, writing a gate that reports
numbers with no provenance. Each of those used to produce a green report and
now produces a loud failure naming the key and the fix.

Where that stops, stated so you know which half of a report the machinery
stands behind: the runner can only re-derive the **two identity keys**, so
those are the only ones it checks on its own. Your `extra` and `info` dicts
are kept off a gate's measured numbers by the collision refusal, and that
refusal lives in `runner.merge_problem_dict` — a gate that does not route
your dict through it (the gates shipped here do) would let your value land on
top of its own, and nothing downstream could tell. That is a rule for
whoever writes the next gate, not something this page can promise you. The
provenance fields are what make a report checkable at all: they give a reader
the module path to go and read. They are a pointer to the evidence, not the
evidence.

Everything else is a convention rather than a contract, and the honest reason
is worth stating: across the two shipped problems the intersection of
module-level function names is **empty**. `grating_coupler` and `phc_bend`
share no callable at all. So the table below describes shapes you will end up
writing, not names anything imports.

| you provide | who consumes it | required? |
|---|---|---|
| `PROBLEM = ProblemSpec(...)` — a config class and one answer per problem-specific gate; no name, `load` derives that | `problems.load`, gates G2 Part C and G4 | yes |
| a `@dataclass` config subclassing `config.BaseConfig` | `cli.apply_overrides` (`--set`), `cli.start_run` (writes `config.json`) | yes |
| geometry as plain numpy / plain data | your own scene builders, your tests, `invdx.viz` | yes, in practice |
| one scene builder per engine you use | your measurement functions | one per engine |
| measurement functions returning plain JSON-able dicts | driver scripts, gates, `runio.save_json` | yes |
| `vg_fn(p, beta) -> (loss, grad)` with `loss = -FOM` | `optimize.run_loop`, `ProblemSpec.gradcheck_case` | only for inverse design |

**Minimum viable problem: a config subclass, a geometry function, one
measurement that is a ratio of two runs, and a `PROBLEM` declaration that
answers both gates — even if both answers are `Unsupported`.** A second
engine, a differentiable figure of merit and an optimizer driver are all
opt-in, and each is a separate day of work. Do not start with them.

The smallest complete example is
[`tests/fixture_problems/tmm_stack.py`](../tests/fixture_problems/tmm_stack.py):
no engine, no GPU, and it earns both gates. It lives under
`tests/` rather than `problems/` because it is a contract fixture, not a
device anyone designs.

---

## Step 0 — decide what to copy

| copy from | when | why |
|---|---|---|
| [`src/invdx/problems/phc_bend.py`](../src/invdx/problems/phc_bend.py) | your problem runs on the toy 2D engine and/or Meep, on CPU | numpy-pure — it imports no jax and no fdtdx, which is exactly what lets `engines/meep_worker.py` import it *inside* the Meep environment so both engines consume one geometry definition. Config + geometry + a handful of ratio measurements, nothing else. |
| [`src/invdx/problems/grating_coupler.py`](../src/invdx/problems/grating_coupler.py) | your problem needs the fdtdx GPU engine and/or adjoint gradients | by far the longest module in the repo, and most of that length is one device's measurement chain: do not copy the file. Copy sections. |

`phc_bend` is the better model for the *shape* of a problem module. `grating_coupler` is
the reference for the shape of an fdtdx measurement chain; these are the
sections worth reading before you write your own:

| section of `grating_coupler.py` | what it shows you |
|---|---|
| `build_scene` | assembling an fdtdx object list + placement constraints |
| `_run`, `_phasor` | running the scene and reading a `PhasorDetector` line back |
| `characterize`, `beam_power_and_tilt` | a measurement and its normalization run |
| `_box_bounds`, `check_energy_closure` | how to write a guard that *refuses* rather than returning a plausible number |
| `design_device`, `build_scene_design` | the differentiable `fdtdx.Device` path |
| `te0_target_on_monitor`, `ce_from_arrays`, `make_ce_value_and_grad` | a traced FOM and its value-and-grad factory |

The rest of this page builds a new problem from scratch, in the `phc_bend`
style, and points at the `grating_coupler` equivalent at each step.

---

## Step 1 — the config subclass

Create `src/invdx/problems/<your_problem>.py` and start with the config. Nothing
tweakable may live anywhere else: scripts never hardcode numbers, because
`cli.start_run` snapshots `config.json` and that snapshot is what makes a run
reproducible months later.

```python
from dataclasses import dataclass

import numpy as np

from ..config import BaseConfig


@dataclass
class SlabConfig(BaseConfig):
    # ---- Geometry (um) ----
    n_slab: float = 2.0
    t_slab: float = 0.5

    # ---- Band ----
    lam_min: float = 1.0
    lam_max: float = 2.5
    n_freq: int = 31

    # ---- Numerics (toy engine) ----
    cells_per_um: int = 40
    pad_um: float = 3.0         # vacuum before and after the slab
    height_um: float = 12.0     # transverse extent of the cell
    toy_steps: int = 6000
    toy_courant: float = 0.5

    @property
    def freqs(self):
        return np.linspace(1.0 / self.lam_max, 1.0 / self.lam_min, self.n_freq)
```

Four rules that bite:

1. **Every field needs a default.** All of `BaseConfig`'s fields have
   defaults, so dataclass inheritance rejects any field of yours without one.
2. **Write float defaults as float literals.** `cli._cast_like` casts a
   `--set` value to the type of the *current* default, so a field declared
   `0` is an int forever and `--set w=0.3` dies inside `int("0.3")`.
   `GratingCouplerConfig.w_s11` carries this warning inline for exactly that reason.
3. **Derived values are `@property`, never fields.** Properties do not appear
   in `dataclasses.asdict`, so `config.json` stays the minimal set of inputs a
   run can be rebuilt from. `PhCBendConfig.freqs` and `GratingCouplerConfig.X0` /
   `cell_x` / `cell_z` are the pattern.
4. **`BaseConfig` fields you never use still appear** in `config.json` and in
   the valid-keys error message. That is accepted noise, not a bug to fix:
   `PhCBendConfig` inherits `min_feature`, `eta_e` and `beta_schedule` and
   reads none of them.

**Check it worked.** A config that round-trips and rejects nonsense:

```bash
uv run python -c "
from dataclasses import asdict
from invdx.problems.slab import SlabConfig
cfg = SlabConfig(n_slab=3.0)
print(len(asdict(cfg)), 'fields'); print(cfg.freqs[:3])
"
```

and, once you have a driver script (step 8), the `--set` path:

```
$ uv run python scripts/22_slab.py --set nonsense=1
unknown config key: nonsense
  valid keys: beta_schedule, cells_per_um, design_grid_per_um, ...
```

If a typo is silently ignored instead, you are not going through
`cli.apply_overrides`.

> **Values a caller passes, not a user.** `grating_coupler` sets `cfg._lams_um` before
> building a scene and restores it in a `finally`. Attributes that are not
> dataclass fields are invisible to `config.json` and to `--set` — fine for
> something a caller sets programmatically, wrong for anything a user should
> be able to tune.

---

## Step 2 — geometry first, and look at it before you simulate

**No simulation until you have looked at the permittivity your code actually
built.** Both shipped problems make this a first-class stage:
`python scripts/06_phc_bend.py --stage eps` prints an ASCII map of all three
layouts and saves the arrays.

For a numpy/toy problem the geometry is just an array:

```python
def epsilon_grid(cfg, layout):
    """Rasterized permittivity. layout "empty" is the normalization run.

    The outermost cells stay vacuum on purpose: the toy engine's first-order
    Mur boundary assumes the vacuum wave speed there (toy/fdtd2d.py).
    """
    nx = int(round((2 * cfg.pad_um + cfg.t_slab) * cfg.cells_per_um))
    ny = int(round(cfg.height_um * cfg.cells_per_um))
    eps = np.ones((nx, ny))
    if layout == "empty":
        return eps
    i0 = int(round(cfg.pad_um * cfg.cells_per_um))
    i1 = i0 + int(round(cfg.t_slab * cfg.cells_per_um))
    edge = max(2, cfg.cells_per_um // 4)
    eps[i0:i1, edge:ny - edge] = cfg.n_slab ** 2
    return eps
```

```bash
uv run python -c "
import numpy as np
from invdx.problems import slab
cfg = slab.SlabConfig()
eps = slab.epsilon_grid(cfg, 'slab')
print(eps.shape, np.unique(eps))
print('slab thickness in cells:', int((eps.max(axis=1) > 1).sum()))
"
```

**For an fdtdx problem the same check exists**, and it is worth the two
minutes. `build_scene` returns `(sim_config, object_list, constraints)` with
nothing placed yet; `fdtdx.place_objects` resolves the constraints, and then
the grid can be read back:

```bash
uv run python -c "
import jax, numpy as np, fdtdx
from invdx.problems import grating_coupler
cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.040)
sim_config, objs, cons = grating_coupler.build_scene(
    cfg, teeth=grating_coupler.uniform_grating_teeth(cfg, 0.6, 0.5))
objects, arrays, params, sim_config, _ = fdtdx.place_objects(
    object_list=objs, config=sim_config, constraints=cons,
    key=jax.random.PRNGKey(0))
eps = 1.0 / np.asarray(arrays.inv_permittivities)[0]
print('eps grid:', eps.shape, 'unique:', np.unique(np.round(eps, 3)))
print('time steps:', sim_config.time_steps_total)
print('wg_mon phasor shape:', arrays.detector_states['wg_mon']['phasor'].shape)
"
```

```
eps grid: (500, 4, 243) unique: [ 1.     2.094 12.271]
time steps: 19669
wg_mon phasor shape: (1, 1, 2, 1, 4, 62)
```

This costs seconds and catches, before you pay for a run: a block placed
outside the cell, a feature that snapped to zero cells, a detector plane with
the wrong orientation, an object that silently spans an axis you meant to
keep free, a grid that is 10x the size you intended.

Two constraints to design around:

- **Toy engine:** `eps` must be `1.0` on the outermost cells. The first-order
  Mur boundary assumes the vacuum wave speed there. Material touching the
  edge does not raise — it reflects, and the reflection sets your noise floor.
- **Boundary rims are part of the measurement.** `phc_bend` carries one extra
  ring of lattice sites *into* the rim; without it, light bypasses the crystal
  through the vacuum around it and the measured in-gap suppression is
  bypass-limited rather than crystal-limited (an effect worth tens of dB —
  see `rod_sites`' docstring and step 1 of
  [`phc-bend-walkthrough.md`](phc-bend-walkthrough.md)).

---

## Step 3 — measure a ratio, with an identical normalization run

Every physical number in this repo is a ratio of two runs that differ only in
the thing under test:

- `phc_bend`: bend output / straight-waveguide output; crystal slab / empty
  cell
- `grating_coupler`: mode power at the waveguide monitor / incident beam power from an
  empty-cell run

**"Differ only" is literal.** The `grating_coupler` module docstring states the
condition that makes its ratio valid: all the building blocks are computed
from detector fields at the same wavelength and the same run duration, so
phasor scaling factors cancel exactly. `GratingCouplerConfig.sim_time_s` repeats the
warning at the field itself: keep it *identical* between a measurement run
and its normalization run. Change it in one and not the other and you get a
wrong answer with no error message.

Checklist for your normalization run — same grid spacing and shape, same
source, same step count / `sim_time_s`, same detector planes, same wavelength
list. The structure is the only difference.

The measurement functions themselves should return **plain JSON-able dicts**
(lists, floats, strings — no numpy scalars, no arrays). That is what lets
`runio.save_json` write them straight into the run directory, and it is why
`toy_bend_transmission` ends in `.tolist()` calls.

---

## Step 4 — anchor it to something you did not fit

A measurement that only agrees with itself proves nothing. Before the problem
is worth optimizing, it needs an anchor — a number you did not choose.

| anchor | cost | example in this repo |
|---|---|---|
| closed-form analytic result | seconds | `gates/g5_crossengine.analytic_transmission` (Airy slab); `grating_coupler.slab_te0_neff` (asymmetric-slab dispersion relation) |
| a literature value | minutes | `phc_bend`'s reference band gap `f = 0.29..0.41`, `GAP_REF` in `scripts/06_phc_bend.py` |
| a second engine | minutes to hours | `phc_bend.meep_bend_transmission` through `engines/meep_bridge.py` |

Prefer the cheapest anchor that exists for your device, and keep it in the
repo as a function, not as a number in a commit message.

### The worked example, end to end

Here is a complete new problem — `src/invdx/problems/slab.py`, a file you
create — that does all of steps 1–4. Normal-incidence transmission through a
lossless dielectric slab, on the toy engine, checked against the Airy
formula.

```python
"""Normal-incidence transmission of a lossless dielectric slab in air.

Units: lengths in um, frequencies in 1/um (f = 1/lambda).
Engine: the self-written 2D toy FDTD (CPU only). Every reported number is a
ratio of two runs that differ only by the slab, so the absolute source
amplitude cancels. Anchor: the Airy formula for a lossless slab, which
contains no fitted parameter.
"""

from dataclasses import dataclass

import numpy as np

from ..config import BaseConfig


@dataclass
class SlabConfig(BaseConfig):
    # ---- Geometry (um) ----
    n_slab: float = 2.0
    t_slab: float = 0.5

    # ---- Band ----
    lam_min: float = 1.0
    lam_max: float = 2.5
    n_freq: int = 31

    # ---- Numerics (toy engine) ----
    cells_per_um: int = 40
    pad_um: float = 3.0         # vacuum before and after the slab
    height_um: float = 12.0     # transverse extent of the cell
    toy_steps: int = 6000
    toy_courant: float = 0.5

    @property
    def freqs(self):
        return np.linspace(1.0 / self.lam_max, 1.0 / self.lam_min, self.n_freq)


def epsilon_grid(cfg, layout):
    """Rasterized permittivity. layout "empty" is the normalization run.

    The outermost cells stay vacuum on purpose: the toy engine's first-order
    Mur boundary assumes the vacuum wave speed there (toy/fdtd2d.py).
    """
    nx = int(round((2 * cfg.pad_um + cfg.t_slab) * cfg.cells_per_um))
    ny = int(round(cfg.height_um * cfg.cells_per_um))
    eps = np.ones((nx, ny))
    if layout == "empty":
        return eps
    i0 = int(round(cfg.pad_um * cfg.cells_per_um))
    i1 = i0 + int(round(cfg.t_slab * cfg.cells_per_um))
    edge = max(2, cfg.cells_per_um // 4)
    eps[i0:i1, edge:ny - edge] = cfg.n_slab ** 2
    return eps


def _ports(cfg):
    """Source line and flux line. Both sit in vacuum and are IDENTICAL in the
    measurement and the normalization run — that identity is what makes the
    ratio mean anything."""
    nx = int(round((2 * cfg.pad_um + cfg.t_slab) * cfg.cells_per_um))
    ny = int(round(cfg.height_um * cfg.cells_per_um))
    edge = max(2, cfg.cells_per_um // 4)
    return {"src": {"i": int(round(1.0 * cfg.cells_per_um)),
                    "j0": edge, "j1": ny - edge},
            "out": ("x", nx - int(round(1.0 * cfg.cells_per_um)),
                    edge, ny - edge)}


def _run(cfg, layout):
    from ..toy import fdtd2d

    eps = epsilon_grid(cfg, layout)
    ports = _ports(cfg)
    dx = 1.0 / cfg.cells_per_um
    fcen = 0.5 * (cfg.freqs[0] + cfg.freqs[-1])
    spread = 1.0 / (np.pi * (cfg.freqs[-1] - cfg.freqs[0]) / 2)
    out = fdtd2d.run(
        nx=eps.shape[0], ny=eps.shape[1], dx=dx, steps=cfg.toy_steps,
        source={**ports["src"], "t0": 4 * spread, "spread": spread,
                "fcen": fcen},
        eps=eps, courant=cfg.toy_courant,
        line_probes={"out": ports["out"]})
    dt = cfg.toy_courant * dx
    return fdtd2d.line_flux_spectrum(out["lines"]["out"], cfg.freqs, dt, dx,
                                     sign=-1.0)


def toy_transmission(cfg):
    """T(f) = P(cell with slab) / P(empty cell). Plain JSON-able dict."""
    p_empty = _run(cfg, "empty")
    p_slab = _run(cfg, "slab")
    return {"freqs": cfg.freqs.tolist(), "T": (p_slab / p_empty).tolist()}


def analytic_transmission(cfg):
    """Airy transmission of a lossless slab in air — the parameter-free
    anchor. Same formula as gates/g5_crossengine.analytic_transmission."""
    n, t = cfg.n_slab, cfg.t_slab
    s = np.sin(2 * np.pi * n * t * cfg.freqs) ** 2
    return 1.0 / (1.0 + ((n ** 2 - 1) ** 2 / (4 * n ** 2)) * s)
```

Two things to notice about `fdtd2d.run`, both of which are silent-wrong-answer
traps in their own right: the source dict carries `fcen`, without which the
pulse is baseband and its spectrum dies away from DC; and
`line_flux_spectrum` takes a `sign` because power flow along `+x` is
`Sx = -Ez*Hy` while along `+y` it is `Sy = +Ez*Hx`. Both are documented at
their definitions in [`toy/fdtd2d.py`](../src/invdx/toy/fdtd2d.py).

**Check it worked** — the whole point of the anchor:

```bash
uv run python -c "
import numpy as np
from invdx.problems import slab
cfg = slab.SlabConfig()
T = np.array(slab.toy_transmission(cfg)['T'])
Ta = slab.analytic_transmission(cfg)
print('max rel err vs Airy:', float(np.max(np.abs(T - Ta) / Ta)))
"
```

```
max rel err vs Airy: 0.010295862885515091
```

About 30 s on a CPU. If your number is 0.5 rather than 0.01, the usual causes
are, in order: a normalization run that is not identical to the measurement
run, a flux sign, a source without a carrier frequency, and material touching
the Mur boundary.

---

## Step 5 — the contracts that fail silently

This is the section to read twice. Every entry below produces a
plausible-looking number and no error. They are collected as executable rules
in [`engines/conventions.py`](../src/invdx/engines/conventions.py) precisely
so a new problem does not have to rediscover them.

| trap | what it looks like when you get it wrong | what to use |
|---|---|---|
| Meep omits the physical ½ in DFT fields, `\|alpha\|²` and fluxes | a clean, consistent factor of 2 (3 dB) that no self-consistency check can see | `conventions.MEEP_POWER_OMITS_HALF`, `conventions.meep_to_physical_power` |
| simulation resolution below the design-grid density | forward fields look right; adjoint gradients come back systematically small (measured 5–8% low at res 40) | `conventions.assert_resolution_covers_design_grid(cfg)`; for the fdtdx Device path also `grating_coupler.assert_design_grid_snaps(cfg)` |
| sparse wavelength sampling in a minimax FOM | the sampled points improve every iteration while the spectrum collapses between them; the optimizer's own readout becomes meaningless | `conventions.assert_fom_sampling_covers_band(spacing_nm, feature_nm)` |
| comparing two engines at one fixed wavelength | tens of dB of disagreement between engines whose spectral peaks actually agree — different discretizations give different *effective* geometry, up to half a cell per edge | `conventions.CROSS_ENGINE_COMPARE_SPECTRA`: compare curves, peak positions and peak values |
| `meep.adjoint` multi-frequency gradient arrives as `(Nx, nf)` | a ravel gives a wrong-*length* gradient that still runs | `conventions.collapse_multifreq_gradient(dJ)` (frequency sum) |
| Meep's `decay_by` default (1e-11) | correct, and ~3.4x slower than 1e-6 at equal accuracy | always pass `cfg.dft_decay_tol` explicitly, as `phc_bend.meep_payload` does |
| taking `abs()` of a flux that then enters a sum | a magnitude is right for a *ratio* and wrong for a *conservation check*: face fluxes can only cancel if inflow and outflow keep opposite signs | `grating_coupler.phasor_line_power` (abs, for ratios) vs `grating_coupler.signed_poynting_flux_x` / `signed_poynting_flux_z` (signed, for sums) |
| mixing an instantaneous time-domain detector with phasor quantities | a number with plausible magnitude and no meaning — this is what made an early energy tally sum to 144–151% | one detector family per quantity; `grating_coupler.energy_budget`'s judgment #1 spells it out |
| averaging away an axis that carries design freedom | the run exits 0, the numbers look ordinary, and the optimizer has been handed a smeared objective | `grating_coupler.ce_from_arrays` raises when the monitor's `ny` does not match `cfg.n_y_cells`; copy that shape of check |

**How to make your own trap loud.** The pattern this repo uses is a function
that raises before the expensive work starts, placed in the problem module,
with the *reason* in the exception text:

- `grating_coupler._box_bounds` refuses to report an energy-closure check whose
  "no loss and no source inside the box" premise is violated by geometry, and
  says which knob to change.
- `grating_coupler.assert_design_grid_snaps` refuses a grid spacing that does not divide
  both the layer thickness and the design pixel, and names the spacings that
  do.
- `grating_coupler.energy_budget` calls `_box_bounds` on its first line, explicitly so
  it raises *before* paying for a simulation.

If a wrong configuration of your problem can produce a number rather than an
error, write that guard now. It is much cheaper than finding it in a result.

---

## Step 6 — tests, which G0 picks up for free

Put tests in `tests/test_<your_problem>.py`. Nothing needs registering:
[`gates/g0_unit.py`](../src/invdx/gates/g0_unit.py) runs pytest over the whole
`tests/` directory, so a file dropped there is in the gate from the next run
onwards.

Test the things that need no simulation first — they are the ones that catch
geometry mistakes in milliseconds. [`tests/test_phc_bend.py`](../tests/test_phc_bend.py)
is the model: rod counts per layout, permittivity values and area fraction, a
symmetry the geometry must obey, and equal source-to-monitor path lengths for
the two runs whose ratio is the measurement. Then one fast physics regression
with coarse settings.

```python
"""Pure-math tests for the slab problem, plus one fast physics regression."""

import numpy as np

from invdx.problems import slab

CFG = slab.SlabConfig(cells_per_um=20, height_um=6.0)


def test_epsilon_grid_binary_and_placed():
    eps = slab.epsilon_grid(CFG, "slab")
    assert set(np.unique(eps)) == {1.0, CFG.n_slab ** 2}
    assert np.all(slab.epsilon_grid(CFG, "empty") == 1.0)
    # the Mur boundary assumes vacuum on the outermost cells
    assert eps[0].max() == 1.0 and eps[-1].max() == 1.0
    assert eps[:, 0].max() == 1.0 and eps[:, -1].max() == 1.0


def test_slab_thickness_in_cells():
    eps = slab.epsilon_grid(CFG, "slab")
    assert (eps.max(axis=1) > 1.0).sum() == round(CFG.t_slab * CFG.cells_per_um)


def test_analytic_peaks_at_half_wave():
    # T = 1 exactly when 2*n*t*f is an integer (half-wave slab)
    cfg = slab.SlabConfig(lam_min=1.0, lam_max=1.0, n_freq=1)
    cfg.n_slab, cfg.t_slab = 2.0, 0.25          # 2*n*t*f = 1 at f = 1
    assert abs(slab.analytic_transmission(cfg)[0] - 1.0) < 1e-12


def test_toy_matches_airy():
    # fast physics regression (~20 s): the measured curve must track the
    # parameter-free analytic anchor across the whole band
    cfg = slab.SlabConfig(n_freq=11, toy_steps=4000)
    Ta = slab.analytic_transmission(cfg)
    T = np.array(slab.toy_transmission(cfg)["T"])
    assert np.max(np.abs(T - Ta) / Ta) < 0.05
```

**Check it worked:**

```bash
uv run python -m pytest tests/test_slab.py -q      # your file alone
make check                                          # G0: the whole suite
```

Keep the physics regression measured in seconds, not minutes. `make check` is
run constantly, and a slow gate is a gate people start skipping.

---

## Step 7 — inheriting the gates, and adding one of your own

Be clear about what a new problem inherits. Four of the six shipped gates are
problem-independent; the other two measure the problem named by `--problem`,
and you get them by declaring a case:

| gate | for a new problem |
|---|---|
| G0 `unit` | free, and it includes your tests the moment you add them |
| G1 `api` | free — fdtdx API surface, GPU visible, Meep bridge ping |
| G3 `physics` | free — vacuum flux conservation, an engine-level check |
| G5 `crossengine` | free — fdtdx vs Meep vs analytic on a dielectric slab |
| G2 `gradcheck` Part C | write `gradcheck_case()` (settings, starting design, `vg_fn`/`value_fn`); the gate supplies the eligibility floor, the sampling, the Richardson extrapolation and the 5% tolerance. Parts A and B are generic and always run. |
| G4 `reciprocity` | write `reciprocity_case()` returning two independently normalized dB numbers; the gate supplies the comparison and the 0.5 dB bound |

Both cases are declared in your module's `PROBLEM`, and both are checked by
running the real gate:

```bash
uv run python scripts/00_check.py --only reciprocity --problem <your_problem>
uv run python scripts/00_check.py --only gradcheck   --problem <your_problem>
```

If a gate genuinely has nothing to check on your problem, say so **in code,
with the argument**:

```python
reciprocity_case=Unsupported(
    "the measurement is p_bend / p_straight: two runs sharing one source and "
    "one normalization, so the normalization cancels in the ratio and there "
    "is nothing left to check")
```

(That is `phc_bend`'s real declaration.) The runner then prints `[n/a]` — or
`[part]`, when only the problem-specific half of a gate was declared away —
with your reason on the same line. It is not a pass, it is not a failure, and
it does not look like either. What you cannot do is say nothing: the slot has
no default, so silence is an import error and the runner turns it into a
`[FAIL]`. Write the reason for someone deciding whether to trust your numbers,
and say what would have to change for the gate to become applicable.

A worked example of a problem that earns both gates, with no engine and no
GPU, is [`tests/fixture_problems/tmm_stack.py`](../tests/fixture_problems/tmm_stack.py).

Adding a gate of your own is a file, not a registration. `gates/runner.discover()` imports
every module in `src/invdx/gates/` whose name starts with `g` and sorts them
by `ORDER`; the runner executes them in order and stops at the first failure.
`REQUIRES` is documentation — the runner does not read it.

```python
"""Gate 6 — the slab problem's own physics anchor: toy-engine transmission
vs the analytic Airy curve, as a CURVE (conventions lesson 6), not at one
frequency.
"""

import numpy as np

from .runner import GateResult

NAME = "slab"
ORDER = 6
REQUIRES = ()          # documentation only; the runner does not read it

TOL = 0.05


def run(cfg, args):
    from invdx.problems import slab

    scfg = slab.SlabConfig(n_freq=11, toy_steps=4000)
    T = np.array(slab.toy_transmission(scfg)["T"])
    Ta = slab.analytic_transmission(scfg)
    err = float(np.max(np.abs(T - Ta) / Ta))
    details = {"max_rel_err": err, "T_toy": T.tolist(),
               "T_analytic": Ta.tolist()}
    if err > TOL:
        return GateResult(NAME, "fail", {
            "reason": f"slab transmission deviates from the analytic Airy "
                      f"curve by {err:.1%} > {TOL:.0%}",
            **details})
    return GateResult(NAME, "ok", details)
```

Save that as `src/invdx/gates/g6_slab.py`.

**Check it worked** — discovery first, then execution:

```bash
uv run python -c "
from invdx.gates import runner
print([(m.ORDER, m.NAME) for m in runner.discover()])
"
uv run python scripts/00_check.py --only slab
```

```
[(0, 'unit'), (1, 'api'), (2, 'gradcheck'), (3, 'physics'), (4, 'reciprocity'), (5, 'crossengine'), (6, 'slab')]
[ok]   G6 slab (18.61s)
```

Two conventions worth keeping: a failing gate's `details["reason"]` is what
the runner prints on one line, so write it for someone who has not read the
gate; and the whole `details` dict lands in `gates_report.json`, so put the
numbers there rather than in a print statement.

If your problem needs a GPU or the Meep environment, say so in `REQUIRES`
(for the reader) and let the gate raise when the prerequisite is missing —
the runner turns that into a fail. `gates/__init__.py` explains why a
silent skip would be worse: a skipped gate and a passing gate look identical
in a summary line.

---

## Step 8 — a driver script

Scripts are thin. They parse arguments, build the config, open a run
directory, call the problem module and save the results — all the physics
lives in the problem module, which is what lets tests and gates reuse it.

```python
#!/usr/bin/env python
"""Dielectric-slab transmission, stage by stage.

  python scripts/22_slab.py --stage eps      # look at the geometry
  python scripts/22_slab.py --stage measure  # T(f) vs the analytic anchor
"""

import os

import numpy as np

from invdx import runio
from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import slab


def stage_eps(cfg, d):
    for layout in ("empty", "slab"):
        eps = slab.epsilon_grid(cfg, layout)
        np.save(os.path.join(d, f"eps_{layout}.npy"), eps)
        print(f"[{layout}] grid {eps.shape}, eps values {np.unique(eps)}")


def stage_measure(cfg, d):
    res = slab.toy_transmission(cfg)
    T = np.array(res["T"])
    Ta = slab.analytic_transmission(cfg)
    res["T_analytic"] = Ta.tolist()
    res["max_rel_err"] = float(np.max(np.abs(T - Ta) / Ta))
    runio.save_json(os.path.join(d, "transmission.json"), res)
    print("\n   f       T_toy    T_airy   rel err")
    for f, a, b in zip(cfg.freqs, T, Ta):
        print(f" {f:.4f}  {a:7.4f}  {b:7.4f}  {abs(a - b) / b:7.2%}")
    print(f"\n[anchor] max relative error vs Airy: {res['max_rel_err']:.2%}")

    from invdx.viz import plots
    plots.plot_transmission(
        [(cfg.freqs, T, "toy FDTD"), (cfg.freqs, Ta, "Airy (analytic)")],
        os.path.join(d, "transmission.png"),
        "slab transmission", ylabel="T")


def main():
    p = base_parser(__doc__)
    p.add_argument("--stage", default="eps", choices=("eps", "measure"))
    args = p.parse_args()
    cfg = apply_overrides(slab.SlabConfig(), args)
    d = start_run(cfg, args, "slab")
    {"eps": stage_eps, "measure": stage_measure}[args.stage](cfg, d)
    print(f"[done] {d}")


if __name__ == "__main__":
    main()
```

`start_run` gives you `config.json`, `cmdline.txt`, `env.txt` and
`hardware.json` in a timestamped run directory, for free, on every
invocation.

**Check it worked:**

```bash
uv run python scripts/22_slab.py --stage measure --set n_freq=7 \
    --set toy_steps=3000 --tag doc
```

```
[run] outputs -> runs/<timestamp>-slab-doc

   f       T_toy    T_airy   rel err
 0.4000   0.8305   0.8373    0.80%
 ...
[anchor] max relative error vs Airy: 0.80%
[done] runs/<timestamp>-slab-doc
```

### Figures you get without writing plotting code

`python -m invdx.viz <run-dir>` walks a run directory and renders every
filename it recognizes. Write these names and the figures are free:

| file you write | what `viz.render_run` does with it |
|---|---|
| `eps_*.npy` | permittivity map per file |
| `field_*.npz` with keys `field`, `eps`, and `extent` or `extent_a` (optional `title`) | steady-state field map over the permittivity |
| `results.json` containing `"history"` | optimization trace |
| `results.json` containing `"spectrum"` | efficiency spectrum |
| `design.npz` with key `eps` | the optimized design's permittivity |

```bash
uv run python -m invdx.viz runs/<dir>          # add --pdf for vector output
```

One caveat, since guessing wrong here costs you a mislabelled figure:
`gap.json` and `bend.json` are **not** generic hooks. `render_run` renders
them with photonic-crystal labels and a hardcoded reference band
(`0.29–0.41`), because they belong to `phc_bend`. For any other problem,
choose your own filename and call `plots.plot_transmission` /
`plots.plot_eps` / `plots.plot_field` directly, as the driver above does.

---

## Step 9 — inverse design, if you need it

Only reach this step once steps 1–6 pass. An optimizer amplifies whatever
your measurement chain believes.

[`optimize.py`](../src/invdx/optimize.py) is the genuinely problem-agnostic
piece: it imports neither fdtdx nor any problem module. It gives you Adam on a
`[0, 1]`-boxed latent vector, the `cfg.beta_schedule` annealing walk, atomic
per-iteration checkpointing, resume, and stopping on iteration count,
wall-clock budget or convergence.

Everything problem-specific hides behind one callable:

```python
from invdx import optimize

state = optimize.run_loop(
    vg_fn,            # vg_fn(p, beta) -> (loss, grad),  loss = -FOM
    p0,               # initial latent array
    cfg,              # owns beta_schedule
    n_iters=40,
    lr=0.02,
    run_dir=d,
    resume=False,
    time_budget_h=None,
)
```

The contract, and the parts of it that are easy to get wrong:

- **`loss = -FOM`.** Every problem here maximizes a figure of merit, so
  `history.csv` records `CE = -loss`. A FOM plugged in with the wrong sign
  optimizes away from your goal and reports it as progress.
- `vg_fn` may instead return `((loss, aux), grad)` with `aux` holding the true
  efficiency and a penalty term; the history columns are
  `optimize.HISTORY_HEADER`.
- `n_iters` is the denominator of the beta schedule. Keep it fixed across a
  resume, or the annealing changes underneath you — `beta_for_iter`'s
  docstring and the resume path in `run_loop` both explain why the
  checkpointed `beta` is ground truth on resume.
- The loop writes `opt_state.npz` atomically (`.tmp.npz` then `os.replace`)
  after every iteration, so an interrupted run loses at most one iteration.

**What has no generic helper, and that you will therefore write yourself:**
the differentiable scene and the traced FOM. There is no problem-independent
`Device` factory in this repo. The `grating_coupler` versions are 1-D-specific — for
instance `design_device` installs `ConicFilter1D(radius_um=..., axis=0)`
followed by `fdtdx.TanhProjection` — so copy and adapt rather than import:

1. `grating_coupler.design_device` — one `fdtdx.Device` over the design window, one voxel
   per design pixel, with the filter → projection parameter chain.
2. `grating_coupler.build_scene_design` — the measurement scene with the grating replaced
   by that Device, plus a checkpointed `fdtdx.GradientConfig`.
3. `grating_coupler.te0_target_on_monitor` + `grating_coupler.ce_from_arrays` — a static target
   computed once outside the trace, and a jnp twin of your measurement read
   off the finished run.
4. `grating_coupler.make_ce_value_and_grad` — assembles the above into
   `jax.jit(jax.value_and_grad(loss))`.

**Check it worked, before you spend GPU-hours:** finite-difference your
gradient with [`richardson_fd.richardson_fd_check`](../src/invdx/richardson_fd.py),
which is the shared core both `gates/g2_gradcheck.py` and
`scripts/15_grating_coupler_optimize.py` use. Pass it an `evaluate(sign, h) -> float`
closure that perturbs one design voxel, and it returns `fd`, `rel_err` and
`fd_consistency` from two step sizes. Two rules learned the expensive way and
written up in [`optimize.md`](optimize.md) and
[`RETRACTIONS.md`](RETRACTIONS.md):

- Check only voxels carrying a meaningful fraction of the peak gradient.
  Below that floor the finite difference measures float32 rounding, not your
  adjoint.
- Do not respond to a gradcheck failure by raising the tolerance. A
  single-step-size FD can fail on *truncation* error while the adjoint is
  correct — which is what the two-step Richardson form exists to separate.

Then read [`optimize.md`](optimize.md) in full: the Device-vs-blocks
equivalence check, why rasterizing a starting design is physics rather than
formatting, and why the optimizer's own printed number is a ranking signal
and never a reportable result.

---

## How do I know I am done?

A new problem is finished when each of these prints what it should:

| # | done means | command |
|---|---|---|
| 1 | the config round-trips and rejects typos | `uv run python scripts/<NN>_<your_problem>.py --set nonsense=1` → `unknown config key` |
| 2 | the geometry is what you meant | `--stage eps`, then look at the array or the rendered map |
| 3 | the measurement and its normalization run differ only in the structure | read your own `_run`: same grid, same steps, same ports |
| 4 | the result agrees with an anchor you did not fit | your `--stage measure` prints the anchor comparison |
| 5 | geometry invariants are pinned by tests | `uv run python -m pytest tests/test_<your_problem>.py -q` |
| 6 | the whole suite still passes | `make check` |
| 7 | the anchor is enforced automatically, not by memory | `uv run python scripts/00_check.py --only <your_problem>` → `[ok]` |
| 8 | someone else can rerun your result from the run directory alone | `runs/<dir>/config.json` + `cmdline.txt` reproduce it |
| 9 | *(inverse design only)* the gradient is checked against finite differences before any long run | `richardson_fd_check` at production settings |
| 10 | both problem-specific gates have an answer, and it is the answer you meant | `--only gradcheck --problem <your_problem>` and `--only reciprocity --problem <your_problem>` → `[ok]`, or `[n/a]`/`[part]` printing the reason you wrote |

If 4 and 7 are missing, you have a simulation, not a measurement.

---

## Honest map: what is generic, what is not

Written down so you can plan, rather than discover it halfway through.

| module | for a new problem |
|---|---|
| `config.py`, `cli.py`, `runio.py` | fully generic. No change needed. |
| `optimize.py` | fully generic — imports no engine and no problem module. |
| `engines/conventions.py` | generic rules; add yours here rather than in your problem module if another problem could hit them too. |
| `engines/meep_bridge.py` | generic. Adding a Meep task means adding `task_<name>(payload, jobdir)` to `engines/meep_worker.py` and registering it in that file's `TASKS` dict; the payload must be plain JSON, with arrays passed separately via `run_job(..., arrays={...})`. Round-trip check: `make smoke-meep`. |
| `gates/g0`, `g1`, `g3`, `g5` | problem-independent; you inherit them. |
| `gates/g2` Part C, `gates/g4` | generic checks over a problem-supplied case. You inherit them by declaring `gradcheck_case` / `reciprocity_case` in your `PROBLEM`, or declare `Unsupported(reason)` and get a labelled gap instead of a silent one (step 7). |
| `viz/plots.py` | filename-driven and mostly free (see step 8), except `gap.json` / `bend.json`, which carry `phc_bend`'s labels. |
| `report.py` | reads keys out of `results.json` (`peak`, `bandwidth_3db`, `linewidth`, `spectrum`, `corners`, `s11`). Emit the same keys and the Markdown table works; emit different ones and write your own. |
| `export/gds.py` | generic for a 1-D binary profile: `export_profile_gds(rho, grid_per_um=..., width_um=..., min_feature_um=..., out=...)` plus a minimum-feature self-check. |
| `export/handoff.py` | partly `grating_coupler`-specific. It exports the design vector, spectra and manifest for any run, but skips the permittivity raster for a config it does not recognize and says so in the manifest notes. |
| `datasets.py` | `grating_coupler`-only today. |
| `fab/` | generic: `ConicFilter1D` / `ConicFilter2D`, `min_feature_1d`, `erode_dilate_1d`, `softmin`, `tanh_projection`. |

---

## Where to go next

- [`phc-bend-walkthrough.md`](phc-bend-walkthrough.md) — the same material as
  a physics tutorial: reproducing a literature benchmark step by step on two
  engines. Read it if step 4's "anchor it" is the part you are least sure of.
- [`optimize.md`](optimize.md) — everything behind step 9.
- [`tolerance.md`](tolerance.md) — once a design exists, how it is evaluated
  against fabrication error.
- [`env.md`](env.md) — the environment split, if anything in this page failed
  to import.
