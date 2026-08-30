"""Concrete design problems. Each module = one problem: a config subclass,
scene builders for the engines, and measurement/FOM definitions with their
convention contracts. Included: `grating_coupler`, a fiber-to-chip grating
coupler, `phc_bend`, a photonic-crystal waveguide bend benchmark.

Scripts and tests still reach a problem by importing its name -- that is the
right wiring for code written FOR one problem, and most of this repo's
scripts are. The registry below is for the other case: code that needs *a*
problem rather than a particular one, which today means the two gates that
used to be hardcoded to `grating_coupler` (G2 Part C, G4).

    from invdx import problems
    spec = problems.load("phc_bend")       # lazy: imports on demand
    spec = problems.from_args(args)        # honours --problem, falls back

`load` imports lazily and one at a time, which is load-bearing rather than
tidy: `grating_coupler` imports jax and fdtdx, while `phc_bend` is
deliberately numpy-pure so `engines/meep_worker.py` can import it INSIDE the
Meep environment. A registry that imported every problem to build itself
would drag jax into that environment and break the bridge.

Registration is this file's dict, not a scan of the package directory, for
the same reason: discovery-by-import is discovery-by-side-effect.

`load` also accepts a dotted module path (anything containing a "."), so a
problem living outside this package -- a test fixture, or a downstream user's
own module -- can be gated without being vendored in here.

What a problem must declare, and why the list is as short as it is:
`contract.py`. How to write one: `docs/new-problem.md`.
"""

import importlib

from .contract import (  # noqa: F401  (re-exported for problem modules)
    GradcheckCase,
    ProblemModule,
    ProblemSpec,
    ReciprocityCase,
    Unsupported,
)

# Name -> import path. Lazy on purpose (see the module docstring).
_REGISTRY = {
    "grating_coupler": "invdx.problems.grating_coupler",
    "phc_bend": "invdx.problems.phc_bend",
}

# What the gates use when nothing asked for a specific problem. It is
# `grating_coupler` because that is what G2 Part C and G4 measured before they
# were parameterized, and a default that changes what `make gates` checks is a
# silent change to the trust ladder.
DEFAULT = "grating_coupler"


def available():
    """Registered problem names, sorted."""
    return tuple(sorted(_REGISTRY))


def load(name):
    """Return the `ProblemSpec` declared by problem `name`.

    `name` is a registered name, or a dotted import path for a problem that
    lives outside this package.
    """
    path = _REGISTRY.get(name)
    if path is None:
        if "." not in str(name):
            raise KeyError(
                f"unknown problem: {name!r}\n"
                f"  registered: {', '.join(available())}\n"
                f"  or pass an importable dotted module path")
        path = name
    mod = importlib.import_module(path)
    if not isinstance(mod, ProblemModule):
        raise AttributeError(
            f"{path} is not a problem module: it declares no PROBLEM. "
            f"Add `PROBLEM = ProblemSpec(...)` -- see problems/contract.py")
    if not isinstance(mod.PROBLEM, ProblemSpec):
        raise TypeError(f"{path}.PROBLEM is {type(mod.PROBLEM).__name__}, "
                        f"expected ProblemSpec")
    return mod.PROBLEM


def from_args(args, default=DEFAULT):
    """`load` the problem named by `--problem`, or the default.

    Takes the whole argparse namespace rather than a string so a script that
    never added the flag still works -- gates are called with several
    different parsers' namespaces.
    """
    return load(getattr(args, "problem", None) or default)
