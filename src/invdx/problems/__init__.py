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

A problem does not name itself. Whatever `load` was asked for IS the name --
the registry key, or the last segment of the dotted path -- and `load` stamps
it onto the spec it returns. That is one fact in one place instead of three
handwritten copies (module path, registry key, a `name=` string) with no
derivation between them. A module that declares a name anyway is refused when
it disagrees with the one it was loaded under, rather than silently
overwritten: the name is the key the gate reports are filed under, and the
failure worth preventing is exactly a report filed under someone else's
label. See `_stamp`.

A name is not an identity, though, and three things follow from that. `load`
also stamps the import path it resolved, which the gates write into
`details["problem_module"]`, so a report says which module produced its
numbers and not merely what that module was called -- provenance the report
carries itself, rather than in the `cmdline.txt` next to it, which does not
survive being copied or aggregated. Neither field is one a problem supplies:
`runner.gate_details` stamps both from the spec `load` returned and refuses a
problem-supplied copy; both fields must be exactly `str`, so a value that
answers `str()` and `==` on its own behalf is refused rather than believed
(`_stamp`, `contract.reject_non_str_identity`); and `runner.run_gates`
derives them from what was ASKED for and compares before the report is
written, because a subject that fills in its own identity has not been
identified -- and neither has one whose identity is confirmed by asking it a
second time. And a dotted path whose last segment
collides with a registry key is refused outright unless it IS that key's
module: without that, `anything.grating_coupler` would be stamped
`grating_coupler` on the strength of its filename alone. See
`_reject_registry_impersonation`. Neither of those constrains an out-of-tree
problem with a name of its own -- `spiral`, `mmi`, `tmm_stack` all load.

How far any of that goes: these are checks against mistakes, not against a
module that means to deceive. Loading a problem imports it, and an imported
module runs here, with this package in reach. A determined one can produce a
report indistinguishable from an honest run and none of the above will say
so. What the rules buy is that the ordinary failures -- a forgotten
declaration, a copied name, a file named after someone else's problem, a
dict merged the wrong way round -- stop being silent. `gates/runner.py` says
the same where the rule lives, and the README says it where someone reading
a report will see it.

What a problem must declare, and why the list is as short as it is:
`contract.py`. How to write one: `docs/new-problem.md`.
"""

import importlib
from dataclasses import replace

from .contract import (  # noqa: F401  (re-exported for problem modules)
    GradcheckCase,
    ProblemModule,
    ProblemSpec,
    ReciprocityCase,
    Unsupported,
    reject_non_str_identity,
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
    """Return the `ProblemSpec` declared by problem `name`, named after `name`.

    `name` is a registered name, or a dotted import path for a problem that
    lives outside this package. Either way it is also the problem's NAME, and
    `load` stamps it onto the returned spec, together with the import path it
    resolved -- the module declares neither. See `_stamp` for what happens
    when it declares one anyway, and `_reject_registry_impersonation` for the
    one dotted path that is refused before it is imported.
    """
    path = _REGISTRY.get(name)
    if path is None:
        if "." not in str(name):
            raise KeyError(
                f"unknown problem: {name!r}\n"
                f"  registered: {', '.join(available())}\n"
                f"  or pass an importable dotted module path")
        path = str(name)
        _reject_registry_impersonation(path)
    mod = importlib.import_module(path)
    if not isinstance(mod, ProblemModule):
        raise AttributeError(
            f"{path} is not a problem module: it declares no PROBLEM. "
            f"Add `PROBLEM = ProblemSpec(...)` -- see problems/contract.py")
    # `type(...) is`, not `isinstance`: a subclass of ProblemSpec can override
    # `__getattribute__`, `__eq__` or `__str__`, and every question the loader
    # and the gate runner ask about a spec -- what is your name, does it match
    # the one you were loaded under, what module did you come from -- is asked
    # OF the spec. A subclass answers those questions itself, so accepting one
    # would mean the identity check is performed by the thing being
    # identified. Rebuilding a plain `ProblemSpec` from the subclass's fields
    # was the alternative and is worse: it reads the same attributes through
    # the same overrides, and it would silently discard whatever the author
    # added the subclass for, which is a bug report turned into a shrug.
    # Read ONCE. `mod.PROBLEM` is an attribute lookup on a module object, and
    # a module can replace its own type (`sys.modules[__name__].__class__`),
    # after which `PROBLEM` is a property that may answer differently each
    # time it is asked. Checking one read and stamping another would mean the
    # object that passed the check is not the object that reaches the gates.
    spec = mod.PROBLEM
    if type(spec) is not ProblemSpec:
        raise TypeError(
            f"{path}.PROBLEM is {type(spec).__name__}, expected "
            f"exactly ProblemSpec (a subclass is refused too: the loader and "
            f"the gate runner ask the spec for its own identity, and a "
            f"subclass can answer for it). Declare "
            f"`PROBLEM = ProblemSpec(...)` -- see problems/contract.py")
    # `path`, not `mod.__name__`: the path is what the registry or the caller
    # asked importlib for, while `__name__` is an attribute the imported module
    # can assign to itself -- and a module writing its own provenance is the
    # thing this field exists to replace.
    return _stamp(spec, _requested_name(name), path)


def _reject_registry_impersonation(path):
    """Refuse a dotted path that would take a registered problem's name.

    `_requested_name` names a dotted path by its last segment, which is the
    module's filename. That is the right rule for `yourpkg.problems.spiral`
    and the wrong one for `yourpkg.problems.grating_coupler`: the second
    would be stamped with a registered problem's name on the strength of a
    filename, and the gates key their reports by the name. Declaring
    `name="grating_coupler"` is already refused; renaming the FILE to
    `grating_coupler.py` reached the same place until this check, without the
    module having to state the claim anywhere.

    Refused before the import, because importing runs the module.

    This rejects only the collision itself. An out-of-tree problem whose last
    segment is not a registry key is untouched, and so is the registered
    module's own path -- `load("invdx.problems.grating_coupler")` is the
    registered `grating_coupler`, spelled out.
    """
    last = path.rsplit(".", 1)[-1]
    registered = _REGISTRY.get(last)
    if registered is None or registered == path:
        return
    raise ValueError(
        f"{path!r} would be loaded under the name {last!r}, but {last!r} is a "
        f"registered problem and it lives at {registered!r}.\n"
        f"  A dotted path is named by its last segment, so this module would "
        f"be stamped {last!r} and its numbers filed in gates_report.json "
        f"under that problem's own keys -- details['problem'] = {last!r}, "
        f"'{last}_f0', '{last}_fd_checks', '{last}_sampling' -- describing "
        f"code from somewhere else entirely.\n"
        f"  Fix: if this is your own problem, rename the module's last "
        f"segment to anything outside the registered set -- those names, and "
        f"only those, are refused here, and today they are: "
        f"{', '.join(available())}. Any other spelling loads as before. If "
        f"you meant the registered problem, ask for it by name: {last!r}.")


def _requested_name(name):
    """The problem name implied by what `load` was asked for.

    A registry key is already the name. A dotted path is named by its last
    segment, which is the module name -- `fixture_problems.tmm_stack` is
    `tmm_stack`, and that is what the gate reports call it.
    """
    if name in _REGISTRY:
        return str(name)
    return str(name).rsplit(".", 1)[-1]


def requested_identity(name):
    """The `(name, module)` a `load(name)` would stamp, from the request alone.

    No import, no spec, nothing read off the problem: this is the identity a
    gate report SHOULD carry, computed from what was asked for. That is what
    makes it usable as a second opinion. `runner._verify_problem_identity`
    calls it rather than loading the problem again and reading the answer off
    the same object the gate read it off -- comparing a spec with itself is
    an assertion that `==` is reflexive, not a check that a report is true.

    Raises the same `KeyError` `load` would for a name that is neither
    registered nor a dotted path, so the backstop cannot silently agree with
    a report by failing to derive anything.
    """
    path = _REGISTRY.get(name)
    if path is None:
        if "." not in str(name):
            raise KeyError(
                f"unknown problem: {name!r}\n"
                f"  registered: {', '.join(available())}\n"
                f"  or pass an importable dotted module path")
        path = str(name)
    return _requested_name(name), path


def identity_from_args(args, default=DEFAULT):
    """`requested_identity` for whatever `--problem` asked for. See `from_args`."""
    return requested_identity(getattr(args, "problem", None) or default)


def _stamp(spec, requested, path):
    """Give the spec the name and module it was loaded under, or refuse.

    A spec that stays quiet gets named here, which is the point: the name is
    derived from the module path or the registry key rather than written out
    a third time by hand, so there is no copy left to drift. `path` -- the
    import path that was actually resolved -- is stamped alongside it, and is
    what the gates report as `problem_module`.

    A spec that names itself something else is REFUSED rather than quietly
    corrected. Overwriting would leave the author believing the name they
    wrote is the name in use; and this name is not decoration -- it is the
    key the gate reports are filed under, so a wrong one files real numbers
    under another problem's label, which is precisely the failure being
    prevented.

    Two things this function deliberately does NOT do any more.

    It does not return the spec unchanged when the values already look
    right. That shortcut asked `spec.name == requested`, i.e. it asked the
    value being checked whether it was correct, and a `str` subclass
    answering True walked out of here still holding whatever it actually
    contained. The stamp is now unconditional: what comes back always
    carries `str(requested)` and `str(path)`, built here, so nothing that
    arrived in those two fields survives into a report.

    And it does not trust `__post_init__` to have run. A frozen dataclass can
    be assembled with `object.__new__` and `object.__setattr__`, which skips
    `__init__` entirely, so the type check is repeated here where every load
    goes through.
    """
    reject_non_str_identity(spec)
    declared = str(spec.name).strip()
    if declared and declared != requested:
        raise ValueError(
            f"{path}.PROBLEM declares name={declared!r}, but it was loaded "
            f"as {requested!r}.\n"
            f"  The name is not a label on the module, it is the key the "
            f"gate reports are filed under: G2 Part C writes "
            f"'{declared}_f0' / '{declared}_fd_checks' / "
            f"'{declared}_sampling' and G4 writes details['problem'], so "
            f"this module's numbers would land in gates_report.json under "
            f"{declared!r} -- the same keys and the same shape as anything "
            f"else filed under that name, with only "
            f"details['problem_module'] ({path!r}) left to say where they "
            f"came from -- and that field is stamped from the path this load "
            f"resolved, not offered by the module, so it is the one field a "
            f"report mislabelled THIS way still tells the truth in.\n"
            f"  Fix: delete the `name=` line. `load` names a problem after "
            f"the registry key or the last segment of the dotted path it "
            f"was asked for, so there is nothing to keep in sync.")
    declared_module = str(spec.module).strip()
    if declared_module and declared_module != path:
        raise ValueError(
            f"{path}.PROBLEM declares module={declared_module!r}, but it was "
            f"imported from {path!r}.\n"
            f"  Fix: delete the `module=` line. `load` stamps the path it "
            f"resolved, which is the one thing in the report that says where "
            f"the measured code actually came from; a declared value would "
            f"be a claim about that, not a record of it.")
    return replace(spec, name=str(requested), module=str(path))


def from_args(args, default=DEFAULT):
    """`load` the problem named by `--problem`, or the default.

    Takes the whole argparse namespace rather than a string so a script that
    never added the flag still works -- gates are called with several
    different parsers' namespaces.
    """
    return load(getattr(args, "problem", None) or default)
