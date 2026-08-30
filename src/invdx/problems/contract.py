"""What a problem module must declare, written as code instead of prose.

Read this before `docs/new-problem.md`'s contract table: the table describes
five things a problem provides, and only the first of them is a contract in
the sense a machine can check. Measured across the two shipped problems, the
intersection of their module-level names is EMPTY -- `grating_coupler` and
`phc_bend` share no function name at all, only the `BaseConfig` ancestor of
their configs. So a Protocol demanding `build_scene` or `characterize` would
be a Protocol that neither shipped problem implements, i.e. a lie with type
annotations on it.

What IS shared, and therefore what this module encodes, is narrow:

    PROBLEM: ProblemSpec     -- a config class and one explicit answer per
                                problem-specific gate (the problem's name and
                                the module it came from are not declared here;
                                `problems.load` derives both from the name it
                                was asked for and the module it resolved to)

`phc_bend` is the yardstick for "necessary": anything it does not have is not
in the required contract. That is why there is no geometry function, no
measurement function and no scene builder here, even though every problem has
some of each -- they have no shared shape to name.

The gate slots are the point of the file. Before this existed, a new problem
inherited G0/G1/G3/G5 for free and silently got NO gradient and NO reciprocity
coverage -- the two checks that catch normalization and adjoint errors, which
are exactly the errors every other check lets through. `ProblemSpec` has no
default for either slot, so a problem that forgets to decide fails at import
with a TypeError naming the missing argument. There are three outcomes and the
gate runner prints them differently:

    a case factory        -> the gate runs; [ok] or [FAIL] as usual
    Unsupported(reason)   -> the gate reports [n/a] (or [part], for a gate
                             whose generic half still ran) and prints the
                             reason on the same line
    nothing               -> import error, which the runner turns into [FAIL]

"Declared inapplicable" and "broken" therefore never look alike, and neither
of them looks like a pass. `gates/__init__.py` explains why that separation
is worth a status of its own rather than a silent skip.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Protocol, Union, runtime_checkable

from ..config import BaseConfig


@dataclass(frozen=True)
class Unsupported:
    """A problem's declaration that a gate does not apply to it.

    The reason is mandatory and is printed on the gate's console line, so it
    is written for someone who has not read the problem module. "Not
    applicable" without an argument is indistinguishable from "nobody got
    around to it", and the second one should not pass review.
    """

    reason: str

    def __post_init__(self):
        if not str(self.reason).strip():
            raise ValueError(
                "Unsupported(reason=...) needs a reason: it is printed on the "
                "gate's console line and is the only thing telling a reader "
                "that the missing coverage was a decision")


# --------------------------------------------------------------------------
# Gate cases -- what a problem hands a gate, and nothing more
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GradcheckCase:
    """Everything G2 Part C needs to finite-difference a design path.

    The problem owns the physics settings (grid, run length, starting design,
    which beta) and the dtype/array-library conventions of its own callables;
    the gate owns the check (eligibility floor, voxel sampling, Richardson
    extrapolation, tolerance). Keeping the split there is what lets a
    non-jax problem be gradient-checked by the same gate.

    vg_fn(p, beta)    -> (loss, dloss/dp), loss = -FOM, same shape as `p`
    value_fn(p, beta) -> float loss, for the finite differences
                         Both take a plain numpy `p`; casting to the working
                         dtype is the problem's job, in the problem's order,
                         because float64-then-cast and float32-throughout do
                         not round identically.
    base              -> numpy array, the design to differentiate AT. Choose
                         it so the gradient has signal: a mid-grey design
                         often has none, and a hard 0/1 one sits on the clip
                         boundary where a central difference degenerates into
                         a one-sided one.
    beta              -> passed back to the callables untouched, whatever its
                         type
    seed              -> selects which voxels get sampled, and nothing else
    info              -> extra numbers for gates_report.json, under names of
                         your own. A name the gate or the runner writes is
                         refused, not merged -- `gates/runner.py`,
                         `merge_problem_dict` and `RESERVED_DETAIL_KEYS`.
    """

    vg_fn: Callable[[Any, Any], Any]
    value_fn: Callable[[Any, Any], float]
    base: Any
    beta: Any
    seed: int = 0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReciprocityCase:
    """The two numbers G4 compares, each measured by the problem.

    The gate deliberately cannot compute these: the whole value of G4 is that
    the two directions are normalized INDEPENDENTLY, so a shared helper doing
    both would be the very bug it looks for. The problem runs both
    excitations, normalizes each by its own reference, and reports both in dB.

    A problem for which the two directions cancel their normalization by
    construction (a ratio of two runs sharing one source) has nothing for this
    gate to check and should declare `Unsupported` rather than report a
    quantity that passes trivially.

    `extra` rides into `gates_report.json` under names of your own; a name
    the gate or the runner writes -- `problem`, `problem_module`, the two
    efficiencies, the mismatch, `seconds` -- is refused rather than merged.
    See `gates/runner.py`, `merge_problem_dict` and `RESERVED_DETAIL_KEYS`.
    """

    fwd_dB: float
    rev_dB: float
    extra: Dict[str, Any] = field(default_factory=dict)


class GradcheckCaseFactory(Protocol):
    def __call__(self) -> GradcheckCase: ...


class ReciprocityCaseFactory(Protocol):
    def __call__(self) -> ReciprocityCase: ...


GradcheckSlot = Union[GradcheckCaseFactory, Unsupported]
ReciprocitySlot = Union[ReciprocityCaseFactory, Unsupported]


# --------------------------------------------------------------------------
# The declaration itself
# --------------------------------------------------------------------------


IDENTITY_FIELDS = ("name", "module")


def reject_non_str_identity(spec):
    """Refuse a spec whose `name` or `module` is not exactly a `str`.

    Called from `ProblemSpec.__post_init__`, and again from
    `problems._stamp` -- deliberately twice, because they cover different
    holes. `__post_init__` covers every spec that was constructed normally;
    `_stamp` covers one that was not, since `object.__new__` plus
    `object.__setattr__` builds a frozen dataclass without running `__init__`
    at all.

    Exact type, not `isinstance`. A `str` subclass IS a str to `isinstance`,
    and is exactly what defeats the checks built on top of these two fields:
    `str(x).strip()` (used to ask "did this module name itself?") and `==`
    (used to ask "does that name match the one we loaded under?") both
    dispatch to the subclass. One object answering "" to the first and True
    to the second passes both questions while carrying a third value
    entirely, which is then what the report says. Nothing downstream can
    detect that, because everything downstream asks the same object.

    The message names the type, because the value will print as an ordinary
    string and the reader would otherwise be looking at a name that appears
    correct.
    """
    for field_name in IDENTITY_FIELDS:
        value = getattr(spec, field_name, "")
        if type(value) is not str:
            raise TypeError(
                f"ProblemSpec.{field_name} must be exactly `str`, got "
                f"{type(value).__name__} (a str subclass counts as the wrong "
                f"type here, even though `isinstance` says otherwise).\n"
                f"  These two fields identify the module a gate report "
                f"describes, and every check on them -- `.strip()`, `==`, "
                f"formatting -- runs on the value itself, so a subclass "
                f"overriding `__str__`/`__eq__` answers those checks instead "
                f"of the value doing so.\n"
                f"  Fix: pass a plain str, or (better) pass nothing at all -- "
                f"`problems.load` stamps both fields from what it was asked "
                f"for.")


@dataclass(frozen=True)
class ProblemSpec:
    """The module-level `PROBLEM = ProblemSpec(...)` every problem declares.

    Neither gate slot has a default. That is the one piece of enforcement in
    this file, and it is aimed at the failure the old arrangement produced by
    default: a new problem quietly shipping without gradient or reciprocity
    coverage, looking exactly as green as one that has both.

    `name` is the exception: it HAS a default, and the default is the right
    answer. A problem module used to hand-write its own name, which made the
    name the third handwritten copy of one fact -- module path, registry key,
    and this string -- with no derivation between them and nothing checking
    them against each other. `problems.load(name)` already knows which
    problem it was asked for, so it stamps that name on the spec it returns.
    Leave it out. `problems.load` rejects a spec that declares a name
    disagreeing with the one it was loaded under, because that name becomes
    the key the gate reports are filed under.

    A spec built by hand and never passed through `load` -- a test's
    deliberately broken variant, say -- may still name itself, and nothing
    stamps or checks that name: there was no request for it to disagree
    with. Its TYPE is checked even so, at construction; see below.

    `module` is the other half of the same idea and is likewise never written
    by a problem module: `load` stamps the import path it handed to
    `importlib` -- the registry entry, or the dotted path the caller asked
    for -- and the gates copy it into `details["problem_module"]`.
    Deliberately NOT `mod.__name__`: that is an attribute the imported module
    can assign to itself, and a module writing down its own provenance is the
    thing this field replaces -- the path `load` handed `importlib` came from
    the request, so it is the one identity in the report with a source
    outside the module it describes. (Do not "simplify" it back; `load` says
    the same in a comment at the line that would change.)
    A name alone does not say where the measured code lives -- two modules can
    end up under one name, e.g. any `grating_coupler.py` in any package -- so
    a report carrying only the name cannot be checked against the tree it
    claims to describe. It is a field on the spec rather than a lookup on the
    side because everything downstream already receives the spec and nothing
    else: a side table would have to be threaded through `from_args`, both
    gates and every test that builds a spec by hand, and could go missing
    exactly where the report is written. The cost is that a hand-built spec
    carries `module=""` -- unstamped, therefore honestly blank rather than
    wrong.

    Both identity fields are enforced to be EXACTLY `str`, not merely
    str-like. The annotation used to be the only thing saying so, and an
    annotation is not a check: a `str` subclass overriding `__str__` and
    `__eq__` satisfied every test `load` and the gate runner performed on
    these two values, because all of those tests ran on the value itself.
    `str.strip()`, `==` and `f"{...}"` all dispatch to the subclass, so a
    value that reports itself as empty when asked and equal when compared
    walks through the loader untouched and lands in `gates_report.json`
    saying whatever it likes. Requiring the exact type costs one `type(...)
    is str` and removes the whole class of override from every check
    downstream -- see `problems._stamp`, which repeats it for a spec built
    without running `__init__`.

    That is a guard against a mistake and a shortcut, not against an
    adversary; see the note at the end of `gates/runner.py`'s module
    docstring for what an imported module can still do.
    """

    config_cls: type
    gradcheck_case: GradcheckSlot
    reciprocity_case: ReciprocitySlot
    name: str = ""
    module: str = ""

    @property
    def label(self):
        """What to call this spec in an error message, named or not.

        An unnamed spec is the ordinary case now, so the messages below
        cannot print `self.name`: on the path that matters most -- a module
        that got the declaration wrong and has not reached `load` yet -- it
        would print an empty string and name nothing.
        """
        if str(self.name).strip():
            return self.name
        cls = getattr(self.config_cls, "__name__", None)
        return f"<unnamed ProblemSpec, config_cls={cls}>" if cls \
            else "<unnamed ProblemSpec>"

    def __post_init__(self):
        reject_non_str_identity(self)
        if not (isinstance(self.config_cls, type)
                and issubclass(self.config_cls, BaseConfig)):
            raise TypeError(
                f"{self.label}: config_cls must be a BaseConfig subclass, got "
                f"{self.config_cls!r} -- cli.apply_overrides and "
                f"cli.start_run are written against the dataclass fields "
                f"BaseConfig defines")
        for slot in ("gradcheck_case", "reciprocity_case"):
            v = getattr(self, slot)
            if not (callable(v) or isinstance(v, Unsupported)):
                raise TypeError(
                    f"{self.label}.{slot} must be a zero-argument factory or "
                    f"Unsupported('why not'), got {v!r}")


@runtime_checkable
class ProblemModule(Protocol):
    """A module that can be handed to `problems.load`.

    Thin on purpose -- see this file's header. `isinstance(mod, ProblemModule)`
    checks only that `PROBLEM` is present; `ProblemSpec.__post_init__` is what
    checks that it means anything.
    """

    PROBLEM: ProblemSpec
