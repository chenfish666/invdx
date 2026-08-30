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

    PROBLEM: ProblemSpec     -- a name, a config class, and one explicit
                                answer per problem-specific gate

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
    info              -> extra numbers for gates_report.json
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


@dataclass(frozen=True)
class ProblemSpec:
    """The module-level `PROBLEM = ProblemSpec(...)` every problem declares.

    Neither gate slot has a default. That is the one piece of enforcement in
    this file, and it is aimed at the failure the old arrangement produced by
    default: a new problem quietly shipping without gradient or reciprocity
    coverage, looking exactly as green as one that has both.
    """

    name: str
    config_cls: type
    gradcheck_case: GradcheckSlot
    reciprocity_case: ReciprocitySlot

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("ProblemSpec.name must be non-empty")
        if not (isinstance(self.config_cls, type)
                and issubclass(self.config_cls, BaseConfig)):
            raise TypeError(
                f"{self.name}: config_cls must be a BaseConfig subclass, got "
                f"{self.config_cls!r} -- cli.apply_overrides and "
                f"cli.start_run are written against the dataclass fields "
                f"BaseConfig defines")
        for slot in ("gradcheck_case", "reciprocity_case"):
            v = getattr(self, slot)
            if not (callable(v) or isinstance(v, Unsupported)):
                raise TypeError(
                    f"{self.name}.{slot} must be a zero-argument factory or "
                    f"Unsupported('why not'), got {v!r}")


@runtime_checkable
class ProblemModule(Protocol):
    """A module that can be handed to `problems.load`.

    Thin on purpose -- see this file's header. `isinstance(mod, ProblemModule)`
    checks only that `PROBLEM` is present; `ProblemSpec.__post_init__` is what
    checks that it means anything.
    """

    PROBLEM: ProblemSpec
