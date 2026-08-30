"""Validation-gate framework.

Gates are ordered, reusable trust checks: unit -> API -> gradcheck -> physics
baseline -> reciprocity -> cross-engine. The runner executes them in order and
stops at the first failure; a gate may honestly SKIP when its preconditions
(e.g. a concrete design problem) don't exist yet. Ordering rationale lives in
engines/conventions.py.

Each gate module defines:
    NAME: str, ORDER: int, REQUIRES: tuple[str, ...]

    REQUIRES is documentation, not dispatch: the runner does not read it.
    A gate whose prerequisites are absent raises, and the runner turns
    that into a fail. Wiring REQUIRES up to skip instead would be worse,
    because a skipped gate is indistinguishable from a passing one in a
    summary line -- which is the failure this whole layer exists to catch.
    run(cfg, args) -> runner.GateResult

Two gates (G2 Part C, G4) measure a concrete problem, chosen by `--problem`.
That creates a case the policy above does not cover: a problem for which the
check has nothing to check -- `phc_bend` has no adjoint to finite-difference,
and its transmission ratio cancels the very normalization G4 exists to
validate. Running the gate anyway would either crash or, worse, pass
vacuously.

The answer is NOT to relax the policy into a skip. It is that the problem
must DECLARE the gap, in code, with a reason:

    reciprocity_case=Unsupported("why this gate has nothing to check here")

`problems/contract.py` gives the slot no default, so a problem that says
nothing fails at import and the runner turns that into a fail -- exactly the
old behaviour for the old reason. A problem that does declare gets a status
of its own ([n/a], or [part] when a gate's generic half still ran) with the
reason printed on the same console line. So the three cases stay three:

    the check ran and passed          [ok]
    the check could not run           [FAIL]   (including "nobody decided")
    the check does not apply, because [n/a] / [part] + the argument for it

The one thing no problem can do is get no coverage quietly, which is what
used to happen by default.

Writing a third such gate: build its `details` with `runner.gate_details`
rather than a dict literal. A gate that measures a problem ends up merging
two authors' keys -- what it measured and whatever free-form dict the problem
handed it -- and a silent merge writes one author's number under the other
author's name, which no reader of the report can detect. `gate_details`
refuses the collision and stamps `problem` / `problem_module` from the loaded
spec, so the module being measured does not supply the fields that identify
it through the path the gate hands it. G4 shipped without that: it spread
`**case.extra` last, and a problem could name itself anything it liked in its
own gate report.

Not knowing about `gate_details` is therefore a bug, so it is not left to
memory -- and not left to a declaration either. `run_gates` requires the two
identity keys from EVERY gate by default: it derives the identity from what
`--problem` asked for -- from the request, not by loading the problem and
reading the answer off the same spec the gate read it off -- and fails the
gate if the result disagrees, or if it carries no identity at all.

A gate that measures no problem, like G0/G1/G3/G5, opts OUT, in the gate
module, with a reason:

    MEASURES_PROBLEM = NoProblem("what this gate measures instead of a device")

Two separate decisions are packed into that line. The POLARITY -- opt out, not
opt in -- came first. It was opt-in once (`MEASURES_PROBLEM = True` to be
checked), which meant the failure the backstop exists for, an author who does
not know the rule, was also the way to turn the backstop off: a new gate
reporting numbers with no identity passed silently. Now writing nothing gets a
loud complaint with both fixes in it, and deleting the line from a gate that
legitimately measures nothing gets a loud false alarm, which is the harmless
direction. The runner never infers the opt-out; a person types it.

The REASON came second, from an audit that showed the polarity was only half
the job. The opt-out was the bare constant `MEASURES_PROBLEM = False`, and
`False` is the same three characters in every module: correct where each of
these four gates typed it, and still correct-LOOKING pasted into a gate that
measures a coupler. The audit did exactly that -- copied G3's declaration and
its explanatory comment into a new gate that reported `CE_fwd_dB` and
`CE_rev_dB` and stamped no identity -- and got `[ok]`. A reason cannot be
neutral that way: "there is no device in the scene" is visibly false in a gate
that has one, which is why `False` is now refused outright with the
replacement spelled out rather than quietly accepted.

The size of that claim, because it is easy to overstate: this makes a copied
opt-out READABLE as wrong. It does not make copying impossible. A reason is a
string and the interpreter cannot check whether it describes the module it was
typed in -- the reader does that, in review. Same boundary as everything else
in this package.

Exactly two things excuse a result from carrying identity, and both are named
here so this reads as the whole rule rather than most of it. One is
`NoProblem(reason)`, above. The other is a result that is already a
`[FAIL]`: a gate that broke before it loaded anything has a real diagnosis in
it, and replacing that with a provenance complaint swaps the cause for a
lecture (G2's Parts A and B fail exactly there). A `[FAIL]` that stamps the
WRONG identity is still caught; the exemption is for absence, not for error.

A gate that always measures one particular problem regardless of `--problem`
declares that problem's name in place of `True`, and is checked against that
instead; before there was a way to say so, such a gate was failed for being
honest. This is NOT a third exemption -- such a gate still owes both identity
keys, and still fails without them. It only changes where the runner gets the
truth it compares against: it resolves the declared name the same request-side
way, without reading anything off the loaded problem, so the declaration
cannot name one problem while the report names another. What it does not check
is that the gate imported the module it named. A gate declaring one name and
loading a different problem to measure produces a self-consistent report about
the wrong thing, and that is a gate lying about its own work -- the far side of
the boundary set out in README, not something this layer claims to cover.

That backstop covers the identity keys only -- for the numbers a gate
measures itself the runner has no independent source, and `gate_details` is
the only thing standing there.

The scope of all of it, stated once here because the rest of this package
reads as if it were absolute: this is provenance as a RECORD. Every check
above catches a gate author or a problem author making a mistake, and none of
them survives a module that is deliberately lying -- an imported module runs
in this process and can rewrite any of it. `runner.py`'s docstring says the
same at the point where the rule lives; the README says it where a reader of
a report will see it.
"""

from .runner import NoProblem

__all__ = ["NoProblem"]
