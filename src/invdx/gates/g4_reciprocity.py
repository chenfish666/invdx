"""Gate 4 — reciprocity: forward vs reciprocal excitation must give the same
coupling efficiency.

This is the gate that caught a factor-2 normalization bug in this codebase's
own mode-overlap path — it is the only single-engine check that validates
*normalization* end-to-end, which is why it outranks the cross-engine gate
for trust in absolute numbers. A convention error that scales both directions
equally is invisible to every other check; one that scales only one of them
shows up here immediately.

Contract:
    1. forward: problem's nominal source -> CE into the target mode
    2. reciprocal: excite from the target mode -> CE back into the source mode
    3. |CE_fwd - CE_rev| within a tight bound (0.2 dB) at the nominal design;
       grayscale high-Q intermediates seen mid-optimization ring longer and
       are noisier, so they need a looser bound (~1.6 dB) or a longer run

Which problem is measured comes from `--problem` (default
`invdx.problems.DEFAULT`); the two measurements come from that problem's
`reciprocity_case()`. The report records both what that problem is called and
the import path it was loaded from (`details["problem_module"]`), because the
name is a label the loader assigns and only the path says which module the
two numbers came out of. Both are stamped by `runner.gate_details` from the
spec the loader returned, and a `ReciprocityCase.extra` carrying either name
is refused — so the module being measured does not supply the fields that
identify it through any path this gate offers it. That is a guard against a
mistake, not against a determined module: see the end of `runner.py`'s
docstring for what an imported problem can still do, and the README for how
that should change the way a report from someone else's problem is read.
This gate deliberately does not know how to produce
them: the value of the check is that the two directions are normalized
INDEPENDENTLY, and a helper here doing both sides would be the very bug it
looks for. What stays here is the comparison, the tolerance and the wording
of the failure.

A problem may declare `reciprocity_case=Unsupported("why not")` — the
runner then prints [n/a] with that reason, which is not a pass and not a
failure and does not look like either. The one thing a problem cannot do is
say nothing: `ProblemSpec` has no default for the slot, so silence is an
import error and the runner turns it into [FAIL].

Cheap-mode tolerance is 0.5 dB, slack enough to absorb the coarse grid and the
short run a case is expected to use; a final design at production resolution
should be held to something much tighter (0.2 dB) — re-tighten per problem
when it matters.
"""

from invdx import problems

from .runner import GateResult, gate_details, not_applicable

NAME = "reciprocity"
ORDER = 4
# This gate measures whatever `--problem` asked for. Read by the runner's
# identity backstop, which then requires the result to carry the two identity
# keys and to carry the right ones. A gate that always measures one
# particular problem regardless of `--problem` declares its NAME here
# instead; see `runner._verify_problem_identity`.
MEASURES_PROBLEM = True
# Documentation, as always here -- and now a property of the PROBLEM rather
# than of this gate: the default problem's case runs two fdtdx scenes, while
# a closed-form one needs nothing. What a case needs is the case's business.
REQUIRES = ("gpu",)

TOL_DB = 0.5


def run(cfg, args):
    spec = problems.from_args(args)
    slot = spec.reciprocity_case
    if isinstance(slot, problems.Unsupported):
        # The spec, not `spec.name`: an [n/a] result carries the same two
        # identity keys every other status does.
        return not_applicable(NAME, spec, slot.reason)

    case = slot()
    mismatch = abs(case.fwd_dB - case.rev_dB)
    # The name says what the numbers are filed under; the module says where
    # they came from. Only the second one can be checked against a tree — and
    # neither is passed in here: `gate_details` reads both off the loaded spec
    # and refuses a `case.extra` that carries a copy of either. This used to
    # be a dict literal ending in `**case.extra`, which put the problem's keys
    # LAST and let the module under measurement name itself.
    details = gate_details(
        spec,
        {"CE_fwd_dB": case.fwd_dB, "CE_rev_dB": case.rev_dB,
         "mismatch_dB": mismatch},
        supplied=case.extra,
        source=f"{spec.name}.reciprocity_case() -> ReciprocityCase.extra")
    if mismatch > TOL_DB:
        return GateResult(NAME, "fail", {
            "reason": f"reciprocity violated on {spec.name}: "
                      f"|CE_fwd - CE_rev| = "
                      f"{mismatch:.3f} dB > {TOL_DB} dB — suspect a "
                      f"normalization/convention bug (a missing factor of 2 "
                      f"on one side is exactly what this looks like)",
            **details})
    return GateResult(NAME, "ok", details)
