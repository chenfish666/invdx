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
"""
