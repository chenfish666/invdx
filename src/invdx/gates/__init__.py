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
"""
