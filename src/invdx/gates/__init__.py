"""Validation-gate framework.

Gates are ordered, reusable trust checks: unit -> API -> gradcheck -> physics
baseline -> reciprocity -> cross-engine. The runner executes them in order and
stops at the first failure; a gate may honestly SKIP when its preconditions
(e.g. a concrete design problem) don't exist yet. Ordering rationale lives in
engines/conventions.py.

Each gate module defines:
    NAME: str, ORDER: int, REQUIRES: tuple[str, ...]
    run(cfg, args) -> runner.GateResult
"""
