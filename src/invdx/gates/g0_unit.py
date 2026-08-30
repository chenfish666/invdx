"""Gate 0 — pure-python unit tests (the pytest suite, seconds)."""

import os

import pytest as _pytest

from .runner import GateResult

NAME = "unit"
ORDER = 0
REQUIRES = ()
# This gate measures no problem module, so it owes the report no
# `problem` / `problem_module` keys. Declared, not inferred: the runner
# requires the two identity keys from every gate by default, precisely so
# that a gate author who writes nothing gets a loud complaint instead of a
# silent exemption. See `runner._declared_problem`.
MEASURES_PROBLEM = False


def run(cfg, args):
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    code = _pytest.main(["-q", os.path.join(root, "tests")])
    status = "ok" if code == 0 else "fail"
    return GateResult(NAME, status, {"pytest_exit_code": int(code)})
