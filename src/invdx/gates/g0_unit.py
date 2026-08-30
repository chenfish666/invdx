"""Gate 0 — pure-python unit tests (the pytest suite, seconds)."""

import os

import pytest as _pytest

from .runner import GateResult, NoProblem

NAME = "unit"
ORDER = 0
REQUIRES = ()
MEASURES_PROBLEM = NoProblem(
    "G0 runs this repository's pytest suite; what it measures is the source "
    "tree, and the one number it reports is pytest's exit code. No "
    "simulation runs, so there is nothing a `--problem` could own here")


def run(cfg, args):
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    code = _pytest.main(["-q", os.path.join(root, "tests")])
    status = "ok" if code == 0 else "fail"
    return GateResult(NAME, status, {"pytest_exit_code": int(code)})
