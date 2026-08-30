"""The problem contract, and the proof that it buys a new problem something.

Two halves:

  * the declaration rules -- a problem cannot stay silent about a gate, and
    cannot declare a gap without an argument for it;

  * the gates, run UNMODIFIED against a third problem that did not exist when
    they were written (`fixture_problems.tmm_stack`). That is the only test
    that separates "an interface exists" from "an interface is useful", so it
    is written both ways round: the gates pass on the correct physics, and
    they FAIL on a deliberately broken version of the same problem. A gate
    that cannot fail is not coverage.

What is NOT covered here, and why: G2's Parts A and B need fdtdx on a GPU, so
this file exercises Part C directly. The full `g2.run()` path on a third
problem is a GPU check -- `scripts/00_check.py --only gradcheck --problem ...`.
"""

from dataclasses import replace

import numpy as np
import pytest

from invdx import problems
from invdx.config import BaseConfig
from invdx.gates import g2_gradcheck as g2
from invdx.gates import g4_reciprocity as g4
from invdx.gates import runner
from invdx.problems.contract import ProblemSpec, Unsupported

from fixture_problems import tmm_stack

FIXTURE = "fixture_problems.tmm_stack"


class _Args:
    """The slice of an argparse namespace the gates actually read."""

    def __init__(self, problem=None):
        self.problem = problem


# --------------------------------------------------------------------------
# The declaration rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", problems.available())
def test_every_registered_problem_declares_a_usable_spec(name):
    spec = problems.load(name)
    assert spec.name == name, "the registry key and the declared name must agree"
    assert issubclass(spec.config_cls, BaseConfig)
    for slot in ("gradcheck_case", "reciprocity_case"):
        v = getattr(spec, slot)
        assert callable(v) or isinstance(v, Unsupported)


def test_a_problem_cannot_stay_silent_about_a_gate():
    """The failure this whole contract exists to prevent.

    Before it, a new problem got no gradient and no reciprocity coverage by
    DEFAULT, and looked exactly as green as one that had both. The slots
    having no default is the entire enforcement mechanism, so it gets a test.
    """
    with pytest.raises(TypeError, match="reciprocity_case"):
        ProblemSpec(name="silent", config_cls=BaseConfig,
                    gradcheck_case=Unsupported("no adjoint"))


def test_a_declared_gap_must_carry_an_argument():
    """"Not applicable" with no reason is indistinguishable from "not done"."""
    for empty in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="reason"):
            Unsupported(empty)


def test_config_cls_must_be_a_baseconfig_subclass():
    with pytest.raises(TypeError, match="BaseConfig"):
        ProblemSpec(name="x", config_cls=dict,
                    gradcheck_case=Unsupported("a"),
                    reciprocity_case=Unsupported("b"))


def test_a_gate_slot_must_be_a_factory_or_an_explicit_unsupported():
    with pytest.raises(TypeError, match="gradcheck_case"):
        ProblemSpec(name="x", config_cls=BaseConfig,
                    gradcheck_case=None,          # "we'll do it later"
                    reciprocity_case=Unsupported("b"))


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_unknown_problem_names_the_alternatives():
    with pytest.raises(KeyError) as e:
        problems.load("no_such_problem")
    for name in problems.available():
        assert name in str(e.value)


def test_a_module_without_a_declaration_is_not_a_problem():
    with pytest.raises(AttributeError, match="PROBLEM"):
        problems.load("invdx.config")


def test_an_out_of_tree_problem_loads_by_dotted_path():
    """A problem in someone else's package must be gateable without vendoring."""
    assert problems.load(FIXTURE).name == "tmm_stack"


def test_from_args_falls_back_to_the_default():
    assert problems.from_args(_Args()).name == problems.DEFAULT
    assert problems.from_args(_Args(FIXTURE)).name == "tmm_stack"


# --------------------------------------------------------------------------
# Statuses: a declared gap is neither a pass nor a failure
# --------------------------------------------------------------------------


def test_every_status_the_runner_can_emit_has_its_own_label():
    statuses = {runner.OK, runner.FAIL, runner.SKIP,
                runner.NOT_APPLICABLE, runner.PARTIAL}
    assert statuses <= set(runner.LABELS)
    assert len(set(runner.LABELS.values())) == len(runner.LABELS)
    # a column of statuses has to stay a column
    assert len({len(v) for v in runner.LABELS.values()}) == 1


def test_not_applicable_is_distinguishable_from_both_pass_and_failure():
    r = runner.not_applicable("reciprocity", "phc_bend", "nothing to check")
    assert r.status not in (runner.OK, runner.FAIL)
    assert runner.LABELS[r.status] != runner.LABELS[runner.OK]
    assert "nothing to check" in r.details["reason"]


def test_not_applicable_refuses_a_silent_gap():
    with pytest.raises(ValueError, match="reason"):
        runner.not_applicable("reciprocity", "somewhere", "")


# --------------------------------------------------------------------------
# G4, unmodified, against a problem it was not written for
# --------------------------------------------------------------------------


def test_g4_covers_the_third_problem():
    res = g4.run(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.OK, res.details
    assert res.details["problem"] == "tmm_stack"
    # a theorem, not a tolerance: the two directions agree to machine precision
    assert res.details["mismatch_dB"] < 1e-9


def test_g4_still_bites_on_a_one_sided_normalization(monkeypatch):
    """The gate's reason for existing, reproduced on the new problem.

    Reporting |t|^2 as a transmittance drops the exit-side admittance ratio.
    Because the two ambients differ, that is a ONE-SIDED error -- exactly the
    class a check that scales both directions equally would never see.
    """
    broken = ProblemSpec(
        name="tmm_stack_unnormalized", config_cls=tmm_stack.TMMStackConfig,
        gradcheck_case=Unsupported("only the reciprocity path is under test"),
        reciprocity_case=lambda: tmm_stack.reciprocity_case(normalized=False))
    monkeypatch.setattr(problems, "load", lambda name: broken)

    res = g4.run(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "reciprocity violated" in res.details["reason"]
    assert "normalization" in res.details["reason"]
    expected = 10.0 * np.log10(tmm_stack.TMMStackConfig().n_out
                               / tmm_stack.TMMStackConfig().n_in)
    assert res.details["mismatch_dB"] == pytest.approx(expected, rel=1e-9)


def test_g4_reports_a_declared_gap_as_a_gap(monkeypatch):
    """`phc_bend` says the gate has nothing to check; it must not say [ok]."""
    res = g4.run(BaseConfig(), _Args("phc_bend"))
    assert res.status == runner.NOT_APPLICABLE
    assert res.details["problem"] == "phc_bend"
    # the argument, not just the verdict
    assert "cancels" in res.details["reason"]


# --------------------------------------------------------------------------
# G2 Part C, unmodified, against a problem it was not written for
# --------------------------------------------------------------------------


def test_g2_part_c_covers_the_third_problem():
    spec = problems.load(FIXTURE)
    f0, checks, n_bad, info = g2._part_c_problem_device(spec)
    assert n_bad == 0, checks
    assert len(checks) == g2.K_SAMPLES
    # the sampling actually had a choice to make, rather than falling back to
    # "the K largest gradients" -- otherwise the eligibility floor is untested
    assert info["n_eligible"] > g2.K_SAMPLES
    assert all(c["rel_err"] < g2.REL_TOL for c in checks)
    assert f0 < 0, "loss = -FOM, and this stack transmits something"


def test_g2_part_c_still_catches_a_wrong_gradient():
    """A gate that cannot fail is not coverage.

    The forward value is left alone and only the reported gradient is scaled,
    which is what a chain-rule or normalization slip in a real adjoint looks
    like: nothing about the physics output moves.
    """
    case = tmm_stack.gradcheck_case()

    def doubled(p, b):
        loss, grad = case.vg_fn(p, b)
        return loss, 2.0 * grad

    broken = ProblemSpec(
        name="tmm_stack_wrong_grad", config_cls=tmm_stack.TMMStackConfig,
        gradcheck_case=lambda: replace(case, vg_fn=doubled),
        reciprocity_case=Unsupported("only the gradient path is under test"))

    _, checks, n_bad, _ = g2._part_c_problem_device(broken)
    assert n_bad == g2.K_SAMPLES, checks
    assert all(c["rel_err"] > g2.REL_TOL for c in checks)


def test_g2_part_c_declines_to_run_where_there_is_no_gradient():
    """`phc_bend` is numpy-pure by design; the gap is declared, not silent."""
    slot = problems.load("phc_bend").gradcheck_case
    assert isinstance(slot, Unsupported)
    assert "numpy-pure" in slot.reason


@pytest.mark.parametrize("key", sorted(g2._GATE_OWNED_INFO_KEYS))
def test_g2_part_c_refuses_to_write_over_a_problems_own_info_key(key):
    """The sampling report has two authors; a shared name is refused, not resolved.

    Whichever side won a merge, the report would still parse and still carry
    every expected key -- the value would just belong to the other author.
    That is unfalsifiable from the report alone, so the collision has to be
    loud at the point it happens, naming the key.
    """
    case = tmm_stack.gradcheck_case()
    colliding = ProblemSpec(
        name="tmm_stack_clashing_info", config_cls=tmm_stack.TMMStackConfig,
        gradcheck_case=lambda: replace(case, info={**case.info, key: "mine"}),
        reciprocity_case=Unsupported("only the info merge is under test"))

    with pytest.raises(ValueError, match=key):
        g2._part_c_problem_device(colliding)
