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

import importlib
import json
import types
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from invdx import problems
from invdx.config import BaseConfig
from invdx.gates import g0_unit as g0
from invdx.gates import g1_api as g1
from invdx.gates import g2_gradcheck as g2
from invdx.gates import g3_physics as g3
from invdx.gates import g4_reciprocity as g4
from invdx.gates import g5_crossengine as g5
from invdx.gates import runner
from invdx.problems.contract import ProblemSpec, ReciprocityCase, Unsupported

from fixture_problems import lying_name, self_naming, tmm_stack

FIXTURE = "fixture_problems.tmm_stack"
# Same physics, filed under a registered problem's name -- once by declaring
# it, once by being spelled that way, once by writing the report's identity
# keys from inside the problem's own dict. All three are refused; see the
# modules.
IMPOSTOR = "fixture_problems.impostor"
IMPERSONATOR = "fixture_problems.grating_coupler"
FORGER = "fixture_problems.self_naming"

# What G2 Part C measures itself and therefore will not let a problem supply.
# Not imported from the gate: a constant there would be a second copy of the
# `merge_problem_dict` argument list, which is the duplication being removed.
# `test_the_gate_measured_info_keys_are_the_ones_this_file_guards` keeps this
# honest against a real run.
GATE_MEASURED_INFO_KEYS = ("grad_max", "n_eligible", "n_voxels")


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
    assert spec.name == name, "load names the spec after what it was asked for"
    assert issubclass(spec.config_cls, BaseConfig)
    for slot in ("gradcheck_case", "reciprocity_case"):
        v = getattr(spec, slot)
        assert callable(v) or isinstance(v, Unsupported)


@pytest.mark.parametrize("name", problems.available())
def test_no_problem_module_writes_its_own_name_down(name):
    """The name is derived, so there must be no handwritten copy left.

    This is the assertion that used to read `spec.name == name` on the loaded
    spec, which could only ever compare `load`'s output against `load`'s
    input. It has to look at the MODULE to say anything: the module path, the
    registry key and a `name=` string were three copies of one fact with no
    derivation between them, and two of them still are, unavoidably. The
    third one was avoidable and is gone.
    """
    mod = importlib.import_module(problems._REGISTRY[name])
    assert not str(mod.PROBLEM.name).strip(), (
        f"{name} hand-writes its own name; delete the `name=` line and let "
        f"problems.load derive it")


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


def test_an_unnamed_spec_still_names_itself_in_its_own_error_messages():
    """A spec is unnamed until `load` stamps it, so the errors cannot use it.

    Without a stand-in the message would open with an empty string, and
    `.gradcheck_case must be ...` names nothing at all -- least of all in the
    case that matters, a module that has not reached `load` yet.
    """
    with pytest.raises(TypeError) as e:
        ProblemSpec(config_cls=BaseConfig, gradcheck_case=None,
                    reciprocity_case=Unsupported("b"))
    msg = str(e.value)
    assert not msg.startswith("."), "an empty name leaves a bare '.slot'"
    assert "BaseConfig" in msg, "the config class stands in for the name"
    assert "gradcheck_case" in msg


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


def test_a_problem_is_named_by_what_was_asked_for_not_by_itself():
    """The module says nothing about its name; `load` supplies it.

    Both halves matter: the module is silent (so there is no third copy to
    drift), and what comes back is named anyway (so the gates, which key
    their reports by the name, still have one).
    """
    assert tmm_stack.PROBLEM.name == "", "the fixture declares no name"
    assert problems.load(FIXTURE).name == "tmm_stack", "the last path segment"
    assert tmm_stack.PROBLEM.name == "", "and load did not mutate the module"


def test_load_refuses_a_problem_that_claims_someone_elses_name():
    """The attack the derivation exists to stop, run for real.

    `fixture_problems.impostor` is a 16-layer dielectric film that declares
    `name="grating_coupler"`. Loaded, its numbers would be written to
    `gates_report.json` as `grating_coupler_f0`, `grating_coupler_fd_checks`,
    `grating_coupler_sampling` and `details["problem"]="grating_coupler"` --
    same keys, same shape, indistinguishable from the real coupler's report.
    """
    with pytest.raises(ValueError) as e:
        problems.load("fixture_problems.impostor")
    msg = str(e.value)
    # both names, because either one alone leaves the reader guessing which
    # of the two is wrong
    assert "grating_coupler" in msg, "what the module declared"
    assert "impostor" in msg, "what was actually asked for"
    # and why it matters, not just that it happened
    assert "gates_report.json" in msg


def test_load_refuses_a_registered_name_claimed_by_a_filename():
    """The same attack with the `name=` line deleted -- and it used to work.

    `fixture_problems.grating_coupler` declares no name at all. It does not
    have to: a dotted path is named by its last segment, so the file's own
    spelling was enough to get it stamped `grating_coupler` and its numbers
    filed under the real coupler's keys. Checking the DECLARED name was
    necessary and not sufficient.
    """
    with pytest.raises(ValueError) as e:
        problems.load(IMPERSONATOR)
    msg = str(e.value)
    assert "grating_coupler" in msg, "the name being claimed"
    assert IMPERSONATOR in msg, "who is claiming it"
    # where the real one lives -- otherwise the reader cannot tell which of
    # the two modules is the impostor
    assert problems._REGISTRY["grating_coupler"] in msg
    assert "gates_report.json" in msg, "and why a name collision matters"


def test_the_filename_alone_takes_the_name_when_the_guard_is_removed(
        monkeypatch):
    """Red first: prove the guard is what stops it, not something upstream.

    With the check disabled the module loads and is stamped
    `grating_coupler` -- exactly the report an audit produced against this
    tree. The one field that still tells the truth is `module`, which is the
    argument for recording it.
    """
    monkeypatch.setattr(problems, "_reject_registry_impersonation",
                        lambda path: None)
    spec = problems.load(IMPERSONATOR)
    assert spec.name == "grating_coupler", "the attack, with the guard off"
    assert spec.module == IMPERSONATOR, "and the field that never agreed"


def test_the_guard_rejects_a_collision_and_nothing_else():
    """It must not become a ban on out-of-tree problems in general.

    The check is on the NAME a path would take, so anything that would take a
    name of its own is untouched -- including the registered module's own
    path, which is the registered problem spelled out in full.
    """
    for path in (FIXTURE, IMPOSTOR, "yourpkg.problems.spiral", "a.b.mmi",
                 problems._REGISTRY["grating_coupler"]):
        problems._reject_registry_impersonation(path)      # no raise
    assert problems.load(FIXTURE).name == "tmm_stack"


# --------------------------------------------------------------------------
# Provenance: what the report says about where its numbers came from
# --------------------------------------------------------------------------


def test_a_loaded_spec_records_the_module_it_was_imported_from():
    assert problems.load(FIXTURE).module == FIXTURE
    assert tmm_stack.PROBLEM.module == "", "the fixture declares no module"


def test_a_spec_cannot_declare_the_module_it_came_from():
    """`module` is a record of an import, so a written-down one is refused."""
    lying = ProblemSpec(
        module="invdx.problems.grating_coupler",
        config_cls=tmm_stack.TMMStackConfig,
        gradcheck_case=Unsupported("only the stamp is under test"),
        reciprocity_case=Unsupported("only the stamp is under test"))
    with pytest.raises(ValueError, match="module="):
        problems._stamp(lying, "tmm_stack", FIXTURE)


def test_g4_records_which_module_it_measured():
    res = g4.run(BaseConfig(), _Args(FIXTURE))
    assert res.details["problem"] == "tmm_stack", "the label"
    assert res.details["problem_module"] == FIXTURE, "and the source"


def test_problem_module_is_the_only_key_g4_gained():
    """The report grew one field; nothing that was already in it moved.

    Asserted on the fixture because the shipped coupler's case needs a GPU,
    and it is the same `details` construction either way.
    """
    res = g4.run(BaseConfig(), _Args(FIXTURE))
    before = {"problem", "CE_fwd_dB", "CE_rev_dB", "mismatch_dB",
              "T_fwd", "T_rev", "normalized_reverse"}
    assert set(res.details) == before | {"problem_module"}


def test_the_module_survives_into_gates_report_json(tmp_path):
    """The point of the field: it has to be there after the run directory is
    the only thing left. `cmdline.txt` says which `--problem` was typed, but
    it does not travel with the report."""
    res = g4.run(BaseConfig(), _Args(FIXTURE))
    res.details["seconds"] = 0.0          # what run_gates adds before writing
    path = runner.write_report([res], str(tmp_path))
    with open(path) as f:
        written, = json.load(f)
    assert written["details"]["problem_module"] == FIXTURE
    assert written["details"]["problem"] == "tmm_stack"


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


@pytest.mark.parametrize("key", GATE_MEASURED_INFO_KEYS)
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


def test_the_gate_measured_info_keys_are_the_ones_this_file_guards():
    """Keep the list above honest without giving the gate a second copy of it.

    `g2` used to export `_GATE_OWNED_INFO_KEYS`; the keys are now simply the
    dict it hands `runner.merge_problem_dict`, so the only way to know what
    they are is to look at what a run produced that the problem did not.
    """
    _, _, _, info = g2._part_c_problem_device(problems.load(FIXTURE))
    measured = set(info) - set(tmm_stack.gradcheck_case().info)
    assert measured == set(GATE_MEASURED_INFO_KEYS)


# --------------------------------------------------------------------------
# The identity keys: the subject of a measurement may not fill them in
# --------------------------------------------------------------------------


def _old_g4_details(spec, measured, supplied=None, source=None):
    """G4's `details` construction as it stood before the guard.

    A dict literal whose last entry was `**case.extra`, i.e. the problem's
    keys applied over the gate's. Reproduced here rather than described,
    because the test that uses it has to show the attack working.
    """
    return {"problem": spec.name, "problem_module": spec.module,
            **measured, **(supplied or {})}


def test_a_problem_can_forge_its_own_identity_once_the_guard_is_removed(
        monkeypatch):
    """Red first: prove the guard is load-bearing, not decorative.

    `fixture_problems.self_naming` states no claim anywhere -- honest
    filename, no `name=`, no `module=` -- so both earlier impersonation
    checks pass it, and `load` stamps it truthfully. With G4 merging
    `case.extra` last, as it did, the module under measurement replaces both
    fields that say which module was measured, and the resulting
    `gates_report.json` is indistinguishable from the shipped coupler's.
    """
    assert problems.load(FORGER).name == self_naming.HONEST_NAME
    assert problems.load(FORGER).module == self_naming.HONEST_MODULE

    monkeypatch.setattr(g4, "gate_details", _old_g4_details)
    res = g4.run(BaseConfig(), _Args(FORGER))

    assert res.status == runner.OK, "and it passes, which is the problem"
    assert res.details["problem"] == self_naming.FORGED_NAME
    assert res.details["problem_module"] == self_naming.FORGED_MODULE


@pytest.mark.parametrize("key", ["problem", "problem_module"])
def test_g4_refuses_a_problem_that_supplies_an_identity_key(key):
    """Green: the same probe, the same gate, unmodified."""
    with pytest.raises(ValueError, match=key):
        g4.run(BaseConfig(), _Args(FORGER))


@pytest.mark.parametrize("key", ["problem", "problem_module"])
def test_g2_part_c_refuses_a_problem_that_supplies_an_identity_key(key):
    """The same probe against the other problem-parameterized gate.

    Part C is called directly: G2's Parts A and B need fdtdx on a GPU, and
    the merge under test is entirely inside Part C. The full `g2.run()` path
    is `scripts/00_check.py --only gradcheck --problem
    fixture_problems.self_naming` on a machine with a card.
    """
    with pytest.raises(ValueError, match=key):
        g2._part_c_problem_device(problems.load(FORGER))


def test_g2_stamps_the_identity_from_the_spec_not_by_hand(monkeypatch):
    """G2's top-level stamp, exercised without a card.

    Parts A and B need fdtdx on a GPU and are stubbed out; what is under test
    is the line that used to be two hand-written `details[...] =` statements
    and is now the same `gate_details` call G4 makes. Part C is declined by
    the spec, so the gate reports [part] and the assertion is about the two
    keys and nothing else.
    """
    monkeypatch.setattr(g2, "_part_a_filter_chain", lambda: 0.0)
    monkeypatch.setattr(g2, "_part_b_fdtdx_fd", lambda cfg: (0.0, [], 0))
    spec = replace(problems.load(FIXTURE),
                   gradcheck_case=Unsupported("part C is not what is tested"))
    monkeypatch.setattr(problems, "load", lambda name: spec)

    res = g2.run(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.PARTIAL
    assert res.details["problem"] == "tmm_stack"
    assert res.details["problem_module"] == FIXTURE


def test_the_refusal_names_the_colliding_key_and_where_to_fix_it():
    """An error that does not name the key sends the reader to read a merge."""
    with pytest.raises(ValueError) as e:
        g4.run(BaseConfig(), _Args(FORGER))
    msg = str(e.value)
    # both collisions, listed together -- one name would leave the reader
    # fixing half of it and hitting the same error again
    assert "['problem', 'problem_module']" in msg
    assert "ReciprocityCase.extra" in msg, "which dict to edit"
    assert self_naming.HONEST_NAME in msg, "whose dict it is"


def test_a_legitimate_extra_is_untouched():
    """The guard must not become a ban on the problem reporting anything."""
    res = g4.run(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.OK
    for key, value in tmm_stack.reciprocity_case().extra.items():
        assert res.details[key] == value


def test_a_legitimate_info_is_untouched():
    _, _, _, info = g2._part_c_problem_device(problems.load(FIXTURE))
    for key, value in tmm_stack.gradcheck_case().info.items():
        assert info[key] == value


@pytest.mark.parametrize("key", sorted(runner.RESERVED_DETAIL_KEYS))
def test_no_problem_supplied_dict_may_carry_a_reserved_key(key):
    """`seconds`, `reason` and `exception` are the runner's, like the two IDs.

    A supplied `seconds` would be silently overwritten by the stopwatch; a
    supplied `reason` would print next to another problem's [ok] on the
    console line. Neither is an identity, but both are the gate's words.
    """
    with pytest.raises(ValueError, match=key):
        runner.merge_problem_dict("a problem's dict", {key: "mine"},
                                  {"measured": 1.0})


# --------------------------------------------------------------------------
# The backstop: a gate that never heard of any of the above
# --------------------------------------------------------------------------


class _ForgetfulGate:
    """A third gate, written by someone who did not read gates/__init__.py.

    It builds `details` with a dict literal and lets the problem's dict land
    on top -- exactly G4's original mistake, which is the point: the question
    is whether the next gate has to remember, not whether the two existing
    ones did.
    """

    NAME = "forgetful"
    ORDER = 99

    def run(self, cfg, args):
        spec = problems.from_args(args)
        case = spec.reciprocity_case()
        return runner.GateResult(self.NAME, runner.OK,
                                 {"problem": spec.name,
                                  "problem_module": spec.module,
                                  **case.extra})


def test_the_runner_catches_a_gate_that_did_not_use_the_guard(monkeypatch):
    """The failure has to be loud without the new gate's author doing anything.

    `run_gates` resolves `--problem` itself and compares it with what the
    result claims, so a forged identity fails the gate before the report is
    written rather than being copied into it.
    """
    monkeypatch.setattr(runner, "discover", lambda: [_ForgetfulGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FORGER))

    assert res.status == runner.FAIL
    assert "problem_module" in res.details["exception"]
    assert self_naming.FORGED_MODULE in res.details["exception"], "what it said"
    assert self_naming.HONEST_MODULE in res.details["exception"], "the truth"


def test_the_backstop_passes_a_gate_that_reports_honestly(monkeypatch):
    """It must not fail every gate; the honest fixture goes through it."""
    monkeypatch.setattr(runner, "discover", lambda: [_ForgetfulGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.OK, res.details
    assert res.details["problem_module"] == FIXTURE


def test_the_backstop_leaves_a_gate_that_measures_no_problem_alone():
    """G0/G1/G3/G5 stamp no identity -- but only because they SAY so.

    The exemption is not "no identity keys present"; it is the gate module's
    own `MEASURES_PROBLEM = NoProblem(...)`. The real module is used rather
    than a stand-in, so this test also fails if that line is deleted from G0.
    """
    res = runner.GateResult(g0.NAME, runner.OK, {"pytest_exit_code": 0})
    runner._verify_problem_identity(g0, res, _Args(FORGER))       # no raise


# --------------------------------------------------------------------------
# The identity fields themselves: a `str` that answers for the loader
#
# Everything above this line guards the DICTS a problem hands over. The
# checks below are about the two fields on the spec, which were annotated
# `str` and never checked, so a `str` subclass could answer the loader's
# questions on its own behalf. See `fixture_problems.lying_name`.
# --------------------------------------------------------------------------

LIAR = "fixture_problems.lying_name"


def _old_stamp(spec, requested, path):
    """`problems._stamp` as it stood before the type check, reproduced.

    Two decisions in it, both taken by asking the value under examination:
    `str(spec.name).strip()` decides whether a name was declared at all, and
    `spec.name == requested` decides whether the spec needs stamping. A value
    that answers "" and True walks out unchanged, still holding what it
    actually contains. Written out rather than described, because the test
    that uses it has to show the forgery working end to end.
    """
    declared = str(spec.name).strip()
    if declared and declared != requested:
        raise ValueError("declared name disagrees")
    declared_module = str(spec.module).strip()
    if declared_module and declared_module != path:
        raise ValueError("declared module disagrees")
    if spec.name == requested and spec.module == path:
        return spec
    return replace(spec, name=requested, module=path)


def test_a_str_subclass_forges_an_identity_when_the_stamp_takes_its_word(
        monkeypatch):
    """Red first: the two fields were guarded by an annotation, i.e. by nothing.

    With the type check removed and the old stamp restored, `load` returns a
    spec that reports itself as unnamed and as already-correct, while the
    characters it actually carries -- and therefore the characters
    `json.dump` writes into `gates_report.json` -- are another problem's.
    """
    monkeypatch.setattr(problems, "reject_non_str_identity", lambda spec: None)
    monkeypatch.setattr(problems, "_stamp", _old_stamp)

    spec = problems.load(LIAR)
    # What the loader saw when it asked, and what the field actually holds.
    assert str(spec.name) == "", "it reports itself as unnamed"
    assert spec.name == "anything at all", "and as equal to whatever it is asked"
    written = json.dumps({"problem": spec.name, "problem_module": spec.module})
    assert lying_name.FORGED_NAME in written, "what the report would say"
    assert lying_name.FORGED_MODULE in written
    assert lying_name.HONEST_NAME not in written, "and what it would not"


def test_the_str_subclass_is_refused_and_the_message_says_it_is_a_type():
    """Green: the same probe, unmodified, and the refusal names the problem.

    The message has to say `str` and say `subclass`. The value prints as an
    ordinary problem name, so a message phrased as "wrong name" would send
    the reader hunting for a typo in a string that does not have one.
    """
    with pytest.raises(TypeError) as e:
        problems.load(LIAR)
    msg = str(e.value)
    assert "must be exactly `str`" in msg
    assert "LyingName" in msg, "the type it actually is"
    assert "subclass" in msg and "isinstance" in msg, "why it got this far"
    assert "ProblemSpec.name" in msg, "which field"


def test_a_hand_built_spec_cannot_take_a_str_subclass_identity_either():
    """The same refusal at construction, which is where a module hits it.

    `fixture_problems.lying_name` has to assemble its spec with
    `object.__new__` precisely because this raises -- and that bypass is why
    `_stamp` repeats the check on every load rather than trusting `__init__`.
    """
    for field in ("name", "module"):
        with pytest.raises(TypeError, match="exactly `str`"):
            ProblemSpec(**{field: lying_name.LyingName("grating_coupler")},
                        config_cls=tmm_stack.TMMStackConfig,
                        gradcheck_case=Unsupported("only the type is tested"),
                        reciprocity_case=Unsupported("only the type is tested"))


def test_the_stamp_no_longer_returns_the_spec_it_was_given():
    """The shortcut asked the value whether it needed correcting.

    Even for an already-correct spec the stamp now rebuilds, so the two
    fields that come back were constructed here from the request rather than
    carried over from whatever arrived.
    """
    spec = problems.load(FIXTURE)
    stamped = problems._stamp(spec, "tmm_stack", FIXTURE)
    assert stamped is not spec, "no pass-through, so nothing survives it"
    assert type(stamped.name) is str and type(stamped.module) is str
    assert (stamped.name, stamped.module) == ("tmm_stack", FIXTURE)


def test_a_problemspec_subclass_is_refused_by_load(monkeypatch):
    """`isinstance` would let a subclass answer for its own identity.

    A subclass can override `__getattribute__`, so `spec.name` becomes a call
    into code the module wrote -- the same hole one level up from the `str`
    subclass, and one the field-level type check cannot see, because by then
    the value has already been fetched through the override. Rebuilding a
    plain spec from the subclass's fields was the alternative: it reads the
    same attributes through the same override, and it would silently discard
    whatever the author added the subclass for.
    """
    class Sneaky(ProblemSpec):
        def __getattribute__(self, item):
            if item == "name":
                return "grating_coupler"
            return super().__getattribute__(item)

    sneaky = Sneaky(config_cls=tmm_stack.TMMStackConfig,
                    gradcheck_case=Unsupported("only the type is tested"),
                    reciprocity_case=Unsupported("only the type is tested"))
    monkeypatch.setattr(problems.importlib, "import_module",
                        lambda path: types.SimpleNamespace(PROBLEM=sneaky))

    with pytest.raises(TypeError) as e:
        problems.load("yourpkg.problems.sneaky")
    msg = str(e.value)
    assert "exactly ProblemSpec" in msg, "not merely an instance of one"
    assert "Sneaky" in msg, "what it actually is"
    assert "subclass" in msg and "answer for it" in msg, "and why that matters"


# --------------------------------------------------------------------------
# A mapping with two answers about its own contents
# --------------------------------------------------------------------------


class _TwoFaced(Mapping):
    """A mapping whose `keys()` and `__iter__` do not agree.

    `dict(m)` asks `keys()`; `for k in m` asks `__iter__`. The merge used to
    do the first and the collision check the second, so anything listed only
    by `keys()` was copied into the report without ever being examined --
    including `reason`, which is the text printed on the console line next to
    the gate's status.
    """

    HIDDEN = {"reason": "this rode in past the check"}
    SHOWN = {"T_fwd": 0.5}

    def __getitem__(self, key):
        return {**self.SHOWN, **self.HIDDEN}[key]

    def __iter__(self):
        return iter(self.SHOWN)                  # what the check used to see

    def keys(self):
        return [*self.SHOWN, *self.HIDDEN]       # what the merge used to copy

    def __len__(self):
        return len(self.SHOWN) + len(self.HIDDEN)


def _old_merge_problem_dict(source, supplied, owned):
    """`runner.merge_problem_dict` before the copy, reproduced.

    One line apart from the real one: the check iterates `supplied` and the
    merge calls `dict(supplied)`, i.e. the keys are fetched twice, by two
    different protocols.
    """
    reserved = runner.RESERVED_DETAIL_KEYS | set(owned)
    clash = sorted(k for k in supplied if k in reserved)
    if clash:
        raise ValueError(f"{source} supplies {clash}")
    merged = dict(supplied)
    merged.update(owned)
    return merged


def test_two_answers_about_the_keys_used_to_smuggle_one_past_the_check():
    """Red: the old merge copied a key its own guard never looked at."""
    merged = _old_merge_problem_dict("probe", _TwoFaced(), {"CE_fwd_dB": -9.7})
    assert merged["reason"] == _TwoFaced.HIDDEN["reason"], "into the report"
    assert "reason" in runner.RESERVED_DETAIL_KEYS, "and it is a reserved key"


def test_the_merge_reads_the_supplied_keys_once():
    """Green: one copy, taken first, and both the check and the merge use it."""
    with pytest.raises(ValueError, match="reason"):
        runner.merge_problem_dict("probe", _TwoFaced(), {"CE_fwd_dB": -9.7})


def test_a_gate_case_that_hides_a_reserved_key_is_refused_by_the_gate():
    """The same mapping through the documented path, not the helper directly."""
    hiding = ProblemSpec(
        config_cls=tmm_stack.TMMStackConfig,
        gradcheck_case=Unsupported("only the extra merge is under test"),
        reciprocity_case=lambda: ReciprocityCase(
            fwd_dB=-9.7, rev_dB=-9.7, extra=_TwoFaced()))
    with pytest.raises(ValueError, match="reason"):
        runner.gate_details(replace(hiding, name="probe", module="a.b.probe"),
                            {"CE_fwd_dB": -9.7},
                            supplied=hiding.reciprocity_case().extra,
                            source="probe's ReciprocityCase.extra")


def test_a_supplied_mapping_that_calls_itself_empty_is_still_merged():
    """`supplied or {}` asked the mapping whether it had anything in it."""

    class _ModestlyEmpty(dict):
        def __bool__(self):
            return False

    details = runner.gate_details(
        problems.load(FIXTURE), {"CE_fwd_dB": -9.7},
        supplied=_ModestlyEmpty({"T_fwd": 0.5}))
    assert details["T_fwd"] == 0.5, "its keys are in the report"


# --------------------------------------------------------------------------
# The backstop's second opinion has to come from somewhere else
# --------------------------------------------------------------------------


def _old_verify_problem_identity(gate_name, res, args):
    """The backstop before it stopped asking the loader. Reproduced.

    Its truth came from `problems.from_args(args)` -- the same call the gate
    made, returning the same object the gate read its identity off. Where the
    loader itself could be made to stamp the wrong value, both sides of this
    comparison carried it and the check passed.
    """
    stamped = {k: res.details[k] for k in runner.ID_KEYS if k in res.details}
    if not stamped:
        return
    spec = problems.from_args(args)
    truth = {"problem": spec.name, "problem_module": spec.module}
    wrong = {k: v for k, v in stamped.items() if v != truth[k]}
    if wrong:
        raise ValueError(f"identity mismatch: {wrong}")


def test_the_old_backstop_agreed_with_a_loader_that_had_been_fooled(monkeypatch):
    """Red: comparing the spec with itself is an assertion that `==` works.

    The lie is placed in `load`, which is what the `str`-subclass probe
    achieves by other means. A gate that uses `gate_details` correctly then
    stamps the wrong identity, and the old backstop -- which called `load`
    again -- got the same wrong answer and agreed.
    """
    lying = replace(problems.load(FIXTURE), name="grating_coupler",
                    module="invdx.problems.grating_coupler")
    monkeypatch.setattr(problems, "load", lambda name: lying)

    res = g4.run(BaseConfig(), _Args(FIXTURE))
    assert res.details["problem"] == "grating_coupler", "the forged report"
    _old_verify_problem_identity("reciprocity", res, _Args(FIXTURE))  # agrees


def test_the_backstop_derives_the_truth_from_the_request_not_from_the_spec(
        monkeypatch):
    """Green: the same forged report, refused, because the sides are independent.

    `problems.identity_from_args` resolves `--problem` through the registry
    and the dotted-path rule and imports nothing, so nothing the loaded
    module does can move it.
    """
    lying = replace(problems.load(FIXTURE), name="grating_coupler",
                    module="invdx.problems.grating_coupler")
    monkeypatch.setattr(problems, "load", lambda name: lying)
    monkeypatch.setattr(runner, "discover", lambda: [g4])

    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "grating_coupler" in res.details["exception"], "what it claimed"
    assert "tmm_stack" in res.details["exception"], "what was asked for"


def test_requested_identity_imports_nothing(monkeypatch):
    """The independence is the point, so it gets an assertion of its own."""
    monkeypatch.setattr(problems.importlib, "import_module",
                        lambda path: pytest.fail(f"imported {path}"))
    assert problems.requested_identity(FIXTURE) == ("tmm_stack", FIXTURE)
    assert problems.requested_identity("grating_coupler") == (
        "grating_coupler", problems._REGISTRY["grating_coupler"])


def test_requested_identity_refuses_a_name_it_cannot_resolve():
    """A backstop that silently derived nothing would silently agree."""
    with pytest.raises(KeyError, match="unknown problem"):
        problems.requested_identity("no_such_problem")


# --------------------------------------------------------------------------
# A gate that honestly measures a problem other than `--problem`
# --------------------------------------------------------------------------


class _CrossProblemGate:
    """A gate that always measures one particular problem.

    Legitimate: a gate can exist to check one reference device on every run,
    whatever `--problem` the rest of the run is about. It uses `gate_details`
    correctly and stamps the spec it actually loaded. The backstop used to
    fail it anyway -- it compared against `from_args(args)` and could reach
    no other conclusion -- with a message accusing the gate of letting a
    problem supply its own identity, which is the opposite of what it did.
    """

    NAME = "cross_problem"
    ORDER = 96
    MEASURES_PROBLEM = FIXTURE

    def run(self, cfg, args):
        spec = problems.load(self.MEASURES_PROBLEM)
        case = spec.reciprocity_case()
        return runner.GateResult(
            self.NAME, runner.OK,
            runner.gate_details(spec, {"mismatch_dB": abs(case.fwd_dB - case.rev_dB)}))


def test_an_honest_gate_measuring_another_problem_is_not_accused_of_forgery(
        monkeypatch):
    """`--problem` names one thing, the gate declares another, both are honest."""
    monkeypatch.setattr(runner, "discover", lambda: [_CrossProblemGate()])
    res, = runner.run_gates(BaseConfig(), _Args("phc_bend"))
    assert res.status == runner.OK, res.details.get("exception", res.details)
    assert res.details["problem"] == "tmm_stack", "what it measured"
    assert res.details["problem_module"] == FIXTURE


def test_the_same_gate_without_the_declaration_is_still_failed(monkeypatch):
    """Red: the misjudgement was real, and the declaration is what fixes it."""

    class _Undeclared(_CrossProblemGate):
        MEASURES_PROBLEM = None

    monkeypatch.setattr(runner, "discover", lambda: [_Undeclared()])
    res, = runner.run_gates(BaseConfig(), _Args("phc_bend"))
    assert res.status == runner.FAIL


def test_the_declaration_is_not_a_way_to_write_any_identity_you_like(
        monkeypatch):
    """It must not become the escape hatch the guard just closed.

    The runner resolves the declared name itself and compares it with what
    the result stamped, so declaring one problem while stamping another
    fails. The only way a gate gets a name into a report is by loading that
    problem, which is what the report then honestly says.
    """

    class _Mismatched(_CrossProblemGate):
        MEASURES_PROBLEM = "phc_bend"        # declared

        def run(self, cfg, args):            # loaded and stamped: not that
            spec = problems.load(FIXTURE)
            return runner.GateResult(self.NAME, runner.OK,
                                     runner.gate_details(spec, {"x": 1.0}))

    monkeypatch.setattr(runner, "discover", lambda: [_Mismatched()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "phc_bend" in res.details["exception"], "what it declared"
    assert "tmm_stack" in res.details["exception"], "what it stamped"


# --------------------------------------------------------------------------
# A gate that writes no identity at all
# --------------------------------------------------------------------------


class _AnonymousGate:
    """A gate that measures a problem and never says which one.

    The backstop used to return early when `details` carried neither
    identity key -- it could not tell this gate apart from G0, which measures
    no problem at all. So the way past the guard was to write less, and the
    result was a passing report with no provenance and nothing complaining.
    """

    NAME = "anonymous"
    ORDER = 97
    MEASURES_PROBLEM = True

    def __init__(self, status=runner.OK):
        self.status = status

    def run(self, cfg, args):
        spec = problems.from_args(args)
        case = spec.reciprocity_case()
        details = {"CE_fwd_dB": case.fwd_dB, "CE_rev_dB": case.rev_dB}
        if self.status == runner.FAIL:
            details["reason"] = "the real diagnosis, which must survive"
        return runner.GateResult(self.NAME, self.status, details)


def test_a_gate_that_stamps_no_identity_at_all_is_caught(monkeypatch):
    """Writing nothing is not a way to have nothing checked."""
    monkeypatch.setattr(runner, "discover", lambda: [_AnonymousGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    exc = res.details["exception"]
    assert "problem_module" in exc and "problem" in exc, "what is missing"
    assert "gate_details" in exc, "and how to supply it"


def test_the_runner_does_not_stamp_the_missing_identity_itself(monkeypatch):
    """It never watched the numbers being produced, so it has nothing to record.

    Filling the fields in would put a value the runner INFERRED under a field
    name that means "recorded", which is the distinction the whole layer
    rests on.
    """
    monkeypatch.setattr(runner, "discover", lambda: [_AnonymousGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert "problem" not in res.details
    assert "problem_module" not in res.details


def test_a_gate_that_already_failed_keeps_its_own_reason(monkeypatch):
    """The exemption: G2 Parts A and B fail before any problem is loaded.

    Requiring provenance from a result that is already a failure would
    replace a real diagnosis with a complaint about bookkeeping.
    """
    monkeypatch.setattr(runner, "discover",
                        lambda: [_AnonymousGate(runner.FAIL)])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert res.details["reason"] == "the real diagnosis, which must survive"
    assert "exception" not in res.details


def test_a_failing_gate_is_still_not_allowed_a_wrong_identity(monkeypatch):
    """The exemption is about absence only; a stamped claim is always checked."""

    class _FailingForger(_AnonymousGate):
        def run(self, cfg, args):
            return runner.GateResult(self.NAME, runner.FAIL,
                                     {"reason": "something broke",
                                      "problem": "grating_coupler",
                                      "problem_module": "invdx.problems.grating_coupler"})

    monkeypatch.setattr(runner, "discover", lambda: [_FailingForger()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "grating_coupler" in res.details["exception"]


def test_an_n_a_result_carries_both_identity_keys():
    """[n/a] is the status a problem chooses for itself, so it says the most."""
    res = g4.run(BaseConfig(), _Args("phc_bend"))
    assert res.status == runner.NOT_APPLICABLE
    assert res.details["problem"] == "phc_bend"
    assert res.details["problem_module"] == problems._REGISTRY["phc_bend"]


def test_a_declared_gap_survives_the_backstop(monkeypatch):
    """The whole [n/a] path through `run_gates`, not just the gate."""
    monkeypatch.setattr(runner, "discover", lambda: [g4])
    res, = runner.run_gates(BaseConfig(), _Args("phc_bend"))
    assert res.status == runner.NOT_APPLICABLE, res.details


# --------------------------------------------------------------------------
# The console line is one line
# --------------------------------------------------------------------------


FORGED_LINE = "[ok]  G4 reciprocity (91.05s)"


def test_a_multi_line_reason_cannot_forge_a_console_line(monkeypatch, capsys):
    """`Unsupported(reason)` is the problem's own text, printed verbatim.

    A newline in it used to put a second, entirely fabricated status line on
    the console under the real one -- and the console is what a person reads
    a gate run from. `gates_report.json` still records the reason as written,
    because a multi-line string inside JSON is a string, not a line.
    """
    spec = replace(problems.load(FIXTURE),
                   reciprocity_case=Unsupported(
                       f"the two directions cancel\n{FORGED_LINE}"))
    monkeypatch.setattr(problems, "load", lambda name: spec)
    monkeypatch.setattr(runner, "discover", lambda: [g4])

    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    out = capsys.readouterr().out
    assert res.status == runner.NOT_APPLICABLE
    assert len([ln for ln in out.splitlines() if ln.strip()]) == 1, out
    assert not any(ln.startswith("[ok]") for ln in out.splitlines()), out
    assert FORGED_LINE in res.details["reason"], "the JSON side keeps the text"
    assert "\n" in res.details["reason"], "newlines and all"


def test_the_collapse_is_applied_to_every_reason_not_only_to_n_a():
    """G2 builds its [part] reason by interpolating the problem's text too.

    The collapse therefore lives at the print, which is the one place every
    status goes through, rather than in `not_applicable`.
    """
    flattened = runner.one_line(f"parts A+B passed\n{FORGED_LINE}")
    assert "\n" not in flattened
    assert flattened.startswith("parts A+B passed [ok] G4 reciprocity")


def test_one_line_leaves_an_ordinary_reason_readable():
    assert runner.one_line("nothing to check here") == "nothing to check here"
    assert runner.one_line("  spaced \t out \n ") == "spaced out"


@pytest.mark.parametrize("mod", [g2, g4])
def test_a_problem_parameterized_gate_declares_that_it_measures_one(mod):
    """The presence half of the backstop is opt-in per gate, so it is asserted.

    Without the declaration a gate that stamps nothing looks exactly like G0,
    which measures no problem at all -- and the way past the guard would be to
    write less.
    """
    assert getattr(mod, "MEASURES_PROBLEM", None) is True


def test_the_declaration_is_read_once(monkeypatch):
    """A module can answer `PROBLEM` differently on the second look.

    Replacing a module's own `__class__` makes `PROBLEM` a property, so the
    object that passes the type check need not be the object that reaches the
    gates. `load` reads it once into a local; this counts the reads.
    """
    honest = tmm_stack.PROBLEM          # unstamped, as a module declares it
    reads = []

    class _CountingModule(types.ModuleType):
        @property
        def PROBLEM(self):
            reads.append(len(reads))
            # after the first read, hand back something else entirely
            return honest if not reads[1:] else replace(
                honest, name="grating_coupler",
                module="invdx.problems.grating_coupler")

    mod = _CountingModule("yourpkg.problems.two_faced")
    monkeypatch.setattr(problems.importlib, "import_module", lambda path: mod)

    spec = problems.load("yourpkg.problems.two_faced")
    assert spec.name == "two_faced", "stamped from the request, as always"
    assert spec.module == "yourpkg.problems.two_faced"
    assert len(reads) == 1, f"PROBLEM was read {len(reads)} times"


# --------------------------------------------------------------------------
# The polarity of the declaration: default-checked, opt out in writing
#
# The three sections above all test a guard that had to be switched ON per
# gate. That is the same shape as the bugs they were written to catch -- a
# rule the next author has to know about and copy -- appearing inside the
# fix for it. These tests pin the reversal: a gate that says nothing is
# asked for provenance, and a gate that owes none says so out loud.
# --------------------------------------------------------------------------


class _ForgotToDeclare:
    """A new gate whose author never heard of `MEASURES_PROBLEM`.

    Nothing about it is malicious and nothing about it is unusual. It loads
    the problem `--problem` asked for, measures two numbers, and reports
    them under their own names -- which is what a gate is for. The only
    thing wrong with it is what is absent, and absence is what nobody
    reviews.
    """

    NAME = "forgot"
    ORDER = 98

    def run(self, cfg, args):
        spec = problems.from_args(args)
        case = spec.reciprocity_case()
        return runner.GateResult(self.NAME, runner.OK,
                                 {"CE_fwd_dB": case.fwd_dB,
                                  "CE_rev_dB": case.rev_dB})


def _opt_in_verify(gate, res, args):
    """`runner._verify_problem_identity` as it stood while the flag was opt-in.

    Reproduced rather than described, for the same reason `_old_stamp` above
    is: a test that asserts the current code is correct proves nothing about
    whether the previous code was wrong. The single line that matters is
    marked; everything else is the shipped logic of that revision, trimmed of
    its error prose.
    """
    gate_name = getattr(gate, "NAME", gate)
    declared = getattr(gate, "MEASURES_PROBLEM", None)
    stamped = {k: res.details[k] for k in runner.ID_KEYS if k in res.details}
    if declared is None and not stamped:
        return                          # <-- the hole: "forgot" lands here
    if declared is None or declared is True:
        name, module = problems.identity_from_args(args)
    else:
        name, module = problems.requested_identity(declared)
    truth = {"problem": name, "problem_module": module}
    missing = sorted(set(runner.ID_KEYS) - set(stamped))
    if missing and declared is not None and res.status != runner.FAIL:
        raise ValueError(f"gate {gate_name!r} carries no {missing}: {truth}")
    wrong = {k: (v, truth[k]) for k, v in stamped.items() if v != truth[k]}
    if wrong:
        raise ValueError(f"gate {gate_name!r} identity mismatch: {wrong}")


def test_red_under_the_old_opt_in_polarity_forgetting_passed_in_silence(
        monkeypatch, capsys):
    """The bug, demonstrated rather than asserted.

    Green report, [ok] on the console, no identity keys in it, and not one
    word anywhere saying so. Note what class of author this catches: the one
    who did not know the rule -- which is the same class the whole layer
    claims to cover.
    """
    monkeypatch.setattr(runner, "_verify_problem_identity", _opt_in_verify)
    monkeypatch.setattr(runner, "discover", lambda: [_ForgotToDeclare()])

    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    out = capsys.readouterr().out
    assert res.status == runner.OK
    assert "problem" not in res.details, "nothing says whose numbers these are"
    assert "problem_module" not in res.details
    assert out.startswith("[ok]"), out
    assert "identity" not in out and "provenance" not in out, out


def test_green_the_same_gate_is_now_asked_where_its_numbers_came_from(
        monkeypatch):
    """Same gate, same fixture, real runner: loud, and with both fixes in it."""
    monkeypatch.setattr(runner, "discover", lambda: [_ForgotToDeclare()])

    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    exc = res.details["exception"]
    assert "problem_module" in exc, "what is missing"
    assert "gate_details" in exc, "how to supply it, if it measures a problem"
    assert "MEASURES_PROBLEM = NoProblem(" in exc, (
        "how to opt out, if it does not -- spelled with its argument, so the "
        "message cannot be satisfied by a constant")


def test_the_gate_is_let_through_by_declaring_the_opt_out(monkeypatch):
    """The escape hatch exists and is one line -- it is just not the default."""

    class _Declared(_ForgotToDeclare):
        MEASURES_PROBLEM = runner.NoProblem(
            "this fixture reports two numbers it made up, from no scene")

    monkeypatch.setattr(runner, "discover", lambda: [_Declared()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.OK, res.details.get("exception", res.details)


def test_the_other_way_out_is_to_stamp_the_identity(monkeypatch):
    """A gate that does measure a problem fixes it by saying which one."""

    class _Stamping(_ForgotToDeclare):
        def run(self, cfg, args):
            spec = problems.from_args(args)
            case = spec.reciprocity_case()
            return runner.GateResult(
                self.NAME, runner.OK,
                runner.gate_details(spec, {"CE_fwd_dB": case.fwd_dB}))

    monkeypatch.setattr(runner, "discover", lambda: [_Stamping()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.OK, res.details.get("exception", res.details)
    assert res.details["problem_module"] == FIXTURE


@pytest.mark.parametrize("mod", [g0, g1, g3, g5])
def test_a_gate_that_measures_no_problem_says_so_in_its_own_module(mod):
    """`vars(mod)`, not `getattr`: the exemption must not be inferable.

    `getattr` would also be satisfied by the runner's default, which is the
    thing being ruled out. Reading the module's own namespace is what makes
    this an assertion about a line someone typed.
    """
    assert "MEASURES_PROBLEM" in vars(mod), (
        f"{mod.__name__} carries no declaration of its own and would be "
        f"held to the default (it measures whatever `--problem` asked for)")
    declared = vars(mod)["MEASURES_PROBLEM"]
    assert isinstance(declared, runner.NoProblem), (
        f"{mod.__name__} must opt out with a reason, not a constant: "
        f"{declared!r}")
    assert declared.reason.strip()


def test_the_four_exempt_gates_do_not_share_one_reason():
    """The point of the reason is that it is about the gate it is typed in.

    Four copies of one sentence would be `MEASURES_PROBLEM = False` again,
    spelled longer: correct in each module only because it says nothing about
    any of them. This asserts they are distinct, and that each names something
    from its own module -- what it measures INSTEAD of a device.
    """
    reasons = {mod.__name__: vars(mod)["MEASURES_PROBLEM"].reason
               for mod in (g0, g1, g3, g5)}
    assert len(set(reasons.values())) == 4, reasons

    # One word each that could not honestly appear in the other three.
    for mod_name, word in [("g0_unit", "pytest"),
                           ("g1_api", "toolchain"),
                           ("g3_physics", "EMPTY cell"),
                           ("g5_crossengine", "slab")]:
        subject = next(r for n, r in reasons.items() if n.endswith(mod_name))
        assert word in subject, (mod_name, word, subject)


@pytest.mark.parametrize("mod", [g0, g1, g3, g5])
def test_the_four_exempt_gates_still_pass_the_backstop(mod):
    """A default-deny flip fails by turning everything red; this is that check.

    Their real `run()` needs a GPU or a meep env, so the result shape they
    return is used directly -- what the backstop reads is `details`, and the
    declaration comes off the module either way.
    """
    res = runner.GateResult(mod.NAME, runner.OK, {"a_number_it_measured": 1.0})
    runner._verify_problem_identity(mod, res, _Args(FIXTURE))     # no raise


def test_an_exempt_gate_that_stamps_an_identity_anyway_is_still_checked():
    """Opting out of being REQUIRED to say is not opting out of being right."""
    res = runner.GateResult(g0.NAME, runner.OK,
                            {"problem": "grating_coupler",
                             "problem_module": problems._REGISTRY["grating_coupler"]})
    with pytest.raises(ValueError, match="does not match"):
        runner._verify_problem_identity(g0, res, _Args(FIXTURE))


@pytest.mark.parametrize("bad", [0, 1, "", "   ", ("tmm_stack",), 1.0])
def test_a_declaration_that_is_none_of_the_three_forms_is_refused(bad):
    """`MEASURES_PROBLEM = 0` must not read as False by truthiness.

    Truthiness is the loose comparison this file refuses everywhere else; a
    declaration nobody can act on is a failure, not a default.
    """
    gate = types.SimpleNamespace(NAME="odd", MEASURES_PROBLEM=bad)
    with pytest.raises(ValueError, match="MEASURES_PROBLEM"):
        runner._declared_problem("odd", gate)


# --------------------------------------------------------------------------
# The opt-out has to carry a reason
#
# The polarity above put "forgot" on the safe side. It did nothing about the
# other way a guard gets switched off, which is copying: `MEASURES_PROBLEM =
# False` was the same three characters in four modules, so it was correct
# wherever it was typed AND wherever it was pasted. These pin the replacement.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_an_opt_out_with_no_reason_is_refused_at_construction(blank):
    """Same rule as `Unsupported`, for the same reason and at the same time.

    At construction, so a gate module that tries it fails at IMPORT -- before
    the runner ever gets a chance to read the declaration as a valid opt-out.
    """
    with pytest.raises(ValueError, match="needs a reason"):
        runner.NoProblem(blank)


def test_the_old_bare_False_is_refused_and_the_message_says_what_to_write():
    """Neither accepted nor ignored: both silences are worse than a failure.

    Accepted, the copyable constant survives the change that was made to
    remove it. Ignored -- falling through to "it measures a problem" -- four
    working gates would fail complaining about missing identity keys, which
    sends the reader looking at `details` instead of at the one line that
    needs editing.
    """
    gate = types.SimpleNamespace(NAME="oldstyle", MEASURES_PROBLEM=False)
    with pytest.raises(ValueError) as exc:
        runner._declared_problem("oldstyle", gate)
    msg = str(exc.value)
    assert "NoProblem(" in msg, "the replacement, spelled out"
    assert "no longer accepted" in msg
    assert "identity" not in msg.split("NoProblem(")[0], (
        "it must not be diagnosed as a provenance problem: the fix is here")


def test_a_new_gate_that_writes_nothing_is_still_asked_where_numbers_came_from(
        monkeypatch):
    """The polarity is unchanged by any of this.

    Requiring a reason from the opt-out would be worth nothing if it also
    made silence an opt-out. `_ForgotToDeclare` declares nothing at all and
    must still fail.
    """
    monkeypatch.setattr(runner, "discover", lambda: [_ForgotToDeclare()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "MEASURES_PROBLEM = NoProblem(" in res.details["exception"]


class _CopiedFromG3:
    """The audit probe: G3's opt-out, pasted into a gate that measures a device.

    Nothing here is invented. `MEASURES_PROBLEM` is read off the real G3
    module, which is what "copy the file header" produces, and the rest is an
    ordinary problem-measuring gate -- it loads what `--problem` asked for,
    reports two coupling efficiencies, and stamps no identity.

    Read what the two tests below do and do not claim. This gate still passes
    the runner's type check after the change, because a reason is a string and
    nothing in this process can know whether it describes the module it was
    typed in. What changed is that the string is now IN the module, next to
    `CE_fwd_dB`, saying there is no device in the scene.
    """

    NAME = "copied"
    ORDER = 99
    MEASURES_PROBLEM = g3.MEASURES_PROBLEM       # literally G3's declaration

    def run(self, cfg, args):
        spec = problems.from_args(args)
        case = spec.reciprocity_case()
        return runner.GateResult(self.NAME, runner.OK,
                                 {"CE_fwd_dB": case.fwd_dB,
                                  "CE_rev_dB": case.rev_dB})


def _pre_noproblem_verify(gate, res, args):
    """`_verify_problem_identity` as it stood while the opt-out was `False`.

    Reproduced rather than described, the same way `_opt_in_verify` above is:
    asserting that today's code is right says nothing about whether the code
    it replaced was wrong. Trimmed of its error prose; the `is False` tests
    are the shape being demonstrated, and the declaration is read with a bare
    `getattr` because that is what the resolver of the day did with it.
    """
    gate_name = getattr(gate, "NAME", gate)
    declared = getattr(gate, "MEASURES_PROBLEM", None)
    if declared is None:
        declared = True                 # absent: held to the default
    stamped = {k: res.details[k] for k in runner.ID_KEYS if k in res.details}
    if declared is False and not stamped:
        return                          # <-- the copied constant lands here
    if declared is True or declared is False:
        name, module = problems.identity_from_args(args)
    else:
        name, module = problems.requested_identity(declared)
    truth = {"problem": name, "problem_module": module}
    missing = sorted(set(runner.ID_KEYS) - set(stamped))
    if missing and declared is not False and res.status != runner.FAIL:
        raise ValueError(f"gate {gate_name!r} carries no {missing}: {truth}")
    wrong = {k: (v, truth[k]) for k, v in stamped.items() if v != truth[k]}
    if wrong:
        raise ValueError(f"gate {gate_name!r} identity mismatch: {wrong}")


def test_red_the_copied_opt_out_used_to_pass_in_total_silence(
        monkeypatch, capsys):
    """The audit finding, reproduced: `[ok]`, no identity, no complaint.

    `False` is put back on the gate to stand in for the pre-change tree,
    because that IS the whole of the pre-change opt-out -- the five lines of
    comment that used to sit around it were never read by anything, which is
    the point. Whoever copied the header got both, and only one of them was
    load-bearing.
    """

    class _Copied(_CopiedFromG3):
        MEASURES_PROBLEM = False        # what G3 declared before the change

    monkeypatch.setattr(runner, "_verify_problem_identity",
                        _pre_noproblem_verify)
    monkeypatch.setattr(runner, "discover", lambda: [_Copied()])

    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    out = capsys.readouterr().out
    assert res.status == runner.OK
    assert not [k for k in runner.ID_KEYS if k in res.details], (
        "two coupling efficiencies, and nothing saying whose")
    assert out.startswith("[ok]"), out
    assert "identity" not in out and "provenance" not in out, out


def test_that_same_copy_is_now_refused_outright_by_the_real_runner(
        monkeypatch):
    """The half of the fix that IS enforced: the old spelling stops working.

    Not because the runner detected a copy -- it cannot -- but because the
    thing that was copyable no longer parses. Anyone carrying `False` forward
    is stopped and told what to write instead.
    """

    class _Copied(_CopiedFromG3):
        MEASURES_PROBLEM = False

    monkeypatch.setattr(runner, "discover", lambda: [_Copied()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "NoProblem(" in res.details["exception"]


def test_green_the_copy_now_carries_a_sentence_that_is_false_where_it_landed(
        monkeypatch):
    """The honest boundary of this fix, asserted rather than claimed in prose.

    It is NOT that the copy is blocked. The gate below is the real audit probe
    -- G3's declaration object, unmodified, on a gate that measures a coupler
    -- and it still passes: the runner accepts any `NoProblem`, and asking it
    to do more would mean asking a string whether it describes its module.

    What the change buys is the second half. The declaration that travelled
    with the copy now says what G3 measures, so the module contains, a few
    lines above `CE_fwd_dB`, the claim that there is no device in the scene.
    That is a review finding a person can make in one read; before, the
    declaration was `False` and said nothing at all.
    """
    monkeypatch.setattr(runner, "discover", lambda: [_CopiedFromG3()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.OK, "the fix makes copying visible, not hard"

    reason = _CopiedFromG3.MEASURES_PROBLEM.reason
    assert reason is vars(g3)["MEASURES_PROBLEM"].reason, "the same object"
    assert "EMPTY cell" in reason and "no device" in reason
    assert "CE_fwd_dB" in res.details, (
        "and this is what it is now sitting next to")


def test_a_str_subclass_declaration_is_refused_with_the_rest():
    """It would carry its own `__eq__` into the comparison made about it."""
    gate = types.SimpleNamespace(
        NAME="liar",
        MEASURES_PROBLEM=lying_name.LyingName(lying_name.FORGED_NAME))
    with pytest.raises(ValueError, match="str SUBCLASS"):
        runner._declared_problem("liar", gate)


# --------------------------------------------------------------------------
# The console line, part two: whitespace was never the whole alphabet
# --------------------------------------------------------------------------

# Erase the current line, move the cursor up one. What follows then lands on
# top of the status line already printed, rather than under it.
CURSOR_UP = "\x1b[2K\x1b[1A"


@pytest.mark.parametrize("payload", [
    CURSOR_UP,                  # ANSI CSI: erase line, cursor up
    "\x08" * 60,                # backspace: same trick within one line
    "\x9b" "1A",                # C1 CSI, the single-byte spelling of the same
    "\r",                       # carriage return: overwrite from column 0
    "\u2028",               # LINE SEPARATOR, a break `split()` does take
    "\u200b",               # zero width space: prints as nothing
    "\u202e",               # right-to-left override: reverses the rest
])
def test_one_line_removes_control_characters_not_only_whitespace(payload):
    """`str.split()` splits on whitespace, and none of these is whitespace.

    The old docstring claimed a string "cannot occupy more than its own
    line". Every payload here breaks that claim in a different way while
    passing `split()` untouched, which is why the claim -- not just the code
    -- had to change.
    """
    flat = runner.one_line(f"the two directions cancel{payload}{FORGED_LINE}")
    assert all(c == " " or c.isprintable() for c in flat), repr(flat)
    assert len(flat.splitlines()) == 1, repr(flat)
    assert "\x1b" not in flat and "\x08" not in flat and "\r" not in flat


# The same alphabet, on the multi-line path. `console_text` keeps newline and
# tab so a traceback stays a traceback, and the first spelling of that keep
# rule was `c.isspace()` -- which is true for \r, \v, \f, \x1c-\x1f, \x85 and
# every Unicode separator. \r was already in the table above, and the function
# that let it through was written in the same change: the attack was known and
# the allowance was written anyway, because "whitespace is layout" reads as
# obviously true. It is not: layout is two characters.
#
# Without this test the keep rule can be reverted to `isspace()` and nothing
# fails -- which was the state this test was added to end.
@pytest.mark.parametrize("payload", [
    "\r",                       # carriage return: overwrite from column 0
    "\v",                       # vertical tab
    "\f",                       # form feed: page break on some terminals
    "\x1c", "\x1d", "\x1e", "\x1f",   # file/group/record/unit separators
    "\x85",                     # NEL, the C1 newline
    " ", " ",         # LINE / PARAGRAPH SEPARATOR
    " ", "　",         # no-break and ideographic space: pad to align
    CURSOR_UP,
    "\x08" * 60,
])
def test_console_text_keeps_layout_not_every_whitespace(payload):
    flat = runner.console_text(
        f"the two directions cancel{payload}{FORGED_LINE}")
    assert all(c in "\n\t" or c.isprintable() for c in flat), repr(flat)
    # one line in, one line out: none of these may split it
    assert len(flat.splitlines()) == 1, repr(flat)


def test_console_text_leaves_a_traceback_readable():
    """The reason the keep list is not empty. Losing these would trade a
    forged status line for an unreadable diagnosis, which is not a trade."""
    tb = ('Traceback (most recent call last):\n'
          '  File "g4.py", line 60, in run\n'
          '\traise RuntimeError("the two directions cancel")\n'
          'RuntimeError: the two directions cancel')
    assert runner.console_text(tb) == tb


def test_an_escape_code_reason_cannot_rewrite_the_line_above_it(
        monkeypatch, capsys):
    """Strictly worse than the newline case, and it survived that fix.

    A newline adds a fabricated line under the real one. `\\x1b[2K\\x1b[1A`
    REPLACES the real one -- the reader never sees the [n/a] at all, only a
    green [ok] where it used to be.
    """
    spec = replace(problems.load(FIXTURE),
                   reciprocity_case=Unsupported(
                       f"the two directions cancel{CURSOR_UP}{FORGED_LINE}"))
    monkeypatch.setattr(problems, "load", lambda name: spec)
    monkeypatch.setattr(runner, "discover", lambda: [g4])

    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    out = capsys.readouterr().out
    assert res.status == runner.NOT_APPLICABLE
    assert "\x1b" not in out and "\x08" not in out, repr(out)
    assert len([ln for ln in out.splitlines() if ln.strip()]) == 1, repr(out)
    assert out.startswith("[n/a]"), repr(out)
    assert CURSOR_UP in res.details["reason"], "the JSON side keeps the text"


# --------------------------------------------------------------------------
# Keys and values that answer the questions asked about them
#
# Same diagnosis as `lying_name`, one layer out: the reserved-name check and
# the identity comparison both run ON data the checked party supplied, and
# `str` subclasses get to answer `__hash__`, `__eq__` and `__ne__`.
# --------------------------------------------------------------------------


class _HashDodgingKey(str):
    """A key that spells `problem` and is not `"problem"` to any dict or set.

    Membership (`k in reserved`, `k in details`) consults `__hash__` first
    and `__eq__` only on a bucket collision, so a key that hashes elsewhere
    is never compared with the name it is spelling. `json.dump` then calls
    `str()` on it and writes the name out in full.
    """

    def __hash__(self):
        return str.__hash__("nothing the guard is looking for")

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True


def test_red_a_dodging_key_is_invisible_to_a_membership_check_and_visible_to_json():
    """Both halves of the damage, without any invdx code in the picture."""
    k = _HashDodgingKey("problem")
    assert k not in runner.RESERVED_DETAIL_KEYS, "the guard it walks past"
    assert k not in {"problem": "x"}, "and the same for the identity lookup"

    blob = json.dumps({k: "grating_coupler", "problem": "tmm_stack"})
    assert blob.count('"problem"') == 2, blob
    assert json.loads(blob) == {"problem": "tmm_stack"}, (
        "one parser's answer; a first-wins parser reads the forgery instead")


def test_a_dodging_key_is_refused_by_the_merge():
    """The documented path a problem's dict travels."""
    with pytest.raises(ValueError, match="not str"):
        runner.merge_problem_dict(
            "a ReciprocityCase's extra",
            {_HashDodgingKey("problem"): lying_name.FORGED_NAME},
            {"CE_fwd_dB": -9.768209911027565})


def test_a_dodging_key_nested_one_level_down_is_refused_too():
    """Nested dicts are serialized by the same `json.dump`."""
    with pytest.raises(ValueError, match="inside 'sampling'"):
        runner.merge_problem_dict(
            "a GradcheckCase's info",
            {"sampling": {_HashDodgingKey("problem"): lying_name.FORGED_NAME}},
            {"grad_max": 1.0})


def test_a_dodging_key_is_refused_by_the_backstop_too(monkeypatch):
    """A gate that never called the merge is still caught before the report."""

    class _DodgingGate:
        NAME = "hash_dodger"
        ORDER = 94
        MEASURES_PROBLEM = True

        def run(self, cfg, args):
            return runner.GateResult(self.NAME, runner.OK, {
                _HashDodgingKey("problem"): lying_name.FORGED_NAME,
                _HashDodgingKey("problem_module"): lying_name.FORGED_MODULE,
                "CE_fwd_dB": -9.768209911027565})

    monkeypatch.setattr(runner, "discover", lambda: [_DodgingGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "not str" in res.details["exception"]


def test_red_the_identity_comparison_is_answered_by_the_value_it_checks():
    """`v != truth[k]` on a `str` subclass is that subclass's opinion."""
    forged = lying_name.LyingName(lying_name.FORGED_NAME)
    assert not (forged != "tmm_stack"), "what the backstop's `!=` returns"
    assert str.__ne__(forged, "tmm_stack") is True, "what it actually holds"


def test_a_gate_stamping_a_str_subclass_identity_is_refused(monkeypatch):
    """Lower severity -- it takes a lying GATE, not a lying problem.

    Worth closing anyway: it is the same diagnosis as `lying_name`, and the
    fix is the same one line. A gate is trusted with its own numbers because
    nothing can second-guess them; identity is the one field that CAN be
    second-guessed, so it should not be given away to the value.
    """

    class _AgreeableGate:
        NAME = "agreeable"
        ORDER = 95
        MEASURES_PROBLEM = True

        def run(self, cfg, args):
            return runner.GateResult(self.NAME, runner.OK, {
                "problem": lying_name.LyingName(lying_name.FORGED_NAME),
                "problem_module": lying_name.LyingName(
                    lying_name.FORGED_MODULE),
                "CE_fwd_dB": -9.768209911027565})

    monkeypatch.setattr(runner, "discover", lambda: [_AgreeableGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    exc = res.details["exception"]
    assert "not a str" in exc
    assert lying_name.FORGED_NAME in exc, "what it was actually carrying"


def test_a_dodging_key_inside_a_list_of_dicts_is_refused(monkeypatch):
    """`json.dump` walks lists, so the key check has to walk them too.

    Found by asking the first version of the check the same question it asks
    of everyone else: under what input is it false? It recursed through
    dicts only, so one `[{...}]` between `details` and the forged key was
    enough to walk past it.
    """
    with pytest.raises(ValueError, match=r"inside 'cases'\[1\]"):
        runner.merge_problem_dict(
            "a GradcheckCase's info",
            {"cases": [{"ok": 1}, {_HashDodgingKey("problem"): "x"}]},
            {"grad_max": 1.0})


def test_the_key_check_terminates_on_a_details_dict_that_contains_itself():
    """A cycle must be reported, not spun on.

    `json.dump` raises on one too, but only after this check has already
    said the report is clean -- and by then the report is what is being
    written.
    """
    loop = {"self": None}
    loop["self"] = loop
    runner._exact_str_keys("a case's extra", loop)         # returns, no hang

    loop[_HashDodgingKey("problem")] = "grating_coupler"
    with pytest.raises(ValueError, match="not str"):
        runner._exact_str_keys("a case's extra", loop)


def test_the_failure_dump_cannot_carry_escape_codes_to_the_console(
        monkeypatch, capsys):
    """The other place a problem's own text reaches a terminal.

    A traceback's last line is the raising exception's message. G2 Part C and
    G4 call into problem code, so a problem that raises can put whatever it
    likes there -- and that dump is printed on the one path where a reader is
    looking hardest.
    """

    class _RaisingGate:
        NAME = "raiser"
        ORDER = 93
        MEASURES_PROBLEM = runner.NoProblem("this fixture never gets as far "
                                            "as building a scene")

        def run(self, cfg, args):
            raise RuntimeError(
                f"the design grid is empty{CURSOR_UP}{FORGED_LINE}")

    monkeypatch.setattr(runner, "discover", lambda: [_RaisingGate()])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    out = capsys.readouterr().out

    assert res.status == runner.FAIL
    assert "\x1b" not in out and "\x08" not in out, repr(out)
    assert "the design grid is empty" in out, "the real diagnosis survives"
    assert out.count("\n") > 3, "and it is still a multi-line traceback"
    assert CURSOR_UP in res.details["exception"], "the JSON side keeps it"


class _KeyHidingDict(dict):
    """A `details` whose `__iter__` and `keys` under-report its own contents.

    The exact shape `merge_problem_dict` was already fixed for, arriving from
    the other side: there the PROBLEM handed over a mapping that answered two
    questions differently, here the GATE returns one. `json.dump` takes the
    C fast path for anything that is a `dict` -- subclass included -- so it
    writes the underlying storage regardless of what these two methods say.
    """

    def __iter__(self):
        return iter([k for k in dict.keys(self) if type(k) is str])

    def keys(self):
        return [k for k in dict.keys(self) if type(k) is str]


def test_a_details_dict_that_under_reports_its_own_keys_is_still_checked(
        monkeypatch):
    """Found by probing the fix above with the question it asks of others.

    Reading the check's keys with `iter(node)` while `json.dump` reads the
    real storage is two sources for one fact -- and the keys only the writer
    sees are exactly the ones worth hiding.
    """
    hidden = _HashDodgingKey("problem")

    class _HidingGate:
        NAME = "hider"
        ORDER = 91
        MEASURES_PROBLEM = True

        def run(self, cfg, args):
            spec = problems.load(FIXTURE)
            d = _KeyHidingDict(runner.gate_details(spec, {"CE_fwd_dB": -9.7}))
            dict.__setitem__(d, hidden, lying_name.FORGED_NAME)
            return runner.GateResult(self.NAME, runner.OK, d)

    gate = _HidingGate()
    leaked = json.dumps(gate.run(BaseConfig(), _Args(FIXTURE)).details,
                        default=str)
    assert leaked.count('"problem"') == 2, (
        "red: the writer sees the forged key even though the mapping denies "
        "it")

    monkeypatch.setattr(runner, "discover", lambda: [gate])
    res, = runner.run_gates(BaseConfig(), _Args(FIXTURE))
    assert res.status == runner.FAIL
    assert "not str" in res.details["exception"]
