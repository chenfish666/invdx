"""A problem that fills in its own identity through the free-form dict.

The third impersonation attempt, and the only one that never states a claim.
`impostor` declares `name="grating_coupler"`; `grating_coupler` claims the
name by being spelled that way. This module does neither -- honest filename,
no `name=`, no `module=`, so `problems.load` stamps it truthfully as
`self_naming` / `fixture_problems.self_naming`. It then overwrites both of
those stamps from inside `ReciprocityCase.extra` and `GradcheckCase.info`,
which are the problem's own dicts and used to be merged into the gate's
`details` LAST.

Its physics is `tmm_stack`'s, unaltered, so that what it produced was a
complete and passing report. An audit of this tree ran the equivalent module
through `scripts/00_check.py --only reciprocity` and got `[ok]` with a
`gates_report.json` whose `details` matched a real GPU run of the shipped
grating coupler on every key except `seconds`:

    "problem": "grating_coupler",
    "problem_module": "invdx.problems.grating_coupler"

Nothing in that file said `fixture_problems.self_naming`, and no gate had to
be touched to get there. The keys below are therefore not decoration: they
are the exact two fields a report is attributed by, which is why no gate may
let a problem supply them. See `gates/runner.py`.
"""

from invdx.problems import GradcheckCase, ProblemSpec, ReciprocityCase

from . import tmm_stack

# What the report should have said, and what this module writes instead.
HONEST_NAME = "self_naming"
HONEST_MODULE = "fixture_problems.self_naming"
FORGED_NAME = "grating_coupler"
FORGED_MODULE = "invdx.problems.grating_coupler"

FORGERY = {"problem": FORGED_NAME, "problem_module": FORGED_MODULE}


def reciprocity_case():
    case = tmm_stack.reciprocity_case()
    return ReciprocityCase(fwd_dB=case.fwd_dB, rev_dB=case.rev_dB,
                           extra={**case.extra, **FORGERY})


def gradcheck_case():
    case = tmm_stack.gradcheck_case()
    return GradcheckCase(vg_fn=case.vg_fn, value_fn=case.value_fn,
                         base=case.base, beta=case.beta, seed=case.seed,
                         info={**case.info, **FORGERY})


PROBLEM = ProblemSpec(
    config_cls=tmm_stack.TMMStackConfig,
    gradcheck_case=gradcheck_case,
    reciprocity_case=reciprocity_case,
)
