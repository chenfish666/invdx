"""An out-of-tree problem that claims, in writing, to be `grating_coupler`.

Not a hypothetical. Back when `name` was hand-written and nothing compared it
to the name `load` had been asked for, running G2 Part C against this module
by its dotted path produced a `gates_report.json` carrying
`grating_coupler_f0`, `grating_coupler_fd_checks`, `grating_coupler_sampling`
and `details["problem"] = "grating_coupler"` -- the same keys, the same
shape, the same file as the real coupler's report, measured on a 16-layer
dielectric film.

The physics here is `tmm_stack`'s, reused unmodified, precisely so that the
only thing wrong with this module is its name. It exists to be REFUSED:
`problems.load` compares the declared name against the one it was asked for.
"""

from invdx.problems.contract import ProblemSpec

from . import tmm_stack

PROBLEM = ProblemSpec(
    name="grating_coupler",          # the lie under test
    config_cls=tmm_stack.TMMStackConfig,
    gradcheck_case=tmm_stack.gradcheck_case,
    reciprocity_case=tmm_stack.reciprocity_case,
)
