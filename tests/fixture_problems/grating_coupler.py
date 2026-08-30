"""An out-of-tree problem that claims `grating_coupler`'s name by FILENAME.

The quieter half of `impostor.py`. That module writes the lie down --
`name="grating_coupler"` -- and is refused for saying so. This one says
nothing at all: it declares no name, and relies on `load` naming a dotted
path after its last segment. Called `fixture_problems.grating_coupler`, it
was stamped `grating_coupler` and its numbers went into
`gates_report.json` under `details["problem"] = "grating_coupler"`,
`grating_coupler_f0`, `grating_coupler_fd_checks`, `grating_coupler_sampling`
-- the real coupler's keys, on a 16-layer dielectric film, with nothing in the
report to tell them apart.

So the check on the declared name was necessary and not sufficient: a name
can be claimed by how a file is spelled. `problems._reject_registry_impersonation`
is what closes it, and it refuses this module BEFORE importing it. The physics
is `tmm_stack`'s, reused unmodified, so that the filename is again the only
thing wrong here.
"""

from invdx.problems.contract import ProblemSpec

from . import tmm_stack

PROBLEM = ProblemSpec(
    # no name= : the claim is made by the file's name, not by this line
    config_cls=tmm_stack.TMMStackConfig,
    gradcheck_case=tmm_stack.gradcheck_case,
    reciprocity_case=tmm_stack.reciprocity_case,
)
