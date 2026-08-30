"""A problem whose NAME is not a string but an object that answers like one.

The fourth impersonation attempt, and the first that does not go through any
dict a problem is invited to fill in. `impostor` declares another problem's
name; `grating_coupler` claims one by being spelled that way; `self_naming`
writes the report's identity keys from inside `ReciprocityCase.extra`. This
one puts the forgery in `ProblemSpec.name` and `ProblemSpec.module`
themselves, as a `str` SUBCLASS -- so it is a string by every test the
loader used to perform, while containing something else.

`ProblemSpec` annotates both fields `str`, and an annotation is not a check.
The two questions the loader asks about them were:

    str(spec.name).strip()          "did this module name itself?"
    spec.name == requested          "does that name match the one we loaded
                                     under?"

Both run ON the value, so the value answers them. `LyingName.__str__`
returns "" (so the first question hears "no name declared, nothing to
refuse") and `LyingName.__eq__` returns True (so the second hears "already
correct, nothing to overwrite"). The spec then reached the gates unchanged,
carrying `grating_coupler` in the field a report is attributed by --
`json.dump` writes a str subclass out as its actual characters.

An audit of this tree took that to its conclusion: a pure-CPU transfer-matrix
problem with a `time.sleep(91)` in it produced a `gates_report.json` with the
same md5 as a real 91.05 s GPU run of the shipped grating coupler.

`PROBLEM` below is assembled with `object.__new__` plus `object.__setattr__`
rather than by calling `ProblemSpec(...)`, because calling it now raises: the
type check in `__post_init__` refuses a name that is not exactly `str`. That
bypass is not a fifth attack so much as the reason `problems._stamp` repeats
the same check on every load instead of trusting that `__init__` ran.

Loading this module is expected to fail. What it is FOR is the pair of tests
that show why: one with the check disabled, where the forgery goes through
and the report is indistinguishable; one with it in place, where the refusal
says the word `str` -- because the value prints as an ordinary name and a
message about names would send the reader looking for a typo.
"""

from invdx.problems import ProblemSpec

from . import tmm_stack

HONEST_NAME = "lying_name"
HONEST_MODULE = "fixture_problems.lying_name"
FORGED_NAME = "grating_coupler"
FORGED_MODULE = "invdx.problems.grating_coupler"


class LyingName(str):
    """A `str` holding `claimed` while answering as if it held nothing.

    Every override here corresponds to one line of the loader that used to
    interrogate this object instead of interrogating the request.
    """

    def __new__(cls, claimed):
        return super().__new__(cls, claimed)

    def __str__(self):
        return ""                       # "I declare no name of my own"

    def __eq__(self, other):
        return True                     # "whatever you loaded me as, I match"

    def __ne__(self, other):
        return False

    def __hash__(self):
        return str.__hash__(self)


def _built_without_init(**fields):
    """A frozen dataclass assembled without running `__init__`.

    `dataclasses.dataclass(frozen=True)` blocks attribute assignment through
    `__setattr__`; `object.__setattr__` is not blocked, and `object.__new__`
    never calls `__init__`, so `__post_init__` does not run. The resulting
    object's `type(...)` is exactly `ProblemSpec`.
    """
    spec = object.__new__(ProblemSpec)
    for key, value in fields.items():
        object.__setattr__(spec, key, value)
    return spec


PROBLEM = _built_without_init(
    config_cls=tmm_stack.TMMStackConfig,
    gradcheck_case=tmm_stack.gradcheck_case,
    reciprocity_case=tmm_stack.reciprocity_case,
    name=LyingName(FORGED_NAME),
    module=LyingName(FORGED_MODULE),
)
