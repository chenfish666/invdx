"""Problem modules that exist only to test the problem contract itself.

They live under tests/ rather than src/invdx/problems/ on purpose: a toolbox
that ships a third "problem" nobody designs anything with is a toolbox with a
fake component in it. What is real here is the contract they exercise, and
that belongs next to the tests that exercise it.

Four of them exist to be REFUSED rather than loaded -- `impostor`, which
writes another problem's name down; `grating_coupler`, which claims one by
being spelled that way; `self_naming`, which claims nothing anywhere and
instead writes `details["problem"]` and `details["problem_module"]` from
inside the free-form dict a problem hands its gate; and `lying_name`, which
puts a `str` SUBCLASS in `ProblemSpec.name` so that the field answers the
loader's questions itself. Each says in its own docstring what a report
carrying its numbers looked like before it was refused.

They are still reachable from the command line, because `problems.load`
accepts a dotted module path:

    PYTHONPATH=tests uv run python scripts/00_check.py \\
        --only reciprocity --problem fixture_problems.tmm_stack

which is also the check that an out-of-tree problem -- someone else's, in
their own package -- can be gated without being vendored into this repo.
"""
