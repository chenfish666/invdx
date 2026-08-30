"""Problem modules that exist only to test the problem contract itself.

They live under tests/ rather than src/invdx/problems/ on purpose: a toolbox
that ships a third "problem" nobody designs anything with is a toolbox with a
fake component in it. What is real here is the contract they exercise, and
that belongs next to the tests that exercise it.

They are still reachable from the command line, because `problems.load`
accepts a dotted module path:

    PYTHONPATH=tests uv run python scripts/00_check.py \\
        --only reciprocity --problem fixture_problems.tmm_stack

which is also the check that an out-of-tree problem -- someone else's, in
their own package -- can be gated without being vendored into this repo.
"""
