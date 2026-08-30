"""Gate registry and ordered execution with [ok]/[FAIL] console output.

Five statuses, and the two added for problem-parameterized gates exist so
that "this problem declared the check inapplicable" can never be read as
either a pass or a breakage:

    ok      everything the gate checks ran and passed
    fail    something ran and failed, OR something that should have run could
            not — including a problem that declares no answer at all, whose
            import raises and lands here
    n/a     the problem explicitly declared this gate inapplicable, with a
            reason, which is printed on the same line
    part    the gate's problem-independent half ran and passed; the
            problem-specific half was declared inapplicable, with a reason
    skip    a precondition of the whole gate is absent (rare — see
            gates/__init__.py on why this is not the answer to a missing
            problem capability)

`n/a` and `part` are never failures, so they do not stop the run and do not
change the exit code. They are also never silent: `not_applicable` refuses to
build a result without a reason, and the reason is what the console line
carries.

This file also holds the rule about who may write which key of `details` --
`gate_details`, `merge_problem_dict` and `RESERVED_DETAIL_KEYS` below, plus
the backstop in `run_gates` that catches a gate which did not use them. The
rule is about the shape of `GateResult.details`, and `details` is defined
here, which is why it is not in `problems/contract.py` next to the dicts a
problem fills in.

WHAT THAT IS AND IS NOT. Provenance here is a record, not a security
boundary, and every claim in this package should be read with that clause
attached. Loading a problem imports it; an imported module runs in this
process, with `invdx.gates` in reach. It can replace `gate_details`, replace
`_verify_problem_identity`, or write `gates_report.json` itself without
simulating anything, and an audit of this repo did the equivalent: a
pure-CPU stand-in with `time.sleep(91)` in it produced a report whose md5
matched a real 91.05 s GPU run. Nothing below detects that, and nothing that
runs in the same process could.

What the checks below DO catch is the whole class of failure that arrives
without an author intending it, which is the class that actually happens: a
gate built its `details` as a dict literal and let the problem's dict land on
top (that was G4, shipped); a problem's `extra` used the same key name as
something the gate measured; a module was copied and kept the name it was
copied from; a file was renamed into a registered problem's spelling; a new
gate reported numbers and never said whose. Each of those produced a green,
well-formed, completely unattributable report, and each is now a loud
failure naming the key and the fix. Those checks are worth having for that
reason alone -- and for no other one. See the README for the same statement
where a reader of a report will meet it.
"""

import importlib
import json
import os
import pkgutil
import time
import traceback
from dataclasses import dataclass, field, asdict


OK = "ok"
FAIL = "fail"
SKIP = "skip"
NOT_APPLICABLE = "n/a"
PARTIAL = "part"

# Fixed width so a column of them stays a column.
LABELS = {OK: "[ok]  ", FAIL: "[FAIL]", SKIP: "[skip]",
          NOT_APPLICABLE: "[n/a] ", PARTIAL: "[part]"}


@dataclass
class GateResult:
    name: str
    status: str                      # one of runner.LABELS
    details: dict = field(default_factory=dict)


# Keys of `details` that no problem may supply, in any gate, ever.
#
#   problem, problem_module  identify the module being MEASURED. A subject
#                            that fills in its own identity has not been
#                            identified; that is the whole reason these two
#                            fields exist, so they are the two that must
#                            never be problem-writable.
#   seconds                  the runner's own stopwatch, written after the
#                            gate returns -- a supplied value would be
#                            overwritten, i.e. silently ignored.
#   reason, exception        what the console line and the failure dump
#                            print. A problem that could set `reason` could
#                            write the text next to another problem's [ok].
#
# Applied to every problem-supplied dict, including ones that end up nested
# (G2's sampling info), because a nested dict is one refactor away from being
# spliced into `details` and because none of these names means anything
# useful inside a problem's own numbers anyway.
RESERVED_DETAIL_KEYS = frozenset(
    {"problem", "problem_module", "seconds", "reason", "exception"})

# The two of those that say WHICH module a report describes. Kept as an
# ordered pair rather than spelled out at each use, because the backstop now
# asks three questions about them (present? both? right?) and three spellings
# of the same pair is how one of them ends up checking only `problem`.
ID_KEYS = ("problem", "problem_module")


def _exact_str_keys(source, mapping):
    """Refuse a `details` key that is anything but an exact `str`, at any depth.

    Every guard over `details` -- the reserved-name check below, the identity
    lookup in `_verify_problem_identity` -- is a `in`/`==` against a key that
    the checked party supplied. Both of those go through `__hash__` and
    `__eq__`, which a `str` SUBCLASS overrides. A key that hashes to something
    of its own is not equal to `"problem"` as far as any set or dict is
    concerned, so it walks past a guard written to catch exactly that name --
    and then `json.dump` calls `str()` on it and writes `"problem"` anyway.
    The result is a report with the key twice: a duplicate-key JSON object
    whose first copy is the forged one, which most parsers resolve by
    last-wins and some by first-wins, i.e. the file no longer has one meaning.

    Same diagnosis as the `ProblemSpec` identity fields, so the same fix: the
    check cannot be run on a value that gets to answer it. `type(k) is str`
    asks the interpreter, not the object. Nothing legitimate loses anything --
    a key in a JSON report is a string in the end regardless.

    It walks the whole structure `json.dump` will walk -- dicts nested in
    dicts, and dicts nested in lists and tuples -- for the same reason the
    reserved-name rule is applied to nested dicts: a nested one is one
    refactor away from being spliced into `details`, and a forged key inside
    a list is written into the report by the same serializer. Stopping at the
    top level would be a guard whose scope a reader has to know.

    `seen` is by object identity, so a dict that refers to ITSELF does not
    recurse forever here. It is not reported either: the cycle is skipped and
    the check passes, because a cycle is not a forged key and this function
    answers one question. Where it then fails depends on the result's status,
    and neither place is this one -- so do not read a pass here as "the report
    will serialise". On a non-FAIL result, `write_report` raises
    `RecursionError` out of `dataclasses.asdict` and leaves a zero-byte
    `gates_report.json`. On a `[FAIL]`, `run_gates` gets there first: its
    console dump passes `json.dumps(res.details, ...)` as the *default
    argument* of a `.get()`, which Python evaluates whether or not the key is
    present, so the `ValueError` for a circular reference fires before the
    report is written at all.

    `seen` covers dicts only. A cycle that runs through a list or a tuple --
    which `merge_problem_dict` will happily accept from a problem's `extra` --
    makes THIS function raise `RecursionError`, caught by `run_gates` and
    turned into a `[FAIL]`. Still loud, still not here, and the report never
    appears. Giving `_nested_dicts` its own `seen` would move that failure
    back into this function's own message; it has not been done, so this
    paragraph is the map.

    Keys are read with `dict.keys`, not `iter(node)`, and values with
    `dict.__getitem__`. This is the `merge_problem_dict` lesson applied to
    the other end: a check and a writer that ask a mapping two different
    questions disagree about what is in it, and the keys only the WRITER
    sees are the ones that matter. `json.dump` takes the C fast path for
    anything `PyDict_Check` accepts -- including a `dict` subclass -- so it
    reads the underlying storage no matter what `__iter__` or `keys` were
    overridden to say. Reading that same storage is what makes this check
    cover what actually gets written.
    """
    seen = {id(mapping)}
    stack = [(mapping, "")]
    while stack:
        node, path = stack.pop()
        is_dict = isinstance(node, dict)
        for k in (list(dict.keys(node)) if is_dict else list(node)):
            if type(k) is not str:
                raise ValueError(
                    f"{source} supplies a details key {k!r}{path} of type "
                    f"{type(k).__name__}, not str.\n"
                    f"  Keys are checked by name -- against the reserved set, "
                    f"and against the two identity keys -- and both checks go "
                    f"through the key's own __hash__/__eq__. A str subclass "
                    f"therefore answers the question being asked about it, "
                    f"walks past the guard, and is still written out as its "
                    f"str() value, producing a report with the same key "
                    f"twice.\n"
                    f"  Fix: use a plain str. If the key is computed, wrap "
                    f"it: str(k).")
            v = dict.__getitem__(node, k) if is_dict else node[k]
            for child, where in _nested_dicts(v, f"{path} inside {k!r}"):
                if id(child) not in seen:
                    seen.add(id(child))
                    stack.append((child, where))


def _nested_dicts(value, path):
    """Every dict reachable from `value` through dicts, lists and tuples."""
    if isinstance(value, dict):
        yield value, path
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _nested_dicts(item, f"{path}[{i}]")


def merge_problem_dict(source, supplied, owned):
    """Merge a problem-supplied dict under gate-owned keys, or refuse.

    `owned` is what the GATE measured; `supplied` is what the PROBLEM handed
    over (`ReciprocityCase.extra`, `GradcheckCase.info`, ...); `source` names
    that container in the error, since the fix is made in the problem module.

    A shared name is refused rather than resolved. Whichever side won a
    silent merge, the report would still parse and would still carry every
    expected key -- the value would simply belong to the other author, which
    is unfalsifiable from the report alone. Raising turns it into a loud
    failure with the colliding name in it.

    `supplied` is COPIED into a plain dict before anything reads it, and the
    copy is what is both checked and merged. Before, the collision check
    iterated `supplied` while the merge called `dict(supplied)` -- two
    different ways of asking a mapping what is in it. `dict()` prefers
    `keys()`; iteration uses `__iter__`. A `Mapping` whose two answers
    disagree therefore had keys that the check never saw and the merge did,
    which is how `reason` and `exception` -- the strings that end up on the
    console line next to an `[ok]` -- could be delivered past a guard written
    to refuse exactly those. One source of keys, read once.

    Scope, so nobody reads more into this than it does: it stops a problem
    from writing gate-owned keys through the documented data path, which is
    the way a well-meaning problem does it by accident and the way a report
    gets quietly mislabelled. It is not a sandbox. Loading a problem imports
    it, and an imported module runs code that could reach into this package
    directly; provenance here is a record, not a security boundary.

    The keys are type-checked before their names are compared
    (`_exact_str_keys`), because comparing names is a set membership test and
    a `str` subclass gets to answer it.
    """
    supplied = dict(supplied)
    _exact_str_keys(source, supplied)
    reserved = RESERVED_DETAIL_KEYS | set(owned)
    clash = sorted(k for k in supplied if k in reserved)
    if clash:
        raise ValueError(
            f"{source} supplies key(s) {clash}, which the gate writes itself. "
            f"Rename them in the problem: a merge would file the problem's "
            f"value under the gate's name, and nothing downstream could tell. "
            f"Written by the gate here: {sorted(owned)}; never problem-"
            f"writable in any gate: {sorted(RESERVED_DETAIL_KEYS)}.")
    # Gate keys applied last as well as checked first: if the check above
    # ever has a hole, the gate's own measurement still wins the merge.
    merged = dict(supplied)
    merged.update(owned)
    return merged


def gate_details(spec, measured, supplied=None, source=None):
    """`details` for a gate that measured a problem: identity stamped here.

    A gate passes what IT measured and, if the problem handed it a free-form
    dict, that dict too. The two identity keys are not arguments: they are
    read off the loaded `spec`, so a gate cannot report the wrong ones and a
    new gate does not have to know they exist. `run_gates` re-checks them
    against the loader afterwards, so a gate that skips this function is
    caught rather than trusted.
    """
    owned = dict(measured)
    owned["problem"] = spec.name
    owned["problem_module"] = spec.module
    # `is None`, not `or {}`: `or` asks the supplied mapping whether it is
    # truthy, and a mapping that answers False for its own contents would
    # have its keys dropped here without anything saying so.
    supplied = {} if supplied is None else supplied
    return merge_problem_dict(source or f"{spec.name}'s gate case",
                              supplied, owned)


@dataclass(frozen=True)
class NoProblem:
    """A gate's declaration that it measures no problem, and why not.

    Same shape, and the same argument, as a problem's `Unsupported(reason)`:
    the two are the opt-outs of this package, one written by a problem about a
    gate and one by a gate about problems in general, so they read alike on
    purpose.

    WHY A REASON. The opt-out used to be the bare constant
    `MEASURES_PROBLEM = False`, whose polarity was right -- writing nothing
    lands on the checked side -- but whose CONTENT was the same five words in
    four modules. An audit copied G3's declaration, comment block and all, into
    a new gate that really did measure a coupler; the gate reported `CE_fwd_dB`
    and `CE_rev_dB`, stamped no identity, and printed `[ok]` without a word.
    Nothing in `False` could have objected, because `False` says nothing about
    the gate it is written in: the same three characters are correct in every
    module that measures no device and wrong in every module that measures one,
    and no reader can tell which they are looking at.

    A reason cannot be neutral in that way. It names THIS gate's subject --
    an empty cell, the installed toolchain, a slab the gate hard-codes -- so
    carrying it into a gate that measures a device puts a visibly false
    sentence next to the numbers. That is the whole of the claim being made
    here, and it is worth stating its size: this makes a copied opt-out
    READABLE as wrong, in review and in the module itself. It does not make
    copying impossible, and nothing in this process could -- a reason is a
    string, and the interpreter cannot know whether it describes the module it
    was typed in. Read `runner.py`'s header for the same boundary drawn around
    the rest of this layer.

    The reason is checked at construction, exactly like `Unsupported`, so an
    empty one is a failure at import of the gate module rather than a silent
    exemption discovered later.
    """

    reason: str

    def __post_init__(self):
        if not str(self.reason).strip():
            raise ValueError(
                "NoProblem(reason=...) needs a reason: it is the only thing "
                "that distinguishes a gate which owes no provenance from a "
                "gate whose author copied the opt-out from next door. Say "
                "what THIS gate measures instead of a problem -- an empty "
                "cell, the installed toolchain, a device the gate hard-codes.")


def _declared_problem(gate_name, gate):
    """Resolve a gate's `MEASURES_PROBLEM`, defaulting to "it measures one".

        absent      the same as True. See below -- this is the whole design.
        True        the gate measures whatever `--problem` asked for -- G2, G4
                    and anything else parameterized the ordinary way.
        NoProblem(  the gate measures no problem (G0/G1/G3/G5), and says what
          reason)   it measures instead. Nothing is required; a stamped
                    identity is still checked, because a gate that stamps one
                    has made a claim. Returned as the instance, so the caller
                    can tell this apart from `True` -- and so that `False`,
                    which this used to be, is not something the caller could
                    still be handed.
        "<name>"    the gate always measures this particular problem, whatever
                    `--problem` says. A cross-problem gate is honest and used
                    to be failed here for it, with a message accusing it of
                    letting a problem name itself.

    `False` is refused, loudly, with the replacement spelled out. It was the
    opt-out until `NoProblem` replaced it, so during any upgrade it is a
    declaration a person genuinely believes in -- and both silent readings of
    it are worse than a failure. Accepting it keeps the copyable constant
    alive; ignoring it (falling through to "measures a problem") would fail
    four working gates with a message about missing identity keys, which is a
    diagnosis pointing away from the actual edit.

    WHY ABSENT MEANS True. This flag used to be opt-IN: a gate said
    `MEASURES_PROBLEM = True` to be checked, and a gate that said nothing was
    read as measuring no problem. So the failure this backstop exists for --
    a gate author who does not know the rule -- was also the way to switch
    the backstop off. A new gate that measured a problem, reported numbers
    and stamped no identity produced a green, unattributable report and not
    one word of complaint, which is the exact shape of the bug the layer was
    written to catch, reappearing inside the catcher. A guard that has to be
    named at each site guarantees the next site forgets it.

    Reversed, "forgot" lands on the safe side. A new gate that declares
    nothing is asked for provenance: loud, and a one-line fix either way,
    because the author either stamps the identity or writes a
    `NoProblem(...)`. And if someone deletes the declaration from G0/G1/G3/G5,
    those gates fail loudly rather than silently -- a false alarm, which is the
    direction an error should point.

    That polarity was never the whole problem, though, and the second half is
    what `NoProblem` is for: writing nothing was loud, but writing the same
    `False` as the module next door was silent, and copying is what people
    actually do. `NoProblem`'s docstring has the audit that found it.

    Anything else raises. A value that is not exactly `True`, a `NoProblem` or
    a non-empty exact `str` is a declaration nobody can act on -- and a `str`
    subclass in particular would carry its own `__eq__` into the comparison
    the caller then makes, which is the move this file refuses everywhere
    else it appears.
    """
    declared = getattr(gate, "MEASURES_PROBLEM", None)
    if declared is None:
        return True                 # not declared: held to the default
    if declared is True:
        return declared
    if isinstance(declared, NoProblem):
        return declared             # the instance: the reason travels with it
    if declared is False:
        raise ValueError(
            f"gate {gate_name!r} declares MEASURES_PROBLEM = False, which was "
            f"the opt-out and is no longer accepted.\n"
            f"  Write the reason the constant could not carry:\n"
            f"    MEASURES_PROBLEM = NoProblem(\"what this gate measures "
            f"instead of a device -- an empty cell, the installed toolchain, "
            f"a structure the gate hard-codes\")\n"
            f"  `False` was the same three characters in four gate modules, "
            f"so it was correct wherever it was typed and stayed correct-"
            f"looking wherever it was copied. A sentence about THIS gate's "
            f"subject reads as false in a gate that measures a device.\n"
            f"  It is refused rather than accepted-as-before or ignored: "
            f"accepting it keeps the copyable constant alive, and ignoring it "
            f"would fail this gate for missing identity keys, pointing a "
            f"reader away from the line that actually needs editing.")
    if type(declared) is str and declared.strip():
        return declared
    raise ValueError(
        f"gate {gate_name!r} declares MEASURES_PROBLEM = {declared!r} "
        f"({type(declared).__name__}), which is none of the three things it "
        f"may be.\n"
        f"  True (or nothing at all): this gate measures whatever "
        f"`--problem` asked for.\n"
        f"  NoProblem('why not'): this gate measures no problem, and owes no "
        f"identity keys. The reason is mandatory and is about THIS gate.\n"
        f"  '<name>': this gate always measures that one problem, whatever "
        f"`--problem` says.\n"
        f"  A str SUBCLASS is refused with everything else: it would answer "
        f"the comparisons made about it on its own behalf. Use a plain str.")


def _verify_problem_identity(gate, res, args):
    """Backstop: check who a result says it measured against the REQUEST.

    `gate_details` is what a gate should use, but "should" is how G4 ended up
    with `**case.extra` last and a problem overwriting its own identity. This
    runs on every result, for every gate, written or not yet written.

    Where the truth comes from is the whole design. It used to come from
    `problems.from_args(args)` -- load the problem again and read the answer
    off the spec. That is the same object the gate read it off, so the check
    reduced to "this value equals itself", and anything that made the loader
    stamp a wrong value made the backstop agree with it. The truth is now
    derived from the request alone (`problems.identity_from_args`, which
    resolves a registry key or a dotted path and imports nothing), so the
    comparison has two independent sides.

    Which problem a gate is entitled to have measured comes from the GATE
    module, as `MEASURES_PROBLEM`. Read `_declared_problem` for what the
    values mean and, more importantly, for why the DEFAULT is "a problem":
    a gate whose author wrote nothing is asked for provenance, not exempted
    from it.

    The string form is a declaration by the gate, not by the problem: a
    problem module cannot reach it, and the runner does not take it as the
    answer either -- it is resolved through `requested_identity` and compared
    with what the result stamped, so a gate can only get a name into a report
    by actually having loaded that problem.

    Three rules, deliberately different:

      * every `details` key must be an exact `str`, on any status and any
        declaration, because everything below is a comparison BY KEY NAME
        and a `str` subclass answers those (`_exact_str_keys`). The two
        identity VALUES are held to the same rule, for the same reason: `!=`
        below would otherwise be answered by the value being checked -- the
        move a lying gate makes, since it owns that value;
      * a stamped identity is always checked, on any status;
      * a MISSING identity is a failure unless the gate declared a
        `NoProblem(reason)`, or the result is already a failure. A
        gate that failed before it got as far as loading anything has a real
        reason in its result, and burying that under a provenance complaint
        would replace the diagnosis with a lecture.

    Missing identity is a failure rather than something the runner fills in.
    Stamping it here would attach a provenance the runner INFERRED to numbers
    it never watched being produced -- exactly the difference between a
    record and a claim that this layer exists to keep. The gate's author
    fixes it once; a reader of a report never has to wonder which of the two
    it is looking at. The same reasoning rules out inferring the opt-out:
    a `NoProblem(reason)` has to be typed by a person, and the reason has to
    be typed about the gate it is typed in.

    It can only police the two identity keys, because those are the only ones
    whose true value the runner can obtain independently. Gate-measured
    numbers (`CE_fwd_dB`, `grad_max`, ...) are still guarded only where the
    gate calls `merge_problem_dict` -- the runner has no second source for
    them. Identity is the half worth a backstop: without it a report cannot
    even be attributed, so every other key in it is unattributed too.
    """
    gate_name = getattr(gate, "NAME", gate)
    _exact_str_keys(f"gate {gate_name!r}", res.details)
    declared = _declared_problem(gate_name, gate)
    stamped = {}
    for k in ID_KEYS:
        if k in res.details:
            v = res.details[k]
            if type(v) is not str:
                raise ValueError(
                    f"gate {gate_name!r} stamped details[{k!r}] as a "
                    f"{type(v).__name__}, not a str: {v!r}.\n"
                    f"  The check below is `v != truth`, and a str subclass "
                    f"answers that itself -- one returning True for __eq__ "
                    f"agrees with whatever it is compared against, so the "
                    f"backstop would confirm an identity it never actually "
                    f"read. The type is asked of the interpreter instead.\n"
                    f"  Fix: build `details` with runner.gate_details(spec, "
                    f"...); it stamps `str` values from the loaded spec.")
            stamped[k] = v
    measures_none = isinstance(declared, NoProblem)
    if measures_none and not stamped:
        return                      # a gate that declared it measures none

    from invdx import problems      # local: gates that need it import it too
    if declared is True or measures_none:
        name, module = problems.identity_from_args(args)
        asked = "--problem"
    else:
        name, module = problems.requested_identity(declared)
        asked = f"{gate_name}.MEASURES_PROBLEM = {declared!r}"
    truth = {"problem": name, "problem_module": module}

    missing = sorted(set(ID_KEYS) - set(stamped))
    if missing and not measures_none and res.status != FAIL:
        says = ("did not declare what it measures, so it is held to the "
                "default (`--problem`)"
                if getattr(gate, "MEASURES_PROBLEM", None) is None
                else f"says it measures a problem ({asked})")
        raise ValueError(
            f"gate {gate_name!r} {says}, but its result carries no "
            f"{missing}.\n"
            f"  A report with no identity keys describes numbers nobody can "
            f"attribute: the run directory says which `--problem` was typed, "
            f"but the report is what gets copied and aggregated, and on its "
            f"own it would not even say that much. The runner will not fill "
            f"these in -- it did not watch the numbers being produced, so "
            f"anything it wrote here would be a guess wearing the same "
            f"field name as a record.\n"
            f"  Fix -- whichever of these is true of this gate:\n"
            f"    it measures a problem: build `details` with "
            f"runner.gate_details(spec, ...), which stamps both from the "
            f"spec `problems.load` returned. Expected {truth}.\n"
            f"    it measures no problem, like G0/G1/G3/G5: write "
            f"MEASURES_PROBLEM = NoProblem(\"...\") in the gate module, "
            f"saying what it measures INSTEAD of a device. The runner will "
            f"not infer that for you -- being asked for provenance you do "
            f"not owe is a loud false alarm you fix in one line, while "
            f"being exempted by default is a silent gap that looks exactly "
            f"like a correct report. The reason is required for the same "
            f"kind of reason the declaration is: a bare constant is correct "
            f"in every module and so says nothing about the one it is in, "
            f"and an opt-out copied from a neighbouring gate reads as "
            f"correct until someone types out why.")

    wrong = {k: (v, truth[k]) for k, v in stamped.items() if v != truth[k]}
    if wrong:
        detail = "; ".join(f"details[{k!r}] = {got!r}, {asked} implies {exp!r}"
                           for k, (got, exp) in sorted(wrong.items()))
        raise ValueError(
            f"gate {gate_name!r} reported an identity that does not match "
            f"the problem it was asked for: {detail}.\n"
            f"  These two keys say which module produced the numbers. Build "
            f"`details` with runner.gate_details(spec, ...), which stamps "
            f"them from the loaded spec and refuses a problem-supplied "
            f"copy.\n"
            f"  If this gate measures a problem OTHER than the one "
            f"`--problem` names, that is legitimate and is declared, in the "
            f"gate module, as MEASURES_PROBLEM = '<name>'; the runner then "
            f"resolves that name itself and expects the result to match it.")


def one_line(text):
    """One printable line: control characters out, then whitespace collapsed.

    Applied where a string is printed as part of the console summary. Several
    of those strings come from the problem being measured -- an
    `Unsupported(reason)` is written by the problem module and reproduced
    verbatim -- and the console line is the surface most people read a gate
    run from. A reason containing a newline used to print a second, entirely
    fabricated status line, so a problem declaring itself inapplicable could
    put `[ok]  G4 reciprocity (91.05s)` under its own [n/a].

    Collapsing whitespace alone did not finish the job, and the docstring
    that said it did was the reason nobody looked again. `str.split()` splits
    on WHITESPACE; `ESC` and `BACKSPACE` are not whitespace, and they are
    what a terminal reads as movement. `\\x1b[2K\\x1b[1A` erases the line and
    moves the cursor up one, so a reason carrying it overwrites the status
    line already printed above -- the forged `[ok]` is not a second line at
    all, it replaces a real one, which is strictly worse than what was fixed.
    A run of `\\x08` does the same within the current line.

    So every character that is neither printable nor newline-or-tab becomes a
    space before the collapse -- the rule is `console_text`'s, applied first,
    and it is deliberately narrower than "whitespace" (read its docstring for
    why `str.isspace()` was the wrong alphabet). `str.isprintable()` is false
    for the whole C0 and C1 ranges, for `DEL`, and for the format category --
    which also removes the zero-width and bidirectional-override characters
    that make a line read as something other than what it contains. `\\u2028`
    and `\\u2029` never reach `split()`: `console_text` has already replaced
    them, so the collapse only ever sees spaces, newlines and tabs.

    The stored `details["reason"]` keeps the original text: `gates_report.json`
    is structured, so neither a newline nor an escape in it is a line or a
    movement -- `json.dump` escapes both. What is stripped here is stripped
    from the console rendering only.
    """
    return " ".join(console_text(text).split())


def console_text(text):
    """The same control-character strip, for output that is legitimately
    multi-line.

    `one_line` is not the only place problem-authored text reaches a terminal.
    The `[FAIL]` branch of `run_gates` dumps the traceback, and a traceback's
    last line is the exception's own message -- written by whatever raised,
    including the problem module. `print()`ing that raw put every escape
    sequence back on the console, on the one path where the reader is
    squinting hardest, so collapsing only the [n/a] line would have fixed the
    symptom that was noticed rather than the thing that was wrong.

    Newlines and tabs survive, because a traceback IS lines and destroying
    them would trade a forged status line for an unreadable diagnosis. The
    forgery this removes is cursor movement and invisible text, not layout.

    Newline and tab by name, NOT `str.isspace()`. `isspace()` is true for
    `\\r`, `\\v`, `\\f`, `\\x1c`-`\\x1f` and `\\x85`, and `\\r` is cursor
    movement -- it returns to the start of the line the reader is on, so an
    exception message ending `...\\r[ok]  G4 reciprocity (91.05s)` replaces the
    traceback's last line, which is the diagnosis, with a forged pass. This
    function was written in the same change that added a `\\r` case to
    `one_line`'s own test table: the attack was known and the allowance was
    written anyway, because "whitespace is layout" reads as obviously true.
    Layout is two characters; the rest of `isspace()` is not layout.

    Keeping `\\n` has a cost, stated here rather than left to be discovered:
    an exception message ending `\\n[ok]  G4 reciprocity (91.05s)` still prints
    that as its own line, in the real status line's format, at the bottom of a
    `[FAIL]` dump -- and `run_gates` breaks after a failure, so it is the last
    thing on screen. That is the original `[n/a]` forgery moved one path over.
    It is not fixed, it is priced: destroying newlines makes a traceback
    unreadable, and an unreadable diagnosis on every real failure costs more
    than a fabricated line printed directly under a `[FAIL]` that is still
    there. So "removes cursor movement" is the claim; "cannot be made to show
    something that looks like a pass" is not, and never was.

    Two more survive on purpose by being printable: characters that occupy
    space but draw nothing (`U+2800`, `U+3164`) and combining marks. They
    cannot move a cursor; they can pad a forged line into column alignment.
    """
    keep = "\n\t"
    return "".join(c if (c.isprintable() or c in keep) else " "
                   for c in str(text))


def not_applicable(gate_name, spec, reason):
    """Result for a gate a problem has declared inapplicable to itself.

    `reason` comes from the problem's `Unsupported(...)` and is mandatory
    there, so this cannot produce a bare "n/a" with nothing to argue with.

    `spec` is the loaded `ProblemSpec`, not a bare name, so an [n/a] result
    carries the same two identity keys an [ok] one does. It used to carry
    only `problem`, which meant the one status a problem gets to choose for
    itself was also the one that said least about where it came from -- and
    the runner's backstop, which now requires both keys from a gate that
    measures a problem, would have had to make an exception for exactly that
    case. A plain string is still accepted here, and the result then honestly
    lacks `problem_module` rather than inventing one -- but read that as a
    signature this function has not removed, not as a usable path: since the
    backstop's polarity was inverted, any gate that has not declared a
    `NoProblem(reason)` fails on the missing key, and a gate that HAS
    declared one has no reason to call this. And the string it passes is still
    compared against what `--problem` resolves to, so it has to spell that
    name exactly -- an odd thing for a gate that measures no problem to get
    right by accident. So the string form is reachable only from a gate that
    measures no problem, calls [n/a] anyway, and names the problem it is not
    measuring. It is kept because removing it would be a signature change for
    no reader, not because anything should take it.
    """
    if not str(reason).strip():
        raise ValueError(f"{gate_name}: an n/a result needs a reason")
    name = getattr(spec, "name", spec)
    details = {"problem": name,
               "reason": f"not applicable to {name}: {reason}"}
    module = getattr(spec, "module", "")
    if module:
        details["problem_module"] = module
    return GateResult(gate_name, NOT_APPLICABLE, details)


def discover():
    """Import every gates.g*.py module, sorted by ORDER."""
    import invdx.gates as pkg

    mods = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("g"):
            mods.append(importlib.import_module(f"invdx.gates.{info.name}"))
    return sorted(mods, key=lambda m: m.ORDER)


def run_gates(cfg, args, only=None, through=None):
    """Execute gates in order; stop at first fail. Returns list[GateResult].

    only    — run just the gate with this NAME
    through — run gates up to and including this NAME
    """
    results = []
    for mod in discover():
        if only and mod.NAME != only:
            continue
        t0 = time.time()
        try:
            res = mod.run(cfg, args)
            # Before the result can reach the report: a gate that let the
            # problem write its own identity fails here, like any other gate
            # failure, whether or not its author knew to guard against it.
            # The module is passed, not just its name, because the gate is
            # also where `MEASURES_PROBLEM` is declared.
            _verify_problem_identity(mod, res, args)
        except Exception:
            res = GateResult(mod.NAME, FAIL,
                             {"exception": traceback.format_exc()})
        res.details["seconds"] = round(time.time() - t0, 2)
        results.append(res)
        label = LABELS[res.status]
        # Collapsed, because part of this string is written by the problem
        # module being measured and the console line is one line by contract
        # -- see `one_line`. The stored value keeps its newlines.
        reason = one_line(res.details.get("reason", ""))
        print(f"{label} G{mod.ORDER} {mod.NAME} ({res.details['seconds']}s)"
              + (f" — {reason}" if reason else ""))
        if res.status == FAIL:
            # `console_text`, not raw: the traceback's last line is the
            # exception's message, and a problem module writes its own. The
            # json.dumps fallback needs no strip (it escapes everything
            # non-ASCII), but it goes through the same call so that there is
            # one answer to "is what run_gates prints sanitized?".
            print(console_text(res.details.get(
                "exception",
                json.dumps(res.details, indent=2, default=str))))
            break
        if through and mod.NAME == through:
            break
    return results


def write_report(results, run_dir):
    path = os.path.join(run_dir, "gates_report.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    return path
