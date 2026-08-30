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


def not_applicable(gate_name, problem, reason):
    """Result for a gate a problem has declared inapplicable to itself.

    `reason` comes from the problem's `Unsupported(...)` and is mandatory
    there, so this cannot produce a bare "n/a" with nothing to argue with.
    """
    if not str(reason).strip():
        raise ValueError(f"{gate_name}: an n/a result needs a reason")
    return GateResult(gate_name, NOT_APPLICABLE,
                      {"problem": problem,
                       "reason": f"not applicable to {problem}: {reason}"})


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
        except Exception:
            res = GateResult(mod.NAME, FAIL,
                             {"exception": traceback.format_exc()})
        res.details["seconds"] = round(time.time() - t0, 2)
        results.append(res)
        label = LABELS[res.status]
        reason = res.details.get("reason", "")
        print(f"{label} G{mod.ORDER} {mod.NAME} ({res.details['seconds']}s)"
              + (f" — {reason}" if reason else ""))
        if res.status == FAIL:
            print(res.details.get("exception", json.dumps(res.details, indent=2, default=str)))
            break
        if through and mod.NAME == through:
            break
    return results


def write_report(results, run_dir):
    path = os.path.join(run_dir, "gates_report.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    return path
