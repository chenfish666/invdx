"""Gate registry and ordered execution with [ok]/[FAIL] console output."""

import importlib
import json
import os
import pkgutil
import time
import traceback
from dataclasses import dataclass, field, asdict


@dataclass
class GateResult:
    name: str
    status: str                      # "ok" | "fail" | "skip"
    details: dict = field(default_factory=dict)


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
            res = GateResult(mod.NAME, "fail",
                             {"exception": traceback.format_exc()})
        res.details["seconds"] = round(time.time() - t0, 2)
        results.append(res)
        label = {"ok": "[ok]  ", "fail": "[FAIL]", "skip": "[skip]"}[res.status]
        reason = res.details.get("reason", "")
        print(f"{label} G{mod.ORDER} {mod.NAME} ({res.details['seconds']}s)"
              + (f" — {reason}" if reason else ""))
        if res.status == "fail":
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
