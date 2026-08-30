#!/usr/bin/env python
"""Validation-gate runner. Every line must print [ok], or an honest
[skip]/[n/a]/[part] carrying its reason; any [FAIL] is a stop-the-line event
for all downstream work.

  make check          -> G0 only (seconds)
  make gates          -> all gates in order
  --only NAME / --through NAME for selective runs
  --problem NAME      which problem G2 Part C and G4 measure

Four of the six gates are problem-independent. The other two read --problem;
a problem that declares one of them inapplicable to itself prints [n/a] (or
[part], when only the problem-specific half of a gate was declared away),
with its reason on the same line -- neither a pass nor a failure, and looking
like neither.
"""

import sys

from invdx.cli import add_problem_arg, base_parser, apply_overrides, start_run
from invdx.config import BaseConfig
from invdx.gates import runner


def main():
    p = base_parser(__doc__)
    p.add_argument("--only", default=None, help="run a single gate by name")
    p.add_argument("--through", default=None,
                   help="run gates up to and including this name")
    add_problem_arg(p)
    args = p.parse_args()
    cfg = apply_overrides(BaseConfig(), args)

    results = runner.run_gates(cfg, args, only=args.only, through=args.through)

    # G0-only runs skip the run-dir ceremony; full runs get a report snapshot
    if not (args.only == "unit"):
        d = start_run(cfg, args, "gates")
        print(f"[report] {runner.write_report(results, d)}")

    if any(r.status == runner.FAIL for r in results):
        sys.exit(1)
    # "all passed" over a list containing an [n/a] would be the exact
    # overstatement the extra statuses exist to prevent -- say what did not run.
    aside = [r.name for r in results
             if r.status in (runner.NOT_APPLICABLE, runner.PARTIAL)]
    if aside:
        print(f"no gate failed; {', '.join(aside)} did not run in full "
              f"(reasons on the lines above and in gates_report.json).")
    else:
        print("all requested gates passed.")


if __name__ == "__main__":
    main()
