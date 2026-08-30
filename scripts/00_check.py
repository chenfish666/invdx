#!/usr/bin/env python
"""Validation-gate runner. Every line must print [ok] (or an honest [skip]);
any [FAIL] is a stop-the-line event for all downstream work.

  make check          -> G0 only (seconds)
  make gates          -> all gates in order
  --only NAME / --through NAME for selective runs
"""

import sys

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.config import BaseConfig
from invdx.gates import runner


def main():
    p = base_parser(__doc__)
    p.add_argument("--only", default=None, help="run a single gate by name")
    p.add_argument("--through", default=None,
                   help="run gates up to and including this name")
    args = p.parse_args()
    cfg = apply_overrides(BaseConfig(), args)

    results = runner.run_gates(cfg, args, only=args.only, through=args.through)

    # G0-only runs skip the run-dir ceremony; full runs get a report snapshot
    if not (args.only == "unit"):
        d = start_run(cfg, args, "gates")
        print(f"[report] {runner.write_report(results, d)}")

    if any(r.status == "fail" for r in results):
        sys.exit(1)
    print("all requested gates passed.")


if __name__ == "__main__":
    main()
