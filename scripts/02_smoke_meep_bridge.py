#!/usr/bin/env python
"""Round-trip test of the cross-env Meep subprocess bridge: ping (version
report) + a tiny vacuum flux simulation. Uses -np 2 to stay out of the way
of any production optimization occupying the CPU cores.
"""

from invdx.cli import base_parser, apply_overrides
from invdx.config import BaseConfig
from invdx.engines import meep_bridge


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = apply_overrides(BaseConfig(), args)

    pong = meep_bridge.run_job("ping", {})
    print(f"[bridge] worker reports meep {pong['meep_version']} "
          f"({pong['processes']} MPI processes, python {pong['python']})")

    res = meep_bridge.run_job(
        "vacuum_flux",
        {"resolution": 20, "dft_decay_tol": cfg.dft_decay_tol})
    print(f"[bridge] vacuum flux from worker: {res['flux']:.6g} "
          f"(res {res['resolution']})")
    print("[done] file protocol verified both directions")


if __name__ == "__main__":
    main()
