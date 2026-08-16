#!/usr/bin/env python
"""CPU tuning sweep for the Meep verification fleet: rank count x MPI
binding policy, measured on the same synthetic benchmark case.

FDTD is memory-bandwidth-bound: past the bandwidth saturation point more
ranks only add scheduling tax, and on multi-socket/chiplet CPUs unpinned
ranks migrate across NUMA domains. This sweep replaces folklore with a
measured table for THIS machine.

  uv run python scripts/12_cpu_tuning.py --ranks 8 16 32 --tag cpu-local
  # on niu36 (dual-socket Xeon), with INVDX_MEEP_ENV pointing at your meep env:
  #   INVDX_MEEP_ENV=<path-to-meep-env> uv run python scripts/12_cpu_tuning.py \
  #       --ranks 16 32 64 --tag cpu-niu36
"""

import argparse
import json
import os
import platform

# MPICH hydra syntax (the conda meep env ships MPICH, not OpenMPI)
BINDINGS = {
    "none": [],
    "core": ["-bind-to", "core"],
    "numa": ["-bind-to", "numa"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--size", type=float, default=6.0)
    p.add_argument("--res", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--tag", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from invdx.engines import meep_bridge

    case = {"resolution": args.res, "size_um": [args.size] * 3,
            "steps": args.steps}
    rows = []
    for n in args.ranks:
        for bname, bargs in BINDINGS.items():
            try:
                r = meep_bridge.run_job("benchmark", case, n_ranks=n,
                                        timeout=1800, mpi_args=bargs)
                rate = r["mcell_steps_per_s"]
            except Exception as e:
                rate = None
                print(f"[cpu] np={n:3d} bind={bname:5s}  FAILED: "
                      f"{str(e).splitlines()[-1][:80]}")
                rows.append({"ranks": n, "bind": bname, "error": True})
                continue
            rows.append({"ranks": n, "bind": bname,
                         "mcell_steps_per_s": rate,
                         "wall_s": r["wall_s"]})
            print(f"[cpu] np={n:3d} bind={bname:5s}  "
                  f"{rate:7.1f} Mcell-steps/s  ({r['wall_s']:.1f}s)")

    best = max((x for x in rows if "mcell_steps_per_s" in x),
               key=lambda x: x["mcell_steps_per_s"], default=None)
    if best:
        print(f"\n[cpu] WINNER on {platform.node()}: np={best['ranks']} "
              f"bind={best['bind']}  {best['mcell_steps_per_s']:.1f} "
              f"Mcell-steps/s")
    out = args.out or f"runs/cpu_tuning_{args.tag}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"host": platform.node(), "case": case, "rows": rows},
                  f, indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
