#!/usr/bin/env python
"""Which run directories can actually be fed to verify / tolerance / handoff.

`runs/` accumulates everything: gate runs, benchmarks, killed optimisations,
finished designs. Only a few carry a design vector, and the difference is not
visible from the directory name. Without this the answer is "read the
checkpoint format, then write your own find" -- which is a fair amount to ask
of someone whose actual question was "which of these can I use".

    make runs                 # or: python tools/list_runs.py [runs-root]
"""
import os
import sys

# file -> what having it makes the run good for
MARKERS = [
    ("design_rho.npy", "verify, tolerance, handoff"),
    ("design_rho_2d.npy", "eval-2d"),
    ("opt_state.npz", "resume"),
    ("results.json", "read results"),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    if not os.path.isdir(root):
        raise SystemExit(f"no such directory: {root}")
    rows, total = [], 0
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        total += 1
        have = [(f, use) for f, use in MARKERS if os.path.exists(os.path.join(d, f))]
        if have:
            rows.append((name, ", ".join(f for f, _ in have),
                         ", ".join(sorted({u for _, u in have}))))
    if not rows:
        print(f"{total} directories under {root}/, none carrying a design or results.")
        return
    w = max(len(r[0]) for r in rows)
    print(f"{'run dir'.ljust(w)}  {'contains'.ljust(34)}  usable for")
    print(f"{'-' * w}  {'-' * 34}  {'-' * 30}")
    for name, files, uses in rows:
        print(f"{name.ljust(w)}  {files.ljust(34)}  {uses}")
    print(f"\n{len(rows)} of {total} directories carry something usable; "
          f"the rest are gate runs, benchmarks or interrupted jobs.")


if __name__ == "__main__":
    main()
