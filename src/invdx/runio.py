"""Run I/O: run directories and master-only JSON/CSV writes.

Ported verbatim from pvgc/core.py (Run I/O section). The MPI guards are kept
on purpose: this module is also imported by engines/meep_worker.py, which runs
under mpirun inside the meep conda env; when mpi4py is absent (invdx env) the
guards degrade to no-ops.
"""

import json
import os
import time


def _mpi_comm():
    """COMM_WORLD if running under MPI, else None (keeps this module meep-free)."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


def am_master():
    c = _mpi_comm()
    return c is None or c.rank == 0


def new_run_dir(tag, base="runs"):
    # timestamp must come from rank 0 only: ranks crossing a second boundary
    # would otherwise create (and write into) different run directories
    d = os.path.join(base, time.strftime("%Y%m%d-%H%M%S") + "-" + tag)
    c = _mpi_comm()
    if c is not None:
        d = c.bcast(d, root=0)
    if am_master():
        os.makedirs(d, exist_ok=True)
    if c is not None:
        c.barrier()
    return d


def save_json(path, obj):
    if not am_master():
        return
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def append_csv(path, header, row):
    if not am_master():
        return
    newfile = not os.path.exists(path)
    with open(path, "a") as f:
        if newfile:
            f.write(",".join(header) + "\n")
        f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
