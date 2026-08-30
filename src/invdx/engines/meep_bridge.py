"""Parent side of the cross-env Meep protocol (runs in the invdx env).

JAX/CUDA and conda pymeep don't share an env, so Meep work is a subprocess:
    1. write jobdir/job.json  (task name + payload) and input .npy arrays
    2. spawn  mpirun -np N <meep-env-python> meep_worker.py <jobdir>
       with PYTHONPATH pointing at invdx/src (worker imports numpy-pure
       invdx modules only)
    3. block, then read jobdir/result.json

File exchange + mpirun is deliberately dumb, which is the point: nothing has
to be serialized live across two mutually incompatible dependency trees, and
a worker that dies leaves its job directory behind to be inspected. Keep -np
small for gate/smoke jobs: a production optimization may be occupying the
CPU cores.
"""

import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np

# Repo root, derived from this file's location (engines/ -> invdx/ -> src/ -> repo):
# never a hardcoded home directory, so this works from any checkout/machine.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_MEEP_ENV = _REPO_ROOT / "spack" / "env" / ".spack-env" / "view"

# `or` (not a get() default) so an *empty* INVDX_MEEP_ENV also falls back —
# "VAR= cmd" shell prefixes set the variable to "" rather than unsetting it.
MEEP_ENV = os.environ.get("INVDX_MEEP_ENV") or str(_DEFAULT_MEEP_ENV)
MEEP_PYTHON = os.path.join(MEEP_ENV, "bin", "python")
MPIRUN = os.path.join(MEEP_ENV, "bin", "mpirun")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meep_worker.py")
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_job(task, payload, arrays=None, n_ranks=2, timeout=600, jobdir=None,
            mpi_args=None):
    """Execute one worker task; returns the parsed result.json dict.

    arrays — optional {name: ndarray} saved as <name>.npy next to job.json;
    the worker addresses them by name.
    mpi_args — extra mpirun arguments (e.g. ["--bind-to", "core"]) for
    placement/pinning experiments; default none.
    """
    if not os.path.exists(MEEP_PYTHON):
        raise RuntimeError(
            f"No Meep environment found at {MEEP_ENV!r} (bin/python missing — "
            "spack env not built yet, or wrong path). Set INVDX_MEEP_ENV to a "
            "directory with bin/python + bin/mpirun, e.g. an existing conda "
            "env with pymeep.")
    if jobdir is None:
        jobdir = tempfile.mkdtemp(prefix="meepjob-")
    os.makedirs(jobdir, exist_ok=True)
    with open(os.path.join(jobdir, "job.json"), "w") as f:
        json.dump({"task": task, "payload": payload}, f, indent=2)
    for name, arr in (arrays or {}).items():
        np.save(os.path.join(jobdir, f"{name}.npy"), np.asarray(arr))

    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = ([MPIRUN, "-np", str(n_ranks)] + list(mpi_args or [])
           + [MEEP_PYTHON, WORKER, jobdir])
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)
    result_path = os.path.join(jobdir, "result.json")
    if proc.returncode != 0 or not os.path.exists(result_path):
        raise RuntimeError(
            f"meep worker failed (task={task}, rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}")
    with open(result_path) as f:
        return json.load(f)
