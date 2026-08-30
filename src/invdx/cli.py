"""Shared CLI plumbing. Every script gets, for free:

  --set KEY=VAL   override ANY field of the passed config without editing code
                  (repeatable), e.g.  --set min_feature=0.15 --set seed=3
  --tag NAME      suffix for the timestamped run directory
  --resolution N  shortcut for --set resolution=N

Each run directory receives config.json / cmdline.txt / env.txt snapshots so
every result is reproducible even after the config class changes later.

The config instance is passed explicitly (apply_overrides(cfg, args)) rather
than parked in a module-level singleton, so two configs can coexist in one
process and nothing mutates state a caller cannot see.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import asdict

from . import runio


def base_parser(desc):
    p = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override any config field (repeatable)")
    p.add_argument("--tag", default="", help="run directory tag")
    p.add_argument("--resolution", type=int, default=None,
                   help="shortcut for --set resolution=N")
    return p


def _cast_like(cur, v):
    if isinstance(cur, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(v)
    if isinstance(cur, float):
        return float(v)
    if isinstance(cur, tuple):
        elem = type(cur[0]) if cur else float
        return tuple(elem(t) for t in v.split(","))
    return v


def apply_overrides(cfg, args):
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VAL, got: {kv}")
        k, v = kv.split("=", 1)
        if not hasattr(cfg, k):
            # It already knows the answer -- the hasattr above just asked it.
            # Sending the reader to the source to find out what they could have
            # typed is a worse answer than printing it.
            import dataclasses
            valid = ", ".join(sorted(f.name for f in dataclasses.fields(cfg)))
            raise SystemExit(f"unknown config key: {k}\n  valid keys: {valid}")
        setattr(cfg, k, _cast_like(getattr(cfg, k), v))
    if args.resolution:
        cfg.resolution = args.resolution
    return cfg


def _env_snapshot():
    """pip freeze + jax devices (if importable) + invdx git hash, best effort."""
    lines = []
    try:
        lines.append(subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60).stdout)
    except Exception as e:  # snapshot must never kill a run
        lines.append(f"pip freeze failed: {e}\n")
    try:
        from . import hardware
        # `jax.devices()` alone prints `[CudaDevice(id=0)]`, which is
        # byte-identical on every CUDA machine -- so a stored run snapshot
        # could not answer "was this measured on the same hardware as that
        # one?". The probe can.
        lines.append(hardware.describe(hardware.probe()) + "\n")
    except Exception as e:
        lines.append(f"hardware probe failed: {e}\n")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        h = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        lines.append(f"invdx git: {h.stdout.strip() or h.stderr.strip()}\n")
    except Exception:
        pass
    return "".join(lines)


def start_run(cfg, args, name, run_dir=None):
    """Create <runs_root>/<timestamp>-<name>[-tag]/ and snapshot config + env.

    run_dir — use this exact directory instead of a fresh timestamped one.
              Needed by jobs whose directory must be predictable before the
              process starts (a requeued Slurm job has to find its own
              checkpoint again); everything else leaves it None.
    """
    tag = name + (("-" + args.tag) if args.tag else "")
    if run_dir:
        d = run_dir
        if runio.am_master():
            os.makedirs(d, exist_ok=True)
    else:
        d = runio.new_run_dir(tag, base=cfg.runs_root)
    runio.save_json(os.path.join(d, "config.json"), asdict(cfg))
    if runio.am_master():
        # Machine-readable twin of the block in env.txt, so a later comparison
        # across machines is a diff rather than a reading exercise.
        try:
            from . import hardware
            runio.save_json(os.path.join(d, "hardware.json"),
                            hardware.as_dict(hardware.probe()))
        except Exception as e:                 # never kill a run over a snapshot
            runio.save_json(os.path.join(d, "hardware.json"),
                            {"probe_failed": str(e)})
    if runio.am_master():
        with open(os.path.join(d, "cmdline.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        with open(os.path.join(d, "env.txt"), "w") as f:
            f.write(_env_snapshot())
    print(f"[run] outputs -> {d}")
    return d


def load_design(path):
    """Accept either a run directory or a direct .npy path; return binary rho."""
    import numpy as np
    if os.path.isdir(path):
        for cand in ("design_rho.npy", "design_x.npy"):
            f = os.path.join(path, cand)
            if os.path.exists(f):
                path = f
                break
    rho = np.load(path)
    return (rho > 0.5).astype(float)
