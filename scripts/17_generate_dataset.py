#!/usr/bin/env python
"""Batch forward-simulation dataset generation for the pvgc coupler.

Samples a batch of grating geometries, runs ONE forward-only fdtdx
simulation per design (through the same INVDX_FAST-gated loop
`problems.pvgc.characterize` already uses), and writes the result as
self-describing npz shards + a JSON manifest under a run directory — the
mechanism `invdx.datasets` provides. What to sample and how to label it is a
research decision left to the caller (`--kind` / `--set`), not this script.

  python scripts/17_generate_dataset.py --tag smoke --kind uniform-grating \
      --n 8 --shard-size 4 --set spacing_um=0.05 --set sim_time_s=0.05e-12 \
      --set L_design=6.0 --set pad_x=2.0 --set dpml=0.6 --set t_box=1.5 \
      --set t_sub=0.8 --set air_above=2.0 --set x_mon_wg=-4.0 --set x_src_wg=-4.5 \
      --set design_grid_per_um=20                        # CPU, seconds/sample

  python scripts/17_generate_dataset.py --tag prod --kind random-rho \
      --n 2000 --shard-size 50 --lams 1.28,1.31,1.34      # production, GPU

  python scripts/17_generate_dataset.py --run-dir runs/<dir> ...  # resume:
      same args, same dir -> shards already on disk are skipped

CPU-vs-GPU is never decided here: fdtdx/jax pick up whatever JAX_PLATFORMS /
CUDA_VISIBLE_DEVICES say at process start, same as every other script.
"""

import sys

from invdx import datasets
from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import pvgc


def main():
    sys.stdout.reconfigure(line_buffering=True)

    p = base_parser(__doc__)
    p.add_argument("--kind", choices=sorted(datasets.SAMPLE_KINDS),
                   default="uniform-grating", help="geometry sampling kind")
    p.add_argument("--n", type=int, default=8, help="number of designs")
    p.add_argument("--shard-size", type=int, default=4,
                   help="samples per npz shard")
    p.add_argument("--lams", default=None, metavar="L1,L2,...",
                   help="optional CE(lambda) spectrum per sample (um); one "
                        "extra empty-cell+grating run pair per sample")
    p.add_argument("--period-range", default="0.4,0.8", metavar="LO,HI",
                   help="uniform-grating: period sampling range (um)")
    p.add_argument("--duty-range", default="0.3,0.7", metavar="LO,HI",
                   help="uniform-grating: duty-cycle sampling range")
    p.add_argument("--rho-beta", type=float, default=64.0,
                   help="random-rho: tanh projection sharpness after "
                        "conic filtering")
    p.add_argument("--rho-eta", type=float, default=None,
                   help="random-rho: projection threshold "
                        "(default: cfg.eta_i)")
    p.add_argument("--run-dir", default=None, metavar="DIR",
                   help="use this exact run dir instead of a fresh "
                        "timestamped one (existing shards are skipped)")
    args = p.parse_args()

    cfg = apply_overrides(pvgc.PVGCConfig(), args)

    lo, hi = (float(v) for v in args.period_range.split(","))
    cfg._period_range_um = (lo, hi)
    lo, hi = (float(v) for v in args.duty_range.split(","))
    cfg._duty_range = (lo, hi)
    cfg._rho_beta = args.rho_beta
    cfg._rho_eta = args.rho_eta if args.rho_eta is not None else cfg.eta_i

    lams = [float(v) for v in args.lams.split(",")] if args.lams else None

    d = start_run(cfg, args, "dataset", run_dir=args.run_dir)

    def on_shard(entry):
        tag = "skip " if entry["skipped_existing"] else "wrote"
        print(f"[shard] {tag} {entry['file']}  n={entry['n_samples']}  "
              f"sha256={entry['sha256'][:12]}...")

    manifest = datasets.generate_dataset(
        cfg, kind=args.kind, n=args.n, run_dir=d,
        shard_size=args.shard_size, lams_um=lams, on_shard=on_shard)

    print(f"[done] {d}/{datasets.MANIFEST_FILE}  "
          f"({manifest['n_samples_written']}/{args.n} samples, "
          f"{len(manifest['shards'])} shard(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
