#!/bin/bash
# One pvgc inverse-design round (script 15) as a requeue-safe batch job.
#
# Nothing site-specific is baked in: partition, account, QoS and the wall
# clock belong on the sbatch command line (or in SBATCH_* environment
# variables), because they differ on every cluster. Submit with e.g.
#
#   sbatch -p gpu -A myproject -t 14:00:00 slurm/pvgc_opt.sbatch
#   sbatch -p gpu --export=ALL,ITERS=40,SIM_TIME=0.8e-12 slurm/pvgc_opt.sbatch
#
# Requeue safety: the run directory is derived from the job ID, which Slurm
# keeps across a requeue, so a preempted job finds its own checkpoint and
# continues with --resume instead of starting over. Script 15 checkpoints
# after every iteration, so at most one iteration is ever lost.
#
# Knobs (environment variables, all optional):
#   RUNS_ROOT   runs/                 where run directories live
#   RUN_DIR     $RUNS_ROOT/pvgc-opt-$SLURM_JOB_ID
#   TAG         m1                    run tag
#   ITERS       40
#   LR          0.05
#   CHECKPOINTS 20
#   SIM_TIME    (unset -> script default 0.8e-12)
#   BUDGET_H    (unset -> no wall-clock budget; set it a bit under -t so the
#               job stops itself cleanly instead of being killed mid-iteration)
#   EXTRA       extra flags passed through verbatim

#SBATCH --job-name=pvgc-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j-pvgc-opt.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUNS_ROOT="${RUNS_ROOT:-runs}"
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/pvgc-opt-${SLURM_JOB_ID:-local}}"
TAG="${TAG:-m1}"
ITERS="${ITERS:-40}"
LR="${LR:-0.05}"
CHECKPOINTS="${CHECKPOINTS:-20}"

args=(--tag "$TAG" --iters "$ITERS" --lr "$LR" --checkpoints "$CHECKPOINTS")

# A run directory that already carries a checkpoint means this is a requeue.
if [ -f "${RUN_DIR}/opt_state.npz" ]; then
    echo "[sbatch] resuming ${RUN_DIR} (requeue $((${SLURM_RESTART_COUNT:-0})))"
    args+=(--resume "$RUN_DIR")
else
    echo "[sbatch] fresh run in ${RUN_DIR}"
    args+=(--run-dir "$RUN_DIR" --gradcheck)
fi

[ -n "${SIM_TIME:-}" ] && args+=(--set "sim_time_s=${SIM_TIME}")
[ -n "${BUDGET_H:-}" ] && args+=(--time-budget-h "$BUDGET_H")
[ -n "${EXTRA:-}" ] && args+=(${EXTRA})

echo "[sbatch] node $(hostname -s), $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s)"
srun uv run python scripts/15_pvgc_optimize.py "${args[@]}"
