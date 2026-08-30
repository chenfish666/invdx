#!/bin/bash
# One grating_coupler inverse-design round (script 15) as a requeue-safe batch job.
#
# Nothing site-specific is baked in: partition, account, QoS and the wall
# clock belong on the sbatch command line (or in SBATCH_* environment
# variables), because they differ on every cluster. Submit with e.g.
#
#   sbatch -p <partition> -A <account> -t 14:00:00 slurm/grating_coupler_opt.sbatch
#   sbatch -p <partition> --export=ALL,ITERS=40,SIM_TIME=0.8e-12 \
#          slurm/grating_coupler_opt.sbatch
#
# Requeue safety: the run directory is derived from the job ID, which Slurm
# keeps across a requeue, so a preempted job finds its own checkpoint and
# continues with --resume instead of starting over. Script 15 checkpoints
# after every iteration, so at most one iteration is ever lost.
#
# Knobs (environment variables, all optional):
#   RUNS_ROOT   runs/                 where run directories live
#   RUN_DIR     $RUNS_ROOT/coupler-opt-$SLURM_JOB_ID
#   TAG         m1                    run tag
#   ITERS       40
#   LR          0.05
#   CHECKPOINTS 20
#   SIM_TIME    (unset -> script default 0.8e-12)
#   BUDGET_H    (unset -> no wall-clock budget; set it a bit under -t so the
#               job stops itself cleanly instead of being killed mid-iteration)
#   GPU_INDEX   (unset -> use whatever Slurm exposes)
#               DEPRECATED. Setting it overrides CUDA_VISIBLE_DEVICES inside
#               the job, which is exactly the action slurm/exp.sbatch refuses
#               ("the one action that reliably breaks the arbitration") -- and
#               this file hard-codes --gres=gpu:1, so Slurm already exports
#               the allocated device. The only remaining legitimate use: a
#               scheduler that sets no CUDA_VISIBLE_DEVICES at all on a
#               shared node without device isolation, where pinning one card
#               manually beats seeing all of them -- and then announce which
#               card you are taking. Leaving it set now prints a warning.
#   EXTRA       extra flags passed through verbatim

#SBATCH --job-name=coupler-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j-coupler-opt.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUNS_ROOT="${RUNS_ROOT:-runs}"
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/coupler-opt-${SLURM_JOB_ID:-local}}"
TAG="${TAG:-opt}"
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

# GPU_INDEX is deprecated (see the knobs list in the header): overriding what
# Slurm exported is how a job wanders onto a card someone else holds. Kept,
# with a warning, for the one case where Slurm exported nothing at all.
if [ -n "${GPU_INDEX:-}" ]; then
    echo "[sbatch] WARNING: GPU_INDEX is deprecated. Overriding" >&2
    echo "[sbatch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} -> ${GPU_INDEX}." >&2
    echo "[sbatch] Trust the allocation unless Slurm exported nothing." >&2
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi

echo "[sbatch] node $(hostname -s), $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s)," \
     "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
srun uv run python scripts/15_grating_coupler_optimize.py "${args[@]}"
