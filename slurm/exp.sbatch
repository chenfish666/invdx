#!/bin/bash
# Run one experiment script as a scheduled job.
#
# The optimisation driver has had slurm/grating_coupler_opt.sbatch for a while. One-off
# experiments did not, so they got launched straight onto whichever GPU looked
# free -- and on a node whose Slurm has no cgroup device isolation, that is
# exactly how two jobs end up on one card. --gres reserves a slot, every device
# stays visible to the process, and nothing complains. The collision shows up
# later as timings that will not reproduce.
#
#   sbatch -p <partition> --gres=gpu:1 -t 2:00:00 slurm/exp.sbatch \
#          EXP-001-forward-smoke uv run python scripts/01_smoke_fdtdx.py --tag smoke
#
# First argument is the experiment label (used for the run directory and the
# output file); everything after it is the command to run. That command is
# handed to srun as-is, so it has to be resolvable there -- "python" on a bare
# PATH usually is not. Use "uv run python", or an explicit interpreter path.
#
# Partition, account, QoS and the wall clock belong on the sbatch command line
# or in SBATCH_* environment variables, because they differ on every cluster.
#
# Knobs (environment variables, all optional):
#   RUNS_ROOT   runs/                       where run directories live
#   RUN_DIR     $RUNS_ROOT/<label>-$SLURM_JOB_ID
#               Exported to the payload as INVDX_RUN_DIR. That export is the
#               whole contract: nothing in this repo consumes it yet (checked
#               2026-08, single occurrence repo-wide); it exists so a payload
#               can drop artifacts into the run directory without being told
#               where it is. If you add a consumer, say so here.
#   AGENT       (unset)                     who is running this; recorded in
#                                           the manifest, because jobs
#                                           submitted under one shared account
#                                           and one git identity have no other
#                                           way to say who produced a result.
#               Free text, but pick ONE spelling per actor and keep to it:
#               manifests holding three spellings of the same submitter cannot
#               be grouped afterwards. Suggested values:
#                 <tool-or-session-name>  (an automated submitter)
#                 exp-<nnn>               (a session working one experiment)
#                 <your-name>             (a human at a shell)
#   EXP_ALLOW_UNSCHEDULED  (unset)          run outside Slurm anyway
#
# What this file will NOT do: set CUDA_VISIBLE_DEVICES. Slurm already sets it
# from the --gres allocation even when it cannot enforce it, so overriding it
# is the one action that reliably breaks the arbitration. See the device check
# below, which fails the job rather than letting that happen silently.

#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j-exp.out

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch [sbatch opts] slurm/exp.sbatch <experiment-label> <command> [args...]" >&2
    exit 2
fi

# `$# -lt 2` does not catch a forgotten label: the first word of the command
# becomes the label and the manifest records `label: uv`. Check the shape
# instead -- an experiment label is EXP-<nnn>-<slug>, or TEST-* for a
# throwaway run.
case "$1" in
    EXP-*|TEST-*) ;;
    *) echo "first argument should be an experiment label (EXP-...), got: $1" >&2
       echo "usage: sbatch [opts] slurm/exp.sbatch <label> <command> [args...]" >&2
       exit 2 ;;
esac
LABEL="$1"; shift

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---------------------------------------------------------------- scheduling
# Running outside the scheduler is what causes the collisions this file exists
# to prevent, so it is refused by default rather than warned about.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    if [ -z "${EXP_ALLOW_UNSCHEDULED:-}" ]; then
        echo "[exp] refusing to run outside Slurm." >&2
        echo "[exp] submit with sbatch, or set EXP_ALLOW_UNSCHEDULED=1 and" >&2
        echo "[exp] announce which device you are taking." >&2
        exit 3
    fi
    echo "[exp] WARNING: running unscheduled (EXP_ALLOW_UNSCHEDULED set)."
fi

# ------------------------------------------------------------- device checks
# Slurm exports CUDA_VISIBLE_DEVICES from the allocation. If it is set but no
# SLURM_*_GPUS is, the value came from somewhere else -- a --export=ALL that
# carried the submitter's shell environment in, most likely -- and it may point
# at a card this job was not given. That is silent: the job runs, on the wrong
# device, at the wrong speed, next to somebody else's work.
SLURM_GPUS="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "$SLURM_GPUS" ]; then
        echo "[exp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} is set but no" >&2
        echo "[exp] SLURM_STEP_GPUS/SLURM_JOB_GPUS is -- this did not come from" >&2
        echo "[exp] the allocation. Drop it from the submitting environment." >&2
        exit 4
    fi
    # Backstop, and honestly a weak one: submitting with CUDA_VISIBLE_DEVICES=1
    # under --export=ALL against a --gres=gpu:1 allocation of device 0 was tried,
    # and Slurm overwrote the exported value -- so on a scheduler that behaves
    # this way, this branch cannot fire. It is kept because schedulers that do
    # NOT overwrite it exist, and because a guard that has been measured not to
    # fire is different from one nobody checked. What it does not cover is the
    # payload setting the variable itself after this script has run; nothing
    # here can see that, which is why the allocated UUID is recorded below.
    #
    # NOTE (inference from documentation, NOT tested -- this node has no cgroup
    # device isolation, so the case cannot be reproduced here): SLURM_JOB_GPUS
    # holds GLOBAL device indices, while on clusters with ConstrainDevices the
    # cgroup remaps visible devices and CUDA_VISIBLE_DEVICES becomes 0..n-1.
    # There a correctly allocated job can legitimately show e.g.
    # SLURM_JOB_GPUS=2 next to CUDA_VISIBLE_DEVICES=0. A plain string compare
    # would refuse that job, so: a COUNT mismatch still fails (wrong on any
    # cluster), but a content mismatch with equal counts only warns.
    if [ -n "$SLURM_GPUS" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        slurm_gpu_n=$(awk -F',' '{print NF}' <<<"$SLURM_GPUS")
        cuda_gpu_n=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
        if [ "$slurm_gpu_n" != "$cuda_gpu_n" ]; then
            echo "[exp] allocation has ${slurm_gpu_n} GPU(s) (${SLURM_GPUS}) but" >&2
            echo "[exp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} lists" >&2
            echo "[exp] ${cuda_gpu_n}. Refusing." >&2
            exit 4
        elif [ "$SLURM_GPUS" != "$CUDA_VISIBLE_DEVICES" ]; then
            echo "[exp] WARNING: allocation says GPU(s) ${SLURM_GPUS} but" >&2
            echo "[exp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}. Counts match," >&2
            echo "[exp] so this is likely cgroup device remapping (indices" >&2
            echo "[exp] rebased to 0). Proceeding; manifest records both." >&2
        fi
    fi
fi

# Without --gres on the command line, SLURM_*_GPUS is empty and BOTH device
# checks above skip silently -- the job then sees every card on the node. Not
# hard-coding #SBATCH --gres here on purpose: this is the generic entry point
# and a CPU-only experiment should not be forced to hold a GPU.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "$SLURM_GPUS" ] \
   && [ -z "${EXP_ALLOW_NO_GPU:-}" ]; then
    echo "[exp] no GPU in this allocation, so the device checks above did" >&2
    echo "[exp] nothing. Submit with --gres=gpu:1, or set EXP_ALLOW_NO_GPU=1" >&2
    echo "[exp] if this experiment genuinely needs no GPU." >&2
    exit 5
fi

# ---------------------------------------------------------------- run layout
RUNS_ROOT="${RUNS_ROOT:-runs}"
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/${LABEL}-${SLURM_JOB_ID:-local}}"
mkdir -p "$RUN_DIR"

# The job log (#SBATCH --output above) lands in the submit directory: outside
# the run directory, outside version control, and deletable by a cleanup pass.
# When a job dies before the manifest records its ending, that log is the only
# trace -- so link it into RUN_DIR and the directory stays self-contained.
# Skipped when --output was overridden at submit time (then the default name
# below does not exist and this script cannot know the real one).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    out_log="${PWD}/slurm-${SLURM_JOB_ID}-exp.out"
    if [ -e "$out_log" ]; then
        ln -sfn "$out_log" "${RUN_DIR}/slurm-${SLURM_JOB_ID}-exp.out"
    fi
fi

# ---------------------------------------------------------------- provenance
# Warn (do not refuse) when nobody says who is running this: a manifest with
# agent <unset> still answers every other question, but the one it was built
# for goes unanswered.
if [ -z "${AGENT:-}" ]; then
    echo "[exp] WARNING: AGENT is unset; the manifest will say agent: <unset>." >&2
    echo "[exp] Set AGENT=<tool-or-session-name> / exp-<nnn> / <your-name> (see the" >&2
    echo "[exp] knobs list in this file's header) so the run has an owner." >&2
fi

# Enough to answer "who ran this, on what, from which commit" without asking
# the person who ran it -- the question a shared submit account cannot
# otherwise answer, because every job under it carries the same git identity.
MANIFEST="${RUN_DIR}/manifest.txt"
{
    echo "label:        ${LABEL}"
    echo "agent:        ${AGENT:-<unset>}"
    echo "node:         $(hostname -s)"
    echo "job_id:       ${SLURM_JOB_ID:-<unscheduled>}"
    echo "partition:    ${SLURM_JOB_PARTITION:-<none>}"
    echo "allocated_gpus: ${SLURM_GPUS:-<none>}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "git_commit:   $(git rev-parse HEAD 2>/dev/null || echo '<not a repo>')"
    echo "git_dirty:    $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "started_utc:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # $* flattens argv: `sh -c 'exit 7'` was recorded as `sh -c exit 7`, so
    # copying the line out of the manifest reran a different command. The
    # manifest exists to reproduce the run.
    echo "command:      $(printf '%q ' "$@")"
} | tee "$MANIFEST"

# Which physical card did we actually get? An index only means something next
# to the allocation that produced it, and on a node without device isolation
# every card is listed regardless of what was allocated -- so resolve the
# allocated index to a UUID, which still means the same thing when someone
# reads this manifest from a different job, on a different day, next to a
# profile that has to be matched to one card.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "visible_gpus:" | tee -a "$MANIFEST"
    nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader \
        | sed 's/^/  /' | tee -a "$MANIFEST"
    if [ -n "${SLURM_GPUS:-}" ]; then
        first_gpu="${SLURM_GPUS%%,*}"
        alloc_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                     | awk -F', ' -v i="$first_gpu" '$1==i {print $2}')
        echo "allocated_gpu_uuid: ${alloc_uuid:-<unresolved>}" | tee -a "$MANIFEST"
    fi
else
    echo "visible_gpus: <no nvidia-smi>" | tee -a "$MANIFEST"
fi

echo "[exp] run dir ${RUN_DIR}"
echo "[exp] ---------------------------------------------------------------"

export INVDX_RUN_DIR="$RUN_DIR"

# `set -e` means a failing srun aborts here, so `status=$?` and everything
# after it never ran: the manifest could structurally only ever record
# success, and a failed job was indistinguishable from one still running.
# Checked against 17 real manifests -- every recorded exit_status was 0, while
# the job logs held five genuine failures (time limit, missing module, gate
# abort) that the manifest never saw.
# Installed BEFORE srun on purpose: declared after it, the trap was
# not in place while srun was actually running, which is the entire
# window a scancel or a time limit lands in.
# A signal (Slurm time limit, scancel) kills this shell without reaching the
# lines below, so record the ending from a trap as well.
trap 'st=$?; echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"; \
      echo "exit_status:  signalled ($st)" >> "$MANIFEST"' TERM INT
status=0
srun "$@" || status=$?


echo "[exp] ---------------------------------------------------------------"
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MANIFEST"
echo "exit_status:  ${status}" | tee -a "$MANIFEST"
exit "$status"
