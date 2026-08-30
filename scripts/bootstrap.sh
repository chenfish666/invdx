#!/usr/bin/env bash
# L1 bootstrap for invdx -- the uv-owned Python/GPU half of the environment.
# Counterpart of spack/bootstrap.sh (L2). Portable: no absolute paths, no
# machine-specific values, nothing that has to be edited before first use.
#
# Scope, following github/scripts-to-rule-them-all: "script/bootstrap ... used
# solely for fulfilling dependencies of the project". So this script installs
# and *verifies*; it does not run simulations, write env.sh, or export
# variables into your shell. L3 wiring stays in env.sh (see env.sh.example),
# L2 stays in spack/bootstrap.sh.
#
# Usage:
#   bash scripts/bootstrap.sh              # GPU environment (the default)
#   bash scripts/bootstrap.sh --cpu-only   # skip the GPU extra and the driver gate
#   bash scripts/bootstrap.sh --dry-run    # run every check, install nothing
#
# Idempotent: a second run says "already matches uv.lock" and touches nothing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CPU_ONLY=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --cpu-only) CPU_ONLY=1 ;;
        --dry-run)  DRY_RUN=1 ;;
        -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) printf 'unknown option: %s (try --help)\n' "$arg" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }
ok()   { printf '    ok    %s\n' "$1"; }
info() { printf '    ..    %s\n' "$1"; }
# Every failure exits with something the reader can actually run next, not a
# bare "error". First argument is the finding, the rest are the way out.
die() {
    printf '\n    FAIL  %s\n' "$1" >&2
    shift
    for line in "$@"; do printf '          %s\n' "$line" >&2; done
    printf '\n' >&2
    exit 1
}

# ---------------------------------------------------------------- uv itself
step "uv"

command -v uv >/dev/null 2>&1 || die \
    "uv is not on PATH." \
    "L1 is uv-owned; uv.lock is the source of truth for this half of the stack." \
    "Next, pick one:" \
    "  curl -LsSf https://astral.sh/uv/install.sh | sh" \
    "  pipx install uv" \
    "then re-run: bash scripts/bootstrap.sh"

ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# Capability probes, not a version floor. A hardcoded ">= 0.x.y" is a proxy for
# the thing that actually matters (does this uv have the two subcommands the
# repo's workflow is built on) and goes stale silently every time upstream
# renames or backports a flag. Ask the binary instead.
uv sync --help 2>/dev/null | grep -q -- '--locked' || die \
    "this uv has no 'uv sync --locked'." \
    "Bootstrap refuses plain 'uv sync': that silently re-resolves and rewrites" \
    "uv.lock when it disagrees with pyproject.toml, which is the opposite of" \
    "what a bootstrap is for." \
    "Next: upgrade uv (uv self update, or reinstall via the installer above)."

uv export --help 2>/dev/null | grep -q 'pylock.toml' || die \
    "this uv cannot export PEP 751 (no pylock.toml in 'uv export --format')." \
    "'make pylock' and 'make env-drift' both need it." \
    "Next: upgrade uv (uv self update)."

ok "uv sync --locked and uv export --format pylock.toml are both available"

# ------------------------------------------------------------------- python
step "Python interpreter"

# Read the constraint rather than restating it: pyproject.toml is the one place
# allowed to say which Pythons this project supports.
REQ_PY="$(sed -nE 's/^requires-python[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' pyproject.toml | head -1)"
[ -n "$REQ_PY" ] || die \
    "could not read requires-python from pyproject.toml." \
    "Expected a line like: requires-python = \">=3.12,<3.13\"" \
    "Next: check pyproject.toml is intact (git diff pyproject.toml)."

# uv does the version comparison; hand-rolled semver in bash is how you end up
# deciding 3.9 > 3.12.
if FOUND_PY="$(uv python find --no-project "$REQ_PY" 2>/dev/null)"; then
    ok "requires-python $REQ_PY satisfied by $FOUND_PY"
else
    SUGGEST="$(printf '%s' "$REQ_PY" | sed -nE 's/.*>=[[:space:]]*([0-9]+\.[0-9]+).*/\1/p')"
    [ -n "$SUGGEST" ] || SUGGEST="$REQ_PY"
    die \
        "no interpreter on this host satisfies requires-python $REQ_PY." \
        "Next: uv python install $SUGGEST" \
        "  (uv will then use its own managed interpreter; no root needed.)"
fi

# ------------------------------------------------------------------- driver
# L0 -- the GPU driver -- is not owned by this repo, but the pinned CUDA wheels
# have a hard floor on it, and docs/env.md has always listed this as the
# migration check. Until now nothing executed it.
step "GPU driver (L0)"

GPU_EXPECTED=0

if [ "$CPU_ONLY" -eq 1 ]; then
    info "--cpu-only: driver gate skipped, GPU extra will not be installed"
else
    # Which CUDA major the wheels want comes from the pin, not from this script.
    CUDA_MAJOR="$(grep -oE 'jax\[cuda[0-9]+\]' pyproject.toml | grep -oE '[0-9]+' | head -1)"
    case "$CUDA_MAJOR" in
        # JAX's install docs require a driver ">= 525 for CUDA 12 on Linux";
        # 525.60.13 is NVIDIA's exact CUDA 12 minimum. One row per CUDA major
        # this repo has ever pinned -- see docs/env.md, "Bootstrap".
        12) MIN_DRIVER="525.60.13" ;;
        "") die \
            "pyproject.toml's [gpu] extra does not name a jax[cudaNN] wheel." \
            "Nothing to check the driver against, and the GPU path is undefined." \
            "Next: restore the pin, or run: bash scripts/bootstrap.sh --cpu-only" ;;
        *)  die \
            "no driver floor recorded for CUDA $CUDA_MAJOR." \
            "The pin moved and this table did not. Guessing a floor here would be" \
            "worse than stopping: too low silently produces a CPU fallback." \
            "Next: look up the minimum driver for CUDA $CUDA_MAJOR in the JAX" \
            "install docs and add a row to the case statement in this script." ;;
    esac

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        info "nvidia-smi not found -- treating this as a host with no NVIDIA driver."
        info "Not an error. The CUDA wheels are still installable (they are just"
        info "files on disk); JAX will report a cpu backend, and the GPU line in"
        info "the verification below is reported rather than enforced."
        info "To skip the 4.5 GB of CUDA wheels entirely: --cpu-only"
    else
        # One line per visible GPU; take the oldest, since that is the one that
        # decides whether the process can initialise.
        DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
                  | tr -d ' \r' | grep -E '^[0-9]' | sort -V | head -1 || true)"
        if [ -z "$DRIVER" ]; then
            info "nvidia-smi is present but reported no driver version."
            info "Usually a stale kernel module after an in-place driver upgrade."
            info "Continuing; the GPU line below is reported, not enforced."
        elif [ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$DRIVER" | sort -V | head -1)" != "$MIN_DRIVER" ]; then
            die \
                "GPU driver $DRIVER is below $MIN_DRIVER, the floor for the CUDA $CUDA_MAJOR wheels pinned in pyproject.toml." \
                "A lockfile cannot fix this: the driver is host state, not a package." \
                "Installing anyway would give you a silent CPU fallback -- slow, and" \
                "indistinguishable from a working install in a summary line." \
                "Pick one:" \
                "  1. Raise the host driver to >= $MIN_DRIVER (needs root; L0 is not" \
                "     owned by this repo)." \
                "  2. Keep the driver and add NVIDIA's CUDA forward-compatibility" \
                "     package (cuda-compat-*), which lets a newer CUDA runtime run on" \
                "     an older driver. NVIDIA supports that path on data-center GPUs" \
                "     only -- check their compatibility matrix for your card first." \
                "  3. Accept CPU: bash scripts/bootstrap.sh --cpu-only" \
                "     (G1..G5 in 'make gates' need a GPU and will fail.)"
        else
            ok "driver $DRIVER >= $MIN_DRIVER (floor for CUDA $CUDA_MAJOR)"
            GPU_EXPECTED=1
        fi
    fi
fi

# ------------------------------------------------------------------ install
step "Dependencies"

SYNC_ARGS=(--locked --extra dev)
if [ "$CPU_ONLY" -eq 0 ]; then
    SYNC_ARGS+=(--extra gpu)
    info "target: core + gpu + dev  (~5.9 GB on disk, mostly CUDA wheels)"
else
    info "target: core + dev only, no CUDA wheels"
    info "if a GPU environment is already installed here, this will REMOVE it"
fi

# --locked, not bare 'uv sync': fail if uv.lock and pyproject.toml disagree
# instead of quietly re-resolving. Same two-level pinning contract as L2, where
# spack.yaml states intent and spack.lock states the one solution.
if uv sync --check "${SYNC_ARGS[@]}" >/dev/null 2>&1; then
    ok "environment already matches uv.lock -- nothing to install"
    SYNCED_NOW=0
else
    if [ "$DRY_RUN" -eq 1 ]; then
        info "changes uv would make:"
        uv sync --dry-run "${SYNC_ARGS[@]}" 2>&1 | sed 's/^/          /'
    else
        uv sync "${SYNC_ARGS[@]}"
    fi
    SYNCED_NOW=1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    step "Dry run"
    info "stopping before verification; nothing was installed or removed"
    exit 0
fi

# --------------------------------------------------------------- verify it
# This is the section spack/bootstrap.sh does not have. That script installs
# and then echoes "done", so a build that produced an unimportable result
# still reads as success. An install is not finished until something has been
# imported out of it.
step "Verification"

if [ "$CPU_ONLY" -eq 1 ]; then
    uv run --no-sync python - <<'PY' || die \
        "the CPU-only environment installed but does not import." \
        "Next: uv run python -c 'import invdx' to see the traceback in full."
import importlib
mods = ["numpy", "scipy", "autograd", "gdstk", "pyevtk", "invdx"]
for m in mods:
    importlib.import_module(m)
    print(f"    ok    import {m}")
print("    ..    jax and fdtdx are NOT installed (--cpu-only).")
print("    ..    'make check' (G0) works; 'make gates' (G1..G5) needs the GPU extra.")
PY
else
    # Pins are read out of pyproject.toml and compared against what is actually
    # importable, so a lock that drifted from the pin fails here rather than in
    # a simulation eight hours later.
    JAX_PIN="$(sed -nE 's/.*jax\[cuda[0-9]+\]==([0-9][^"]*)".*/\1/p' pyproject.toml | head -1)"
    FDTDX_PIN="$(sed -nE 's/.*fdtdx==([0-9][^"]*)".*/\1/p' pyproject.toml | head -1)"

    uv run --no-sync python - "$JAX_PIN" "$FDTDX_PIN" "$GPU_EXPECTED" <<'PY' || die \
        "the installed environment did not pass verification (see above)." \
        "Next, in order:" \
        "  uv sync --locked --extra gpu --extra dev   # re-run the install" \
        "  uv run python -m invdx.hardware            # what JAX thinks it is on" \
        "  make env-drift                             # is uv.lock still fresh?"
import sys
from importlib.metadata import version

jax_pin, fdtdx_pin, gpu_expected = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
bad = []


def check(label, got, want):
    if got == want:
        print(f"    ok    {label} {got}")
    else:
        print(f"    FAIL  {label} {got}, pyproject.toml pins {want}")
        bad.append(label)


check("jax", version("jax"), jax_pin)
check("jaxlib", version("jaxlib"), jax_pin)
check("fdtdx", version("fdtdx"), fdtdx_pin)

import jax

devs = jax.devices()
kinds = sorted({d.platform for d in devs})
print(f"    ok    jax.devices() -> {len(devs)} x {','.join(kinds)}")
if gpu_expected and "gpu" not in kinds:
    print("    FAIL  a driver above the floor was found, but JAX sees no GPU")
    bad.append("gpu-backend")
elif not gpu_expected:
    print("    ..    no GPU was expected on this host; backend not enforced")

# The fdtdx pin is load-bearing, not conservative: engines/fdtdx_fixes.py
# subclasses an internal fdtdx class to fix an axis-order bug specific to the
# pinned release. If that import breaks, the pin and the vendored fix have
# come apart -- which is exactly the failure a version equality check alone
# would miss.
try:
    from invdx.engines.fdtdx_fixes import GaussianBeamSource  # noqa: F401
    print("    ok    invdx.engines.fdtdx_fixes binds to fdtdx " + version("fdtdx"))
except Exception as exc:  # noqa: BLE001
    print(f"    FAIL  invdx.engines.fdtdx_fixes does not import: {exc}")
    bad.append("fdtdx_fixes")

sys.exit(1 if bad else 0)
PY
fi

# ------------------------------------------------------------------- done
step "L1 ready"
if [ "${SYNCED_NOW:-0}" -eq 0 ]; then
    info "nothing changed on this run"
fi
cat <<'EOF'
    Next:
      make check                 # G0, pure-python gates, ~5 min on CPU
      make gates                 # G0..G5 (G1..G5 need the GPU)
      make env-drift             # uv.lock fresh, pylock.toml in step with it

    L1 is only half the environment. Meep (L2) and the glue (L3) are separate:
      bash spack/bootstrap.sh    # L2, hours on a cold cache
      cp env.sh.example env.sh   # L3
    See docs/env.md.
EOF
