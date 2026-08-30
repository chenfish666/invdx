#!/usr/bin/env bash
# spack bootstrap for invdx — 可攜，無絕對路徑。
# 用法：bash spack/bootstrap.sh
set -euo pipefail

SPACK_ROOT="${SPACK_ROOT:-$HOME/spack}"
SPACK_VERSION="v1.2.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/env"

if [ ! -d "$SPACK_ROOT" ]; then
    echo "==> cloning spack $SPACK_VERSION to $SPACK_ROOT"
    git clone -c feature.manyFiles=true --depth 1 --branch "$SPACK_VERSION" \
        https://github.com/spack/spack.git "$SPACK_ROOT"
else
    echo "==> reusing existing spack at $SPACK_ROOT"
fi

# shellcheck disable=SC1091
. "$SPACK_ROOT/share/spack/setup-env.sh"

echo "==> spack version: $(spack --version)"

echo "==> detecting external packages (user scope: ~/.spack/)"
spack external find || true
spack compiler find || true

echo "==> concretizing invdx meep env"
spack -e "$ENV_DIR" concretize

echo "==> installing invdx meep env (this takes hours on cold cache)"
spack -e "$ENV_DIR" install --fail-fast

echo "==> done. view at: $ENV_DIR/.spack-env/view"
