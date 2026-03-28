#!/usr/bin/env bash
# Install MLX from the TurboQuant KV-cache fork for local testing.
# Upstream fork/branch: https://github.com/arozanov/mlx/tree/feature/turboquant-kv-cache
#
# Prerequisites (macOS, Apple Silicon):
#   - Xcode Command Line Tools (xcode-select --install)
#   - cmake >= 3.25 on PATH (brew install cmake) — pip's build isolation may not see it otherwise
#   - A dedicated venv is strongly recommended.
#
# If the build fails with only "make: *** [all] Error 2", the real error is a few lines
# above that. Use a verbose build:
#   MLX_TURBOQUANT_VERBOSE=1 ./scripts/install-mlx-turboquant.sh
#
# Usage:
#   python3 -m venv ~/.local/mlx-turboquant && source ~/.local/mlx-turboquant/bin/activate
#   ./scripts/install-mlx-turboquant.sh
#   pip install mlx-lm
#   cd /path/to/claude-code-local && ~/.local/mlx-turboquant/bin/python3 proxy/server.py
#
# Environment overrides:
#   MLX_TURBOQUANT_REPO     default https://github.com/arozanov/mlx.git
#   MLX_TURBOQUANT_REF      default feature/turboquant-kv-cache
#   MLX_TURBOQUANT_EDITABLE set to 1 for editable clone + pip install -e
#   CMAKE_BUILD_PARALLEL_LEVEL  default 8 (MLX setup.py uses -jCPU_COUNT if unset — can OOM)
#   MLX_TURBOQUANT_NO_ISOLATION  set to 1 for pip --no-build-isolation (uses your venv cmake)
#   MLX_TURBOQUANT_VERBOSE       set to 1 for pip -v (full compiler errors)

set -euo pipefail

REPO="${MLX_TURBOQUANT_REPO:-https://github.com/arozanov/mlx.git}"
REF="${MLX_TURBOQUANT_REF:-feature/turboquant-kv-cache}"

# MLX setup.py passes -j$(nproc) to cmake unless this is set — high -j often causes OOM / flaky C++ builds.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

# Prefer Homebrew cmake/ninja if installed
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

PIP=(python3 -m pip install)
if [[ "${MLX_TURBOQUANT_VERBOSE:-0}" == "1" ]]; then
  PIP+=(-v)
fi
if [[ "${MLX_TURBOQUANT_NO_ISOLATION:-0}" == "1" ]]; then
  PIP+=(--no-build-isolation)
fi

echo "==> Removing broken / partial mlx installs (if any)..."
python3 -m pip uninstall -y mlx mlx-metal 2>/dev/null || true

if [[ "${MLX_TURBOQUANT_EDITABLE:-0}" == "1" ]]; then
  CLONE_ROOT="${MLX_TURBOQUANT_CLONE:-${TMPDIR:-/tmp}/mlx-turboquant}"
  echo "==> Cloning ${REPO} (ref ${REF}) to ${CLONE_ROOT}..."
  rm -rf "${CLONE_ROOT}"
  git clone --depth 1 --branch "${REF}" "${REPO}" "${CLONE_ROOT}"
  echo "==> Editable install (CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL})..."
  python3 -m pip install --upgrade pip setuptools wheel
  "${PIP[@]}" -e "${CLONE_ROOT}"
else
  echo "==> Installing MLX from git ${REPO} @ ${REF}"
  echo "    (parallelism: CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL})"
  python3 -m pip install --upgrade pip setuptools wheel
  "${PIP[@]}" "git+${REPO}@${REF}"
fi

echo "==> Verifying mlx.core (required for proxy/server.py)..."
python3 -c "import mlx; import mlx.core as mx; print('OK mlx', getattr(mlx, '__version__', '?'), mx.__name__)"

echo ""
echo "Run server from repo root:"
echo "  ~/.local/mlx-turboquant/bin/python3 proxy/server.py"
echo ""
echo "If pip install mlx-lm overwrites mlx, re-pin:"
echo "  python3 -m pip install --force-reinstall --no-deps \"git+${REPO}@${REF}\""
echo ""
echo "If build still fails: MLX_TURBOQUANT_VERBOSE=1 MLX_TURBOQUANT_NO_ISOLATION=1 ./scripts/install-mlx-turboquant.sh"
echo "and scroll up for the first C++ / Metal error above 'Error 2'."
