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
# Usage (must use a venv — Homebrew / system Python is "externally managed" and will refuse pip):
#   python3 -m venv ~/.local/mlx-turboquant
#   source ~/.local/mlx-turboquant/bin/activate
#   ./scripts/install-mlx-turboquant.sh
#
# Or without activating (explicit interpreter):
#   MLX_TURBOQUANT_PYTHON="$HOME/.local/mlx-turboquant/bin/python3" ./scripts/install-mlx-turboquant.sh
#
#   pip install mlx-lm
#   cd /path/to/claude-code-local && ~/.local/mlx-turboquant/bin/python3 proxy/server.py
#
# Environment overrides:
#   MLX_TURBOQUANT_REPO     default https://github.com/arozanov/mlx.git
#   MLX_TURBOQUANT_REF      default feature/turboquant-kv-cache
#   MLX_TURBOQUANT_EDITABLE set to 1 for editable clone + pip install -e
#   CMAKE_BUILD_PARALLEL_LEVEL  default 8 (MLX setup.py uses -jCPU_COUNT if unset — can OOM)
#   MLX_TURBOQUANT_NO_ISOLATION  default 1 (--no-build-isolation: use venv + brew cmake; avoids flaky pip-build-env)
#                                set to 0 to use pip's isolated build env only
#   MLX_TURBOQUANT_VERBOSE       set to 1 for pip -v and CMAKE_VERBOSE_MAKEFILE (shows real compile error)
#   MLX_TURBOQUANT_PYTHON        path to venv python3 (optional if VIRTUAL_ENV is set)

set -euo pipefail

if [[ -n "${MLX_TURBOQUANT_PYTHON:-}" ]]; then
  PY="${MLX_TURBOQUANT_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PY="${VIRTUAL_ENV}/bin/python3"
else
  echo "ERROR: Install must run inside a virtualenv (PEP 668 blocks pip on Homebrew Python)."
  echo "  source ~/.local/mlx-turboquant/bin/activate"
  echo "Or set:"
  echo "  MLX_TURBOQUANT_PYTHON=\"\$HOME/.local/mlx-turboquant/bin/python3\" $0"
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Not executable: $PY"
  exit 1
fi

REPO="${MLX_TURBOQUANT_REPO:-https://github.com/arozanov/mlx.git}"
REF="${MLX_TURBOQUANT_REF:-feature/turboquant-kv-cache}"

# MLX setup.py passes -j$(nproc) to cmake unless this is set — high -j often causes OOM / flaky C++ builds.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

# Prefer Homebrew cmake/ninja if installed
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

# Default: build outside pip's isolated overlay (your log showed failures under pip-build-env-*/overlay).
MLX_TURBOQUANT_NO_ISOLATION="${MLX_TURBOQUANT_NO_ISOLATION:-1}"

if [[ "${MLX_TURBOQUANT_VERBOSE:-0}" == "1" ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

PIP=("$PY" -m pip install)
if [[ "${MLX_TURBOQUANT_VERBOSE:-0}" == "1" ]]; then
  PIP+=(-v)
fi
if [[ "${MLX_TURBOQUANT_NO_ISOLATION}" == "1" ]]; then
  PIP+=(--no-build-isolation)
fi

echo "==> Using Python: $PY"
echo "==> pip build isolation: $([[ "${MLX_TURBOQUANT_NO_ISOLATION}" == "1" ]] && echo off || echo on)"
echo "==> Removing broken / partial mlx installs (if any)..."
"$PY" -m pip uninstall -y mlx mlx-metal 2>/dev/null || true

echo "==> Build prerequisites in venv (needed for --no-build-isolation)..."
"$PY" -m pip install --upgrade pip setuptools wheel 'cmake>=3.25' ninja

if [[ "${MLX_TURBOQUANT_EDITABLE:-0}" == "1" ]]; then
  CLONE_ROOT="${MLX_TURBOQUANT_CLONE:-${TMPDIR:-/tmp}/mlx-turboquant}"
  echo "==> Cloning ${REPO} (ref ${REF}) to ${CLONE_ROOT}..."
  rm -rf "${CLONE_ROOT}"
  git clone --depth 1 --branch "${REF}" "${REPO}" "${CLONE_ROOT}"
  echo "==> Editable install (CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL})..."
  "${PIP[@]}" -e "${CLONE_ROOT}"
else
  echo "==> Installing MLX from git ${REPO} @ ${REF}"
  echo "    (parallelism: CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL})"
  "${PIP[@]}" "git+${REPO}@${REF}"
fi

echo "==> Verifying mlx.core (required for proxy/server.py)..."
"$PY" -c "import mlx; import mlx.core as mx; print('OK mlx', getattr(mlx, '__version__', '?'), mx.__name__)"

echo ""
echo "Run server from repo root:"
echo "  ~/.local/mlx-turboquant/bin/python3 proxy/server.py"
echo ""
echo "If pip install mlx-lm overwrites mlx, re-pin:"
echo "  \"$PY\" -m pip install --force-reinstall --no-deps \"git+${REPO}@${REF}\""
echo ""
echo "If build still fails, capture the real compiler line:"
echo "  MLX_TURBOQUANT_VERBOSE=1 ./scripts/install-mlx-turboquant.sh 2>&1 | tee /tmp/mlx-build.log"
echo "Then search the log for 'error:' above the final 'Error 2'."
