#!/usr/bin/env bash
# Install MLX from the TurboQuant KV-cache fork for local testing.
# Upstream fork/branch: https://github.com/arozanov/mlx/tree/feature/turboquant-kv-cache
#
# Prerequisites (macOS, Apple Silicon):
#   - Xcode Command Line Tools (xcode-select --install)
#   - cmake >= 3.25 (e.g. brew install cmake)
#   - A dedicated venv is strongly recommended so you do not break other projects.
#
# Usage:
#   python3 -m venv ~/.local/mlx-turboquant && source ~/.local/mlx-turboquant/bin/activate
#   ./scripts/install-mlx-turboquant.sh
#   pip install mlx-lm
#   python3 proxy/server.py   # or your usual entrypoint
#
# Environment overrides:
#   MLX_TURBOQUANT_REPO  default https://github.com/arozanov/mlx.git
#   MLX_TURBOQUANT_REF   default feature/turboquant-kv-cache
#   MLX_TURBOQUANT_EDITABLE  set to 1 to clone and pip install -e (for hacking the fork)

set -euo pipefail

REPO="${MLX_TURBOQUANT_REPO:-https://github.com/arozanov/mlx.git}"
REF="${MLX_TURBOQUANT_REF:-feature/turboquant-kv-cache}"

echo "==> Removing PyPI mlx wheels from this environment (if any)..."
python3 -m pip uninstall -y mlx mlx-metal 2>/dev/null || true

if [[ "${MLX_TURBOQUANT_EDITABLE:-0}" == "1" ]]; then
  CLONE_ROOT="${MLX_TURBOQUANT_CLONE:-${TMPDIR:-/tmp}/mlx-turboquant}"
  echo "==> Cloning ${REPO} (ref ${REF}) to ${CLONE_ROOT}..."
  rm -rf "${CLONE_ROOT}"
  git clone --depth 1 --branch "${REF}" "${REPO}" "${CLONE_ROOT}"
  echo "==> Editable install (builds from source)..."
  python3 -m pip install --upgrade pip setuptools wheel cmake
  python3 -m pip install -e "${CLONE_ROOT}"
else
  echo "==> Installing MLX from git ${REPO} @ ${REF} (builds from source; first time can take several minutes)..."
  python3 -m pip install --upgrade pip setuptools wheel cmake
  python3 -m pip install "git+${REPO}@${REF}"
fi

echo "==> Verifying import..."
python3 -c "import mlx; import mlx.core as mx; print('mlx', getattr(mlx, '__version__', '?'), 'core', mx)"

echo ""
echo "Done. Next: pip install mlx-lm"
echo "If pip upgrades mlx back to PyPI, re-pin the fork with:"
echo "  python3 -m pip install --force-reinstall --no-deps \"git+${REPO}@${REF}\""
echo "Check: python3 -c \"import mlx; print(mlx.__file__)\""
