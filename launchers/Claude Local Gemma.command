#!/bin/bash
# Claude Code — local Gemma 4 31B IT (MLX via mlx-community conversion of google/gemma-4-31B-it)

CLAUDE_BIN="$HOME/.local/bin/claude"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLX_PYTHON="${MLX_PYTHON:-$HOME/.local/mlx-server/bin/python3}"
SERVER="${REPO_ROOT}/proxy/server.py"

export MLX_GEMMA_VARIANT="${MLX_GEMMA_VARIANT:-4bit}"
# Or pin exactly: export MLX_MODEL=mlx-community/gemma-4-31b-it-8bit

if ! lsof -i :4000 >/dev/null 2>&1; then
  "$MLX_PYTHON" "$SERVER" >/tmp/mlx-server-gemma.log 2>&1 &
  echo "  Loading Gemma 4 31B IT (MLX)..."
  while ! curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; do
    sleep 2
  done
fi

clear
echo ""
echo "  → Claude Code with LOCAL AI (Gemma 4 31B IT / MLX)"
echo "  → MLX_GEMMA_VARIANT=${MLX_GEMMA_VARIANT}"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 --permission-mode auto
