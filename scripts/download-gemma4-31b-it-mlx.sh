#!/usr/bin/env bash
# Prefetch Gemma 4 31B IT weights for MLX (Hugging Face Hub cache).
#
# Google model card (license / terms): https://huggingface.co/google/gemma-4-31B-it
# This script loads the mlx-community conversion compatible with mlx-lm.
#
# Environment:
#   GEMMA_PYTHON        python to use (default: python3)
#   MLX_MODEL           if set, prefetch this repo id instead
#   MLX_GEMMA_VARIANT   4bit | 5bit | 6bit | 8bit | bf16 | mxfp4 | mxfp8 | nvfp4 (default: 4bit)
#
# Example:
#   GEMMA_PYTHON=~/.local/mlx-server/bin/python3 ./scripts/download-gemma4-31b-it-mlx.sh

set -euo pipefail

PY="${GEMMA_PYTHON:-python3}"
if [[ -n "${MLX_MODEL:-}" ]]; then
  MODEL="$MLX_MODEL"
else
  V="${MLX_GEMMA_VARIANT:-4bit}"
  MODEL="mlx-community/gemma-4-31b-it-${V}"
fi

echo "Prefetching MLX model: $MODEL"
echo "(First run downloads into the Hugging Face cache; size depends on variant.)"
export _GEMMA_MLX_PREFETCH="$MODEL"
"$PY" -c 'from mlx_lm.utils import load; import os; m=os.environ["_GEMMA_MLX_PREFETCH"]; load(m); print("Done:", m)'
unset _GEMMA_MLX_PREFETCH
