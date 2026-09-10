#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
# Never install/upgrade packages or update the protected Llama checkout here.
exec python "$HERE/run.py" --config "${QWEN_CONFIG:-$HERE/qwen25-base.yaml}" "$@"
