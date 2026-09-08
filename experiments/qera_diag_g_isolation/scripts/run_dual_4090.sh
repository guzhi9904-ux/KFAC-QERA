#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Deliberately ignore old CONFIG/RUN_DIR variables from other experiments.
CONFIG_PATH="${DIAG_G_CONFIG:-$HERE/configs/llama3.1-8b-dual4090.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
if [ "$#" -eq 0 ]; then
  set -- run
fi
exec python "$HERE/run.py" --config "$CONFIG_PATH" "$@"
