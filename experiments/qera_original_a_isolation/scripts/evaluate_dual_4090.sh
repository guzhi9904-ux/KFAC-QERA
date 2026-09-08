#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$HERE/configs/llama3.1-8b.yaml}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/share/home/tm902089733300000/a913520780/chengkang/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec python "$HERE/run.py" --config "$CONFIG" evaluate \
  --dual-gpu --batch-size "${EVAL_BATCH_SIZE:-8}" --ce-chunk-tokens "${EVAL_CE_CHUNK_TOKENS:-2048}" "$@"
