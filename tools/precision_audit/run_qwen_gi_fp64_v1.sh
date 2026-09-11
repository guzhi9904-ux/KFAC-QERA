#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/share/home/tm902089733300000/a913520780/chengkang
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$BASE/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$BASE/qera_official_bf16_pydeps:$BASE/QERA-official-bd7fc86/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$BASE/conda_envs/qera-original-a/bin/python" -B "$TOOLS_DIR/qwen_gi_fp64_v1.py" \
  --source-code "$BASE/KFAC-QERA-qwen25-base-v1" \
  --run-dir "$BASE/qera_runs/qwen2.5-7b-base-mxint3-v1" \
  --output-dir "$BASE/qera_diagnostics/qwen_gi_fp64_v1" \
  --harness-source "$BASE/QERA-harness-3823cfe" \
  --official-qera-root "$BASE/QERA-official-bd7fc86" \
  --word-reference-dir "$BASE/qera_runs/official-word-ppl-4096-existing-artifacts" "$@"
