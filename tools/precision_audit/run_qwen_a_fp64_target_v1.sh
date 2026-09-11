#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/share/home/tm902089733300000/a913520780/chengkang
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$BASE/conda_envs/qera-original-a/bin/python" -B \
  "$TOOLS_DIR/qwen_a_fp64_target_v1.py" \
  --tools-dir "$BASE/precision_tools_25813e5/tools/precision_audit" \
  --v2-helper "$BASE/qwen_a_audit_v2/tools/precision_audit/qwen_full_a_audit_v2.py" \
  --run-dir "$BASE/qera_runs/qwen2.5-7b-base-mxint3-v1" \
  --output-dir "$BASE/qera_diagnostics/qwen_a_fp64_target_v1" \
  --device cuda:0 "$@"
