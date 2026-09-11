#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/share/home/tm902089733300000/a913520780/chengkang
MODE="${1:-inspect}"
if (( $# > 0 )); then shift; fi
case "$MODE" in
  inspect) EXTRA=() ;;
  replay) EXTRA=(--replay --device cuda:0) ;;
  *) printf '%s\n' 'Usage: run_qwen_a_diagnostic_v2.sh inspect|replay [options]' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$BASE/conda_envs/qera-original-a/bin/python" -B \
  "$TOOLS_DIR/qwen_full_a_audit_v2.py" \
  --helper-file "$BASE/precision_tools_25813e5/tools/precision_audit/qwen_full_a_audit_v1.py" \
  --run-dir "$BASE/qera_runs/qwen2.5-7b-base-mxint3-v1" \
  --module model.layers.1.mlp.down_proj "${EXTRA[@]}" "$@"
