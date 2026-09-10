#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/share/home/tm902089733300000/a913520780/chengkang
COMMAND="${1:-run}"
if (( $# > 0 )); then shift; fi
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$BASE/conda_envs/qera-original-a/bin/python" -B \
  "$TOOLS_DIR/full_a_all_precision_r8_v1.py" "$COMMAND" \
  --repo-dir "$BASE/KFAC-QERA-98ad0c5" \
  --run-dir "$BASE/qera_runs/llama3.1-8b-diag-g-256/mxint3_full_g_v1" \
  --output-dir "$BASE/qera_diagnostics/full_a_all_precision_r8_v1" \
  "$@"
