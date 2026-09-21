#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
BASE=$(cd -- "$REPO/.." && pwd)
RUN=${1:-"$BASE/qera_runs/k_structure_audit_v2/run_01"}
mkdir -p "$RUN/logs"
trap 'code=$?; printf "K_AUDIT_EXIT %s %s\n" "$code" "$(date -Is)"' EXIT
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
printf 'K_AUDIT_START %s\n' "$(date -Is)"
"$BASE/conda_envs/qera-original-a/bin/python" -u -B "$HERE/runner.py" \
 --ko-run "$BASE/qera_runs/ko_increment_v1/run_01" \
 --v-run "$BASE/qera_runs/v_attention_proto_v1/run_01" --output "$RUN"
