#!/usr/bin/env bash
set -euo pipefail
CONFIG=$(realpath "$1")
RUN=$(realpath "$2")
TOTAL_HOURS=${3:?Supply cumulative total hours}
TEST_WINDOWS=${4:-}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
PY="$REPO/../conda_envs/qera-original-a/bin/python"
mkdir -p "$RUN/logs"
TAG=$(date +%Y%m%d_%H%M%S)_$$
trap 'result=$?; printf "RESUME_EXIT %s %s\n" "$result" "$(date -Is)"; printf "%s\n" "$result" > "$RUN/logs/resume_${TAG}.exit"' EXIT
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd -- "$REPO"
printf 'RESUME_START %s total_hours=%s\n' "$(date -Is)" "$TOTAL_HOURS"
ARGS=()
if [[ -n "$TEST_WINDOWS" ]]; then ARGS+=(--test-windows "$TEST_WINDOWS"); fi
"$PY" -u -B "$HERE/resume_eval.py" "$CONFIG" "$RUN" --total-hours "$TOTAL_HOURS" "${ARGS[@]}"
