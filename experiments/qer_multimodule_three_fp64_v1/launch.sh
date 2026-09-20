#!/usr/bin/env bash
# Explicitly invoked by the user/agent; no scheduler, monitoring, or auto-retry.
set -euo pipefail
CONFIG=$(realpath "$1")
RUN=$(realpath -m "$2")
ENTRY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$ENTRY/../.." && pwd)
PY="$REPO/../conda_envs/qera-original-a/bin/python"
mkdir -p "$RUN/logs"
trap 'result=$?; printf "LAUNCHER_EXIT %s %s\n" "$result" "$(date -Is)"; printf "%s\n" "$result" > "$RUN/logs/launcher.exit"' EXIT
printf '%s\n' "$$" > "$RUN/logs/launcher.pid"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd -- "$REPO"
printf 'LAUNCHER_START %s\n' "$(date -Is)"
for stage in prepare pilot run; do
    printf 'STAGE_START %s %s\n' "$stage" "$(date -Is)"
    "$PY" -u -B "$ENTRY/runner.py" "$CONFIG" "$RUN" "$stage"
    printf 'STAGE_COMPLETE %s %s\n' "$stage" "$(date -Is)"
done
