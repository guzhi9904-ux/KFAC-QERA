#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
CONFIG=$(realpath "$1")
RUN=$(realpath -m "$2")
STAGE=${3:-run}
mkdir -p "$RUN/logs"
TAG=$(date +%Y%m%d_%H%M%S)_$$
trap 'code=$?; printf "PROTOTYPE_EXIT %s %s\n" "$code" "$(date -Is)"; printf "%s\n" "$code" > "$RUN/logs/${STAGE}_${TAG}.exit"' EXIT
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
printf 'PROTOTYPE_START %s stage=%s\n' "$(date -Is)" "$STAGE"
"$REPO/../conda_envs/qera-original-a/bin/python" -u -B "$HERE/runner.py" "$CONFIG" "$RUN" "$STAGE"
