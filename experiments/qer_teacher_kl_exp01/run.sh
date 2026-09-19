#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "$(id -un)" == cck ]] || { echo 'Only cck permitted'; exit 77; }
grep -q '/labgpu.slice/' /proc/self/cgroup || { echo 'A labgpu lease is required'; exit 78; }
RUN="$(realpath -m -- "$1")"
case "$RUN" in /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp01_v2/*) ;; *) exit 79;; esac
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/$(date +%Y%m%d_%H%M%S)-$$.log") 2>&1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data1/cck/cache/huggingface
PYTHON=/home/cck/miniconda3/envs/kfac-qera/bin/python
cd "$TOOLS"
"$PYTHON" -B test_math.py
"$PYTHON" -B test_model.py
exec "$PYTHON" -u -B run.py --config "$TOOLS/config.json" --protocol "$TOOLS/protocol.md" --output "$RUN" --stage "${2:-all}"
