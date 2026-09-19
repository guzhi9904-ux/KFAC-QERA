#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "$(id -un)" == cck ]] || exit 77
grep -q '/labgpu.slice/' /proc/self/cgroup || exit 78
RUN="$(realpath -m -- "$1")"
case "$RUN" in /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp02/*) ;; *) exit 79;; esac
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
"$PYTHON" -B test_structure.py
exec "$PYTHON" -u -B replay.py --config "$TOOLS/config.json" --output "$RUN"
