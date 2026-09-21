#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a913520780/chengkang
REPO="$BASE/KFAC-QERA-teacher-kl"
PY="$BASE/conda_envs/qera-original-a/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HOME="$BASE/huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
set +e
"$PY" experiments/full_model_attention_slim_v2/run.py "${1:-resume}"
status=$?
echo "FULL_MODEL_SLIM_EXIT $status"
exit "$status"
