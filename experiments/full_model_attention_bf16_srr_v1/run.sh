#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a913520780/chengkang
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HOME="$BASE/huggingface_cache" HF_DATASETS_CACHE="$BASE/huggingface_cache/datasets" HF_HUB_CACHE="$BASE/huggingface_cache/hub"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$BASE/qera_official_bf16_pydeps:$BASE/QERA-official-bd7fc86/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE/KFAC-QERA-teacher-kl"
set +e
"$BASE/conda_envs/qera-original-a/bin/python" -B experiments/full_model_attention_bf16_srr_v1/run.py "${1:-resume}"
status=$?
echo "BF16_SRR_EXIT $status"
exit "$status"
