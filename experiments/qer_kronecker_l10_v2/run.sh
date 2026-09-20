#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: run.sh CONFIG RUN_DIRECTORY prepare|pilot|run|report' >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${QER_PYTHON:-python}" -B "$here/controller.py" "$1" "$2" "$3"
