#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${1:?Usage: bash run.sh CONFIG_JSON OUTPUT_DIR pilot|formal|report}"
: "${2:?Output directory required}"
: "${3:?Explicit pilot, formal or report stage required}"
export QER_PORTABLE_CONFIG="$(realpath -- "$1")"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${QER_PYTHON:-python}"
"$PYTHON" -B "$HERE/test_suite.py"
"$PYTHON" -B "$HERE/test_construction_fixture.py"
"$PYTHON" -B "$HERE/test_portable.py"
"$PYTHON" -B "$HERE/test_device_layout.py"
exec "$PYTHON" -u -B "$HERE/controller.py" --output "$2" --stage "$3" --workers 1
