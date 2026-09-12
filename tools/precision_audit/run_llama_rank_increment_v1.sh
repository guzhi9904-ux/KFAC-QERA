#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-doctor}"
if (( $# > 0 )); then shift; fi
(cd -- "$TOOLS_DIR" && sha256sum -c SHA256SUMS.llama_rank_increment)
PROFILE="${LLAMA_RANK_INCREMENT_PROFILE:-$TOOLS_DIR/server_profile.sh}"
if [[ "$PROFILE" == "$TOOLS_DIR/server_profile.sh" && -f "$TOOLS_DIR/SHA256SUMS.llama_rank_increment_profile" ]]; then
  (cd -- "$TOOLS_DIR" && sha256sum -c SHA256SUMS.llama_rank_increment_profile)
fi
if [[ ! -f "$PROFILE" ]]; then
  echo 'Set LLAMA_RANK_INCREMENT_PROFILE to your private server configuration.' >&2
  exit 2
fi
source "$PROFILE"
: "${RANK_INCREMENT_PYTHON:?}" "${RANK_INCREMENT_REPO:?}" "${RANK_INCREMENT_RUN:?}"
: "${RANK_INCREMENT_FP64:?}" "${RANK_INCREMENT_DA:?}" "${RANK_INCREMENT_QERA:?}"
: "${RANK_INCREMENT_HARNESS:?}" "${RANK_INCREMENT_WORD_REFERENCE:?}" "${RANK_INCREMENT_DESTINATION:?}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$RANK_INCREMENT_PYTHON" -B "$TOOLS_DIR/llama_rank_increment_v1.py" "$COMMAND" \
  --repo-dir "$RANK_INCREMENT_REPO" \
  --run-dir "$RANK_INCREMENT_RUN" \
  --source-fp64-dir "$RANK_INCREMENT_FP64" \
  --source-da-dir "$RANK_INCREMENT_DA" \
  --official-qera-root "$RANK_INCREMENT_QERA" \
  --harness-source "$RANK_INCREMENT_HARNESS" \
  --word-reference-dir "$RANK_INCREMENT_WORD_REFERENCE" \
  --output-dir "$RANK_INCREMENT_DESTINATION" \
  "$@"
