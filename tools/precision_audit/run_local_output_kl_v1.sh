#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-doctor}"
if (( $# > 0 )); then shift; fi
(cd -- "$TOOLS_DIR" && sha256sum -c SHA256SUMS.local_output_kl)
PROFILE="${LOCAL_OUTPUT_KL_PROFILE:-$TOOLS_DIR/server_profile.sh}"
if [[ "$PROFILE" == "$TOOLS_DIR/server_profile.sh" && -f "$TOOLS_DIR/SHA256SUMS.local_output_kl_profile" ]]; then
  (cd -- "$TOOLS_DIR" && sha256sum -c SHA256SUMS.local_output_kl_profile)
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "Set LOCAL_OUTPUT_KL_PROFILE to your private server configuration file." >&2
  exit 2
fi
# User-owned configuration stays outside the public source tree.
source "$PROFILE"
: "${LOCAL_OUTPUT_PYTHON:?}" "${LOCAL_OUTPUT_REPO:?}" "${LOCAL_OUTPUT_RUN:?}"
: "${LOCAL_OUTPUT_FP64:?}" "${LOCAL_OUTPUT_RANK64:?}" "${LOCAL_OUTPUT_QERA:?}"
: "${LOCAL_OUTPUT_HARNESS:?}" "${LOCAL_OUTPUT_WORD_REFERENCE:?}" "${LOCAL_OUTPUT_DESTINATION:?}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$LOCAL_OUTPUT_PYTHON" -B \
  "$TOOLS_DIR/local_output_kl_v1.py" "$COMMAND" \
  --repo-dir "$LOCAL_OUTPUT_REPO" \
  --run-dir "$LOCAL_OUTPUT_RUN" \
  --source-fp64-dir "$LOCAL_OUTPUT_FP64" \
  --source-rank64-dir "$LOCAL_OUTPUT_RANK64" \
  --official-qera-root "$LOCAL_OUTPUT_QERA" \
  --harness-source "$LOCAL_OUTPUT_HARNESS" \
  --word-reference-dir "$LOCAL_OUTPUT_WORD_REFERENCE" \
  --output-dir "$LOCAL_OUTPUT_DESTINATION" \
  "$@"
