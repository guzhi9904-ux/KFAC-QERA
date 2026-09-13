#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-doctor}"
if (( $# > 0 )); then shift; fi
(cd -- "$TOOLS_DIR" && sha256sum -c SHA256SUMS.g_permutation_kl)
PROFILE="${G_PERMUTATION_PROFILE:-$TOOLS_DIR/server_profile.sh}"
if [[ "$PROFILE" == "$TOOLS_DIR/server_profile.sh" && -f "$TOOLS_DIR/SHA256SUMS.g_permutation_kl_profile" ]]; then
  (cd -- "$TOOLS_DIR" && sha256sum -c SHA256SUMS.g_permutation_kl_profile)
fi
if [[ ! -f "$PROFILE" ]]; then
  echo 'Set G_PERMUTATION_PROFILE to your private server configuration.' >&2
  exit 2
fi
source "$PROFILE"
: "${G_PERMUTATION_PYTHON:?}" "${G_PERMUTATION_REPO:?}" "${G_PERMUTATION_RUN:?}"
: "${G_PERMUTATION_FP64:?}" "${G_PERMUTATION_RANK64:?}" "${G_PERMUTATION_LOCAL:?}"
: "${G_PERMUTATION_QERA:?}" "${G_PERMUTATION_HARNESS:?}" "${G_PERMUTATION_WORD_REFERENCE:?}" "${G_PERMUTATION_DESTINATION:?}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$G_PERMUTATION_PYTHON" -B "$TOOLS_DIR/g_permutation_kl_v1.py" "$COMMAND" \
  --repo-dir "$G_PERMUTATION_REPO" \
  --run-dir "$G_PERMUTATION_RUN" \
  --source-fp64-dir "$G_PERMUTATION_FP64" \
  --source-rank64-dir "$G_PERMUTATION_RANK64" \
  --source-local-dir "$G_PERMUTATION_LOCAL" \
  --official-qera-root "$G_PERMUTATION_QERA" \
  --harness-source "$G_PERMUTATION_HARNESS" \
  --word-reference-dir "$G_PERMUTATION_WORD_REFERENCE" \
  --output-dir "$G_PERMUTATION_DESTINATION" "$@"
