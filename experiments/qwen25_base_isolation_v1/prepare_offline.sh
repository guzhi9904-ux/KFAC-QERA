#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../../.." && pwd)"
RAW="$BASE/qwen_offline_raw_b5f90f4"
WT="$BASE/huggingface_cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
test -f "$RAW/calibration.jsonl.gz"
test -f "$RAW/calibration.source.json"
test -f "$WT/wikitext-test.arrow"
exec bash "$HERE/run_server.sh" prepare --offline-raw-dir "$RAW" --wikitext-cache-dir "$WT" "$@"
