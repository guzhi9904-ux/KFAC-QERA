#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local model directory}"
: "${RUN_DIR:?Set RUN_DIR to an output directory outside the git repository}"
CONFIG="${CONFIG:-configs/qwen_smoke.yaml}"

qera-exp --config "$CONFIG" doctor
qera-exp --config "$CONFIG" prepare-data
qera-exp --config "$CONFIG" quantize
qera-exp --config "$CONFIG" plan

SHARDS=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"state/shard_plan.json")))["shard_count"])')
for SHARD in $(seq 0 $((SHARDS - 1))); do
  qera-exp --config "$CONFIG" collect --shard "$SHARD" --solve
done

qera-exp --config "$CONFIG" evaluate --dataset wikitext2
qera-exp --config "$CONFIG" evaluate --dataset c4
qera-exp --config "$CONFIG" analyze
