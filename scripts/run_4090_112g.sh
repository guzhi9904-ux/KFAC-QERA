#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a cached Hugging Face model id or local model directory}"
: "${RUN_DIR:?Set RUN_DIR to an output directory outside the git repository}"

CONFIG="${CONFIG:-configs/rtx4090_112g_smoke.yaml}"
RUN_C4="${RUN_C4:-1}"
CPU_THREADS="${CPU_THREADS:-14}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

qera-exp --config "$CONFIG" doctor
qera-exp --config "$CONFIG" prepare-data --role calibration
qera-exp --config "$CONFIG" prepare-data --role wikitext2
if [[ "$RUN_C4" == "1" ]]; then
  qera-exp --config "$CONFIG" prepare-data --role c4
fi

qera-exp --config "$CONFIG" quantize
qera-exp --config "$CONFIG" plan

python -c 'import json,os; p=json.load(open(os.path.join(os.environ["RUN_DIR"],"state/shard_plan.json"))); print("shards=%d retained_raw_GiB=%.2f disk_free_GiB=%.2f" % (p["shard_count"],p["estimated_retained_raw_statistics_bytes"]/2**30,p["disk_free_bytes_at_plan"]/2**30)); [print("WARNING:", x) for x in p["warnings"]]'

SHARDS=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"state/shard_plan.json")))["shard_count"])')
for SHARD in $(seq 0 $((SHARDS - 1))); do
  COLLECT_STATE=$(printf "%s/state/collect_shard_%04d.json" "$RUN_DIR" "$SHARD")
  SOLVE_STATE=$(printf "%s/state/solve_shard_%04d.json" "$RUN_DIR" "$SHARD")
  if [[ ! -f "$COLLECT_STATE" ]]; then
    qera-exp --config "$CONFIG" collect --shard "$SHARD"
  fi
  if [[ ! -f "$SOLVE_STATE" ]]; then
    qera-exp --config "$CONFIG" solve --shard "$SHARD"
  fi
done

qera-exp --config "$CONFIG" evaluate --dataset wikitext2
if [[ "$RUN_C4" == "1" ]]; then
  qera-exp --config "$CONFIG" evaluate --dataset c4
  qera-exp --config "$CONFIG" analyze
else
  qera-exp --config "$CONFIG" status
fi
