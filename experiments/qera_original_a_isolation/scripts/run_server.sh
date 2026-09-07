#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$HERE/configs/llama3.1-8b.yaml}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/share/home/tm902089733300000/a913520780/chengkang/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "$HERE/run.py" --config "$CONFIG" doctor
python "$HERE/run.py" --config "$CONFIG" plan

RUN_DIR="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["run_dir"])' "$CONFIG")"
SHARDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_count"])' "$RUN_DIR/plan.json")"
mkdir -p "$RUN_DIR"

for ((SHARD=0; SHARD<SHARDS; SHARD++)); do
  echo "[runner] shard=$SHARD/$((SHARDS-1)) collect"
  python "$HERE/run.py" --config "$CONFIG" collect --shard "$SHARD"
  echo "[runner] shard=$SHARD/$((SHARDS-1)) roots"
  python "$HERE/run.py" --config "$CONFIG" roots --shard "$SHARD"
  echo "[runner] shard=$SHARD/$((SHARDS-1)) solve"
  python "$HERE/run.py" --config "$CONFIG" solve --shard "$SHARD"
done

python "$HERE/run.py" --config "$CONFIG" evaluate
