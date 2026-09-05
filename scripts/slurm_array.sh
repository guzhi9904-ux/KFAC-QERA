#!/usr/bin/env bash
#SBATCH --job-name=qera-ag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-0
#SBATCH --output=qera-ag-%A_%a.out

set -euo pipefail
: "${CONFIG:?Export CONFIG before sbatch}"
: "${MODEL_PATH:?Export MODEL_PATH before sbatch}"
: "${RUN_DIR:?Export RUN_DIR before sbatch}"

source .venv/bin/activate
qera-exp --config "$CONFIG" collect --shard "$SLURM_ARRAY_TASK_ID" --solve
