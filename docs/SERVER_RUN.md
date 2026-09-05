# Server run

## 1. 环境

```bash
git clone https://github.com/guzhi9904-ux/KFAC-QERA.git
cd KFAC-QERA
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
```

先按服务器 CUDA 版本安装合适的 PyTorch。模型建议提前下载到本地只读目录。

```bash
export MODEL_PATH=/srv/models/Qwen-or-Llama
export RUN_DIR=/srv/experiments/qera/qwen-smoke
export HF_HOME=/srv/cache/huggingface
CONFIG=configs/qwen_smoke.yaml
```

## 2. Smoke run

```bash
qera-exp --config "$CONFIG" doctor
qera-exp --config "$CONFIG" prepare-data
qera-exp --config "$CONFIG" quantize
qera-exp --config "$CONFIG" plan
```

查看分片数量和预计 dense A/G 主存：

```bash
python -c 'import json,os; p=json.load(open(os.path.join(os.environ["RUN_DIR"],"state/shard_plan.json"))); print(p["shard_count"]); [print(s["index"],s["module_count"],round(s["estimated_dense_accumulator_bytes"]/2**30,2)) for s in p["shards"]]'
```

逐个完成所有 shard：

```bash
SHARDS=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"state/shard_plan.json")))["shard_count"])')
for SHARD in $(seq 0 $((SHARDS-1))); do
  qera-exp --config "$CONFIG" collect --shard "$SHARD" --solve
done
```

之后运行：

```bash
qera-exp --config "$CONFIG" evaluate --dataset wikitext2
qera-exp --config "$CONFIG" evaluate --dataset c4
qera-exp --config "$CONFIG" analyze
qera-exp --config "$CONFIG" status
```

## 3. Full run

Smoke 通过后换新 `RUN_DIR`，再改为：

```bash
CONFIG=configs/qwen_full.yaml
# 或 CONFIG=configs/llama3_full.yaml
# 宽 MLP 可行性运行：CONFIG=configs/llama3_full_cholesky.yaml
```

不得把 smoke 的 token、quantization、statistics 或 corrections 目录复制到 full run。

## 4. Slurm

先在登录节点执行 `doctor`、`prepare-data`、`quantize` 和 `plan`，再根据 `shard_plan.json` 的 `shard_count` 修改 `scripts/slurm_array.sh` 中的 array 上限。

```bash
sbatch --export=ALL,CONFIG="$CONFIG" scripts/slurm_array.sh
```

数组任务全部完成后，再单独提交 evaluation。

## 5. 本地 dataset cache

若计算节点不能联网，先用 `datasets.save_to_disk()` 保存数据，然后在 YAML 对应 role 中设置：

```yaml
local_path: /srv/datasets/wikitext-2-raw-v1
```

`local_path` 可指向 `Dataset` 或 `DatasetDict`。C4 可使用本地 validation 子集。

## 6. 资源注意

full A/G 对宽 MLP 很重。单个线性层的 FP64 accumulator 约为：

```text
8 × (d_in² + d_out² + d_in + d_out) bytes
```

`runtime.max_dense_ram_gib_per_shard` 只限制 accumulator 估算值，不包含模型、autograd saved tensors、分解 workspace 和 Python 进程开销。服务器实际可用主存应留出至少 30% 余量。`plan` 同时报告保留 raw A/G 所需的预计磁盘容量和当前空闲空间。

Llama-3 8B 的宽 MLP 会让 FP64 `eigh` 很慢。建议：

1. 先跑 2 层 smoke；
2. 保留一份 `eigh` smoke 作为旧实验对齐；
3. 若全层不可行，另建新 `RUN_DIR` 使用 `root_method: cholesky`；
4. 不要在同一个运行目录中途改变 root method、rank 或 tokenizer。

## 7. 断点

- `quantize` 会校验并复用已有量化文件；checkpoint 变更时直接报错。
- 完成的 collection shard 会由 `state/collect_shard_XXXX.json` 跳过。
- evaluation 以 dataset/configuration/window 为键续跑。
- 默认保留该模块的 raw A/G。只有明确不再需要重新求解或复查时，才把 `cleanup_raw_after_solve` 设置为 `true`。

RTX 4090 24GB / 112GB RAM 的现成配置、缓存路径和模型选择见 [SERVER_4090_112G.md](SERVER_4090_112G.md)。
