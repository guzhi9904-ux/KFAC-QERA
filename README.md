# QERA-style MXINT4 Full-A / Full-G Factorial Experiments

用于检查低秩量化误差补偿中，输入二阶矩 `A` 与输出梯度二阶矩 `G` 的对角/完整矩阵近似是否真的改善语言模型 PPL。

仓库实现同一组 MXINT4-B32 权重上的 2×3 实验：

| 方法 | A | G |
|---|---|---|
| `AD_GI` | diagonal | identity |
| `AD_GD` | diagonal | diagonal |
| `AD_GF` | diagonal | full |
| `AF_GI` | full | identity |
| `AF_GD` | full | diagonal |
| `AF_GF` | full | full |

支持 Hugging Face `AutoModelForCausalLM` 中采用 Llama/Qwen 投影命名的模型，默认覆盖每层的 `q/k/v/o/gate/up/down_proj`。模块由运行时自动发现，不固定层数、hidden size 或 GQA 维度。

## 目录

```text
configs/                 Qwen、Llama 3 的 smoke/full 配置
docs/METHOD.md           目标函数、G^(1/2) 与闭式解
docs/SERVER_RUN.md       单机和 Slurm 运行方法
reference_results/       当前 Qwen2.5-1.5B 与旧对角实验的小型结果文件
scripts/                 服务器运行脚本
src/qera_exp/            实验实现
tests/                    不加载大模型的单元测试
```

模型权重、token 缓存、A/G 矩阵、低秩因子和逐窗口评测结果默认写到 `RUN_DIR`，并被 `.gitignore` 排除。

## 安装

```bash
git clone <your-repository-url>
cd qera_mxint4_full_ag
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

安装与你的 CUDA 驱动匹配的 PyTorch。建议先把模型下载到服务器本地，实验配置默认 `local_files_only: true`。

## 最小运行

```bash
export MODEL_PATH=/srv/models/your-qwen-or-llama
export RUN_DIR=/srv/experiments/qera-smoke
CONFIG=configs/qwen_smoke.yaml

qera-exp --config "$CONFIG" doctor
qera-exp --config "$CONFIG" prepare-data
qera-exp --config "$CONFIG" quantize
qera-exp --config "$CONFIG" plan
qera-exp --config "$CONFIG" collect --shard 0 --solve
qera-exp --config "$CONFIG" evaluate
qera-exp --config "$CONFIG" analyze
```

`plan` 可能产生多个 shard。必须完成 `RUN_DIR/state/shard_plan.json` 中的所有 shard 后再评测。完整流程见 [docs/SERVER_RUN.md](docs/SERVER_RUN.md)。

## 推荐顺序

1. 用 `qwen_smoke.yaml` 或 `llama3_smoke.yaml` 跑 2 层、8 个窗口。
2. 检查 BF16 teacher PPL、纯 MXINT4 PPL、因子 shape、有限值和 rank 前缀。
3. 再使用 `qwen_full.yaml` 或 `llama3_full.yaml` 跑全层；宽 MLP 的可行性版本为 `llama3_full_cholesky.yaml`。
4. 每个模型使用独立 `RUN_DIR`，不要跨 tokenizer 复用 token 文件或校正因子。

## 输出

关键输出位于：

- `evaluation/ppl_summary_wikitext2.csv`
- `evaluation/ppl_summary_c4.csv`
- `analysis/figures/ppl_vs_rank_wikitext2_focused.png`
- `analysis/figures/ppl_vs_rank_c4_focused.png`
- `analysis/rank_energy_by_projection.csv`
- `analysis/figures/rank_energy_by_projection.png`
- `analysis/factorial_contrasts_*.csv`

当前本地实验的对照结果放在 [reference_results](reference_results/README.md)。这些结果只用于检查趋势，不应与新模型的 tokenizer/PPL 直接横向比较。

## 范围

- 权重：MXINT4，4-bit signed shared-exponent block，block size 32，fake quantization。
- 激活：默认 BF16，不做激活量化。
- 统计：clean teacher 上、true next-token CE 的 token-wise empirical-Fisher/KFAC 因子。
- 部署：`Wq x + L(R^T x)` 独立低秩支路。
- PPL：`exp(sum NLL / sum valid prediction tokens)`。

本仓库没有提交任何模型权重、Hugging Face token 或大体积实验中间文件。
