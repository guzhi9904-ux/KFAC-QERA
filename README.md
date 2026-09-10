# QERA-style MXINT4 Full-A / Full-G Factorial Experiments

## 当前研究入口（2026-09-11）

下文保留初始 2×3 实验的用法；后续隔离实验不要与初始入口混用。

- [数值诊断与同根 FP64 / 四个 rank 双 PPL](tools/precision_audit/README.md)：最近发布工具的唯一源码归档，各阶段共用一份冻结 helper。
- [Qwen2.5-7B Base MXINT3](experiments/qwen25_base_isolation_v1/README.md)：包含离线数据准备与续跑；Full-A 数值异常仍未解决，不能当作完成的跨模型结果。
- [问题与处理日志](docs/research/ISSUE_LOG.md) · [PPL 阶段总账](docs/research/PPL_SUMMARY.md)。

**正在运行或需要续跑的服务器目录不要 `git pull`、覆盖脚本或升级环境。** 本次 GitHub 更新不要求重新部署正在跑的双 PPL 实验。工具按哈希冻结，新任务请使用独立目录与固定 commit。

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
configs/                 Qwen、Llama 与 RTX 4090 服务器的 smoke/full 配置
docs/METHOD.md           目标函数、G^(1/2) 与闭式解
docs/SERVER_RUN.md       单机和 Slurm 运行方法
reference_results/       当前 Qwen2.5-1.5B 与旧对角实验的小型结果文件
scripts/                 服务器运行脚本
src/qera_exp/            实验实现
tests/                    不加载大模型的单元测试
```

模型权重、token 缓存、A/G 矩阵、低秩因子和逐窗口评测结果默认写到 `RUN_DIR`，并被 `.gitignore` 排除。raw A/G 默认保留，用于审计、归档和受控的后续求解。

## 附加隔离实验

不含 G、严格区分 Full-A 与 Diag-A，并对齐官方 QERA 校准集和求解器的 rank 8/16/32/64 实验位于 [experiments/qera_original_a_isolation](experiments/qera_original_a_isolation/README.md)。该目录有独立环境、配置、断点和运行脚本，不修改原 2×3 A/G 实验实现。

## 安装

```bash
git clone https://github.com/guzhi9904-ux/KFAC-QERA.git
cd KFAC-QERA
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

RTX 4090 24GB、112GB RAM 的配置和当前缓存资源对应命令见 [docs/SERVER_4090_112G.md](docs/SERVER_4090_112G.md)。

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
