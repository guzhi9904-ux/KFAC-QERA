# QERA 原设置下的 Full-A / Diag-A 隔离实验

本目录是附加实验，不导入、修改或复用 `src/qera_exp` 的实验逻辑。目标是在同一份权重、同一批校准 token、同一个 MXINT4 权重量化结果上，只改变输入二阶统计中的非对角项，比较：

- `BF16`
- `W4_MXINT`
- `QERA_DIAG_R{8,16,32,64}`
- `QERA_FULL_R{8,16,32,64}`

主配置为 `configs/llama3.1-8b.yaml`。Llama 模型路径已经固定为：

```text
/share/home/tm902089733300000/a913520780/chengkang/modelzoo/Meta-Llama-3.1-8B
```

## 数学定义

令原权重为 `W`，官方 MXINT4 量化权重为 `Wq`，量化误差为 `E = W - Wq`。输入行向量的二阶矩为：

```text
Rxx = E[x^T x]
S_full = Rxx^(1/2)
S_diag = diag(sqrt(diag(Rxx)))
```

对 `S E^T` 做截断 SVD：

```text
S E^T = U Sigma V^T
A_r = S^(-1) U_r
B_r = Sigma_r V_r^T
y_hat = x Wq^T + x A_r B_r
```

Full-A 与 Diag-A 的唯一区别是 `S` 是否保留 `Rxx` 的非对角项；两者都不使用 G，也不加 damping。rank 64 保存一次完整 A/B，rank 8、16、32 使用其严格前缀。

## 与官方设置的对齐

| 项目 | 本实验 |
|---|---|
| 官方源码 | `ChengZhang-98/QERA@bd7fc86a2e44d41f95b9b0421f27f5624dd37064` |
| 校准集 | `DKYoon/SlimPajama-6B` 前 5120 条原始文本，官方预处理后的前 256 个窗口 |
| 数据获取 | 锁定实际 dataset revision，streaming 读取固定前缀 |
| 序列长度 | 2048 |
| 校准 / PPL batch size | 4 / 4 |
| 评测集 | `Salesforce/wikitext` / `wikitext-2-raw-v1` test |
| 权重量化 | 官方 `mxint_quantizer(width=4, block_size=32, block_axis=-1)` |
| Full-A 平方根 | 官方 SciPy blocked `sqrtm` 路径，FP64 |
| Full-A 外积 / 累加 | FP32 / FP64 |
| Diag-A 平方和 / 累加 | FP32 / FP32（官方 diagonal 路径） |
| 求解 | 官方 `_compute_scales_and_error_for_fc`，FP32 |
| ranks | 8、16、32、64 |

`doctor` 会同时核验官方 commit 和五个关键源码文件的 SHA-256。官方该 commit 没有 Qwen 模型适配，所以 Llama 结果是主对照；`qwen2.5-7b.yaml` 仅作为相同协议向 Qwen 的迁移实验，不能冒充论文复现。

## 服务器初始化

在 KFAC-QERA 仓库根目录执行：

```bash
conda env create -f experiments/qera_original_a_isolation/environment.yml
conda activate qera-original-a
bash experiments/qera_original_a_isolation/scripts/setup_official_qera.sh
```

新 Conda 环境固定官方使用的 Python、PyTorch 和 Transformers 版本，不会改变现有 `frontierguard` 环境。

### 下载并冻结 SlimPajama 与 WikiText-2

服务器目前没有 SlimPajama。数据下载必须单独做一次；不要在正式长任务中临时联网：

```bash
export HF_HOME=/share/home/tm902089733300000/a913520780/chengkang/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

python experiments/qera_original_a_isolation/run.py \
  --config experiments/qera_original_a_isolation/configs/llama3.1-8b.yaml \
  prepare-data --allow-download
```

此步骤不会下载完整的 48 个 SlimPajama 分片。它先解析并记录 Hugging Face dataset revision，再按数据集顺序 streaming 读取官方使用的前 `20 × 256 = 5120` 条原始文本，随后直接调用官方 QERA 的拼接、tokenize 和 2048-token 分块函数，最后保存前 256 个校准窗口、原始文本前缀哈希和 token 文件 SHA-256。通常只需读取第一个数据分片；完成后实验不再依赖数据网络。

该优化保持校准文本及预处理定义不变，但数据获取入口不是官方非流式 `load_dataset` 全量下载路径。因此结果应标为“官方数据前缀与预处理对齐”，而不是“未经改动的官方数据加载入口复现”。

## 审查与运行

```bash
CONFIG=experiments/qera_original_a_isolation/configs/llama3.1-8b.yaml
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" doctor
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" plan
```

正式运行：

```bash
export CONFIG=experiments/qera_original_a_isolation/configs/llama3.1-8b.yaml
nohup bash experiments/qera_original_a_isolation/scripts/run_server.sh \
  > /share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-original-a-isolation/pipeline.log 2>&1 &
```

查看进度：

```bash
tail -f /share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-original-a-isolation/pipeline.log
```

服务器关机后重复同一条 `nohup` 命令即可续跑。collect 每 32 个窗口做一次一致性 checkpoint；root 和 solve 按输入组/投影层保存；PPL 每个窗口写盘。

也可以单独运行某一阶段：

```bash
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" collect --shard 0
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" roots --shard 0
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" solve --shard 0
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" evaluate --only QERA_FULL_R32
python experiments/qera_original_a_isolation/run.py --config "$CONFIG" summarize
```

## 保留内容

运行目录中长期保留：

- `data/*.safetensors`：冻结的校准和评测 token；
- `statistics/raw/`：每个共享输入组的 FP64 `Rxx_sum`、直接对角和计数；
- `statistics/roots/`：Full-A 与 Diag-A 根；
- `corrections/{full,diag}/`：rank 64 的 A/B；
- `evaluation/wikitext2_per_window.csv`：逐窗口配对 NLL；
- `evaluation/ppl_summary_wikitext2.csv`：最终 PPL 表；
- `state/`、`plan.json`、`doctor.json`：断点与审计记录。

论文与源码：<https://arxiv.org/abs/2410.06040>，<https://github.com/ChengZhang-98/QERA>。
