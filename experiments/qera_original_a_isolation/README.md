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
| Full-A 平方根 | 官方 SciPy blocked `sqrtm` 路径，FP64；复数结果按官方 float cast 语义丢弃虚部并记录诊断 |
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

### 已完成求解后，双 RTX 4090 仅续跑评测

保留原始 YAML 和 `run_dir`，在原环境中执行：

```bash
export CONFIG=experiments/qera_original_a_isolation/configs/llama3.1-8b.yaml
bash experiments/qera_original_a_isolation/scripts/evaluate_dual_4090.sh
```

该入口仅运行 WikiText-2 的 10 组评测，默认 batch size 8，模型上下文长度仍为 2048。权重使用
`device_map=balanced`，两张卡的权重预算各 10 GiB，为 logits 和激活留出空间。
只改变加载模型时的内存预算，原实验指纹、数据、统计组和补偿保持原样。
可通过 `EVAL_BATCH_SIZE=4` 调整评测批次；用 `--only BF16` 仅评测基线。
每次加载的实际设备分配和运行参数记录在 `evaluation/runtime/`。
进度日志同时显示两张卡的 PyTorch 峰值 allocated/reserved 显存。

评测默认对每个窗口的全部 2047 个预测位置一次执行完整词表交叉熵，然后沿用原先的
逐窗口 FP32 NLL 求和与 mask。一次前向仍输入整个 batch 的完整 2048-token 窗口。
这样避免完整 shifted-logits 副本以及整批 log-softmax 工作区，并在下批前向之前释放 logits。
如需进一步减少 CE 工作区，可设置 `EVAL_CE_CHUNK_TOKENS=256`，该值只改变
loss 计算的分块大小，不改变模型前向的序列长度或 attention 上下文。此处是附加评测脚本的
显存优化，不应标为未经改动的官方评测入口；分块与原实现的数值一致性由
测试覆盖，跨设备的末位浮点差异仍可能存在。进程中断后用相同入口续跑。

### 完整流水线

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

## 独立官方 harness Word PPL 核对（先 BF16）

这一步不调用上面的 token-window 评测器。QERA Table 16 报告的是 **Word ppl**，
原来的 `138 × 2047 = 282486` 是预测 token 数；两者不能直接比较。

新增入口锁定官方 QERA commit `bd7fc86a2e44d41f95b9b0421f27f5624dd37064`
所绑定的 harness 子模块 commit `3823cfec41c016378acbcc8616dd1ac92c15edd4`，
来自 `ChengZhang-98/lm-evaluation-harness`，不安装浮动最新版。
运行前核验 task、预处理、HFLM、评测器、指标与 rolling 工具文件的 SHA256。
WikiText 文档预处理、rolling likelihood 与 Word PPL 聚合均由该 harness 执行。
数据为 `EleutherAI/wikitext_document_level / wikitext-2-raw-v1 / test`，
不用之前冻结的 138 个窗口，不需要重新下载 SlimPajama。

模型使用已有本地 BF16 权重，eager attention，2048-token 上下文、无 chat template，
调用官方 QERA 的 `auto-balanced` 设备分配。默认 harness batch size 1，
与官方 `HFLM(model)` 的有效默认值一致；两卡分配权重不代表数据并行。
不使用自定义 CE 分块。`bootstrap_iters=0` 只省略标准误 bootstrap，不改变 PPL 点估计。
权重、tokenizer、数据文本与运行协议都记录指纹。模型文件哈希阶段会逐文件输出日志；
正式评测进度由 harness 的文档级进度条输出。

在 **现有 qera-original-a 环境**、仓库根目录执行：

```bash
bash experiments/qera_original_a_isolation/scripts/setup_harness.sh
export CUDA_VISIBLE_DEVICES=0,1
export CONFIG=experiments/qera_original_a_isolation/configs/llama3.1-8b.yaml
bash experiments/qera_original_a_isolation/scripts/run_harness_word_ppl.sh \
  --stage bf16 --allow-download
```

安装脚本固定当前 torch、transformers、datasets 等核心包版本为 pip constraints，
不升级 PyTorch/CUDA；存在依赖冲突时停止而非放宽约束。只补装 pinned harness 和缺少的依赖，
安装过程直接显示下载进度，沿用服务器现有 pip 镜像配置。
`--allow-download` 允许获取 harness 的文档级 WikiText 数据集；本地模型仍强制离线加载。
该参数只控制下载权限，不改变评测协议。数据准备好后可以不加。

先查看 `evaluation_harness_word_ppl/bf16_reference_check.json`。
论文 BF16 参考值为 **7.55**；默认“接近”阈值为绝对差 **0.05**，
这是本实验的工程检查阈值，不是论文规定的误差范围，也不代表已经严格复现全部论文设置。
不接近则保存结果、返回 `NEEDS_REVIEW`（退出码 2），不启动量化对照。
确认基线后，在同一环境、同样参数下执行：

```bash
bash experiments/qera_original_a_isolation/scripts/run_harness_word_ppl.sh \
  --stage compare --allow-download
```

它校验并复用已完成的 BF16 结果，再运行：

| 配置 | 论文 Word PPL 参考 | 使用内容 |
| --- | ---: | --- |
| W4_MXINT | 8.78 | 官方 MXINT4，block size 32，重新量化已有 BF16 权重 |
| QERA_DIAG_R32 | 8.45 | 已保存 Diag-A 补偿的 rank32 前缀 |
| QERA_FULL_R32 | 8.33 | 已保存 Full-A 补偿的 rank32 前缀 |

比较阶段验证所有低秩文件的哈希与形状，不访问或重算 roots、A 或校准统计量。
这里评测的是**已有 A-isolation 产物**，量化与补偿加载继续沿用该独立实验的实现；
不能把它描述成从校准到评测全部未经修改的官方 PTQ 流水线。
`--stage all` 可自动执行上述基线检查和通过后的三个对照，默认仍然只跑 `bf16`。

输出统一放在原 run_dir 下新增的 `evaluation_harness_word_ppl/`：

- `protocol.json`：源码、模型、数据、环境及评测参数；
- `BF16/results.json` 等：完整 harness 原始结果和逐文档 samples；
- `BF16/complete.json` 等：配置完成记录与结果校验和；
- `bf16_reference_check.json`：7.55 基线差值检查；
- `word_ppl_summary.csv`：独立 Word PPL 结果表。

旧 `evaluation/` 不修改。新入口按**完整配置**续跑：完成的配置校验后跳过，
被中断的那一个配置重新评测，不支持逐文档断点。更改 batch size、源码、权重、
数据或核心环境后应使用新的 `--output-dir`，避免混用协议。
不要同时启动多个进程写同一个输出目录。

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
