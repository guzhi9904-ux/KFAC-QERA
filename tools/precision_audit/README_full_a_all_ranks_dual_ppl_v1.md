# Full-A 四个 rank，双 PPL 评估（独立 v1）

本轮：Llama-3.1-8B / MXINT3 / Full-A / GI、DG、GF / rank 8、16、32、64。
每个协议 14 个主配置：BF16、W3-only、3 方法 × 4 rank。
不包含 DA、MXINT4、Qwen 或 FP64 A-root 构造实验。

## 固定输入与不变项

- 原模型、tokenizer、Wq、A/G 统计、原始代码及已完成 A5 输出只读。
- 新因子来源：`qera_diagnostics/full_a_all_precision_r8_v1/factors/` 的全部 672 份已保存 FP64 rank64 因子。
- 不重新收集 A/G，不重新量化，不做新 SVD 或 inverse solve，不修改 root 定义、floor、阻尼或 shrinkage。
- 检查阶段会读取原统计并按原协议重建 G-root，与 A5 记录的 root bits 精确比较。这不是重新收集 G，也不是 FP64 A-root 构造。
- 保存一份 BF16 rank64 因子，按 rank 取前缀并 contiguous 部署。所有前缀从源 FP64 直接转 BF16，仍采用 `(input @ A) @ B` hook，不合并进 Wq。
- 两种 PPL 共用同一份 BF16 因子。各自核验模型参数/buffer/device_map 不随 correction 配置改变；跨协议核对 correction bits。

## 高 rank 检查，不重新做 SVD

源 A5 的独立 full SVD 已完成，保存了 r8 尾能量证据。本入口检查：

1. 源 672 份因子的身份、哈希、shape、dtype、r8 直接舍入一致性。
2. 精确相同的 A/G root bits。
3. 白化后的左列正交、右行正交、投影残差，以及奇异分量能量降序。
4. r8/16/32/64 的加权残差、随 rank 非增，以及从源 r8 尾能量减去新增正交分量能量后的吻合度。
5. BF16 舍入后有限性及加权误差 proxy。目标上升保留 flag；不自动改数值方法或忽略失败。

前四项超阈值直接停止、不提交该模块。这是对原 SVD 证据的延伸检查，不宣称重新独立证明了一次全谱 SVD 最优性。BF16 因子 proxy 也不是 BF16 激活执行误差。

## 评估协议与控制

### token-PPL

- 冻结的 138 × 2048 窗口；282486 prediction tokens；batch=8；CE chunk=256。
- 复用原 BF16 / balanced 模型加载器、eager attention 和原 NLL 累加顺序。
- 先完整回放三组 FP64 r8，与 A5 的实际部署 tensor bits 精确比较。
- 工程回放阈值：PPL 差 ≤1e-5 且最大窗口 NLL 差 ≤1e-3。不是统计显著性阈值；报告实际差值。
- 三组通过后才继续 BF16、W3 和更高 rank。r8 这次有意重新前向，是对新 rank-aware 路径的回归测试，不是独立重复实验。

### word-PPL

- 之前的 4096-context、BF16、eager、auto-balanced、无 chat template 协议。
- 固定 QERA `bd7fc86a2e44d41f95b9b0421f27f5624dd37064` 和 harness `3823cfec41c016378acbcc8616dd1ac92c15edd4`，验证 6 个 harness 文件哈希。
- `EleutherAI/wikitext_document_level / wikitext-2-raw-v1 / test`，完整文档，无 limit。
- 与旧 QERA wrapper 一样使用 `HFLM(model)`，检查有效 `max_length=4096`，保存实际 HFLM batch size；不要把 wrapper 的 `batch_size="auto"` 误解成已经确认 HFLM 在自动 batching。
- 调用固定 harness 的 `simple_evaluate`，原 task 的预处理、rolling likelihood、word-perplexity 聚合不改。仅将 `bootstrap_iters` 设为 0（不计算 stderr），保留全部原始 samples，便于保存文档级 NLL/词数。不给出统计显著性结论。
- 旧 `protocol.json` 的数据文本指纹、模型文件身份、包版本必须匹配；旧 BF16 和 MXINT3 FULL_GD_R8 原始 JSON 必须存在并通过哈希/协议绑定验证。
- 新入口先回放上述两个旧配置，word-PPL 绝对差均需 ≤1e-4 才评估新因子。不能拿小数仅 4 位的旧 CSV 替代原始 results/complete/protocol JSON。
- 每份完整结果保存 harness JSON、逐文档 NLL/计分词数，按原 task 返回的 word 指标复算总 PPL。不是从 token-PPL 换分母，也不是新的独立数据域。

## 服务器准备

上传 `full_a_all_ranks_dual_ppl_v1.zip` 到 BASE。选择空闲双 4090，各至少 18 GiB 可用。不要与 Qwen 或其他实验争用同一对 GPU。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
mkdir -p "$BASE/full_a_all_ranks_dual_ppl_v1_tools"
unzip -n "$BASE/full_a_all_ranks_dual_ppl_v1.zip" -d "$BASE/full_a_all_ranks_dual_ppl_v1_tools"
nvidia-smi
```

启动器使用原 `qera-original-a` 环境 Python 与旧 word-PPL 使用的隔离依赖目录 `qera_official_bf16_pydeps`。不 pip/conda 安装或升级，不 git pull，不覆盖原仓库。

默认新输出：

```text
/share/home/tm902089733300000/a913520780/chengkang/qera_diagnostics/full_a_all_ranks_dual_ppl_v1/
```

### 1. doctor（只读预检，无模型评估）

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
bash "$BASE/full_a_all_ranks_dual_ppl_v1_tools/run_full_a_all_ranks_dual_ppl_v1.sh" doctor
```

会校验源因子和旧 word-PPL 协议，因此哈希阶段可能需要几分钟，每 30 秒有心跳。
如果提示缺少旧 word reference，先定位真实的 `protocol.json` 与 `BF16/results.json`、`mxint3/FULL_GD_R8/results.json` 所在根目录，再给所有命令追加：

```bash
--word-reference-dir /实际的旧4096评估根目录
```

默认查找 `.../qera_runs/official-word-ppl-4096-existing-artifacts/`。不要为了通过预检重命名 2048-context 旧结果，也不要把汇总 CSV 伪装成原始 JSON。文件缺失时先回传报错核对。

### 2. pilot

检查 L0 gate_proj、down_proj 的三种方法、四个 rank，再完整评估一次 4096 BF16 作为参考控制。
pilot 不是完整 14 配置或全部模块通过。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_a_all_ranks_dual_ppl_v1.log"
nohup bash "$BASE/full_a_all_ranks_dual_ppl_v1_tools/run_full_a_all_ranks_dual_ppl_v1.sh" \
  pilot --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

看到 `PILOT COMPLETE` 后再正式启动。`Ctrl+C` 退出 tail 不会停止 nohup 实验。

### 3. 正式跑 / 续跑（同一条命令）

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_a_all_ranks_dual_ppl_v1.log"
nohup bash "$BASE/full_a_all_ranks_dual_ppl_v1_tools/run_full_a_all_ranks_dual_ppl_v1.sh" \
  run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

预计完整前向配置数量：token 14（包含三组 r8 回放）；word 15（14 主配置 + 1 个额外旧 DG 控制）。pilot 完成的 BF16 word 会复用。时间预算从进程启动开始包含审计耗时。

不能将过去约 2 分钟/配置当成这套新入口的承诺：模型重载/部署哈希、全模块数值检查、服务器 I/O 都增加耗时。用 pilot 日志估算实际资源与耗时。检查是逐模块串行，GPU 数不代表检查耗时自动减半。

## 断点边界

- 检查：每个模块 × 方法提交，保存四个 rank 的检查和 BF16 rank64 因子；中断最多重做一个未提交检查。
- token：每 8 个窗口提交，已完成批次保留；中断最多重跑当前 8 窗口。
- word：每个完整配置提交；若被关机/信号打断，保留之前配置，当前配置从头重跑。没有实现 partial-document 续跑，不用部分 test 结果冒充完整 PPL。
- 时间预算到达或 SIGINT/SIGTERM 后退出码 75 为正常暂停；word 在下一次模型 forward 边界检查预算，不改变模型输出。检查期间在当前模块提交后暂停。
- 同输出目录有互斥锁；不要并行启动两份。更换任何协议、来源或脚本版本必须使用新输出目录。
- 不自动删除旧 checkpoint、原始结果或源因子。

估算新 BF16 rank64 因子约 0.94 GiB（以实际 shape 为准）；加 JSON、token loss 和 harness samples，建议新目录至少 8 GiB 空闲。不复制原模型或完整 A/G。

## 查看结果

```bash
RUN=/share/home/tm902089733300000/a913520780/chengkang/qera_diagnostics/full_a_all_ranks_dual_ppl_v1
cat "$RUN/token_ppl/ppl_summary.csv"
cat "$RUN/token_ppl/comparisons.csv"
cat "$RUN/word_ppl/ppl_summary.csv"
cat "$RUN/word_ppl/comparisons.csv"
cat "$RUN/rank_check_status.json"
```

- `rank_checks.csv`：2688 条模块/方法/rank 检查，包含 BF16 rounding flags。
- `token_ppl/per_unit.csv`、`paired_units.csv`：逐窗口和成对窗口结果。
- `word_ppl/per_unit.csv`、`paired_units.csv`：逐文档和成对文档结果。
- `word_ppl/<配置>/results.json`：完整 harness 原始结果与 samples。
- `*_control.json`：工程回放检查。两类 PPL 不合成一张混合排名。

## 本地验证与局限

48 项本地测试通过（含旧工具回归）。CPU 合成测试覆盖矩形方向、四个 rank、尾能量校验、篡改拒绝、直接 BF16 前缀、检查续跑、token 批次续跑、word 完整配置提交/中断不提交、原任务词数聚合与控制失败阻断。Bash 语法检查通过。
旧辅助脚本仍原哈希；脚本不运行服务器实验。本地没有 4090/CUDA 2.3 环境与服务器固定 harness，因此实际显存、速度、旧 word 协议文件位置、官方样本结构和数值回放须由 doctor/pilot 验证。遇到不匹配停止，不静默更改配置。
