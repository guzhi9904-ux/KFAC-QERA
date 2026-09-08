# MXINT3 × frozen A256/G256：实验设置审计与运行方案

本轮只改变权重量化位宽，优先验证 MXINT3；**不收集 Full-G，不重新收集 A/G**。
所有新文件写入原始 G256 结果目录的 `mxint3_v1/`，不覆盖 MXINT4 或 `full_svd_v1/`。

## 审计结论与实验矩阵

| 项目 | MXINT3 设置 | 与完成的 MXINT4 对照关系 |
|---|---|---|
| 模型 | 原清单固定的 Llama-3.1-8B checkpoint，224 个 projection | 相同权重、模块范围；不新增 embedding/lm_head 量化 |
| 量化 | 官方 pinned QERA MXINT，width=3，block_size=32，block_axis=-1 | 仅 width 4→3；不换 quantizer，不加激活量化 |
| A | 已存 Diag-A / Full-A roots，FP32 teacher，256×2048 positions | 原文件及 SHA256 原样复用，不重新求根或改变阻尼 |
| G | 已存 FP64 raw diagonal G，256×2047 prediction positions | 同一 FP32 teacher 梯度统计，不依赖新 Wq |
| G 定义 | 对有效位置聚合 `[d(sequence CE SUM)/d(layer output)]²` | 包含 future-token 路径；不是 per-loss Fisher；没有 Full-G |
| G 处理 | 原 mean-diagonal normalization、relative floor=1e-6 | 继承原配置，不增加调参 |
| 求解 | FP32，full_matrices=True，官方 A inverse，TF32 关闭 | 沿用成功的 full_svd_v1 数值路径 |
| Rank | 8 / 16 / 32 / 64 | 全部相同；求解 rank64 后取精确 factor prefixes |
| 评估 | 冻结 WT2 138×2048，282486 prediction tokens，BF16 | 同一 batch/chunk、低秩 hook、token-NLL/PPL 代码 |
| 基线 | BF16、W3_MXINT | 全部重新评估；BF16 保留历史 PPL 回归门禁 |
| 四组补偿 | diag_gi / full_gi / diag_gd / full_gd，各四个 rank | 固定 A/rank 比 G，固定 G/rank 比 A，共18配置 |

平均权重位数约3.25包含共享指数，不包含 BF16 低秩补偿及未量化参数开销。
这里是量化数值仿真：Wq 以 BF16 tensor 保存/计算，不是打包3-bit推理内核，也不声称获得推理加速。

### 不能简单修改旧脚本的原因

1. 旧 `pipeline.quantize` 硬编码 width=4；新阶段明确使用 width=3，并逐模块检查 FP32/BF16 量化完全一致。
2. 旧 GI 来自原 MXINT4 保存的补偿；这些因子不能复用。新清单清空旧 GI 指针，四组补偿都保存在新目录。
3. 新 GI 基线调用**原封不动的 pinned 官方 `compute_ab`，传入 MXINT3 配置**。GD 所用 full-SVD 求解器在 G=I 时逐模块对比这一基线，四个 rank 产品相对误差均须≤0.001。
4. 旧 W4 和旧 GI PPL 不能作为 W3 的数值回归目标。新清单仅保留 BF16 回归阈值0.01，其余由独立官方 GI 数值门禁和固定配对评估控制；不伪造未知 W3 参考分数。
5. 原版被 manifest 固定的代码不能改。本扩展仅新增子包/启动脚本，复用原 evaluator 的统计、batch、hook和CSV逻辑，只替换配置命名和新产物路由。

## 开始前的服务器审计

必须先完成 `full_svd_v1` 的18个 MXINT4评估。准备阶段只读验证：

- 原始G清单、代码哈希、PASS G256 checkpoint及快照元数据；
- 相同父清单的 full-SVD 数值审计与18个连续完整评估记录、原 control gates；
- 冻结 calibration/WT2 数据及 A raw/roots 哈希；
- 新目录无不明文件，续跑初始化指纹与已有内容完全一致。

模型文件在 quantize/solve/evaluate 消费时重新核验。G 文件仅以经校验的元数据引用；不会复制/重新收集。
父实验、新实验及MXINT4比较目录同时持有进程锁，防止并发写入。新目录中的权重和补偿不接受指向父目录的记录。
清单固定代码，开始运行后不要再修改这些源文件；变更协议应另建版本。

本地设置审计不是对远程文件的现场核验；只有服务器 `prepare` 输出 PASS 后，才说明该服务器输入通过审计。

## 运行

继续使用原 `qera-original-a` 环境（torch 2.3.0、transformers 4.44.2）；**不要安装仓库根 requirements**。
需要两张可见≥20 GiB GPU。没有 backward；磁盘建议至少25 GiB用于新 Wq/四组补偿及日志，旧统计仍需保留。

在服务器仓库根目录先运行审计：

```bash
bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh prepare
```

成功后启动量化、求解、18组评估：

```bash
nohup bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh run \
  > /share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-diag-g-256/mxint3_v1/run.log 2>&1 &
```

必须先 `prepare`，使目标日志目录存在。默认读取原始 `configs/llama3.1-8b-dual4090.yaml`；若原始配置另存，用 `DIAG_G_CONFIG=/absolute/path/original.yaml`，不要传派生的 `mxint3_v1/config.json`。

分阶段及续跑（同一环境、同一清单/配置）：

```bash
bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh quantize
bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh solve
bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh evaluate
bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh evaluate --only FULL_GD_R64
bash experiments/qera_diag_g_isolation/scripts/run_mxint3.sh summary
```

`run` 会再次审计并跳过已完成产物/窗口；有中断可继续同一命令。完整配置才进入汇总CSV，不把半途窗口当最终PPL。
若新 Identity-G 门禁失败，保留诊断上下文并审计；不要放宽阈值或切换 SVD 模式。

新目录关键输出：

- `audit.json`、`manifest.json`、`config.json`：输入/协议与来源。
- `quantized/`：新 BF16-emulated MXINT3 Wq。
- `corrections/{diag_gi,full_gi,diag_gd,full_gd}/`：新补偿、GI门禁与来源。
- `statistics/effective_g/`：复用G的有效对角尺度及floor诊断。
- `evaluation/ppl_summary_wikitext2.csv`、`evaluation/wikitext2_per_window.csv`、`evaluation/status.json`。

SSE日志仍为rank64的加权重建误差，不是PPL；不同权重度量下的SSE绝对值不能直接横比。

## 分析计划

主比较：每个 A/rank 上的 `GI→GD` 配对NLL/PPL增量，在MXINT3与MXINT4之间如何变化。
次比较：每个 G/rank 上的 `Diag-A→Full-A` 增量。
同时报告PPL绝对/相对下降、每token NLL变化和逐窗口改善比例；保留无补偿Wq与BF16差距。
不把更大的原始PPL差值直接认定为更好的误差恢复比例，也不将不同rank混合推断bit效应。
这是同一teacher统计在更大量化扰动下的检验，结果可能改善也可能退化；不事先保证G增益增大。

## 测试

```bash
python -m pytest experiments/qera_diag_g_isolation/tests \
  experiments/qera_diag_g_isolation/audits/test_identity_g.py \
  experiments/qera_diag_g_isolation/full_svd_v1/test_resume.py \
  experiments/qera_diag_g_isolation/mxint3_v1/test_mxint3.py
```

覆盖新配置/旧配置隔离、父产物不变、续跑与初始化中断、损坏G/错误宽度/错误指针拒绝、FP32/BF16一致性门禁、GI失败不提交、四组新补偿路由、完整18组汇总与BF16-only回归。
有同级官方 `QERA` checkout 时，额外用哈希匹配的真实 MXINT quantizer/official compute_ab 做CPU小矩阵、Diag/Full-A、rank8/16/32/64数值检查；没有则明确跳过这两项。
CPU测试不代替服务器torch2.3/CUDA上的224模块正式门禁或完整模型评估。
