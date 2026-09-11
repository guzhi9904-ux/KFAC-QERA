# Rank-64 分组删除机制实验 v1

Llama-3.1-8B Base / 冻结 MXINT3 / 224 个线性投影。新增普通残差 SVD；复用已有 FA+GI、FA+DG、FA+GF。发布时仅完成本地 CPU 验证，服务器 doctor、pilot、完整评估仍需执行。

## 实验设置 review

| 项目 | 本轮决定 | 原因 |
| --- | --- | --- |
| 普通 SVD | 对同一份 `E_t=(W.float()-Wq.float()).T` 做 FP64 full SVD，保留前 64 分量 | 补齐无 A/G 权重的量化误差补偿基线；分解 W 会换研究问题 |
| 主比较 | plain_svd、FA+GI、FA+DG、FA+GF | 后三者固定 A，仅改变 G；普通 SVD 是附加参照 |
| DA | 本轮不加入 | 避免把 A 与 G 的变化混在主机制比较中；已有 DA 终点仍有效 |
| 因子来源 | 已完成 A5 FP64 因子及其已评估 rank64 BF16 文件 | 复用既有奇异分量顺序，不重新 SVD 或重新因子分解补偿乘积 |
| 分组 | 1–8、9–16、…、57–64，共 8 组 | 每个模块按其自身分解的前 64 分量顺序分组 |
| 干预 | 从完整 rank64 独立删除一组，224 模块同时删相同组号 | 每个条件实际剩 56 个分量；不是累积删除，也不是单层定位 |
| 执行方式 | 右因子对应 8 行置零，两次 GEMM 均保留 rank64 shape | 减少内核形状变化带来的数值混杂；补偿不合并进 Wq |
| 主损失 | 相对各自完整 rank64 的逐 token 平均 NLL 增量 | 直接衡量语言建模损伤，保留正负值 |
| 辅助损失 | 同一 BF16 teacher 的全词表 KL 增量 | 衡量输出分布偏移，与真实标签 NLL 分开解释 |
| 几何量 | 删除分量的权重能量、冻结 FA 输出误差 proxy、总残差误差增量 | 检验重建尺度与任务损伤是否一致；显式保留交叉项 |

设计参考 [FWSVD 原文第 3 节与 5.5.1 节](https://arxiv.org/html/2207.00112)：其按奇异值分组并在模型各层同时删除同一组，比较任务损伤与重建误差。**本实验只覆盖已保留的 top-64 补偿子空间，不是原文的全谱分组复现。**

## 定义与可支持的结论

对方法 m、模块 l，保存的补偿为 `C_ml = L_ml @ R_ml`。令 `I_g` 为第 g 组的 8 个索引：

```text
D_mlg = L_ml[:, I_g] @ R_ml[I_g, :]
C_ml,minus-g = C_ml - D_mlg
```

上式定义数学干预；实际前向仍是 BF16 `(x @ L) @ R_masked`，不声称等于先合并矩阵再前向的数值结果。保留原因子中 Σ 的位置，不作平方根均分或额外归一化。

在同一评估 token 集上：

```text
damage_NLL(m,g) = [sum NLL(model_m,minus-g) - sum NLL(model_m,64)] / N
damage_KL(m,g)  = mean KL(teacher || model_m,minus-g)
                 - mean KL(teacher || model_m,64)
N = 138 * 2047 = 282486
PPL_removed / PPL_full = exp(damage_NLL)
```

`summary.csv` 同时保存每个配置的绝对 mean NLL / PPL / KL；`damage.csv` 使用**该方法自己的** full64 基线。不同方法的 full64 已经不同，不能把某组绝对 damage 更大直接解释成方法更好。

可检验的问题：在已保留的补偿子空间内，靠前分组是否承担更多任务相关作用；从 GI 到 DG/GF 后，分组顺序与实际 NLL/KL 损伤是否更一致；权重/激活重建能量能否解释损伤。

解释边界：

- GF 曲线更有序可支持当前子空间内的任务重要性排序证据；不预设一定单调，也不预设 GF 每组 damage 都更大。
- 较小重建能量却较大 NLL/KL 损伤提示重建尺度与功能作用不一致。需要同时看绝对基线，不能只画归一化柱图。
- 后组接近零，只说明该方法当前 full64 附近对该干预不敏感；不能推断第 65 位以后均不重要。
- 负 NLL damage 或负 KL damage 合法，意味着删组后该指标改善。保留原值，不能截断到零。
- 224 模块同步变化包含跨层相互作用。各组 damage 不可相加，也不能据此归因某一层、证明经验 G 等于真实 Fisher，或证明局部二次近似严格成立。

建议先画 8 组的原始 damage 曲线及探索性区间，再画 `damage_NLL` 对共同权重/FA 能量的散点；四方法保持相同坐标轴并附 full64 基线。每方法仅 8 点，不把相关系数当充分机制证明。

## 两套明确区分的评估协议

**主 NLL / token-PPL**：冻结的 138×2048 全有效窗口，batch=8、CE chunk=256、BF16、balanced 双 GPU、eager、无 cache，复用原 NLL 累加顺序。每次提交 8 个窗口，最后 2 个窗口也提交。保存逐 token NLL 与原协议窗口 NLL；headline 使用原累加顺序，不改用 CPU 求和替代。

先完整回放 BF16、W3-only、FA+GI/DG/GF full64 五个控制。各自必须匹配旧 full-A dual evaluator 的模型参数、buffer、device map、补偿 bits；PPL 绝对差 ≤1e-5、最大窗口 NLL 差 ≤1e-3。五组全部通过才删除任何分组。阈值是工程回放容差，不是统计显著性阈值。普通 SVD 没有旧终点，先得到完整 full64 基线再做其删组。

**辅助 teacher-KL**：相同 138×2048 窗口，独立标注 batch=1；一个完整 BF16 teacher 在 GPU0，一个完整 BF16 student 在 GPU1。相同 eager 前向，最后一个无预测标签位置不计分。全词表 FP64 log-softmax / centered-logit logsumexp，每块 128 个位置跨卡传输。每窗口原子提交；每配置重新 teacher 前向，避免保存巨大的全词表 logits 缓存。KL 阶段需要先通过上述五组完整 NLL 控制。

两模型都必须完全驻留各自 GPU；先跑全部窗口的两卡 BF16 self-KL，逐 token 最大值 ≤1e-9。仅 FP64 舍入范围内的负 KL（≥-1e-10）归零并记录归零前最小值；**KL 差不归零**。两协议实际参数/buffer 都绑定同一冻结 BF16/Wq，补偿 bits 跨协议精确一致。KL 的 batch1 不冒充冻结 batch8 NLL 的重复测量。

每协议 38 个配置：BF16、W3-only、4 方法×（full64+8 次独立删组），完成后各有 32 行 damage。共 76 个完整评估条件；KL 每个条件含 teacher/student 两次前向。没有新增 word-PPL、任务 benchmark、量化或 A/G 采集。

窗口差采用配对统计；探索性 95% 区间使用 circular moving-block bootstrap，block=8 个连续窗口、2000 次、seed=20260911。不是把 282486 token 当独立样本，也不是经过多重比较校正的显著性结论。

## 几何指标

用实际 BF16 舍入后的因子提升 FP64 计算；所有方法用相同 `S_A`。设 `R=E_t-C`：

```text
removed_weight_fro2 = ||D||_F^2
removed_output_sse_calibration_A = ||S_A D||_F^2
delta_total_weight_sse = ||R+D||_F^2 - ||R||_F^2
                       = ||D||_F^2 + 2 <R,D>
delta_total_output_sse_calibration_A = ||S_A(R+D)||_F^2 - ||S_A R||_F^2
```

通过 rank64 Gram 矩阵计算，CPU 测试与显式大矩阵公式逐项核对。`S_A` 是既定 FP32 root 数值升 FP64，因此是冻结校准度量的 proxy，不是独立 heldout 激活误差，也不是 BF16 执行损伤。各模块几何量之和不是模型全局输出误差。程序保存逐模块结果及全部 224 模块完整时的组级总和。

## 来源、安全与续跑

- 只读复用 `full_a_all_precision_r8_v1` 的 FP64 源因子和 `full_a_all_ranks_dual_ppl_v1` 已评估的 full64 因子/证书/逐 token 结果。
- 检查源哈希、身份、rank、dtype、shape、root bits、证书来源和原评估部署 bits；FA 因子复核 FP64 直接转 BF16，避免重复大矩阵证书计算。仅 plain_svd 新做 SVD，并检验残差等于第 65 位起的奇异值平方和。
- 冻结 helper 一字不改；不重新收集 A/G、不重建 A-root、不换 Wq、不改共享环境。新输出与全部来源目录不能重叠。
- 新代码/来源/协议通过 `experiment.json` 绑定身份；输出目录非空但不属于该实验时拒绝。不得边跑边替换入口代码；改版本要用新输出目录。
- 数值失败立即停止，不能降低阈值或跳过失败配置来凑 COMPLETE。损坏文件、身份变化、部署不符均拒绝续跑。
- 因子/几何量按模块保存；NLL 按 8 窗口保存；KL 按 1 窗口保存。未提交文件不代表完成，重启只从最后已提交单元恢复。
- `--max-hours` 从启动起计入来源审计；`--max-new-units` 计因子提交、NLL batch 或 KL window。几何计算也检查时间预算。到预算或收到 SIGINT/SIGTERM，完成当前单元后退出 75；相同命令继续。不要并发写同一输出目录。

## 服务器使用

上传离线包到 BASE，解压到新的工具目录。原环境必须是 `qera-original-a` / torch 2.3.0+cu121；选择空闲双 4090，各至少 18 GiB 可用；输出至少 8 GiB 可用空间。KL 每卡放完整 8B 模型，真实显存以 pilot 验证为准，不能静默 offload 或换模型精度。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
mkdir -p "$BASE/grouped_rank_ablation_v1_tools"
unzip -n "$BASE/grouped_rank_ablation_v1.zip" -d "$BASE/grouped_rank_ablation_v1_tools"
bash "$BASE/grouped_rank_ablation_v1_tools/run_grouped_rank_ablation_v1.sh" doctor
```

启动器每次核验包内 SHA256；默认不传命令也是 doctor。doctor 是只读来源与运行环境审计，不执行模型评估。继承接口中的 `--harness-source` / `--word-reference-dir` 仅用于受保护路径；本入口强制 token 分支，不导入或运行 word harness。

启动器内有固定已知服务器路径。若已完成的 full-A dual 结果存于别处，所有命令统一追加 `--source-rank64-dir /实际目录`；需要该目录的完整原始 JSON、safetensors，不接受仅有汇总 CSV 的替代。

先执行 pilot：检查 L0 gate/down 两个大模块的四种因子/几何量，回放 teacher 首 batch8，验证两模型 self-KL，并执行这两个模块 GF 删第 8 组的 BF16 hook smoke。此时其余目标使用 Wq-only；smoke 不是正式机制终点，不写入主表，也不保证完整 224 个补偿因子部署的显存。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/grouped_rank_ablation_v1_pilot.log"
nohup bash "$BASE/grouped_rank_ablation_v1_tools/run_grouped_rank_ablation_v1.sh" \
  pilot --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

看到 `PILOT COMPLETE` 后正式运行。下列同一命令也用于退出 75 后续跑：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/grouped_rank_ablation_v1.log"
nohup bash "$BASE/grouped_rank_ablation_v1_tools/run_grouped_rank_ablation_v1.sh" \
  run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

`run` 按顺序 prepare → token → KL。也可分别运行 `prepare`、`token`、`kl`；`token`/`kl` 要求全部因子已准备，`kl` 还要求五个完整 NLL 控制已通过。`summarize` 只重建已有结果的表格，但仍需通过相同来源/环境审计。默认输出：

```text
$BASE/qera_diagnostics/grouped_rank_ablation_v1/
  experiment.json
  plain_factors/                      # 新普通 SVD，FP64 与 BF16
  checked_factors/                    # full_gi / full_gd / full_gf
  geometry/                           # 每模块每方法 8 行
  group_geometry.csv
  group_geometry_totals.csv
  token_ppl/{summary,damage,per_window,paired_windows}.csv
  teacher_kl/{summary,damage,per_window,paired_windows}.csv
  token_ppl/status.json
  teacher_kl/status.json
```

两个 status 都要 `complete=true`、`completed=38`、`damage_rows=32`。回传 experiment、pilot、两个 status、5 个 `*_control.json`、两套 CSV 和几何 CSV；保留所有窗口状态、部署 JSON、token 文件供复核。`PILOT COMPLETE` 或 `prepare` 完成不等于机制实验完成。

## 本地测试与离线构建

测试不下载模型、不安装依赖；覆盖普通 SVD 尾能量、已有因子复用、独立 mask、实际 hook、残差交叉项、全词表 KL、原子续跑、防篡改和控制失败阻断。

```bash
cd tools/precision_audit
python -B -m unittest -v test_grouped_rank_ablation_v1
python -B run_cpu_tests.py
python -B build_grouped_rank_ablation_v1.py --output-dir /本地输出目录
```

离线包只包含新入口、启动器、说明、专用测试、构建器、5 个未修改的依赖 helper 和独立校验清单，不包含模型、统计、旧结果或其他正在运行实验的代码。完整回归 runner 不在最小包内，解压后可运行第一条专用测试。发布维护者在完成 review/test 后可用 `--update-checksums` 更新本实验的校验清单，再构建确定性 ZIP。
