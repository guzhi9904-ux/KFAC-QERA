# KFAC-QERA 实验问题、处理与验证日志

最后更新：2026-09-11（Asia/Shanghai）。

## 维护约定

本文件记录“遇到了什么问题、如何诊断、采取了什么处理、是否真的验证解决”，不是只记录成功结果。
后续在本工作区处理本研究的异常、修改实验协议、交付新代码或收到服务器结果时，继续更新本文件。

- 每个问题保留现象、证据、处理、验证结果和剩余不确定性；不要把旧结论直接删掉。
- 严格区分：已观察、推断、已实现但未实机验证、已实机验证、待处理。
- 日志中的历史事实以用户贴出的服务器日志/结果和本地可核查代码为依据；未见完整操作记录的处理，不补写成已执行。
- 不将 `AUDIT_COMPLETE`、`COMPLETE` 或单元测试通过等同于科学假设成立。
- 更新本日志不修改冻结代码、manifest、统计、权重、因子或原始评估结果。
- 自 2026-09-11 起，以仓库 `docs/research/ISSUE_LOG.md` 和同目录 `PPL_SUMMARY.md` 为维护入口；此前工作区根目录 Markdown 为历史副本，不再双份更新。GitHub 源码归档不表示服务器已更新。

## 当前状态速览

| 项目 | 当前状态 |
|---|---|
| Llama MXINT4/MXINT3，GI/DG 对照 | 已收到完整 r8/16/32/64 评估结果 |
| Llama MXINT3 Full-G | 收集、求解、评估已完成；旧结果受到数值问题干扰，不能直接据此否定 Full-G |
| L0 `o_proj` 数值诊断 | 已发现保存的 Full-A 根近奇异，FP32/FP64 correction 差异很大 |
| L0 `o_proj` r8：OLD/FP64/ZERO | 实机完成，OLD 精确回放；FP64 和 ZERO 均大幅改善 |
| 全模块 FA+GI/DG/GF FP64 求解，r8 | 六组完整汇总已回传；用户日志三组 OLD 控制通过 |
| 同根 FP64 四 rank × token/word PPL | 两套各 14 组汇总已回传并复算；完整控制、rank 检查与逐单元文件待核验 |
| 从原始 FP64 A 统计重新构造 FP64 root | 尚未实现、未启动，也不会由第一步脚本自动执行 |
| Qwen Full-A 求解爆炸 | 已有诊断工具；当前对话没有确认修复闭环，不标为已解决 |
| 真实 DG vs 置换 DG | 已讨论并记录，尚未实施 |

## 固定术语与实验边界

- `diag_gi` / `full_gi`：Diagonal-A / Full-A，G 为单位阵。
- `diag_gd` / `full_gd`：Diagonal-A / Full-A，G 为对角梯度二阶统计。
- `diag_gf` / `full_gf`：Diagonal-A / Full-A，G 为模块输出通道的稠密二阶统计。
- 名称开头的 `diag/full` 指 A，不是 G；Full-G 也不等于全模型完整 Fisher，不含所有跨模块相关项。
- 模型磁盘配置 BF16 与运行时加载 FP32 是两个概念。历史收集日志明确加载 `dtype=float32`；精度干预实验最终仍按 BF16 部署。
- Llama 校准：SlimPajama 固定 256×2048，共 524288 输入 tokens；G 的有效 next-token prediction 数为 256×2047。
- Llama 当前 WikiText2 评估：138×2048，共 282486 prediction tokens。不要把评估窗口数误写成校准 256，也不要写成未经核实的 146/147。
- Qwen 当前审计显示 WikiText2 143 个评估窗口；采用模型自己的 tokenizer，不要求跨模型 token 数相同。跨模型直接比较绝对 PPL 需要注意 tokenizer 不同。
- 不同 A/G 定义下的加权 SSE 不是共同标尺，不能只凭不同方法各自 SSE 的大小判断方法优劣。

## 2026-09-08｜ISSUE-01：DG 收集完成后 Identity-G 回归失败

**状态：运行阻塞已解除；观察到 full/reduced SVD 路径差异，不夸大为底层根因已完全确定。**

现象：DG 完成 256 窗口，耗时 14991 秒（4 小时 9 分 51 秒），进入 solve 的首个模块 `model.layers.0.self_attn.k_proj` 即失败：

```text
Identity-G regression failed ... diag
r8=0.0026605430624281577
r16=0.003140141929978237
r32=0.0032122227632065637
r64=0.0033287365922664474
unchanged_tolerance=0.001
```

这些约 0.3% 的差异是回归检查所比较的低秩 correction 产品相对差异，不是 PPL 差异，也不是 G 收集误差。

诊断证据（18:55 的 identity-audit）：

- 新旧求解输入逐项完全一致：relative Frobenius=0、max absolute=0。
- official replay 对已保存因子产品：四个 rank 均为 0 漂移。
- old/new input + full SVD：均为 0 漂移。
- old/new input + reduced SVD：复现上述约 0.3% 漂移。
- FP64 reference 对旧结果也并非 0 漂移，不能把“FP64 不同”直接当成修复成功。

处理：使用独立版本 `full_svd_v1` 恢复与旧基线一致的 full-SVD 路径，保留原回归阈值，不通过放宽阈值绕过。交付包 `qera_full_svd_resume_22ec3cb.zip`，对应历史提交 `22ec3cb98205617c7c2c98478819f6ac87db1dcf`。

验证：19:13 后日志显示逐模块成功保存 `diag_gd` / `full_gd`，后续收到完整评估结果。无需重收已完成的 DG。

经验：数值执行路径也属于实验协议；Identity-G 回归失败时应先做输入与路径审计，不能把差异解释为 G 的收益。

## 2026-09-08～09｜ISSUE-02：下载、日志查看与续跑目录问题

**状态：相关操作阻塞已解决；保留通用复发检查。**

1. raw.githubusercontent.com 的 curl 长时间 0 字节：改用上传已经打包的 ZIP 获取指定版本文件，随后实验成功进入求解。不要把网络超时当成计算卡住，也不盲目改代码。
2. `tail: cannot open ... No such file`：首次创建日志时可能早于后台重定向完成。改用 `tail -n 60 -F "$LOG"` 等待文件出现；日志实际出现后再判断后台是否正常启动。这个现象本身不能证明进程仍在运行。
3. `bash: experiments/.../run_full_g.sh: No such file or directory`、Exit 127：用户当前位于 `chengkang` 父目录，没有进入项目目录；应先 cd 到正确项目或使用绝对脚本路径。这次启动失败，不是旧进程没退出。
4. `PAUSED ... Time budget/signal reached`：安全预算暂停。使用同一个实验目录和续跑命令恢复，不能新建空输出假装续跑。
5. 新精度实验退出码 75：用于正常预算暂停。Ctrl+C 退出 `tail -F` 不会杀死独立 nohup 任务；不要用 `killall` 处理不明进程。

新交付的精度实验启动器使用工具目录定位和绝对服务器路径，减少工作目录错误。

## 2026-09-08～10｜ISSUE-03：Full-G 收集很慢，服务器租期有限

**状态：通过分片和事务检查点完成了实际实验。**

资源与现象：双 RTX4090 24 GB，内存上限 224 GiB；Full-G 分 4 个 shard，每片 56 个目标模块，每片遍历 256 个窗口。
pilot 日志前几窗口约 83～87 秒，包含保存的窗口可能更慢；CPU 进程峰值约 104.85 GiB，GPU 峰值约 [15.442,18.723] GiB。不要把早期粗 ETA 当成保证。

处理：使用独立 `full_g_v1` 与事务检查点，时间预算到期/信号到来后提交进度。已完成分片和当前分片累积统计保留；服务器重开后按原路径继续。历史包 `qera_full_g_8b49f28.zip`。

实机证据：2026-09-09 10:32:56 明确 `COMMITTED shard=2 window=122/256`，随后 `PAUSED`；后来用户确认 Full-G 跑通并提供完整 GF 评估结果。

边界：Full-G 保存的是累计二阶统计，不是把每个 token 的梯度全部永久保存。旧收集器具体检查点频率应看其冻结配置/日志，不把新精度评估的“每 8 窗口”误套到 Full-G 收集器。

## 2026-09-09｜ISSUE-04：Qwen 跨模型实验与现有 Llama 任务隔离

**状态：独立实验目录与基础数据准备已落实；Qwen 求解仍有单独未闭环问题。**

问题：新租服务器共享同一存储，直接改旧仓库/环境/输出可能影响 Llama Full-G 运行或续跑。

处理：Qwen 使用独立目录 `KFAC-QERA-qwen25-base-v1`，独立输出 `qera_runs/qwen2.5-7b-base-mxint3-v1`；不更新 Llama 冻结仓库或共享 conda 依赖。

模型选择：最初缓存是 Qwen2.5-7B-Instruct；用户决定改用 base 版本，经 ModelScope 下载到 `modelzoo/Qwen2.5-7B`。`modelscope: command not found` 后已有成功下载日志；安装的具体步骤未在当前可见记录中确认，不补写命令为历史事实。

验证：Qwen audit 显示 196 targets、112 A groups；pilot 的运行时日志记录 FP32、balanced、eager、use_cache=False、bias_policy=unchanged。

经验：文件隔离并不消除共享磁盘 I/O 竞争；不同主机也应确认 GPU UUID/资源归属。跨模型对齐实验规则和原始文本来源，不是复用 Llama tokenizer 的 token ID。

## 2026-09-09｜ISSUE-05：Qwen 初始化拒绝 dataset_cache

**状态：已越过该初始化错误，后续进入数据准备。**

```text
RuntimeError: Unknown output contents; refusing adoption: ['dataset_cache']
```

原因定位：初始化的输出归属检查把准备数据产生的私有缓存目录当成未知内容。

处理：初始化逻辑对 `dataset_cache` 做受约束的识别/检查，不直接删除整个输出目录，也不允许任意未知文件混入。当前本地 `qwen25_base_isolation_v1/run.py` 可见相关逻辑；交付历史包 `qwen_init_fix.zip`。

验证：下一次报错已进入 `data.prepare_data` 的数据集网络访问，说明原来的目录检查阻塞已解除，不代表后续阶段也都通过。

## 2026-09-09｜ISSUE-06：Hugging Face 不可达，Qwen 校准数据无法获取

**状态：离线数据路径已实现，后续 Qwen 数据审计/收集启动成功。**

```text
Couldn't reach 'DKYoon/SlimPajama-6B' on the Hub
curl: Failed to connect to huggingface.co port 443 ... Connection timed out
```

对齐依据（Llama 原始校准元数据）：

```text
dataset_id: DKYoon/SlimPajama-6B
resolved_revision: b5f90f419b7489cdba26fdbc8c022fcb5562f968
raw_prefix_rows: 5120
raw_text_sha256: 84067fe5da1470cca552135d1ee3e2089231b249053336b842721877c1ba03ab
calibration: 256 x 2048
```

处理：提供离线原始文本前缀与 WikiText 本地缓存读取路径，核对 revision、原文行数和文本哈希后再用 Qwen tokenizer 分词。不是复用 Llama 已分词 calibration.safetensors，也不是换一份任意数据绕过网络错误。
相关文件：`build_qwen_offline_bundle.py`、`qwen_offline_raw_b5f90f4/`、`qwen_offline_fix_b5f90f4.zip`、Qwen 独立实验的 `data.py` / `README_OFFLINE.md`。

验证：19:09 Qwen audit PASS；后续收集 A 启动。143 是 Qwen WikiText2 评估窗口数，校准仍固定 256×2048。

## 2026-09-09｜ISSUE-07：Qwen 收集 A 时 lm_head 显存不足

**状态：后续 A 收集完成；具体服务器修复操作尚缺完整记录。**

故障点：`lm_head(hidden_states)`，尝试分配 4.64 GiB，GPU 仅剩 1.31 GiB；日志同时报告 PyTorch allocated 17.12 GiB、reserved but unallocated 4.63 GiB。

解释边界：大 logits 分配和显存分配状态参与了这次 OOM；仅凭 reserved 数值不能断言全部由碎片造成，也不能保证只开 expandable_segments 就一定解决。

已知后续：19:35 日志显示 A 已运行到 windows=149–152；21:40 显示 shard_013 window=256/256 COMMITTED，14/14 片完成。因此收集阶段恢复并完成。

待补：实际服务器采用了什么配置/补丁、是否改变 batch 或 allocator 设置、补丁哈希。当前本地 A 收集仍调用完整 `model(...)`，不能写成“已通过跳过 lm_head 修好”。未取得操作记录前不把推测当成修复步骤。

## 2026-09-10｜ISSUE-08：Qwen Full-A GI 求解爆炸

**状态：未闭环，不能标为已解决。**

目标：`model.layers.1.mlp.down_proj`。对角 A 路径保存成功，Full-A 路径出现：

```text
Mean squared error ... 65477908.0
Weighted objective increased: 3.130071415508088 -> 25140.280772709644
```

根文件元数据虽然 `status=PASS`，且 residual=8.834077794745269e-12，但这不保证逆变换稳定：

```text
raw_max_imaginary: 0.00030793979535172883
normalized_max_imaginary: 4.2528577634844754e-07
```

已做：提供独立只读诊断工具 `qwen_full_a_audit_v1.py`，支持输入审计和 FP32/FP64 逆变换 replay。
未确认：具体 A-root 谱/条件数、是否与 Llama 异常完全同因、修复后全模型 PPL。不要把 Llama 单模块 FP64 成功自动当成 Qwen 已修复。

## 2026-09-10｜ISSUE-09：Llama 原始 FA+GF PPL 异常，常规 SSE 检查却通过

**状态：已定位一个会显著影响 endpoint 的数值异常模块；全模块审计尚未闭环。**

同为 MXINT3 / Full-A，旧结果：

| rank | GI PPL | DG PPL | GF PPL |
|---|---:|---:|---:|
| 8 | 9.518781685 | 9.473556011 | 12.587309223 |
| 16 | 9.297341811 | 9.225305026 | 10.013711412 |
| 32 | 9.034099330 | 8.958447708 | 9.772800422 |
| 64 | 8.802484942 | 8.688604156 | 8.758126413 |

问题：GI/DG 四个 rank 回归通过，不替代真实稠密 GF 在低 rank 的检查。rank64 的 SSE 通过，也不能代替 r8/r16/r32 检查。

处理：先用 `full_g_rank_audit_v1.py` 审计 224 模块、真实 GF 的各 rank；随后用 `full_g_target_probe_v1.py` 定点检查 L0 `o_proj` 的 A-root 奇异谱、保存因子、FP64 目标重测、FP64 SVD 与逆变换重求解。

已有诊断材料：用户提供的 `E:/改作业/a_spectrum.json`、`E:/改作业/comparison.csv`，以及对应服务器诊断日志。

主要证据：

- L0 `o_proj` 的保存 FP32 A-root 提升为 FP64 后，最小奇异值约 1.597e-10、最大约 0.230288，条件数约 1.442e9。
- r8 correction/error 范数比，旧→FP64 重求解：GI 约 3.265→0.097；DG 约 4.805→0.077；GF 约 24.412→0.164。
- 旧 GF r8 加权 SSE 约 0.00546275653，新 FP64 约 0.00546275583，二者很接近；局部加权目标可能对某些方向很不敏感，不能用“目标接近最优”排除严重 endpoint 问题。
- 该模块 GF 根诊断未触发特征值 floor；不能笼统归为“Full-G floor 导致失败”。

当时仍不能仅凭这些数值断言 PPL 劣化已被解释，因此开展下面的实际单模块替换实验。

## 2026-09-10｜ISSUE-10：单模块 FP64 / ZERO 干预验证了 endpoint 影响

**状态：已实机完成，控制逐窗口精确回放。**

工具：`full_g_precision_r8_v1.py`，交付包 `full_g_precision_r8_v1.zip`。
独立输出：

```text
/share/home/tm902089733300000/a913520780/chengkang/qera_diagnostics/full_g_precision_r8_v1/
```

设置：全模型 FA+GF / MXINT3 / r8；只改变 L0 `o_proj` correction，其他 223 个模块和 Wq 不变；新因子 FP64 求解后直接转换 BF16，仍用两次 GEMM 部署。

- OLD：原始全模型因子。
- FP64：只换目标模块的 FP64 重求解因子。
- ZERO：只关闭目标模块补偿，保留 Wq。

运行：pilot 保存 OLD 8/138 窗口后正常 Exit 75；去掉 pilot batch 限制后续跑。18:56:39 三组 COMPLETE。
实验 identity：`1ca323d55403f44e4f79e8b60435e06c3dcb493ab0341cd9c5e813b4bd5236f8`。

| arm | NLL sum | PPL | prediction tokens | windows |
|---|---:|---:|---:|---:|
| OLD | 715449.2136230469 | 12.587309222749287 | 282486 | 138 |
| FP64 | 632377.4028320312 | 9.380330726026495 | 282486 | 138 |
| ZERO | 632767.3254394531 | 9.393287575280118 | 282486 | 138 |

控制证据：ppl_observed=ppl_reference；absolute_ppl_difference=0；max_window_nll_absolute_difference=0。不是只在 tolerance=0.01 内接近，而是此次实际完全相同。

配对结果：

| comparison | delta NLL | delta PPL | improved windows |
|---|---:|---:|---:|
| FP64−OLD | −83071.81079101562 | −3.206978496722792 | 138/138 |
| ZERO−OLD | −82681.88818359375 | −3.194021647469169 | 136/138 |
| FP64−ZERO | −389.922607421875 | −0.012956849253622948 | 94/138 |

结论与边界：

1. 旧目标 correction 的确在该全模型上下文中严重损害 endpoint；仅去掉它就大幅恢复。
2. 按 NLL 的这条分解路径，ZERO 已获得 OLD→FP64 改善的约 99.53%；不能把全部 3.21 PPL 降幅宣传成新补偿的额外正向重建收益。
3. FP64 比 ZERO 小幅更好，但 token 有改善也有恶化。FP64−ZERO 改善 143774 个、恶化 138703 个、持平 9 个；不将相关 token 当独立重复宣称显著性。
4. 替换后 PPL 低于旧 GI/DG，但 GI/DG 仍用旧求解因子，不能据此完成公平的方法排序。
5. 干预定位到“整个新求解路径”，尚未分别归因于加权乘法、SVD、A-side solve 或 G-side solve。
6. 这是已知异常上的事后诊断，仅针对该模型、该模块、r8 和冻结 WikiText2；不等同全模型数值正确。

结果入口：`evaluation/ppl_summary.csv`、`evaluation/comparisons.csv`、`evaluation/control_check.json`；因子与局部诊断在 `factors/l0_o_fp64.{safetensors,json}`。

## 2026-09-10｜ISSUE-11：澄清“求解精度”与“A-root 信息精度”

**状态：实验定义已明确，避免后续混因。**

曾产生的误解：听到“重建 G 根”时，以为在修复 G；或者以为把 A-root `.double()` 就恢复了 A-root 原本丢失的精度。

明确区分：

1. **第一步（已做单模块，正扩展全模块）**：保存的 FP32 A-root 原值 → FP64 容器 → FP64 加权乘法/SVD/逆变换 → BF16 部署。没有恢复 FP32 root 已舍入掉的信息。
2. **第二步（未做）**：原始 FP64 A 统计 → 重新构造并保留 FP64 A-root → FP64 求解 → BF16 部署。在第一步基础上检验 root 构造/保存精度的额外影响。

G 根来自冻结 G 统计，按旧协议重建 FP32 根后提升到 FP64；不是重收 G、不是增加 shrinkage，也不是升级 root 构造协议。旧 G-root 没有归档的逐 bit 数值不能凭空核对，因此措辞应为“相同冻结统计及根构造协议”，不声称已验证新旧 G-root 逐 bit 相同。

## 2026-09-10｜ISSUE-12：扩展第一步到全模块，统一 GI/DG/GF 求解精度

**状态：用户已授权，代码已交付；未收到实机 pilot/最终结果。**

决策：先建立全模块统一求解精度的 GI/DG/GF 基线，再考虑第二步 root 重建。只先评估 r8，不立即铺开四个 rank。

代码与交付：

- [主程序](full_a_all_precision_r8_v1.py)
- [运行说明](README_full_a_all_precision_r8_v1.md)
- [启动器](run_full_a_all_precision_r8_v1.sh)
- [独立运行包](full_a_all_precision_r8_v1.zip)
- ZIP SHA256：`4b9fef861ca6ebd311be94fbd06fc900e556f36fe83c50be51f27292a33afb86`

范围：224 模块 × full_gi/full_gd/full_gf，共 672 份新求解；FP64 full SVD/rank64 两侧 solve，取前 8 分量 BF16 部署。
旧/新 × 三种方法，共六组完整评估。正式 run 先回放三个 OLD，失败即阻止新配置评估。

保存 rank 澄清（2026-09-10，用户询问后核对代码）：不是只保存 rank8。每个新因子文件包含 `A_fp64`（输入维度×64）、`B_fp64`（64×输出维度），另附本轮实际部署的 `A_bf16_r8` / `B_bf16_r8`。后续 r16/r32/r64 可以从这份 FP64 rank64 因子切片并直接转 BF16，不需要重做 SVD 或收集 A/G；但必须增加对应 rank 的数值检查和独立评估记录。当前脚本只评估 r8，局部目标/尾能量检查也聚焦 r8，不代表高 rank 已验证。不要直接修改冻结脚本中的 RANK 后继续写入当前输出。

验证与保护：

- 36 项本地 CPU 测试通过，包含矩形 gate/down 方向、根定义、SVD 尾能量、事务续跑、异常不提交、文件篡改拒绝、六组模拟评估以及旧工具回归。
- Bash 语法与 LF 行尾检查通过，ZIP 每个成员的哈希与本地文件一致。
- 旧 helper 文件保持原哈希，未修改冻结仓库及旧结果。
- pilot：L0 gate_proj 和 down_proj 各求解三种方法，覆盖较大 G/A 侧矩阵；随后评估一批 OLD full_gf。
- 每完成一种方法的一个模块就提交；评估通常每 8 窗口提交。强制关机最多重做当前未提交事务，不是在单次 SVD 内部续算。
- 数值失败不自动降精度、跳过模块、加阻尼或改用伪逆；记录 failure 后停止。
- 新因子 payload 预计约 3.87 GiB，不复制整个模型或全量 A/G。求解串行使用 cuda:0，评估使用双 GPU；双卡不会自动让一次 SVD 加速一倍。

独立服务器路径：

```text
工具：.../chengkang/full_a_all_precision_r8_v1_tools/
输出：.../chengkang/qera_diagnostics/full_a_all_precision_r8_v1/
日志：.../chengkang/full_a_all_precision_r8_v1.log
```

正式启动/续跑入口（先完成 README 的上传、解压和 pilot）：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_a_all_precision_r8_v1.log"
nohup bash "$BASE/full_a_all_precision_r8_v1_tools/run_full_a_all_precision_r8_v1.sh" \
  run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

待回收：pilot 的单方法耗时和资源峰值；最终 `evaluation/ppl_summary.csv`、`evaluation/comparisons.csv`、`solve_metrics.csv` 和三份 `control_check.json`。
尚不能给实机总时间或成功结果；不将之前单模块的十几分钟外推到全模块。

## 开发侧小问题（不改变科学实验定义）

- 原子 safetensors 保存后 fsync：Windows 对只读 `rb` 文件描述符报 Bad file descriptor，改为 `r+b` 后测试通过；原子替换和目录同步保留。
- Windows 合成测试临时目录删除失败：safetensors mmap 张量仍被引用。通过释放测试引用，及新脚本对 A-root 使用独立复制，解除映射生命周期问题；未删除或覆盖真实实验源文件。

## 后续问题清单

- [ ] 全模块 FP64 大模块 pilot：实际可运行性、峰值和耗时。
- [x] 全模块 r8：GI/DG/GF 同精度 PPL/NLL 汇总已回传（2026-09-11）；局部 flags 仍待 solve_metrics.csv 独立核验。
- [ ] 是否扩展 r16/r32/r64；需新一轮明确决定，不由当前脚本自动执行。
- [ ] 第二步 FP64 A-root 构造精度实验；尚未实现。
- [ ] 必要时拆分加权乘法、SVD、A-side solve、G-side solve 的精度贡献。
- [ ] Qwen Full-A 求解爆炸的独立根因与修复后验证。
- [ ] 置换 DG 机制对照：每模块内打乱对角元素、保留分布，多固定种子；不需要重收 G。
- [ ] 数值基线稳定后再讨论 G shrinkage/稳健处理，不能将其与精度修复混为一次实验。

## 2026-09-10：PPL 多阶段结果归档

状态：已完成本地汇总；不代表待回传实验已经完成。

- 新增 [PPL 阶段总账](PPL_SUMMARY.md)，按 A1–A6 / Q1 / H1–H3 区分原始 W4、原始 W3、GF 扩展、单模块干预、待回传全模块 FP64、未实施 root 精度实验和历史表。
- Excel 分“主线总览 / 其他口径 / 数据明细”，保留 135 条来源记录，主线 47 条；共享基线和 OLD 回放不是独立重复实验。
- 100 条有 NLL 的记录复算 PPL 通过，最大差约 1.78e-15；35 条 word-PPL 缺少 NLL，不声称完成独立复算。
- 特别纠正归档歧义：本地确有旧 Qwen7B 的 147 窗口、Qwen1.5B 的 146 窗口结果；它们不能代替当前 Qwen base MXINT3 / 143 窗口实验。本文“未收到完整 Qwen PPL 表”指当前这条新主线，Full-A 求解异常仍未因此解决。
- word-PPL 源文件模型/数据集标注不完整，部分 context=4096；与当前 Llama 138 窗口 token-PPL 分开，不能混成排名。
- A4 的 9.380331 是只替换 L0 o_proj 的结果，不写成全模块 FP64；A5 新结果保持“待回传”。
- 未修改实验代码、冻结产物或服务器进程。

## 2026-09-11：全模块同根 FP64 r8 结果回收

状态：六组评估完成；用户状态显示三组 OLD 控制通过。汇总和成对差异已本地复算，逐窗口及所有模块 flags 未独立核验。

| 方法 | OLD PPL | FP64 PPL | ΔPPL |
| --- | ---: | ---: | ---: |
| FA+GI | 9.518781684549507 | 9.502215248226642 | -0.01656643632286503 |
| FA+DG | 9.473556010647131 | 9.446071640927759 | -0.027484369719372737 |
| FA+GF | 12.587309222749287 | 9.379707357395414 | -3.2076018653538725 |

- 数据：Llama-3.1-8B / MXINT3 / r8；每组 138 窗口、282486 prediction tokens。
- 处理：沿用既定根定义，全模块 FP64 加权/SVD/两侧求解，仍 BF16 部署；没有重收 A/G、重建高精度 A-root、增加阻尼或 shrinkage。
- 同精度比较：GF − DG = -0.06636428353234436 PPL，111/138 窗口改善；GF − GI = -0.12250789083122804，125/138；DG − GI = -0.05614360729888368，117/138。
- 六组 OLD/FP64 的汇总 NLL→PPL 相符；六组成对 ΔNLL、Δmean NLL、ΔPPL 相符；token 胜/平/负数均合计 282486。窗口胜负为用户回传计数，未从原始逐窗口数据独立复算。
- 单模块 FP64 GF = 9.380330726026495，全模块 GF = 9.379707357395414，净差仅 -0.0006233686310803677。此前大幅终点退化主要由 L0 o_proj 的坏 correction 驱动；不能从净差小推断其他模块逐个无数值问题。
- 结论修正：原始 FA+GF r8 的巨大退化不是有效的方法比较依据。同精度下 GF 优于 DG 优于 GI，但 GF 相对 DG 的方法增量约 0.70% PPL，不能把旧→新约 3.21 PPL 的修复量宣传为 Full-G 方法收益。
- 范围限制：单模型、单数据协议、r8；无 IID-token 显著性结论；r16/32/64 与 FP64 A-root 构造仍未验证。
- 已同步更新 PPL 总账和 Excel A5，保留原始值、A4 单模块结果及所有历史来源 ID。本地记录总数 141，主线 53；未修改实验代码或服务器产物。
- 后续：补回 solve_metrics.csv / control JSON；在已存 rank64 因子上独立验证高 rank，或讨论置换 DG 机制对照。暂不因本轮结果自动启动下一实验。

## 2026-09-11：四个 rank × 双 PPL 独立入口

状态：已实现本地 CPU 测试，待服务器 doctor/pilot；没有新实机 PPL，不能标成完成。

验证：48 项本地测试通过（含旧工具回归），Bash 语法通过；CUDA/harness 实机尚未运行。

交付包：`full_a_all_ranks_dual_ppl_v1.zip`，SHA256 `d096a99f141c28fe0db42a1204a3f87c06987fffa476b1883f1d6f2810785626`。10 个打包源文件逐一哈希核对，旧 helper 原哈希不变。

- 新工具：`full_a_all_ranks_dual_ppl_v1.py` / `run_full_a_all_ranks_dual_ppl_v1.sh`；详见 `README_full_a_all_ranks_dual_ppl_v1.md`。
- 范围：当前 Llama MXINT3 Full-A 三种 G，r8/16/32/64，BF16/W3 参照；每种评估协议 14 个主配置。未扩展 DA、MXINT4、Qwen。
- 复用 A5 全部 672 份 FP64 rank64 因子。没有新 SVD、inverse solve、A/G 收集、量化或 A-root 构造升级。
- 检查使用同 root bits、白化因子正交/投影残差/能量顺序、源 r8 尾能量延伸检查，以及每个 rank 的 BF16 舍入 proxy。保留 rounding flags，不自动换算法。
- token 保持 138×2048，先回放三组 r8；word 接固定 4096 harness 完整 WikiText2 文档，先回放 BF16 和旧 FULL_GD_R8。旧 word 协议 JSON/数据/模型/包版本与因子绑定不符即停止。
- 澄清：旧 word-PPL 的 4096 独立评估入口已经在代码中找到；之前汇总 CSV 元数据不足不等于不存在该实现。仍须服务器原 protocol/results/complete 文件完成来源核验。
- 续跑边界：每模块/方法检查、每 8 个 token 窗口、每个完整 word 配置。word 被中断时重跑当前配置，不声称支持文档内续跑。
- 新目录 `qera_diagnostics/full_a_all_ranks_dual_ppl_v1/`；旧代码、A5 输出和运行环境不修改。
- 待回收 doctor/pilot，特别是旧 word reference 路径、源根 bitwise 复算、4096 有效上下文、旧 PPL 回放、实际显存和耗时。

## 2026-09-11：双 PPL pilot 回传与源码集中归档

状态：pilot 已完成；用户确认正式 run 已启动，尚未收到完整四 rank 双 PPL 结果。

- 实机日志：`COMMITTED WORD BF16 word_ppl=7.552676109 documents=62`，随后 `PILOT COMPLETE: two large modules checked`。BF16 word 控制与此前参考相符；不等于三组 token r8 控制、旧 DG word 控制及全部配置均已通过。
- 同段日志含模型对象被当成 repo id 的报错和 `not a git repository`。程序继续并提交了 BF16 word 结果，说明它们未使该次 pilot 中断；仅据截取日志不能确定完整调用来源，本次不擅自修改冻结 harness 消除警告。
- GitHub 检查时 HEAD 为 `d488bb82c0783af52127334273319a6d5028d10a`（2026-09-09）。Qwen 隔离入口和之后的数值诊断工具尚未纳入版本控制。
- 本次整理：Qwen 原入口纳入 `experiments/qwen25_base_isolation_v1/`；独立诊断脚本集中至 `tools/precision_audit/`；问题记录与 PPL 总账集中至本目录。首页提供索引，不在每个版本目录重复复制 helper；ZIP、缓存、模型与大实验产物不提交。
- 已发布诊断脚本、启动 shell、测试和冻结 word helper 与工作区原文件逐字节一致，另存 `SHA256SUMS`。保留单份冻结 word helper 是现有协议哈希要求，不更换为动态导入；不改求解数学、阈值或输出身份。
- 整理后旧测试依赖工作区相对位置，直接 discover 有两项导入失败；新增 `run_cpu_tests.py` 只设置测试路径，不修改冻结实现。该入口下 55 项 CPU 测试通过，5 个 shell 脚本语法通过。
- Qwen 默认本地 Windows 环境的测试在 SciPy `sqrtm/schur` 原生调用处中止；仅在本地测试进程设置 `MKL_THREADING_LAYER=SEQUENTIAL`、`OMP_NUM_THREADS=1` 后，全套 30 项测试通过。该现象支持本地线程运行库相关问题的排查方向，但尚未单独定位，不将其当作服务器 Qwen Full-A 数值异常的根因或修复。未升级任何依赖，未改服务器启动脚本。
- 运行隔离：未连接或操作服务器。用户继续原工具目录和原指令；不要向运行中/待续跑 checkout 执行 `git pull` 或覆盖脚本。源码归档不会自动部署到共享文件系统。

## 2026-09-11：四 rank token/word 汇总回收

状态：用户回传两份各 14 行汇总；已补充 Excel 与 PPL 总账，不表示独立重做服务器全过程审计。

- 归档编号 A5T / A5W，复用 A5 FP64 rank64 因子前缀，BF16 部署。没有改任何实验代码、A/G、root 或服务器产物。
- token 协议：2048 context、138 窗口、282486 prediction tokens。word 协议：4096 context、62 documents、241335 scored words。二者分开存储和比较。
- 新增 P142–P169 / S9–S10 共 28 行，保留原先 141 条记录的 ID 和所有原始值。总计 169 条；134 条有 NLL 的记录复算通过，最大 PPL 差约 1.78e-15；35 条旧 word 行仍缺 NLL。
- 新 token 三组 r8 的 NLL/PPL 与 A5 精确一致；BF16/W3 token 基线一致。汇总重现不是逐窗口控制重审，也不是独立重复实验。
- 同 rank 的 GF 在两套汇总中均优于 GI/DG。DG 在 token 四 rank 都优于 GI，但在 word r8/r32 略差于 GI；不写成所有对角 G 设置普遍改善。
- word 平均 NLL 由 NLL sum / 241335 核对；文档数没有误作分母。Excel 新增文档数、scored words 字段，复算公式按指标选择正确分母，并检查链接及分母变化后可重算。
- 继续待回收完整 control JSON、rank checks、逐窗口/逐文档文件；不作 IID-token 显著性声明。Qwen 和原始 FP64 A-root 构造问题仍未由本轮解决。

## 2026-09-11：DA 同根 FP64 补齐与 Qwen 故障回顾

状态：新代码已实现并通过65项 CPU 测试（含55项旧回归），两份新增 shell 语法通过；待两台双4090服务器验证，尚无新实机结果。

- 用户确认 Llama 与 Qwen 分别用双卡4090。共享只读模型、统计和工具；独立输出、日志，不修改旧 checkout/conda/manifest/检查点。
- 新 `tools/precision_audit/diag_a_fp64_dual_v1.py` / 启动 shell：补齐 Llama MXINT3 DA+GI/GD/GF 的 FP64 加权/SVD/求解，rank64存储与BF16直接前缀部署，四rank双PPL。复用冻结评估器，不再复制一套helper。
- Diag-A复用保存的FP32向量，要求严格正值；遇零/负数中止，避免 dtype 改变旧epsilon处理。没有新阻尼、收缩或高精度root构造。G沿用已完成FA FP64路径的构造/归一化/floor/FP32输出定义。
- 新构造的G-root与已完成FA FP64因子证书的root bits逐一核对；来源实验与原环境不符或root bits漂移时中止。需要保留原FA FP64实验目录，不修改其任何文件。
- 每模块/方法原子提交FP64及BF16因子和四rank尾能量检查；token八窗口提交、word完整配置提交。真实CUDA显存、耗时、旧DA r8 token控制和旧word控制须服务器验证。
- Qwen A已在14/14 shard收完。最终已知失败是layer1 down_proj Full-A GI求解；MSE=65477908.0，加权SSE 3.130071415508088→25140.280772709644。小的构根残差是在转FP32前计算，不是逆稳定性证明。此前lm_head OOM是另一问题，不混为同一根因。
- 新Qwen shell只调用已冻结只读诊断：inspect存根结构，再固定FP32 SVD比较FP32/FP64 A侧逆求解。只写新日志，不自动修改correction、不重收A。等实际回传再判断修复路线。
- 本次同时纳入上一轮已完成的A5T/A5W总表文档更新；未改变任何旧工具payload字节。
- 详细启动与续跑见 `tools/precision_audit/README_diag_a_fp64_dual_v1.md`。本地CPU测试不代表服务器已修复或实验完成。

## 2026-09-11：Qwen inspect 回收与诊断版本门禁修复

状态：inspect完成，Qwen数值根因尚待replay；v2仅修复独立诊断入口，不修原求解器或实验数据。

- 用户回传：root/raw和所列源码哈希校验完成；root为FP32的18944×18944，有限、无零行、对角正，相对不对称度1.8559991818521334e-10。没有计算root奇异值/条件数，不能据此证明稳定可逆。
- 发现v1 replay比较口径错误：manifest来自importlib.metadata.version，脚本却直接对比torch.__version__，导致2.3.0与2.3.0+cu121被误判。用户逐项核对八项distribution均match=True，runtime=2.3.0+cu121，CUDA12.1。
- 新增qwen_full_a_audit_v2.py与独立shell；同来源核对全部八项依赖，另核对已确认的CUDA构建，真实不一致仍拒绝。复用哈希固定v1数学/读文件函数，不修改其字节，不改变torch版本字段、不伪造或重写manifest。
- 原始A和root精度、固定FP32 SVD与逆求解比较协议不变；不重新收A/G、不跑PPL、不覆盖correction。Llama正在运行的旧工具目录与进程不触碰。
- 部署保持离线上传新小包，不要求服务器clone/pull。此前precision_tools_25813e5包的SHA256SUMS带CRLF，导致Linux读取带回车文件名；脚本payload本身为LF。用户可在管道中过滤清单回车，不需重传旧包。新增.gitattributes清单LF约束，新包验证ZIP内原始行尾而非仅按splitlines解析。
- 用户已回传Llama `DA PILOT COMPLETE`，BF16 word=7.552676109、62文档；不表示全224模块和全部DA PPL已完成。
- 本地72项CPU测试通过，新增shell语法通过；覆盖版本口径、每项真实依赖变更拒绝、CUDA构建拒绝、旧helper不变及LF清单校验。真实Qwen GPU replay待用户运行。

## 2026-09-11：Qwen逆求解重放回收，补齐单模块全程FP64

状态：v2逆精度重放已完成；新单模块全程FP64工具已实现，82项CPU测试及shell语法通过，实际GPU结果待回传。

- v2环境与源文件核验通过。目标layer1 down_proj、FA+GI、rank64，固定FP32 SVD；SVD截断前后加权SSE为3.1300714155080875→2.5269765301450184。
- FP32逆后SSE=25140.28077270964，逆相对残差232.01365795075017，精确复现原求解报错；仅逆改FP64后SSE=2.5269765362471652，残差7.777469480784339e-08；转回FP32后SSE=2.526983302202024，残差8.390460451371457e-05。
- FP64逆后的correction范数17731974.0606312，原error范数约27.305743，范数比约649386；未加权MSE仍4630996.880998676。局部加权SSE恢复不等于BF16部署/PPL已经安全。没有重收A，也没有证明root信息精度足够。
- 与Llama的区别：Llama成功干预是同根下乘法、SVD、两侧求解全FP64后BF16部署；Qwen此轮只改逆精度。Llama有保存root条件数约1.44e9与单模块PPL干预证据，Qwen尚无root谱和PPL证据。两轮固定U诊断的U来源也不同，不能把残差大小直接当病态程度排序。
- 用户确认先对Qwen同一目标补齐全程FP64（按FP64理解原消息末尾FP6）。新增qwen_a_fp64_target_v1.py及独立shell；同根Full-A、G=I、原Wq/error，full SVD和两侧solve全部FP64，保存rank64及直接BF16转换，检查r8/16/32/64。
- 保留Llama精度门槛：逆残差和相对尾能量差均1e-9。失败保存隔离候选与检查，不静默放宽、不加入阻尼/伪逆。不覆盖旧corrections、不重建root、不重收统计、不跑PPL。
- 输出qera_diagnostics/qwen_a_fp64_target_v1，report.json为完成事务标志，包含候选状态与数值门槛；DIAGNOSTIC_COMPLETE不是科学PASS。退出2表示完整诊断但门槛/有限性失败；候选永不自动部署。中断只需重做这一个模块，已完成重跑核验后复用。
- 新离线小包依赖前两份已解压helper，只读使用；无需服务器访问GitHub，不覆盖Llama正在运行的工具或共享conda。BF16指标明确仅为舍入因子在FP64下的proxy，不是实际BF16激活执行。

## 新条目模板

```text
日期 / ISSUE-ID / 简短问题名
状态：已观察 / 诊断中 / 已实现待实机验证 / 已验证解决 / 待处理
实验与模块：
现象和原始报错：
证据与定位（事实/推断分开）：
采取的处理（哪些因素保持不变）：
代码版本、工具包与输出路径：
测试/实机验证结果：
结论边界及尚未解决的问题：
下一步：
```
