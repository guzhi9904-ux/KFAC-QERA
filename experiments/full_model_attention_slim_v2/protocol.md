# 全模型 Structure-aware Attention QER 实验协议

版本：v2.0 / 2026-09-21。执行平台：双 RTX 4090。模型：Llama-3.1-8B Base。

本文供另一个 Codex 完成代码、核验服务器资产、执行实验和汇报结果。它是本轮完整执行协议，取代 Downloads 中旧 full_model_structure_aware_attention_experiment.md 的执行设置，以及旧成本审阅中关于 WikiText-2 校准、旧因子直接复用的假设。历史结果及父实验文件保留。

用户已授权本轮实现和服务器运行。完成本文规定的只读审计、配置冻结和技术验收后，按顺序继续执行，不为正常阶段切换反复请求确认；若遇到身份不一致、数值验收失败、资源不足或需要改变科学设置，保存现场并报告具体阻塞，不擅自降低预算或换方法。本协议不授权删除父资产、停止他人任务、购买资源或扩展研究范围。

## 1. 本轮只回答什么

前期单模块实验支持一个候选规则：Q/O 用 Marginal，K 用 Token-joint-one，V 用 Attention-aware，MLP 用 A-only。

本轮检验：在固定量化权重、固定各模块 rank、统一通用校准集的条件下，这个按 projection type 冻结的规则能否改善完整量化模型的预测保持和任务质量。

主比较是：

1. 全 A-only → Attention Marginal：输出预测敏感性在整模型 QER 中是否提供增益。
2. Attention Marginal → Structured-Attention：冻结的 K/V 几何替换是否进一步提供增益。

不以全局曲率拟合误差作为主目标，不构造完整 Hessian，不新做 ALS、Full-fit、Sequence、Token-joint-3、Direct Query、rank allocation 或 MLP 新方法；不按层选 winner。已有单模块结果是开发证据，不是全模型结论。

## 2. 固定配置

| 项目 | 本轮固定值 |
|---|---|
| Teacher | 当前项目已核验的 Llama-3.1-8B Base checkpoint |
| Teacher/候选前向与反向 | FP32；原 checkpoint 是 BF16 不等于本轮 BF16 推理 |
| 目标权重 | 32 层 × q/k/v/o/gate/up/down = 224 个 Linear |
| 非目标权重 | embedding、lm_head、RMSNorm 等保持 teacher |
| 量化 | 当前冻结的 MXINT3；group/block size 32，axis=-1；以父量化器源码和资产身份为准 |
| 残差 | E = W_teacher_FP32 - Wq；所有候选共用同一 Wq |
| 校准集 | DKYoon/SlimPajama-6B，冻结的 256 个窗口 |
| 上下文 | L=2048；T=L-1=2047 个预测位置 |
| 标签 | 每窗口 1 套 teacher predictive labels，各预测位置各采样一个类别 |
| Rank | 每个有补偿的 Linear rank64，不按 head 分配 |
| 统计/根/求解 | FP32 原始激活与梯度，转 FP64 收缩/累加；FP64 roots 和 dense SVD |
| 阻尼 | eta_A=eta_G=0.001，trace-relative，继承已验收 solver |
| 部署 | W_deploy = Wq + float32(P64 @ Q64)，乘法先在 FP64 完成 |
| 层规则 | 全 32 层统一，无 L0/L31 特例 |

本轮评价量化数值与补偿的质量，不是 MXINT3 打包内核性能测试。不能由 FP32 dense 部署的显存和时间宣称实际压缩推理收益。理论 rank64 指存储因子定义的 C64；FP32 舍入后的稠密矩阵不宣称严格代数 rank64。

运行前读取实际 config 验证 hidden=4096、intermediate=14336、32 query heads、8 KV heads、head_dim=128；不匹配则停止并定位模型身份，不能默默套用尺寸。

## 3. 校准数据：与 QERA/SRR 对齐

### 3.1 来源及确定预算

本地已核查：

- QERA/experiments/configs/w-only-uniform-rank.yaml：slim_pajama_6b、256、2048。
- srr/experiments/configs/srr_3bit_rank64_iter1.yaml：相同校准设置。
- 两份 ptq_pipeline.py 的 calibration loader 均使用 perplexity_max_seq_length 作为校准长度。
- 我们的 qera_original_a_isolation 历史实验也采用 256×2048。

因此本轮是 **256×2048=524288 个输入 tokens**，对应 **524032 个 next-token 预测位置**；不是 512×1024，也不是 256/512 篇独立文章。不能只因总 tokens 相同就改变上下文长度。

历史预处理读取固定前 5120 条原始文本，使用官方 QERA 的拼接、tokenize、分块流程，取前 256 个窗口。5120 是原文前缀数量，不是拟合窗口数。

优先复用已经冻结的 calibration.safetensors。重建时必须采用已记录的 tokenizer、特殊 token 设置、num_workers=8、map 批处理、原文顺序、分块/尾部处理和官方源码版本；不能用“全文一次拼接”自行替代官方预处理。预处理 worker/map 边界可能影响实际窗口。

### 3.2 已有缓存线索，不等于已完成本轮服务器验收

当前租用服务器历史根目录：

~~~text
/share/home/tm902089733300000/a913520780/chengkang
~~~

9 月 21 日只读审计记录的优先数据资产：

~~~text
qera_runs/llama3.1-8b-original-a-isolation/data/calibration.safetensors
qera_runs/llama3.1-8b-original-a-isolation/data/calibration.json
~~~

张量文件约 8 MiB。实际运行先查存在性、metadata、input_ids/attention_mask 的 [256,2048] 形状、tokenizer 身份、内容哈希和顺序，再绑定到新 run。

另一台本地 A6000 Linux 服务器历史上也有：

~~~text
/data1/cck/datasets/prepared/Llama-3.1-8B/qera-bd7fc86-ctx2048-workers8/
~~~

这不是双 4090 租用服务器的同一路径，不能混写。其历史 SlimPajama revision 为 b5f90f419b7489cdba26fdbc8c022fcb5562f968；双 4090 资产必须读取自己的 metadata 核实，不能仅按名称假定相同。

### 3.3 Teacher labels 本轮重新冻结一次

近期 K/V 实验拟合数据是 WikiText-2，旧 sampled labels 不适用于这批 SlimPajama 窗口。

在原 teacher、teacher-forced 固定文本上下文上，对每个窗口 j 和每个预测位置 t=0,...,2046：

\[
y_{jt}\sim p_{\rm teacher}(\cdot\mid {\rm text}_{j,\le t}).
\]

保存 [256,2047] 标签、逐窗口 hash、抽样算法、RNG 状态/种子、位置 mask、teacher 身份。默认基种子 20260921；若实现采用按窗口派生种子，冻结明确的派生函数，恢复运行不得换标签。使用完整 softmax 分布，不用 top-k 截断或 temperature 修改。

每个窗口只采一套，所有模块/方法/分组共用。标签采样结果不回灌文本，不做 free-running generation。QERA/SRR 对齐的是校准文本和预算；predictive labels 是我们方法需要的额外统计来源，不声称它们是 QERA 官方标签协议。

定义 sampled sum-NLL：

\[
\ell_j=-\sum_{t=0}^{T-1}\log p_{\rm teacher}(y_{jt}\mid {\rm text}_{j,\le t}).
\]

g 是该标量对对应模块输出的真实梯度。不能用 ground-truth CE、mean-NLL 或 teacher 对自身 KL 的零梯度替换。原数据文件的 labels=input_ids 也不是这里的 predictive labels。

## 4. 冻结候选

为避免与 C4 语料混淆，表中模型 C4 显示为 Structured-Attn。

| ID | Q | K | V | O | gate/up/down | 补偿模块数 |
|---|---|---|---|---|---|---:|
| Teacher | 原权重 | 原权重 | 原权重 | 原权重 | 原权重 | 0 |
| C0 Quant-only | 无 | 无 | 无 | 无 | 无 | 0 |
| C1 MLP-A | 无 | 无 | 无 | 无 | A-only | 96 |
| C2 All-A | A-only | A-only | A-only | A-only | A-only | 224 |
| C3 Attn-Marginal | Marginal | Marginal | Marginal | Marginal | A-only | 224 |
| C4 Structured-Attn | Marginal | Token-joint-one | Attention-aware | Marginal | A-only | 224 |

“无”指使用冻结 Wq、不加补偿。C2/C3/C4 的补偿参数预算完全一致；C1 只是固定 MLP 背景，不是同预算方法比较。

C3→C4 同时改变 K 与 V，评价的是这套组合规则的效果，不能单独归因于 K 或 V。本轮不自动增加 K-only/V-only 消融。L0 K 的历史退化保留在背景说明，不因此修改该层。

C2 称 QERA-style A-only，而非未经改动的官方 QERA 复现：本轮阻尼、FP64 solver、FP32 部署均按当前实验规范统一。对齐数据不等于完整复现 QERA/SRR 算法；本轮不另加 SRR 候选。

## 5. 统计与数学定义

以下 j 遍历 N=256 个窗口，u 遍历 L=2048 个模块位置。普通 A 含全部 L 个输入位置；梯度最后一个无 loss 位置理论为零，G 和 token contraction 使用 NT 归一化，保持父实现。

所有统计来自原 teacher 轨迹，不能逐层量化/补偿后重新收集上游改变的激活。

### 5.1 A-only 与 Marginal

\[
A_M=\frac{1}{NL}\sum_{j,u}x_{ju}x_{ju}^{\top},\qquad
G_M=\frac{1}{NT}\sum_{j,u}g_{ju}g_{ju}^{\top}.
\]

A-only 用 (A_M,I)；Marginal 用 (A_M,G_M)。每层 q/k/v 共用一份普通 A，gate/up 共用一份普通 A；o 和 down 各自独立。

### 5.2 K Token-joint-one：必须沿用真实初始化

\[
A^{(0)}=A_M,\qquad G^{(0)}=I_m,\qquad m=1024.
\]

一次同步更新：

\[
A_1=\frac{\sum_{j,u}\|g_{ju}\|^2x_{ju}x_{ju}^{\top}}{NTm},
\quad
G_1=
\frac{\sum_{j,u}(x_{ju}^{\top}A_Mx_{ju})g_{ju}g_{ju}^{\top}}
{NT\|A_M\|_F^2}.
\]

必须用完整 N256、未阻尼的 A_M。不能用 running A、A_1、I 或 diag(A_M) 替换；不能用 ||x||² 替换 xᵀA_Mx；不继续第二轮。第一轮 gauge 和 solver 内部 gauge 与父实现保持一致。

### 5.3 V Attention-aware：沿用已验收 GQA 定义

设 P^a 是真实 teacher 的第 a 个 query head 注意力概率，包含真实 RoPE、causal mask 和 GQA。a 映射至 KV head b(a)。x 是 v_proj 的原输入，delta^a 是 attention mixing 输出、进入 o_proj 之前的相应 head 梯度：

\[
\bar x_{jt}^{a}=\sum_u P_{j,tu}^{a}x_{ju}.
\]

\[
A_V=\frac{1}{NLH_q}\sum_{j,t,a}\bar x_{jt}^{a}(\bar x_{jt}^{a})^\top,
\quad H_q=32.
\]

\[
G_{V,b}=\frac{1}{NT}\sum_{j,t,a:b(a)=b}
\delta_{jt}^{a}(\delta_{jt}^{a})^\top,
\quad
G_V=\operatorname{blockdiag}(G_{V,0},\ldots,G_{V,7}).
\]

A_V 为 4096×4096，G_V 为 1024×1024，保持完整 WV 上单次 rank64 求解，不按 head 分 rank。G 的块对角性是冻结近似的一部分，不声称捕获所有跨 head/位置曲率。

高效且等价的 A 累计：

\[
\sum_{t,a}\bar x_t^a(\bar x_t^a)^\top
=X^\top\left(\sum_a(P^a)^\top P^a\right)X.
\]

优先复用 v_math.py 的 effective/normalize。新采集器可以直接取得 o_proj 输入的梯度 delta，或使用已验收 local replay；必须先与父定义数值比较。不能将 g_v 本身当成 delta。

### 5.4 统一求解

对每份 raw A/G 沿用父 gauge、trace-relative damping，记最终正定矩阵为 A_d/G_d：

\[
C_{64}=G_d^{-1/2}[G_d^{1/2}EA_d^{1/2}]_{64}A_d^{-1/2}.
\]

FP64 对称根与 dense SVD。禁止静默 eigenvalue clipping、额外 epsilon、随机 SVD 或换 Cholesky 定义。

A-only 可使用数学等价的单侧实现，跳过 G=I 的大矩阵分解，但必须先与父双侧路径核验，保持同一阻尼和目标。共享 A 的 eigendecomposition/roots 尽量复用。

保存 P64/Q64，部署时先做 FP64 P64@Q64 再转 FP32，与 Wq 相加。不能改成 float32(P64)@float32(Q64)，或 BF16 两层 hook。

## 6. 复用资产审计

执行 Codex 首先读取当前仓库及目标目录适用的 AGENTS.md，然后只读检查服务器 GPU、占用任务、可用主机内存、文件系统配额、模型、代码和下列资产。

| 资产 | 可复用条件 | 不满足时 |
|---|---|---|
| SlimPajama tokens | tokenizer、[256,2048]、预处理与内容身份一致 | 按冻结历史协议重建并记录；不换成 WT2 |
| Teacher | config/tokenizer/权重身份与当前父实验一致 | 报告，不能换模型继续 |
| 全 224 模块 Wq | 量化器、配置、teacher 与逐模块 hash 一致 | 未覆盖模块按同一量化器生成一次；冻结共享全集 |
| 旧 SlimPajama A/因子 | 文本、轨迹、FP32变量/FP64累加、归一化、阻尼/部署全部兼容 | 重算；不能仅凭 Full-A 名称复用 |
| 最近 WT2 的 x/g/A/G/因子 | 与本轮拟合语料不同 | 不用于本轮构造，只保留历史证据 |
| 旧 ground-truth CE G | 与 predictive labels 定义不同 | 不复用 |
| 统计/solver/V/GQA 代码 | 冻结源码，完成新 collector 数值回归 | 修复实现，不改变方法 |
| 评测 tokens/tasks | 协议与本轮一致，冻结身份 | 准备所需小型固定数据 |
| 旧模型分数 | 仅当完整候选部署身份与评价协议完全相同 | 默认重评，不拼接旧 BF16/WT2补偿分数 |

尤其：旧“44 组局部因子可直接复用”的预算不再适用。没有证据前按全部 416 份独立因子需要生成安排。

审计输出：

- parent_assets_audit.json：逐项存在性、身份、维度、精度、可用/不可复用原因。
- reuse_plan.md：逐模块/统计族引用与缺口。
- frozen_config.json、data_manifest.json、teacher_identity.json、quantization_manifest.json。
- estimated_cost.md：基于实时资源及 pilot 更新。

不需要为新实验复制整份 teacher、旧 TB 缓存或完整 Hugging Face cache。已有父目录只读引用。

## 7. 流式执行，控制双卡与主机内存

### 7.1 Pass A：完整普通 A_M

原 teacher 前向，统计每层 QKV/O/gate-up/down 四类输入 Gram。无反向。匹配的旧 A 可逐项跳过，否则全新收集。同步生成并冻结本轮 predictive labels；无需再为采样单独新增完整 teacher 遍历。

Q/K/V 共享 A，gate/up 共享 A；不重复累计同一输入。每份 A 完成后保存 raw FP64 与计数，冻结 N256 身份。

### 7.2 Pass B：predictive gradients 与结构统计

完整 A_M 已存在后，在同一标签下共享 teacher 前向/反向，统计：

- q/k/v/o 的 Marginal G；
- K 的 A1/G1；
- V 的 A_V/G_V。

不计算 MLP G。每窗口及时将 x/g/P/delta 转成最终统计增量并释放，不持久化原始大缓存。

全层模型在两卡分片运行，不默认单卡可放下 FP32 8B teacher。先测试全层/分组 hook 的峰值内存；必要时按连续 4–8 层一组统计，组内共享反向。分组大小是工程参数，不能改科学估计器。每组均用完整 N256 与同一 labels。

全 32 层 attention probabilities 单窗口约 16 GiB FP32，普通 A 全集约 61 GiB FP64；不能全部堆在 GPU 或无约束地复制到 CPU。统计组完成即保存/释放。CPU 与两卡通信、checkpoint 重算计入时间。

需要多组时允许重复 teacher 遍历，明确报告额外代价；不按每个 head 或每个模块各做一次完整反向。严禁用少窗口、少 token、截断后续梯度或更换 attention 近似来解决 OOM。

### 7.3 求解与冻结

卸载 teacher 后进行离线 roots/SVD；双卡可各运行一个 worker，依据实测显存安排大 MLP 求解。共享 roots 的模块连续处理。

独立补偿份数：

| 因子族 | 数量 |
|---|---:|
| 全模块 A-only | 224 |
| Attention Marginal | 128 |
| K Token-joint-one | 32 |
| V Attention-aware | 32 |
| 合计 | 416 |

MLP A-only 在 C1/C2/C3/C4 共享；Q/O Marginal 在 C3/C4 共享。所有候选 manifest 在正式评价前冻结，不根据 PPL 修改。

## 8. Pilot 与数值验收

先用校准窗口 0/1 做工程 pilot，不用于选择方法。验收通过后，可按事务记录计入正式累计且只计一次；否则清理本 run 的未提交临时状态并从冻结边界重算。

1. 数据：[256,2048] 输入、[256,2047] sampled labels，合法词表、无 padding 歧义；sample/token/label 顺序绑定。
2. Teacher：FP32，eval mode，dropout 关闭；attention 实现、TF32 开关、checkpoint 行为固定并记录。新采集路径与父参考 hidden/output 数值一致。
3. 梯度：共享 backward 与小规模独立 reference 一致；最终无 loss 位置梯度为零或通过父数值容差。不 detach 中间目标输出而切断跨层梯度。
4. K：完整维度两窗口 contraction 与 token_step 同步第一轮公式比较；FP64 相对误差 ≤1e-10。pilot A_M 可以用两窗口验证公式，正式 G1 一律使用全 N256 A_M。
5. V：P、x、delta 构造的 g_v 与真实 v_proj 输出梯度比较，FP32→FP64 路径相对误差 ≤1e-5；显式 bar_x Gram 与高效 Gram 的 FP64 相对误差 ≤1e-10。核查 GQA mapping 和 G 块结构。
6. 统计：finite、对称；负特征值仅容许相对舍入级误差（父阈值 1e-10），不静默裁剪。
7. Solver：阻尼后 condition ≤1e8，root reconstruction ≤1e-10，weighted-SVD tail/back-transform ≤1e-8，沿用父验收定义。
8. 部署：逐模块目标权重 hash、未量化权重身份、候选切换和 teacher 恢复；补偿舍入带来的代理目标变化沿用父 ≤1e-4 检查，失败需报告。
9. 全模型：teacher self-KL 与评价实现正常，所有候选无 NaN/Inf；单次前向显存与分块 LM-head CE/KL 验收。
10. 断点：小规模中断/恢复后统计、标签、窗口计数和候选身份与不中断参考匹配，不重复/漏计窗口。

数值失败先修代码。不能根据真实 KL/PPL 调整阈值、阻尼、层规则。数学定义正确但全模型效果差不属于实现失败，必须报告。

## 9. 评价数据与指标

本轮固定三类评价。除了小规模实际 KL，不新增 alpha 扫描、曲率评分或单模块 sweep。

### 9.1 实际 teacher-KL：16 个既有 validation 窗口

复用近期模块实验相同的 WikiText-2 validation 16×2048 输入，重新评价 Teacher、C0–C4 的全模型联合干预。这些窗口已参与开发期观察，标记为 development/held-out-from-fitting，不称全新独立测试。

KL 使用完整 teacher softmax、方向 teacher→candidate，对所有有效预测位置平均，不采样评价标签，不以二次近似替代。

\[
{\rm Recovery}(C)=100\left(1-\frac{{\rm KL}(C)}{{\rm KL}(C0)}\right)\%.
\]

先汇总 KL 再取比值，不能平均每窗口 recovery。报告绝对 KL、每窗口 KL/NLL、有效 tokens 和 recovery；若分母为零则 recovery 记 NA。

不永久存全部 logits。默认每候选评价时重算小批 teacher reference 并及时释放，以增加少量前向换取低存储；允许在内存做有界 reference cache，不能重新引入全数据 logits 文件。

### 9.2 Token-PPL：所有六个状态

**WikiText-2 test**：优先读取历史 original-a-isolation/data/wikitext2.safetensors 及 metadata；核实 tokenizer、revision、官方预处理、窗口/尾部定义后冻结全量已有 test 窗口，L=2048，不自行取前 16/128 窗口。实际可能为 138 等数量，必须读取张量后填写，不硬编码历史另一路径的 141。

**C4 固定子集**：本协议选定英文 validation 的 128 个 2048-token 窗口作为第二语料预算，报告名称为“C4-en validation fixed128 token-PPL”，不是全 C4 PPL。

先查项目既有 C4 子集缓存：只有来源、split、tokenizer、长度、样本 hash、抽样/分块规则可核验才使用。若已存在合格 N≥128 的固定子集，固定其原始顺序前 128 个窗口；若没有，按固定 revision 的英文 validation、固定 shard/文档顺序流式读取，每篇文档 tokenization 后取连续不重叠完整 2048 块、丢弃短尾，按顺序累计前 128 块。不跨文档拼接，不随机重抽，不下载整个 C4。

数据准备必须在查看本轮模型分数前完成；dataset repo/config/revision、shard/文档索引、tokenizer special-token 策略、128 个窗口及 hash 写入 eval_manifest。缓存不可验证就重建，不随效果换子集。无法获得该数据时报告缺项，不拿 WT2 第二份冒充域外结果。

两语料所有状态使用相同输入与有效 mask，窗口内评分位置 0..L-2；记录尾部丢弃量。采用：

\[
{\rm NLL}=\frac{\sum_w{\rm NLL\_sum}_w}{\sum_wT_w},
\qquad {\rm PPL}=\exp({\rm NLL}).
\]

禁止平均窗口 PPL，禁止混用 word-PPL/rolling harness 分母。本轮“数据对齐 QERA/SRR”指校准，不自动把历史不同 PPL 数值拉来比较。

### 9.3 下游：固定五个任务，不自动追加旧七任务

为控制预算，本协议采用原附件五个任务，0-shot、完整标准评价 split：

| harness task | 主报告项 |
|---|---|
| hellaswag | acc_norm |
| piqa | acc_norm |
| winogrande | acc |
| arc_easy | acc_norm |
| arc_challenge | acc_norm |

评价 Teacher、C0、C2、C3、C4 共 5 个模型状态；C1 只做 KL/PPL。共 25 个模型-任务组合，不添加 MMLU/BBH/BoolQ/CSQA。

冻结 lm-eval-harness commit、任务 YAML、数据版本、标准 split、0-shot、无 chat template、tokenizer 和 FP32 dense 部署。若该固定 harness 缺少表列 metric，先核查任务定义并在正式评分前记录可用 metric 与原因，不能事后选较好指标。

建议固定 max_length=4096，batch_size=1 起步；为吞吐可依据无分数导向的内存 pilot 调整 batch_size，但所有状态使用同一任务长度/截断策略，记录实际运行 batch。精度、task 定义和有效样本集合不能因候选不同而改变。

正式结果不使用 limit。短 pilot 的分数不参与方法选择；缓存 key 必须含候选完整身份，禁止跨候选复用模型响应。旧 BF16 factor-hook 的结果不能作为本轮 FP32 分数。

## 10. 结果解释与统计汇报

主表：

| 状态 | KL16 | KL recovery | WT2 NLL/PPL | C4-fixed128 NLL/PPL | 五任务各自分数 | 五任务宏平均 |
|---|---:|---:|---:|---:|---|---:|
| Teacher | | | | | | |
| C0 | | | | | | |
| C1 | | | | | 不执行 | 不执行 |
| C2 | | | | | | |
| C3 | | | | | | |
| C4 | | | | | | |

另报 C0→C1、C1→C2、C2→C3、C3→C4、C2→C4 的绝对差值，明确符号和单位。NLL/PPL/KL 越低越好；任务准确率越高越好。

保留逐窗口/逐题结果，可用 2000 次配对 bootstrap 报探索性区间。文本窗口未必独立，16 窗口的区间不代表训练随机性或跨语料稳健性；没有多种子就不声称消除了统计估计噪声。跨零不是等效证明。

若 C4 优于 C3：支持“这套冻结的 K/V 组合几何在本设置有效”，不证明每一层/每个 K/V 都改善，不反推全局 Hessian 拟合更好。

若 C4 不优于 C3：报告统一规则在新校准分布及全模型联合干预下的边界，不能仅归因于样本不足、depth dependence 或某单一模块。不能在本轮改用层级 oracle 或调参追平。

需披露：方法曾在 WT2 开发；本轮改为 SlimPajama 校准；WT2 部分 validation/test 已被查看。因此跨语料评价有价值，但不称所有测试数据从未接触。

## 11. 保存与存储预算

### 11.1 永久保存

- 协议、代码版本/补丁、环境、配置、teacher/tokenizer/Wq 身份。
- 冻结校准 tokens、predictive labels、评价 tokens 或可验证资产引用、mask 与 hash。
- 去重后的最终 raw FP64 A/G、计数/归一化系数、样本绑定。
- 每份唯一 P64/Q64、rank、阻尼、谱摘要、数值验收、部署 hash。
- 候选 manifest 按模块引用因子，不复制完整候选模型。
- 逐窗口 NLL/KL/token 数、逐题选项分数/预测/计分结果、task 配置。
- 阶段 wall time、GPU call time、峰值 GPU/CPU、磁盘使用/读写、完成与恢复记录。

### 11.2 不新增全量持久保存

x/g、attention probabilities、delta、bar_x、autograd 图、完整 vocab logits、sequence score、完整 U/V、每候选 dense C/W_deploy、roots/inverse-roots 全集。

最终 raw A/G 足够复算 roots；P64/Q64 足够重建部署权重。不要同时永久保存 U、等比例 A、gauged A、damped A 和多套 roots。保留尺度/转换记录。

统计检查点只保留本 run 当前活动层组的当前/上一代；完成并校验最终统计后释放已被取代的本 run 临时文件。使用原子提交和哈希，不覆盖父目录。删除必须验证绝对路径位于本 run 的临时子目录；不清理旧服务器资产。

### 11.3 估算，单位 GiB

| 项目 | 最终容量 |
|---|---:|
| 普通 A：每层三个 4096²、一个 14336²，FP64 | 61 |
| Attention Marginal G | 8.5 |
| K Joint-one 额外 A/G | 4.25 |
| V Attention-aware 额外 A/G，按完整矩阵上界 | 4.25 |
| 最终 raw statistics 合计 | 78 |
| 416 份 rank64 P64/Q64 | 1.8125 |
| 224 模块 FP32 Wq，若需新增唯一副本 | 26 |

Wq 已有时永久新增约 80 GiB 加少量元数据/评价输出；需要新增 Wq 时约 106 GiB。峰值计划 120–160 GiB，建议正式运行前确认至少 **200 GiB 个人实际可用空间**，不是共享存储池 df 总剩余量。未核实配额时不能宣称空间足够。

上述不含下载全 C4、复制 teacher 或历史缓存；均不应发生。全量 Attention x/g 约 1.13 TiB，全部 P 约 4 TiB；本协议通过流式累计避免它们。

checkpoint 建议按活动层组每 32/64 窗口或约 20–30 分钟提交，pilot 后冻结频率。不要每 8 窗口写全 78 GiB 状态。根据主机 RAM 选择组大小，不将 112 GiB 本地 A6000 服务器内存假定为租用服务器内存。

## 12. 运行时间与费用

这是排期估算，不是全层新 run 的实测承诺。SlimPajama 与近期 WT2 的 N/L 相同，主变化是旧局部因子不能默认复用。

| 阶段 | 双 4090 初始规划 |
|---|---:|
| 只读审计、数据绑定、collector/数值 pilot | 0.5–1 小时 |
| Pass A、标签冻结 | 0.5–2 小时 |
| Pass B、Attention/K/V 统计 | 2–4 小时 |
| 去重 roots/SVD、候选冻结 | 1–3 小时 |
| KL16 + 六状态双语料 PPL、部署核验 | 约 1–2 小时 |
| 五状态 × 五个下游任务 | 另计约 4–12 小时，低置信度 |

整体先预留约 **10–24 小时级**；分组重复遍历、I/O 争用和 FP32 下游请求较长时可超出。统计/求解/双 PPL 可先按约 6–12 小时排期，不机械使用旧 12 模块十小时乘以模块数。

历史依据：缓存齐全的四层 K Joint-one 总约 8.8 分钟；V effective-stat kernel 约 0.72–0.74 秒/层/窗口，32层×256仅此 kernel 串行约 1.65 小时；旧 shared gradient 约 11.5 秒/窗口但目标层和 collector 不同；down 大矩阵 SVD 明显慢于 K/V。

工程验收后必须更新实测 ETA：

\[
t_{\rm remaining}\approx\sum_{\rm groups}N_{\rm remaining}\,t_{\rm window,group}
+t_{\rm solves}+t_{\rm evaluation}.
\]

额外记录数据准备、哈希、checkpoint 和候选切换耗时。GPU 异步/并行 kernel 时间之和不是 wall time。每个阶段及每 32/64 窗口报告进度和剩余量。

费用只按实际双卡实例小时价 × wall-clock 小时计算；无当前单价时报告 GPU-hours/实例小时，不编造金额。若 pilot 显示资源显著超预算，先报告实测瓶颈和不改变科学定义的分组/流式方案；不能自动换 H100 或延长租赁购买。

## 13. 实现组织与恢复

建议新增独立目录，不修改已冻结父实验：

~~~text
qera_mxint4_full_ag/experiments/full_model_attention_slim_v2/
  README.md
  protocol.md
  config.yaml
  run.py
  run.sh
  audit.py
  data.py
  collect.py
  solve.py
  evaluate.py
  report.py
~~~

阶段命令至少包括 audit / prepare-data / pilot / collect-a / collect-functional / solve / freeze / eval-kl-ppl / eval-downstream / report / resume。工程代码可复用父实现，不要求为目录形式重写成熟数学函数。

服务器输出：

~~~text
<server-root>/qera_runs/full_model_attention_slim_v2/run_01/
  protocol.md
  manifest.json
  frozen_config.json
  audit/
  data/
  statistics/
  factors/
  candidates/
  verification/
  kl/
  ppl/
  downstream/
  resources/
  summary/
  READOUT.md
~~~

manifest 绑定数据、teacher、Wq、代码、统计定义、solver、部署、评价协议；resume 时不匹配即拒绝接续。每阶段完成需有计数、文件哈希和 completion record。部分任务失败保留已完成结果，可按相同身份续跑；不能生成表示全部成功的空 completion。

无 stdout 长时间静默：后台日志需含活动层组、窗口进度、当前阶段、峰值资源与预计时间。服务/进程在已授权服务器会话内启动，保留 PID/日志/启动命令，避免重复启动两个 collector 争用显存。

## 14. 正式执行顺序

1. 读取本文、适用 AGENTS.md、父代码和服务器配置；完成只读资产及资源审计。
2. 冻结 SlimPajama tokens 与评价数据/task 清单；绑定 teacher/完整 Wq。
3. 完成数值与资源 pilot，确定流式分组、checkpoint 与实测预算。
4. Pass A，生成/冻结唯一 predictive labels 和完整 A_M。
5. Pass B，全 N256 采集所有缺失 Attention/K/V 统计。
6. 完成 416 份以内去重求解与部署验收；冻结全部 C0–C4 manifest。
7. 运行预定 KL16 与两个语料 PPL；技术正常即继续固定五任务下游，不按分数改模型或删掉输家。
8. 输出完整报告、所有状态与差值、失败项、实际时间/存储、复用清单及局限；同步轻量报告到本地，默认不下载 78 GiB stats。

## 15. 交付与完成标准

必须交付：实现源码及可恢复命令；完整冻结配置；资产审计；数值验收；六状态 KL/PPL；五状态五任务结果；候选与因子身份；逐样本证据；资源账单；READOUT.md。

READOUT 必须回答：

1. C0→C2 恢复多少预测与任务质量？
2. 固定 MLP A-only 后 C1→C2 的 Attention 补偿增量多少？
3. C2→C3 的 predictive Marginal 增量多少？
4. C3→C4 的结构规则组合增量多少？跨语料/任务是否一致？
5. 固定规则是否在 SlimPajama 校准与全模型联合量化下成立？若不成立，哪些指标具体失败？
6. 新统计/因子实际计算多少，复用了哪些；耗时、显存、主机 RAM、永久/峰值存储多少？

全部技术与评价任务完成才能标记 EXPERIMENT_COMPLETE；负结果也可以完成。数据/任务缺失时列为 PARTIAL/BLOCKED，不把预估或旧分数填成新结果。

## 16. 本地依据与优先复用代码

- [QERA 校准配置](../QERA/experiments/configs/w-only-uniform-rank.yaml)
- [SRR 3bit rank64 配置](../srr/experiments/configs/srr_3bit_rank64_iter1.yaml)
- [历史 SlimPajama 实验与预处理](../qera_mxint4_full_ag/experiments/qera_original_a_isolation/README.md)
- [双4090历史资产审计](../outputs/server_cache_audit_20260921/REPORT.md)
- [A6000服务器历史数据记录](../outputs/server_inventory_20260918/README.md)
- [旧成本审阅：仅保留资源计算依据](full_model_attention_review_cost_storage_20260921.md)
- [Shared teacher collector](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/shared_teacher.py)
- [Token-joint contraction](../qera_mxint4_full_ag/experiments/qer_kronecker_l10_v2/sketch_math.py)
- [K Joint-one 定义](../qera_mxint4_full_ag/experiments/qer_k_token_one_v1/README.md)
- [V effective variables 与归一化](../qera_mxint4_full_ag/experiments/qer_v_attention_proto_v1/v_math.py)
- [V collector](../qera_mxint4_full_ag/experiments/qer_v_attention_proto_v1/v_statistics.py)
- [FP64 weighted-SVD solver](../qera_mxint4_full_ag/experiments/qer_teacher_kl_exp03/ag_math.py)
- [K Joint-one 历史结果](../qera_mxint4_full_ag/outputs/chat_handoff_20260921/k_token_one_v1/READOUT.md)

另一 Codex 应沿用本文的科学设置完成实现与运行；可以优化计算组织，不因看到新统计量或局部负结果扩展方法搜索。
