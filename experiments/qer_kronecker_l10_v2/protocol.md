# L10 paired full-budget extension (2026-09-20)

This addendum overrides only the corresponding L31-specific clauses below.
Target: model.layers.10.self_attn.q_proj, MXINT3, rank64, full S0/S1/S2.
Reuse the completed L31 run's exact 32 fit /16 evaluation texts and saved labels,
including the original parent 8x4 labels. This is an exploratory paired layer
extension motivated by already observed L31 results, not a fresh unseen test set.
All 15 candidates, 224 fit gradients,256 eval gradients,240 primary KL rows,
3840 projections,48 bounds and article bootstrap tests remain mandatory.

The L10 W0/Wq asset is imported read-only from the frozen functional-gradient
run; teacher W0 and official MXINT3 source/config hashes are verified.
L31 factors, gradients, and fitted corrections are never reused for L10.

Engineering changes: exact FP64 article-grouped S=g.T@x contractions;
feature-panel sample Gram; two workers on distinct CUDA devices only in offline
phases. The FP32 teacher continues to occupy both GPUs in one process.
Gauge, alternating G-then-A updates, three synchronous Token-joint rounds,
ALS tolerances, exact dense SVD, deployment and all original checks are unchanged.
Reduction order changes are validated on toy and actual pilot data at 1e-10.
Evaluation g is additionally saved in its exact native FP32 representation
(~8 GiB), so output cap=128 GiB and entry free disk>=144 GiB.

The original L31 protocol follows for provenance; L31-only descriptions refer
to the parent experiment, not this L10 target.

---

# QER 曲率 sketch 与样本预算诊断：双 RTX 4090 完整实验协议

版本：v1.0，2026-09-20。

状态：执行协议，尚未实现或运行。本文件规定本轮研究问题、统计对象、数据预算、求解与评价规则；不代表任何预期方法已经有效。本轮使用双 RTX 4090，不使用 A6000 的单卡运行设置。

## 1. 研究主线与唯一核心问题

固定一个模块的量化权重和 rank64 补偿预算，研究：

> 不同低成本曲率统计所构造的 Kronecker 度量，其近似质量和 weighted-SVD 补偿质量，如何随标签采样数与文本覆盖增加而变化？

整轮只验证：

\[
\text{teacher-KL curvature }H
\longrightarrow K=A\otimes G
\longrightarrow C_{K,64}
\longrightarrow q_H(E-C_{K,64})
\longrightarrow\text{actual teacher KL}.
\]

本轮不优化 residual-specific mismatch，不对 H(E) 做 SVD，不加入谱集中正则，不搜索 rank、模块分配、标量补偿幅度或新的 A/G 参数化，不做全模型联合补偿、PPL、下游任务评测。

本轮同时回答两个具体问题：

1. 同样的 teacher 信息经不同 Kronecker 构造处理后，哪些近似能生成更好的 rank64 补偿？
2. 增加标签采样与增加独立文本，哪一种更能改善当前结果？

不得预设 Full-fit、Token-joint 或 Sequence-one-step 必然优于 Marginal。

## 2. 理论桥与实验要观察的量

固定 E=W-Wq，R=E-C，rank(C)≤64。定义：

\[
q_H(R)=\tfrac12\operatorname{vec}(R)^\top H\operatorname{vec}(R),\qquad
q_K(R)=\tfrac12\|G^{1/2}RA^{1/2}\|_F^2.
\]

对 A,G 正定，代理问题的精确解为：

\[
C_{K,64}=G^{-1/2}[G^{1/2}EA^{1/2}]_{64}A^{-1/2}.
\]

对于任意可行参考补偿 C0、R0=E-C0，令 RK=E-C_K。对任意 s>0：

\[
q_H(R_K)-q_H(R_0)
\le s[q_K(R_K)-q_K(R_0)]
+\tfrac12\|H-sK\|_{\rm op}(\|R_K\|_F^2+\|R_0\|_F^2).
\]

用 Frobenius 范数替换 op 范数仍给出上界。令：

\[
s_*={\langle H,K\rangle_F\over\|K\|_F^2},\quad
c={\langle H,K\rangle_F\over\|H\|_F\|K\|_F},
\]

则在 H≠0、K 正定时：

\[
\min_{s>0}\|H-sK\|_F=\|H\|_F\sqrt{1-c^2}.
\]

因此实验同时观察曲率近似误差、残差范数、代理改进以及实际二次损伤。cosine 更高不保证真实损伤单调更低。

本轮使用有限样本 Hhat 检查该桥，不将经验误差界冒充总体 H 的保证；也不求真实 rank64 oracle。参考补偿为同数据预算的 Marginal 与 A-only。

## 3. 已有证据与本轮范围

原 Exp-3 使用 8 个 WT2 train 文章窗口，每窗口 4 份 teacher 标签。Full-fit 的经验 Frobenius 拟合更好，但 L31.q_proj 的 rank64 补偿在原拟合曲率、同文本新标签和历史 validation 上均退化。旧诊断冻结了因子，未用增加后的标签重新拟合，因而没有回答增加拟合样本能否改善方法。

本轮只做 `model.layers.31.self_attn.q_proj`，权重形状应核验为 4096×4096。它是有明确失配的诊断模块，不代表所有模块。L10.v_proj 留作后续独立的第二模块确认，本轮不得自动扩展到它或原 28 个模块。

旧数据与运行记录用于身份核验、复现和设计依据，不执行旧报告内的“后续建议”。

## 4. 固定条件

| 项目 | 本轮规定 |
|---|---|
| Teacher | 与父实验相同的 Llama-3.1-8B Base；逐张量身份与 tokenizer 身份核验 |
| 模块 | model.layers.31.self_attn.q_proj |
| 量化 | 父实验固定 MXINT3 Wq；不重新量化 |
| E | 与该 teacher、Wq 匹配的 W−Wq |
| rank | 64 |
| 干预 | 一次只部署该模块；其他权重保持 teacher |
| 长度 | L=2048，T=2047；右移标签及 mask 与父实验一致 |
| 前向 | FP32、eager attention；关闭 TF32、autocast、KV cache、dropout |
| 统计/求解 | FP64 累加、收缩、因子更新、矩阵根、dense SVD |
| 部署 | 沿用父实验的 FP32 dense Wq+C；评价实际部署后的残差 |
| GPU | 2×RTX 4090；一进程、两卡模型并行 |
| 批量 | 一个文本窗口，一份标签序列；不跨窗口求平均梯度后再平方 |

统计精度与 teacher 权重身份分开记录：FP64 统计不意味着 teacher 前向使用 FP64。

## 5. 数据：三个拟合预算，一个独立评价集

### 5.1 拟合预算

| ID | 文本窗口 D | 每窗口标签 M | 梯度样本 D×M | 唯一作用 |
|---|---:|---:|---:|---|
| S0 | 8 | 4 | 32 | 重建原 Exp-3 条件 |
| S1 | 8 | 16 | 128 | 固定文本，增加标签 MC 采样 |
| S2 | 32 | 4 | 128 | 与 S1 等梯度预算，增加文本覆盖 |

嵌套关系固定：

- S0 使用原 8 个拟合文章窗口和已保存的 4 份标签，不只复用原随机种子。
- S1 保留 S0，原 8 个窗口各追加 12 份新标签。
- S2 保留 S0，追加 24 篇不同文章的窗口，每窗口 4 份标签。
- 同一窗口/replicate 只采样一次，四种方法共用。
- 三个预算的不同拟合样本总数为 8×16+24×4=224。
- 不做 32×16，不在看到结果后改变 D 或 M。

S1 对比 S0 主要识别固定文本下 MC 精度变化；S2 对比 S0 同时改变文本覆盖与总体估计精度；S2 对比 S1 是等梯度样本预算的数据配置比较，不能称为纯粹只改变一个因素。

32 个窗口是本轮诊断预算，不是“通用模型校准已足够”的声明。

### 5.2 新文本选择

沿用父实验 WT2 数据版本和分词。每篇文章只用首个符合条件的 2048-token 窗口，不把一个文章切成多个窗口充作独立文本。

在未使用过的 WT2 train 文章中构造 eligible pool，排除：

1. 原 8 个 fit 文章和所有已知历史评价文章；
2. 文档/title、token hash 重复；
3. 与既有 fit/eval 窗口共享连续 64-token 片段的候选。

对 eligible pool 按 SHA256(`qer-ksample-v1/data` + 文档稳定 ID) 排序，依次选取 24 篇追加 fit 文章，再选 16 篇 evaluation 文章；每次纳入均执行与已选集合的去重。所有窗口 ID、全文档 ID、token 范围、mask、hash 写入 manifest。若合格文章不足，报告缺口，不静默换语料、缩短长度或重复取同文章。

选样规则在生成补偿前冻结，不依据 loss、曲率或效果筛选文章。本轮是同语料独立文章诊断，不宣称跨域泛化。

### 5.3 独立评价集

固定 16 个新文章窗口，每窗口 16 份独立 teacher 标签，共 256 个评价梯度样本。

- 不与拟合文章或旧评价文章重合。
- 标签随机流与拟合侧隔离。
- 候选与超参数冻结后才计算正式效果。
- 不用评价集选择迭代数、阻尼、候选或拟合预算。
- 历史 8×64 validation 不作为本轮主评价，不为新候选默认重跑。
- 16×16 不保证分辨小差异；不确定就报告不确定，不按方法追加评价样本。

### 5.4 Teacher 标签定义

固定 teacher-forced 文本。对每个有效预测位置 j 独立采样 y_j~p_teacher(·|context_j)，完整词表、不用 top-k/top-p；采样标签不送回上下文。

一份标签序列包含 T 个采样标签。使用 sum-NLL，不能改成 mean-NLL。标签采样脱离 autograd；反传时将已保存标签视为常量。

新标签随机种子由 SHA256(`qer-ksample-v1/fit` 或 `/eval`，窗口 token hash，replicate ID) 派生；保存实际标签、种子及 hash。复用父标签时记录其旧身份，不重新生成假定等价的标签。

## 6. 统一统计定义与归一化

每窗口 c、每份标签 k：

\[
\ell_c^{(k)}=-\sum_{j=1}^{T}\log p_j(y_j^{(k)}),\quad
g_{ct}^{(k)}=\partial\ell_c^{(k)}/\partial h_{ct},\quad
S_c^{(k)}=\sum_{t=1}^{L}g_{ct}^{(k)}x_{ct}^{\top}.
\]

定义 N=DM，列优先 vec：

\[
\widehat H_{\rm full}={1\over NT}\sum_{c,k}
\operatorname{vec}(S_c^{(k)})\operatorname{vec}(S_c^{(k)})^\top,
\quad
\widehat q_{\rm full}(R)={1\over2NT}\sum_{c,k}\langle S_c^{(k)},R\rangle_F^2.
\]

这里 H 已吸收 1/T；窗口等权，窗口内标签等权。g_t 包含全部预测位置对模块位置 t 的影响，不能换成单个同位置 token loss 的梯度。

同位置联合统计目标为：

\[
\widehat H_{\rm pos}={1\over NT}\sum_{c,k,t}
(x_{ct}x_{ct}^\top)\otimes(g_{ct}^{(k)}g_{ct}^{(k)\top}).
\]

边际统计：

\[
A_m={1\over DL}\sum_{c,t}x_{ct}x_{ct}^\top,\qquad
G_m={1\over N L}\sum_{c,k,t}g_{ct}^{(k)}g_{ct}^{(k)\top}.
\]

输入 A 不中心化、不对角化，每窗口计一次，不因 M 增加重复加权。S0 与 S1 的 A_m 必须相同（直接共享同一统计资产）；S2 重新累计 32 窗口。

所有梯度都来自 teacher predictive 标签，不使用真实标签 CE-G，不使用 teacher 与自身 KL 的零梯度外积。

## 7. 四种 metric 构造

每种方法在每个 S0/S1/S2 预算上独立构造；数据、E、rank、求解器一致。全部原始因子进入统一 gauge、阻尼和 SVD。

### 7.1 Marginal

\[
A_{\rm raw}=A_m,\qquad G_{\rm raw}={L\over T}G_m.
\]

因此 K_raw=(L/T)A_m⊗G_m，与本轮 canonical 二次评分一致。

### 7.2 Token-joint，Sketch A 型固定三轮

目标为 H_pos 的 Kronecker 近似。初始化 A^(0)=A_m、G^(0)=I；固定三轮，同一轮两侧都使用上一轮因子：

\[
A^{(i+1)}={\sum_{c,k,t}(g_{ct}^{(k)\top}G^{(i)}g_{ct}^{(k)})x_{ct}x_{ct}^\top
\over NT\|G^{(i)}\|_F^2},
\]

\[
G^{(i+1)}={\sum_{c,k,t}(x_{ct}^\top A^{(i)}x_{ct})g_{ct}^{(k)}g_{ct}^{(k)\top}
\over NT\|A^{(i)}\|_F^2}.
\]

每个完整同步轮次后做保持 Kronecker 积不变的 gauge 归一化。保存三轮变化；主候选固定为第 3 轮。不得将其替换为交替更新而不记录版本；同步更新不要求目标逐半步单调，也不因三轮结束就声称全局最优或收敛。

逐位置 x、g 采用缓存流式收缩，固定标签反复使用，不每轮重新采样。只累计边际 Gram 或 S 不能重建本方法所需的全部联合统计。

### 7.3 Sequence-one-step，Sketch B 型

从 A^(0)=I_n、G^(0)=I_m 出发，两侧独立使用旧单位阵做一轮同步更新：

\[
A_{\rm raw}={1\over NTm}\sum_{c,k}S_c^{(k)\top}S_c^{(k)},\qquad
G_{\rm raw}={1\over NTn}\sum_{c,k}S_c^{(k)}S_c^{(k)\top}.
\]

m=n=4096 仍保留符号及维度检查。本方法不是从 Marginal 出发的一轮 ALS，不做第二轮，不将 G 更新依赖本轮刚计算的 A。

两因子乘积的原始幅值不保证等于 H 的幅值；保存原始定义，用统一 s* 作分析尺度校准，不按评价效果缩放补偿。

### 7.4 Sequence Full-fit

目标为 ||Hhat_full−A⊗G||_F²，从同预算 canonical Marginal 初始化。每轮依次：

\[
G\leftarrow{\sum_{c,k}S_c^{(k)}AS_c^{(k)\top}\over NT\|A\|_F^2},\qquad
A\leftarrow{\sum_{c,k}S_c^{(k)\top}GS_c^{(k)}\over NT\|G\|_F^2}.
\]

第二步使用刚更新的 G。保留父实验设置：最多 20 完整轮，relative J 改善≤1e−6 且 Kronecker product relative change≤1e−4，连续两轮同时成立才停止；每轮 gauge；不进行原始因子的特征值裁剪。

J=||A||_F²||G||_F²−2〈Hhat_full,A⊗G〉可为负。不得把负 J 当误差范数或据此算拟合误差下降百分比。保存全部轮次，最终收敛轮或第 20 轮为唯一主候选，不按评价选 checkpoint。不把 ALS 终止称为已证明全局最优。

## 8. 统一求解、基线及候选数量

### 8.1 因子处理

统一顺序：原始统计及 canonical 系数 → 对称性/PSD 检查 → gauge → 相对阻尼 → FP64 dense SVD → FP32 部署。

gauge 为 A←A/||A||F，G←G||A_old||F，保持 K 不变。

相对阻尼：

\[
A_s=A+10^{-3}{\operatorname{tr}(A)\over n}I,
\qquad G_s=G+10^{-3}{\operatorname{tr}(G)\over m}I.
\]

保存 K_raw 与 K_solve=A_s⊗G_s。补偿解释及主近似质量用 K_solve，raw 结果辅助分析。矩阵根由 FP64 对称特征分解得到；逆白化使用 solve，不显式求逆；采用 dense SVD，不换随机截断 SVD。阻尼后条件数上限每侧 1e8；失败时报告，不自动调阻尼使其通过。

### 8.2 部署残差

先得到 FP64 rank64 因子 P64,Q64，构造理论 C64。沿用父实验 FP32 因子/乘法及最终权重写入顺序，并在 manifest 中记录具体实现。

评价实际残差：R_deploy=float64(W_teacher_FP32)−float64(W_deploy_FP32)。不能用 E−C64 的评分冒充实际部署结果。

报告理论 R64 与 R_deploy 的差异。FP32 运算可能使实际残差不严格符合精确 rank64 数学模型；定理分析严格版使用 R64，部署损伤使用 R_deploy，不混用。此轮不自动切为 BF16 因子部署。

### 8.3 参考基线

- None：Wq，不补偿。
- A8：同原 8 个拟合文本的 A_m，G=I，rank64。
- A32：同 32 个拟合文本的 A_m，G=I，rank64。

A-only 的输入阻尼沿用同一 eta_A；G 侧使用 I（标量阻尼不改变精确补偿），明确记录。S0、S1 共用 A8，S2 对照 A32。不使用旧 256 窗口 A64 冒充同预算基线。

共 12 个 AG 候选 + A8 + A32 + None = 15 个不同部署。所有 15 个都接受主评价，不先筛选赢家再测 KL。不再加入普通 SVD64 或旧 functional-gradient 候选。

## 9. 曲率近似质量与理论诊断

### 9.1 无完整 H 的精确经验收缩

对于某个参考集合的 N 个 S，令 B_ij=〈S_i,S_j〉F：

\[
\|\widehat H\|_F^2={1\over N^2T^2}\sum_{i,j}B_{ij}^2,
\quad
\langle\widehat H,K\rangle_F={1\over NT}\sum_i\operatorname{tr}(GS_iAS_i^\top),
\quad
\|K\|_F^2=\|A\|_F^2\|G\|_F^2.
\]

由此计算 cosine c、s*、相对误差 d=sqrt(1−c²)、绝对误差 delta_F=||Hhat−s*K||F。采用分块/流式 Gram，不能构造 (mn)×(mn) 的 H。对舍入导致的微小负平方量给出受限 roundoff guard 和原值记录；较大负值直接失败。

两套 reference：

1. 各方法对应的 fit Hhat_full：说明训练拟合；不同预算 fit reference 不同，不能把其 raw 误差直接横比作总体改善。
2. 同一新 evaluation Hhat_full：所有预算/方法共享，作为主要外推诊断。

评价侧 s* 是事后尺度不变诊断，不参与因子、阻尼、补偿或候选选择。还应报告 canonical 未校准 q_K(E) 和 q_K(R)，避免尺度不变诊断掩盖实现归一化错误。

经验 Hhat 自身含 MC 与有限文本噪声；绝对 cosine 及 Frobenius 误差不视为总体无偏估计，不按它单独宣布方法胜负。

### 9.2 误差界诊断

每个 AG 候选对同预算 Marginal、A-only，使用理论残差 R64，计算：

\[
U=s_*[q_K(R_K)-q_K(R_0)]
+\tfrac12\delta_F(\|R_K\|_F^2+\|R_0\|_F^2).
\]

报告真实经验差值、代理改进项、误差项、U、U/qhat_full(E) 及两侧残差范数。核验经验差值≤U（考虑严格数值容差）；若 U 很大，报告界在本例缺乏约束力，不改变理论目标或为使界变小重新选因子。

额外保存 ||C||F、两侧条件数、白化谱尾、代理恢复率，作为解释变量。谱集中不是优化目标，不把代理恢复率接近 1 当成真实补偿成功。

## 10. 独立 q_full 与实际 KL 评价

### 10.1 完整二次评分

每个评价样本同时给所有 15 个部署残差打分：

\[
q_{ck}(R)={\langle S_{ck},R\rangle_F^2\over2T}.
\]

先窗口内对 16 份标签平均，再 16 窗口等权平均。共 256×15=3840 个样本级评分；保存未归约数值与每窗口结果。模型反向共享，不为每个候选重复反向。

定义恢复率 gamma_H(C)=1−qhat_full(R)/qhat_full(E)。不裁剪负恢复率。

### 10.2 Actual teacher KL

在相同 16 个评价窗口、完整部署幅度 alpha=1 下，逐候选替换目标模块，其他权重保持 teacher，计算每预测位置平均 KL(p_teacher||p_candidate)。使用完整词表，FP64 归约；稳定 log-softmax 与零值容差遵循原实现。

teacher 前向结果按窗口复用；可按位置分块算 KL，避免全窗口 FP64 logits 常驻 GPU。15×16=240 个主候选-窗口前向评价，不计 teacher/self-KL 及数值复核；KL 不依赖采样标签，不做 16 次重复 KL。

定义 gamma_KL=1−KL_candidate/KL_None。每次替换后恢复 teacher 目标模块，并核验恢复；不得累计部署多个候选。

除全部窗口 self-KL 外，对每个候选在固定第一个评价窗口做直接替换与等价输出扰动路径交叉核验，不做多 alpha 搜索。记录新候选的 q_full/KL 差异；不预设旧方向的二阶有效性已自动覆盖新方向。

## 11. 比较、置信区间和判断

### 11.1 预先指定比较

主比较为每个预算中三种非 Marginal 方法对同预算 Marginal，共 9 对。定义正值为改善：

\[
d_H={q_{\rm Marginal}-q_{\rm method}\over q_{\rm None}},\qquad
d_{\rm KL}={KL_{\rm Marginal}-KL_{\rm method}\over KL_{\rm None}}.
\]

样本量比较：每种方法 S1−S0、S2−S0、S2−S1，统一用“旧/参照损伤减新损伤”的方向，并明确正值含义。A-only 同预算比较辅助判断新增信息是否超过输入统计改善。

实际意义门限沿用 0.02，即未补偿损伤的 2 个百分点，不是相对 Marginal 损伤下降 2%。报告连续效应量，不只报告类别。

### 11.2 不确定性

bootstrap 固定 2000 次，种子由 `qer-ksample-v1/bootstrap` 派生，所有候选/对比共用重采样索引。

- 主区间：对 16 个文章窗口的已平均结果做配对 cluster bootstrap，每次重新计算总体分子/None 分母之比。q 与 KL 均可给出该文章抽样区间。不要把窗口当 token 独立重采样。
- MC 辅助区间：固定 16 个窗口，每窗口内对 16 份标签配对重采样，仅用于 q；描述条件 MC 误差。
- 不将上述两种区间宽度相加，不采用未经论证的双重 bootstrap 来重复计入方差。
- 这些区间均不涵盖重新拟合随机性。本轮没有多个独立 fit 重复，不能据此宣称某方法普遍更稳定。
- 百分位 95% 区间是探索性逐对区间，不控制 9 个主对比的整体错误率；不将挑出一个显著结果作为普遍优势结论。

判定：区间下端>+0.02 为明确改善且超过门限；上端<−0.02 为明确退化且超过门限；整个区间落在 [−0.02,+0.02] 内为当前评价精度下差异受限；其余为未确定，保留方向及区间。不得把未确定写成相等。

若 None 分母≤1e−12，记录该归一化不可判定并报告原始差值，不用强行抬高分母生成结论。

## 12. 双 RTX 4090 执行安排

### 12.1 运行前检查

核验两张 RTX 4090 的实际可用显存、GPU UUID、驱动/CUDA/PyTorch、算子实现、cgroup 内存上限、CPU 核数、磁盘余量和吞吐。不得把本地 A6000 服务器的 112 GB RAM 自动套到租赁服务器。

已有双 4090 运行记录提供可复用的模型并行路径，但不是本轮显存/时间保证。推荐沿用：

- GPU0：embed_tokens、rotary_emb、layers 0–15；
- GPU1：layers 16–31、最终 norm、lm_head。

记录精确 device_map。一个进程承载一个 FP32 teacher，不启动两个全模型副本，不使用每卡复制模型的 DDP。设备调度与磁盘路径由实际服务器确认，旧绝对路径仅作为资产查找线索。

### 12.2 梯度收集

只获取目标模块所需的 x、g、S，不给全模型参数开启梯度。目标输出设为可求导 leaf，保留从该模块输出到 logits 的完整下游路径；上游 teacher 前向可 no_grad，不能切断目标 q_proj 经 attention 到预测的路径。

使用已验证的 suffix checkpoint 路径（如有）；checkpoint 不应重复累计 hook。逐样本释放图；新标签可重放前向，但不改变 teacher activation。采样、logit/softmax 临时张量、梯度以及统计缓存分阶段释放/搬至 CPU。

收集阶段两卡都用于 teacher，不同时安排大规模 FP64 ALS 或 eig/SVD。OOM 时停止并报告，不静默改 BF16、长度、模块或样本数。

### 12.3 离线统计与求解

拟合数据收集后卸载 teacher，再从缓存构造四种 metric、运行迭代及 SVD。独立因子任务可按 GPU 分配，但只在 pilot 验证显存、CPU RAM、I/O 后开启；默认按任务串行，避免两个任务同时占满主存。

4090 的 FP64 dense 计算需要单独计时。两卡不会自动加速单个 dense eig/SVD；不得套用 BF16 吞吐或单卡 A6000 耗时作承诺。

候选冻结后重载同一 teacher，收集评价 S、为所有候选打分并做 KL。最后卸载 teacher，离线完成参考曲率 Gram、metric 收缩及理论界分析。

### 12.4 缓存与磁盘预算

针对 4096×4096 的 S，FP64 每份约 128 MiB：

| 缓存 | 数量/格式 | 近似体积 |
|---|---|---:|
| fit S | 224 份 FP64 | 28 GiB |
| eval S | 256 份 FP64 | 32 GiB |
| fit g | 224×2048×4096，teacher 原生 FP32 | 7 GiB |
| fit x | 32×2048×4096，FP32，每窗口一次 | 1 GiB |
| eval x（若保留） | 16×2048×4096，FP32 | 0.5 GiB |
| 因子/根/候选/临时文件 | 依实现 | 另留余量 |

建议在不含模型 checkpoint 的实验目录预留至少 128 GiB 空闲磁盘。保存 x/g 的 FP32 原始值后转 FP64 收缩，不对 teacher FP32 梯度额外量化；不必为每个预算重复存同一 S，也不缓存巨大的全词表每样本 logits。

CPU 使用有界缓存，原则上只保留少量 S/块，目标在 64 GiB 可用主存下执行；推荐可用 96 GiB 以上。实际可行性由 pilot 决定，不一次将 60 GiB 的 S 与模型/中间量全部载入 RAM。输出分块原子写入、哈希与断点恢复，保留原资产只读。

## 13. Pilot、验收和停止规则

### 13.1 仅拟合侧 pilot

使用 S0 首个窗口的前两份已保存标签，完成：

1. teacher/Wq/tokenizer/样本身份；双卡 forward；self-KL。
2. 目标输出 x/g 获取、S=Σgxᵀ，与目标权重 autograd 梯度核验。
3. 两种投影路径〈S,R〉与Σ_t g_tᵀRx_t 核验。
4. Marginal、Token-joint 一轮同步、Sequence-one-step 和 Full-fit 一完整轮的收缩核验。
5. FP64 dense rank64 SVD、逆变换、FP32 部署与恢复。
6. 少量 S 的 Gram 及 metric contraction 双实现核验；小维合成数据可显式构造 H 检查列优先 vec、T/L 系数、所有更新和误差界。
7. 反向、缓存写读、各类矩阵收缩、SVD、KL 单次时间和资源峰值。

pilot 只验证实现，不查看新评价集效果。合格的 pilot 样本进入正式缓存，不重复计入预算。

### 13.2 数值门限

沿用父实验并明确跨硬件例外：

- teacher 参数与 Wq 身份要求精确；跨硬件 activation 相对误差≤1e−5，父 S 重放相对误差≤1e−4。后者是沿用已记录双 4090 迁移容差，不扩展为所有运算的容差。
- 本轮 S 对当前 autograd 梯度、投影等价检查≤1e−5（接近零的分母用明示的绝对保护规则）。
- FP64 对称性及独立收缩检查≤1e−10；PSD 负特征值不得低于 −1e−10×最大特征值。
- 根重建≤1e−10；SVD 代理损失与谱尾恒等式、逆变换≤1e−8。
- 理论 C64 到实际部署的代理目标相对漂移≤1e−4；接近零目标单独报告绝对误差。
- self-KL 与重复路径使用父实验的尺度保护规则；最大输出干预路径 KL 相对误差≤2%。

所有比例的分母保护、绝对容差应在 pilot 后正式 manifest 中写明，不按单个候选放宽。重大失败停止相关阶段并保留记录，不以换精度或裁剪数值掩盖失败。

### 13.3 资源与时间门限

单卡 pilot GPU peak allocated 建议≤21.5 GiB，并检查 reserved 和设备实际可用显存；主存不得超过真实限制的 85%。文件系统始终保留临时写入和恢复余量。

完成 pilot 后，用实测吞吐写出分阶段估算：224 份拟合收集、Token-joint 三预算各三轮、Full-fit 三预算最多各20轮、Sequence-one-step、全部 SVD、256 份评价反向及 240 次候选前向、参考 Gram/metric 收缩和 I/O。不能把“480 个不同样本”直接等同于全部计算工作量。

正式预算应按上述上界加资源余量与实际租期一起冻结。当前未连接服务器，本文不虚构小时数或服务器内存。若估计超过实际预算，先报告瓶颈与可完成阶段，不能静默减少预定候选、窗口、迭代上限或精度。预算不足时保留检查点，不自动续租或终止他人进程。

## 14. 顺序与复用要求

1. 只读盘点既有资产，记录来源、哈希及实际路径；缺少原 Wq/标签/teacher 身份时报告，不替造复现条件。
2. 冻结 32 个 fit、16 个 eval 窗口及全部标签 ID；不计算评价效果。
3. 完成 pilot，核验与记录计算预算。
4. 本机双卡重建 S0 的统计；父缓存用于复核。不要将 A6000 与本轮双 4090 的不同数值来源混成某一个主 fit。
5. 收集共224份 fit S/g 和32份 x，给 S0/S1/S2 建索引视图。
6. 对三预算构造四方法，生成12个候选和A8/A32，冻结 manifest/因子/部署权重 hash。
7. 对16×16新评价样本收集S并给全部15候选评分；完成240个实际KL主评价。
8. 完成 fit/eval 曲率近似质量与理论界分析、区间及报告。
9. 核验输出完整、停止本实验进程、记录GPU/租约状态；不操作无关服务。

不按阶段7效果回到阶段6调参。若发现实现错误，版本化全部受影响产物并解释重跑范围。

## 15. 交付产物

建议新目录：`outputs/qer_kronecker_sketch_sample_pilot_dual4090_<run_id>/`，不覆盖任何已有 Exp-3 或 functional-gradient 输出。

至少包含：

```text
protocol.md
manifest.json                     # 样本/代码/硬件/归一化/预算/冻结时间
teacher_identity.json
data/fit_windows.json
data/eval_windows.json
data/sample_index.json            # 224 fit、256 eval 与各预算索引
data/labels/                      # 实际标签及 hash
cache/fit_x/                      # 32 窗口
cache/fit_g/                      # 224 份
cache/fit_S/                      # 224 份
cache/eval_S/                     # 256 份
factors/<budget>/<method>/        # raw/solve、迭代日志、谱/条件数
corrections/                      # 12 AG + A8 + A32，理论与部署身份
scores/q_per_sample.csv           # 3840 行
scores/kl_per_window.csv          # 240 主行，self/check另表
summary/curvature_quality.csv
summary/bound_diagnostics.csv
summary/paired_comparisons.csv
summary/sample_budget_effects.csv
verification.json
resource_usage.json
RESULTS.md
READOUT.md
```

报告只需以下主图表：

1. 三预算四方法的 evaluation curvature error/cosine 表或图，同时区分 raw/solve。
2. 同一候选的 q_full 和 actual KL 恢复率及配对区间。
3. S1 vs S2 等梯度预算比较。
4. 代理改进、误差项、excess-loss bound 与实际差值的对照表。

精确数据、每窗口结果和不确定结果全部保留，不只展示最好候选。

## 16. 结果解释与下一步边界

| 观察 | 允许的解释 | 不能直接推出 |
|---|---|---|
| S1 比 S0 改善 | 增加拟合标签可能有帮助 | 文本覆盖已充分、方法已稳定 |
| S2 比 S1 改善 | 等梯度预算下当前新增文本配置更有用 | 任意语料或模块都应少采标签 |
| fit 更好、eval curvature 更差 | 有经验拟合到独立参考的失配证据 | 已唯一证明过拟合机制 |
| eval curvature 更好、q_full 更差 | 全局近似指标不足以排序该实例的补偿 | 所有 Kronecker metric 无效 |
| Token-joint/one-step 更好 | 相应既定构造在本模块值得确认 | 已完成新的通用 QER 算法 |
| q_full 改善、KL 不改善 | 新方向上需检查二阶近似与部署数值 | 立刻更换 A/G 或引入新目标 |
| 区间未确定 | 当前预算分辨力不足 | 两方法相等 |

本轮不因某个局部现象重写研究主线。后续是否增加模块、文本或拟合重复，应由本轮完整结果决定并另立协议。

## 17. 本地参考资产与文献定位

资产路径相对本文件：

- [原 Exp-3 协议](qer_teacher_kl_exp03_full_fit_ag_20260918.md)
- [原 Exp-3 manifest](../outputs/qer_teacher_kl_exp03_20260918/results/manifest.json)
- [Exp-3 冻结诊断](../outputs/qer_teacher_kl_exp03_diagnosis_20260918/RESULTS.md)
- [双4090历史执行身份](../outputs/functional_gradient_4090_run04_final_20260920/identity.json)
- [双4090历史资源记录](../outputs/functional_gradient_4090_run04_final_20260920/resource_usage.json)
- [既有FP64收缩与weighted-SVD代码](../qera_mxint4_full_ag/experiments/qer_teacher_kl_exp03/ag_math.py)

原 Exp-3 identity：`34fd1ee3b82701316a3ce90808b911402f197b90538b97a7c0060a6ec47dc388`。

原诊断 identity：`41b88a5b3c6cd0ca0afcb822d7b0675dd31239794ec92f2e47ecf1a1ebb02fbd`。

双4090历史运行 identity：`e230c9dfb2e2188f18c2d042dbe339456e49bdd22beaafa78f2a6b03b30e61ad`。

Token-joint 与 Sequence-one-step 分别参考 [YAQA §3.2 Sketch A/B](https://arxiv.org/html/2505.22988v2#S3.SS2) 的统计构造。本轮明确其采样、归一化、初始化和更新顺序，应用于固定残差的 rank64 weighted-SVD；不将 YAQA rounding 的性能保证直接移植为本轮 QER 保证。

执行原则：每个新增计算必须服务于“曲率近似→固定rank补偿→实际KL”中的一个环节。按本文件实现与验证，不自动启动服务器实验或扩大租赁资源。
