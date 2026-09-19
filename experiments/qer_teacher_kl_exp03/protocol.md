# QER 实验三：从完整 predictive curvature 拟合 single-Kronecker metric，并生成 weighted-SVD 补偿

版本：v1.1，2026-09-18。修订：明确完整曲率与 marginal metric 的 canonical 系数、两种尺度自由度以及阻尼的尺度等变性。

状态：**实验设计文档；尚未实现或执行 Exp-3。科学问题、核心对照和评价原则已确定；执行参数须按第 12 节在正式运行前冻结。本文不代表已获得实验结果，也不授权自动启动服务器实验。**

## 1. 核心问题与实验定位

> 从完整 predictive curvature 直接拟合 single-Kronecker metric，能否在保持 weighted-SVD 闭式低秩求解的同时，比 marginal 构造更准确地保留补偿收益，并生成真实 functional damage 更低的 rank-64 correction？

本轮沿用以下研究主线：

- **Exp-1**：在已测试模块、残差方向和扰动幅度上，验证完整二次评分与实际 KL 的一致性；这一支持不自动覆盖所有新补偿方向。
- **Exp-2**：在固定方向上，分别诊断删除位置交叉项、再以边缘统计乘积替代联合统计带来的误差。
- **Exp-3**：保持 single-Kronecker 与 weighted-SVD 求解形式，改变左右因子的构造方式，真正生成并评价新补偿。

核心比较是 **Marginal-AG64 vs Full-fit-AG64**。不预设 Full-fit 一定改善，也不根据结果是否符合预期更换主模块、数据或判定规则。

### 1.1 本轮需要区分的两个问题

1. **metric 是否更准确？** 对同一组补偿，拟合 metric 是否更准确地预测完整曲率下的补偿收益？
2. **correction 是否更好？** 拟合 metric 生成的新补偿，是否降低完整曲率评分与实际 KL？

两者分别评价。不能只凭曲率拟合更好宣布补偿更好，也不能只凭一个补偿改善宣布 metric 在所有相关方向上更准确。

### 1.2 本轮不做什么

- 不把双层的 task-aware metric optimization 作为本轮算法。
- 不引入 sum-of-Kronecker、补偿的直接梯度优化、全模型联合补偿、rank 搜索、PPL 或下游任务。
- 不重新量化，不修改原 SVD64/A64 补偿，不覆盖 Exp-1/2 的资产和结果。
- 不把单次拟合结果称为 single-Kronecker 家族的 QER 能力上限。

## 2. 背景证据与结论边界

Exp-2 中，`model.layers.10.self_attn.v_proj` 在三个固定方向上的两步结构误差和总误差均达到原协议的实质差异标准；`model.layers.31.self_attn.q_proj` 的部分第二步比较误差较小，但整体仍有未确定项。

因此，本轮以 L10.v_proj 为预先指定的重点模块，保留 L31.q_proj 为对照模块，并完整报告二者。原结果只是选择后续构造的依据，不证明新构造必然有效。

必须区分：

\[
K_{\mathrm{fit}}\in\arg\min_{K=A\otimes G,\ A,G\succeq0}
\|H-K\|_F^2
\]

与概念性的：

\[
K_{\mathrm{task}}\in\arg\min_{K=A\otimes G,\ A,G\succ0}
q_H(E-C_K).
\]

后者仅用于说明研究目标；不假设其最优值必然达到，不在本轮求解。前者的最优 metric 不必产生后者意义下的最佳补偿。

实现所得因子记作 `A_fit, G_fit` 或带帽估计量。只有给出相应求解证据，才能称其为特定经验拟合问题的最优解；ALS 停止不等于已证明全局最优。

## 3. 固定设置与比较组

| 项目 | 约定 |
|---|---|
| 模型 | 沿用 Exp-1/2 的 Llama-3.1-8B Base teacher，核验权重、tokenizer 和运行身份 |
| 模块 | `model.layers.10.self_attn.v_proj`、`model.layers.31.self_attn.q_proj` |
| 量化 | 沿用原 MXINT3 的固定 `Wq`，不重新量化 |
| 误差 | `E = W - Wq`；原 `R_none` 应与其数值约定一致 |
| 补偿 rank | 64；主实验不搜索其他 rank |
| 干预 | 一次只替换一个目标模块；其余模块保持 teacher 权重 |
| 输入长度 | 沿用 L=2048，T=2047；有效预测位置和 mask 与父实验一致 |
| 评价窗口 | Exp-1/2 保存的 8 个 WT2 validation 窗口 |
| 评价标签 | 沿用每窗口 64 份已保存的 teacher 标签，共 512 份；所有候选共用 |
| 数值前向 | 沿用父实验 FP32/eager，关闭 TF32、autocast 和 KV cache；额外变更必须版本化 |

四个补偿组，加一个不补偿参照：

| 名称 | 补偿来源 | 本轮角色 |
|---|---|---|
| None | C=0，残差 E | 损伤与恢复率的共同分母 |
| SVD64 | 原冻结普通 SVD64 补偿 | 历史基线 |
| A64 | 原冻结 activation-aware 补偿 | 历史基线；原统计来源保持不变 |
| Marginal-AG64 | 新拟合数据的边缘二阶矩，双侧 weighted SVD | 核心对照 |
| Full-fit-AG64 | 同一新拟合数据的完整曲率拟合因子，双侧 weighted SVD | 新构造 |

核心后两组必须共享相同的 E、rank、拟合样本及权重、weighted-SVD 实现、精度、归一化和阻尼规则。数据完全一致意味着窗口、teacher 标签、有效位置和计数一致，不只是使用同一语料名称。

原 A64 的数据来源不同，因此不承担“只改变因子构造方式”的因果比较。A64 和 SVD64 冻结复用；新 AG 两组不得覆盖其文件。

## 4. 数据划分与信息隔离

### 4.1 拟合集 D_fit

使用新的文本窗口，计算 Marginal-AG 和 Full-fit-AG 所需的全部统计。

- 本文建议优先考虑 WT2 train 的固定窗口，减少额外语料差异；**具体数据版本、文档、token 区间、窗口数量和选取规则尚待冻结**。
- 核验与 D_eval 的样本不重合；记录文档标识、token 区间和 token hash。不能只换窗口编号而实际重用相同内容。
- 不使用原 8 个评价窗口的 activation、gradient、完整曲率或 KL 拟合新因子。
- 若需要选择阻尼、迭代规则等，使用拟合侧单独划分的开发集 D_dev；不得使用 D_eval 选参。
- teacher 标签在原 teacher 分布下采样，不把采样标签送回上下文。

### 4.2 历史固定评价集 D_eval

\[
D_{\mathrm{fit}}\cap D_{\mathrm{eval}}=\varnothing.
\]

原 8 个 WT2 validation 窗口完全不参与 factor fitting，但已经在 Exp-1/2 中用于研究判断，因此称为**不参与拟合的历史固定评价集**，不称为从未接触过的盲测集。

- 因子、补偿、超参数、代码身份冻结后，才执行正式评价。
- 不按 D_eval 结果选择拟合初始化、迭代 checkpoint、阻尼或采样量。
- 正式评价后若改方法，单独登记新版本，不能覆盖本轮失败或不确定结果。
- 同窗口更换 teacher 标签只检验标签采样外推，不等于新文本泛化。

## 5. 完整曲率与 marginal 基线的统一定义

### 5.1 符号与 teacher 梯度

目标 Linear 的权重为 `d_out × d_in`，输入 x_t 和输出梯度 g_t 均按列向量记。

对窗口 c、teacher 标签 replicate k：

\[
\ell_c^{(k)}=-\sum_{j\in\mathcal J_c}\log p_j(y_j^{(k)}),
\qquad g_{ct}^{(k)}=\frac{\partial\ell_c^{(k)}}{\partial h_{ct}}.
\]

使用求和 NLL，不改为 mean loss；不使用真实标签 CE，也不对 teacher 自身 KL 的零梯度求外积。g_t 包含所有有效预测位置通过下游计算对模块位置 t 的影响。

定义每个窗口和 replicate 的完整共享权重梯度：

\[
S_c^{(k)}=\sum_{t=1}^{L}g_{ct}^{(k)}x_{ct}^{\top}.
\]

必须先按位置相加，再形成二次统计。令 vec 为列优先向量化：

\[
H_D=\frac1T\mathbb E_{c,k\sim D}
[\operatorname{vec}(S_c^{(k)})\operatorname{vec}(S_c^{(k)})^\top],
\]

\[
q_{H_D}(R)=\frac12\operatorname{vec}(R)^\top H_D\operatorname{vec}(R)
=\frac1{2T}\mathbb E_{c,k\sim D}[\langle S_c^{(k)},R\rangle_F^2].
\]

**本文件将 1/T 吸收到 H 中。** 这就是本轮的 canonical curvature convention，也可记为 H_bar。若代码内部使用未吸收系数的统计缓存，必须在进入因子拟合、阻尼、求解和评分路径前按本节定义转换并给出等价核验；不得以“整体缩放不影响 SVD”为由省略系数。

窗口等权，每窗口内部 replicate 等权。H_fit 与 H_eval 分别由对应数据估计，不把经验估计写成已知的总体曲率。

### 5.2 Marginal-AG

在 D_fit 上计算完整通道、未中心化的二阶矩：

\[
A_{\mathrm{marg}}=\frac1L\mathbb E_c\sum_t x_{ct}x_{ct}^{\top},
\quad
G_{\mathrm{marg}}=\frac1L\mathbb E_{c,k}\sum_t g_{ct}^{(k)}(g_{ct}^{(k)})^\top.
\]

这里 G_marg 始终指未额外吸收 L/T 的边缘二阶矩。canonical metric 定义为：

\[
\boxed{K_{\mathrm{marg}}=\frac LT A_{\mathrm{marg}}\otimes G_{\mathrm{marg}}.}
\]

于是：

\[
q_{K_{\mathrm{marg}}}(R)
=\frac{L}{2T}\operatorname{tr}(G_{\mathrm{marg}}RA_{\mathrm{marg}}R^\top),
\]

与 Exp-2 的 sep 定义在相同数据上严格一致。这里 A 每个窗口只计一次，不能因 K 次反传重复计数。不得引入通道对角化、中心化或未登记的低秩压缩。

为交给统一的双因子求解接口，在做 gauge 归一化和 damping 前先设置：

\[
A_{\mathrm{raw}}^{\mathrm{marg}}=A_{\mathrm{marg}},\qquad
G_{\mathrm{raw}}^{\mathrm{marg}}=\frac LT G_{\mathrm{marg}}.
\]

接口使用 `K_raw = A_raw ⊗ G_raw`。原始边缘统计与 canonical 因子使用不同字段保存，避免吸收两次或漏掉 L/T。Full-fit 拟合的目标 H_fit 已含 1/T，其输出因子直接按 `K_raw = A_fit ⊗ G_fit` 使用，不再额外乘 L/T。

### 5.3 两种尺度自由度与统一处理顺序

必须区分：

1. **因子 gauge**：`(A,G) → (cA,G/c)`，c>0，K 完全不变。
2. **metric 整体尺度**：`K → sK`，s>0，二次目标乘 s。无阻尼时最优补偿集合不变，但原始预测分数和 Frobenius 拟合误差会变化。

两组统一按以下顺序处理：

```text
原始统计/拟合目标的计数与系数核验
→ canonical K_raw（完整曲率含 1/T；marginal 含 L/T）
→ 保持 K_raw 不变的因子 gauge 归一化
→ 共同的相对阻尼规则
→ K_solve 与 weighted SVD
→ raw/solve 评分分别报告
```

“同尺度定义”指双方对应相同的每预测位置二次评分约定，不是强行使 trace、Frobenius 范数或 q_K(E) 相等。完整曲率与 marginal 的幅值差可能来自真实统计失配，必须保留报告。不得利用 D_eval 对齐幅值，也不得把漏掉 L/T、T 或样本均值系数的实现错误解释为无关缩放。

## 6. Full-fit-AG：直接拟合经验完整曲率

### 6.1 拟合目标

\[
\min_{A,G\succeq0}\|\widehat H_{\mathrm{fit}}-A\otimes G\|_F^2.
\]

这是一种有明确数学依据的全局曲率拟合基线，不等于直接优化最终补偿收益。拟合阶段不使用实际 KL 或评价集上的补偿质量选择因子。

### 6.2 无需显式 H 的交替更新

以非零半正定 A 初始化。对固定 A，精确块更新为：

\[
G\leftarrow
\frac{\mathbb E_{\mathrm{fit}}[SAS^\top]}{T\|A\|_F^2}.
\]

随后用更新后的 G：

\[
A\leftarrow
\frac{\mathbb E_{\mathrm{fit}}[S^\top GS]}{T\|G\|_F^2}.
\]

这些统计保留交叉位置贡献，例如：

\[
SAS^\top=\sum_{t,u}g_t(x_t^\top A x_u)g_u^\top.
\]

允许流式累计、缓存 S 或使用数值等价的收缩，不能为了省计算退回仅含 t=u 的 marginal 统计。固定同一批样本完成确定性 ALS；若改为随机 minibatch 更新，则属于不同求解器，不能沿用确定性单调性验收。

### 6.3 因子尺度和收敛记录

固定 gauge 为 `||A||_F = 1`。若原 A 范数为 a，则同时执行：

\[
A\leftarrow A/a,\qquad G\leftarrow aG.
\]

保持 A⊗G 不变。不能只归一化 A 而不补偿 G。

监测去掉常数项的精确经验目标：

\[
J(A,G)=\|A\|_F^2\|G\|_F^2
-\frac2T\mathbb E_{\mathrm{fit}}[\operatorname{tr}(GSAS^\top)].
\]

J 与原 Frobenius 目标只差 `||H_fit||_F^2`，可以比较同一数据上不同 K 的拟合质量；J 可以为负，不能将 J 本身称为误差范数。若报告绝对/相对 Frobenius 误差，须额外核验常数项，或明确使用了什么估计。

记录每步 J、乘积变化量、因子谱、对称误差、运行成本和停止原因。不以 gauge 未固定时的单个因子变化判断收敛。

主初始化建议使用吸收 L/T 并做 gauge 归一化后的 marginal canonical 因子；拟合侧可预先加入单位阵初始化作求解稳定性检查。初始化数量、选择规则、停止阈值和最大迭代次数均须预先冻结。若选择多个结果中的一个，只按拟合侧预设准则选择，不按 D_eval 选择。

## 7. 正则化与 weighted-SVD 补偿生成

### 7.1 Raw metric 与 solve metric 分开

每组至少保存：

1. 原始统计、样本计数、L/T 或 1/T 系数，以及 canonical/gauge 处理后的 `A_raw, G_raw`；统计缓存与 metric 因子字段分开。
2. 对称化、谱检查结果及任何数值修复记录。
3. 阻尼系数和实际求解因子：`A_solve, G_solve`。
4. 生成的补偿、实际部署表示和残差。

本方案采用共享的相对阻尼形式，具体 eta 值仍待冻结：

\[
A_{\mathrm{solve}}=A_{\mathrm{raw}}+
\eta_A\frac{\operatorname{tr}(A_{\mathrm{raw}})}{d_{\mathrm{in}}}I,
\]

\[
G_{\mathrm{solve}}=G_{\mathrm{raw}}+
\eta_G\frac{\operatorname{tr}(G_{\mathrm{raw}})}{d_{\mathrm{out}}}I.
\]

两个 AG 组使用相同的 eta 规则，而非强行使用相同绝对 lambda。eta 的具体值或拟合侧选择规则在正式评价前冻结。若 trace 为零、因子存在实质负特征值或条件数导致数值验收失败，停止并记录原因，不静默换规则。

拟合过程不通过未声明的 damping 改写 raw 拟合目标。最终解对应 solve metric，而不是 raw metric。若存在特征值裁剪、floor 或自适应阻尼，必须事先定义并单独记录，不能隐藏为数值细节。

**尺度边界：固定绝对阻尼与相对阻尼不能混为一谈。** 定义：

\[
\mathcal D_\eta(M)=M+\eta\frac{\operatorname{tr}(M)}d I.
\]

对 a>0，有 `D_eta(aM) = a D_eta(M)`。因此，对 a,b>0 且 eta 规则不变：

\[
\mathcal D_{\eta_A}(aA)\otimes\mathcal D_{\eta_G}(bG)
=ab\,[\mathcal D_{\eta_A}(A)\otimes\mathcal D_{\eta_G}(G)].
\]

- 当 ab=1 时，这是 gauge 变化，阻尼后的 K_solve 不变。
- 当 ab=s 时，这是 metric 整体缩放，阻尼后的 K_solve 也只乘 s；理论上最优补偿集合不变，原始 q 分数乘 s，归一化收益不变。

因此，不能声称本方案的 trace-relative damping 必然受纯整体尺度影响。canonical 定义仍然必须统一，以保证统计、拟合目标、原始分数和工程接口可比。固定绝对 lambda、绝对特征值 floor、隐藏的绝对 epsilon、尺度相关的自适应分支等可能破坏上述性质，必须逐项检查。机器精度范围内的舍入误差另行核验。

### 7.2 补偿公式

对正定求解因子，记：

\[
B=G_{\mathrm{solve}}^{1/2}EA_{\mathrm{solve}}^{1/2},
\qquad B_{64}=[B]_{64}.
\]

\[
\boxed{
C=G_{\mathrm{solve}}^{-1/2}B_{64}A_{\mathrm{solve}}^{-1/2}
}
\]

它精确求解理想精度下的：

\[
\min_{\operatorname{rank}(C)\le64}
\frac12\|G_{\mathrm{solve}}^{1/2}(E-C)A_{\mathrm{solve}}^{1/2}\|_F^2.
\]

验证该值与 B 的 rank-64 截断尾能量的一半一致，并检查变换回原空间的重构误差。

“一次 weighted SVD”指给定 metric 后的补偿求解；不意味着统计、因子拟合及矩阵平方根无需计算。若使用近似/随机 SVD，必须单独披露求解误差，不能仍声称实际计算是精确闭式解。

构造优先采用稳定的高精度线性代数；与父实验对齐的实际 FP32 补偿/因子表示也必须保存。评价使用实际部署后的同一残差，不能 q_H 评价高精度理论 C，而 KL 使用另一份舍入后的 C。若通过两个低秩因子执行，核验其乘积与记录的 C 一致。

## 8. 统一评价：补偿质量和 metric 质量分别报告

### 8.1 完整曲率评分

四个补偿与 None 在 D_eval 上共用完整曲率裁判：

\[
q_H(E-C)=\frac1{2T}\mathbb E_{\mathrm{eval}}
[\langle S,E-C\rangle_F^2].
\]

这里 H 默认指评价集的 teacher-sampled 完整曲率估计。所有方向复用同一窗口和标签，先计算配对差，再估计不确定性。

新补偿不能从已有三个方向的评分插值取得 full 分数；需要重放得到其完整投影。已有标签可以复用，已有方向标量不足以代替新方向统计。

### 8.2 实际 KL

针对每个模块与补偿，以 `Wq + C = W - (E-C)` 替换目标模块，其他模块保持原 teacher。计算：

\[
\mathrm{KL}(C)=\frac1{NT}\sum_{c,j\in\mathcal J_c}
D_{\mathrm{KL}}\bigl(p_{cj}^{\mathrm{teacher}}\|p_{cj}^{C}\bigr).
\]

这是完整幅度 alpha=1、teacher 到干预模型的全词表 KL；采用父实验一致的位置、归一化与数值计算路径，不用采样标签 NLL 差替代 actual KL。

新 AG 补偿必须重新评价。历史基线可在完整身份一致并通过重放校验后复用旧 KL，否则统一重算；所有复用均记录出处。

### 8.3 补偿恢复率和核心配对差

\[
\gamma_H(C)=1-\frac{q_H(E-C)}{q_H(E)},
\qquad
\gamma_{\mathrm{KL}}(C)=1-\frac{\mathrm{KL}(C)}{\mathrm{KL}(0)}.
\]

None 的恢复率为 0；恢复率可能为负，不裁剪。分母接近零时标记不可稳定解释，不强行输出比值。

核心比较：

\[
\Delta_H=q_H(E-C_{\mathrm{marg}})-q_H(E-C_{\mathrm{fit}}),
\]

\[
\Delta_{\mathrm{KL}}=\mathrm{KL}(C_{\mathrm{marg}})-\mathrm{KL}(C_{\mathrm{fit}}).
\]

正值表示 Full-fit 更好。并报告：

\[
d_H=\frac{\Delta_H}{q_H(E)},\qquad
d_{\mathrm{KL}}=\frac{\Delta_{\mathrm{KL}}}{\mathrm{KL}(0)}.
\]

所有补偿质量比较均使用这把共同裁判尺子，不能比较各方法按自己的 metric 得出的自评分。

本轮 gamma 是“相对 None 的补偿恢复率”。Exp-2 的 gamma 是“A64 相对 SVD64 的优势除以 None 损伤”，二者不是同一个量，报告不得混名。

### 8.4 Metric 的收益预测准确度

保持拟合完成的 K 冻结；不在 D_eval 上重新计算 marginal 因子或重新拟合因子。对同一候选集合：

\[
\mathcal C=\{C_{\mathrm{svd}},C_A,C_{\mathrm{marg}},C_{\mathrm{fit}}\},
\]

交叉评价两个 metric：

\[
\gamma_K(C)=1-\frac{q_K(E-C)}{q_K(E)},
\quad e_K(C)=\gamma_K(C)-\gamma_H(C).
\]

每个 K 都评价全部四个候选，不能只评价自己生成的补偿。分别报告 raw 和 solve metric 的有符号误差、绝对误差，以及候选集合上等权 MAE；solve metric 为补偿生成相关的主要版本，raw metric 用于解释拟合与阻尼影响。

候选集合包含方法自身生成的解，因此 MAE 只描述该有限集合，不外推为整个 rank-64 可行空间的统一误差保证。

gamma 对 K 的整体正比例缩放不变。原始 q_K 数值仍需按正确归一化报告，不允许在评价集上事后拟合比例系数美化绝对误差。

## 9. 不确定性与结果判定

### 9.1 不确定性范围

- D_eval 固定 8 个窗口，每窗口 K_eval=64；使用相同标签计算所有候选。
- 原始配对差沿用窗口内样本方差的 SE：`SE² = Σ_c s_c² / (N² K_eval)`。
- 比值、恢复率、MAE 差使用 2000 次窗口内分层配对 bootstrap；所有候选和两个 metric 共用同一重采样索引，seed 在 manifest 中冻结。
- 不重采样窗口，不把 token、候选或模块当作额外独立样本。
- 上述区间只描述固定评价窗口、固定已生成补偿下的标签采样不确定性，不包含拟合数据、初始化或未知文本分布的不确定性，也不是多重比较同时覆盖区间。
- actual KL 在固定窗口和固定模型下不依赖 teacher 标签采样，不能套用 q_H 的 MC 区间。报告逐窗口与总体 KL，并通过数值重放判断接近数值噪声的差异。

### 9.2 预先冻结实际关注阈值

正式评价前，必须明确：

- `tau_H`：完整曲率归一化改善 d_H 的实际关注阈值。
- `tau_KL`：actual KL 归一化改善 d_KL 的实际关注阈值。
- `tau_metric`：两个 metric 的收益预测 MAE 改善阈值。
- 数值复现、分母下限、PSD 判定和 SVD 求解验收容差。

这些阈值尚未在讨论中确定，本文不伪装为已有共识；不能沿用 Exp-2 的结构误差阈值而不解释其是否适用于新任务，也不能看完正式效果再设置。

### 9.3 判定规则

对 d_H 的区间 `[l,u]`：

- `l > tau_H`：明确且达到实际关注程度的改善。
- `u < -tau_H`：明确且达到实际关注程度的退化。
- 区间完全落入 `[-tau_H, tau_H]`：在预算内差异较小。
- 其他情况：当前预算下未确定；包括已支持正向变化、但是否达到实际阈值仍未确定的情形。

actual KL 根据冻结阈值与数值稳定性分为改善、退化或差异较小/数值未确定，不赋予虚构的标签采样显著性。

| 观察结果 | 可报告结论 |
|---|---|
| q_H 与实际 KL 均达到预设改善标准 | 该模块上 Full-fit 构造产生了更好的 rank-64 补偿 |
| 补偿改善，metric MAE 未明确改善 | 补偿质量改善；不能同时宣布收益判断整体更准 |
| metric MAE 改善，补偿未改善 | 有限候选集上的预测更准，尚未转化为更好补偿 |
| q_H 改善但 KL 退化 | 二次目标收益未转化为实际 functional damage 收益 |
| 差异区间跨判定边界 | 当前预算下不确定，不记为方法失败 |
| 数值、数据身份或拟合求解验收失败 | 实现/求解问题，暂不作科学效果判断 |
| 验收通过但本轮未改善或明确退化 | 本轮 full-Fisher Frobenius 拟合构造未改善 QER；不否定整个 single-Kronecker 家族 |

两个模块分别判定。只在 L10 改善时只能声称 L10 上改善；不能隐去 L31 的退化或不确定结果。

## 10. 数学与实现验收

### 10.1 小尺寸显式核验

正式模型试跑前，构造可显式形成 H 的小矩阵例子，验证：

1. vec 方向与 Kronecker 因子次序一致。
2. 显式二次型与 `E[<S,R>²]/(2T)` 相等。
3. `SAS^T`、`S^TGS` 收缩及块更新与显式目标一致。
4. 精确块更新在容差内不增加同一经验 Frobenius 目标。
5. 保持乘积的 gauge 归一化不改变目标或评分。
6. marginal 评分与 Exp-2 sep 的 L/T 约定一致。
7. weighted-SVD 目标值等于变换后矩阵的截断尾能量；rank、回变换和部署残差一致。
8. 相对阻尼规则在互逆因子重标度下保持 solve metric 不变。
9. 对纯整体缩放 K→sK，相对阻尼后的 K_solve 只乘 s，评分乘 s、恢复率不变；在 rank 边界非退化的小例子上，补偿 C 在数值容差内一致。存在奇异值并列时核验最优目标与解集合的性质，不强求求解器输出同一个基。
10. canonical marginal 的 `(L/T) A_marg⊗G_marg` 与统一接口 `A_raw⊗G_raw` 完全等价；Full-fit 输出不重复吸收 L/T。专门检查隐含绝对 epsilon/floor 是否破坏尺度等变性。

### 10.2 真实模型 pilot

只使用拟合侧数据执行小预算 pilot，核验：

- S 与目标 Linear 的共享权重 NLL 梯度一致。
- 每个窗口的输入只计入 A_marg 一次，梯度样本计数正确。
- 两条等价收缩路径在真实维度上通过数值检查。
- 精度、缓存与流式策略的误差可接受。
- 测量实际墙钟、GPU 峰值、进程及作业 cgroup 主机内存、缓存磁盘量；不能将 Exp-2 的成本直接当作 Exp-3 成本。

S 的大小为 `d_out × d_in`；保存所有 replicate 的 S 可能占大量空间。正式选择缓存 S、流式磁盘读取或重放反传时，必须记录成本与精度。不得显式构造真实模型中 `(d_out*d_in)²` 大小的 H，也不得未经声明加入额外统计近似。

### 10.3 正式评价前冻结

保存因子、补偿、数据清单和代码哈希，再执行 D_eval 评价。对旧基线的 full 投影进行父记录重放核验；核验失败先排查身份和数值路径，不继续解释新方法效果。

## 11. 执行顺序与产物

### 11.1 执行顺序

1. 核验父模型、Wq、冻结补偿、8 个评价窗口和标签的身份。
2. 选定并登记新拟合窗口，检查与评价数据不重合。
3. 完成小尺寸数学核验和拟合侧真实模型 pilot。
4. 根据成本冻结第 12 节 manifest；正式运行不按效果加样本或改阈值。
5. 在相同拟合样本上计算 marginal 统计与 Full-fit 统计。
6. 拟合并记录 raw 因子；按共同规则生成 solve 因子。
7. 生成两份新 rank-64 补偿，通过数值验收并冻结。
8. 在 D_eval 上评价 None 和四个补偿的 q_H、actual KL，并交叉评价两个 metric。
9. 复算配对汇总、不确定性、判定及资源成本，形成报告。

本文交付只包含方案。服务器命令、资源租约、代码部署和正式运行属于后续实施阶段。

### 11.2 建议产物结构

```text
qer_teacher_kl_exp03/<run_id>/
  protocol.md
  manifest.json
  identity.json
  data/fit_windows.json
  data/eval_windows.json
  data/fit_sample_manifest.json
  factors/<module>/marginal_raw.*
  factors/<module>/full_fit_raw.*
  factors/<module>/*_solve.*
  fit_history.csv
  numerical_checks.json
  corrections/<module>/*
  eval/full_scores.csv
  eval/kl_by_window.csv
  eval/metric_cross_scores.csv
  summary/correction_quality.csv
  summary/metric_prediction_error.csv
  summary/paired_comparisons.csv
  resource_usage.json
  verification.json
  SHA256SUMS
  RESULTS.md
```

大统计缓存可单独存放，但需明确文件身份和复现路径。不要求将所有大张量复制回本地；不能因为清理缓存而失去重建统计所需的数据、标签、代码和配置身份。

## 12. 正式运行前的参数冻结表

标记“待冻结”表示当前设计尚未确定具体执行值，不是实验中可自由调整的参数。填完并版本化后才进入正式采集和评价。

| 项目 | 当前决定或待办 |
|---|---|
| 两个模块、原量化误差、rank | 已定：第 3 节；rank=64 |
| 比较组 | 已定：None + 四个补偿组 |
| D_eval | 已定：原 8 个 WT2 validation 窗口；须核验哈希 |
| K_eval | 已定：每窗口原 64 份标签，全候选配对 |
| 拟合语料和窗口来源 | 待冻结：建议 WT2 train，登记确切版本、位置及不重合检查 |
| N_fit、K_fit | 待 pilot 成本测量后冻结；两 AG 组完全相同 |
| 拟合采样 seed 和选择顺序 | 待冻结并保存逐样本 manifest |
| 是否设置 D_dev | 待冻结；若选参则只能使用拟合侧开发数据 |
| ALS 初始化、选择规则 | 待冻结；建议 marginal 初始化，拟合侧单位阵初始化作稳定性检查 |
| 迭代上限、收敛阈值 | 待 pilot 后冻结，只据拟合侧数值收敛与成本确定 |
| 统计精度和缓存策略 | 待 pilot 冻结；优先高精度累计，改变精度必须核验 |
| weighted-SVD 实现与精度 | 待冻结；两 AG 组共用，声明精确或近似求解 |
| gauge | 已定：A 的 Frobenius 范数为 1，G 同步补偿 |
| canonical metric 系数 | 已定：H 含 1/T；K_marg=(L/T) A_marg⊗G_marg；marginal 的 L/T 在统一接口中吸收到 G_raw |
| 处理顺序与尺度验收 | 已定：canonical 系数→保乘积 gauge→相对阻尼→SVD；核验 gauge 与整体缩放两种情形 |
| eta_A、eta_G 及条件数规则 | 待冻结；两 AG 组共享同一相对阻尼/选择规则 |
| tau_H、tau_KL、tau_metric | 待冻结；不据正式评价结果选择 |
| 数值、PSD、分母下限容差 | 待冻结，依据数学测试和拟合侧 pilot |
| bootstrap | 已定：2000 次窗口内配对重采样；seed 待写入 manifest |
| 资源与存储预算 | 待 pilot 测量后冻结；不直接沿用 Exp-2 成本估计 |

## 13. 可成立的结论与禁止越界的解释

若 Full-fit 在某模块上改善 q_H 和实际 KL，可表述为：

> 在当前数据、量化误差和 rank-64 条件下，直接利用完整曲率拟合的 single-Kronecker metric 比 marginal 构造生成了更好的补偿，说明该结构仍能承载对本任务有用、而 marginal 构造未充分利用的信息。

若收益预测也改善，可另外表述为“在本轮共同候选集合上更准确地预测了补偿收益”。不能将有限集合结论外推到全部可行补偿。

若未改善，只能表述为：

> 本轮 full-Fisher Frobenius 拟合、采样预算、数值处理与求解设置下，没有获得明确的 QER 改善，或观察到了退化。

不能据此否定整个 single-Kronecker 家族。可能原因包括拟合准则与补偿收益不一致、有限样本、求解质量、阻尼以及结构限制；本轮不自动区分这些原因。

若 A_s 只是另一固定输入侧因子，A_s⊗G 仍是 single-Kronecker 构造，不代表表达结构扩大。多项 Kronecker 之和或直接迭代补偿留待后续单独设计。

## 附录 A：理论桥梁与本轮不作的保证

对固定残差 R：

\[
|q_H(R)-q_K(R)|\le\frac12\|H-K\|_2\|R\|_F^2
\le\frac12\|H-K\|_F\|R\|_F^2.
\]

因此全局拟合具有控制二次型误差上界的依据，但不保证每个具体方向或最终补偿质量单调改善。

若真实最优 C_star 存在，C_hat 精确最小化 q_K，且在两者对应残差上均有绝对误差不超过 delta，则真实目标 regret 不超过 2 delta。更直接地：

\[
q_H(E-C_{\mathrm{hat}})-q_H(E-C_{\mathrm{star}})
\le\frac{\|H-K\|_2}{2}
\left(\|E-C_{\mathrm{hat}}\|_F^2+\|E-C_{\mathrm{star}}\|_F^2\right).
\]

本轮不能计算真实最优解，因此不报告测得的全局 regret 或已证近最优。

仅有 rank 约束的补偿集合无界，不能把未加范数限制的可行集 supremum 当成自动有限的误差目标；加入约束后也不能未经证明沿用原 weighted-SVD 闭式解。

普通的曲率接近不直接保证与一般 H 最优补偿的子空间接近。B_K 的谱间隙可作为其自身截断的稳定性诊断，不作为接近真实最优补偿空间的证据。

## 附录 B：前序资产与方法依据

本地资料：

- [Exp-1 v2 协议](qer_teacher_kl_exp01_local_protocol_20260918_v2.md)
- [Exp-2 结构消融协议](qer_teacher_kl_exp02_structure_ablation_20260918.md)
- [Exp-2 结果与核验](../outputs/qer_teacher_kl_exp02_20260918/RESULTS.md)
- [幅度扩展协议](../outputs/qer_teacher_kl_alpha_extension_20260918/results/protocol.md)

父资产位置按 Exp-2 记录，实施时重新核验，不以路径字符串代替身份验证：

```text
teacher: /data1/cck/models/Llama-3.1-8B/ms-snapshot-20260916
Exp-1: /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp01_v2/20260918_r1
alpha extension: /data2/cck/KFAC-QERA/runs/qer_teacher_kl_alpha_extension/20260918_r1
Exp-2: /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp02/20260918_r1
```

方法依据：

- Van Loan 与 Pitsianis，[Approximation with Kronecker Products](https://users.cs.duke.edu/~nikos/reprints/C-001-KronApprox.pdf)：最近 Kronecker 乘积逼近。
- Koroko 等，[Efficient Approximations of the Fisher Matrix in Neural Networks using Kronecker Product Singular Value Decomposition](https://arxiv.org/abs/2201.10285)：直接拟合 Fisher 的 Kronecker-SVD 方法。
- Martens 与 Grosse，[Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://proceedings.mlr.press/v37/martens15.html)：K-FAC 方法背景。

上述文献支持方法来源，不提供本轮 QER 补偿一定改善的保证，也不构成本轮新颖性的证明。
