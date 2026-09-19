# QER 实验二：位置交叉项与输入—敏感性联合统计的消融

版本：v1.0，2026-09-18。状态：**完整实验方案，尚未实现或执行本实验**。

## 1. 本轮只回答什么

实验一已验证 teacher 采样曲率与小扰动 KL 对齐；幅度扩展进一步显示，在当前两个模块、六个残差方向上，完整幅度 alpha=1 的总体二次预测偏差不超过约 1.53%，A64 与 SVD64 的排序保持。

因此，本轮固定这些方向，依次检查：

1. 去掉共享权重的模块位置交叉项，会丢失多少方向性曲率信息？
2. 再将输入与敏感性的联合二阶统计替换为边缘统计的乘积，会额外丢失多少信息？
3. 这些变化是否影响同 rank 补偿收益的判断，还是主要表现为共同缩放？

**本轮是结构诊断，不是新补偿方法比赛。** 不生成新 C，不改变 rank，不重新量化，不选择 A_s 或 sum-of-Kronecker，不跑全模型 PPL、下游任务或新的 alpha 曲线。结论只决定后续优先研究哪类结构。

## 2. 固定设置与资产

| 项目 | 固定设置 |
|---|---|
| 模型 | 原实验 Llama-3.1-8B Base teacher；核对完整身份 |
| 模块 | `model.layers.31.self_attn.q_proj`、`model.layers.10.self_attn.v_proj` |
| 量化与补偿 | 原 MXINT3 Wq；原普通 SVD64、A64 补偿；不重新求解 |
| 方向 | 原冻结 FP32 `R_none`、`R_svd64`、`R_A64`，每模块三份 |
| 数据 | 原保存的 WikiText-2 validation 八个窗口，不重新分词或挑选 |
| 长度与计数 | 每窗口 L=2048 个模块位置，T=2047 个有效预测位置；N=8 个窗口 |
| 标签 | 复用原 `samples/wXX_kYYY.safetensors`，每窗口 K=64，共 512 份标签 |
| 采样预算 | 固定重放全部 K=64；K=4 做 pilot，K=16 可出进度，但不据结果提前停止 |
| 数值路径 | 原 FP32/eager；关闭 TF32、autocast、KV cache；同一 sampled-NLL seed 与 checkpoint 路径 |
| 新增统计 | 同八窗口的 A_diag、三个方向评分及配对标量 |
| 执行平台 | 本地 A6000，优先一张；沿用已验证的调度与资源释放方式 |

所有期望均针对**这八个固定窗口的经验分布及 teacher 采样标签**。本轮不估计对未知语料的泛化误差。

父运行：

```text
/data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp01_v2/20260918_r1
```

父 identity：`1cf6aaa613ec828a857e279a33878f550dce4d340ad04053ce9b59b960d15b28`。

幅度扩展运行：

```text
/data2/cck/KFAC-QERA/runs/qer_teacher_kl_alpha_extension/20260918_r1
```

扩展 identity：`fd5032f4334bcd16217e7c7b1ecb50ce5389475d3b405aaa30791f2ef6c07fa7`。

模型：`/data1/cck/models/Llama-3.1-8B/ms-snapshot-20260916`。

可复用的冻结父源码：`/home/cck/projects/qer_teacher_kl_exp01_v2_r1`；本地源码对应 `qera_mxint4_full_ag/experiments/qer_teacher_kl_exp01/`。

本轮新建代码与结果目录，记录父身份和实际源码哈希。旧源码、父输入、标签、方向与结果只读；不覆盖原 A、因子或判定。

## 3. 三种评分的精确定义

### 3.1 固定梯度定义

对窗口 c、replicate k，直接读取保存的 teacher 标签，构造求和 NLL：

$$
\ell_c^{(k)}=-\sum_{j\in\mathcal J_c}\log p_j(y_j^{(k)}),\qquad
g_{ct}^{(k)}=\frac{\partial\ell_c^{(k)}}{\partial h_{ct}}.
$$

预测标签在原 teacher 下独立采样；不把标签送回上下文。不使用真实标签 CE，不对 teacher 自身 KL 的零梯度求外积，不改成 mean-loss。

定义 a_ctk(R)=g_ct^(k)ᵀ R x_ct。j 是预测位置，t 是被干预模块的位置；g_ct 包含全部有效预测通过下游网络对 h_ct 的影响。

### 3.2 full：保留全部模块位置交叉项

$$
q_{\rm full}(R)=\frac1{2T}\mathbb E_{c,y}
\left[\left(\sum_{t=1}^{L}g_{ct}^{\top}Rx_{ct}\right)^2\right].
$$

### 3.3 pos：只删除模块位置交叉项

$$
q_{\rm pos}(R)=\frac1{2T}\mathbb E_{c,y}
\sum_{t=1}^{L}(g_{ct}^{\top}Rx_{ct})^2.
$$

二者之差为

$$
q_{\rm full}(R)-q_{\rm pos}(R)
=\frac1T\mathbb E_{c,y}\sum_{t<u}a_{ct}(R)a_{cu}(R).
$$

交叉项可以为正或负。q_pos 可能高估，也可能低估 full；它不是对 full 的一般上界或下界。预测标签独立不能消除模块位置交叉项。

### 3.4 sep：将位置内联合统计替换为池化边缘统计的乘积

定义

$$
A_{\rm diag}=\frac1L\mathbb E_c\sum_t x_{ct}x_{ct}^{\top},\qquad
G_{\rm diag}=\frac1L\mathbb E_{c,y}\sum_t g_{ct}g_{ct}^{\top}.
$$

本轮第三项固定为

$$
\boxed{q_{\rm sep}(R)=\frac{L}{2T}
\operatorname{tr}(G_{\rm diag}R A_{\rm diag}R^{\top}).}
$$

L/T 必须保留；不再用比例号或事后迹归一化对齐数值。若代码吸收系数，须证明与上述定义相等并保存原始计数。

此处 `diag` 指去掉模块位置的交叉块，**A_diag、G_diag 在通道维度上仍是完整矩阵**，不取矩阵对角线。

第二步删除的是混合二阶量的联合依赖：E[(xxᵀ)⊗(ggᵀ)] → E[xxᵀ]⊗E[ggᵀ]。池化同时涉及窗口和位置差异；本轮不能再把失配归因到具体某一类上下文。普通 E[xgᵀ] 或 Pearson 相关性不能替代这个检验。

## 4. A_diag 必须单独收集，但不改变已有 A64

$$
\widehat A_{\rm diag}=\frac1{NL}\sum_{c=1}^{N}\sum_{t=1}^{L}x_{ct}x_{ct}^{\top}.
$$

- 输入来自原 teacher 在八个诊断窗口上的目标 Linear 输入。每模块计数应为 N×L=16384。
- 使用全部 2048 个位置，包括可能对有效预测无贡献、梯度为零的位置；不另删末位置。
- 每个输入只进入 A_diag 一次；K 次反传不能导致重复累计。
- 收集未中心化二阶矩，采用分块 FP64 Gram 和 FP64 累加。仅八个窗口，优先保持数值定义直接明确。
- 原始 sum、count、A_diag、对称化误差和输入哈希分开保存。仅允许数值对称化，不加阻尼、不做 floor、裁剪或迹归一化。
- **不用原 SlimPajama A 替代本轮 A_diag。** 原 SlimPajama A 继续作为被冻结 A64 方向的来源，保持不变。

这里不要求 Cholesky 或求逆，因此 A_diag 即便秩不足，也不构成需要正则化的理由。

## 5. 一次反传同时得到三个评分

对每个 `(module, c, k)`，使用同一个 x、g 和三份冻结 R。每个方向记录：

$$
d_{ck}(R)=\sum_t a_{ctk}(R),\qquad
b_{\rm full,ck}(R)=\frac{d_{ck}(R)^2}{2T},
$$

$$
b_{\rm pos,ck}(R)=\frac1{2T}\sum_t a_{ctk}(R)^2.
$$

sep 不必显式永久收集 G。先在 FP64 中为每个方向构造

$$
M_R=R\widehat A_{\rm diag}R^{\top},\qquad
b_{\rm sep,ck}(R)=\frac1{2T}\sum_t(g_{ct}^{(k)})^{\top}M_Rg_{ct}^{(k)}.
$$

随后统一归约：

$$
\widehat q_m(R)=\frac1N\sum_c\frac1K\sum_k b_{m,ck}(R),
\qquad m\in\{\mathrm{full,pos,sep}\}.
$$

该 sep 与将经验 A_diag、G_diag 代入第 3 节公式严格对应。它对梯度位置使用的 A 是**全八窗口的池化 A_diag**，不是逐窗口 A_c，也不随 replicate 改变。

实现细节：

- 沿用父实验 FP64 `x @ R.T` 和 FP64 最终投影；a_t 数组只需短暂保留。
- M_R、sep 收缩与所有标量归约用 FP64。不得为了省时偷偷再引入通道对角化、低秩截断或随机 trace 估计。
- 可直接计算 g M_R 后点积；也可每个 replicate 临时计算 Γ_ck=Σ_t g_tg_tᵀ，再对三个 M_R 做 `tr(M_R Γ_ck)`。两条实现均须数值核验，pilot 后选择并冻结；Γ 不长期保存。
- 保存每次的 full、pos、sep 和有符号 d；无需保存全部 activation、gradient 或每个 replicate 的大 Gram。
- 为避免重复做 R x，允许在同一窗口内缓存三份确定的 FP64 R x，前提是重放输入核验通过。不能缓存跨窗口的错误输入。
- 记录实际矩阵乘法时间和内存；没有额外反传不等于没有额外计算。

预期正式记录数为 2×8×64=1024 个原子单元，每单元包含三个方向的九个评分，展开后 3072 行方向记录。原 512 份标签跨模块复用，不重新抽样。

## 6. 必须记录的比较：分数、差值和共同缩放

### 6.1 三个有符号误差

每个方向、每条配对记录计算：

$$
e_{\rm position}=b_{\rm pos}-b_{\rm full},\qquad
e_{\rm dependence}=b_{\rm sep}-b_{\rm pos},\qquad
e_{\rm total}=b_{\rm sep}-b_{\rm full}.
$$

核验 e_total=e_position+e_dependence。三项均报告均值、SE、区间及相对原 q0(R) 的比例。q0 是父实验冻结的小幅度 `q_KL`，不在本轮拟合。

若两步差异显著、符号相反且总误差小，标记误差抵消；不能只看 sep≈full 就说两步分别合理。

### 6.2 同 rank 补偿收益

对每种评分定义

$$
\Delta_m=\widehat q_m(R_{\rm svd64})-\widehat q_m(R_{A64}).
$$

正值预测 A64 更好，负值预测 SVD64 更好。构造每条记录的收益 u_m,ck=b_m,ck(R_svd64)−b_m,ck(R_A64)，对 u_pos−u_full、u_sep−u_pos、u_sep−u_full 进行配对误差分析。

同时只读列出两份独立于本轮结构选择的现有参照：

- 局部 KL 差值 Δ0=q0(R_svd64)−q0(R_A64)；
- 幅度扩展 alpha=1 的实际 KL 差值 Δ1。

MC 结构误差主要对照同标签 full，以便消除共同采样波动；原 KL 用于功能解释，不用于拟合 sep 的尺度。

### 6.3 判断是否只是共同缩放

固定每模块内部的未补偿方向为参照：

$$
\nu_m(R)=\frac{\widehat q_m(R)}{\widehat q_m(R_{\rm none})},\qquad
\gamma_m=\frac{\Delta_m}{\widehat q_m(R_{\rm none})}.
$$

nu 比较相对损伤，gamma 比较以未补偿损伤为单位的收益。共同正倍数不会改变它们，也不会改变固定模块、固定 rank 的最优补偿。

先报告原始尺度，再报告 nu、gamma 和各步骤的差异。不能给每个方向单独拟合系数。原始误差大但相对量接近，只能说“与共同缩放相容”，不能根据三个方向证明整个可行空间中都成立。

## 7. 配对不确定性与预先固定的判读规则

### 7.1 评分与差值的标准误差

对任意每条记录的标量 v_ck（可以是 b、e、u 或收益误差），计算

$$
\widehat\mu_v=\frac1N\sum_c\overline v_c,\qquad
\widehat{\mathrm{SE}}(v)=\sqrt{\frac1{N^2}\sum_c\frac{s_c^2(v)}K}.
$$

报告描述性的 `mean ± 1.96 SE`。差值必须先在同 c、k 内相减再算方差，不能把两个评分当作独立估计器。不可将 2048 个位置或三个方向当成独立 MC 重复。

固定 N=8、K=64；不因差异不显著而无限追加 K。所有区间仅描述固定窗口上的标签采样误差，不是泛化区间，也不是多重比较的同时覆盖保证。

### 7.2 比值与归一化收益

对 nu、gamma 及其步骤差异，使用固定 2000 次分层配对 bootstrap，seed=`2026091802`：每个窗口内部重采样 64 个 replicate，下标同时作用于所有评分及所有方向；不重采样窗口，不按 token 重采样。

每次重算比值，取 2.5%/97.5% 分位数。若分母不为正或接近数值零，标为 `RATIO_UNRESOLVED`，不要删掉异常重采样后强行给区间。这里的区间同样仅作有限样本描述。

### 7.3 工程容差与三态判定

本轮在查看结果前固定以下容差，仅用于组织报告，不是理论界或全局方法优劣标准：

| 比较量 | “小误差”带 |
|---|---|
| 各步骤的原始评分误差 | ±10%×该方向原 q0(R) |
| 各步骤的补偿收益误差 | ±20%×原局部 KL 收益 Δ0 |
| 各步骤归一化收益 gamma 的差异 | ±0.05，即未补偿损伤的 5 个百分点 |

原两模块 Δ0 均为正；执行时核验，不静默替换尺度。对任一误差区间：

- 完全位于误差带内：`SMALL_WITHIN_BUDGET`。
- 完全位于误差带某一侧之外：`MATERIAL_DIFFERENCE`。
- 其余情况：`UNRESOLVED_AT_K64`，不能把“未检出差异”说成近似足够。

排序另行报告：Δ_m 区间全正为 `A64_PREFERRED`，全负为 `SVD64_PREFERRED`，跨零为 `RANKING_UNRESOLVED`。若点估计反转但区间跨零，只称“点估计反转且不确定”，不宣布可靠反转。

各窗口的均值、方向误差和排序全部保存。八窗口只作分解展示，不把窗口/幅度重复比较累计成大量独立成功样本。

## 8. 实施顺序与数值验收

### 阶段 A：资产核验和必要的数学检查

读取实际仓库约束，只读核验 teacher、输入、512 份标签、六份方向及父记录。原投影不能恢复 pos/sep，标签存在也不能免除反传；本轮明确需要重放。

用 FP64 小矩阵测试以下恒等式：

1. full/pos 的差值等于显式交叉项，构造正、负交叉项两个例子。
2. 显式经验 A、G 的 Kronecker/trace 评分与 M_R 收缩一致，且正确保留 L/T。
3. 若显式展开输入与梯度的所有独立配对，得到相同的池化 sep；不误实现为仅窗口内打乱或固定位置打乱。
4. e_total 的分解、配对 SE、共同倍数下 nu/gamma 不变；bootstrap 保留配对。
5. rank 反转、不确定区间、零分母及重复记录拒绝的分析逻辑。

toy 相对误差基准≤1e-10；零参照使用绝对误差并记录尺度。旧数学/模型测试可复用，不为本轮重复大模型 KL 曲线。

### 阶段 B：收集诊断 A 与构造 M_R

八个 teacher 前向可同时收集两个模块的输入。保存 FP64 sum/count/A_diag，确认每模块 16384 个位置、无重复累计。对同一小批 x 和固定 R 核对直接输出 SSE 与 trace 目标。

逐模块构造三份 FP64 M_R，记录对称误差和构造耗时。不得调用父实验的阻尼选择器或低秩求解器。只发生数值异常时停止修实现，不能通过加阻尼或裁剪掩盖。

### 阶段 C：真实梯度 pilot

两个模块分别重放窗口 0 的原 k=0..3，得到三方向的全部评分：

- teacher hidden、目标输入与父路径一致；保持已验证的 sampled seed 与 checkpoint 逻辑。
- 对每模块、每方向，将四个带符号 d 组成向量，与父记录比较；相对 L2 差异≤1e-5。另存逐项绝对误差，避免对接近零的单个 d 使用不稳定相对误差。
- FP64 直接 `g M_R` 与临时 Γ 路径至少对一个真实 replicate、三个方向交叉核验，评分相对差异≤1e-8。非有限值或明显负的 PSD 二次评分判为数值失败，不截零掩盖。
- 数值合法的四次 pilot 记录可以进入正式 k=0..3，避免重复累计。
- 记录各评分收缩、反传、显存/RSS 和时间，确定正式实现。

### 阶段 D：正式重放与分析

每模块每窗口重放全部原 k=0..63。按原子单元提交九个评分，进度 K=16 可展示但最终统一 K=64。

结束后每模块每方向用完整 512 个 d 的向量再核对父记录，相对 L2 差异≤1e-5；总体 full 与父 q_hat 相对差异≤1e-5。若失败，本轮结构差异暂不解释，先定位路径变化。

复算归约、误差分解、配对 SE、bootstrap 和判定；保存所有失败方向。采集完成与“近似足够”是两个不同状态，负面或不确定结果也属于完整实验结果。

## 9. 资源与成本控制

原运行单张 A6000、FP32 2048 上下文已验证可行，GPU allocated 峰值约 33.76GiB；1024 次采样反传及原投影累计约 66.5 分钟。本轮反传次数不变，但增加 sep 收缩和诊断 A 准备，**总时长必须由 pilot 外推，不承诺等于原耗时**。

先按既有方式申请一张 A6000 和 64GiB 主机内存，不自动租卡或占用其他作业设备。按模块运行、冻结全部参数，仅目标输出与下游启用梯度，使用原已验证的后续层重计算。

若输入/输出维度为 4096，一份 FP64 方阵约 128MiB；实际按张量形状计算 A_diag、三个 M_R、可选临时 Γ 和工作区需求。按模块释放，必要时仅保留当前需要的 M_R，不缓存所有梯度。

比较 `g M_R` 与临时 Γ 的实测开销，允许选择经验证等价的实现；不在正式运行中混用不同精度或偷偷增加随机估计。GPU 使用时段结束时保存为 `PAUSED`，按既有调度方式释放资源，之后续跑。

## 10. 结果应该怎样解释

| 观察 | 允许的结论 |
|---|---|
| 两步在原始评分和相对收益上都小 | 当前方向上没有观察到需要恢复这些信息的证据；不证明所有候选都足够 |
| full→pos 明显失配，pos→sep 较小 | 优先研究模块位置交叉项；还不能断言必须多个 Kronecker 项 |
| full→pos 较小，pos→sep 明显失配 | 优先研究池化后的输入—敏感性联合依赖 |
| 两步都大、符号相反、总误差小 | 近似之间存在抵消，不能把每一步分别视为合理 |
| 原始评分差异大、归一化收益接近 | 与共同缩放相容，固定 rank 的选择损失可能较小 |
| 归一化收益失配或可靠排序反转 | 已测候选上的补偿选择信息受到损害 |
| 区间太宽 | 当前预算不能判断，不自动升级为更复杂算法 |

共同报告数值与区间，不仅给状态标签。尤其不能由 sep 失败推出“单个 Kronecker 家族不够”：普通边缘乘积不是该家族的最优拟合；重新估计因子或 A_s 可能仍属于单个 Kronecker 结构。

同样，sum-of-Kronecker 通常不保留一次截断 SVD 的闭式解。若后续考虑它，需要另行讨论求解与预算。本轮不给出该算法选择。

本轮每模块仅两种 rank64 候选加一个无补偿控制。即使评分准确，也没有证明曲率对整个低秩可行集准确；后续新方法产生的补偿需要独立验证。

## 11. 保存要求与最小报告

建议新代码：`qera_mxint4_full_ag/experiments/qer_teacher_kl_exp02/`。

远端结果：`/data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp02/<run_id>/`。

本地小型结果：`outputs/qer_teacher_kl_exp02_20260918/`。

```text
protocol.md, config.resolved.yaml, identity.json, environment.json
parent_verification.json, numerical_audit.json, pilot.json, status.json
diagnostic_A/                # FP64 sum/count/A_diag、输入身份
direction_metrics/           # M_R 或可核验重建信息、哈希和精度
records/                    # 原子保存 module/window/replicate 的三个方向
scores.csv                  # 3072 行：d、b_full、b_pos、b_sep、三个 e、标签/R/A 哈希
summary.csv                 # 六方向×三评分、误差均值/SE/区间与三态判定
advantages.csv              # 两模块三评分的收益、配对收益误差、排序
normalized.csv              # nu、gamma、配对 bootstrap 区间
windows.csv                 # 每窗口分解与排序
resource_usage.csv, scheduler_receipt.json
verification.json, RESULTS.md, figures/, logs/
```

每条记录还应保存 L、T、loss reduction、位置归约定义、数值实现版本。一个原子单元包含同次反传的全部方向，缺一不可；续跑按 ID 跳过完整单元，不重复计数。

报告保持集中：

1. 六方向 full/pos/sep 及原 q0、alpha=1 KL 的对照表。
2. 两步误差和总误差的有符号图，显示配对区间，标出可能的抵消。
3. 每模块同 rank 收益、归一化收益、排序和逐窗口表现。
4. 数值复现、有效记录数、实测成本、失败/不确定项及结论边界。

标准库或独立分析脚本应从原始标量重算均值、配对 SE、恒等式和计数。bootstrap 可用固定种子再次复算；不要求重复 GPU 采集。大张量留服务器，小型结果和图同步本地，保存 SHA256 清单。

## 12. 已有理论与本轮贡献边界

输入与梯度二阶统计的乘积分离是已有 K-FAC 思路；共享权重下存在不同近似，不把本轮 sep 等同于所有 K-FAC。参考 [Martens & Grosse, 2015](https://proceedings.mlr.press/v37/martens15.html) 与 [共享权重下的 K-FAC](https://arxiv.org/html/2311.00636v2)。

本轮补充的是：在实际 QER 残差方向上，沿统一 teacher 目标分解结构误差，检验其对有限 rank 补偿收益的影响，并记录有限计算预算下的不确定性。这里不宣称提出了新的 Fisher 恒等式、K-FAC 原理或重构算法。

## 13. 给执行窗口的交接提示词

> 请按本协议实施 QER 实验二。使用已有 SSH/调度方式连接本地服务器，检查可用 A6000、仓库约束与父资产。复用实验一 `20260918_r1` 的 teacher、八个 validation 输入、512 份 teacher 标签和六个冻结方向；幅度扩展结果只读用于解释，不重跑 KL。不重新量化、求解补偿或采集旧 CE-G。先完成小矩阵恒等式与配对分析检查，再从同八窗口收集完整通道的 A_diag（不是矩阵对角线，也不是旧 SlimPajama A）。用同一次反传同时计算 full、pos、sep，正确保留 L/T；正式固定 K=64。sep 可用 FP64 M_R 收缩或等价的临时梯度 Gram，不引入额外结构近似。pilot 与正式 full 投影须复现父记录。按协议报告两步有符号误差、配对收益误差、共同缩放归一化与不确定性，支持原子续跑，释放调度资源并同步小型结果。本轮止于结构诊断；不自动进入 A_s、新补偿、sum-of-Kronecker、rank 分配或全模型评估。遇到真实资产/数值/资源阻塞，保存已完成工作并明确原因。
