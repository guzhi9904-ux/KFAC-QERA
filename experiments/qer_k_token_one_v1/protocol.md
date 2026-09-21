# K Token-joint-one：一次同步更新的 QER 对照实验

版本：v1.0，2026-09-21。平台：双 RTX 4090。

状态：已核对本地Token-joint与Sensitivity-A实现及产物格式；尚未实时验收远端文件，未实现或启动本轮计算。已有K audit、Direct Query、Sensitivity-A实验不重跑。本轮只增加Token-joint-one一个候选。

## 1. 唯一主问题与比较表

> 沿用已有Token-joint的真实初始化，只进行第一轮同步更新，是否比Sensitivity-A进一步降低实际teacher-KL；与已有三轮Token-joint相比，后两轮还带来多少收益？

固定四层K、N256、每窗口一套原标签、rank64及原16个validation窗口。

| 方法 | A | G | 本轮处理 |
|---|---|---|---|
| Marginal | A_M | G_M | 复用 |
| Sensitivity-A | A^(1)的正标量倍 | G_M | 复用 |
| **Token-joint-one** | A^(1) | G^(1) | **唯一新增候选** |
| Token-joint-3 | A^(3) | G^(3) | 复用已有Token-joint |
| Sequence | 已有Sequence-one-step | 已有Sequence-one-step | 复用 |

另复用None（仅量化）作为恢复率分母，不作为新方法。原A-only、Direct Query不加入本轮主表和计算任务；其历史结果保留在原报告中。

不增加单位阵双侧初始化版本、第二轮候选、rank1、ALS、Full-fit、coupled-K、RoPE方法、其他模块或层选择器。不要求预先得到某种性能排序。

## 2. 对研究解释的修订

这个比较成立，但以下表述必须限制在当前数据、rank和已测残差上：

1. 若Joint1优于Sensitivity-A，说明在保持A方向一致的条件下，换成第一轮G有实际收益；不等于证明某类统计信息在所有设置中不可替代。
2. 若Joint1与Token3点估计相近，说明后两轮的实测增益较小；不能仅因差异不显著就宣布等效或普遍不需要后两轮。
3. 若Token3优于Joint1，说明继续更新A/G有收益；不能说后两轮才获取了新的样本信息。所有方法使用同一缓存，后续轮次改变的是同一统计的压缩与重加权。
4. Sensitivity-A→Joint1可以隔离本实现中第一轮G替换的效果；Joint1→Token3同时变化A和G，不能进一步唯一归因于某一侧。

以下所有“<”若用于方法排序，必须注明指恢复率；实际KL越低越好，不能混用方向。正式表格直接报告数值与差值。

## 3. 与现有代码完全一致的数学定义

### 3.1 固定符号与初始化

N=256，L=2048，T=2047，n=4096，m=1024。使用原完整K输入x_{c,u}及原teacher采样sum-NLL输出梯度g_{c,u}，模块位置u遍历0,…,L−1。

\[
A_M=\frac1{NL}\sum_{c,u}x_{c,u}x_{c,u}^{\top},
\qquad
G_M=\frac1{NT}\sum_{c,u}g_{c,u}g_{c,u}^{\top}.
\]

沿用当前Token-joint的实际初始化：

\[
\boxed{A^{(0)}=A_M,\qquad G^{(0)}=I_m.}
\]

不是A^(0)=I_n。A_M必须是原始、未阻尼的输入二阶矩，不使用其逆、矩阵根、gauge后的归一化矩阵或Sensitivity-A替代。

### 3.2 只做一轮同步更新

\[
\boxed{
A^{(1)}=
\frac{\sum_{c,u}\|g_{c,u}\|_2^2x_{c,u}x_{c,u}^{\top}}
{NTm}
}
\]

\[
\boxed{
G^{(1)}=
\frac{\sum_{c,u}(x_{c,u}^{\top}A_Mx_{c,u})g_{c,u}g_{c,u}^{\top}}
{NT\|A_M\|_F^2}
}.
\]

两侧均使用旧A_M和I。不能用新A^(1)计算G^(1)，不能将xᵀA_Mx替换成||x||²，不能进行第二轮。

取

\[
a=A^{(1)}/\|A^{(1)}\|_F,
\qquad g=\|A^{(1)}\|_F G^{(1)},
\]

作为与父实现一致的gauge后的因子，再调用同一阻尼和SVD求解器。gauge不改变A⊗G。

该定义就是原三轮Token-joint在第一轮同步更新并gauge之后停止，不重新定义其初始化或求解问题。

## 4. 直接复用Sensitivity-A的累计量

已核查本地`qer_k_sensitivity_v1/s_statistics.py`，最终原始统计保存：

```text
statistics/<slug(module)>/raw.safetensors
    U, D, A_raw, G_raw
```

其中

\[
U=\sum\|g\|^2xx^\top,\qquad
D=\sum\|g\|^2,\qquad
A_{sens}=U/D.
\]

所以无需重新累计完整A：

\[
\boxed{A^{(1)}=U/(NTm)=\frac{D}{NTm}A_{sens}.}
\]

优先读取U直接形成A^(1)，同时核对A_raw=U/D和D/(NT)=tr(G_M)。保存比例D/(NTm)，避免把表中“同一个A方向”误写成原始tensor数值完全相同。

若仅有完整、身份可信的A_raw与D，可用其恢复等比例A^(1)，记录重构路径和舍入误差；当前源码通常同时保存U，默认不走fallback。

若仅有已阻尼A、矩阵根或补偿权重，不足以替代原始U。若必需产物缺失，报告具体缺失，不自动重跑Sensitivity-A或整个Token-joint。

这里复用的是A及其来源，不是旧补偿的P/Q：G改变后白化残差和SVD都需重新计算。

## 5. 固定设置与数据一致性

| 项目 | 固定值 |
|---|---|
| 模块 | L0/L10/L20/L31 k_proj，四层全部报告 |
| Teacher/Wq | 与KO_RUN、SENS_RUN相同的teacher、tokenizer与MXINT3 Wq；不重新量化 |
| 统计 | 原256个冻结2048-token训练窗口；每窗口原1套teacher标签 |
| 梯度 | 真实K输出完整1024维sum-NLL梯度；不换ground-truth CE |
| rank | 完整1024×4096矩阵上的rank64，不分head |
| 数值 | 缓存FP32，收缩/Gram/根/SVD为FP64；关闭TF32/autocast |
| 阻尼 | 继承父solver的trace-relative规则，预期η_A=η_G=1e−3 |
| 部署 | Wq+float32(C64)，C64在FP64由P64@Q64形成 |
| 评价 | 原16个validation窗口，每次只换一个K，其余权重保持teacher FP32 |
| 新增 | 四个补偿、64个主要实际KL点 |

所有256个窗口按原顺序参与，不做N8/N128性能分支、不重采样标签、不增加训练随机种子。

## 6. 资源复用审计

### 6.1 本次已核查及尚未核查的范围

已从本地源码确认：

- Token-joint从A_M、I初始化，固定三轮同步更新；本轮公式与其第一轮一致。
- Sensitivity-A最终保存U/D/A_raw/G_raw，能够恢复第一轮A。
- Sensitivity-A已有独立校验、求解、逐窗口KL与配对报告实现。
- 原K/O保存真实K梯度；原BASE_RUN保存同层共享QKV输入。

本地没有本轮所需远端完整输出的实时验收结果。源码证明产物格式，不等于证明当前服务器文件完整。执行时须核对manifest、完成标记、receipt与hash。

### 6.2 三个父运行

- BASE_RUN：原`multimodule_three_fp64_v1/run_01`，提供数据、标签和QKV输入。
- KO_RUN：原`ko_increment_v1/run_01`，提供K梯度、A_M、G_M、W0/Wq及既有三方法结果。
- SENS_RUN：已完成`k_sensitivity_v1/run_01`，提供U/D、Sensitivity-A补偿及64条KL。

以上是相对`qera_runs`的默认路径；实际位置由manifest绑定。旧K audit与Direct Query不是本轮依赖，禁止调用其收集流程。

### 6.3 具体复用路径

| 资源 | 位置 | 用途 |
|---|---|---|
| U、D、Sensitivity-A原因子 | SENS_RUN/statistics/<module>/raw.safetensors | 直接恢复A^(1)，不再累计完整A |
| Sensitivity-A来源/完成记录 | 同目录parent_evidence.json、complete.json及运行manifest/complete | 绑定同一N256、X/g来源与G |
| 原A_M、G_M | KO_RUN/modules/<module>/statistics/N256.safetensors，A_m/G_canonical | 旧初始化A_M及一致性检查 |
| 原QKV输入 | BASE_RUN/cache/x/<qkv_input_slug>/wXXXX.safetensors | 计算xᵀA_Mx |
| 原K梯度 | KO_RUN/cache/g/<module>/<sample_id>.safetensors | 累计新G，不重新反向 |
| 原标签与提交记录 | BASE_RUN/cache/labels、两运行cache/commits | 核验梯度身份，不重新算loss |
| W0/Wq | KO_RUN/quantized/<module>.safetensors | 固定残差E及部署 |
| Marginal、Token3、Sequence补偿 | KO_RUN/modules/<module>/corrections/N256__<method>.safetensors | 复验旧评价，不重新拟合 |
| Sensitivity-A补偿 | SENS_RUN/corrections/<module>/sensitivity_weighted.safetensors | 直接复用 |
| 原KO评价 | KO_RUN/scores/validation/wXXXX/<module>___<candidate>.json | Marginal、Token-joint、Sequence-one-step、None，共256条 |
| Sensitivity-A评价 | SENS_RUN/scores/validation/wXXXX/<module>.json | 共64条 |

路径中的module使用既有slug函数。主比较复用的旧KL合计320条：4层×(4个旧补偿方法+None)×16。不是另算一遍历史全部结果。

不得复用Token3最终A/G作为第一轮因子。旧`progress.safetensors`通常覆盖到第三轮；第一轮是否有额外快照只能作为可选交叉核验，不是本轮依赖。

### 6.4 资产验收

生成`parent_assets_audit.json`，记录每项来源identity、文件/张量hash、样本计数及verified/missing/identity_mismatch状态。

重点核验：SENS_RUN与KO_RUN使用同一文本、标签、X/g和Wq；A_M未阻尼；Token3确实是同一初始化的三轮同步版本；每条旧KL与相同16窗口及其candidate freeze绑定。

已有Sensitivity-A逐文件证据可复用以避免重复大范围扫描，但必须检查资产未改变、receipt和提交记录仍匹配。不能只凭历史报告或文件名跳过当前身份检查。

父目录只读，统计新run独立。无需复制40GiB输入/梯度。缺失缓存时不自动重新采集；先报告受影响项，可继续不依赖缺失项的工作。

## 7. 实施步骤

### 7.1 数学pilot，不加载teacher

固定前两个训练窗口，先L20再覆盖四层。使用真实完整A_M：

1. 核对u=diag(X A_M Xᵀ)与逐行xᵀA_Mx的FP64计算一致。
2. 核对gᵀdiag(u)g的矩阵实现与小规模直接外积累加一致。
3. 对前两个窗口调用原`token_step`一次，初始化A_M、I；核对新实现的局部第一轮A/G、分母和同步语义。不得在正式256窗口上重跑三轮作参考。
4. 核对父Sensitivity-A的U/D/A关系及A^(1)正标量关系。

FP64纯代数比较容差1e−10，记录绝对误差及参考范数。若xᵀA_Mx出现显著负值，检查PSD、精度和资产，不通过逐点clamp把公式静默改掉；纯舍入量级异常按父数值规则记录处理。

pilot仅临时重算两个窗口的A用于交叉核验；正式A仍直接复用SENS_RUN的U。pilot生成的正式G累计只计数一次。

### 7.2 一遍缓存扫描，只新累计G

每层读取固定原A_M，令V=0。按窗口执行：

```text
X = cached_x.reshape(L, n).float64()
g = cached_k_gradient.reshape(L, m).float64()
u = row_sum((X @ A_M) * X)
V += g.T @ (g * u[:, None])
```

最终

\[
G^{(1)}=V/(NT\|A_M\|_F^2).
\]

不得将u做每窗口归一化、clipping、开方后当新权重，或按head分开归一化。所有L位置均计入原始计数，梯度为零的位置自然不贡献V。

同一遍扫描顺手累计Σu、Σu²、Σu||g||²和极值。可验证：

\[
\sum_{c,t}u_{c,t}\approx NL\|A_M\|_F^2,
\]

\[
\operatorname{tr}(G^{(1)})\approx
\frac{\sum_{c,t}u_{c,t}\|g_{c,t}\|^2}{NT\|A_M\|_F^2}.
\]

第一式依赖A_M正是同一N256的原始输入二阶矩；检查容差按FP64累计误差设置并与父规范一致。两式只核验实现，不构成新研究任务。

两个GPU各一个离线worker。按模块原子保存当前与上一份V累计检查点、窗口列表、统计量和父资产证据，不永久保存256份G。先不加载teacher。

### 7.3 一次gauge、四次补偿求解

由父U恢复A^(1)，配合新G^(1)，调用与原Token-joint一致的gauge和solver。保存未gauge的A1/G1与gauge后因子，明确比例和hash。

原始PSD、正则化condition、根/逆根、SVD及FP32部署漂移均沿用父门限。不因层间效果改变阻尼或rank。

只做完整W_K的rank64截断SVD。四层候选同时冻结后进入validation。原Sensitivity-A的A根可否复用不作为优化前提，默认由同一solver按规范重新生成，避免混用尺度。

### 7.4 实际teacher-KL

跨双4090加载原teacher，每个validation窗口生成一次reference，四个新候选共享。逐模块干预，其他模块保持原teacher FP32。

新增64个主要KL点。首窗口重放四层×5旧类=20个旧点（Marginal、Sensitivity-A、Token3、Sequence、None），并将四个新候选各重复一次，共24个数值复验。

沿用完整前向评价和父容差：self-KL≤1e−10；旧值复验及重复误差≤max(1e−12,1e−7·|KL|)。不换后缀实现，不增加α扫描或新的评价曲率统计。

## 8. 报告及预先约定的解释

主表按第1节五方法逐层列出实际KL与恢复率，None另列分母。固定主对比：Joint1−Sensitivity-A、Joint1−Marginal、Token3−Joint1；Sequence只作既有参照。

\[
\mathrm{Recovery}(M)=100\left(1-\frac{KL_M}{KL_{None}}\right).
\]

统一先按预测token合并KL再算比率；报告绝对KL差、恢复率pp差、剩余KL变化及16个窗口配对结果。保留四层，不用层平均掩盖反例。

可沿用2000次固定seed配对窗口bootstrap；重采样后重算均值之比。该区间是探索性，16窗口可能相关、验证集已被多轮使用，也不覆盖训练统计随机性。

| 实测恢复率关系 | 合理表述 |
|---|---|
| SensA < Joint1，Joint1与Token3差距小 | 第一轮G替换带来收益；后两轮在本设置的增量较小，给出差值及区间 |
| SensA与Joint1差距小，Joint1 < Token3 | 第一轮G替换未显示明显额外收益；继续联合更新有收益 |
| Marginal < SensA < Joint1 < Token3 | 本设置支持逐步更新带来的性能递进；不是全局曲率拟合或迭代单调性的证明 |
| Joint1差于SensA或Token3 | 如实报告；一轮双侧更新不保证优于只换A，不自动增加新候选 |
| 四层排序不同 | 保留差异，不事后组合层级方法选择器 |

“≈”仅作附带区间的描述，不以差异不显著宣称等效。本轮不任意新增统一等效门限；若要在论文中正式宣称无需后两轮，需结合预先合理的容忍范围、效果不确定性和成本证据。

解释主线可以写为“独立边际→单侧强度加权→一轮联合更新→继续联合更新”，但不能预先把它写成严格递增的定理，也不能说后续迭代引入了新样本信息。

## 9. 成本与资源

主体依然读取约32GiB X与8GiB K梯度；从SENS_RUN额外读取四份U/D等小规模统计。没有训练数据teacher前向、后续网络反向或attention捕获。

新开销为每窗口X@A_M（4096维）、一个1024维加权G Gram，共1024个窗口-模块组合，再加四次SVD与评价。虽然只新增G，计算xᵀA_Mx仍需要完整输入侧矩阵乘法，不能说只是O(n)的输入范数。

A^(1)直接复用省去了正式A侧Gram；不重新拟合Marginal、SensA、Token3或Sequence。用前两个窗口与首模块求解时间估算总时长；两个worker重叠调用的计时之和不是墙钟时间。

增量输出预算8GiB，检查点只保留当前与上一完整代；遵循原共享存储预留要求，不自动删除父资产。模型仅在KL阶段加载，不在每个GPU各放一份完整FP32 teacher。

## 10. 交付与完成条件

至少保存：

```text
manifest.json
parent_assets_audit.json
pilot/one_step_equivalence.json
statistics/<module>/raw.safetensors  # A1, G1, V及尺度/来源
statistics/<module>/input_weight_summary.json
factors/<module>/raw_and_regularized.safetensors
corrections/<module>/token_joint_one.safetensors
candidate_freeze.json
scores/validation/...
summary/results.csv
summary/per_window.csv
summary/paired_comparisons.csv
resource_usage.json
READOUT.md
verification.json
complete.json
```

完成条件：四层N256一致性验收、四套新候选、320条旧KL身份绑定、64个有效新KL点及数值复验通过。负结果也正常完成，不自动继续第二轮、新初始化或新方法。

## 11. 已核查的本地复用入口

- [Token-joint初始化和同步轮次](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/fitting.py)
- [token_step、分母与部署路径](../qera_mxint4_full_ag/experiments/qer_kronecker_l10_v2/sketch_math.py)
- [Sensitivity-A最终U/D保存与检查点](../qera_mxint4_full_ag/experiments/qer_k_sensitivity_v1/s_statistics.py)
- [Sensitivity-A资产身份与旧G规范检查](../qera_mxint4_full_ag/experiments/qer_k_sensitivity_v1/s_assets.py)
- [Sensitivity-A求解与补偿路径](../qera_mxint4_full_ag/experiments/qer_k_sensitivity_v1/s_solve.py)
- [完整前向KL评价结构](../qera_mxint4_full_ag/experiments/qer_k_sensitivity_v1/s_evaluate.py)
- [配对报告和实际墙钟时间注意事项](../qera_mxint4_full_ag/experiments/qer_k_sensitivity_v1/s_report.py)

复用reader/solver/evaluator时建立独立新实验identity，不更改父运行的冻结源码、receipt或完成记录。旧实验入口的完整runner不能直接作为本轮runner，因为其统计任务与候选列表不同。
