# Direct Query-Marginal K：直接统计 A/G 的 QER 实验

版本：v2.1，2026-09-21。平台：双 RTX 4090。此次仅调整协议定位与结果表述，方法和实验配置不变。

本协议取代 audit 后此前拟议的 K 方法阶段方案。用户已确认 audit 完成；本协议不取代或重跑已完成的 audit。本轮 Direct Query-Marginal 实验尚未由本聊天实现或启动；本地已有 K/O、V 复用代码，远端资产存在性需执行时核验。

## 1. 本轮唯一问题

> Direct Query-Marginal geometry 是否优于标准 Marginal-K？

在相同N256、rank64和冻结评价条件下，以实际teacher-KL回答这个问题。这是整个候选构造的效果比较，不是只改变source/query grouping的因果消融。

这是一个明确的近似方法实验。真实 teacher 保留 RoPE/GQA；仅在构造新的代理几何时忽略 K 梯度中的 source-dependent 相对旋转。不把忽略旋转解释成精确重组，也不要求先证明它足够小。

主比较只有 **Marginal-K vs Direct Query-Marginal K**。已有 A-only、Token-joint、Sequence-one-step 和仅量化结果作为参照，直接复用，不重拟合。

不做 ALS、Full-fit、source/query 耦合、分离秩审计、分组曲率拟合、MLP或其他模块新方法、rank allocation。V 保持现有结果。

## 2. 固定设置

| 项目 | 固定值 |
|---|---|
| 模型与量化 | 既有 K/O 增量实验相同 teacher、tokenizer、MXINT3 Wq及源码版本 |
| 模块 | L0、L10、L20、L31 的 k_proj，四层全部报告 |
| 统计 | 原 N256 冻结训练窗口，每窗口2048 tokens、1整套原teacher采样标签 |
| loss | 原 sampled next-token sum-NLL，T=2047个预测位置 |
| rank | 完整 W_K 的 rank64；不分head配置rank |
| 评价 | 原16个validation窗口，实际KL(teacher || candidate) |
| 干预 | 每次只替换一个K模块，其余权重保持teacher FP32 |
| 精度 | teacher及原生统计变量FP32；Gram累加、根分解、SVD为FP64；关闭TF32/autocast |
| 正则与部署 | 原冻结求解器、trace-relative阻尼、FP32部署规则 |

前2个训练窗口只用于实现pilot，验收通过后其统计计入N256，不另做N8性能实验、不重采样标签。非重叠窗口不等于独立文章。

## 3. 真实变量与明确的近似

使用列向量记号。输入 x_u∈R^n，query head a 的旋转前投影 q_t^a=W_Q^a x_t∈R^d，使用 KV head b(a)。令 p_tu^a 为真实 teacher attention probability，v_u^{b(a)} 为真实 V 投影，z_t^a=Σ_u p_tu^a v_u^{b(a)}。

定义真实下游梯度和 logit 梯度：

\[
\delta_t^a=\frac{\partial\ell}{\partial z_t^a},\qquad
e_{tu}^a=\frac{\partial\ell}{\partial s_{tu}^a}
=p_{tu}^a(\delta_t^a)^\top(v_u^{b(a)}-z_t^a).
\]

s是已经包含1/√d缩放、送入masked softmax的attention logits；mask项不求梯度。所有e来自真实teacher及原冻结标签，不能换成真实文本ground-truth CE，不能把概率p当成e。

真实K score为

\[
S_K=\frac1{\sqrt d}\sum_{t,u,a}
P_{b(a)}\Omega_u^\top\Omega_tq_t^a\,e_{tu}^a x_u^\top.
\]

本轮仅在代理构造中忽略 Ω_uᵀΩ_t，定义

\[
\widetilde x_t^a=\sum_u e_{tu}^a x_u,
\qquad
\widetilde S_K=\frac1{\sqrt d}\sum_{t,a}P_{b(a)}q_t^a(\widetilde x_t^a)^\top.
\]

P_b将d维向量嵌入完整K输出的第b个块。必须用**旋转前的q**，不能在实现时混用rotated q。

一般而言，S_K≠S̃_K。真实e已经受到RoPE及下游网络影响，并不意味着被省略的旋转已由e完整补偿；这项近似最终由实际QER结果检验。

## 4. GQA下两个Gram的完整定义

预期配置为n=4096、Hq=32、Hkv=8、d=128、m=Hkv·d=1024。执行前由真实config与repeat_kv映射核验，禁止静默硬编码错位。

令N=256、L=2048、T=2047、Z=NLHq。直接定义

\[
A_{\rm qry}=\frac1Z\sum_{c,t,a}
\widetilde x_{c,t}^a(\widetilde x_{c,t}^a)^\top
\in\mathbb R^{n\times n},
\]

\[
\overline G_{\rm qry}=\frac1{dZ}\sum_{c,t,a}
P_{b(a)}q_{c,t}^a(q_{c,t}^a)^\top P_{b(a)}^\top
\in\mathbb R^{m\times m}.
\]

为沿用V prototype的每预测位置代理尺度，提供求解器的G设为

\[
G_{\rm qry}=\frac{LHq}{T}\overline G_{\rm qry}
=\frac1{NTd}\sum_{c,t,a}P_{b(a)}q_{c,t}^a(q_{c,t}^a)^\top P_{b(a)}^\top.
\]

这只是对整个K乘一个正标量；采用同样trace-relative阻尼时不会改变精确weighted-SVD补偿。保存原始计数及两种归一化关系，避免L/T/head因子混乱。

**所有L个模块位置计入固定归一化；无loss梯度的位置其e为零。不要因某行e为零就在G中选择性删除q。**

此定义的G天然为KV-block-diagonal：第b块汇总所有使用该KV head的query heads，但不含不同KV块的交叉项。同一KV组各query head也按outer-product分别累加，不先把q相加再平方。

这是一套完整尺寸的single-Kronecker候选，不是逐head独立求解。它有三类明确近似：省略相对旋转、使用逐query/head边际矩、分离q与x̃的联合依赖。不给它“精确Fisher”标签。

单头情况下，除不影响补偿的整体尺度，公式退化为A=E[x̃x̃ᵀ]、G=E[qqᵀ]/d。预测梯度信息主要进入新A，而不是要求每个名为G的因子都必须是CE梯度矩。

## 5. 精确加速：仍然只统计这两个Gram

令X∈R^{L×n}，每个head的D_a=[e_tu^a]∈R^{L×L}。则

\[
\sum_a(D_aX)^\top(D_aX)
=X^\top\left(\sum_aD_a^\top D_a\right)X.
\]

每窗口：

1. 从真实p、v、δ构造D_a，按head流式处理；
2. 在FP64累加B=Σ_a D_aᵀD_a；
3. 累加A_sum=XᵀBX；
4. 各KV块累加对应原始query的qᵀq/d；
5. 最后按第4节统一归一化。

这与显式形成全部x̃后计算Gram相等，不引入新的数学近似。它保留D_aᵀD_a中的带符号跨source项；不能改成只用其对角线、|e|或e²逐元素权重。

不形成完整H、每query权重矩阵M_t或8192个分离秩对象。复用V统计代码的收缩结构，但将attention概率矩阵替换为真实logit梯度矩阵D，并将G输入改成旋转前q。

## 6. 最小正确性pilot：不要求近似成为恒等式

先在训练窗口0、1、L20检查，再覆盖其他三层。检查内容限于实现必需项：

- 实际teacher、Wq、文本、标签、sum-NLL、mask和GQA映射与父实验匹配；
- 由缓存down梯度恢复的δ，与真实连通反向得到的δ一致；
- e公式与真实softmax backward一致；若后端无法直接暴露logits梯度，使用包含真实RoPE的K链式回传与缓存/autograd g_k比较；
- e行和在浮点误差范围内为零，mask位置无贡献；
- FP64显式Gram与XᵀBX相符；可先在固定截取的head/query计算块核验代数，不截断模型上下文、不改变正式统计；
- 无RoPE代理内部的两种S̃求和方式一致。

**禁止把S̃_K与真实∇W_K ℓ不相等判为失败。该差异就是方法有意接受的近似，不是数值验收门。**

FP32与真实反向比较沿用相对Frobenius误差1e−5量级；FP64纯代数比较1e−10。记录绝对误差与参考范数；近零行使用明确的绝对尺度，不用不稳定相对误差判断。

pilot不新增性能候选、不计算RoPE分离谱、不做ALS，不用validation选统计方式。通过后直接收集N256。

## 7. 资产复用与实际执行

BASE_RUN为原q/v/down实验；KO_RUN为K/O增量实验。两者只读。

| 资产 | 来源与用途 |
|---|---|
| 原N256文本、标签、同16窗口validation | BASE_RUN/data与cache/labels；原样复用 |
| QKV输入X | BASE_RUN/cache/x；核对重放输入 |
| down输出梯度 | BASE_RUN/cache/g；恢复局部下游梯度 |
| K输出梯度 | KO_RUN/cache/g；pilot真实链式法则验收 |
| K的W0/Wq | KO_RUN/quantized；直接复用，不重新量化 |
| 四种K补偿与逐窗口KL | KO_RUN/modules、scores/validation；不重新拟合或整套评价 |

每个训练窗口一次teacher无梯度前向捕获四层真实变量。借用down输出梯度Λ，令r为attention后残差，得到

\[
\lambda=\Lambda+J_{\mathrm{MLP}\circ\mathrm{RMSNorm}}(r)^\top\Lambda,
\qquad\delta=W_O^\top\lambda.
\]

这只需局部MLP VJP。r必须从真实前向捕获，不能从归一化QKV输入反推。q可由原q_proj输出捕获；δ、p、v均保持真实RoPE/GQA路径。

N256主体共256次共享无梯度前向、1024次局部MLP VJP及四模块统计；不用每模块/head完整反向。两个pilot窗口额外共享完整反向用于验收。

无需把所有窗口的attention、e或x̃落盘。每窗口释放中间量，保存累计A/G的原子检查点与该窗口数值检查。只读取必需父资产，不复制整个旧缓存。

缺少down缓存时先报告成本变化；fallback是原标签完整共享反向，不是改统计公式。远端路径、hash和实际缓存完整性尚须核验。

可复用源码入口：

- [V真实变量捕获与局部delta](../qera_mxint4_full_ag/experiments/qer_v_attention_proto_v1/v_capture.py)
- [V流式Gram结构](../qera_mxint4_full_ag/experiments/qer_v_attention_proto_v1/v_math.py)
- [V累计统计与检查点](../qera_mxint4_full_ag/experiments/qer_v_attention_proto_v1/v_statistics.py)
- [K/O资产读取](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_assets.py)
- [K/O后缀评价与基线格式](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_evaluate.py)

新增实现与输出保持独立identity，不修改既有运行的冻结代码或receipt。

## 8. 补偿求解和实际KL验证

检查原始A/G对称、PSD及维度；沿用父实验阻尼与数值门限，记录条件数，不因结果差临时调阻尼。预期trace-relative η_A=η_G=1e−3，执行前绑定实际父配置。

整个K矩阵统一求解：

\[
C_{\rm qry,64}=G^{-1/2}[G^{1/2}EA^{1/2}]_{64}A^{-1/2}.
\]

统计G的block结构不意味着给各head各64 rank。四层候选同时冻结hash后再进入validation。

部署沿用 `Wq + float32(C64)`；不改成先把两个因子各转FP32再相乘。实际KL针对实际W_deploy计算，并记录部署舍入误差。

主结果表每层含：仅量化、A-only、Marginal、Token-joint、Sequence-one-step、Direct Query-Marginal。已有4×5×16=320条结果复用；新增4×16=64个主要KL点，另有teacher reference与少量路径复验。

同一窗口teacher logits共享；复用既有后缀评价须通过真实候选与完整前向的一致性检查。父结果身份或路径无法匹配时不能直接拼接百分比。

报告：绝对KL、恢复率、相对Marginal的恢复率百分点变化、相对Marginal剩余KL降低比例、相对Token/Sequence差距；逐窗口值保留。

\[
\mathrm{Recovery}=100\left(1-\frac{\mathrm{KL}_{candidate}}{\mathrm{KL}_{None}}\right).
\]

KL先按预测token合并，再计算比率。配对bootstrap可沿用历史探索性实现；16个窗口的区间不涵盖训练统计随机性，也不能称作新的独立测试集证明。

L0必须保留。原Marginal恢复率高意味着绝对百分点空间小，但剩余KL的相对改善仍可能有意义。

## 9. 预先约定的结果解释

| 结果 | 本轮可以支持的结论 |
|---|---|
| Query优于Marginal，接近Token/Sequence | 这种忽略相对旋转的简单query几何在本设置有用，值得独立复验 |
| Query优于Marginal但仍有差距 | 简单候选有部分收益；不能唯一归因于遗漏RoPE或某类coupling |
| Query未改善或退化 | 当前Direct Query-Marginal构造不成立，结束本候选，不自动升级ALS |
| 层间结果不同 | 完整报告四层差异；不事后设计层选择器 |

两者同时改变 grouping、head marginalization 及 RoPE treatment，因此首次正结果证明的是整个 Direct Query-Marginal 构造在本实验设置下有效，而不是单独证明 query grouping 是唯一收益来源。负结果也不能区分signed competition无用、忽略旋转不合适或Marginal分离失真。

这些解释边界不增加本轮消融任务。先完成一个最简单候选的实际QER检验，再决定是否需要其他研究。

## 10. 资源估计与交付

旧8窗口审计的30–90分钟估计和2小时上限不再适用于本协议。本轮是N256新Gram统计，不应继续引用旧耗时承诺。

时间主要由256次共享捕获、1024次局部VJP以及每窗口四模块的DᵀD与XᵀBX决定。具体应复用V正式运行的对应阶段日志，并用前两个窗口测量D构造与Gram增量；不要按旧12模块完整实验的总时间线性缩放。

输出至少包含：父资产审核、冻结配置、pilot数值检查、每窗口累计统计检查点、四层raw/regularized A/G、四套补偿、64个新KL点、逐窗口配对结果、分阶段计时/峰值显存、READOUT。

最终READOUT先回答唯一主问题：**在完全相同的N256和rank64预算下，Direct Query-Marginal geometry是否在实际teacher-KL上优于标准Marginal-K？** 不把“是否精确重组真实K梯度”重新设置成方法的成功标准。
