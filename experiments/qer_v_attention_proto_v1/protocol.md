# V attention-aware prototype：资源复用与三步验证协议

版本：v1.0，2026-09-21。平台：双 RTX 4090。

状态：已完成本地历史源码与资产格式核查；尚未登录服务器核验缓存存在性，尚未实现或运行本 prototype。本文不声称方法成立，不启动服务器计算，也不覆盖历史代码或结果。

## 1. 本轮唯一问题

> 对真实 Llama GQA，将 V 权重的精确梯度改写为 attention-derived effective variables 的外积和，再对这些变量作边际分离，能否改善固定 rank64 的量化误差补偿？

只验证三件事：

1. **Check 1**：真实 GQA 上的梯度恒等式与局部梯度复用是否正确。
2. **Check 2**：构造一套明确的完整 V 权重尺度的 A/G，检查维度、PSD、阻尼与一次全矩阵 rank64 求解。
3. **Check 3**：与原始输入 Marginal、Sequence-one-step 比较实际 teacher-KL。

不进行全局曲率拟合，不引入 Full-fit、Token-joint 新候选、K-attention-aware、rank 分配、head-wise rank 分配、额外样本量扫描、完整模型联合部署或新下游任务。

本轮不是已认可的 q/k/v/o 补齐实验的重命名；它是独立的 V 方法 prototype，数据和基线尽量复用前期实验。

## 2. 模块、数据与固定条件

| 项目 | 固定内容 |
|---|---|
| Teacher | 与已有 12 模块实验完全相同的 Llama-3.1-8B Base、FP32权重、tokenizer及配置 |
| 模块 | layers 0、10、20、31 的 self_attn.v_proj，共4个 |
| 量化 | 对应模块现有 MXINT3 Wq，不能重新量化 |
| 残差 | E=W0−Wq，使用匹配的teacher权重 |
| rank | 整个 W_V 上统一 rank64 |
| 拟合数据 | 父实验N256：WikiText-2 train的256个冻结窗口 |
| 长度 | L=2048，T=2047 |
| 标签 | 每窗口1份原有teacher采样标签，原样复用、不重新抽样 |
| 评价 | 父实验同一16个WikiText-2 validation窗口 |
| 主指标 | 实际KL(p_teacher || p_quantized+correction)，每预测位置平均 |
| 部署范围 | 一次仅干预一个V模块，其他模块保持原始FP32 |
| 数值 | teacher及原生activation/梯度FP32；统计/收缩/矩阵根/SVD为FP64；禁用TF32/autocast |
| 求解/部署 | 沿用父实验相对阻尼与FP32 dense Wq+float32(C64)路径 |

四层均为预先指定的正式结果：L20用于先做实现核验；L10/L20/L31已有Sequence收益；L0是原有Sequence未改善的对照。不能根据prototype效果删除L0或只报告成功层。

256个窗口不是256篇独立文章。父数据为拼接文本的非重叠2048-token窗口；文章独立性不能假定。

## 3. 已核查的历史实现与资源边界

本地识别到与用户描述的12模块、128/256预算、三方法FP64实验相符的实现：

`qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/`

源码明确：四层q/v/down，共12模块；方法为Marginal、Token-joint三轮同步、Sequence-one-step；各方法共享文本、标签及连通梯度；保存逐模块原生FP32 x/g；求解与收缩使用FP64；实际验证逐模块替换。

**证据边界：本地没有本次12模块服务器输出目录与完整冻结运行日志。以下“存在”指源码规定会生成，实际文件是否保留、对应哪个run、是否完整，必须在服务器只读盘点。不能把本地源码阅读写成已经完成远端资产验收。**

### 3.1 资源复用清单

令 PARENT 为实际完成N256实验的根目录。路径中的 slug 必须调用父实现的 slug 函数，不手工猜测模块名替换方式。

| 资源 | 父目录相对路径/来源 | 本轮用途与状态 |
|---|---|---|
| 冻结数据 | data/fit.safetensors、validation.safetensors、相应json、data/index.json、data/freeze.json | 直接复用N256文本、顺序、mask、hash；不运行新选样 |
| 实际采样标签 | cache/labels/<sample_id>.safetensors及receipt | 身份核验及Check 1完整反向参考；正式局部重放不重新采样 |
| 原始QKV输入x | cache/x/<slug(qkv_input_group)>/wXXXX.safetensors | V的真实归一化输入；q/k/v共享；复用并与重放输入核对 |
| down_proj输出梯度 | cache/g/<slug(layer.mlp.down_proj)>/<sample_id>.safetensors | 四层后续网络反向信号，作为局部重放的主要资源 |
| v_proj输出梯度 | cache/g/<slug(layer.self_attn.v_proj)>/<sample_id>.safetensors | 校验新的GQA梯度表达；不当作delta_t^a直接使用 |
| 缓存提交清单 | cache/commits/<sample_id>.json | 绑定文本、标签和相关缓存哈希，识别缺失/损坏 |
| 量化权重 | quantized/<slug(v_module)>.safetensors、quantized/freeze.json | 直接复用W0/Wq及量化身份 |
| 基线因子 | modules/<slug(v_module)>/factors/N256/<method>/raw.safetensors | 分析与身份核验；无需重新拟合 |
| 基线部署 | modules/<slug(v_module)>/corrections/N256__Marginal.safetensors及N256__Sequence-one-step.safetensors | 直接复用P64/Q64/W_deploy与数值audit |
| 基线逐窗口KL | scores/validation/wXXXX/<slug(v_module)>___<candidate>.json | 读取N256__Marginal、N256__Sequence-one-step、None三类；构成配对比较 |
| 基线汇总 | summary/results.csv、summary/results.json | 交叉核对；正式统计使用逐窗口值而非四舍五入百分比 |
| 冻结清单/环境 | manifest.json、candidate_freeze.json、complete.json、teacher身份及source/config pins | 确认确实对应用户的N256表 |

只读取以上4个V模块和4个down模块需要的资产；不默认复制或逐文件重哈希整个216GiB历史缓存。首次对必需资产校验一次，后续按只读、文件身份与检查点复用，避免每个head/候选重复全量校验。

### 3.2 历史代码未持久化的资源

- 原evaluation.py只持久化teacher数值结果，reference logits在内存中逐窗口复用，不能假定磁盘存在可直接使用的teacher logits缓存。
- 原collector不保存各head的attention probability、attention聚合输出梯度delta、attention后残差输入r。
- 原qkv输入缓存是归一化后的输入，不是完整decoder block的原始residual输入；不得据此直接重建整个block残差支路。
- 没有保存bar_x_t^a，不能从已有Marginal A/G反推出它。

这些量需要短期重放，或在Check 1后从等价局部计算获得。

### 3.3 父运行绑定

执行者提供实际PARENT路径后，生成parent_assets_audit.json，记录：父identity、实际配置、源代码hash、每个必需文件的存在性/哈希/形状/dtype、样本与标签绑定、256/16窗口身份、4×3×16=192条既有baseline窗口结果的完整性。

若源码版本与本地所读版本不同，以冻结运行的代码与配置为准并记录差异。使用独立的父资产reader保留旧identity，不把旧receipt改写成新identity，不直接修改父run以适应新实验。

缺少逐窗口结果而仅有百分比汇总时，不能伪造配对结果；可以在相同冻结权重/输入上重放缺失评价，计入额外成本。缺少down梯度时可采用同标签的完整共享反向作为等价fallback，但必须重新估算并记录执行路径，不能宣称已使用缓存加速。

## 4. 精确GQA定义

先从实际模型配置核验头数与排列。预期Llama-3.1-8B：输入n=4096、query heads Hq=32、KV heads Hkv=8、head维d=128、V输出m=Hkv*d=1024；配置或形状不符则停止检查绑定，不硬编码reshape绕过。

使用列向量记号。第a个query head使用KV head b(a)，映射必须由实际repeat_kv/reshape顺序验证。令：

\[
v_u^b=W_V^b x_u,\quad
z_t^a=\sum_u\alpha_{tu}^a v_u^{b(a)},\quad
\bar x_t^a=\sum_u\alpha_{tu}^a x_u.
\]

alpha必须来自原teacher的真实Q/K、RoPE、causal mask、scale及softmax；不省略RoPE，不平均各head的attention，不更换attention backend。只改变V时同层alpha保持不变。

sampled loss为原冻结teacher标签的sum-NLL，定义：

\[
\lambda_t={\partial\ell\over\partial y_t},\quad
\delta_t=W_O^\top\lambda_t,
\]

delta按query head切分为delta_t^a。它是o_proj输入（各query-head聚合输出拼接后）的梯度，不是GQA重复前v_proj输出梯度。

令P_b把d维向量嵌入完整m维V输出空间的第b个KV行块：

\[
\widetilde\delta_t^a=P_{b(a)}\delta_t^a.
\]

精确梯度恒等式为：

\[
\boxed{
S_V=\nabla_{W_V}\ell
=\sum_{t,a}\widetilde\delta_t^a(\bar x_t^a)^\top.
}
\]

源位置表示为：

\[
g_{V,u}^b=\sum_{a:b(a)=b}\sum_t\alpha_{tu}^a\delta_t^a,
\qquad
S_V=\sum_u g_{V,u}x_u^\top.
\]

所有t/u循环保留L个模块位置，损失只使用T个有效预测位置，遵循父实验mask。不能把无损失位置的梯度人为赋非零，也不能将协方差计数偷偷改成T而不改全协议。

## 5. 复用down梯度：只重放局部MLP反向

Llama某层最后的计算为：

\[
r=h+y,\quad h_{out}=r+\operatorname{MLP}(\operatorname{RMSNorm}(r)).
\]

down_proj之后只接残差加法，因此已缓存：

\[
\Lambda={\partial\ell\over\partial\operatorname{down\_proj}(\cdot)}
={\partial\ell\over\partial h_{out}}.
\]

从实际teacher无梯度前向捕获r（post_attention_layernorm的输入）后，把r当独立leaf重放该层norm+MLP：

\[
\lambda={\partial\ell\over\partial y}
=\Lambda+J_{\operatorname{MLP}\circ\operatorname{RMSNorm}}(r)^\top\Lambda.
\]

再用原W_O求delta=W_O^T lambda。这是相同样本的链式法则复用，不是把当前层截断后定义新损失。无需正式阶段重复反向后续网络，也无需为V计算Q/K路径的反向。

每个拟合窗口：

1. 一次无梯度teacher前向，捕获四层r与真实attention概率；与已缓存QKV输入比对。
2. 每层读取对应sample的原down梯度，局部重放norm+MLP并计算lambda/delta。
3. 与原V梯度进行源位置聚合一致性检查；累计新A/G。
4. 释放本窗口attention及临时变量，写入有界检查点。

attention捕获必须不改变原eager数值路径；若需重算Q/K/softmax，必须在Check 1中与teacher实际attention输出核验。原teacher所有权重冻结，dropout关闭，KV cache关闭。

这一路径需要256次共享无梯度前向和4×256次局部MLP VJP，不是4×256次完整模型反向。Check 1的少量完整反向另计。可只运行到最后一个所需捕获点，不能省略计算目标r所需的residual支路。

## 6. Check 1：两层数值核验，通过前不收全量统计

### 6.1 Pilot样本与顺序

固定父fit窗口0、1及其原label，先L20，再L0/L10/L31；所有四层均通过才进入正式统计。不根据这些样本的补偿效果选择方法。

对两窗口取得正常完整teacher反向的W_V梯度参考；可一次连通反向获取四层，不能逐模块切断中间图。只在必要的目标参数上启用权重梯度，不为整个模型保存参数梯度。

### 6.2 局部重放检查

- 完整前向与重放的QKV输入、r、attention聚合输出、head排列一致。
- 缓存down梯度与同一标签完整反向一致。
- 由缓存Lambda得到的lambda/delta，与正常反向得到的o_proj输出/输入梯度比较。
- 由alpha、delta聚合得到的g_V，与原缓存/正常反向g_V比较。
- 检查repeat_kv对应关系，确认多个query head的贡献加到同一个KV权重行块。

### 6.3 FP64代数与FP32 autograd分开检查

将相同的原生FP32 X、alpha、delta转换为FP64，只改变收缩精度：

- 表达A：先算各KV head的g_V64，再形成S_source=g_V64^T X。
- 表达B：先算bar_X^a=alpha^a X，再按head嵌入累加S_effective。
- 两者在FP64下验证求和重排恒等性。
- S_effective另外与正常FP32 teacher的W_V autograd梯度、缓存g_V^T X比较。

冻结容差：FP64重排相对Frobenius误差≤1e−10；FP32 autograd/局部重放相对误差≤1e−5。记录绝对Frobenius误差、max-abs误差和参考范数；比例分母保护固定1e−12，对参考范数≤1e−12的退化检查单独报告，不能用大分母掩盖错误。

这些误差门限不同，不能要求FP32参考也达到FP64代数精度。若失败，停止并定位；不直接放宽门限进入效果比较。失败不视为方法无效，而是尚未完成正确实现。

正式256窗口中，每层持续校验由alpha/delta重建的g_V与父缓存，发现漂移停止该阶段。正常完整反向仅在上述pilot使用，不重复执行全部256份。

## 7. Check 2：唯一预注册的A/G构造

本prototype只测试以下一种定义，不搜索head权重、不按效果调归一化。

所有窗口c、模块位置t、query head a等权，N=256：

\[
A_{attn}={1\over NLH_q}\sum_{c,t,a}\bar x_{ct}^a(\bar x_{ct}^a)^\top,
\]

\[
G_{moment}={1\over NLH_q}\sum_{c,t,a}\widetilde\delta_{ct}^a(\widetilde\delta_{ct}^a)^\top.
\]

先删除该outer-product表示的不同(t,a)项之间的交叉项，再分离两侧联合统计，对应canonical proxy：

\[
K_{raw}={LH_q\over T}A_{attn}\otimes G_{moment}.
\]

交给现有双因子接口时设置：

\[
\boxed{A_{raw}=A_{attn},\qquad G_{raw}={LH_q\over T}G_{moment}.}
\]

等价地，G_raw的每个KV block为：

\[
(G_{raw})_b={1\over NT}\sum_{c,t}\sum_{a:b(a)=b}
\delta_{ct}^a\delta_{ct}^{a\top}.
\]

A/G不中心化。delta来自predictive标签，不是真实标签CE。scalar归一化必须保存以便复核；不得按validation校准metric或补偿幅度。

### 7.1 完整矩阵与head结构的限定

- A_raw尺寸n×n，G_raw尺寸m×m，与完整W_V匹配。
- 由于嵌入向量只在一个KV block非零，G_raw天然是KV-head block diagonal。
- 不同KV head间非对角块被舍弃；同KV head中来自不同query head/位置的交叉项也未显式保留。
- A_raw是在全部query heads上合并的一个公共输入度量。
- 这是一种有额外近似的候选metric，不是完整GQA Fisher的精确分解。
- 仍对整个W_V做一次rank64 SVD；不能每head各rank64或临时分配64/Hkv。

“full-matrix”只指维度覆盖完整W_V并全局求解，不声称保留全部跨head曲率。报告中必须写明此限定。

### 7.2 精确运算重排：避免逐head大协方差

按行排列X∈R^(L×n)，每head的P_a∈R^(L×L)，bar_X_a=P_a X。则每窗口：

\[
\sum_a\bar X_a^\top\bar X_a
=X^\top\left(\sum_aP_a^\top P_a\right)X.
\]

优先在FP64中逐head累计B=ΣP_a^T P_a，再计算X^T B X。G的每个block直接累计delta_a^T delta_a。这与显式bar_X协方差定义精确等价，不做attention平均、低秩截断、采样位置或改变精度。

在Check 1两窗口上，将该快速A计算与显式逐head bar_X^T bar_X比较，相对误差≤1e−10，再用于正式统计。B为L×L约32MiB FP64，避免对每head都执行一次n×n协方差累计。正式不保存全部bar_X。

梯度identity的pilot仍显式构造bar_X，避免两个被测路径共享同一个实现错误。

### 7.3 数值性质与求解

统一按父求解顺序：PSD/对称性检查 → gauge（保持K不变）→相对阻尼 → dense FP64根与SVD → FP32部署。

\[
A_s=A+10^{-3}{\operatorname{tr}A\over n}I,
\qquad
G_s=G+10^{-3}{\operatorname{tr}G\over m}I.
\]

由冻结父配置核对eta_A=eta_G=1e−3，不因prototype效果更换。若实际父配置不同，先解决版本绑定，不能悄悄混用。

保存raw谱/秩诊断/trace、solve条件数、阻尼量、block结构与头映射。Raw PSD可以奇异；负特征值低于−1e−10×最大特征值视为失败。两侧solve条件数≤1e8，根重建误差≤1e−10，SVD谱尾恒等式及逆变换误差≤1e−8。不做未登记的特征值裁剪。

\[
C_{64}=G_s^{-1/2}[G_s^{1/2}E_VA_s^{1/2}]_{64}A_s^{-1/2}.
\]

使用线性solve逆白化，不显式求逆。复用父dense SVD，不使用随机SVD，不用head级求解拼接替代。FP32部署为Wq+float32(C64)，不是先将P/Q各自转FP32再相乘；必须匹配父实现。

计算理论与部署残差，代理目标相对漂移≤1e−4；实际KL必须评价真实W_deploy。所有四个候选及hash冻结后才进入validation。

## 8. Check 3：实际QER对照

每个V模块仅比较：

1. 原始输入Marginal：父N256__Marginal。
2. Attention-aware Marginal：本轮唯一新候选。
3. Sequence-one-step：父N256__Sequence-one-step。

另保留父None（Wq）作为恢复率分母。不把“原始输入Marginal”称为未阻尼Raw metric。

评价alpha=1，逐模块干预，同16个窗口，teacher到student方向KL，完整词表、每预测位置平均，沿用父stable_kl和FP64归约。不给validation采样teacher标签，不计算q_full或曲率cosine。

### 8.1 基线复用与最少重放

- 优先导入父4模块×3基线×16窗口=192条逐窗口记录（两方法及None）。
- 新增4候选×16窗口=64条主KL记录。
- 原teacher logits未持久化时，每验证窗口重新生成一次并在四个新候选间共享；self-KL必须≤1e−10。
- 固定validation窗口0，重放四模块的三种父基线，共12条复核记录；与父逐窗口结果比较，误差≤max(1e−12,1e−7×|KL_parent|)，并核验权重、token及运行数值身份。
- 若重放不通过，不直接拼接旧新汇总；先定位环境/实现差异，必要时将全部192条基线统一重评并另记成本。不能放宽容差使两种环境强行可比。
- 四个新候选在窗口0各重复一次，共4条重复检查；原候选身份与新候选恢复必须通过权重hash检查。

正常新增主工作为64条KL；另有12条基线核验、4条新候选重复、teacher/self检查。不得把核验遗漏或统一基线重评当作已包含在64次中。

可复用每窗口的teacher前缀，从目标层起做后缀前向，但必须先与完整前向核对；最小实现可直接沿用旧全前向，避免prototype同时引入未经验证的新评价路径。

## 9. 指标、区间与解释

对方法m，使用全16窗口合并的KL（长度相同则窗口等权）：

\[
\gamma_m=1-{KL_m\over KL_{None}}.
\]

同时报告绝对KL，不只报告恢复率。主要配对效应为：

\[
d_{AM}={KL_{Marginal}-KL_{Attn}\over KL_{None}},\qquad
d_{AS}={KL_{Sequence}-KL_{Attn}\over KL_{None}}.
\]

正值表示Attention-aware更好。实际意义门限固定0.02（未补偿损伤的2个百分点）。如果None≤1e−12，归一化标为不可判定，报告绝对差，不抬高分母制造恢复率。

本轮16窗口已被用于前期分析，不是最终盲测。窗口不保证文章独立：有文档归属则按文档组做配对重采样；没有时报告固定16窗口的逐窗口效应，并将2000次窗口配对bootstrap明确标为探索性、未控制窗口相关性。随机种子固定由`qer-vattn-proto-v1/bootstrap`派生，方法共用索引。区间不涵盖fit重新采样，四层比较不声称整体显著性保证。

“Attention-aware接近Sequence”不能只看点估计：在可解释的配对区间条件下，d_AS整个区间落入[−0.02,+0.02]才支持当前门限内接近；区间宽则未确定。描述性效果与统计证据分列。

| 结果 | 允许的结论 | 不允许自动推出 |
|---|---|---|
| Attn≈Sequence且优于Marginal | 此有效变量metric在相同预算下有望达到Sequence的QER效果 | 已证明Sequence收益主要来自这一机制 |
| Attn优于Marginal但弱于Sequence | 该具体构造有用，但仍有性能差距 | 差距唯一来自跨位置或跨head项 |
| Attn无收益/退化 | 本构造在当前条件下无效或有风险，应保留负结果 | 精确前向/梯度恒等式错误，或所有attention-aware方法无效 |
| L0与中后层模式不同 | 模块深度依赖值得解释 | 删除L0以美化结论 |

新方法同时改变A/G的变量、head相关处理及其估计方差；一次效果比较不能单独归因给A的混合输入。当前不增加拆A/G的额外ablation。

## 10. 双4090资源安排与可复用程度

### 10.1 保留原环境与资产

使用原双4090 FP32模型并行环境，预计前16层在GPU0、后16层及输出头在GPU1；以实际冻结device_map为准。只读父run和模型目录，新建独立prototype输出，不能修改仍在运行的旧任务。

原实现已共享反向且已双卡离线拟合，本轮不能把这两点再次算成新增提速。主要新增节省来自：缓存down梯度、只构造4个新补偿、复用既有基线结果，以及有效A的精确重排。

### 10.2 分阶段执行

1. CPU只读资产盘点及哈希绑定。
2. 两个fit窗口、四层的Check 1，包括少量完整teacher反向。
3. 每fit窗口一次无梯度teacher前向、四次局部MLP反向及A/G累计。
4. 累计结束后卸载teacher，双卡离线构造/求解四模块；worker队列每卡至多一个求解任务。
5. 四候选冻结后重载teacher，执行Check 3。

第3阶段可由每层所在GPU做相关统计，但必须基于实测显存调度；模型驻留时不同时启动大规模eig/SVD。不可为了“用满两卡”创建两个完整FP32 teacher副本。

### 10.3 缓存策略

- 父文件直接只读复用，不复制全量x/g缓存到新run。
- P_a、r、delta、bar_X只在当前窗口/当前head或层短期保留；不落盘256窗口×32heads的全部attention或有效输入。
- 每层只需要持续保留A累计（4096² FP64约128MiB）、G的8个128² FP64 block（约1MiB）、计数和checkpoint。完整G为1024² FP64约8MiB。
- 每个head的P在2048长度下约16MiB FP32；32heads约512MiB。一次不要让所有层的中间量与完整autograd图都留在显存，可移到CPU并逐层释放。
- B=ΣP^T P约32MiB FP64；显式bar_X每head约64MiB FP64，仅pilot或局部核验暂存。
- 新增输出以因子、4份部署、pilot证据和表格为主；建议预留16GiB独立空闲空间，并按实现实测冻结上限。不得将父已有缓存容量算作新增写盘需求。
- 原缓存的首次哈希与读取仍有成本；按样本顺序读取，一次核验后不在head循环里重复读盘/哈希。
- 采用全局窗口提交检查点：四层累计同一批窗口后原子提交generation，记录最后已计窗口，断点恢复不得重复累计。

### 10.4 时间估计与停止条件

不沿用旧12模块“十几个小时”作为本轮耗时预测；也不因只新增64个KL点就声称整个prototype很便宜。主要新增矩阵成本在256×4个module-window的attention与有效协方差累计。

pilot必须分别测：必需资产读取、完整检查反向、无梯度共享前向、局部MLP反向、head概率提取、快速A累计、G累计、SVD及KL。A的显式/快速双实现只在pilot，不在全部256窗口重复。

正式估算用逐阶段实测和实际服务器CPU/cgroup内存、磁盘吞吐；资源记录需同步CUDA计时，区分双卡并行时间与求和GPU工作量。租用两卡不代表单个SVD自动两倍加速。

Check 1失败立即停止；Check 2任一模块失败则保留诊断、不将剩余成功层包装成完整四层结果。预算不足或OOM则检查点退出，不静默改精度、样本数、head数或rank，不自动续租。任何等价性能优化必须在正式前通过数值对照，不以效果作为验收标准。

## 11. 工作量与复用计数

| 项目 | 数量/范围 |
|---|---|
| 新文本/新标签 | 0 |
| 拟合窗口 | 复用256个 |
| 正式完整teacher反向 | 主路径0；pilot少量完整反向另计 |
| 正式共享无梯度前向 | 256次 |
| 局部MLP VJP | 4×256=1024次 |
| 新A/G | 4套，非12套 |
| 新全矩阵rank64补偿 | 4份 |
| 新主KL | 64个candidate-window点 |
| 复用基线KL | 192条逐窗口记录 |
| 额外KL核验 | 12条基线重放、4条新候选重复，teacher/self另计 |
| 新评价梯度/完整H/Gram/Full-fit | 0 |
| 新联合test或PPL任务 | 0 |

如果父缓存不完整导致fallback，上表必须按实际额外工作更新，不能继续声称“正式完整反向0”。

## 12. 输出与验收

建议新目录：`outputs/qer_v_attention_aware_prototype_<run_id>/`，服务器根路径由现有工作区决定。

```text
protocol.md
manifest.json                       # 新identity，父identity，数值/硬件/方法冻结
parent_assets_audit.json             # 实际远端存在性与hash；区别于本文源码审计
reuse_map.json                      # 每项复用/重放/新算资产及原因
head_mapping.json                   # GQA heads、行块、o_proj切分
pilot/gradient_identity.json
pilot/local_replay.json
pilot/effective_A_reassociation.json
pilot/resource_estimate.json
statistics/progress/                # 原子累计检查点、已计窗口
statistics/<module>/raw.safetensors
factors/<module>/solve_audit.json
corrections/<module>/attention_aware.safetensors
candidate_freeze.json
scores/validation/                  # 新KL及baseline来源引用
summary/per_window.csv
summary/results.csv
summary/paired_comparisons.csv
verification.json
resource_usage.json
RESULTS.md
READOUT.md
```

最终报告必须包括：

1. 四层Check 1误差及GQA映射，局部重放是否通过。
2. A/G定义、shape、G块对角性质、阻尼和条件数、全矩阵rank64验收。
3. 四层三方法的绝对KL、恢复率、相对Marginal/Sequence的配对变化；保留L0。
4. 基线复用校验与所有新增64主点是否齐全。
5. 真实用时与资源复用收益；不以理论省下的反向次数代替实际加速测量。
6. 明确本结果是prototype诊断，不能直接声称方法成立或机制被唯一解释。

## 13. 本地核查来源

以下是本次实际读取的源码，链接相对本文位置：

- [12模块实验README](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/README.md)：模块、双预算、方法、缓存、评价范围。
- [冻结数据实现](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/dataset.py)：256/128嵌套窗口、16validation、标签ID与来源。
- [缓存收集实现](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/collection.py)：共享QKV输入、逐模块g、实际labels、commit绑定。
- [共享teacher与梯度](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/shared_teacher.py)：连通反向、原生FP32缓存、临时teacher logits。
- [量化资产导入](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/assets_local.py)：W0/Wq身份及父来源。
- [已有三方法拟合](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/fitting.py)：N256基线因子/部署路径、双卡离线队列。
- [实际KL与结果路径](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/evaluation.py)：逐模块validation记录与基线分母。
- [求解部署实现](../qera_mxint4_full_ag/experiments/qer_kronecker_l10_v2/sketch_math.py)：FP64求解与Wq+float32(C64)。
- [数值冻结参数](../qera_mxint4_full_ag/experiments/qer_kronecker_l10_v2/plan.json)：rank、阻尼、验收容差；最终需与父run实际冻结副本一致。

这些来源说明“哪些资产可以复用、怎样复用”，不证明服务器缓存尚在。执行者首先完成第3节的父资产实际验收，再运行三个Check。
