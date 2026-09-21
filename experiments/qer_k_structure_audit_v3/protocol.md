# K 专项实验：source/query 分组、RoPE 可分离性与 QER 几何

版本：v1.0，2026-09-21。平台：双 RTX 4090。

**本文件是实验协议，不代表实验已经运行。当前可执行主体是 K 的小规模机制审计；N256 新方法比较是后续阶段，必须先补齐真实 RoPE/GQA 下候选的数学定义。执行者不得自行把无 RoPE 公式当作真实 Llama 方法，直接启动 N256。**

## 1. 主任务与范围

唯一主任务：研究 K 的计算结构是否能帮助我们构造更适合固定 rank64 QER 的单 Kronecker 度量。

主线保持为：

\[
\text{teacher-KL}\longrightarrow H
\longrightarrow K=A\otimes G
\longrightarrow C_{K,64}
\longrightarrow\text{实际 teacher-KL}.
\]

现有 K 结果提供动机：A-only 明显弱于 Marginal；中后层 Token-joint / Sequence-one-step 又优于 Marginal。待检验的解释是：**按 query 聚合不同 keys 的有符号竞争信息，是否提供了 Marginal 未有效保留的 QER 信息？**

这不是已证实的机制。V 的成功也不能替代 K 的验证。

本轮保留 L0/L10/L20/L31 的 k_proj，四层均报告。只读取已有 Q/O/V/MLP 资产以支持 K 的计算；不新增这些模块的方法或性能实验，不做 rank allocation、Full-fit、耦合系数搜索、全模型联合部署。

最终成功标准是同一 Wq、rank、统计预算下的实际 teacher-KL 改善。全局曲率拟合、分离秩和分组二次分数都是解释与构造依据，不能取代最终标准。

## 2. 对附件的审阅结论

### 2.1 保留的判断

- source-centric 与 query-centric 是同一个精确权重梯度的两种组织方式，不是两份独立观测。
- 先验证 query-only 的可计算构造，再讨论 source/query 耦合，研究顺序合理。
- 分别平均两组 A 和 G，会引入额外的交叉 Kronecker 项；这不是对两个原始度量的简单平均。
- 两个 Kronecker 度量的非负加权和是 PSD，但一般不再支持原来的一次双侧加权截断 SVD。共享某一侧等特殊情形除外。

### 2.2 必须收紧的表述

**第一，明确两层近似。** source/query 分组先各自删除不同的 edge 交叉项；将分组统计再分解成 A⊗G，又引入一次联合依赖近似。因此不能直接把最终 Marginal 说成“完整保留了 source coupling”。

**第二，凸组合不等于恢复完整 coupling。**

\[
q_\lambda=\lambda q_{\rm src}+(1-\lambda)q_{\rm qry},\quad 0\leq\lambda\leq1
\]

是合法 PSD 代理，但不自动比任一项更接近 q_full。它给两类交叉项不同权重，并遗漏 query 和 source 都不同的项。精确分组目标下，对角项的总权重为 1，不能笼统称其“必然重复计算”。

**第三，直接平均因子还有尺度规范问题。** 同一个 K 可以写成 (cA)⊗(G/c)。对 A/G 分别平均的结果会受这种任意规范影响。即使以后研究耦合，也必须明确两套度量的相对尺度。

**第四，从两项和再压缩为一个 Kronecker，在计算上可能便宜，但不因此成为 QER 最优。** 它改善的是对该混合代理的逼近，仍需实际补偿验证。本轮不做这个分支。

**第五，真实 RoPE 下附件的简单 query-only A/G 尚未闭合。** 不能把无 RoPE 的 q_t 直接拿出来，把其余 source 依赖全部塞进一个标量权重。这是本轮先做小审计的原因。

**第六，Sequence 比 query-only 好也不直接证明需要耦合。** 差距还可能来自分离近似、跨 query/head 项、估计噪声、正则化等；不能从性能排序唯一反推机制。

## 3. 固定对象、数据与阶段预算

| 项目 | 本轮约定 |
|---|---|
| 模型 | 绑定既有 K/O 增量运行的 teacher、tokenizer、配置和源码版本 |
| 精度 | teacher 原生 FP32；关闭 autocast/TF32；统计收缩 FP64 |
| 模块 | model.layers.{0,10,20,31}.self_attn.k_proj |
| 量化 | 直接复用对应 MXINT3 Wq，禁止重新量化 |
| rank | 后续候选始终为完整 W_K 上的 rank64，不按 head 分配 |
| 训练母集 | 原 N256 个冻结 WikiText-2 train 窗口，长度 L=2048 |
| 预测位置 | T=2047；模块位置包含 L 个位置，不能混用两个计数 |
| 标签 | 每窗口原有一整套 teacher 采样标签，sum-NLL；不重新采样 |
| 小审计窗口 | 原窗口编号 0、32、64、96、128、160、192、224，共8个 |
| 实现 pilot | 上述窗口中的 0、32；先 L20，再验收其余三层 |
| 既有验证集 | 同一16个 validation 窗口，本阶段仅读取既有结果 |
| 正式候选预算 | 若后续方法定义成立，再用完整 N256×1，与旧基线公平比较 |

8 个窗口用于机制和实现诊断，**不重新拟合一个 N8 方法去和 N256 基线比较**。每窗口一份采样标签允许计算下面的经验分数，但不足以精确估计总体曲率或区分文本覆盖与标签 Monte Carlo 噪声。

这些窗口不是独立文章。当前阶段不追加新标签、N128 分支、seed 扫描或新测试集评价。

阶段安排：

| 阶段 | 新计算 | 交付与结束位置 |
|---|---|---|
| A：资产与恒等式验收 | 2个训练窗口，四层 K | 正确性、资源可用性和耗时记录；失败则停止 |
| B：K 机制审计 | 扩展到固定8个训练窗口 | source/query 分组诊断、真实 RoPE 分离性、经过 X 后的误差 |
| C：候选数学冻结 | 根据 A/B 结果完成单一候选定义 | 在真实 RoPE/GQA 下明确 A/G、归一化、丢弃项和成本；不是自动参数搜索 |
| D：正式 QER 比较 | C 完成后，N256×1；4个新补偿 | 复用基线，新增4×16个实际 KL 评价点 |

**本版直接实施 A/B。C 是尚待结果支持的研究工作，D 是已固定的比较框架；不得把尚未定义的候选写成已可运行算法。**

## 4. 从精确 K score 定义审计量

令样本 c 的 teacher 采样 sum-NLL 为 ℓ_c。固定其他模块，x_u 是 K 的真实输入，R=E−C。定义

\[
S_c=\nabla_{W_K}\ell_c,\qquad
\widehat q_{\rm full}(R)=\frac1{2NT}\sum_c\langle S_c,R\rangle_F^2.
\]

此处是局部 teacher predictive Fisher 的采样二次形式，不是实际 KL；本轮不拟合或显式构造 H。

### 4.1 无 RoPE、单头：作为代数起点

\[
s_{tu}=q_t^\top k_u/\sqrt d,\quad
e_{tu}=\partial\ell/\partial s_{tu},\quad
\widetilde x_t=\sum_u e_{tu}x_u.
\]

\[
S_K=\frac1{\sqrt d}\sum_{t,u}e_{tu}q_tx_u^\top
=\sum_u g_{k,u}x_u^\top
=\frac1{\sqrt d}\sum_tq_t\widetilde x_t^\top.
\]

其中 g_{k,u}=Σ_t e_{tu}q_t/√d。source/query 两种精确梯度相同，不存在谁更准确。

设 z_t=Σ_u p_{tu}v_u，δ_t=∂ℓ/∂z_t，则

\[
e_{tu}=p_{tu}\,\delta_t^\top(v_u-z_t),\qquad
\sum_u e_{tu}=0.
\]

因此对任意与 u 无关的向量 μ_t，

\[
\widetilde x_t=\sum_u e_{tu}(x_u-\mu_t).
\]

K 的有效变量包含带正负号的竞争与抵消；不能把它当作 V 的概率加权均值。本轮只检验这一恒等式，不把去中心化另立为新方法。

### 4.2 真实 RoPE/GQA：正式实验使用的表达

query head a 使用 KV head b(a)。令 P_b 把 d 维向量嵌入完整 K 输出的第 b 个块。使用实际实现的 RoPE 线性映射 Ω_t：

\[
\widetilde q_t^a=\Omega_tq_t^a,\quad
\widetilde k_u^b=\Omega_uk_u^b,
\]

\[
f_{tu}^a=\frac{e_{tu}^a}{\sqrt d}\Omega_u^\top\widetilde q_t^a,
\qquad
S_c=\sum_{t,u,a}P_{b(a)}f_{tu}^a x_u^\top.
\]

e 来自真实 causal attention 的 softmax 梯度；被 mask 的位置严格不贡献。RoPE 转置按实际 forward 的线性算子实现，不把近似浮点旋转的逆、转置和重新构造的位置参数混为一谈。

GQA 本身不破坏单个 query head 的无 RoPE 外积分解；它要求把多个 query head 的贡献放回同一 KV 权重块，并处理 head 之间的交叉项。

定义完整权重尺度的 source/query 梯度块：

\[
S_u^{\rm src}=g_{k,u}x_u^\top,
\qquad
M_t=\sum_{u,a}P_{b(a)}f_{tu}^a x_u^\top.
\]

\[
\sum_u S_u^{\rm src}=\sum_tM_t=S_c.
\]

这里 M_t 已汇总所有 query heads，保留同一个 query 内的跨 head 项。不能未经声明改成先对每个 head 平方、再求和。

### 4.3 两个分组代理及其区别

\[
\widehat q_{\rm src}(R)=\frac1{2NT}\sum_{c,u}\langle S_{c,u}^{\rm src},R\rangle_F^2,
\]

\[
\widehat q_{\rm qry}(R)=\frac1{2NT}\sum_{c,t}\langle M_{c,t},R\rangle_F^2.
\]

source 先汇总读取同一个 key 的所有 query/head，然后平方；query 先汇总一个 query 的所有 key/head，然后平方。两者一般互不支配，也都不是 q_full 的单调下界或上界。

q_full−q_src 是被删去的 source 间交叉项总和；q_full−q_qry 是被删去的 query 间交叉项总和。差值可以为负，不能解释为“丢失了多少非负能量”。

再由这些分组对象构造单 Kronecker，还需额外近似。附件无 RoPE、单头的

\[
A_{\rm qry}=\mathbb E[\widetilde x_t\widetilde x_t^\top],
\qquad G_{\rm qry}=\mathbb E[q_tq_t^\top]/d
\]

是这种额外近似的候选，**不是本版真实 Llama 已冻结的方法**。

## 5. 阶段 A：数值恒等式与缓存验收

在窗口0、32、四层 K 上逐项验收：

1. teacher、文本、mask、位置参数、采样标签、sum-NLL 归一化与父实验完全匹配。
2. 局部重放前向与冻结 teacher 相符；没有换 attention backend、丢失残差或重新采样标签。
3. source score `g_k.T @ x` 与新的完整反向 `∂ℓ/∂W_K` 相符。
4. 由 e、RoPE、GQA 重新组装的 score 与 source score 相符；同时检查 reconstructed g_k 与已缓存 g_k。
5. 检查 e 的 softmax 恒等式与有符号行和；零梯度行、全 mask 等异常单独记录。
6. 使用相同原生数值张量、FP64 收缩，检查两种求和组织的一致性。

数值门限沿用历史量级：FP32 与真实 autograd 相对 Frobenius 误差≤1e−5；纯 FP64 相同项重排≤1e−10。报告参考范数、绝对误差和最大元素误差。对于接近零或严重抵消的参考，使用预先记录的绝对误差尺度，不以不稳定相对误差直接判成机制失败。

纯重排的抵消安全尺度可用 Σ_{t,u,a}||f_{tu}^a||₂||x_u||₂；该尺度只用于零值情况的数值验收，不能用来掩盖非退化 score 与 autograd 不符。

实际 FP32 softmax 中，中心化恒等式可有舍入误差。先核对 softmax backward 的实现、张量精度和 mask；不能把误差归咎于理论公式后放宽阈值。

新增完整反向仅用于两个 pilot 窗口的正确性参考，四层共享一次连接完整的计算图。不要按模块或按 head 各做一次全模型反向。

若数值验收失败，停止后续审计。先修复身份、梯度或数学实现；不得带着错误做低秩性分析。

## 6. 阶段 B1：分组究竟改变了什么

### 6.1 固定残差方向

每层取5个已有、冻结的部署残差：

1. E=W0−Wq；
2. R_A-only；
3. R_Marginal；
4. R_Token-joint；
5. R_Sequence-one-step。

补偿残差统一按 `W0 − W_deploy` 在 FP64 中相减，明确其对应实际部署。不能一部分用理论 C64，另一部分用 FP32 部署残差。

四层×五方向×八窗口：每个方向记录 full/source/query 三种二次分数及逐位置 signed contraction。它们共享同一文本和标签，方便配对分析。

### 6.2 无需形成每个 M_t

定义

\[
d_{tu}^a(R)=\frac1{\sqrt d}(\widetilde q_t^a)^\top\Omega_u R^{b(a)}x_u,
\quad a_{tu}^a(R)=e_{tu}^a d_{tu}^a(R).
\]

则 query contraction 是 Σ_{u,a}a_{tu}^a，source contraction 是 g_{k,u}ᵀRx_u；两边再求和都等于 ⟨S,R⟩。按 head/query 分块做 Q 与旋转后 Rx 的收缩即可，不落盘 L 个完整 d×n 或 m×n 矩阵。

一次获得 e 后复用全部5个方向；先计算 Rx，再进入 head 循环，不对每个 pair 重算矩阵乘法。所有归一化固定为 1/(2NT)。

### 6.3 报告与解释

- 三种分数的绝对值、逐窗口分布、相对 E 的剩余比例；分母接近零时报告 NA。
- 同层同 rank 的4个已有补偿，在三种分数下的排序与配对分数差。
- q_full−q_src、q_full−q_qry 的带符号差值；注明它们只对应当前8份采样。
- 旁列旧16窗口的实际 KL 与恢复率，清楚标注训练侧诊断和验证侧预测损伤不是同一指标。

不因8窗口经验 q_qry 更接近 q_full，就声称新 A/G 必然有效；也不因二次排序与旧 KL 排序不一致就直接否定结构假设。有限样本、局部近似及训练/验证差异尚未被隔离。

## 7. 阶段 B2：RoPE 的破坏有多大，以及 X 是否改变结论

### 7.1 正式定义分离性对象

固定 c,t,a，把所有有效 source 的贡献排成列：

\[
F_t^a=[f_{tu}^a]_{u\le t}\in\mathbb R^{d\times(t+1)}.
\]

其 source 分离秩为 rank(F_t^a)。计算

\[
\rho_k(F)=\frac{\sum_{i=1}^k\sigma_i(F)^2}{\|F\|_F^2},
\quad k\in\{1,2,4,8\}.
\]

F=0 时 rho 未定义，记录零能量项，不设为1。不跨 heads 平均 F 后再算秩。

查询位置固定为0基编号：127、383、639、895、1151、1407、1663、1919；四层、八窗口、全部32 query heads，合计8192个小谱对象。每个对象只需 F Fᵀ 的128×128特征分解；不做128×2048的完整 SVD。

另外在 pilot 检查 t=0 和末位置的零梯度/因果行为，不把这些可能退化的点并入非零分离性统计。

同时记录 ||F||²、rho 的中位数/分位数，以及聚合能量比 Σ||F_k||²/Σ||F||²，避免大量低能量样本支配平均值。这些 query 是固定的诊断位置，不把其未加权均值当作所有位置的无偏估计。

### 7.2 无 RoPE 的代数对照

在同一组固定 e 和 q 上，定义仅用于核验的

\[
F_{t,0}^a=q_t^a[e_{tu}^a]_u/\sqrt d.
\]

非零时应为 rank1。它验证分离秩计算和“旋转使左方向依 source 变化”的代数判断。

**这不是去掉 RoPE 后重新运行的模型，也不是无 RoPE 模型的曲率。** 不比较它的实际 KL，不用它替代正式因子。

### 7.3 必须经过 activation 再检查

令 X_{≤t} 的各行为 x_uᵀ：

\[
M_t^a=F_t^aX_{\le t},\qquad
\widetilde M_{t,k}^a=(F_t^a)_kX_{\le t}.
\]

F 的低秩能量集中，不保证经过 X 后仍然准确。为控制成本，固定子集上计算上述量：

- query 只取383、1663；
- 每个 KV group 取一个 query head，a=4b+(j mod 4)，b=0,…,7；j为审计窗口列表中的序号；
- 四层×八窗口×两个query×八heads，共512个对象；
- 仅计算 k=1、4 的近似误差；其他 k 只用于谱摘要。

预期 head 映射 a//4 必须先由实际模型验证。配置不同则修改并重新冻结协议，不硬编码错误映射。

报告

\[
\epsilon_{M,k}=\sqrt{\frac{\sum\|M_t^a-\widetilde M_{t,k}^a\|_F^2}{\sum\|M_t^a\|_F^2}}.
\]

并对第6节五个冻结 R 的对应 KV 块，报告

\[
\epsilon_{R,k}=\sqrt{\frac{\sum\langle M_t^a-\widetilde M_{t,k}^a,R^{b(a)}\rangle_F^2}{\sum\langle M_t^a,R^{b(a)}\rangle_F^2}}.
\]

分母接近零时同时给出绝对误差并标为不可稳定归一化。这里先按对象平方再聚合，是局部贡献诊断，不是完整 q_full 的近似误差证书。

F_k 可由 F Fᵀ 的前 k 特征向量 U_k 构成：F_k=U_kU_kᵀF。实现时用 U_k(U_kᵀF X)，避免生成所有大矩阵。只保留必要的当前对象，不保存整批 M。

### 7.4 本阶段结论边界

- rho1高、经过X和固定R后误差也小：支持研究“每query/head少量可分离项”的具体近似。
- rho1高但经过X后失真大：仅看 RoPE pair 的谱不足以指导 QER。
- rho1低但rho4等较高：说明rank1表达受限，不代表单 Kronecker 度量不可用。
- 全部低：当前 query 外积压缩动机变弱；不能据此宣布所有 K 方法无效。
- 无论哪一种，均未证明某套 A/G 在真实 KL 下优于 Marginal。

**不设置“rho1>某阈值就自动启动N256”的硬门。** 低秩分解还存在各项缩放规范、跨项/跨head舍弃和统计成本问题；这些属于阶段C必须解决的定义，不能由一个rho阈值代替。

## 8. 资源复用：已核查的本地代码与远端边界

2026-09-21 本地存在且已阅读：

- `experiments/qer_multimodule_three_fp64_v1/`：原 q/v/down 共享采集、数据、标签及基线格式。
- `experiments/qer_ko_increment_v1/`：K/O 增量采集、原 down 梯度局部重放、K 四种基线及后缀评价。
- `experiments/qer_v_attention_proto_v1/`：真实 attention 捕获、由 down 梯度恢复 δ、FP64收缩及 V prototype。

以上路径相对于 `qera_mxint4_full_ag/`。**本次没有 SSH 远端盘点；源码存在不等于服务器缓存仍完整。旧 README 中的授权只属于旧任务，不构成本轮自动启动服务器任务的指令。**

令 BASE_RUN 为原12模块运行，KO_RUN 为K/O增量运行。K新资产在KO_RUN，原文本、标签、QKV输入与down梯度在BASE_RUN；不要只寻找一个父目录。

| 资源 | 已知位置/读取接口 | 本轮用途 |
|---|---|---|
| 文本与索引 | BASE_RUN/data，原 Source reader | 8个训练窗口及身份绑定 |
| 冻结标签 | BASE_RUN/cache/labels/<sample_id>.safetensors | 不重新抽样 |
| QKV输入 | BASE_RUN/cache/x/<slug(qkv_input)>/wXXXX.safetensors | K输入与重放核对 |
| down输出梯度 | BASE_RUN/cache/g/<slug(down)>/<sample_id>.safetensors | 恢复局部下游梯度 |
| K输出梯度 | KO_RUN/cache/g/<slug(k_proj)>/<sample_id>.safetensors | source score与恒等式参考 |
| K量化资产 | KO_RUN/quantized/<slug(k_proj)>.safetensors | W0/Wq/E与哈希 |
| 四种K补偿 | KO_RUN/modules/<slug(k_proj)>/corrections/N256__<method>.safetensors | 五方向中的四个部署残差 |
| 原始因子 | KO_RUN/modules/<slug(k_proj)>/factors/N256/<method>/raw.safetensors | 来源核验，不重拟合 |
| K逐窗口KL | KO_RUN/scores/validation/wXXXX/<slug(k_proj)>___<candidate>.json | 4×5×16=320条已有记录，含None |

slug、receipt和identity使用冻结代码的定义，不手工改文件名或旧receipt。只校验本轮必需资产；不遍历复制全部数百GiB历史缓存。

已有 cache 通常没有持久化 e、query RoPE张量及本轮F/M诊断。已有 K g 和 A/G 不能反推出这些量，仍需少量前向和局部重放。

### 8.1 首选重放路径

对每个审计窗口，仅一次完整 teacher 无梯度前向，捕获四层所需的真实调用参数和中间变量，按层/head流式处理。

由于 block 输出为 r+MLP(RMSNorm(r))，已有 down 输出梯度 Λ 等于该 block 输出梯度。用一次局部 MLP VJP 得到

\[
\lambda=\Lambda+J_{\mathrm{MLP}\circ\mathrm{RMSNorm}}(r)^\top\Lambda,
\qquad\delta=W_O^\top\lambda.
\]

由真实 p、v、z 与 δ 计算 e。无需为每个head反向贯穿后续所有层。使用 `v_capture.local_delta` 的思想，但增加真实Q/RoPE捕获和K验证；不能直接把V的statistics函数改名复用。

qkv归一化输入不能还原完整残差r，r必须从真实前向捕获。K/O的 `ko_replay` 提供另一条已实现的完整局部block VJP核验路径。

若 down 缓存缺失，报告缺失并重新估算完整共享反向成本；不得默认重建全部N256。若 K 缓存缺失，只重算8窗口需要的K梯度，标记新来源。

### 8.2 内存、磁盘与时间

- teacher按原方案跨两张4090部署。不要在两卡各加载一份完整FP32 8B模型。
- 统计按head/query块处理，缓存p/e/RoPE中间量只保留当前窗口；不要积累8窗口attention。
- FP64离线收缩/小谱可在释放teacher后用两worker运行；模型常驻时需预留显存，不盲目并发两份大工作集。
- 不存完整H、不存全部M，不重跑N256旧基线，不补N128，不重算320条旧KL。
- 默认增量输出上限8GiB，另保留临时空间；仅存选定小对象、标量摘要及receipt。不复制父缓存。
- pilot必须分列：模型加载、前向捕获、局部VJP、五方向分组收缩、小谱、经过X诊断、I/O与峰值显存。
- 报告总耗时估计采用实测pilot外推，加载单独计时；样本数量小不意味着FP64收缩必然便宜。
- 本轮机制审计采用2小时累计活跃计算上限作为防失控边界。到上限原子保存并报告未完成项，不删减预设四层或悄悄降低精度；不自动转入N256。

成本规模是8次共享前向、32次局部MLP VJP及两个pilot完整反向参考，加离线收缩。不是8×4×heads次全模型反向。8192个小谱和512个经过X的对象仍需计时，不能只按forward速度估算。

## 9. 阶段 C：新候选必须先填写的数学清单

阶段B结束后，用一页说明确定是否继续某一个query-only候选。至少完整给出：

1. 真实RoPE/GQA下，被压缩的精确对象是什么。
2. A和G的具体统计公式、张量维度及计数归一化。
3. source位置、query位置、head三个轴分别如何汇总，哪些交叉项被删除。
4. 若使用SVD分解F或M，如何固定各外积项的缩放规范；不得认为精确外积分解唯一决定Marginal因子。
5. G是否KV-block-diagonal、是否保留跨head块；“完整W上求解”不等于G含完整跨head信息。
6. 能否在N256预算内获得统计；小审计抽取query的诊断不能直接替代正式全位置估计。
7. 与无RoPE单头候选的关系：严格退化一致，还是引入了额外权重，必须写清楚。
8. 固定一个全层通用方案，不根据四层validation选择不同构造、分离秩或阻尼。

这份清单完成前，不建立伪造的 `Query-aware` 实验结果行。即使审计支持低分离秩，具体方法仍是研究结果，不是执行脚本可以自行补全的细节。

本轮不耦合source/query；不通过拟合其Kronecker和来绕过上述定义。

## 10. 阶段 D：后续正式比较的冻结框架

只有阶段C闭合后，新增一套K候选。公平条件继续固定：N256、每窗口原有1份标签、rank64、同16窗口validation、同一teacher/Wq；四层均报告。

比较 A-only、Marginal、Token-joint、Sequence-one-step 和新候选；仅量化None作为恢复率分母。旧5类记录直接复用，新增4个补偿、64个主要KL点；首窗口各候选部署做完整前向核验，另有teacher reference/self-KL检查。

原始A/G必须PSD；阻尼、白化、逆根、SVD和FP32部署规则继承父配置，通常为trace-relative η_A=η_G=1e−3，执行前核对实际冻结值。求解完整W_K的一个rank64问题：

\[
C_{K,64}=G^{-1/2}[G^{1/2}EA^{1/2}]_{64}A^{-1/2}.
\]

部署按父路径 `Wq + float32(C64)`，实际KL评价真实部署矩阵；不擅自换成FP32因子先相乘。记录理论补偿与部署残差的差异。

主要报告绝对KL与

\[
\mathrm{Recovery}=100\left(1-\frac{\mathrm{KL}_{candidate}}{\mathrm{KL}_{None}}\right),
\quad
\Delta_M=100\frac{\mathrm{KL}_{Marginal}-\mathrm{KL}_{new}}{\mathrm{KL}_{None}}.
\]

同时报告相对Marginal剩余KL降低比例。L0原97.24%恢复率只剩2.76pp绝对空间，但1pp改善对应约36%的剩余KL降低；L0不能被预设为“应该没有收益”的负例。

逐窗口配对不确定性作为探索性报告，bootstrap重采样后重新计算KL均值之比；不能平均逐窗口恢复率替代主指标。16窗口并非独立文章，且验证集已被多轮使用，不能称作新的独立泛化证明。新候选冻结后若需要正式泛化声明，再单独定义尚未用于开发的test；本轮不自动启动它。

不要用统一2pp门限把约1.8pp的L20既有差距直接判成无意义。先报告每层绝对效果、剩余KL变化和配对区间；“接近Sequence”若要作为正式结论，需事先另定与问题尺度一致的等效界限。

结果解释保持有限：新候选优于Marginal支持其QER有效性；接近Sequence支持它以较简单结构达到类似效果，不能唯一证明Sequence收益来自query竞争。新候选失败也只否定该具体构造。

## 11. 交付文件与执行约束

建立独立新run，旧run只读。最少交付：

```text
manifest.json
parent_assets_audit.json
frozen_probe_plan.json
acceptance/gradient_identity.json
acceptance/rope_gqa_mapping.json
diagnostics/grouped_scores.csv
diagnostics/separability.csv
diagnostics/post_activation_error.csv
diagnostics/residual_direction_error.csv
resources/timings.json
resources/peak_memory.json
READOUT.md
```

READOUT必须依次回答：

1. K score的source/query/RoPE/GQA恒等式是否通过？
2. 固定残差上两种分组改变了哪些分数，变化是否由少数窗口主导？
3. RoPE分离性在高能对象上怎样，乘X及作用于实际残差后是否仍然成立？
4. 哪些结论受8窗口×1标签限制？
5. 目前是否足以**定义**一个候选，尚缺哪一步数学定义？
6. 实际用了多少共享前向、完整反向、局部VJP、时间与新增磁盘？

不得将小审计完成标记为“新K方法已验证”，不得自动运行其他模块或N256正式候选。协议变更、新候选定义或样本计划变化必须另存版本并记录依据。

## 12. 本地可复用源码入口

- [K/O资产与梯度缓存](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_assets.py)
- [K/O局部重放与后缀评价](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_replay.py)
- [K基线拟合与借用输入统计](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_fit.py)
- [已有K逐窗口KL与结果格式](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_evaluate.py)
- [V attention捕获与局部delta恢复](../qera_mxint4_full_ag/experiments/qer_v_attention_proto_v1/v_capture.py)
- [原共享teacher反向](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/shared_teacher.py)

这些是复用入口，不是要求改写旧实验。新增K审计实现应隔离文件、输出及identity，保持历史结果可复现。
