# K Sensitivity-weighted Marginal：实验协议与资源复用审计

版本：v1.0，2026-09-21。平台：双 RTX 4090。

状态：已核查本地源码、已有服务器存储审计快照和资产格式；本轮尚未实现、未SSH实时验收、未启动计算。旧K audit与Direct Query实验已经完成，不重跑。本协议只新增一个Sensitivity-weighted Marginal候选。

## 1. 唯一主问题

> 在相同N256、rank64和实际teacher-KL评价条件下，仅把输入二阶矩改成真实K梯度强度加权、保留原完整Marginal G，能否优于标准Marginal-K，并获得接近已有Token-joint的补偿效果？

本轮研究最终QER效果，不优化全局Fisher拟合，也不要求先证实所有方向上的条件协方差假设。

主比较：Marginal vs Sensitivity-weighted Marginal。

已有A-only、Token-joint、Sequence-one-step及None作为参照，原样复用。已完成Direct Query可在结果表中作为历史附列，不新增其计算，也不改变本轮主比较。

不加入rank1、ALS、Full-fit、coupled-K、RoPE新构造、head-wise rank、其他模块、N128、标签重采样或层选择器。

## 2. 候选的完整定义

### 2.1 理论动机及适用边界

固定teacher上下文c和source位置u，令g_{c,u}为原teacher独立采样各预测位置标签、对sum-NLL反向得到的真实k_proj输出梯度。定义

\[
B_{c,u}=\mathbb E_y[g_{c,u}g_{c,u}^{\top}\mid c].
\]

若条件二阶矩主要改变强度：

\[
B_{c,u}\approx s_{c,u}G_0,\qquad s_{c,u}\ge0,
\]

则对文本和模块位置取统一平均，有

\[
\mathbb E[xx^\top\otimes B]
\approx
\frac{\mathbb E[sxx^\top]}{\mathbb E[s]}
\otimes G_{\rm marg},
\qquad G_{\rm marg}=\mathbb E[B].
\]

由于E_y[||g||²|c]=tr(B)，可以用已有采样梯度估计强度。该推导针对token位置对角统计，不声称恢复完整sequence交叉项。假设作用于条件二阶矩B，不是要求单次rank1的ggᵀ与完整G成比例。

### 2.2 本轮冻结的经验统计

每层独立，N=256、L=2048、T=2047。x_{c,u}∈R^{4096}，g_{c,u}∈R^{1024}，所有模块位置u=0,…,L−1均按原缓存使用。

\[
w_{c,u}=\|g_{c,u}\|_2^2,
\quad U=\sum_{c,u}w_{c,u}x_{c,u}x_{c,u}^{\top},
\quad D=\sum_{c,u}w_{c,u},
\]

\[
\boxed{A_{\rm sens}=U/D,\qquad G_{\rm sens}=G_{\rm Marginal}.}
\]

G的首选资产是KO_RUN该模块`statistics/N256.safetensors`中的原始`G_canonical`：

\[
G_{\rm canonical}=\frac1{NT}\sum_{c,u}g_{c,u}g_{c,u}^{\top}.
\]

它相对按NL平均的G只差L/T正标量；不改变当前trace-relative阻尼下的加权SVD解。保存这个归一化，不把T和L混用。

**使用完整1024维g的平方范数，复用完整1024×1024 G。** 不按head计算独立w，不把G换成Direct Query的KV块对角G，不对白化后的g取范数。

所有256窗口的U、D全局累计后只除一次。禁止先将每窗口权重归一化再平均；禁止clipping、开方权重、按token/head筛选或新增温度。末位置若原g为零，自然不贡献。

必须D>0且有限，否则停止该模块并报告退化原因。使用FP64计算w与累计U、D。

### 2.3 只改变A，明确处理历史因子的尺度规范

历史`factors/N256/Marginal/raw.safetensors`中的`G_raw`不一定数值等于`G_canonical`。当前代码的gauge操作为

\[
(A,G)\mapsto(A/\|A\|_F,\ \|A\|_F G).
\]

因此优先读取原始统计中的G_canonical，并核对旧G_raw与之相差正标量、归一化形状一致。

若原始统计文件缺失而旧Marginal raw因子完整，可复用旧G_raw作为等价的全局尺度版本，记录该分支及比例；不得把阻尼后的G_solve当原始G再阻尼一次。若无法确定来源或正标量关系，先解决资产身份，不默默换G。

新候选保存`A_raw=A_sens`和所选原始G；可直接调用原求解器，无须为了沿用旧gauge而改变G文件。若内部进行gauge，另存求解时尺度，不能声称gauge后的tensor哈希仍与父G相同。

## 3. 与现有Token-joint的关系：已从源码核实

当前`token_step`的同步A更新为

\[
A_{new}=\frac1{NT\|G_{old}\|_F^2}
\sum_{c,u}(g_{c,u}^{\top}G_{old}g_{c,u})x_{c,u}x_{c,u}^{\top}.
\]

当前Token-joint初始化G_old=I，因此第一轮

\[
A_{first}=U/(NTm),\qquad m=1024.
\]

故新A与第一轮A形状相同，仅整体尺度不同。但Token-joint同时更新G并继续迭代，本候选保留Marginal G，**不是复用完整的Token-joint第一轮因子对**。

实际研究定位：检查Token-joint中最简单的A侧强度加权，是否足以获得实际QER收益。不把这个更新本身包装为新的统计技巧，也不声称它是K专属公式。

## 4. 固定实验设置

| 项目 | 固定内容 |
|---|---|
| 模型 | 与现有KO_RUN完全相同的teacher、tokenizer、配置、数值精度和源码版本 |
| 模块 | L0/L10/L20/L31的k_proj，共4个 |
| Wq | 直接复用KO_RUN对应MXINT3量化权重，不重新量化 |
| rank | 完整1024×4096权重上的rank64，不按head分配 |
| 统计数据 | 原N256个非重叠2048-token训练窗口 |
| 标签与梯度 | 原每窗口1整套teacher采样标签、真实sum-NLL梯度；不重新采样、不换ground-truth CE |
| 输出侧度量 | 每层原完整Marginal G，来源与哈希记录 |
| 新统计 | 每层仅新算一个A_sens |
| 验证 | 原16个validation窗口，T=2047，实际KL(teacher || candidate) |
| 干预 | 每次只替换一个K，其余权重保持teacher FP32 |
| 求解 | FP64矩阵根与SVD，原trace-relative阻尼，预期η_A=η_G=1e−3 |
| 部署 | 原路径Wq+float32(C64)，不改为两个FP32因子先相乘 |
| 新候选数量 | 4层各1个，共4个；主要新增KL点为4×16=64 |

非重叠窗口不等于独立文章。单份标签提供有噪声的强度估计；本轮不追加采样数分支。

## 5. 资源审计：已掌握的证据与当前边界

### 5.1 本地审计结论

已阅读并核对：

1. 原12模块采集和拟合源码；QKV输入按同层共享，原生FP32缓存；统计FP64。
2. K/O增量源码；K输入借用原QKV缓存，K梯度另存KO_RUN；四种基线、量化资产和逐窗口KL均有明确路径。
3. Token-joint更新及进度保存源码；每轮覆盖`progress.safetensors`，不是逐轮永久保留。
4. Direct Query源码；其已完成结果可以引用，其attention收集、G及audit依赖不应带入本轮。
5. 本地保存的2026-09-21 10:05–10:21服务器只读审计报告；当时确认原BASE_RUN缓存和部分张量schema，未覆盖后来K/O及Direct Query结果的完整性。

历史快照显示原QKV输入FP32形状[1,2048,4096]、每文件32MiB。该记录证明当时存在，不证明现在仍可读，也不是本次实时SSH验收。

**本轮尚未连接服务器。以下复用路径已由源码核实；实际文件内容、哈希和完成状态须按本节逐项验收，不能直接写成已验证完整。**

### 5.2 三个运行目录的职责

- BASE_RUN：`qera_runs/multimodule_three_fp64_v1/run_01`，提供冻结文本、标签、QKV输入及身份链。
- KO_RUN：`qera_runs/ko_increment_v1/run_01`，提供K梯度、K的原Marginal统计、四种补偿与KL。
- DIRECT_RUN：`qera_runs/k_direct_query_v1/run_01`，可选读取已有Direct Query结果；不是新方法的统计依赖。

这些是已知默认相对路径。执行时从实际manifest绑定父目录，不因名字为run_01就认定身份相同。slug与sample_id使用冻结reader，不手工猜测。

### 5.3 逐项复用清单

| 资源 | 路径或来源 | 复用方式 / 验收要点 |
|---|---|---|
| 冻结文本/索引 | BASE_RUN/data/index.json、freeze.json、fit与validation文件 | 相同256/16窗口、2048长度、token hash和顺序 |
| 原标签 | BASE_RUN/cache/labels/<sample_id>.safetensors | 只用于梯度来源绑定，无需重新计算loss |
| 四层QKV输入 | BASE_RUN/cache/x/<slug(qkv_input)>/wXXXX.safetensors | 每层256份；确认K与同层Q共享输入，形状/dtype/hash |
| 四层真实K梯度 | KO_RUN/cache/g/<slug(k_proj)>/<sample_id>.safetensors | 每层256份；必须是K，不是Q、V或down梯度；绑定标签、sum-NLL和x_hash |
| 原始完整G | KO_RUN/modules/<slug(k_proj)>/statistics/N256.safetensors，key=G_canonical | 首选直接复用；四层各1份 |
| 原A与规范记录 | 同文件A_m及statistics/N256_A.safetensors | 只核验旧gauge/归一化；不能替代A_sens |
| Marginal raw factors | KO_RUN/modules/<slug(k_proj)>/factors/N256/Marginal/raw.safetensors | 核对G形状与比例；必要时使用等比例G_raw |
| 量化/teacher模块权重 | KO_RUN/quantized/<slug(k_proj)>.safetensors及freeze.json | W0/Wq、量化配置和teacher身份；直接构造E |
| 四种已有补偿 | KO_RUN/modules/<slug(k_proj)>/corrections/N256__<method>.safetensors | A-only、Marginal、Token-joint、Sequence-one-step；不重拟合 |
| 已有KL | KO_RUN/scores/validation/wXXXX/<slug(k_proj)>___<candidate>.json | 4层×5类×16=320条，含None；校验candidate freeze与token hash |
| Direct Query KL，可选 | DIRECT_RUN/scores/validation/wXXXX/<slug(k_proj)>.json | 可附列64条；必须匹配相同Wq、teacher、tokens，不重算 |
| teacher模型、环境 | 原manifest/model source/config | 只在最终KL阶段加载；离线A统计不需要完整模型 |
| 旧teacher logits | 旧评价代码仅在内存中缓存 | 不假定已落盘；最终16窗口重新生成reference即可 |

G、X、g能复用；新的加权A一般需要计算。**不能从A_m与G_marg两个平均矩阵反推出Σ||g||²xxᵀ。**

### 5.4 第一轮Token-joint A是否可直接复用

默认不能。源码只维护最新progress；完成三轮后，raw与progress通常都是最终状态，`iterations.json`中的谱摘要也不能重建第一轮矩阵。

若实际远端另有第一轮完整快照，则只有在其明确绑定同一N256、G_old=I、iteration=1、同步更新和gauge记录后，才允许作为U的等比例来源。不得把最终Token-joint A当成第一轮A。

本协议默认采用可靠路径：**从原X/g重算一次U、D**。若发现可复用第一轮快照，可作数值交叉核验；不以寻找快照延迟主实验。

### 5.5 不需要的资产与禁止的错误复用

- 不需要down梯度、attention probabilities、e、q、RoPE张量、V有效输入或sequence S。
- 不需要重跑K audit或Direct Query；新的资产reader不要沿用Direct Query对audit完成目录的强制依赖。
- 不复用Direct Query G、三轮Token-joint G或CE语料标签的旧G。
- 不复用原A的矩阵根、原加权残差的SVD，因为A已改变。
- 原G的根只有在原始尺度、阻尼及数值版本完全相同且确有持久化时才可复用；本地源码不保证存在，默认重新分解这个1024×1024 G即可。

## 6. 开跑前的只读资产验收

输出`parent_assets_audit.json`，对每项标记：verified / missing / identity_mismatch / optional_not_used。

1. 绑定BASE_RUN和KO_RUN的manifest、data freeze、源码/模型配置及teacher身份；检查四模块W0和当前评价teacher匹配。
2. 核对1024份X、1024份K梯度及其receipt、样本、label_hash、x_hash和原提交清单。检查全部张量有限、shape/dtype匹配。
3. 核对4份G_canonical、旧Marginal raw因子和16份基线补偿。
4. 核对320条旧KL来自相同16窗口，KL方向、预测token数和candidate freeze一致。
5. 可选Direct Query只读取匹配的新方法结果与必要身份记录，不触发其收集流程。

对40GiB大缓存，优先边验证边统计、保存原子提交；文件哈希可在一次加载时计算并用于后续恢复。不要先全量读取校验一次、正式计算又读一次、每个阶段再重新哈希全部目录。

父目录只读，不改receipt或旧identity。恢复时核对新检查点以及已验证资产的身份/变更记录；检测变化必须重新验证，不能以只看文件名代替内容绑定。

缓存缺失时只报告具体文件和受影响模块；本协议不自动重跑N256 teacher backward。可以继续完成不依赖缺失项的审计和离线模块，但最终不能把不完整四层结果标记为正式完成。

## 7. 执行流程：完全离线的统计与求解

### 7.1 最小数学pilot

使用原训练窗口0、1，先L20，再检查其余层的读取/绑定。无需加载teacher、无需新增autograd。

在预先固定小切片上核验：

- 直接Σw·xxᵀ与矩阵表达Xᵀdiag(w)X一致；
- 使用(X·sqrt(w))ᵀ(X·sqrt(w))等价计算只是一种实现，不是把定义中的w改为sqrt(w)；
- G_old=I时，当前token_step的A更新与加权A只差整体正标量；此检查可用小维度纯代数，不运行完整三轮Token-joint；
- 权重全为同一正常数的代数对照退化为普通输入Gram。

FP64相对误差门限1e−10，记录绝对误差和参考范数；不以接近零对象的不稳定相对误差判失败。pilot的正式全窗口累计须计数一次，不重复加入N256。

### 7.2 一遍统计

对每层、每窗口，执行：

```text
X = cached_x.reshape(L, n).float64()
g = cached_k_gradient.reshape(L, m).float64()
w = sum(g * g, output_channel)
U += X.T @ (X * w[:, None])
D += sum(w)
```

统计时顺便保存每窗口Σw、Σw²、最大w、零权重数量和是否有限；由此报告权重集中程度。可给出D²/Σw²，但称为权重有效数量，不是独立文本/梯度样本量，不据此调权重或追加实验。

完成256窗口后：A_sens=sym(U/D)。保存U、D、计数和A_sens。至少检查

\[
\boxed{D/(NT)\approx\operatorname{tr}(G_{\rm canonical}).}
\]

两者使用同一梯度缓存，FP64容差1e−10量级；这是一项廉价的全数据归一化/错梯度检查。若使用了等比例G_raw，先按审计比例还原比较，不直接混用trace。

不要为了重新证明G相同再做一遍完整gᵀg统计。其内容来源与hash依靠父资产验收，trace关系提供额外一致性检查。

双4090离线阶段各一个worker，队列包含四个模块。该阶段不加载完整teacher，因此显存主要用于X、g和4096×4096 A累加器。

按模块保存当前和上一份完整累计检查点，记录已计入窗口集合、父资产hash和配置identity；不要为256窗口各永久保存一份A矩阵。中断后不重复计数。

### 7.3 四次全矩阵求解

沿用父solver，对原始A_sens与完整G施加相同trace-relative阻尼，得到A_solve、G_solve：

\[
C_{64}=G_{solve}^{-1/2}
[G_{solve}^{1/2}EA_{solve}^{1/2}]_{64}
A_{solve}^{-1/2}.
\]

沿用父PSD容差、condition≤1e8、矩阵根/逆根、SVD和部署误差检查。禁止因某层效果差而改阻尼、rank或谱截断。

保存原始与正则化因子、P64/Q64、部署权重及audit。部署为`Wq+float32(P64@Q64)`，其中乘积在FP64形成；实际评价使用真实W_deploy。

四套候选和配置hash全部冻结后，才进入validation。

## 8. 评价：复用基线，只增加64个主要KL点

加载原teacher跨双卡部署，禁用autocast/TF32；每窗口teacher reference只生成一次，供四个新候选复用。每次只替换一个K，结束后恢复原权重并核验。

主要新增4×16=64个实际KL点。不增加小α曲线、q_full估计、全模型联合替换或test/PPL任务。

采用已验证的Direct Query完整前向评价结构，但改用新候选reader，**不调用其旧收集runner或强制audit依赖**。本轮无需新增后缀缓存优化。

首validation窗口按已有路径核验四层×5旧类=20个基线点，并对四个新候选各重复一次；这些24个是数值复验，不是新方法或额外统计预算。

参考self-KL≤1e−10；旧KL复验/重复容差沿用max(1e−12,1e−7·|KL|)。失败时先核对环境、teacher和路径，不把不同运行的分数强行拼接，也不扩大容差掩盖偏差。

主报告包含：

- 每层绝对KL、仅量化KL和恢复率；
- 新方法相对Marginal恢复率变化pp；
- 新方法相对Marginal剩余KL降低百分比；
- 相对Token-joint、Sequence-one-step差距；
- 16窗口逐点KL和探索性配对区间；
- 权重集中程度与统计/求解成本，四层全部保留。

恢复率按汇总KL计算，不平均逐窗口比率：

\[
\mathrm{Recovery}=100(1-\overline{KL}_{candidate}/\overline{KL}_{None}).
\]

可沿用2000次固定seed配对窗口bootstrap，重采样后重新计算均值之比；16窗口非独立文章且已用于多轮开发，区间不包含训练统计随机性，不声称独立泛化证明。

不把低于2pp一律判成无意义；L20旧Token收益约1.8pp，L0剩余损伤很小，应同时报告绝对KL、pp与剩余KL比例。没有预先等效界限时，不因差异“不显著”就声称与Token相等。

## 9. 解释规则

| 结果 | 允许的解释 |
|---|---|
| 新方法优于Marginal，达到接近Token的点估计 | 简单A侧强度加权在当前预算和残差上有用；正式“等效”仍需相应证据 |
| 新方法有部分改善 | 此简化获得部分QER收益，不唯一归因于某种总体曲率结构 |
| 新方法无改善或退化 | 当前||g||²加权构造不足，不能直接证明梯度方向变化是唯一缺失因素 |
| 层间差异明显 | 四层完整报告，不事后增加层选择器或调权重 |

正结果不证明所有上下文上B=sG成立；负结果不能区分单次采样噪声、权重集中、方向依赖和残差相关性等原因。本轮不因此自动新增消融或复杂方法。

本构造数学上适用于其他Linear；当前只在K检验，不提前写成“K必须用强度加权、Q/O不需要”的结论。

## 10. 资源规模与开销控制

按当前预期shape，四层256窗口需要读取：

| 项目 | 数值体积/计算规模 |
|---|---:|
| 原X | 4×256×2048×4096×4 bytes = 32GiB |
| 原K g | 4×256×2048×1024×4 bytes = 8GiB |
| 主要缓存读取总量 | 约40GiB，不含小receipt/模型/基线文件 |
| 四层新的A | 4×4096²×8 bytes = 512MiB |
| 每窗口加权Gram | 一次4096维Gram，共1024次；不是只计算scalar |
| 新补偿求解 | 4次完整weighted SVD |
| 新评价 | 64个主要KL点，另24个复验及16个teacher reference |

与Direct Query相比，完全省去N256 teacher捕获、1024次局部MLP VJP、attention与DᵀD构造；但仍有40GiB I/O和FP64 Gram。不能沿用8窗口audit的30–90分钟估计，也不按旧完整12模块实验耗时比例硬推。

新输出软预算8GiB，建议至少16GiB实际可用增量空间，另遵循服务器既有共享存储预留要求；无需再复制40GiB缓存。不得为腾空间自动删除父实验资产。

前两个窗口记录读取/校验、转移、w与Gram时间，完成第一个模块后记录根/SVD时间；根据双worker实际并发和文件系统吞吐更新预计完成时间。只报告测量支持的估计，不因达到时间预估就修改样本量或精度。

## 11. 最小交付与完成条件

```text
manifest.json
parent_assets_audit.json
pilot/math_checks.json
statistics/<module>/raw.safetensors
statistics/<module>/weight_summary.json
statistics/<module>/progress/latest.json
corrections/<module>/sensitivity_weighted.safetensors
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

正式完成要求：四层完整N256、四套冻结补偿、64个有效新KL点、原320条基线来源可追溯、数值复验通过。保留所有层，负结果同样算实验完成。

READOUT先回答“只改A的强度加权是否改善Marginal的实际teacher-KL”，再解释与Token第一轮A的关系、有限样本边界和资源节省。不把全局Fisher拟合或audit重跑写成完成前提。

## 12. 本地已核查证据入口

- [历史服务器存储审计快照](../outputs/server_cache_audit_20260921/REPORT.md)：确认原BASE_RUN在审计时存在、QKV输入schema及旧进度覆盖策略；不是当前实时状态。
- [K/O输入、梯度与父资产reader](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_assets.py)
- [K/O拟合与原A/G资产继承](../qera_mxint4_full_ag/experiments/qer_ko_increment_v1/ko_fit.py)
- [原Marginal原始统计路径与归一化](../qera_mxint4_full_ag/experiments/qer_kronecker_l10_v2/experiment.py)
- [Token-joint的I初始化与覆盖式progress](../qera_mxint4_full_ag/experiments/qer_multimodule_three_fp64_v1/fitting.py)
- [token_step和FP32部署路径](../qera_mxint4_full_ag/experiments/qer_kronecker_l10_v2/sketch_math.py)
- [gauge、阻尼和weighted SVD](../qera_mxint4_full_ag/experiments/qer_teacher_kl_exp03/ag_math.py)
- [已验证的完整前向KL评价结构](../qera_mxint4_full_ag/experiments/qer_k_direct_query_v1/d_evaluate.py)
- [逐窗口配对结果与bootstrap结构](../qera_mxint4_full_ag/experiments/qer_k_direct_query_v1/d_report.py)

本协议不修改上述冻结源码。后续新增实现应独立建目录与identity，复用必要reader和solver，不继承旧实验的收集步骤、方法列表或任务扩展。
