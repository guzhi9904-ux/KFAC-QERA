# QER 本地实验第一步：重建 A、核验 Cholesky 求解、验证 teacher-KL 曲率

版本：v2.0，2026-09-18。状态：**方案已制定，本文所述新实验尚未实现或运行**。

本版替代 v1.0 的历史资产方案。全部统计、量化权重与补偿方向在本地重新生成；不要求取回旧服务器的 A、CE-G、Wq 或 GI/DG/GF 因子。已有结果只用于选择两个诊断模块，不作为本次数值参照。

## 1. 要完成什么，为什么按这个顺序

当前研究目标是理解固定 rank 下，什么信息能够帮助选择更好的量化残差补偿。第一步先建立可靠的局部损伤参照，暂不选择新的 G 结构。

本轮包含一个准备环节和一个科学实验：

| 环节 | 明确问题 | 本轮交付 | 通过后能说明什么 |
|---|---|---|---|
| 准备：A 与求解审计 | 本地收集的 A 是否正确？同一个求解目标下，Cholesky 能否替代对称平方根？ | 两个模块的原始 A、求解度量、分解及数值核验 | 新基线的统计和数值求解可信 |
| 实验一：teacher 曲率验证 | 在真实量化残差方向上，teacher 采样曲率能否预测小扰动 KL？ | 六个方向的曲率—KL 对齐、采样误差及运行成本 | 得到后续检验结构近似的可用参照 |

本轮不收集全模型 A/G，不重新采集旧 ground-truth CE-G，不开展 KFAC 优劣比较，不做 rank 分配，不跑全模型 PPL 或下游任务。**本轮会采样 teacher 梯度，但只保存方向投影，不构造完整 G。**

原因是：验证曲率估计器只需要若干固定残差方向，不需要先解决“怎样压缩联合统计”这个后续问题。先用普通 SVD 与 A 加权 SVD 生成方向，可以避免把未经验证的新 G 同时放进方向构造与评分参照。

## 2. 固定设置与数据划分

| 项目 | 设置 |
|---|---|
| 模型 | 本地 Meta-Llama-3.1-8B Base 原始 checkpoint；冻结本次文件和张量身份 |
| 目标模块 | `model.layers.31.self_attn.q_proj`、`model.layers.10.self_attn.v_proj` |
| 选择原因 | 历史敏感性补偿中的改善案例与反例；不是代表性随机样本 |
| 量化 | 本地新生成 MXINT3；`width=3, block_size=32, block_axis=-1` |
| 输入统计 | 完整 A，仅上述两个模块 |
| 补偿 rank | 两种补偿均为 rank64；另外保留未补偿方向作为控制 |
| A 校准数据 | 已准备的 SlimPajama 256 个固定窗口，每个 2048 token |
| 曲率诊断数据 | WikiText-2 **validation** 前 8 个符合预处理规则的完整窗口，每个 2048 token |
| 模型数值路径 | FP32、eval、batch=1；参数冻结；关闭 autocast、TF32 和 KV cache |
| 注意力 | 初始 eager；更换实现或设备映射须重新通过 pilot 并记录配置 |
| 求解 | FP64 分解、完整 SVD 和线性求解；不使用随机 SVD |
| 曲率采样 | 每窗口 K=4 试跑；正式 K=16，按精度追加到 K=64 上限 |
| 干预幅度 | 初始 alpha=0、0.05、0.1、0.2；按预定规则追加更小值 |

A 校准只用 SlimPajama；生成的补偿在查看诊断 KL 前冻结。8 个 validation 窗口用于**同输入上的曲率与 KL 配对验证**，不是最终泛化测试，也不用于调阻尼或挑选补偿。

若 prepared 数据没有 validation，使用冻结 tokenizer、QERA 预处理与固定文档顺序生成，保存实际 token 张量和哈希。不得静默使用历史 test。记录两个数据源与窗口 ID，并排查完全相同的窗口；不把这个检查说成排除了所有文本重叠或预训练污染。

## 3. A 到底收集什么

使用列向量约定：W 属于 R^(m×n)，h=Wx。令有效输入位置集合为 I：

$$
S_A=\sum_{(c,t)\in\mathcal I}x_{ct}x_{ct}^{\top},\qquad
N_A=|\mathcal I|,\qquad A_{\rm raw}=S_A/N_A.
$$

具体规则：

- x 来自**未量化 teacher 的目标 Linear 输入**；采集时不安装补偿或干预权重。
- 收集未中心化二阶矩，不减均值。`cov(x)` 不能代替此处的 A。
- 完整无 padding 窗口收集全部 2048 个模块输入位置，故 256 个窗口应有 N_A=524288；若有 padding，显式排除并报告实际计数。
- 同一次 teacher 前向可同时收集两个模块；只挂这两个输入 hook。
- 使用 FP32 分块 Gram 乘法、FP64 累加，关闭 TF32；初始 token chunk=256，实际值进入配置。先进行下述 FP64 对照检查。
- N_A 是输入位置数，不是窗口数，也不是 next-token loss 数；不额外乘 2。
- 不裁剪 activation，不做迹归一化，不在采集阶段加阻尼或特征值 floor。

这里 A 的 2048 个输入位置与后面 KL 的 2047 个预测位置不同，来自两个目标的定义。A 基线度量所有有效层输出的平均 MSE；teacher-KL 曲率按有效预测数归约。不要通过偷偷更改 mask 或分母让两者计数相等。

**永久保存三层产物，不能只保存一个“root”：**

1. `S_A`（FP64）、N_A、已完成窗口 ID、模型/数据/hook/mask/精度信息。
2. `A_raw`（FP64），保留原始统计；记录对称性、谱范围和累计误差诊断。
3. `A_solve`、阻尼参数、Cholesky L、求解记录；与 A_raw 分开。

只改 teacher 标签梯度的定义，不会自动改变 A 的定义。A 是否可复用取决于模型、输入数据、预处理、统计位置及数值路径是否一致。本次按用户选择重新采集，不依赖历史复用。

## 4. Cholesky 替代平方根：目标相同，分解不同

对正定的求解度量 A_solve，设

$$
A_{\rm solve}=LL^{\top}=SS^{\top},\qquad S=A_{\rm solve}^{1/2}.
$$

L 是下三角 Cholesky 因子，S 是对称平方根。两者均满足

$$
\operatorname{tr}(R A_{\rm solve}R^{\top})
=\|RL\|_F^2=\|RS\|_F^2.
$$

对 E=W0-Wq、rank(C)≤64：

$$
C_A=[EL]_{64}L^{-1},\qquad
C_{\rm sym}=[ES]_{64}S^{-1}.
$$

两者在精确算术下具有相同最优目标值；最优解唯一时得到相同 C。截断处奇异值重合时可能得到不同的同优解，不能用 C 的逐元素相等作为唯一验收条件。

实现使用三角求解恢复右因子，不显式计算 L 的逆。若 EL=UΣV^T，可保存

$$
P=U_r\Sigma_r^{1/2},\qquad
Q=\Sigma_r^{1/2}V_r^{\top}L^{-1},\qquad C_A=PQ,
$$

其中通过求解 `L.T @ Q.T = (sqrt(Sigma_r) @ V_r.T).T` 得到 Q。权重布局是 m×n，因此白化为 **E@L**。SRR 的转置权重实现使用 `L.T` 左乘，不能直接照搬到本布局。

### 4.1 正则化单独决定，不能混入分解比较

先对原始统计做数值对称化：A_sym=(A_raw+A_raw^T)/2，并记录改变量。随后

$$
A_{\rm solve}=A_{\rm sym}+\lambda I,\qquad
\lambda=\eta\,\operatorname{tr}(A_{\rm sym})/n.
$$

本协议的起始工程规则：按 eta=`0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3` 顺序，选取满足下列条件的最小值：正定、谱条件数≤1e8、FP64 Cholesky 相对重构误差≤1e-10。规则在查看 KL 前执行；两个分解参照必须使用**同一个 A_solve**。

若 A_sym 的最小特征值小于 `-1e-6 * lambda_max`，先判为统计/数值问题并用 FP64 Gram 核查，不能直接靠大阻尼掩盖。若谱非有限、尺度为零或上述 eta 范围均不通过，标记 `A_SOLVE_UNRESOLVED`，保存诊断后停止依赖 A 的正式方向生成；不自动放宽规则。

这些阈值是实现前固定的工程约束，不是最优阻尼理论。必须报告实际 eta、lambda、前后谱及原始目标变化，因为

$$
\operatorname{tr}(R A_{\rm solve}R^T)
=\operatorname{tr}(R A_{\rm sym}R^T)+\lambda\|R\|_F^2.
$$

Cholesky 不改变目标；加阻尼会改变目标。后续若研究阻尼优劣，另设实验，不能在本轮按 KL 选阻尼。

### 4.2 必做的有限数值审计

先用 SlimPajama 的前 1 个窗口完成 collector pilot，再收集 256 个窗口。pilot 的激活可临时保存在 CPU，审计结束后释放。

| 检查 | 做法 | 初始工程标准 |
|---|---|---|
| Gram 精度 | 同一批 FP32 x，比较生产 Gram 与将 x 提升 FP64 后的 Gram | 相对 Frobenius 误差≤1e-5 |
| 目标恒等式 | 同一批 x、固定 E 与一个固定随机 probe，比较直接平均输出 SSE 和 tr(R A_raw R^T) | 相对误差≤1e-5；直接参照用 FP64 |
| 分解精度 | 比较 LL^T、SS^T 与同一个 A_solve | 相对 Frobenius 误差≤1e-10 |
| 求解最优值 | 比较 Cholesky 与对称平方根的残差目标，并与各自奇异值尾和核对 | 相对误差≤1e-6 |
| 原始目标 | 对两种 C 同时记录 J_raw 与 J_solve | 区分正则化与分解的影响 |
| 截断稳定性 | 记录第 64、65 个奇异值及其间隔，另报两种 C 的差异 | 退化时不强求逐元素相等 |
| 成本 | 分别记录分解、SVD、求解耗时和内存 | 不预设某分解一定更快 |

相对误差使用参照量绝对值作为分母；若参照为零，单独报告绝对误差和零方向状态，不套用相对判据。对称平方根参照用 FP64 `eigh` 构造；这里验证数学等价性，不要求重跑 SciPy 的 Schur 实现。若生产 Gram 精度失败，可切换 FP64 Gram，重新完成全部收集并赋予新统计身份，不混合两种路径。

## 5. 本地生成 Wq 和六个固定方向

### 5.1 固定新的量化资产

从本地 checkpoint 提取两个模块的 W0，提升为 FP32。使用固定官方 QERA 源码中的 MXINT 量化器，在 W0 的 FP32 值上生成 Wq，保存 FP32 张量、量化器源码哈希和完整配置。

固定官方 QERA 提交参考：`bd7fc86a2e44d41f95b9b0421f27f5624dd37064`。新 Wq 只生成一次，后续所有方向共同使用这份张量，不让不同求解器各自重新量化。可以额外记录原 BF16 输入的量化是否一致，但本轮定义以冻结的 FP32 量化结果为准。

这是本次新的量化资产，不声明与历史 Wq 逐字节相同。只量化这两个模块来定义方向；teacher 本身始终保持原始权重。

### 5.2 每个模块的三个方向

令 E 在 FP64 中由冻结的 W0、Wq 相减得到：

| 方向 ID | 补偿 C | 定义的残差 | 用途 |
|---|---|---|---|
| `R_none` | 0 | E | 未补偿量化方向，作为控制 |
| `R_svd64` | [E]_64 | E−[E]_64 | 同 rank 的普通 SVD 残差 |
| `R_A64` | [EL]_64 L^(-1) | E−C_A | 同 rank 的 A 加权残差 |

两模块共六个方向。其中**四个是 rank64 补偿后的残差，两个是未补偿控制**；不能把三类都说成相同 rank 的方法比较。

FP64 求解并保存低秩 P、Q、奇异值和求解元数据，构造 R 后最终冻结为 FP32。方向构造与转换误差单独记录；之后曲率估计和权重干预只使用这份相同的 FP32 R。方向保持原始幅度，不按范数归一化。

本轮不模拟“两次 BF16 GEMM”的部署，不将因子降为 BF16。alpha=1 的浮点端点也不代表历史部署路径。BF16 因子存储与实际部署误差是后续补偿验证的议题。

Cholesky 和对称平方根产生的 C 只用于第 4 节的数值对照，正式方向统一使用通过审计的 Cholesky 结果，不额外扩充方向数。

## 6. 实验一实际计算什么

### 6.1 teacher KL 干预路径

一次只改变一个目标模块，其余模型保持 teacher 状态：

$$
W(\alpha)=W_0-\alpha R,\qquad \delta h_t=-\alpha R x_t.
$$

在 FP64 中构造候选权重后转 FP32，记录实际加载的差值。结束或异常时恢复 W0，核对恢复后的张量哈希。

对窗口 c 的有效预测位置集合 J_c、T_c=|J_c|：

$$
D_c(\alpha;R)=\frac1{T_c}\sum_{j\in\mathcal J_c}
\operatorname{KL}(p_j\|p_j^{(\alpha,R)}),\qquad
D(\alpha;R)=\frac{\sum_cT_cD_c(\alpha;R)}{\sum_cT_c}.
$$

p_j 是固定 teacher 的完整词表分布，KL 的左侧始终不变。采样标签不改变输入文本，不进行自由生成。

### 6.2 teacher 标签采样得到方向性曲率

每窗口、每个 replicate k，在所有有效预测位置**条件独立**采样：

$$
y_j^{(k)}\sim\operatorname{Categorical}(p_j),\qquad
L_c^{(k)}=-\sum_{j\in\mathcal J_c}\log p_j(y_j^{(k)}),\qquad
g_t^{(k)}=\frac{\partial L_c^{(k)}}{\partial h_t}.
$$

loss 必须求和，标签采样不参与求导。对共享该权重的全部模块位置先求和，再平方：

$$
d_{c,k}(R)=\sum_t(g_t^{(k)})^T R x_t,\qquad
b_{c,k}(R)=\frac{d_{c,k}(R)^2}{2T_c},
$$

$$
\widehat q_c(R)=\frac1K\sum_k b_{c,k}(R),\qquad
\widehat q(R)=\frac{\sum_cT_c\widehat q_c(R)}{\sum_cT_c}.
$$

在精确实数计算、固定输入和 teacher 分布下，此估计对局部 KL 二次项无偏；实际 FP32 前向与采样存在数值误差，必须通过 pilot 检查。局部光滑性成立时：

$$
D(\alpha;R)=\alpha^2q(R)+O(\alpha^3).
$$

定义与实现必须保持以下区分：

1. g_t 来自 teacher 采样标签，不来自 ground-truth CE；teacher 对自身 KL 的一阶梯度为零，也不能直接拿来做外积。
2. j 是预测位置，t 是模块位置。g_t 包含有效预测对 h_t 的全部下游影响。
3. `square(sum_t projection_t)` 才保留共享权重的位置交叉项；`sum_t square(projection_t)` 是另一个近似。
4. 模块位置求和覆盖全部输入位置，预测 mask 仅用于 loss。不要凭经验删除可能影响有效预测的模块位置。
5. sum-loss 配合上述除以 T 的规则；用 mean-loss 会再引入 T 的缩放错误。
6. 同一梯度同时评分该模块的三个方向；不裁剪梯度，不对 G 做归一化、阻尼或 floor，因为这里根本不构造 G。

可借鉴分块词表的解析 NLL seed：在最终 hidden 处计算 `(p-one_hot(y)) @ W_head`，再反传到目标模块；必须先与直接 autograd 验证。采样和 seed 都用完整词表，不使用 top-k 截断。

## 7. 本地资源与执行顺序

### 7.1 已知环境，仅作为检查入口

以下来自 2026-09-16 准备记录；不代表当前 GPU 空闲或本实验已运行。

| 项目 | 路径或信息 |
|---|---|
| SSH | `cck@172.31.136.236`，使用已有登录方式，不在实验文档或日志保存密码 |
| 硬件 | 用户确认 3×RTX A6000，系统内存约 112GB；GPU 为共享资源 |
| 激活入口 | `/home/cck/.config/kfac-qera/activate.sh` |
| 主仓库 | `/home/cck/projects/KFAC-QERA` |
| 近期工具快照 | `/home/cck/projects/KFAC-QERA-local-20260916` |
| 官方 QERA | `/home/cck/projects/vendor/QERA-bd7fc86` |
| Python 环境 | `/home/cck/miniconda3/envs/kfac-qera` |
| 已记录版本 | PyTorch 2.3.0+cu121，Transformers 4.44.2 |
| Llama | `/data1/cck/models/Llama-3.1-8B/ms-snapshot-20260916` |
| SlimPajama 原始数据 | `/data1/cck/datasets/raw/SlimPajama-6B/b5f90f4-prefix5120` |
| WikiText 原始数据 | `/data1/cck/datasets/raw/wikitext2-token/b08601e04326c79dfdd32d625aee71d232d685c3/dataset` |
| 已准备的 Llama token 数据 | `/data1/cck/datasets/prepared/Llama-3.1-8B/qera-bd7fc86-ctx2048-workers8` |
| 结果根 | `/data2/cck/KFAC-QERA/runs` |
| 临时目录 | `/home/cck/scratch/KFAC-QERA` |

这次只需本地模型、数据和量化/实验源码；**旧服务器未迁入 A/G/Wq/因子，不构成阻塞**。需要确认当前 teacher 身份和新产物关联，无需证明其与历史大权重完全相同。

若服务器缺 SRR 源码，可参考本协议公式实现并核验；SRR 代码包不是本实验运行依赖。

### 7.2 顺序与停止条件

1. **只读 doctor。** 读取实际仓库 AGENTS.md；检查 GPU UUID、空闲显存、其他作业、系统/cgroup 可用内存、磁盘空间、环境与代码来源。保存 `environment.json`。不终止他人的任务，不直接运行旧双 4090/180GiB 内存入口。
2. **CPU 数学检查。** 用小矩阵验证 A、分解方向与低秩求解；用两个共享位置、两个预测位置的小词表模型枚举标签组合，验证 KL Hessian、完整方向梯度平方及归约。验证解析 seed、权重恢复和续跑计数。
3. **新建 Wq 与 A pilot。** 生成两个模块的 Wq；用一个 SlimPajama 窗口验证 A 采集与 FP64 对照，测资源。A 前向使用 no_grad，可同时挂两个输入 hook。
4. **本地收集 A。** 对 256 个固定窗口前向；保存原始统计。完成正则化、Cholesky/对称平方根审计，生成六个新方向并冻结。
5. **曲率 GPU pilot。** 两个模块各用一个 validation 窗口、`R_A64`、一组标签和 alpha=0/0.1，验证完整采样反传与 KL 路径。L10 反传更长，不能只测 L31。
6. **正式配对实验。** 两模块、8 窗口、六方向，先 K=4 工程轮，再 K=16，必要时追加到 K=64；按第 9 节判定。完成报告后止于本实验，不自动进入完整 G 收集。

若 A 审计未通过，仍可继续不依赖 A 的 CPU/资源排查，但不能用缺少 `R_A64` 的结果冒充完整六方向实验。某方向曲率判定失败不等于任务没做完；诚实保留失败或不确定的科研结论。

### 7.3 内存和算力边界

- 从一张可分配的 A6000 开始。8B FP32 权重本身约 30GiB，2048 长度反传是否能放下必须实测。
- 所有参数 `requires_grad=False`；曲率阶段仅令目标模块输出启用梯度，不为上游保存反传图，不建 optimizer。按模块分别执行，避免 detach 较晚模块而意外切断较早模块梯度。
- 两个输入维度为 4096 时，一份 FP64 A 约 128MiB，两份约 256MiB；实际维度从配置和张量核对。分解/SVD 工作区及模型激活另算。
- 分块词表、逐窗口处理、及时释放图；teacher 和干预模型依次运行，无需两份 FP32 模型同时常驻。
- 必要时使用验证过的下游激活重计算或两张 A6000 的显式映射。eval 下某些 checkpoint 开关不生效，必须实际检查；重复前向不能重复累计 hook。
- CPU offload 先测 RSS；112GB 总内存不是本任务全部可用额度。不能只删除旧脚本 180GiB 门禁继续原样运行。
- 容量不足不静默改成 BF16、缩短正式上下文或删预测位置。不要求三卡同时空闲，也不自动租 GPU。
- 保存原始 A 后可释放模型再做矩阵求解，避免同时占用大模型和 SVD 工作区。

pilot 必须记录 GPU allocated/reserved 峰值、CPU RSS、device map，以及模型加载、A 前向/Gram、一次反传、一次干预 KL、分解和 SVD 的实际耗时。

成本口径：A 收集约 256 次 teacher 前向，两个 hook 共用。曲率按模块分别重算时，8×2×K，即 K=4/16/64 时累计约 64/256/1024 次反传，每次覆盖三个方向。初始 KL 约 2×8×3×3=144 次干预前向，另计 teacher、重复控制、pilot 和额外 alpha。对称平方根对照与矩阵求解单独计时；不预先承诺总时长。

## 8. 概率、KL 与随机性

模型前向 FP32；完整词表 softmax/logsumexp、KL 归约、方向投影的最终点积与平方使用 FP64。可按 token chunk=128 分块处理词表。

小 KL 不宜通过两个接近的 cross-entropy 相减。令 d=z_alpha−z_0，p=softmax(z_0)，v=d−E_p[d]，可以使用

$$
\operatorname{KL}(p\|p_\alpha)=\log E_p[e^v]
=\log\left(1+E_p[e^v-1-v]\right).
$$

以 `expm1`、`log1p` 和小 v 的稳定余项计算，并与 FP64 toy 参照核对。不要无条件把负 KL 截成零。完整词表张量只临时保留当前窗口，处理完即释放。

使用固定 base seed `20260918`；由窗口 ID 与 replicate ID 派生独立随机流。不同预测位置独立抽样，两个模块复用同一窗口/replicate 的标签，三个方向共用同次梯度。保存实际标签、采样实现、设备、精度和哈希，不只保存 seed。

K=16/64 在已有样本上追加，不能因前一批结果不好看重新抽取。每次清理梯度，确认一条投影对应一组标签。方向冻结后才开始曲率采样与 KL 测量。

## 9. 怎样判定实验一完成，以及结论是否成立

以下阈值是本版预先固定的工程验收标准，不是理论界，也不证明未来补偿选择的精度。

### 9.1 数值路径先有效

| 项目 | 标准 |
|---|---|
| alpha=0 重复前向 | 平均 self-KL≤1e-10；另报逐 token 最大值和 logits 差异 |
| 实际权重差值 vs −alpha R | cosine≥0.999999，差向量相对 L2 误差≤1e-3 |
| 实际模块输出差值 vs −alpha R x | cosine≥0.999，相对 L2 误差≤2% |
| 模块输入 | 干预前后保持相同；否则检查范围、缓存或恢复错误 |
| KL 信号 | 平均 KL>100×max(重复前向 self-KL floor, 1e-12) |
| 恢复和有限性 | 恢复哈希一致；所有投影、概率和评分有限 |

非零方向才使用余弦和相对误差。差向量误差不是两个范数之差。无效点保留记录并给出原因，不参加局部曲率拟合。

### 9.2 小扰动区间由 KL 自己决定

计算 kappa(alpha)=D(alpha;R)/alpha²。初始 alpha=0.05、0.1、0.2；要求三个相邻有效点满足 `max(kappa)/min(kappa)-1≤10%`，并报告 log(KL) 对 log(alpha) 的斜率。

若没有平台，依次追加 0.025、0.0125、0.00625，最多三点。不得越过无效点拼凑相邻区间。若小 alpha 已低于数值分辨率，停止继续减小。

在已经测量的 alpha 中，选择满足条件的最小三点组，取三点 kappa 均值为 q_KL，同时报告范围。选择只能依据数值有效性和 KL 平台，不能看哪个区间最贴近 MC。q_KL 是有限扰动参照，不称为精确 Hessian。

### 9.3 单独估计 teacher 标签采样噪声

固定 8 个窗口，令 s_c² 为 K 个 b_(c,k) 的样本方差，omega_c=T_c/sum(T_c)：

$$
\widehat{\mathrm{SE}}_{MC}
=\sqrt{\sum_c\omega_c^2s_c^2/K}.
$$

报告近似 95% 区间 `q_hat ± 1.96 SE_MC`。它只描述条件于这些窗口的标签采样噪声，不是对新语料的泛化区间；有限 K 与自适应停止下也不承诺严格覆盖率。2047 个预测位置不能冒充独立 replicate。

K=4 不作最终判定。K=16 若任一方向的相对区间半宽大于 20%，该模块三方向统一追加到 K=64；达到上限仍不够则报告不确定。无须为追求通过无限增加采样。

### 9.4 每方向结论

数值路径有效、有局部平台、相对 MC 区间半宽≤20%，且 `abs(q_hat-q_KL)/q_KL≤20%`，才记为 `PASS_LOCAL_ALIGNMENT`。

| 状态 | 含义 |
|---|---|
| `PASS_LOCAL_ALIGNMENT` | 通过当前有限预算和工程容差下的局部一致性检查 |
| `NUMERICAL_INVALID` | 路径或数值控制不可靠，先修实现 |
| `LOCAL_RANGE_UNRESOLVED` | 预设 alpha 中未找到可信平台 |
| `MC_INCONCLUSIVE` | 采样上限后仍不足以判断 |
| `MISMATCH_TO_INVESTIGATE` | 数值与采样精度足够，但仍不对齐 |

六方向全部通过，才能写“在这两个模块、六方向、八窗口上通过”。部分通过如实报告，不删除失败方向。这里不要求 R_A64 的 KL 一定低于 R_svd64，也不把两者偶然排序作为实验成功条件。

## 10. 文件、断点与最小报告

建议在仓库新建 `experiments/qer_teacher_kl_exp01/`，结果使用独立根：

```text
/data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp01_v2/<run_id>/
  protocol.md
  config.resolved.yaml
  environment.json
  source_manifest.json
  data_manifest.json
  status.json
  quantized/               # 本地新 Wq 与量化定义
  input_stats/             # S_A、count、A_raw、窗口进度与数值身份
  solve_metrics/           # A_solve、lambda、L、分解审计
  factors/                 # FP64 SVD64/A64 因子与求解元数据
  directions/              # 六份冻结 FP32 R 与来源链
  samples/                 # teacher 实际采样标签
  records/                 # 原子提交的窗口/replicate/alpha 记录
  a_audit.csv
  mc_projection.csv
  kl_path.csv
  summary.csv
  resource_usage.csv
  report.md
  figures/
  logs/
```

关键记录：

- `a_audit.csv`：模块、N_A、统计 hash、对称/Gram 误差、谱、eta/lambda、条件数、LL^T/SS^T 误差、J_raw/J_solve、尾奇异值目标、rank 边界间隔、求解成本。
- `mc_projection.csv`：模块、方向、窗口、replicate、T、带符号 d、b、标签 hash、R hash。
- `kl_path.csv`：模块、方向、窗口、alpha、T、KL_sum/mean、self-KL、权重/输出扰动误差、点有效性及原因。
- `summary.csv`：模块、方向、K、q_hat、MC_SE/区间、局部 alpha、q_KL、相对误差、状态。
- `resource_usage.csv`：阶段、实际 GPU UUID、显存/RSS 峰值、耗时、device map、重计算/offload 配置。

配置、teacher、数据、量化器、方向、代码与数值实现共同构成运行身份。不兼容变更另开 run_id；同一路径的 K/alpha 扩展只追加样本。

A 每 8 个窗口原子保存统计和已完成 ID；中断只重算未提交部分。曲率每个 `(module, window, replicate)` 一次性提交三个投影；KL 按 `(module, direction, window, alpha)` 提交。避免重复计数，不为每个窗口永久保存一份完整 Gram。

资源时段结束正常保存为 `PAUSED`。核心资产缺失使用具体缺项状态，但旧产物缺失不属于缺项。全部要求的记录齐全才记采集完成，科研判定独立保存。

报告只需四块：A 数值审计表；六方向 `KL/alpha²` 与 MC 误差带；逐方向状态及 K 收敛；实测总成本与失败项。无需长期保留所有激活、梯度和完整 logits，保留可复算的身份、标签、原始 A、因子与标量投影。

## 11. 可复用源码与理论结论的边界

以下是工作区/快照中的相对路径；执行者应核对实际来源，只复用必要逻辑，不直接运行旧启动器。

| 来源 | 可借鉴 | 必须调整或注意 |
|---|---|---|
| 官方 QERA `src/qera/quantize/quantizers/mxint.py` | 冻结 MXINT 量化规则 | 本轮独立生成 Wq，不要求旧 manifest |
| `srr/src/qera/ptq_pipeline.py` | Cholesky 因子路线 | 其 `H/2`、`L.T`、FP32 转存和固定加 1e-6 不能原样套用 |
| `srr/src/qera/statistic_profiler/scale.py` | 输入 Gram hook | 明确本轮收集 A 而非 2A；保留原始 FP64 累加和计数 |
| `qera_mxint4_full_ag/experiments/qera_diag_g_isolation/math_ops.py` | 分块 NLL hidden seed | 旧真实标签改为 teacher 采样标签 |
| `fisher_qer_int4_reaudit/src/factor_stream.py` | 标签采样、seed、hook 去重 | 不继承旧 BF16、裁剪或完整 Gram 采集 |
| `fisher_qer_int4_reaudit/src/path_audit.py`、`metrics.py` | FP32 路径、恢复与稳定 KL | 用本轮模型、方向与协议重新核验 |
| `fisher_qer_int4_reaudit/src/exp02_pair.py` | 方向投影组织方式 | 旧逐位置平方求和不等于本轮完整共享权重曲率 |

环境记录参考 `server_setup_20260916/SETUP_README.md`、`receipt/assets.json`、`receipt/environment-verified.json`。旧准备文件中的历史复现协议不自动覆盖本版新实验定义。

已有结论与本轮补充的边界：

- [QERA](https://arxiv.org/html/2410.06040v1) 已有输入二阶矩加权低秩重构；用不同合法因子表示同一正定度量不构成新的重构目标。本轮补充实际 collector、正则化和求解精度审计。
- teacher 分布下 score 外积与局部 KL 曲率的关系属于标准模型 Fisher 理论，可参见 [Martens & Grosse, 2015](https://proceedings.mlr.press/v37/martens15.pdf)。本轮不把恒等式当作创新，而是验证实现、QER 实际方向、小扰动区间与有限预算可计算性。
- 本轮没有检验 A⊗G 丢失多少联合信息，也没有证明任何 G 比 QERA 更好。只有参照可靠后，才比较同一批 teacher 统计下的联合与分离评分，避免把标签来源改变误判成结构近似的效果。
- 本轮不是全模型部署或泛化验证；两个被挑选模块的结论不能外推到全部层、完整量化幅度或任务准确率。

## 12. 可直接交给另一个 Codex 窗口的提示词

> 请按附件 v2.0《QER 本地实验第一步：重建 A、核验 Cholesky 求解、验证 teacher-KL 曲率》在本地 Linux 实施。使用已有 SSH 方式连接 `cck@172.31.136.236`，先只读检查仓库约束、环境、实际可用 A6000/内存和模型数据。用户决定不取回历史 A/G/Wq/因子；旧产物缺失不是阻塞，不要启动历史迁移。建立独立实验目录，先完成小模型数学检查，再生成两个目标模块的新 MXINT3 Wq，pilot 后收集 SlimPajama 256×2048 的完整 A。原始 A 与阻尼后的 A_solve 分开保存，以同一 A_solve 对照 Cholesky 和对称平方根；通过后生成每模块未补偿、SVD64、A64 三个固定残差。随后完成两个模块各一个 2048 窗口的 FP32 曲率 pilot，再按协议执行 8 个 validation 窗口的六方向曲率—KL 验证和报告。teacher 标签独立采样、sum-loss、模块位置求和后平方；不要沿用旧 ground-truth CE-G。支持原子续跑，记录实测资源和失败状态。当前只收集方向投影，不扩展全模型 A/G、不做 rank 分配、不租 GPU。可在授权范围内完成新代码、核验和本地运行，无需为每个可逆步骤重复请求确认；遇到真实资源或资产阻塞时给出具体原因与已完成产物。
