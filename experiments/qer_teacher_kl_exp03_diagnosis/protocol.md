# Exp-3 Diagnosis：原拟合曲率与同文本新标签的两阶段诊断

版本：v1.0，2026-09-18。

状态：**实验方案，尚未实现或执行。本文只规定诊断内容；不启动服务器任务，不重新拟合或修改 Exp-3。**

## 1. 核心问题

> Full-fit-AG64 的收益预测与实际完整曲率之间的落差，在原拟合经验目标内部就已经存在，还是主要在更换 teacher 标签或文本之后出现？

本轮不是扩大版 Exp-3，也不是新方法比赛。固定原来的 metric 和补偿，只做两项诊断：

1. **阶段 A：复用原 S 缓存。** 在构造因子时使用的 8×4 个完整梯度样本上，计算补偿的真实经验完整曲率评分。
2. **阶段 B：相同文本，独立新标签。** 保持原 8 个拟合窗口不变，每窗口新采样 16 份 teacher 标签，只评价已有补偿。

完成后与 Exp-3 已有的历史评价集结果并列解释，再决定是否值得增加拟合窗口或标签重做拟合。本方案不自动执行第三步。

重点观察 L31.q_proj，但两个原模块全部保留，不只报告符合预期的模块。

## 2. 诊断动机与既有结果

Exp-3 的核心比较为 Full-fit-AG64 相对 Marginal-AG64：

| 模块 | 历史评价集 q_H | 历史评价集 actual KL |
|---|---|---|
| L10.v_proj | 点估计改善，配对区间仍未确定 | Full-fit 相对 Marginal 降低 8.22% |
| L31.q_proj | Full-fit 明确退化 | Full-fit 相对 Marginal 升高 136.64% |

L31 上，Full-fit solve metric 对自身补偿预测恢复率约 99.98%，历史评价集完整曲率恢复率约 29.82%；对 Marginal 补偿则分别为约 86.85% 和 70.26%。这说明冻结 metric 与完整曲率存在收益判断失配，但尚未定位它发生在哪一层。

Exp-3 使用每模块 8 个拟合窗口、每窗口 4 份标签，共 32 个 S；历史评价为另一批 8 个窗口、每窗口 64 份标签。本轮将“换标签”和“换文本”分开检查。

## 3. 固定资产、比较组与禁止事项

### 3.1 父运行

```text
Exp-3 identity:
34fd1ee3b82701316a3ce90808b911402f197b90538b97a7c0060a6ec47dc388

远端父目录：
/data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp03/20260918_r1

本地结果：
outputs/qer_teacher_kl_exp03_20260918/
```

实施时核验父 identity、manifest、evaluation_freeze、样本清单和实际 tensor hash。路径正确不等于资产身份正确。

### 3.2 固定设置

| 项目 | 本轮约定 |
|---|---|
| 模块 | `model.layers.10.self_attn.v_proj`、`model.layers.31.self_attn.q_proj` |
| 模型与数值路径 | 原冻结 FP32/eager teacher；关闭 TF32、autocast、KV cache，继承父实验配置 |
| 窗口 | 原 Exp-3 `data/fit_windows.safetensors` 的全部 8 个窗口，不重新分词或选文本 |
| 长度 | L=2048，T=2047；有效位置和 mask 与父实验一致 |
| rank | 原补偿 rank=64，不重新求解 |
| 核心候选 | None、Marginal-AG64、Full-fit-AG64 |
| 上下文基线 | 原冻结 SVD64、A64；统一在同一次投影中报告 |
| metric | 原 marginal/full-fit 的 raw 与 solve 因子，全部只读 |
| damping | 原 eta_A=eta_G=0.001 trace-relative damping；不更改或重新选参 |
| canonical convention | H 已含 1/T，保存的 marginal canonical 因子已吸收 L/T，不重复乘系数 |

五个候选均使用 Exp-3 实际评价的部署残差。新 AG 文件中的 `R` 是 FP32 teacher 权重与 FP32 `W_deploy` 的差，以 FP64 保存；不能替换成理想高精度 C64 对应的另一个残差。

### 3.3 禁止变更

- 不重新量化、拟合 A/G、增加 ALS 迭代、更换初始化、调整 damping 或重新运行 SVD。
- 不用新标签更新任何 factor 或 correction。
- 不增加文本窗口，不自动扩大标签预算，不根据结果选择有利 seed。
- 不重新跑原 validation 曲率或 actual KL；历史结果只读引用。
- 本轮不新增 actual KL。结论限于完整二次目标诊断，不把它写成新文本上的 functional damage 验证。
- 不覆盖父目录；新标签、记录和报告写入独立 diagnosis 运行目录。

## 4. 共用数学定义

对固定 teacher 窗口 c 与标签 replicate k：

\[
\ell_c^{(k)}=-\sum_{j\in\mathcal J_c}\log p_j(y_j^{(k)}),\qquad
S_c^{(k)}=\sum_{t=1}^{L}g_{ct}^{(k)}x_{ct}^{\top}.
\]

teacher 标签从各有效预测位置的完整词表分布独立采样，不送回上下文。使用求和 NLL，而非 mean loss 或真实标签 CE。

对冻结的实际残差 R_a，保存有符号投影与二次分数：

\[
d_{cka}=\langle S_c^{(k)},R_a\rangle_F,\qquad
b_{cka}=\frac{d_{cka}^2}{2T}.
\]

在每个阶段分别等权归约：

\[
\widehat q_D(R_a)=\frac18\sum_{c=1}^{8}\frac1{K_D}\sum_{k=1}^{K_D}b_{cka}.
\]

用实际 None 残差 R_0 作为共同参照：

\[
\gamma_D(a)=1-\frac{\widehat q_D(R_a)}{\widehat q_D(R_0)}.
\]

核心差值与归一化差值：

\[
\Delta_D=\widehat q_D(R_{\mathrm{marg}})-\widehat q_D(R_{\mathrm{fit}}),
\qquad d_D=\frac{\Delta_D}{\widehat q_D(R_0)}.
\]

正值表示 Full-fit 更好。这里 d_D 是汇总后的归一化比较量，与逐样本投影 d_cka 区分。分母非正或低于父实验预设下限 1e-12 时标记比值不可稳定解释，不改变分母、不裁剪恢复率。

None、SVD64、A64 的实际残差沿用父实验构造和哈希，不能直接以未核验的 Wq 重建替代。正文中的 E−C 是简写，实际计算以冻结 R 为准。

## 5. 阶段 A：原拟合样本上的经验目标诊断

### 5.1 输入与工作量

每模块读取原缓存：

```text
cache/<module_slug>/w00_k000.safetensors
...
cache/<module_slug>/w07_k003.safetensors
```

按 `data/fit_sample_manifest.json` 验证文件和 S 的哈希、窗口、标签身份。每模块 32 个 S，两个模块共 64 个原子样本；五个候选共 320 行评分。

流式读取一次 S 后计算全部候选，采用 FP64 内积与归约。不得加载 teacher 或重跑反传来完成这一主步骤；缺失缓存则报告具体缺失资产，不静默改成新样本。

这一步没有模型反传成本，但存在缓存 I/O 和矩阵运算成本，因此称为“低成本缓存诊断”，不承诺绝对免费。

### 5.2 补偿质量

计算每个候选的 q_old、gamma_old，以及核心 Delta_old、d_old，另列 8 个窗口分解。必须用原 S 得到完整曲率评分，不能用 q_K 自评分替代。

**本阶段是对固定经验目标的直接计算，不是独立泛化估计。** 补偿本来就依赖这些标签，不能对 32 个样本做普通 bootstrap 后宣称获得了无偏的样本外置信区间。

报告精确到数值容差的经验差值、窗口分解和实际幅度即可；本阶段不出泛化显著性判定，也不与阶段 B 套用相同的统计解释。

### 5.3 同时核验 raw 与 solve metric 的拟合质量

对四个冻结 metric：

- Marginal raw；
- Full-fit raw；
- Marginal solve；
- Full-fit solve。

在同一原拟合 S 集合上计算：

\[
J_{\mathrm{old}}(A,G)
=\|A\|_F^2\|G\|_F^2
-\frac2T\mathbb E_{\mathrm{old}}[\operatorname{tr}(GSAS^\top)].
\]

J 与 `||H_old−A⊗G||_F²` 只差相同常数，因此：

\[
J_{\mathrm{old}}(K_1)-J_{\mathrm{old}}(K_2)
=\|\widehat H_{\mathrm{old}}-K_1\|_F^2
-\|\widehat H_{\mathrm{old}}-K_2\|_F^2.
\]

不用显式构造 H，也不需要为比较计算其巨大常数项。J 为负是正常的，不能据此计算“负误差”或未定义的误差下降百分比。

raw marginal 和 raw full-fit 的 J 应与原 ALS 起点、终点记录对齐；solve J 用于判断共同阻尼之后的拟合次序是否改变。保持 canonical 尺度，不给各个 K 额外拟合缩放系数。

**这项计算需要矩阵收缩，成本高于单纯 S 与 R 的内积，但仍不需要模型反传。** 与投影成本分开记录；尽量复用同一次缓存读取。

### 5.4 冻结 metric 的交叉评分

对所有候选计算或经哈希核验后复用父实验的：

\[
q_K(R_a),\qquad
\gamma_K(a)=1-\frac{q_K(R_a)}{q_K(R_0)}.
\]

同一 K 和同一 R 下，q_K 不依赖评价文本，因此无需按阶段重新拟合 factor。分别比较：

\[
e_{K,\mathrm{old}}(a)=\gamma_K(a)-\gamma_{\mathrm{old}}(a).
\]

重点记录两个新候选的排序以及 Full-fit 自身恢复率预测的落差；五个候选中的 None 只是参照，不纳入四个非零补偿的 MAE。

### 5.5 第一阶段可成立的结论

若 raw Full-fit 的 J 更低，而其补偿在 q_old 下更差，可报告：

> 在同一个经验完整曲率上，更好的 raw Frobenius 拟合，经本次冻结的阻尼与求解流程后，没有转化为当前误差和 rank-64 条件下的更好补偿。

若 solve Full-fit 的 J 仍更低而补偿更差，则还可说明：拟合次序在阻尼后没有反转，但最终任务次序仍相反。

若 solve 的拟合次序反转，必须将正则化列为可能环节；不能将问题全部归给 raw 拟合准则。即使 solve J 更低，也不表示已经排除阻尼对解的其他影响。

上述现象排除了“只有换了评价文本才导致差异”这一单独解释，但不证明拟合数据充分、不证明增加数据无效，也不证明整个 single-Kronecker 家族不足。

## 6. 阶段 B：相同拟合文本上的独立新标签评价

### 6.1 固定采样预算

本小诊断采用：

| 项目 | 固定值 |
|---|---|
| 窗口 | 原 8 个 fit windows，token 张量和 mask 不变 |
| 新标签次数 | 每窗口 K_new=16 |
| 标签数 | 共 128 份标签，两个模块共用 |
| 模块—样本记录 | 每模块 128，共 256 个原子记录 |
| 候选分数 | 5 个候选，共 1280 行 |
| 新采样 base seed | 2026091804，独立 diagnosis 命名空间 |
| Bootstrap | 2000 次窗口内配对重采样，seed=2026091805 |

K_new=16 是本版为控制成本设置的诊断预算，不表示保证足够判定。未确定时如实报告，不自动增加到 32 或 64；如确需扩展，另行制定新版本并保留本轮结果。

在运行前冻结 seed 派生算法与实际 RNG 配置。确认没有复用 Exp-3 的原随机流；保存真实标签和哈希，不能只记录 seed。独立抽样允许个别 token 取值相同，不能为制造“标签不同”而拒绝正常样本。

### 6.2 评价路径

1. 使用冻结 teacher 对原拟合文本前向。
2. 对有效预测位置从完整词表分布独立采样新标签；不回填上下文。
3. 按原 sum-NLL 路径反传得到 x、g 或 S。
4. 同一次样本上计算全部五个冻结残差的投影和分数。
5. 不更新因子，不调整补偿，不修改 damping。

可采用 `sum_t g_t^T R x_t` 直接投影，不要求永久保存全部新 S。pilot 中验证其与 `<S,R>` 的等价性后冻结实现；新标签和标量记录必须保存，以支持重放。

不生成新 gradient Gram、不跑 ALS 或 SVD。若缓存新 S 有明确复现收益，单独登记存储预算，不默认复制父实验全部大缓存。

### 6.3 不确定性

对每个窗口、每个新标签先计算配对差：

\[
u_{ck}=b_{ck,\mathrm{marg}}-b_{ck,\mathrm{fit}}.
\]

原始 Delta_new 的条件标准误差：

\[
\widehat{\mathrm{SE}}^2=\frac1{8^2}\sum_{c=1}^{8}\frac{s_c^2(u)}{16}.
\]

对归一化 d_new、恢复率和预测误差，使用 2000 次窗口内配对 bootstrap；同一窗口的所有候选共用抽样索引。固定 metric 预测 gamma_K 不随 bootstrap 重新拟合。

区间只描述固定 8 个拟合文本、固定补偿下的新标签采样不确定性，不包含拟合不确定性、文本泛化或多重比较同时覆盖。

不将旧标签与新标签合并为“更多独立测试样本”；旧标签参与了补偿构造，始终单列。

### 6.4 判定规则

继承 Exp-3 的实际幅度阈值：`tau=0.02`，单位为相对 None 损伤的 2 个百分点。

阶段 A：对固定经验 d_old，分别报告数值正负与是否超过 ±0.02；这是经验目标描述，不标注统计显著性。

阶段 B：对 d_new 的 95% 配对区间 `[l,u]`：

- `l > 0.02`：Full-fit 明确且达到实际阈值的改善。
- `u < -0.02`：Full-fit 明确且达到实际阈值的退化。
- 区间全部在 `[-0.02,0.02]`：差异在实际预算内。
- 其他：当前 K_new=16 下未确定。

同时报告区间是否排除零，区分“方向已明确”和“幅度是否达到阈值”。配对比较是主判断，不通过两个独立候选的边际区间是否重叠作判断。

## 7. 两阶段与历史评价的联合解释

| 阶段 A：原文本原标签 | 阶段 B：原文本新标签 | 与历史 validation 结果合并后的解释 |
|---|---|---|
| Full-fit 经验退化，且其 raw 拟合更好 | 仍明确退化 | 失配不只发生在原标签集；优先研究拟合目标与冻结求解链路，不只归因于换文本 |
| Full-fit 经验改善 | 新标签明确退化 | 支持对原标签样本适应过强或曲率估计不稳定；不据一次新批次证明唯一原因 |
| Full-fit 经验改善 | 新标签仍明确改善 | 若 validation 退化，更支持文本覆盖/分布迁移问题；同时核对 fit/eval 预处理差异 |
| Full-fit 经验退化 | 新标签改善 | 原 32 样本的经验判断与新估计不同，需保留有限样本解释，不能按阶段 A 单独推广 |
| 任一关键阶段未确定 | 未确定或方向不一致 | 保留不确定性，不强行归类为结构、标签或文本原因 |

不能将“阶段 A 经验差值”和“阶段 B 条件区间”当成同一种统计量。两阶段差值可以描述性展示，但不直接用两个独立 SE 拼接出对训练适应程度的严格推断。

历史 validation 已用于发现问题，不是本诊断新的盲测集。这里只引用原冻结结果，不重新选择有利评价窗口或将三个数据层面混合汇总。

## 8. 运行前验收与数值边界

### 8.1 资产与精度

- 前后核验父统计、标签、因子、补偿和部署残差的哈希。
- 检查原 S 缓存每模块恰为 8×4 条；跨模块同窗口、同 replicate 使用同一标签。
- 阶段 A 的 S 内积、平方、J 收缩和最终归约均使用 FP64；不得对 S 再低秩压缩或对角化。
- 冻结 metric 已含正确 canonical 系数，不能再次吸收 L/T 或 T。
- 使用原保存的 raw/solve 因子，不“为了稳定”再次加阻尼或重新做 gauge。

### 8.2 核验项目

1. 在每模块至少一条原样本上验证投影路径与缓存身份；若原记录有同残差投影，直接做重放对照。
2. 重算 raw marginal/full-fit 的 J，与父 fit_history 起终点核对。
3. 核验本轮 q_K 交叉评分与父实验相同因子、相同 R 的记录一致。
4. 在小矩阵上检查 J 差等于显式 Frobenius 误差平方差，区分 raw/solve。
5. 阶段 B 每模块首条新样本核验两种投影收缩等价性；使用父实验 `contraction_tolerance=1e-10` 的既定误差定义，近零值用冻结的绝对/尺度保护规则。
6. 核验全部投影平方除以 2T、样本计数、窗口等权、配对差和恢复率恒等式。

J 重放建议使用 `|J_new−J_old| <= 1e-8 * max(1, |J_new|, |J_old|)` 的数值一致性标准；该标准仅用于重算同一量，不用于把小科学效果判为改善。若父实现已有更严格且适用的规则，可在运行前替换并记录，不能结果出来后调整。

继承父 ratio floor=1e-12；新标签与父实验共享采样和反传实现，只改变独立随机流和输入窗口集合角色。正式采集前冻结代码与 manifest。

验收失败先报告资产或数值问题，不把它归为科学上的方法退化，不自动重新拟合修复。

## 9. 执行顺序与资源

1. 建立独立 diagnosis 目录，登记父身份和只读依赖。
2. 冻结阶段 A/B 的候选、K_new、seed、判定规则和代码身份。
3. 完成阶段 A，输出经验 q_old、四个 metric 的 J、交叉预测与窗口分解。
4. 完成阶段 B 的数值 pilot；pilot 样本属于预定新标签序列，通过后只计数一次。
5. 完成全部 8×16 新标签评价，报告完整记录与不确定性。
6. 与历史评价结果并列生成诊断报告，提出是否值得扩大数据的有条件建议；不启动重拟合。

阶段 A 验收通过后，阶段 B 按本方案完成，不因为阶段 A 出现有利或不利效果而提前停止。真实资源/数值阻塞另行记录为未完成，不替换样本凑齐。

资源规划：阶段 A 可流式 CPU 或 GPU 计算，以文件 I/O 和收缩实测为准。阶段 B 可沿用 1 张 A6000、64 GiB 主机内存作为资源申请起点；新反传计数约为 Exp-3 正式 1024 次评价的四分之一，但加载、核验和 I/O 不按该比例缩放，不能据此保证精确墙钟时间。

不下载父大缓存到本地来绕过服务器内存预算，不生成完整高维 H，不重跑模型训练。若申请 GPU，结束时确认诊断任务停止并释放对应租约；不影响其他任务。

## 10. 结果产物与必备表格

建议独立目录：

```text
qer_teacher_kl_exp03_diagnosis/<run_id>/
  protocol.md
  manifest.json
  parent_integrity_before.json
  parent_integrity_after.json
  numerical_checks.json
  stage_a/old_fit_scores.csv
  stage_a/metric_objectives.csv
  stage_a/metric_cross_scores.csv
  stage_a/by_window.csv
  stage_b/samples/wXX_kYYY.safetensors
  stage_b/sample_manifest.json
  stage_b/new_label_scores.csv
  stage_b/by_window.csv
  summary/three_domain_comparison.csv
  summary/metric_prediction_gaps.csv
  bootstrap.json
  resource_usage.json
  verification.json
  SHA256SUMS
  RESULTS.md
```

所有原子分数记录包含模块、窗口、标签身份、候选、R_hash、有符号投影、平方分数及阶段；历史结果的引用注明父文件及哈希。

必须提供以下三张主表：

### 表 A：三个数据层面的补偿质量

每模块、每候选分别列：

- 原 fit 文本＋旧标签：q_old、gamma_old，无泛化 CI。
- 原 fit 文本＋新标签：q_new、gamma_new 与条件区间。
- 原 validation 文本：引用 Exp-3 的 q_eval、gamma_eval 与原区间。

另列三个层面的 `Marginal−Full-fit` 配对差，不将它们合并为一个总均值。

### 表 B：拟合目标与任务结果

每模块列四个 raw/solve metric 的 J_old、两份补偿的 q_old、核心 d_old，以及 raw 与 solve 的拟合优劣次序是否一致。

### 表 C：自评分落差与候选排序

对 marginal/full-fit 的 raw/solve metric，列两份新补偿的 gamma_K，并对照 gamma_old、gamma_new、gamma_eval。注明在哪一层发生“metric 认为 Full-fit 更优，完整曲率却认为 Marginal 更优”。

可附四个非零补偿的预测 MAE，但它不是本小诊断的主决策量，也不能用一个平均数掩盖核心候选排序。

独立复核至少覆盖计数、投影平方、J 差、窗口归约、配对差与 bootstrap。只记录已做的核验，不把读过验收报告写成重新执行了全部 tensor 检查。

## 11. 最终报告用语

本轮的最强可能结论是：

> 在原经验完整曲率上，Full-fit 的曲率拟合目标优于 Marginal，但其冻结的 rank-64 补偿反而更差；该现象不能仅由新旧评价文本不同解释。

如果原文本新标签也复现退化，可进一步说明该落差不局限于原来的 32 个标签样本。仍不宣布已找到唯一机制、不宣布样本规模不重要、不宣布单 Kronecker 的理论能力不足。

若原经验目标改善而新标签退化，结论应改为标签层面的样本外失配；若同文本新标签保持改善而新文本退化，则把文本覆盖/迁移作为后续重点。所有解释都要保留数值处理与有限样本边界。

本小诊断不决定新算法、不修改原结果，也不承诺“增加数据一定解决”。它提供的是下一步是否扩大数据、优先增加标签还是文本的证据。

## 12. 参考文件

- [Exp-3 实验设计](qer_teacher_kl_exp03_full_fit_ag_20260918.md)
- [Exp-3 最终报告](../outputs/qer_teacher_kl_exp03_20260918/RESULTS.md)
- [正式 manifest](../outputs/qer_teacher_kl_exp03_20260918/results/manifest.json)
- [评价前冻结清单](../outputs/qer_teacher_kl_exp03_20260918/results/evaluation_freeze.json)
- [原拟合窗口清单](../outputs/qer_teacher_kl_exp03_20260918/results/data/fit_windows.json)
- [原 S 样本清单](../outputs/qer_teacher_kl_exp03_20260918/results/data/fit_sample_manifest.json)
- [原 ALS 记录](../outputs/qer_teacher_kl_exp03_20260918/results/fit_history.csv)
- [收益预测误差](../outputs/qer_teacher_kl_exp03_20260918/results/summary/metric_prediction_error.csv)
