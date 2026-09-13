# 真实 G 与置换 G：固定 rank64 的输出坐标对应关系诊断

本实验检验：保留输出侧度量的谱性质，打乱它与原模型输出通道的对应关系后，低秩补偿的最终 teacher KL 是否增大。它承接 `local_output_kl_v1`，不进行 rank 分配，也不把真实 G 获胜作为完成条件。

## 固定设计

- Llama-3.1-8B Base、MXINT3，固定 Full-A 和 Wq，补偿 rank64。
- 固定层 0/10/20/31，每层 q/k/v/o/gate/up/down，全部 28 模块，包括先前 GF 退化的模块。
- 每次仅替换一个模块，其余保持 BF16。独立恢复权重，输入逐位一致，BF16 两次 GEMM。
- 九个配置：原 GI、DG、GF，以及 DG/GF 各三个置换版本。
- 种子固定为 20260913/20260914/20260915。CPU `randperm` 的实际种子由 SHA256(version|module|seed) 推导；同模块同种子的 DG/GF 使用相同 P，保存完整置换数组。不筛选置换，不按结果更换种子。若某个 G 对该置换不变，保留并标记为有效的零变化对照，不重新抽取。

使用 `R_permuted = R[p][:,p]`。这里 R 是历史求解使用的 FP32 G-root，实际度量为 `G_eff = R R^T`，因此 `G_eff_permuted = P G_eff P^T`。同时置换行列精确保留元素取值；通过双射及逆置换恢复核验，可以在数学上保证特征值、迹、条件数相同，无需再做一次大矩阵谱分解。保留的是冻结有效度量的谱，并非未经 normalization/floor 的原始统计。

Full-A、量化误差 E、模型输出通道和部署顺序均不置换。重新对 `A_root E^T R_permuted` 执行原 FP64 full SVD 和逆求解，取 rank64，再直接转 BF16；不会对原补偿因子简单换行列冒充重新求解。

## 数值与复现控制

1. 只读取已有 A/G 统计、FP64 因子和上轮局部实验，不重新收集统计、不构造新的 A-root。
2. 依照原 normalization/floor 重建 DG/GF 的 FP32 root，必须与原 FP64 求解证书中的 root bits 一致。重建后缓存到新目录，后续续跑复用；不是 root 精度升级。
3. 保留原 FP64 full-SVD/solve 路径；逆残差上限 1e-9，实际 rank64 加权残差与奇异值尾和的归一差上限 1e-9。不自动改 damping、换 solver 或降 rank。
4. 保存每个新因子的 FP64 值、直接 BF16 转换、奇异值和置换数组。中断仅重做尚未提交的一个因子。
5. 每窗口重新计算真实 GI/DG/GF，逐项对照上轮局部实验的 SSE/KL/NLL、teacher NLL 和输出能量。容差为 `1e-8 + 1e-10*abs(old_sum)`，作用于每窗口总和。失败不提交当前窗口，不能只比较总体均值。
6. 每窗口都有 BF16 teacher/student self control，KL 最大值不超过 1e-9；每个目标捕获恰好一次，检查输入相等、补偿后输出和异常时权重恢复。开始及结束核对完整 BF16 背景。

## 评估与解释

继续使用上轮 WikiText2 的 138×2048 窗口、batch1，共 282486 个预测位置。Teacher KL 为完整词表 `teacher||student`，FP64 中心化 logsumexp，chunk128；NLL 为 FP32 CE、FP64 求和；局部输出 SSE 为实际 BF16 输出之差的 FP64 平方和。保持与上轮逐窗口协议一致，不运行另一套 word-PPL。

主要比较是 `KL(permuted G) - KL(real G)`，正值表示真实 G 更好。不要求置换 G 劣于 GI。NLL 为辅助结果，SSE 描述局部误差，不能用各自 G 下的训练目标优劣替代最终预测指标。

每个模块分别报告两个 G 方法的三个种子差值，以及先在同一窗口上平均三个种子、再进行配对统计的结果。95% 区间使用循环 block bootstrap，block8窗口、2000次、seed20260913。区间条件于这三个固定置换，不反映所有随机置换的不确定性，不是三次随机化显著性检验，不把 28×3 当作独立重复，不做多重比较校正。

全体模块都报告。真实 G 有优势可支持输出坐标对应关系有价值；无优势或混合结果同样有效。普通 GF 置换同时改变对角对应关系，不能单独归因于非对角信息；不证明精确 Fisher 或真实任务全局最优。本轮 WikiText2 已被查看，明确标为事后机制诊断，不能用于挑选主结果配置。

## 运行阶段

需要与此前实验一致的两张空闲 GPU、现有环境和来源目录。启动前检查新结果所在磁盘至少 24 GiB 可用，主要用于 dense G-root 缓存和逐 token 记录。所有新文件写入独立目录；不安装依赖、不更新旧仓库、不修改模型或来源。

- `doctor`：核验来源、原局部 28 模块的完整逐 token 结果及代码，不做新 root/SVD/前向。
- `pilot`：L0 o_proj 与 L31 down_proj，12 个置换因子，各评估前2窗口的9配置。pilot 窗口独立于正式结果。
- `prepare`：pilot 通过后补齐全28模块的168个置换因子，不运行正式前向。
- `run`：要求匹配 pilot；自动补齐因子后进行全28模块×138窗口×9配置。每次保存一个完整九配置配对窗口。
- `summarize`：核验已有完整模块并输出阶段汇总，不运行求解/前向。
- `pack`：只对全部模块完整、匹配 pilot 的结果重新核验并打包。

`--max-hours 10` 是预算，不是耗时承诺。退出75表示预算/信号暂停；重跑同一命令和同一目录即可续跑。单个 SVD 或九配置窗口是原子单元，实际退出可能晚于预算；不要在旧进程未退出时启动第二个实例。需要另一次进程读取来源时会重新核验，耗时可见于 hash 日志。

## 服务器命令模板

上传私有 profile 离线包及 `.zip.sha256` 后：

```bash
BASE=/path/to/upload
TOOLS="$BASE/g_permutation_kl_v1_tools"
PY=/path/to/existing/python
cd "$BASE"
sha256sum -c g_permutation_kl_v1.zip.sha256 &&
mkdir "$TOOLS" &&
"$PY" -m zipfile -e "$BASE/g_permutation_kl_v1.zip" "$TOOLS"
export CUDA_VISIBLE_DEVICES=0,1
bash "$TOOLS/run_g_permutation_kl_v1.sh" doctor &&
bash "$TOOLS/run_g_permutation_kl_v1.sh" pilot --max-hours 10
```

看到 `PILOT COMPLETE` 后：

```bash
LOG="$BASE/g_permutation_kl_v1_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$TOOLS/run_g_permutation_kl_v1.sh" run --max-hours 10 > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

`run` 自动先求解后评估；也可先执行 `prepare --max-hours 10` 单独准备因子。暂停后重复相应启动命令。若 pilot 暂停，重复 pilot，不能跳过。

## 数据位置

输出目录由私有 profile 的 `G_PERMUTATION_DESTINATION` 指定。

| 文件或目录 | 内容 |
|---|---|
| experiment.json / source_baselines.json | 协议、来源哈希、原局部逐窗口值 |
| roots/ | 与历史逐位匹配的 DG/GF root 缓存和证书 |
| factors/ | 168个新补偿、FP64/BF16/置换/奇异值及数值证书 |
| windows/ | 各模块138个完整九配置逐 token 配对记录及状态 |
| pilot_windows/ / pilot.json | 独立 pilot 及通过标志 |
| summary.csv | 252行，28模块×9配置，SSE/KL/NLL均值 |
| paired.csv | 224行，28模块×2方法×(3种子+种子平均) |
| per_window.csv | 3864行，逐模块逐窗口汇总和复现检查 |
| factor_audit.csv | 168个置换因子的数值与置换摘要 |
| g_permutation_kl_v1_summary.tar.gz | 全部结果CSV、来源与协议、背景、pilot，以及224份root/因子元数据；不含权重或张量 |

完整包仅在全部实验完成时生成。根和因子的元数据记录服务器真实路径、大小和哈希，原始逐 token 数组保留在服务器。

开发验证：`python -B -m unittest test_g_permutation_kl_v1 -v`。打包：`python -B build_g_permutation_kl_v1.py --output-dir /path/to/release --server-profile /path/to/private/profile --update-checksums`。CPU测试不能替代服务器pilot与正式逐窗口复现检查。
