# 全模块 FA+GI/GD/GF：第一步 FP64 求解，r8

实现及 CPU 测试完成；本机没有服务器 GPU，真实 CUDA 试跑与评估尚未执行。请先运行 pilot，把内存和耗时测出来。

## 冻结范围与科学问题

模型 Llama-3.1-8B、原 MXINT3 Wq、原 A/G 统计、原 Full-A FP32 根、原评估窗口与协议不变。研究范围是全部 224 个线性投影，三个方法 full_gi/full_gd/full_gf。全模型推理仍为 BF16。

这不是第二步。不会从原始 A 统计重新构造 FP64 A-root，也不会重新收集 A/G。

- A：读取保存的 FP32 Full-A root 原值，提升到 FP64。
- GI：单位阵。
- DG：从原始冻结 DG 检查点按原计数、归一化、floor 构造 FP32 对角根。
- GF：从已有冻结 FP64 Gram 按原始 FP64 eigh、归一化和 floor=1e-6 重建 FP32 根。核对其对角与原 DG 一致。
- 求解：上述 FP32 根提升到 FP64，加权乘法、full SVD、rank64 A/G 两侧 torch.linalg.solve 均用 FP64。保留前 8 个分量用于本轮评估。
- 部署：保存新 FP64 rank64 因子，同时冻结直接转换的 BF16 r8 因子；仍使用原 `(input @ A) @ B` 两次 GEMM hook，不合并进 Wq。

本实现沿用已经完成的单模块 FP64 路径。它改变整个求解路径，不单独归因于 SVD 或某侧逆变换。GI/DG/GF 都统一采用稠密 G 路径；不使用 reduced SVD、随机 SVD、自动伪逆、阻尼或数值失败回退。

重要限制：A-root 直接复用存储文件；G-root 是按相同冻结协议重建，不是从旧运行保存的 G-root 文件读取。因此能保证原始统计和根构造协议不变，不能声称新旧 G-root 的逐 bit 一致性已被验证。新运行会记录实际使用的 A/G 根张量哈希。旧运行没有归档的 G-root 精度信息不会凭空恢复。

## 六组评估

OLD full_gi/full_gd/full_gf 各重新评估一次；FP64 full_gi/full_gd/full_gf 各评估一次。总计 6×138 个 WikiText2 窗口。只评估 r8，不自动扩展 r16/r32/r64。

正式 run 顺序：

1. 按冻结 manifest、文件哈希、DG 检查点核对输入。
2. 三个 OLD 逐一回放，对照原结果；任何一个失败立即停止，FP64 评估不会继续。
3. 逐模块、逐方法求解和保存，共 224×3=672 份。已有有效新因子直接跳过。
4. 三个 FP64 全模型评估及成对汇总。

三个 OLD 沿用原 PPL tolerance（从配置读取），同时报告最大窗口 NLL 偏差。PPL gate 不等价于逐窗口严格相同；报告中应检查实际回放偏差，尤其对小幅改善。

所有六组都核对实际部署的 parameters/buffers 逐字节哈希和 device_map 一致，只有 correction 允许不同；每个配置还绑定准确的因子文件哈希。各组都使用全部 224 个模块，不混入单模块替换实验的新因子或检查点。

评估固定 BF16、原模型加载器及 eager attention、balanced 双 GPU、batch=8、最后一批 2 个窗口、CE chunk=256、138×2048、282486 prediction tokens。PPL 沿用原始逐窗口 NLL 求和顺序，每次模型加载后的首批与原 evaluator 的 NLL 函数精确核对。附加保存 token NLL；其 FP64 累加仅是诊断，不改变 headline NLL。

## 上传和试跑

把 full_a_all_precision_r8_v1.zip 上传到 BASE 目录；解压到独立工具目录，不更新旧仓库，不覆盖之前的单模块工具包或输出。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
mkdir -p "$BASE/full_a_all_precision_r8_v1_tools"
unzip -n "$BASE/full_a_all_precision_r8_v1.zip" -d "$BASE/full_a_all_precision_r8_v1_tools"
nvidia-smi
```

请确保两张 4090 空闲、各至少 18 GiB 可用。启动器显式使用原 qera-original-a 环境的 Python，不安装或升级任何依赖。

pilot 固定两个模块：L0 gate_proj（较大的 G 侧）和 L0 down_proj（较大的 A 侧），每个都求解 GI/DG/GF；然后试评估 OLD full_gf 的一个 batch。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_a_all_precision_r8_v1.log"
nohup bash "$BASE/full_a_all_precision_r8_v1_tools/run_full_a_all_precision_r8_v1.sh" \
  pilot --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

正常首轮 pilot 在保存 OLD full_gf window=8/138 后，输出 PAUSED，退出码 75 是正常预算暂停。pilot 不是完整科学回归通过。记录每次 COMMITTED 求解的耗时、CPU 峰值和 GPU 峰值，先检查是否能容纳最大矩阵；不要用之前一个 o_proj 的时间外推全模型。

如 pilot 被服务器租期中断，再运行 pilot 会跳过已经保存的因子。已完成首批评估也会跳过。

## 正式运行与续跑

pilot 正常完成并检查资源后，Ctrl+C 退出 tail，再执行：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_a_all_precision_r8_v1.log"
nohup bash "$BASE/full_a_all_precision_r8_v1_tools/run_full_a_all_precision_r8_v1.sh" \
  run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

服务器重启后，用同一 run 命令。旧 pilot 的 6 份求解和评估首批会复用。每次启动独立计算最多 10 小时预算；租期不足时调整该参数，并预留一个大型求解事务的收尾时间。

- 求解阶段：完成一种方法的一个模块就提交，最多重算一个未提交的求解事务，包括该方法根重建/SVD/逆变换/检查。不会因中断而重算已保存的其他方法。
- 评估阶段：每 8 个窗口原子提交，最多重算最后未提交的 batch。
- 时间或信号只在安全边界停止，不保证正好到秒结束，无法在一次大型 GPU 分解内部保存进度。
- 不要同时启动两个实例。输出目录文件锁拒绝并发写入。
- 不要覆盖脚本、编辑 experiment.json、混用输出或删除有效旧检查点来绕过校验。
- 数学检查失败或 OOM 会停止，不会跳过模块、降精度或自动更换算法。记录在 last_failure.json，贴出报错后再分析。

可选命令 solve 只生成因子，evaluate 要求全部新因子齐全并重新确认 OLD 控制结果后评估。正常使用 pilot/run 即可。
开发用 --max-new-solves N / --max-new-batches N 限制本次新增事务，正式续跑记得去掉；它们不改变实验身份。

## 资源与隔离

求解时只使用 cuda:0，逐模块串行，不把整个模型常驻显存。评估才使用两张 GPU。两张卡不会自动并行分摊一个 full SVD；本版也没有并行求解 worker。

理论新因子 payload 约 3.87 GiB（FP64 rank64 + BF16 r8，三种方法全部模块），另有少量 JSON/CSV 和约 6.8 MB 的 FP32 token loss。不会复制完整模型或全量 A/G 根；G 根只保留哈希和诊断。脚本按实际模型 shape 计算存储量并检查剩余空间，建议至少留 8 GiB 空闲余量。

程序会读大文件并做哈希，即便输入只读，仍可能争用共享磁盘带宽。代码和输出隔离不代表计算或 I/O 资源隔离；不要与主实验争抢相同 GPU。

## 输出与解读

```text
/share/home/tm902089733300000/a913520780/chengkang/qera_diagnostics/full_a_all_precision_r8_v1/
  experiment.json
  factors/{full_gi,full_gd,full_gf}/model__layers__*.safetensors
  factors/{full_gi,full_gd,full_gf}/model__layers__*.json
  solve_metrics.csv
  solve_status.json
  evaluation/common_deployment.json
  evaluation/{full_gi,full_gd,full_gf}/control_check.json
  evaluation/{full_gi,full_gd,full_gf}/deployment_{OLD,FP64}.json
  evaluation/{full_gi,full_gd,full_gf}/{OLD,FP64}.json
  evaluation/{full_gi,full_gd,full_gf}/tokens/{OLD,FP64}/batch_*.safetensors
  evaluation/ppl_summary.csv
  evaluation/comparisons.csv
  evaluation/paired_windows.csv
  evaluation/per_window.csv
  evaluation/status.json
```

求解检查要求 FP64 两侧逆变换相对残差 ≤1e-9、r8 目标与 SVD 尾能量之差除以补偿前目标的绝对值 ≤1e-9、因子有限、BF16 张量为直接转换。保存原/新 correction 范数、SSE 和舍入敏感性。
BF16 舍入因子的 FP64 目标是局部代理，不是实际 BF16 激活执行；若比未补偿目标还差会记录 rounded_objective_increased_flag，但不会偷偷重新平衡因子或修改方法。最终应同时检查这些 flags 和 PPL。

主要交回三个文件：evaluation/ppl_summary.csv、evaluation/comparisons.csv、solve_metrics.csv，另附三个 control_check.json。
comparisons 包括三种方法各自 FP64−OLD，以及 FP64 精度统一后的 DG−GI、GF−GI、GF−DG。差值命名中前项减后项，负 ΔNLL/ΔPPL 为前项更好。

优先用 FP64 三方法的同精度结果比较 G 的增量；OLD→FP64 用于描述求解路径影响。不把相关 token 当独立实验重复做显著性声明，不把一次 r8/WikiText2 结果扩展为所有 rank、数据域、模型的结论。

这仍然是第一步；FP64 A-root 重建没有实现或自动排队。

## 测试

新脚本的合成 CPU 测试覆盖所有方法的根定义、矩形投影方向、SVD 目标、直接 BF16 转换、672 份事务的缩小版续跑、异常不提交、文件篡改拒绝、六组评估续跑、Wq/参数不变及配对汇总。辅助脚本保持原哈希。新测试与原单模块、审计回归合计 36 项通过。真实 PyTorch 2.3.0+cu121 / 双4090 大矩阵资源测试需要服务器 pilot。
