# Llama MXINT3 Full-G rank audit v1

独立数值审计，不是求解器修复。只需上传 `full_g_rank_audit_v1.py`，不需要修改实验仓库。

## 边界

- 读取冻结的 manifest、A 根、原始 G、BF16 模型权重、W3 权重和已保存 Full-G 因子。
- 每个模块同时检查 `diag_gf` / `full_gf` 的 rank 8/16/32/64。
- 不重新收集 A/G，不执行模型前向/反向，不生成或覆盖补偿因子，不改原阈值。
- 所有输出写到单独指定的审计目录，拒绝覆盖非空无主目录或与已知实验/模型/代码路径重叠的目录。
- 原始 G 的特征分解与开方仍然昂贵。单个 k_proj 的试跑耗时不能代表大 MLP 模块。

## 环境与启动

使用原来的 `qera-original-a` 环境（torch 2.3.0+cu121）以及一张空闲 RTX4090。
建议保留原双卡服务器的 224 GiB RAM 档位，但审计只使用选中的一张 GPU；不得与其他任务争抢该 GPU。

```bash
conda activate /share/home/tm902089733300000/a913520780/chengkang/conda_envs/qera-original-a
BASE=/share/home/tm902089733300000/a913520780/chengkang
RUN="$BASE/qera_runs/llama3.1-8b-diag-g-256/mxint3_full_g_v1"
OUT="$BASE/qera_diagnostics/full_g_rank_audit_v1"
LOG="$BASE/full_g_rank_audit_v1.log"

# 先检查一个新模块；预算结束退出码 75 是正常暂停。
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$BASE/full_g_rank_audit_v1.py" \
  --run-dir "$RUN" --output-dir "$OUT" --device cuda:0 \
  --max-new-modules 1 >> "$LOG" 2>&1 &
tail -n 80 -F "$LOG"
```

看到 `COMMITTED` 和 `PAUSED` 后退出 tail，再执行全量扫描。只改预算，不改输入/输出/精度选项。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$BASE/full_g_rank_audit_v1.py" \
  --run-dir "$RUN" --output-dir "$OUT" --device cuda:0 \
  --max-hours 10 >> "$LOG" 2>&1 &
tail -n 80 -F "$LOG"
```

按租期调整 `--max-hours`。预算为协作式停止：当前整个模块检查完、报告提交后退出，不能中断正在进行的大矩阵运算。
模块报告采用临时文件、fsync、原子替换提交；中断后已提交模块会跳过，最多重算正在检查的一个模块。
不删除任何检查点或临时文件。不允许同一输出目录同时启动两个审计进程。

## 报告

- `status.json`：审计进度。`AUDIT_COMPLETE` 只表示扫描完，不代表方法或原实验 PASS。
- `rank_metrics.csv`：全量逐模块、方法、rank 指标。默认完成时 224×2×4=1792 行。
- `flagged_rows.csv`：触发初筛规则的行，必须结合原始数值解释。
- `modules/*.json`：逐模块事务报告，保留输入身份、旧 rank64 指标、重建 G 谱诊断。
- `audit_config.json`：脚本 SHA256、环境、数值选项和初筛阈值。

主要字段：

| 字段 | 含义 |
|---|---|
| `sse_ratio` | 固定 A/G 度量下，补偿后 SSE / 补偿前 SSE |
| `weight_mse` | 非加权的权重重建 MSE，不是激活误差或 PPL |
| `left_orthogonality` | `S_A L` 各列的正交性偏差；反变换的初筛信号，不是 A 侧直接方程残差 |
| `correction_over_error_norm` | 补偿矩阵范数 / 原量化误差范数；大不一定错，供排序 |
| `bf16_factor_rounding_product_relative` | 因子 BF16 舍入后转回 FP32 相乘，相对原 FP32 乘积的差异；不是真实 BF16 两次 GEMM，也不包含输入激活 |
| `rank64_saved_sse_difference_over_before` | 重算 rank64 SSE 与旧报告的绝对差 / 未补偿 SSE |
| `g_root_condition_spectral_proxy` | FP64 有效 G 的谱给出的根条件数代理，不是保存后 FP32 根或 Full-A 的实测条件数 |

`SCREEN` 标记阈值仅用于排查，不改变正式实验门槛，不自动宣布某模块错误。
无标记仍不能排除端到端 NLL 退化、BF16 执行问题或校准目标泛化问题。
禁止跨不同 A/G 度量直接比较 SSE 的大小。

## 第二轮：对某个可疑模块深入检查

把模块名替换成第一轮报告中的实际模块；使用全新的输出目录。

```bash
MODULE=model.layers.0.mlp.down_proj
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$BASE/full_g_rank_audit_v1.py" \
  --run-dir "$RUN" --output-dir "$BASE/qera_diagnostics/full_g_detail_$MODULE" \
  --device cuda:0 --module "$MODULE" --fp64-check --svd-reference \
  > "$BASE/full_g_detail_$MODULE.log" 2>&1 &
```

FP64 检查只把同一份已保存 FP32 根和因子升精度再评价，不重建 FP64 A、不重新求解因子，也不能恢复 FP32 已损失的信息。
`--svd-reference` 重新对加权误差矩阵做 full SVD，输出奇异值尾能量作为理想局部目标参考，不执行反变换或保存新因子。
不同精度、参考 SVD 选项、模块选择或脚本版本不能混写同一审计目录。

## 本地验证范围

CPU 小矩阵测试覆盖：与冻结 G 根实现一致性、直接 SSE 定义、SVD 尾能量对照、rank 单调性标记、非有限值、输入哈希、输出隔离、模块端到端只读性和报告续跑。
未在本地运行真实 224 模块 GPU 审计；真实耗时、显存峰值和退化原因仍需服务器输出确认。
