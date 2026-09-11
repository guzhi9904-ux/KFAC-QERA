# Llama DA 同根 FP64 与 Qwen 只读排查

状态：已实现 CPU 合成测试；服务器 CUDA、真实冻结产物和 harness 仍须 doctor/pilot 验证。不要把本地测试通过写成实验已完成。

## 隔离与部署

两台机器分别使用两张 4090；共同读取模型、统计和本目录工具没有写冲突。共享磁盘仍可能有 I/O 争用。不要在旧 checkout 执行 git pull，不要覆盖旧工具或在共享 conda 中 pip install。新工具不触碰 Llama Full-G 和 Qwen 原输出。

从 GitHub 在**新目录** clone 并 checkout 交付的固定 commit。工具随仓库只有一份；两台机器共享读取，不再为每阶段复制 helper。启动后保持该工具目录不变。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
TOOLS="$BASE/KFAC-QERA-tools-20260911-da-qwen-v1/tools/precision_audit"
# 在完成新目录 clone / 固定 commit checkout 后：
(cd "$TOOLS" && sha256sum -c SHA256SUMS)
hostname
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv
pgrep -af 'diag_a_fp64_dual_v1.py|qwen_full_a_audit_v1.py|qwen25_base_isolation_v1/run.py' || true
```

日志放输出目录外，避免初始化拒绝未知文件。不要同时启动两份同一任务；Llama 有独立输出锁。锁报错时先查进程，不删除锁或检查点。

## Llama：DA + GI / GD / GF

- MXINT3；复用原 Wq、FP32 Diag-A root、256×2048 校准统计。没有重新收集 A/G、量化权重或重建高精度 A-root。
- G 与已完成 FA FP64 路径一致：GI、原 DG normalization/floor、原 Full-G FP64 eigendecomposition→FP32 root，再提升到 FP64 求解。Full-G root 原来未归档，所以从冻结统计按原算法重建，不声称已与未保存的原 root 逐 bit 比较。
- 与**已完成 FA FP64 实验的 G-root bit 证书**逐一核对；不匹配即停止。因此需要保留 `qera_diagnostics/full_a_all_precision_r8_v1` 的实验、因子和证书，不从中部署 FA 因子，也不修改它们。
- FP64 加权矩阵、`full_matrices=True` SVD、A 侧逐元素逆、G 侧 dense solve；保存完整 rank64 FP64 因子及直接转成 BF16 的 rank64 因子，r8/16/32/64 取前缀。
- Diag-A 为正时逐元素逆与对角矩阵求解是同一个数学问题。遇到零/负数直接停止，避免 FP32→FP64 改变旧 epsilon 行为；不自动增加 floor、阻尼或 pinv。
- 每个模块/方法在四个 rank 上检查实际加权 SSE 与 SVD 尾能量及单调性；BF16 舍入 proxy 保留异常标志，不作为实际 PPL。
- 先跑两个大模块 × 三种 G 的 pilot；word 选中时同时完整重放 BF16 word 控制。
- token 保持 138×2048、batch8、CE chunk256，正式评估前重放三组**原始 DA r8**控制；容差 PPL 1e-5、逐窗口 NLL 1e-3，失败停止，不放宽。
- word 保持旧 4096 官方 harness、完整 62 文档协议，先重放 BF16 与旧 FA+DG r8 控制。这个 FA 控制验证共用评估路径，不混入 DA 主表。
- 两种协议各 14 个主配置：BF16、W3、3种 G×4rank。共 28 行，OLD 控制另存，不用旧 FP32 DA 结果冒充新路径。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
TOOLS="$BASE/KFAC-QERA-tools-20260911-da-qwen-v1/tools/precision_audit"
mkdir -p "$BASE/qera_diagnostics/logs"
bash "$TOOLS/run_diag_a_fp64_dual_v1.sh" doctor

# doctor 成功后，pilot 用前台运行，能直接看到异常。
bash "$TOOLS/run_diag_a_fp64_dual_v1.sh" pilot --max-hours 2

# 看到 DA PILOT COMPLETE 后正式启动；先确认没有另一个同任务进程。
LOG="$BASE/qera_diagnostics/logs/diag_a_fp64_dual_v1.log"
nohup bash "$TOOLS/run_diag_a_fp64_dual_v1.sh" run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 80 -F "$LOG"
```

服务器重开后，在拥有这两张卡的同一任务机器重新设置 BASE/TOOLS/LOG，重跑**同一条 run 命令**即可。`--max-hours` 应比服务器租期短，预算不是硬终止期限：当前 SVD/模型加载等操作可能超过边界。不得改变 protocol、数据或工具源码来续跑。

检查点：求解按**一个模块/方法**提交；token 每**8窗口**；word 每**一个完整配置**。突然断电只重做尚未提交的单位。退出75是主动暂停，不是科学检查通过；退出1等须看 traceback，不要反复盲跑。

```bash
RUN="$BASE/qera_diagnostics/diag_a_fp64_dual_v1"
cat "$RUN/solve_status.json"
cat "$RUN/token_ppl/status.json" "$RUN/word_ppl/status.json"
cat "$RUN/token_ppl/ppl_summary.csv"
cat "$RUN/word_ppl/ppl_summary.csv"
```

另存 `rank_checks.csv`、`factors/<method>/*.json`、各协议 `comparisons.csv`、`per_unit.csv`、`paired_units.csv` 和 control JSON。尚未创建的文件表示对应阶段还没提交。

## Qwen：先找原故障，再做只读重放

A 收集已经完成（14/14 shard、256窗口）。最终中断发生在 `solve-gi` 的 `model.layers.1.mlp.down_proj`、Full-A 分支，而不是 A 收集中途：

```text
diag_gi official_mse = 1.0600419045658782e-05
full_gi official MSE = 65477908.0
Weighted objective increased: 3.130071415508088 -> 25140.280772709644
```

此前 root 元数据 residual=8.834077794745269e-12 是 FP64 构根后、转 FP32 前的检查，不能保证存下的 FP32 root 在逆求解中稳定。这里仍是待定位原因，不直接移植 Llama 的结论。

在 Qwen 那台双卡机器执行以下**只读**命令（`rg` 若未安装，用 `grep -nE` 替代，不必安装包）：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
RUN="$BASE/qera_runs/qwen2.5-7b-base-mxint3-v1"
CODE="$BASE/KFAC-QERA-qwen25-base-v1"
TOOLS="$BASE/KFAC-QERA-tools-20260911-da-qwen-v1/tools/precision_audit"
hostname
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv
pgrep -af 'qwen25_base_isolation_v1/run.py|qwen_full_a_audit_v1.py' || true
tail -n 160 "$BASE/qera_runs/qwen25-base-v1.log"
rg -n -C 4 'Traceback|Weighted objective increased|Mean squared error|OutOfMemory|paused-qwen' "$BASE/qera_runs/qwen25-base-v1.log"
cat "$RUN/roots/model__layers__1__mlp__down_proj.json"
find "$RUN/statistics/a" -maxdepth 2 -type f -name '*json' -print
sha256sum "$CODE/experiments/qwen25_base_isolation_v1/stages.py"

# 不使用 GPU 的 root 结构检查；每次独立新日志，拒绝覆盖。
mkdir -p "$BASE/qera_diagnostics/logs"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
INSPECT="$BASE/qera_diagnostics/logs/qwen_a_inspect_${STAMP}.log"
(set -o noclobber; bash "$TOOLS/run_qwen_a_diagnostic_v1.sh" inspect > "$INSPECT" 2>&1)
tail -n 100 "$INSPECT"

# 只有 inspect 正常结束、选中 GPU 空闲时才做数值重放。
REPLAY="$BASE/qera_diagnostics/logs/qwen_a_replay_$(date +%Y%m%d_%H%M%S)_$$.log"
(set -o noclobber; nohup bash "$TOOLS/run_qwen_a_diagnostic_v1.sh" replay > "$REPLAY" 2>&1 &)
tail -n 100 -F "$REPLAY"
```

不要在 Qwen 主实验运行时启动 GPU replay。该步骤只用 cuda:0，不加载整个模型、不更新 correction、不写回实验目录。日志如报 manifest/code hash 不一致，不改 manifest 绕过，先回传检查。检查脚本对原环境及数据哈希有约束，启动前预期选中卡至少16GiB空闲，但不是显存容量保证。

重放固定同一保存 FP32 A-root、同一 Wq/error、同一次 FP32 full SVD 的 U/右因子，只比较 FP32 与 FP64 A 侧逆求解（另测 FP64解转FP32）。它**不是全程 FP64 SVD 实验**，也不验证重新构造高精度 A-root。最终 `AUDIT_COMPLETE_NOT_AN_EXPERIMENT_PASS` 表示诊断完成，不表示 Qwen 修好。回传这两个日志后再决定下一修复，暂不重新收集 A。
