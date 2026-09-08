# MXINT3 Full-G v1 — 双 RTX4090，可断点续跑

本版本只新增 Full-G；不修改任何旧实验代码、manifest、量化权重、A/G 统计或补偿。
新输出固定为原始 G256 目录下的 `mxint3_full_g_v1/`，与 `mxint3_v1/` 并列。
原始配置文件仍使用 `configs/llama3.1-8b-dual4090.yaml`，不要改用派生目录中的 JSON。

## 对齐的实验设置

| 项目 | 固定设置 |
| --- | --- |
| 模型及目标 | 同一 Llama-3.1-8B checkpoint，32 层 × 7 投影 = 224 模块 |
| 环境/硬件 | 原 `qera-original-a` 环境，torch 2.3.0、transformers 4.44.2；两张 RTX4090 |
| 量化 | 已完成 MXINT3 的同一 Wq：width=3，block_size=32，block_axis=-1；激活不量化 |
| A | 只读复用原来的 128 组 A256 roots；对角 A 和完整 A 均不重算、不新增 damping |
| G 校准数据 | 同一 256 个长度 2048 的冻结窗口，固定顺序，batch_size=1 |
| Teacher | FP32、参数冻结、eager attention、CPU saved activations、use_cache=False；同一双卡 device map |
| 梯度 | 同一 sequence next-token CE **sum** 的投影输出梯度；CE seed chunk=128 |
| 统计位置 | 每窗口前 2047 个有效预测位置；每模块 524032 个位置，不去均值 |
| Full-G | `sum(delta_t @ delta_t.T) / 524032`；模块内全通道，不是跨模块协方差，也不是逐 loss Fisher |
| 统计精度 | FP64 Gram 乘法 + FP64 CPU 累计；独立对角线沿用旧 FP64 square/reduce；TF32 关闭 |
| G root | FP64 对称化、trace/d 归一化、特征值下限 1e-6、FP64 eigh；root 转 FP32 |
| 求解 | FP32、`full_matrices=True` SVD；同一官方 A 逆运算；最大 rank64 的前缀取 8/16/32/64 |
| 评估 | 原 BF16 evaluator、batch=8、CE chunk=256、同一 WT2 138 窗口/282486 prediction tokens |

新增 `DIAG_GF_R{8,16,32,64}` 与 `FULL_GF_R{8,16,32,64}`，共 8 组。
`DIAG/FULL` 指 **A**，`GI/GD/GF` 分别指单位/对角/完整 **G**。
原 MXINT3 的 18 组逐窗口结果在哈希与协议检查后导入，明确记录 `reused_from`，并非重新评估。
最终汇总 26 组，不包含 MXINT4；新旧结果不是独立重复实验。

## 内存、时间、磁盘

- 全部原始 FP64 G 矩阵为 110.5 GiB。默认分 4 批，每批 8 个 Transformer 层、56 个模块，
  常驻累计矩阵约 27.625 GiB，另有 teacher、CPU 激活、临时张量和文件缓存。
- 每批都要跑完整 256 个窗口，共 **1024 次窗口前向/反向**，不是把校准样本数增加到 1024。
  同一模块仍只用原来的 256 窗口。不允许在已开始的 run 中修改分批或精度。
- 旧 DG 的四小时不能当作本次预计总时长。仅按原先约 58.6 秒/窗口计算，重复前向/反向部分
  就约 16.7 小时；另加 FP64 Gram、检查点 I/O、矩阵分解、求解和评估。实际以 pilot 为准。
- Gram 按 512 个输出通道分行做 FP64 GEMM，降低 GPU 临时矩阵占用；不保存全批梯度。
- 开始前至少留出 **160 GiB 可用磁盘**，实际建议 200 GiB 以上；这是在旧结果之外的空间。
  只保留每批当前已提交快照，以及正在写入的新快照；不会保存 32 份全量 G。
- 224GB 套餐并不保证实际 cgroup 可用内存或运行峰值。先 pilot 查看内存/显存，
  若 OOM 则停止排查，**不会自动降精度、改 batch 或跳层**。

## 必须通过的数值门禁

1. 每个完整窗口：Full-G 对角线 vs 独立旧实现 FP64 对角累计，relative L2 ≤ 1e-10。
2. 每批完成：重收集的对角 G256 vs 冻结 G256，relative L2 ≤ 1e-6；teacher NLL 相对差 ≤ 1e-6。
3. 每个模块、每种 A、每个 rank 前缀：新**稠密矩阵**求解器输入单位 G，补偿乘积复现旧 GI，≤ 0.001。
4. 同样输入本次重收集的对角 G，补偿乘积复现旧 GD，≤ 0.001。
5. Full-G 显著非 PSD、非有限数、G 逆运算残差过大、加权 SSE 上升均停止；不放宽原 GI 门禁。

新 `audit.json` 只是输入/协议审计通过，不代表以上 GPU 数值门禁已通过。
同卡也不保证每次逐 bit 一致，因此必须保留这些检查。
Full-G 的 SSE 和旧 GD 的 SSE 使用不同度量，不能直接比较大小来声称模型更好。
主要结果看固定 A/rank 下 GF 相对 GD 的 WT2 NLL/PPL，并保留逐窗口数据。

## 服务器操作

使用原环境，**不要执行根目录 requirements 安装或升级 torch/CUDA**。
仓库更新前先查看 `git status --short`；有服务器本地更改时不要覆盖。
所有路径（代码、模型、旧结果和新结果）在重启后保持不变，且必须位于持久存储。
容器销毁/租期到期会被清空的临时盘不适合保存检查点。

以下命令在仓库根目录执行。日志放在原 G 目录中，避免污染尚未初始化的新输出目录。

```bash
export DIAG_G_CONFIG="$PWD/experiments/qera_diag_g_isolation/configs/llama3.1-8b-dual4090.yaml"
LOG=/share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-diag-g-256/full_g_v1.log

# 第一次先完成输入审计 + 4 个真实收集窗口，写检查点后退出；退出码 75 是预期暂停。
nohup bash experiments/qera_diag_g_isolation/scripts/run_full_g.sh pilot >> "$LOG" 2>&1 &
echo $!
tail -n 60 -f "$LOG"
```

pilot 的 4 个窗口会直接用于正式实验，不需要删除或重算。
观察 `COMMITTED`、`[paused-full-g]`、`cpu_process_peak_GiB`、`cgroup_peak_GiB`、`gpu_peak_GiB`。
`rough_collect_eta` 是当前批采样速度外推，仅供参考，**不包含求解和评估**。
`tail` 中 Ctrl+C 只退出日志查看，不会停止后台实验。

确认 pilot 结束、内存和耗时可接受后：

```bash
# 示例：当前服务器剩余约 12 小时，给程序 11 小时时间预算。
# 按实际剩余时长修改，建议至少留出 30–60 分钟完成慢窗口/检查点。
nohup bash experiments/qera_diag_g_isolation/scripts/run_full_g.sh run --max-hours 11 >> "$LOG" 2>&1 &
echo $!
tail -n 60 -f "$LOG"
```

服务器重新启动后，激活原环境、进入**相同代码路径**，重新设置以上环境变量和 LOG，
再执行相同 `run --max-hours ...` 命令即可。不要同时启动两个进程；OS 文件锁会拒绝并发写入。
正常完成退出码 0，预算/信号/pilot 安全暂停为 75，实际错误是非零失败且有 traceback。

手动安全停止可以对你记录并确认的 Python 进程 PID 发 `kill -TERM PID`。
程序会等待当前工作单元结束再提交；`--max-hours` 是协作预算，不是硬超时。
不要直接 `kill -9`，除非必须；SIGKILL、断电只能恢复到此前已落盘的检查点。

## 恢复粒度与故障语义

- **收集**：默认每 8 个窗口保存。到达时间预算、收到 SIGTERM/SIGINT、pilot 结束或批次完成时，
  额外保存当前完整窗口。一个窗口的所有模块成功才可提交；异常时 RAM 中的半窗口不保存。
- **事务**：先逐模块写新 generation、fsync 并计算哈希，再写完整 STATE，最后原子替换 CURRENT 指针。
  写到一半被杀时，旧指针仍有效。重启只加载它；未发布片段不会被加进统计。
- **滚动清理**：只清理本版本带匹配 OWNER 的过期/未提交 generation；当前原始 G 永久保留。
  不递归删除、不清理旧 DG/MXINT3 目录；出现外来文件或 symlink 拒绝自动清理。
- **求解**：每模块/每种 A 的门禁及 GF 补偿分别提交；已通过且哈希一致的补偿不重算。
  中断时正在做的 SVD/eigh 需要重做，但不会重新收集 G。
- **评估**：沿用原 evaluator，每批 8 个窗口提交；重新启动时跳过已评估窗口及完成的配置。
- **配置/代码不一致、文件损坏**：停止并报告，不静默从零开始或自动放宽容差。
  不支持改变路径后直接续跑，manifest 使用绝对路径；迁移应另行审计。

可分别执行 `prepare`、`collect`、`solve`、`evaluate`、`summary`。
`evaluate --only FULL_GF_R64` 可以只算一组；`summary` 不需要 GPU，但服务器仍需原 Python 依赖。

结果位置：

```text
<原 G256 目录>/mxint3_full_g_v1/
  manifest.json / audit.json
  statistics/checkpoints/shard_000..003/CURRENT.json
  statistics/checkpoints/shard_*/gen_*/layer_*.safetensors
  statistics/audit_shard_*.json
  statistics/runtime.json
  diagnostics/solver_gates/*.json
  corrections/diag_gf/*.safetensors
  corrections/full_gf/*.safetensors
  evaluation/ppl_summary_wikitext2.csv
  evaluation/wikitext2_per_window.csv
```

## 本地验证

```bash
python -m pytest experiments/qera_diag_g_isolation/full_g_v1/test_full_g.py -q
PYTHONPATH=src python -m pytest experiments/qera_diag_g_isolation experiments/qera_original_a_isolation tests -q
```

CPU 测试包含矩阵方向、对角退化、所有 rank 前缀、真实 collector 循环的重复启动、
半窗口反向异常、checkpoint 在不同写入阶段被打断、提交后崩溃、求解部分完成恢复、
评估先落盘再暂停、协议/哈希拒绝和 26 组汇总。CPU 测试不能代替双 4090 实测和数值门禁。
