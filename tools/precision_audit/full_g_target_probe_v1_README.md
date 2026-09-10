# Llama MXINT3 Full-G 单模块定点诊断

仅调查 `model.layers.0.self_attn.o_proj`。不重收 A/G，不改冻结代码、权重、因子、manifest 或原评估记录。

## 实验内容

1. `diagnose`：同一模块比较已保存的 FA+GI / FA+DG / FA+GF，在 r8/16/32/64 测量补偿范数、普通权重 MSE、各自加权 SSE 和统一 FA+GI SSE。检查保存的 A 根矩阵的 FP64 奇异谱。使用保存的 FP32 根矩阵提升到 FP64，执行新的 FP64 SVD 和线性求解，对照理论尾能量。另用完全相同的 U（先转为 FP32）比较 A 侧 FP32/FP64 线性求解。
2. `evaluate`：每个 rank 先复跑原全模型 FA+GF 控制配置，再只将第 0 层 o_proj 的补偿换成已保存的同 rank FA+DG。其余 223 个模块、量化权重、BF16 两次矩阵乘法、eager attention、balanced 双卡分配、batch=8、CE chunk=256、138 个 WikiText2 窗口全部不变。原控制 PPL 回归容差沿用冻结配置，控制失败阻止该 rank 的替换评估。

数值诊断的新因子只在内存中使用，绝不保存或用于 PPL。没有加入 damping、shrinkage、伪逆或梯度裁剪。GI/DG 的因子引用来自 Full-G manifest，DG 根来自原始 DG checkpoint，非重新收集的 Full-G 对角近似。

这是针对已观察异常的事后诊断，不是新的正式方法基准。不同 G 下的加权 SSE 不可直接横向比较；`saved_common_fa_gi_sse` 才是三者共同的 A 加权误差口径。FP64 产品不同也不自动表示 FP32 错误，须结合尾能量差距、奇异值间隔与逆变换残差。FP64 重求解仍使用保存的 FP32 A/G 根，并不恢复它们转换为 FP32 之前的精度。

## 文件与资源

将压缩包上传到服务器 `/share/home/tm902089733300000/a913520780/chengkang`。
包内两个运行必需文件必须放在同一目录：

- `full_g_target_probe_v1.py`
- 原封不动的 `full_g_rank_audit_v1.py`（SHA256 必须为 `1c39e84f7f2c1ac9dfdcbb4ee7291760cd047c8222cae0cd9404e0dd5276d3e9`）

`diagnose` 使用一张空闲 4090；`evaluate` 使用两张空闲 4090。启动前每张所需卡至少有 18 GiB 空闲显存；这不是运行峰值的保证。不与其他 GPU 任务并行运行。会读取共享存储，仍可能与其他服务器的任务产生 I/O 竞争。

保持原 `qera-original-a` 环境，不升级 torch/transformers/accelerate。CPU 合成数据与模拟模型测试通过；尚未在真实服务器 GPU/模型上实测。

## 1. 解压与变量

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE"
unzip -n full_g_target_probe_v1.zip -d "$BASE/full_g_target_probe_v1_tools"
conda activate "$BASE/conda_envs/qera-original-a"

PROBE="$BASE/full_g_target_probe_v1_tools/full_g_target_probe_v1.py"
REPO="$BASE/KFAC-QERA-98ad0c5"
RUN="$BASE/qera_runs/llama3.1-8b-diag-g-256/mxint3_full_g_v1"
OUT="$BASE/qera_diagnostics/full_g_target_probe_v1"
python -B "$PROBE" --help
nvidia-smi
```

新终端需要重新设置这些变量。脚本读取冻结 manifest 中的绝对路径；若原实验代码或文件已被移动/改写会拒绝运行，请不要删除或绕过检查。

## 2. 先做数值诊断

```bash
LOG="$BASE/full_g_target_probe_v1_diagnose.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$PROBE" diagnose \
  --repo-dir "$REPO" --run-dir "$RUN" --output-dir "$OUT" \
  --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

`Ctrl+C` 只退出这里的 tail，不会停止后台 Python。完成后提供：

```text
$OUT/diagnostics/a_spectrum.json
$OUT/diagnostics/comparison.csv
$OUT/diagnostics/full_gi.json
$OUT/diagnostics/full_gd.json
$OUT/diagnostics/full_gf.json
```

先讨论诊断结果也可以，不必立即启动全部 PPL 评估。

## 3. 双卡评估 pilot：只跑 r8 控制的一个 batch

等 diagnose 进程退出，确认两张卡都空闲后运行：

```bash
LOG="$BASE/full_g_target_probe_v1_evaluate.log"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$PROBE" evaluate \
  --repo-dir "$REPO" --run-dir "$RUN" --output-dir "$OUT" \
  --only-rank 8 --max-new-batches 1 --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

pilot 正常会保存 `CONTROL_FULL_GF_R8 window=8/138` 后暂停，退出码 75。此时没有完成 PPL，也还没有开始单模块替换。

## 4. 继续完成 r8 的控制和替换

等 pilot 退出后，删除 batch 限制，其他路径不变：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$PROBE" evaluate \
  --repo-dir "$REPO" --run-dir "$RUN" --output-dir "$OUT" \
  --only-rank 8 --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

查看：

```bash
cat "$OUT/evaluation/control_check_r8.json"
cat "$OUT/evaluation/ppl_comparison.csv"
```

负的 `delta_ppl` / `delta_mean_nll` 表示单模块替换优于此次重跑控制。改善只能说明该模块在当前全模型背景下贡献了退化，不能自动归因于某个数值步骤，也不能证明剩余模块没有问题。

## 5. 补齐 r16/32/64

删除 `--only-rank 8`，r8 已完成部分会自动跳过：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -B "$PROBE" evaluate \
  --repo-dir "$REPO" --run-dir "$RUN" --output-dir "$OUT" \
  --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

总计四个控制和四个替换配置，每个 138 窗口。每个新配置会重新加载 BF16 模型。

## 断点、保护与输出

- 原实验所有路径只读；新输出必须与模型、原结果、checkout 和工具脚本路径分离。独立目录锁阻止重复 probe 进程。不锁住原实验，不调用原 pipeline 的 evaluate/summarize 写入函数。
- 诊断：A 奇异谱和每个方法分别原子保存。若方法中途被强制杀死，只重做未提交的方法。
- 评估：每完成一个原 batch（通常 8 窗口，末 batch 为 2）原子保存。重启只重做未提交的 batch。没有更改 batch size 来规避 OOM。
- `--max-hours`/SIGTERM/SIGINT 为协作暂停，在当前计算单元完成并保存后退出。可能超过时间预算一个数值方法/一个 batch 加载计算时间；预留服务器关机余量。突然断电/SIGKILL 无法保存正在计算的单元。
- 路径、脚本 SHA、源 manifest、输入因子身份、软件版本和协议绑定到 `probe_config.json`。修改代码或固定设置须使用新的输出目录，不删除校验。预算与 only-rank 不改变实验身份。
- `evaluation/ppl_comparison.csv`：只包含控制通过且两边均完成 138 窗口的比较。
- `evaluation/paired_windows.csv`：逐窗口配对 NLL 差异。
- `evaluation/control_check_r*.json`：原 GF 控制回归结果。
- `evaluation/CONTROL_FULL_GF_R*.json` / `HYBRID_O0_GD_R*.json`：含逐 batch 续跑记录。
- `diagnostics/status.json`：DIAGNOSTICS_COMPLETE 仅表示数值诊断计算完成。
- `evaluation/status.json`：仅在四个 rank 的配对都完成时为 COMPLETE。

若报 CONTROL FAILED、哈希不符、FP64 求解失败或 OOM，保留完整日志并反馈，不放宽容差或改原数据。
