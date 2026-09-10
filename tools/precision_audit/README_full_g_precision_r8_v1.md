# Llama MXINT3 FA+GF：r8 单模块精度干预

这是独立的事后诊断实验。实现及 CPU 测试完成；真实双卡 4090 的结果必须在服务器运行后得到。

## 三组与冻结边界

- OLD：原始 FA+GF 全模型，rank=8，重新评估。
- FP64：仅 model.layers.0.self_attn.o_proj 的 correction 改为 FP64 重求解得到的因子。
- ZERO：仅该模块不安装 correction hook；该模块的 Wq 保留不变。

保持原始 MXINT3 Wq、A/G 统计、其他 223 个模块的 correction、tokenizer 与评估窗口不变。
不重新收集统计、不训练、不增加 shrinkage、不改变 G floor、不更改旧实验文件。
对实际部署的全模型 parameters/buffers、BF16 correction 张量逐字节哈希，要求三组只有目标 correction 不同，device_map 也相同。这个保证针对张量，不承诺不同运行的所有中间激活逐 bit 相同。

FP64 路径沿用此前诊断：将存储的 FP32 Full-A 根和按冻结协议重建的 FP32 Full-G 根提升为 FP64，完成加权矩阵乘法、full SVD、rank64 逆变换求解，取前 8 个分量。不是重新从原始 A 统计计算一个 FP64 根。
在新目录保存 FP64 rank64 因子，再直接转换出 BF16 rank8 因子；部署仍是原来的 BF16 两次矩阵乘法 correction hook，不合并进 Wq。检查 FP64 逆变换残差和 r8 加权目标与 SVD 尾能量的一致性。

评估沿用原始 138×2048 WikiText2 窗口、282486 个 prediction tokens、batch=8（最后一批 2）、CE chunk=256、BF16、balanced 双卡、原加载器及 attention 设置。使用原始 NLL 归约顺序，首批额外与冻结 evaluator 的 NLL 函数做精确一致性检查。
OLD 完成后按原 control_ppl_tolerance 与冻结 FULL_GF_R8 结果比较，失败立即阻止 FP64/ZERO。报告同时保留最大逐窗口 NLL 回放差异；PPL gate 通过不代表逐窗口逐 bit 完全相等。

## 部署与启动

将 full_g_precision_r8_v1.zip 上传到服务器 BASE 目录。不要解压进旧仓库或旧输出目录，也不要编辑三个 Python 脚本。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
mkdir -p "$BASE/full_g_precision_r8_v1_tools"
unzip -n "$BASE/full_g_precision_r8_v1.zip" -d "$BASE/full_g_precision_r8_v1_tools"
nvidia-smi
```

要求两张 4090 空闲、各至少 18 GiB 可用。启动器使用原 qera-original-a 环境的绝对 Python 路径，不需要安装或升级依赖。
代码/数据隔离不能消除共享磁盘 I/O 竞争；不要与主实验共享正在占用的 GPU。

建议先 pilot，生成/冻结新因子，然后评估 OLD 的第一批 8 个窗口，提交后以退出码 75 暂停：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_g_precision_r8_v1.log"
nohup bash "$BASE/full_g_precision_r8_v1_tools/run_full_g_precision_r8_v1.sh" \
  --max-new-batches 1 --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

看到 COMMITTED OLD window=8/138 和 PAUSED 后，Ctrl+C 退出 tail，再启动正式续跑：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
LOG="$BASE/full_g_precision_r8_v1.log"
nohup bash "$BASE/full_g_precision_r8_v1_tools/run_full_g_precision_r8_v1.sh" \
  --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

服务器重启后也使用上述正式续跑命令。每次启动重新获得最多 10 小时预算，可调整 --max-hours；不要同时启动两个实例。目录内文件锁会拒绝重复实例。
时间预算/SIGTERM/SIGINT 在安全边界生效，不保证严格到秒停止；为租期预留停机余量。
每个 batch 原子提交 token 文件和进度 JSON，通常每 8 个窗口保存一次；意外掉电只需重算最后未提交的 batch。保存完成的新因子不会重新求解。完整 OLD/FP64/ZERO 配置也会跳过。
不要覆盖源码或删除输出中的检查点再续跑；源码/输入/环境不一致会拒绝混用结果。

## 输出

所有新增结果仅在：

```text
/share/home/tm902089733300000/a913520780/chengkang/qera_diagnostics/full_g_precision_r8_v1/
  experiment.json
  factors/l0_o_fp64.safetensors
  factors/l0_o_fp64.json
  evaluation/control_check.json
  evaluation/deployment_OLD.json
  evaluation/deployment_FP64.json
  evaluation/deployment_ZERO.json
  evaluation/OLD.json
  evaluation/FP64.json
  evaluation/ZERO.json
  evaluation/ppl_summary.csv
  evaluation/comparisons.csv
  evaluation/per_window.csv
  evaluation/paired_windows.csv
  evaluation/tokens/{OLD,FP64,ZERO}/batch_*.safetensors
  evaluation/status.json
```

主要看 ppl_summary.csv、comparisons.csv、control_check.json；数值诊断看 factors/l0_o_fp64.json。
comparisons.csv 的差值是名称所示的前项减后项，例如 FP64_minus_OLD 为 FP64−OLD，负 ΔNLL/ΔPPL 表示 FP64 更好。
PPL 使用原生逐窗口 NLL 总和。额外的 token_nll_sum_fp64 是对已算出的 token loss 进行 FP64 累加，不是 FP64 模型推理；可能与原生归约略有差别，不能混为同一 headline 指标。
逐 token 差异用于检查正负抵消，不把相关 token 当独立重复做显著性声明。本轮不计算 teacher-KL。

三组各需要一次完整评估，ZERO 不是免费的。新增 token 文件若 loss 为 FP32 总计约 3.4 MB，目标 FP64 因子约 4 MiB，另有少量 JSON/CSV；不会复制完整模型、A/G 或其余 correction。启动时哈希校验会读已有大文件，可能耗时数分钟，终端有心跳日志。

## 解释边界

- FP64 比 OLD 好：支持这一个模块的求解精度路径对既有 endpoint 劣化有贡献，不证明所有模块或所有 rank 已修复，也不单独定位 SVD/逆变换中的哪一步。
- OLD 比 ZERO 差且 FP64 比 ZERO 好：支持旧 correction 有害，新 correction 在该全模型上下文中有效。
- PPL 近似不变：不能推出数值问题不存在或模块本质上不敏感；还需查看 token/window 正负抵消以及 BF16 部署后的实际改变。

这是在冻结评估集上针对已知异常的事后诊断，不是无偏的全模型方法优劣结论。

## 本地验证

CPU 测试覆盖数学目标、直接 BF16 转换、只改目标模块、零补偿语义、三组批次续跑、输入文件不变、token 检查点篡改拒绝、控制回放失败阻断等。连同旧辅助脚本回归测试共 27 项通过。未在本机执行真实 CUDA 模型评估，pilot 是服务器端必需的验证步骤。
