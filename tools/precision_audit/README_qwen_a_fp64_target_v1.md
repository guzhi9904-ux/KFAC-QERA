# Qwen 单模块同根全程 FP64 诊断

固定 Qwen2.5-7B Base / MXINT3 / `model.layers.1.mlp.down_proj` / Full-A + G=I。保持已有保存的FP32 root、Wq和数据不变，将加权矩阵乘法、full_matrices=True SVD、A侧和恒等G侧求解都放到FP64。没有重建FP64 root，没有重新收集A/G，没有阻尼/伪逆/失败fallback，不部署到模型、不跑PPL。

这才与Llama第一步的求解精度范围对齐。前一轮Qwen replay只改逆求解为FP64，SVD仍是FP32；二者不同。新程序固定同一原始量化error口径 `(W.float()-Wq.float()).T`，再提升FP64。

## 产物和检查

- 新目录 `qera_diagnostics/qwen_a_fp64_target_v1`；求解对象只有一个模块，保存rank64两侧FP64因子、两侧直接BF16转换因子和完整加权矩阵奇异值（不是A-root谱）。
- 检查r8/16/32/64：加权SSE与SVD尾能量、逆残差、correction/error范数比、普通权重MSE、BF16舍入proxy。
- 使用Llama同根精度门槛：逆残差1e-9，目标相对尾能量差1e-9；失败仍保留完整诊断与隔离候选因子，不放宽门槛、不回写正式correction。
- BF16 proxy是在FP64中计算BF16舍入因子的乘积，不等于实际BF16激活执行或PPL。其目标变大标志单独保留；数值检查通过不自动授权部署。
- `report.json` 的 `status=DIAGNOSTIC_COMPLETE` 只表示诊断完整，必须一起看 `checks` / `candidate_status`。退出0表示FP64门槛和BF16有限性通过，不代表BF16 proxy全通过或科学成功；退出2表示数值门槛/有限性失败但已保存诊断。traceback表示执行失败。
- 一个模块完成后原子提交report；同指令重跑会核验已完成产物，不重复求解。中途突然退出只重算这一个模块，不重收统计。输出目录有进程锁，不并发启动同一命令。

## 离线部署

不要求服务器clone/pull。把 `qwen_a_fp64_target_v1.zip` 上传到BASE并解压到新目录，不修改Llama正在使用的原包。需保留前两个包：`precision_tools_25813e5`、`qwen_a_audit_v2`；新入口只读复用其经SHA256固定的基础工具，不复制helper。

在Qwen机器执行：

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE"
unzip -n qwen_a_fp64_target_v1.zip
QTOOLS="$BASE/qwen_a_fp64_target_v1/tools/precision_audit"
(cd "$QTOOLS" && sha256sum -c SHA256SUMS.qwen_fp64_target)

hostname
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv
pgrep -af 'qwen25_base_isolation_v1/run.py|qwen_full_a_audit_v[12].py|qwen_a_fp64_target_v1.py' || true
```

确认cuda:0空闲（至少18GiB；不保证峰值一定足够），不要与另一Qwen任务争抢这张卡。只使用本机cuda:0；另一台机器的Llama不受代码或GPU影响，共享磁盘仍可能有读I/O争用。

```bash
mkdir -p "$BASE/qera_diagnostics/logs"
LOG="$BASE/qera_diagnostics/logs/qwen_a_fp64_target_v1_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$QTOOLS/run_qwen_a_fp64_target_v1.sh" > "$LOG" 2>&1 &
tail -n 100 -F "$LOG"
```

结果：

```bash
OUT="$BASE/qera_diagnostics/qwen_a_fp64_target_v1"
cat "$OUT/rank_metrics.csv"
cat "$OUT/report.json"
```

另有 `experiment.json`（来源/代码/环境/阈值）、`candidate_factors.safetensors`，执行异常时有 `failure.json`。不把候选文件复制进原RUN的corrections目录；先回传report和CSV再决定下一步。若报告未生成，回传日志结尾。
