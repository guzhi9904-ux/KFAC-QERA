# Qwen Full-A 诊断 v2：版本核对口径修复

v1 冻结脚本不修改。v2 复用其全部哈希核验、root 检查、SVD 和逆求解函数，新增独立入口，日志明确记录 v2 和 v1 的两个代码哈希。不是修复 Qwen correction，也不重建 root、收集 A/G 或跑 PPL。

原因：原实验 environment 用 `importlib.metadata.version`，v1 replay 却与 `torch.__version__` 直接比较。用户回传八项 distribution 版本全部一致（torch=2.3.0），运行时为 2.3.0+cu121、CUDA12.1。v2 按同一来源逐项严格比较全部八项版本，另外严格核对上述已确认 runtime/CUDA；不截去版本后缀，不伪造 torch.__version__，不放宽真实版本差异。

本轮 inspect 已验证目标 root/raw 文件哈希；FP32 root 18944平方，有限、无零行、对角正，相对不对称度约1.856e-10。这些不是条件数或可逆稳定性证明。Qwen 根因仍待 replay。

## 离线上传，不 clone/pull

上传 `qwen_a_audit_v2.zip` 到 BASE；它只新增 v2 工具，**依赖保留原 `precision_tools_25813e5` 包中的 v1 helper**，启动时严格核对其 SHA256。不覆盖 Llama 正在使用的文件。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE"
unzip -n qwen_a_audit_v2.zip
QTOOLS="$BASE/qwen_a_audit_v2/tools/precision_audit"
(cd "$QTOOLS" && sha256sum -c SHA256SUMS.qwen_v2)

# 在 Qwen 机器确认选中 GPU 没有其他实验进程。
hostname
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv
pgrep -af 'qwen25_base_isolation_v1/run.py|qwen_full_a_audit_v[12].py' || true

mkdir -p "$BASE/qera_diagnostics/logs"
REPLAY="$BASE/qera_diagnostics/logs/qwen_a_replay_v2_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$QTOOLS/run_qwen_a_diagnostic_v2.sh" replay > "$REPLAY" 2>&1 &
tail -n 100 -F "$REPLAY"
```

需要选中卡至少16GiB空闲（不代表保证不会 OOM）。会重新做只读文件/root检查，然后固定FP32 root、error和同一组FP32 SVD triplets，比较 FP32逆、FP64逆、FP64逆转回FP32；无阻尼或fallback。保留单次诊断的完整日志；中断重新执行这一小诊断，不需要重跑A收集。`AUDIT_COMPLETE_NOT_AN_EXPERIMENT_PASS` 不是方法正确性/科学通过标志。

优先回传 `environment`、`svd`、`fp32_inverse`、`fp64_inverse`、`inverse_comparison`、`fp64_inverse_cast_fp32`，异常时回传 traceback。保持旧原始实验、manifest、共享conda与Llama任务不变。

打包检查：此次校验清单强制LF，并直接验证ZIP内原始字节及其每项SHA256；修复上一包清单CRLF导致Linux把回车当文件名的问题。旧包可继续通过 `tr -d '\r' < SHA256SUMS | sha256sum -c -` 校验，无需覆盖它。
