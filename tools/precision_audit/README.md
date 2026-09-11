# 数值诊断与精度实验

这里集中归档先前单独交付的脚本；不为每个 ZIP 再复制一套 helper，不提交 ZIP、模型、统计或评估大文件。脚本和 shell 与已交付版本逐字节相同，`SHA256SUMS` 保存校验值。旧阶段 README 保留其独立实验的操作和边界。

## 选择入口

| 阶段 | 入口 / 说明 | 已知状态（用户回传，2026-09-11） |
| --- | --- | --- |
| 全模块 Full-G 原始 rank 审计 | [full_g_rank_audit_v1.py](full_g_rank_audit_v1_README.md) | 已执行；不是数值修复 |
| 目标模块谱与 FP64 重求解 | [full_g_target_probe_v1.py](full_g_target_probe_v1_README.md) | L0 o_proj 已诊断 |
| A4：单模块 OLD / FP64 / ZERO | [full_g_precision_r8_v1.py](README_full_g_precision_r8_v1.md) | r8 三组完成 |
| A5：全模块 FA+GI/DG/GF 同根 FP64 | [full_a_all_precision_r8_v1.py](README_full_a_all_precision_r8_v1.md) | r8 六组完成；保存 rank64 因子 |
| A5 扩展：四个 rank × token/word PPL | [full_a_all_ranks_dual_ppl_v1.py](README_full_a_all_ranks_dual_ppl_v1.md) | 已回收两套各14行汇总；原始控制文件待回收 |
| DA 扩展：同根 FP64 × 四 rank × 双 PPL | [diag_a_fp64_dual_v1.py](README_diag_a_fp64_dual_v1.md) | 新实现，待服务器 doctor/pilot；不覆盖旧阶段 |
| Qwen Full-A 只读诊断 | [qwen_full_a_audit_v2.py](README_qwen_a_diagnostic_v2.md) | v2修正版本记录口径；复用冻结v1数学函数，异常未关闭 |

上述 FP64 实验仍使用既定根矩阵的数值，不是“原始 FP64 A 重新构造高精度 root”。四 rank 扩展复用 A5 因子，没有重新收集 A/G 或重做 SVD。token-PPL 与 4096 word-PPL 分开保存，不能相互换算或混合排名。结果见[总账](../../docs/research/PPL_SUMMARY.md)，故障与处理见[日志](../../docs/research/ISSUE_LOG.md)。

新增 DA 入口需要重新做 DA 的 FP64 SVD/逆求解，但不收集统计；共用冻结双 PPL evaluator。两台双卡机器的启动、续跑和 Qwen 只读排查见[操作说明](README_diag_a_fp64_dual_v1.md)。

## 已在跑的实验

**什么也不必更新。** 继续使用原 `*_tools/`、原输出目录和原续跑命令。本次只归档源码，不接触服务器进程、环境、检查点或后续续跑依赖。

所有阶段共用这里的一份冻结 helper；`frozen_harness_word_ppl.py` 是已发布双 PPL 包要求的固定哈希副本，不能用随时变化的模块引用替换。它与 `experiments/qera_original_a_isolation/harness_word_ppl.py` 的固定版本相同。

## 未来重新部署（不是当前运行任务的更新步骤）

从独立 checkout 的固定 commit 导出工具。将下面 `REV` 替换为实际完整提交号，并选择**尚不存在**的工具目录；不要覆盖正在运行的工具目录，也不要在冻结的旧实验 checkout 中拉取更新。

```bash
REV=<固定的完整提交号>
TOOLS=/path/to/new_precision_tools
mkdir "$TOOLS" && git archive "$REV:tools/precision_audit" | tar -x -C "$TOOLS"
(cd "$TOOLS" && sha256sum -c SHA256SUMS)
```

导出保持原文件名，兼容脚本内相邻 helper 的导入。随后按对应阶段 README 审计外部数据、源码和依赖路径。旧 README 中的 ZIP 只是此前交付方式；本目录导出提供相同运行文件，不需要下载多份 ZIP。

## 本地 CPU 回归

```bash
python -B tools/precision_audit/run_cpu_tests.py
```

CPU 合成测试不能代替 CUDA、服务器冻结依赖及最终端到端控制。当前 pilot 仅验证两个大模块和 BF16 word 控制；不能宣称全部 rank / 配置已完成。
