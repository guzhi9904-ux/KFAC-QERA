# Qwen DA/FA 同根 FP64 + 双 PPL

这是独立新实验入口，不更新旧 Qwen runner，不触碰 Llama 代码、输出、进程或共享 conda。

当前发布 v1a 修复 Python namespace package 的 `__file__=None` 导入审计兼容性：核对包的唯一搜索目录属于冻结 checkout 且有 manifest 源码锚点，实际 `.py` 哈希校验不减弱。数学、数据与两种 PPL 实现不变。v1 在导入前置审计退出，尚无求解/PPL提交；v1a 使用新输出，不修改旧 `experiment.json` 的版本指纹，也不删除失败记录。

## 固定设置与范围

- Qwen2.5-7B **Base**，原 MXINT3 / block32 / axis-1 的 Wq；QKV bias 不变。
- 只读原 `qwen2.5-7b-base-mxint3-v1` manifest、模型、112 组已保存 A-root、196 个 Wq。
- A 校准仍来自原 256×2048；不重新收 A/G、不重建 root、不重新量化。
- DA：原 FP32 对角 root 向量转 FP64，行缩放/full SVD/除法全 FP64。非正 root 直接报错，不新增 floor。
- FA：直接复用已经通过单模块实测的哈希固定 `qwen_a_fp64_target_v1.solve`，同 FP32 root 转 FP64，乘法/full SVD/两侧逆求解全 FP64，G=I。
- DA+GI 与 FA+GI 全 196 模块重新求解；不把旧 FP32 correction 当作 FP64 结果。单目标诊断因子此次也不自动收编，pilot 用相同冻结函数重算。
- 保存 rank64 FP64 因子及直接转换的完整 BF16 因子，评估截取 8/16/32/64；两条 PPL 使用同一套因子。
- 数值门槛保留 1e-9：逆残差、实际加权 SSE 与 SVD 尾能量相对差、rank 单调性；另外 BF16 因子有限且舍入代理无超过既定容差的目标上升。失败隔离候选，不放宽、不回退 FP32、不用伪逆或阻尼。
- BF16/W3 + 两种 A×四个 rank = **每种 PPL 10 个配置**。这次没有 DG/GF。

## 两种 PPL 不是同一指标

1. Token-PPL：复用原 Qwen evaluator 的模型加载、CE 实现和冻结 Qwen tokens，143×2048，batch8，CE chunk256；每个窗口2047 prediction tokens。原 BF16/Wq 若已有完整结果则校验回归；没有时明确记 `NO_FROZEN_REFERENCE`，不伪称通过旧结果回归。
2. Word-PPL：沿用 Llama 的固定 lm-eval harness、QERA `auto-balanced` 放置、4096 context、完整62篇文档和241335 scored words；换成 Qwen 自身模型/tokenizer、不应用 chat template。使用相同 `HFLM(model)` 与 `simple_evaluate(batch_size="auto")` 路径，实际 HFLM batch 值单独记录，不把请求参数误当实际批量。调用冻结 harness 的滚动评分/分词/词数/聚合，并复用已发布双 PPL helper 的结果校验。**不拿 Llama 的 BF16 PPL 当 Qwen 的回归阈值**。

Word 数据集、依赖、harness 和原 4096 协议必须一致；离线缺依赖/缓存就停止，不安装或下载。原始文档/hash相同，不复用 Llama token IDs。

## 离线部署

上传 `qwen_gi_fp64_v1a.zip` 到共享 BASE，解压出独立工具目录。ZIP 内包含所需冻结 helper；不要求替换之前两份工具包，也不要求服务器 git clone/pull。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE"
unzip -n qwen_gi_fp64_v1a.zip
QTOOLS="$BASE/qwen_gi_fp64_v1a/tools/precision_audit"
(cd "$QTOOLS" && sha256sum -c SHA256SUMS.qwen_gi_fp64)
hostname
nvidia-smi
pgrep -af 'qwen25_base_isolation_v1/run.py|qwen_a_fp64_target_v1.py|qwen_gi_fp64_v1.py' || true
```

只在 Qwen 的双4090机器启动，不与其他 Qwen GPU 程序并跑。共享模型/旧输出只读；同时读共享磁盘可能有 I/O 竞争，但不会修改 Llama 续跑文件。新包包含旧 helper 的冻结副本，仓库仍只有一份对应源文件。

已有依赖路径由 shell 固定：`KFAC-QERA-qwen25-base-v1`、`QERA-official-bd7fc86`、`QERA-harness-3823cfe`、`qera_official_bf16_pydeps`、共享数据缓存与原 `official-word-ppl-4096-existing-artifacts/protocol.json`。不更新这些目录。

## Pilot

```bash
mkdir -p "$BASE/qera_diagnostics/logs"
LOG="$BASE/qera_diagnostics/logs/qwen_gi_fp64_v1a_pilot.log"
nohup bash "$QTOOLS/run_qwen_gi_fp64_v1.sh" pilot --max-hours 2 >> "$LOG" 2>&1 &
tail -n 100 -F "$LOG"
```

- 三个模块：L0 gate_proj、L0 down_proj、L1 down_proj；DA/FA 共6次求解，均检查四个 rank。
- Token BF16 只测前8窗口，保存在新输出 `smoke/`，**不进入正式结果表**。
- Word BF16 测完整62篇，保存在 `word_ppl/BF16/`，是完整正式基线，后续 run 校验后复用。
- 六个成功因子事务也由 run 复用。不重复收统计。
- 见 `PILOT COMPLETE` 才表示所选 pilot 完成。CPU 测试不代替该 GPU pilot；若 OOM/门槛失败请回传，不改变精度或门槛。

## 完整运行与续跑

Pilot 进程正常结束、GPU空闲后：

```bash
LOG="$BASE/qera_diagnostics/logs/qwen_gi_fp64_v1a_run.log"
nohup bash "$QTOOLS/run_qwen_gi_fp64_v1.sh" run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 100 -F "$LOG"
```

`run` 默认两种 PPL 全跑，顺序：剩余 DA/FA 求解 → token10配置 → word10配置。输出根独立：
`$BASE/qera_diagnostics/qwen_gi_fp64_v1a`。

服务器重开后重新设置 BASE/QTOOLS/LOG，重复**相同 run 命令**。不要同时启动第二个进程，新输出锁会拒绝并发写入。

- 求解：每“模块×方法”事务落盘；中断只重做当前未完成方法。失败数值候选只有 `.failed.json`，不当成功因子。
- Token：每8窗口事务，最多重算当前未提交批次。
- Word：每完整配置事务；中途停止需重跑当前配置的62篇，已完成配置保留。不是每文档续跑。
- `--max-hours 10` 是程序软预算，不延长服务器租期。应给租期留出提交余量。退出75表示安全暂停；SIGKILL/断电依靠已提交事务恢复。
- 每次重启会重新哈希校验实际源文件，所以开头审计不等于重做统计/SVD；未消费的 raw A 只保留来源绑定，不再读取/重新计算。
- `solve` 只做全求解；`evaluate` 只做评估且必须已有全部所需因子；`--protocol token/word` 可单独续跑某种评估，默认 both。不能因此将单种完成说成双指标完成。

## 结果与检查

```bash
OUT="$BASE/qera_diagnostics/qwen_gi_fp64_v1a"
cat "$OUT/evaluation/ppl_summary_wikitext2.csv"
cat "$OUT/word_ppl/ppl_summary.csv"
cat "$OUT/evaluation/status.json"
cat "$OUT/word_ppl/status.json"
```

两份状态都 COMPLETE，且各10配置才表示全部评估完成。逐窗口在 `evaluation/wikitext2_per_window.csv`，逐文档在 `word_ppl/per_document.csv`。另有 `rank_metrics.csv`、`inputs.json`、`experiment.json`、`word_protocol.json` 和实际部署 tensor hashes。异常在 `failure.json`；旧 failure 文件可能保留历史，判断当前进度应看最新日志/事务状态。

新路径不证明原 root 构造精度足够，不是重建 FP64 root 的第二步；数值检查通过也不是方法显著有效的结论。
