# Llama DA GI 分组 rank 增量诊断

问题：即使全模型增加 rank 改善 PPL，是否仍存在被其他层收益掩盖的负收益层组？本轮是 WikiText2 上的事后诊断，不用于选择主结果配置，不是 rank 分配算法。

## 固定实验

- Meta-Llama-3.1-8B Base，MXINT3，DA+GI（对角输入二阶统计，G=I）。
- 复用 `diag_a_fp64_dual_v1` 已完成的 rank64 FP64 因子及其直接 BF16 转换；本轮不采集 A/G、不构造 root、不做 SVD/逆求解、不重新量化。
- 每个模块以 BF16 两次 GEMM 部署。所有 224 个目标模块始终有一个完整补偿；仅改变规定模块的前缀长度。
- 六个固定配置如下。层号从 0 开始，每组包含 8 层、56 个线性投影。

| 配置 | rank32 的范围 | 其余目标模块 |
|---|---|---|
| DA_R16 | 无 | rank16 |
| DA_R32 | 全部32层 | 无 |
| DA_L00_07_R32 | L0-7 | rank16 |
| DA_L08_15_R32 | L8-15 | rank16 |
| DA_L16_23_R32 | L16-23 | rank16 |
| DA_L24_31_R32 | L24-31 | rank16 |

每个层组分别从同一个全模型 r16 基准出发，**不是连续升级，也不是删分量**。本轮不含 DG/GF、不自动细分异常组、不搜索新 rank。没有发现负收益也是有效结果。

## 评估与门禁

- Token-PPL：冻结 138×2048 无 padding 窗口，计 282486 个预测位置，batch8、CE chunk256，严格保留历史 evaluator 的 CE dtype 和归约顺序。它不是 `local_output_kl_v1` 的新 batch1 FP32 CE 口径。另存逐token loss 及 FP64 诊断求和，主指标仍按历史归约。
- Word-PPL：冻结 QERA/harness 官方 wrapper，context4096，62文档、241335词，完整 rolling likelihood。两种 PPL 分开解释；word 并非 token 换分母。
- 每个协议先重放全 r16、全 r32，再评估四组。与各自历史端点逐窗口/逐文档比较，最大单元 NLL 差不超过 1e-3，PPL 绝对差不超过 1e-5；这是工程重放门禁，不是统计显著阈值。
- 核对原因子文件哈希、FP64 到 BF16 直接转换、历史部署的 r16/r32 前缀、每个新配置的全模型参数/缓冲区/设备映射与实际补偿位记录。历史记录无法补证 hook 次数，故每次新成功前向均独立检查每个目标补偿恰好执行一次。
- 原 source setup 会审计已有 FA/G 来源证书及旧控制，可能耗时；不调用其求解/构根入口。读取的 source metadata 保存于新目录，不修改来源目录或共享环境。
- 继承既有运行环境及两张4090的设备映射；请在两卡均空闲时运行，不与已有 Qwen/G 采集任务抢卡。

## 文件与续跑

`doctor`：只核验来源和历史端点，不运行模型前向，也不创建正式结果。

`pilot`：六配置分别运行首8个 token 窗口，并重放端点；检查点独立于正式实验。pilot 不替代完整 token/word 端点门禁。

`run --protocol both`：两个协议各六配置。token 每8窗口原子提交（末批2窗口）；word 每个完整配置提交，中断会重跑当前 word 配置。exit75 为预算/信号暂停，重复相同命令和目录即可继续。未完整提交的临时文件不会计入结果。

`run --protocol token` 或 `word`：只做一个协议，后续可在同一目录补另一协议。只有两个协议各六配置都完成，才生成正式汇总包。

`pack`：重新核验现有完整结果并打包，不运行前向；仍会核验冻结来源与环境。缺任一协议或端点门禁失败，不生成“完整”汇总包。

输出位置由私有 profile 的 `RANK_INCREMENT_DESTINATION` 指定，必须与来源目录分离。日志放在该目录外。

| 路径 | 内容 |
|---|---|
| experiment.json / source_audit.json | 冻结来源、代码哈希、原端点逐单元值及各模块原FP64目标 |
| pilot/ / pilot_gate.json | 独立小规模前向及端点控制 |
| token/DA_*.json、token/tokens/ | 正式逐窗口、逐token损失、执行次数和部署证书 |
| word/DA_*/ | 完整 harness 结果、逐文档损失、部署和执行记录 |
| token/ 与 word/ 的 ppl_summary.csv | 六配置绝对PPL/NLL、升级模块数及新增补偿参数量 |
| comparisons.csv / paired_deltas.csv / per_unit.csv | 相对全r16的差值、改善/退化单元数及完整逐单元数据 |
| nonadditivity.json | 全r32的NLL变化减去四组单独变化之和；不将其直接归因到某一层对 |
| status.json | 两协议各六配置完成状态 |
| llama_rank_increment_v1_summary.tar.gz | 协议、来源摘要、端点控制和全部CSV；不含权重、因子或逐token张量 |

`delta_mean_nll > 0` 表示相对全 r16 的负收益。所有四组均报告；计数与配对明细用于后续相关性适当的分析，不将 token 当作 IID 重复。不同层组的有限干预收益不能简单相加。

## 服务器运行

将带私有 profile 的 `llama_rank_increment_v1.zip` 及 `.zip.sha256` 上传到服务器。公开源码只含通用配置接口，具体机器路径仅进入离线包。

```bash
BASE=/path/to/upload-directory
TOOLS="$BASE/llama_rank_increment_v1_tools"
PY=/path/to/existing/python
cd "$BASE"
sha256sum -c llama_rank_increment_v1.zip.sha256
mkdir "$TOOLS"
"$PY" -m zipfile -e "$BASE/llama_rank_increment_v1.zip" "$TOOLS"
export CUDA_VISIBLE_DEVICES=0,1
bash "$TOOLS/run_llama_rank_increment_v1.sh" doctor
bash "$TOOLS/run_llama_rank_increment_v1.sh" pilot --max-hours 2
```

确认 `PILOT COMPLETE` 后，启动正式实验：

```bash
BASE=/path/to/upload-directory
TOOLS="$BASE/llama_rank_increment_v1_tools"
export CUDA_VISIBLE_DEVICES=0,1
LOG="$BASE/llama_rank_increment_v1_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$TOOLS/run_llama_rank_increment_v1.sh" run --protocol both --max-hours 10 > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

`tail -f` 中 Ctrl+C 仅停止看日志。若末尾 `PAUSED (75)`，确认旧进程退出后重复正式启动命令。失败门禁不能绕过或放宽后当作同一实验。预期历史 token-PPL 约为 r16=10.020961、r32=9.640474；门禁使用完整源记录，不使用这些四舍五入展示值。

正式完成后可查看结果包位置：

```bash
source "$TOOLS/server_profile.sh"
cat "$RANK_INCREMENT_DESTINATION/status.json"
ls -lh "$RANK_INCREMENT_DESTINATION/llama_rank_increment_v1_summary.tar.gz"
```

## 开发验证与打包

```bash
cd tools/precision_audit
python -B -m unittest test_llama_rank_increment_v1 -v
python -B build_llama_rank_increment_v1.py --output-dir /path/to/release --update-checksums --server-profile /path/to/private/server_profile.sh
```

profile 需要设置 `RANK_INCREMENT_PYTHON/REPO/RUN/FP64/DA/QERA/HARNESS/WORD_REFERENCE/DESTINATION`（实际变量均带 `RANK_INCREMENT_` 前缀），以及原环境的 HF 离线缓存路径和 PYTHONPATH。CPU测试不能替代服务器CUDA pilot与完整端点门禁。
