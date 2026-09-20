# 12模块共享采样流程

首轮范围固定：层0/10/20/31，每层q_proj、v_proj、down_proj。四种方法为
Marginal、Token-joint、Sequence-one-step、Full-fit；rank64、MXINT3/group32。
这是新协议、新输出目录，不修改历史L10/L31实验。

## 统计与评价

- WikiText-2 raw官方train，固定的128/256个嵌套、互不重叠的2048-token窗口。
  每窗口只抽一套教师标签，四种方法和12模块使用完全相同的输入与标签。
  窗口由整份文本分块，不声称256个独立文章；不同窗口仍可能来自同一文章。
- FP32教师参数、输入、autograd门限保持原样。统计、收缩、SVD仍为FP64，TF32关闭。
  Token-joint仍3轮；Full-fit仍最多20轮、连续两轮同时满足原目标与乘积变化门限。
- validation固定16窗口：分别只替换一个模块，报告12模块×2预算×4方法及量化基线。
- 官方test的全部完整2048窗口：同时替换12模块，评价8个预注册候选和量化基线，
  其余权重保留FP32。报告真实KL及真实下一词NLL/PPL，不使用采样标签计算PPL。
  每块预测2047词，丢弃末尾不足一块的词；与滑窗PPL或其他分词协议不能直接混比。
- 所有候选在评价前冻结，没有根据test选择方法。旧train数据可能重复使用；
  官方test过去可能已经用于其他实验，因此不宣称它是从未见过的新盲测集。

## 省掉的工作与仍有的成本

256套教师标签的12模块统计，从3072次独立反传改为256次连接完整计算图的共享反传。
每窗口的参考输出和教师logits也共享。q/v共用输入缓存，梯度保留原生FP32。
主流程不保存稠密S，不计算eval梯度、Gram、曲率拟合误差或理论界。
需要这些机制诊断时应另立固定样本子集的任务，本版本没有自动启动诊断的入口。

四种方法的计算公式保持不变。两卡离线任务按模块/预算分配；每次新进程先串行
初始化CUDA线性代数，规避旧版PyTorch的并发首次调用问题。迭代历史保留标量，
矩阵只保留当前原子checkpoint；补偿保存P64/Q64/W_deploy和审计，不重复保存可重建残差。

实际Llama形状下，256窗口的x/g约216GiB；全部主要张量约387GiB，设置512GiB上限、
启动时至少544GiB文件系统空闲，并额外核对账户配额。它们是**磁盘**，不是显存。
不保存S额外省约608GiB。断点续跑校验缓存哈希，不覆写已完成结果。

每个模块的拟合和SVD仍要独立做；down_proj的输入维度14336，不能用q_proj耗时直接
外推，也不能把反传次数减少12倍说成总时长减少12倍。默认验证前向1728次；test前向
为9×test窗口数，加教师自身重放和首窗口重复性检查。资源日志记录分阶段实测耗时。

## 服务器运行

在KFAC-QERA-teacher-kl目录，用原环境；以下变量均为shell变量，不修改环境配置文件。

```bash
PY=../conda_envs/qera-original-a/bin/python
ENTRY=experiments/qer_multimodule_v1

# 只做一次，从已有官方缓存导出新目录。会核对train/validation与迁移数据逐行一致。
$PY -B "$ENTRY/materialize_dataset.py" \
  --cache-dir ../huggingface_cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3 \
  --compare-existing ../teacher_kl_assets/geometry_inputs/wikitext2 \
  --output ../teacher_kl_assets/multimodule_wikitext2

$PY -B "$ENTRY/configure.py" \
  --from-config ../qer_ksample_l10_v2.json \
  --wikitext ../teacher_kl_assets/multimodule_wikitext2 \
  --quantized-run ../qera_runs/functional_gradient_4090_v1/run_04 \
  --output-parent ../qera_runs/multimodule_v1 \
  --hours 10 --output ../qer_multimodule_v1.json

CONFIG=../qer_multimodule_v1.json
RUN=../qera_runs/multimodule_v1/run_01
$PY -B "$ENTRY/runner.py" "$CONFIG" "$RUN" prepare
CUDA_VISIBLE_DEVICES=0,1 $PY -B "$ENTRY/runner.py" "$CONFIG" "$RUN" pilot
```

`prepare`只冻结数据、量化资产及工作量表；`pilot`只用一个历史训练窗口核查12模块
共享梯度、独立权重autograd、原精度、显存与自KL，不跑正式实验，也不能提供完整拟合ETA。

确认租期和配额能覆盖后，**用户自行启动**正式流程：

```bash
mkdir -p "$RUN/logs"
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 $PY -u -B "$ENTRY/runner.py" "$CONFIG" "$RUN" run \
  2>&1 | tee -a "$RUN/logs/console.log"
```

阶段依次是collect→fit→evaluate→report；也可把`run`换成单独的阶段名。
相同源码/配置/输出目录重跑同一命令会从已提交结果恢复。
已冻结源码或配置发生变化时拒绝混跑，应保留旧输出并新建目录。
`--hours`为整个run累计活跃时间上限，需在开始前设定；达到上限会停止而非擅自续租。
前台命令不自动防断线，长实验可在用户自己的tmux会话里运行。

看日志：`tail -f "$RUN/logs/console.log"`。
结果：`summary/results.csv`、`summary/results.json`；只有完整覆盖预注册数据后才生成
`complete.json`和`EXPERIMENT_COMPLETE`。`resources.json`含阶段耗时、显存峰值和累计时长。

## 验证

```bash
$PY -B "$ENTRY/test_shared.py"
$PY -B "$ENTRY/test_workflow.py"
```

覆盖跨层/同层连接梯度与独立权重autograd、checkpoint、异常恢复、精确序列矩公式、
四种拟合与冻结旧实现对照、断点续跑、部分缓存恢复/损坏拒绝、评价范围与结果完整性。
本地CPU测试与服务器固定PyTorch2.3环境均需通过；真实GPU短验收另存acceptance.json。
