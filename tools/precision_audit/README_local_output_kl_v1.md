# 固定 rank64：局部输出重构误差与最终预测损伤

本实验检验：固定 Full-A、MXINT3 Wq、rank64，GF 是否在某些模块留下更大的局部输出误差，却产生更小的最终 teacher KL。所有结果都保留，不以该现象出现作为运行成功的门槛。

## 冻结设计

- 模型：Meta-Llama-3.1-8B Base；第 **0、10、20、31** 层，每层 q/k/v/o/gate/up/down 全部7类投影，共28模块。这是预先固定的覆盖样本，不代表全部224模块。
- 三个完整补偿：**FA+GI、FA+DG、FA+GF，rank64**。复用已评估、同根FP64求解后直接转BF16的因子，不做SVD、不收A/G、不重新量化、不删分量。
- 每次仅目标模块使用原Wq及其两次GEMM补偿，其余模块保持同一个BF16模型。每个方法结束即恢复原权重并移除钩子，异常也恢复。
- 评估：冻结WikiText2 test 138×2048窗口，与SlimPajama校准文本分离。每窗口计2047个预测位置，全程batch1。GPU0 teacher，GPU1 student，继承上一实验两张4090的运行环境；需要两卡空闲。
- 每个模块/窗口先运行BF16自对照，验证两卡模型self-KL≤1e-9、目标输入/输出及NLL完全一致；每个补偿的目标输入也必须与teacher逐值相同。
- 在补偿钩子之后捕获实际BF16输出。局部误差计算只用位置0..2046，与最终预测标签ids[1:]对齐；所有输出坐标都计入。
- 原grouped脚本只作为已冻结的来源审计、加载与KL数值helper导入，**不调用删组、plain SVD、全模型量化或几何准备入口**。

## 指标与解释

对同一目标模块，令delta_y为补偿后输出减BF16输出：

- `output_sse_per_position = mean_token sum_channel(delta_y²)`，实际输出差以FP64相减/平方/求和。
- `output_mse_per_element` 再除以目标模块输出维数。
- `output_nmse = sum(delta_y²)/sum(y_teacher²)`，零分母记空值。
- `teacher_kl_per_token = mean KL(P_teacher || P_student)`，完整词表、FP64 centered-logsumexp、chunk128，复用已测KL实现。
- `nll_per_token_batch1`：显式FP32 cross_entropy、chunk128，每token结果转FP64求和；记录同一前向teacher NLL及delta。**这是新batch1口径，不能直接与旧batch8 NLL作严格端点回归**。该脚本不输出word-PPL，也不把局部模型当作全模型部署结果。

`paired.csv` 对每个模块分别给出GF−GI、GF−DG。`relative_output_sse_change`为(GF SSE−baseline SSE)/baseline SSE。原始SSE差、KL差、NLL差及各自95%探索性配对block-bootstrap区间全部保留。固定seed20260912、2000次、循环8窗口块；不是IID token检验，也没有多重比较校正。

主图可用横轴`relative_output_sse_change`、纵轴`kl_delta`：右下象限表示局部重构误差增大、最终KL减少。一个点是一个模块；全部28个模块均应报告。NMSE与模块内GF/GI比例的分母在方法间相同。

该现象支持局部误差大小不能充分决定预测损伤；不独立证明当前梯度Gram是精确Fisher，不证明GF全局最优，也不能将模块损伤相加解释全模型性能。用训练时同一个G给GF自身目标评分并非独立验证，本轮核心指标均来自独立评估文本上的实际前向，不新增G加权代理指标。

## 输出、审计和续跑

独立目录由私有服务器配置中的`LOCAL_OUTPUT_DESTINATION`指定，不能与来源目录重叠。

- `experiment.json`：来源证书、因子哈希、代码哈希、模块列表与计分协议；变化后拒绝混用输出。
- `background.json`：运行前后两模型参数/缓冲区与冻结BF16来源逐位核对。
- `pilot.json`、`pilot_windows/`：L0 o_proj、L31 down_proj，各2窗口×3方法，独立于正式数据。
- `windows/<module>.json`、`windows/<module>/window_XXXX.safetensors`：每次完整提交GI/DG/GF三组KL/NLL/局部SSE、teacher NLL和参考能量。中断最多重算当前配对窗口。
- `teacher_windows/`：每个窗口跨模块复用的teacher NLL位证书。
- `summary.csv`：完成后的84行方法指标；`paired.csv`：56行配对对照；`per_window.csv`：3864行配对窗口数据。
- `status.json`：完成模块数量与总数；只有28/28才为complete。
- `local_output_kl_v1_summary.tar.gz`：正式运行结束自动打包；含协议/状态/三张CSV/背景与pilot，不含模型、因子、token数组。也可通过`pack`重新生成。

`doctor`只审计来源与环境，不执行前向；会读取和校验既有大文件，首次可能较慢。`pilot`通过后才允许`run`。达到`--max-hours`或`--max-new-units`时返回75，重复相同命令继续；exit75不是完成。实际OOM或门禁失败返回非零并记录`last_failure.json`，不要自动忽略错误。

预计token检查点约0.7GiB，启动要求至少4GiB空闲。运行时间由GPU前向决定，不承诺10小时内完成；10小时是单次执行预算，支持续跑。每窗口含teacher、BF16自对照、三种补偿共5次前向。

## 服务器完整运行命令

上传`local_output_kl_v1.zip`与`local_output_kl_v1.zip.sha256`到服务器。新建专用工具目录，保留现有工具/模型/统计目录。带私有配置的离线包可直接运行；Git中的公开版本不包含具体机器路径。

```bash
BASE=/path/to/upload-directory
TOOLS="$BASE/local_output_kl_v1_tools"
PY=/path/to/existing/python
cd "$BASE"
sha256sum -c local_output_kl_v1.zip.sha256
mkdir "$TOOLS" && "$PY" -m zipfile -e "$BASE/local_output_kl_v1.zip" "$TOOLS"
(cd "$TOOLS" && sha256sum -c SHA256SUMS.local_output_kl)
export CUDA_VISIBLE_DEVICES=0,1
bash "$TOOLS/run_local_output_kl_v1.sh" doctor
bash "$TOOLS/run_local_output_kl_v1.sh" pilot --max-hours 2
```

确认末尾`PILOT COMPLETE`后启动正式实验：

```bash
BASE=/path/to/upload-directory
TOOLS="$BASE/local_output_kl_v1_tools"
export CUDA_VISIBLE_DEVICES=0,1
LOG="$BASE/local_output_kl_v1_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$TOOLS/run_local_output_kl_v1.sh" run --max-hours 10 > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

`tail -f`中Ctrl+C只停止看日志。若日志末尾为`PAUSED (75)`，确认旧进程已结束后重复上面正式启动段即可续跑，工具和输出目录都不变。若pilot被预算打断，则重复pilot；不要越过门禁。

完整结束后结果包在：

```text
$LOCAL_OUTPUT_DESTINATION/local_output_kl_v1_summary.tar.gz
```

可在空闲两卡环境中核验或手工打包（目前这两个入口仍执行相同的严格来源/环境审计）：

```bash
bash "$TOOLS/run_local_output_kl_v1.sh" pack
source "$TOOLS/server_profile.sh"
cat "$LOCAL_OUTPUT_DESTINATION/status.json"
```

## 开发验证

```bash
cd tools/precision_audit
python -B -m unittest test_local_output_kl_v1 -v
python -B build_local_output_kl_v1.py --output-dir /path/to/new/release --update-checksums
```

为已有服务器制作可直接运行的私有包，可另加`--server-profile /path/to/private/server_profile.sh`。该配置只进入离线ZIP，不加入Git。配置需提供以下变量：`LOCAL_OUTPUT_PYTHON`、`LOCAL_OUTPUT_REPO`、`LOCAL_OUTPUT_RUN`、`LOCAL_OUTPUT_FP64`、`LOCAL_OUTPUT_RANK64`、`LOCAL_OUTPUT_QERA`、`LOCAL_OUTPUT_HARNESS`、`LOCAL_OUTPUT_WORD_REFERENCE`、`LOCAL_OUTPUT_DESTINATION`；并设置该服务器原有的`PYTHONPATH`、HF离线缓存环境。已有实验与模型的具体路径由使用者的私有配置决定。

CPU测试覆盖真实钩子顺序、FP64局部SSE、FP32 CE、直接KL对照、非目标权重不变、失败恢复、配对窗口原子性、续跑/篡改拒绝与汇总方向。CPU通过不等于服务器CUDA pilot通过。
