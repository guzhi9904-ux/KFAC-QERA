# k/o增量与20模块A-only基线：冻结协议 v1

授权：2026-09-21用户要求补代码、规范并直接启动；后续明确A-only补全部20模块。不持续监听。

## 研究问题与范围

新增L0/L10/L20/L31的k_proj、o_proj，共8模块，检查attention投影类型差异。
同时给原12个q/v/down模块补A-only。原12模块三方法及仅量化基线全部沿用冻结validation记录，不重新计算。
新的44份补偿：8×4 + 12×1；新增16×(8×5+12)=832次验证候选前向，不含参考与数值复验。
只做逐模块validation。禁止把结果写为联合test、完整模型压缩或q_full恢复率。

## 固定统计与方法

原WikiText2 train前256个冻结不重叠2048-token窗口，每窗口一套相同教师采样标签；不重采样。
原validation16窗口完全复用；每窗口计2047个预测token。rank64，MXINT3 group32，teacher与部署FP32，统计与SVD FP64，TF32关闭。
Marginal、Token-joint（三次同步更新）、Sequence-one-step沿用原算式、归一化、相对阻尼、数值检查和部署规则；不跑Full-fit。
A-only：完整非中心化输入二阶矩A_m，输出侧原始G=I；最小化输入加权重建误差，使用同一个FP64求解器。
输出侧相对阻尼只把I乘以标量，不新增梯度信息。该名称指输入侧QERA目标基线，不宣称完全复现任意官方QERA工程版本。
新增k复用同层q的输入和A统计；原12模块复用原A统计；o输入和A新采集。k与v梯度禁止互换。

## 缓存down梯度与局部反向

原down输出梯度等于同层block输出梯度，作为完整block的VJP种子；重放保留残差、norm、MLP、mask、position和RoPE参数。
每窗口完整无梯度前向捕获四个block真实调用参数；逐层局部反向同时获得k/o梯度。block输入不长期落盘。
前两fit窗口四层验收：局部q/v与冻结q/v及新完整共享反向比较；新增k/o与完整反向比较；首窗口另做k/o权重autograd检查。
梯度相对误差门限1e-5，输出前向1e-7，不改变旧数学/部署门限。局部快捷路径未通过则使用完整共享反向，绝不放宽门限。
所有借用tensor必须有已有receipt，验证父identity、tensor/file哈希、样本、标签和输入绑定；父目录只读。

## 评价

从目标层的真实teacher输入开始运行后缀，复用参考logits。两个fit窗口、全部20模块仅量化干预验证后缀与完整前向等价；不用于选择候选。
首validation窗口再对每个新增实际候选做完整前向复验；不通过即停止，不选择较好分数。快捷路径pilot未通过则全程完整前向。
同一窗口中一个模块替换，其他模型参数保持teacher FP32；每次恢复原权重并核验精确hash。
KL方向teacher→candidate。每模块按token汇总后计算100*(1-KL_candidate/KL_quantized)。另报告相对A-only的实际KL降低百分比。
summary/results.csv/json合并原12模块冻结的三方法与基线记录，并记录每份来源路径/hash；新旧identity不混淆。

## 执行、资源与恢复

双4090，离线两worker；8小时累计活跃时间上限，180GiB增量输出上限，初始可用磁盘190GiB，保留32GiB恢复空间。
估计新增输入/梯度72GiB，统计、因子、44补偿和临时文件另留余量。前缀及block输入按窗口释放。
资源检查每30秒；仅新run目录的完整磁盘扫描每10分钟一次，不扫描父338GiB历史树。
小样本验收完成后启动run。OS锁避免重复进程，原子样本/候选/窗口检查点可复用。源码、配置改变必须新run。
实际采集与拟合耗时以pilot及进度日志为准，预估运行4–6小时不含编写调试。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE/KFAC-QERA-teacher-kl"
CONFIG="$BASE/qer_ko_increment_v1.json"
RUN="$BASE/qera_runs/ko_increment_v1/run_01"
"$BASE/conda_envs/qera-original-a/bin/python" -B experiments/qer_ko_increment_v1/configure.py \
 --parent-run "$BASE/qera_runs/multimodule_three_fp64_v1/run_01" \
 --output-parent "$BASE/qera_runs/ko_increment_v1" --output "$CONFIG"
bash experiments/qer_ko_increment_v1/run.sh "$CONFIG" "$RUN" pilot
tmux new-session -d -s qer_ko_increment "bash '$BASE/KFAC-QERA-teacher-kl/experiments/qer_ko_increment_v1/run.sh' '$CONFIG' '$RUN' run >> '$RUN/logs/run.log' 2>&1"
tail -n 80 -f "$RUN/logs/run.log"
```

正式完成标记INCREMENT_EXPERIMENT_COMPLETE及INCREMENT_EXIT 0。pilot的EXIT0只表示验收完成。
换实例后确认旧进程已退出，在共享存储同目录用同一CONFIG/RUN执行同一run命令即可；不要删除锁文件或重置累计时间。
