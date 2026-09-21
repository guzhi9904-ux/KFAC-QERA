# V attention-aware prototype 双4090执行入口

科学协议全文见protocol.md。该文件开头“尚未运行”是收到的设计文档原文，实际运行状态以新run记录为准。
只新增四层V的一个Attention-aware候选；原N256 Marginal/Sequence/None原样复用。无A-only/Token新候选、无test/PPL/Full-fit任务。

实现通过仅目标self_attn调用的output_attentions=True取得真实eager概率；全模型仍output_attentions=False，避免保存32层的attention。
真实GQA排列通过当前Transformers.repeat_kv验证。正式统计依次在该层所在GPU进行FP64收缩，避免与模型同时做eig/SVD；候选求解才卸载teacher并双卡离线。
Check1失败立即停止，不降门限。输入/梯度原生FP32；代数重排和统计FP64。每4窗口四层共同提交一次原子generation；断电最多重算未提交的3窗口，保留当前和前一个generation，已完成窗口不重复累计。
资产审计只读取所需的4V/4down/QKV/标签约72GiB原始缓存及必要基线，不扫描/复制整个父缓存。续跑检查文件签名和receipt，沿用首次哈希验收。
父141窗口test未全部完成不影响其已冻结N256候选和完整16窗口validation；审计会明确记录该边界。
累计活跃时间8小时保护、新增磁盘16GiB上限；初始空闲48GiB（含32GiB恢复空间）。不自动续租。小样本只验算与计时，不按补偿效果挑选层/参数。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE/KFAC-QERA-teacher-kl"
CONFIG="$BASE/qer_v_attention_proto_v1.json"
RUN="$BASE/qera_runs/v_attention_proto_v1/run_01"
"$BASE/conda_envs/qera-original-a/bin/python" -B experiments/qer_v_attention_proto_v1/configure.py \
 --parent-run "$BASE/qera_runs/multimodule_three_fp64_v1/run_01" \
 --output-parent "$BASE/qera_runs/v_attention_proto_v1" --output "$CONFIG"
mkdir -p "$RUN/logs"
bash experiments/qer_v_attention_proto_v1/run.sh "$CONFIG" "$RUN" pilot 2>&1 | tee "$RUN/logs/pilot.log"
tmux new-session -d -s qer_v_attention "bash '$BASE/KFAC-QERA-teacher-kl/experiments/qer_v_attention_proto_v1/run.sh' '$CONFIG' '$RUN' run >> '$RUN/logs/run.log' 2>&1"
tail -n 80 -f "$RUN/logs/run.log"
```

pilot通过标记CHECK1_PASSED_ALL_FOUR_LAYERS。正式完成标记V_ATTENTION_PROTOTYPE_COMPLETE及PROTOTYPE_EXIT 0。
原启动命令也用于相同源代码/config下断点续跑；确认旧进程已退出后启动，OS锁防重入，不删除锁文件。
结果见summary/results.csv、paired_comparisons.csv、per_window.csv、RESULTS.md。区间为2000次探索性配对窗口bootstrap，四层共用索引；窗口相关性及fit重采样未包含，不能当作最终盲测或机制证明。
