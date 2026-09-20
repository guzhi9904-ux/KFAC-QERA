# 12模块、三方法、FP64夜间实验

用户2026-09-20明确授权后台运行，次日自行查看日志；不创建监控或自动续租任务。
独立于qer_multimodule_v1的新版本，保留原版本源码及输出。

- 模块：层0/10/20/31，每层q_proj、v_proj、down_proj，共12个。
- 方法仅Marginal、Token-joint（固定三轮同步）、Sequence-one-step；Full-fit被配置校验拒绝。
- 统计/收缩/开方/SVD使用原FP64实现；教师及原生x/g、最终部署权重FP32，TF32关闭。
- 同一WikiText-2 train嵌套128/256个2048词窗口，每窗口一套教师采样标签。
  保持原采样种子和窗口排序，方法共享同一组输入、标签和连通反传。
- validation16窗口逐模块干预：12×(2预算×3方法+量化基线)；正式test全部141完整窗口，
  同时干预12模块，6个候选及量化基线，其余权重FP32。报告KL及NLL/PPL。
- 先冻结全部候选再评价，不按test选方法。无eval梯度、Gram、曲率拟合和理论界诊断。
- 两卡离线拟合，串行初始化CUDA线性代数；复用原数学及断点续跑机制。
- x/g缓存约216GiB磁盘；原保守的512GiB输出上限、544GiB空闲要求保留。
  文件系统空闲不代表账户剩余配额。10小时为累计活跃上限，不是完成时间保证。

## 后台流程

launch.sh依次执行prepare、真实共享梯度pilot、正式collect/fit/evaluate/report。
任一阶段失败立即退出，保留日志和已提交检查点，不自动跳过验收或重新启动。
在仓库目录配置并用tmux启动：

```bash
PY=../conda_envs/qera-original-a/bin/python
ENTRY=experiments/qer_multimodule_three_fp64_v1
$PY -B "$ENTRY/configure.py" --from-config ../qer_multimodule_v1.json \
 --wikitext ../teacher_kl_assets/multimodule_wikitext2 \
 --quantized-run ../qera_runs/functional_gradient_4090_v1/run_04 \
 --output-parent ../qera_runs/multimodule_three_fp64_v1 \
 --hours 10 --output ../qer_multimodule_three_fp64_v1.json
```

运行配置及目录固定后，用tmux调用launch.sh并重定向到对应logs/console.log。
同一源码、配置和输出目录可续算；运行中不要修改该版本源码或配置。

```bash
RUN=../qera_runs/multimodule_three_fp64_v1/run_01
tail -n 80 -f "$RUN/logs/console.log"
```

全部成功会记录EXPERIMENT_COMPLETE和LAUNCHER_EXIT 0；结果在summary/results.csv。
logs/launcher.exit只在退出时出现，0表示完整流程成功，非0表示检查点处失败退出。
不删除已有缓存；不登录613，不使用其管理员账户。

## 检查

test_shared.py验证连通共享反传、原权重autograd、异常恢复及评价恢复。
test_workflow.py验证数学对照、断点/损坏检测、三方法配置与评价完整性。
保留旧Full-fit的微型CPU数学回归测试，但正式配置及工作量中不包含Full-fit。
