# 换实例后只续跑评价

2026-09-21用户要求租期结束、重新租用后断点继续。原实验72个候选已完成，评价尚未完成。
新实例须能访问同一个共享存储目录，保留原模型、环境、仓库、配置和约338GiB输出，并提供双4090。
此工具不管理租赁、不搬迁大文件、不自动监控或续租。

原10小时额度耗尽时，直接重新执行原runner会再次停止。不要改原配置或清空resources.json。
本工具只对当前进程的资源管理器提供明确的累计时间额度，并写入runtime_extensions审计记录；
原科学配置/源码/manifest/数值门限不改，历史活跃时间不清零。16小时是含已用约10小时的总额度。
入口固定为evaluate，全部候选和数据仍通过原哈希检查；完整窗口跳过，部分窗口只补缺少候选。
原OS文件锁仍生效，实例终止后会由系统释放，无需手动删除.run.lock。

```bash
BASE=/share/home/tm902089733300000/a913520780/chengkang
cd "$BASE/KFAC-QERA-teacher-kl"
RUN="$BASE/qera_runs/multimodule_three_fp64_v1/run_01"
CONFIG="$BASE/qer_multimodule_three_fp64_v1.json"
test -f "$RUN/candidate_freeze.json" && test -x "$BASE/conda_envs/qera-original-a/bin/python"
nvidia-smi
tmux new-session -d -s qer_three_resume \
 "bash '$BASE/KFAC-QERA-teacher-kl/tools/three_fp64_resume/launch.sh' '$CONFIG' '$RUN' 16 >> '$RUN/logs/resume.log' 2>&1"
tail -n 80 -f "$RUN/logs/resume.log"
```

先确认旧任务已退出；不要在旧任务仍运行时另起一个。命令只启动一次，已有同名tmux会话会拒绝重复启动。
最终EXPERIMENT_COMPLETE/EXPERIMENT_ALREADY_COMPLETE及RESUME_EXIT 0表示成功。
结果仍在原目录summary/results.csv，旧logs/launcher.exit可能保留旧失败码，不能据此判断此次续跑。

## 2026-09-21：缩短测试集检查

用户要求以16或32个测试窗口快速检查趋势。本轮固定选原test顺序的前16个窗口，
选择依据是运行成本，不是测试成绩。保留原141窗口数据及manifest，不改科学代码和候选。
先对已确认的旧评价Python进程发送SIGINT并确认退出，再用新的tmux会话续跑：

```bash
tmux new-session -d -s qer_three_test16 \
 "bash '$BASE/KFAC-QERA-teacher-kl/tools/three_fp64_resume/launch.sh' '$CONFIG' '$RUN' 16 16 >> '$RUN/logs/test16.log' 2>&1"
tail -n 80 -f "$RUN/logs/test16.log"
```

第四参数为测试窗口数，可选16或32；第三参数仍为含历史时间的累计小时上限。
验证集仍全部16窗口。已有逐候选原子记录复用，测试只补选定前缀缺失项，到16窗口自动退出。
选择清单写入evaluation_subsets/test_first_16.json；子集结果写入summary/test_subset_16/results.csv和results.json。
仅子目录写complete.json，日志成功标志TEST_SUBSET_COMPLETE 16及RESUME_EXIT 0，绝不把原141窗口实验标为完成。
test是12模块联合部署，不能据此推断每层每模块的test排名；固定前缀也不是随机全测试集估计。
