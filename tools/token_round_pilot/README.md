# Token轮数低成本对照

固定L10.q_proj、原S2的32训练窗口×4标签（128梯度样本），FP64，rank64。
只读复用历史第1/2/3轮和第3轮部署权重，追加第4/5轮，比较第1/2/3/5轮。
使用已冻结的正式WikiText-2 validation16个2048词窗口；不读官方test，不新增反向。
所有候选先冻结再评价。同步更新的整体尺度可能奇偶振荡，因此单独报告因子方向与尺度，
不将原始Kronecker积幅值变化直接解释为方向未收敛。KL按相同窗口配对，并报告2000次窗口bootstrap。
单模块短测不能证明全层收敛，也不代表128/256个不同文本窗口；窗口不声称独立文章。
限时15分钟、输出8GiB。旧结果与正式12模块流程均不改写。

```bash
PY=../conda_envs/qera-original-a/bin/python
$PY -B tools/token_round_pilot/test_rounds.py
CUDA_VISIBLE_DEVICES=0,1 $PY -u -B tools/token_round_pilot/run.py \
 --source ../qera_runs/kronecker_l10_v2/run_02 \
 --validation ../qera_runs/multimodule_v1/run_01 \
 --output ../qera_runs/token_round_pilot_v1/run_01
```
