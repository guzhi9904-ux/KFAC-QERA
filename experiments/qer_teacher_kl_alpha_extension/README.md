# 实验一幅度扩展

只读复用父实验 `20260918_r1` 的 teacher、8 个输入窗口和 6 个冻结 FP32 残差。新增 alpha=0.3/0.5/0.75/1.0 的 192 个非零 KL 点，另测 16 个 self-KL 控制和 4 个 pilot 点。只运行前向，不调用采集、量化生成、SVD 或 MC 反传。

`extend.py` 在导入父代码前验证其全部源码哈希；核验父实验身份、原记录、小幅度基准、输入、方向及量化来源。加载模型后核验全部 teacher 参数哈希、设备映射和环境版本。原路径、稳定 KL 和权重恢复复用冻结父实现，原代码与记录不修改。

`analysis.py` 固定原 q_KL；逐窗口使用自己的三个小幅度点建立基准。无效点保留但不参与非线性解释；不设新的通过标准。输出总体和逐窗口的 rho、有符号偏差、同 rank 补偿差值及窗口胜负。

在 cck 的 labgpu 作业内运行：

```bash
/bin/bash /home/cck/projects/qer_teacher_kl_alpha_extension_r1/run.sh /data2/cck/KFAC-QERA/runs/qer_teacher_kl_alpha_extension/20260918_r1
```

重用相同输出目录即可断点续跑；实验身份不一致会拒绝。原子记录按模块、方向、窗口和 alpha 唯一命名，并由文件锁阻止并发写同一运行。扩展源代码、配置或父资产变化时建立新目录。

本地分析测试：`python test_analysis.py`。测试覆盖 token 加权与逐窗口基准、禁止大幅度点重拟合、排序反转识别、无效点处理和重复记录拒绝。
