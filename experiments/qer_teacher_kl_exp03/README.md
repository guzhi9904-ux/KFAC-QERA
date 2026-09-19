# Exp-3：canonical full-fit single-Kronecker 与 weighted SVD

仅 cck、labgpu 任务内执行。父源码与前序资产只读；运行目录和源码目录独立。

`run.sh OUTPUT pilot` 只使用拟合侧数据，生成共享权重梯度、收缩与线性代数实测。根据 pilot 的数值和预算验收，将 plan、identity、pilot SHA256 和成本决策写入独立 manifest.json；随后 `run.sh OUTPUT formal` 才能继续。正式评价前，数据/因子/补偿再次写入不可变哈希清单。用户本次“根据要求完成第三次实验”已授权实施；方案首页关于当时只交付设计的说明不构成再次请求启动许可的要求。

计划 8 个新 WT2 train 文章前缀窗口 × 4 teacher 标签，保留所有 8×64 历史评价标签。仅 marginal 初始化，20 个确定性 ALS 周期上限，两个连续周期满足相对 J 改善 ≤1e-6 且 Kronecker 乘积变化 ≤1e-4 时停止。最大周期停止只表示预算用尽，不表示全局收敛。无开发集和超参数搜索；两个 AG 组共享 eta_A=eta_G=1e-3。判定阈值事先定为 None 损伤的 2 个百分点，以及收益预测 MAE 的 2 个百分点，与实验二结构误差预算不同。

H 含 1/T；canonical marginal G 吸收 L/T 一次；随后做保持乘积的 Frobenius gauge，再施加 trace-relative damping。全通道 FP64，无对角化、低秩统计、随机 trace、隐藏 floor 或自适应阻尼。S 和梯度 Gram 分别保存在每个原子样本中，A 每个窗口只累计一次。ALS 流式读取 S。

weighted SVD 使用 FP64 dense gesvd；理想补偿为两个 rank-64 因子的乘积，同时保存 dense FP32 补偿和实际部署权重。评价 q_H 与 actual KL 使用同一个实际部署残差；历史冻结方向另外投影以核验父记录。新 KL 每窗口重复两次；所有候选共用 full 曲率裁判，两个冻结 metric 分别交叉评价四个补偿，raw/solve 分开报告。

`test_ag.py` 覆盖显式 H/vec/块更新、ALS 单调性、canonical marginal、gauge、相对阻尼尺度等变性、weighted-SVD 截断尾能量和无效 PSD。`test_analysis.py` 覆盖配对方差、四态判定、分母失效以及 bootstrap 配对与尺度不变性。正式运行后另行从原始标量独立复核。
