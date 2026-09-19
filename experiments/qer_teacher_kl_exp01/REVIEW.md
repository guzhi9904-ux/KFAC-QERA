# 实验协议 v2 审阅与实施决定

2026-09-18。结论：数学目标明确，可以实施；正式结果由数值审计和预先规定的判据决定。

- `E @ L` 及 `L.T @ Q.T = B.T` 与 m×n 权重布局一致。对称平方根只作同一 A_solve 的对照，不另选阻尼，不用补偿逐元素相等代替目标核验。
- teacher 采样使用每预测位置独立标签，sum-loss；同一共享权重的模块位置投影先求和后平方。CPU 枚举测试同时检查 Fisher/KL 二阶项和错误归约的差异。
- 模型固定 FP32/eager，softmax 和 KL 用 FP64。解析 seed 在 logits 的 FP64→FP32 反向边界转为 FP32，再经过 FP32 head GEMM；这与直接 autograd 保持一致，避免额外完整词表反传图。
- 现有 prepared WikiText 是 test；本实验从冻结 raw validation 按官方 QERA 预处理重新生成前八个完整窗口，记录原始 split、处理顺序、worker 数和 token 哈希。
- 一张 A6000 先试，显式启用目标层之后的逐层 non-reentrant checkpoint；目标 hook 不在重计算层内，且 eval 下通过实际小 Llama 验证前向与梯度。用户已确认三卡可申请，容量不足可另开设备映射不同的新运行，不降精度、不裁上下文。
- 六个方向先冻结，再采样或测 KL。K 按 4→16→64 追加，模块内三个方向统一扩展；alpha 区间选择只看 KL 平台和路径有效性。
- 原始 A 累加值和计数存于同一原子文件；MC 每次提交三个方向；KL 每个干预点提交。身份包含代码、配置、协议、teacher、数据、量化器与方向。异常恢复 teacher 权重，保存失败或暂停状态。
- 仅使用 cck，通过本地 labgpu 调度申请资源；不使用管理员或外部付费 GPU。旧 A/G/Wq 缺失不影响本轮，也不运行旧启动器。

测试、环境适配和 pilot 必须通过后才能正式计算。K=64 仍不满足精度时按协议报告 MC_INCONCLUSIVE，不继续追求通过。

理论来源：[QERA 原论文](https://arxiv.org/html/2410.06040v1)提出输入统计加权的量化误差重构；[Martens 与 Grosse 2015](https://proceedings.mlr.press/v37/martens15.html)讨论模型 Fisher 及其 Kronecker 近似。本轮直接验证给定方向上的曲率估计器，不把该恒等式或 Cholesky 替代本身作为创新。
