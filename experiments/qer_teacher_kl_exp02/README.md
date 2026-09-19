# QER 实验二：结构消融

固定父实验的模型、输入、512 份采样标签及六份残差，重放 1024 次反传。独立收集八窗口的完整通道 FP64 A_diag，每模块 16384 个输入位置，不使用原 SlimPajama A 替代诊断 A。

每次反传同时计算 full、pos、sep。sep 完整使用 R A_diag R.T；真实 pilot 交叉核验直接 gM 与临时梯度 Gram，并以两次测量的中位时间按模块选择后冻结。两条路径均保留完整 FP64 通道矩阵及 L/T 系数，无随机 trace 或额外结构近似。

评分、误差与收益的区间由配对标量计算。比值使用固定 2000 次窗口内配对 bootstrap（PCG64，seed 2026091802）；下标同时应用于所有方向、评分和模块。分母 ≤1e-12 或任意重采样分母无效时整项标为 RATIO_UNRESOLVED，不丢弃重采样。误差带和状态严格按协议。

通过 cck 的 labgpu 调度运行 `run.sh /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp02/20260918_r1`。相同目录和身份支持原子续跑；文件锁阻止并发写同一运行。原源码及资产只读，身份或代码变化时必须新建运行。

`test_structure.py` 覆盖六组实质检查：正负交叉项、Kronecker/trace/全独立配对的 sep 等价式、配对 SE 与误差分解、共同缩放与 bootstrap 配对、分类/排序/零分母/重复拒绝、PSD 数值失败。

参考：[Martens & Grosse 2015](https://proceedings.mlr.press/v37/martens15.html)、[共享权重 K-FAC](https://arxiv.org/html/2311.00636v2)。本实现的边缘乘积只是指定的一个消融对象，不代表所有共享权重 K-FAC 或单 Kronecker 最优拟合。
