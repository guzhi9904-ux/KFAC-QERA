# FP32统计与迭代短测

目的：验证昂贵的矩阵乘法能否采用FP32，并测量实际收益。代码与旧实验、12模块正式配置隔离。
此版本不自动修改正式精度、不启动128/256窗口的完整实验。

固定L10的q_proj/v_proj/down_proj，覆盖4096×4096、1024×4096、4096×14336三种真实权重尺寸。
使用Exp03原始fit窗口0/1及原保存k0教师标签；KL检查窗口0/1及2/3，全部来自原train池。
不读取官方validation/test效果。2/3只是在此短测拟合之外，不是全新盲测数据。

四方法均对照；Token-joint3轮，Full-fit固定8轮（不宣称达到正式收敛）。
FP64与FP32使用同一输入、标签、量化权重、初始化规则和迭代数。
FP32分支只改变大矩阵乘法；样本间累计、标量、因子保存为FP64，TF32关闭。
开方、加权变换、dense gesvd、恢复和目标检查保留FP64。部署仍为FP32 Wq+FP32(C64)。

事先固定的验收门限见bootstrap.py/PLAN：

- 规范化A/G及Kronecker乘积相对差异<=1e-4；所有迭代标量交叉检查通过。
- 在FP64参考阻尼度量下，补偿后目标相对差异<=1e-4。
- 每个窗口的实际KL差异<=max(1e-8,1e-3×该窗口量化基线KL)，即基线非退化时恢复率差异<=0.1个百分点。
- FP32原始因子允许微小负特征值：绝对值既不得超过最大特征值的1e-5，也不得超过阻尼lambda的10%。
  这项是新分支明确登记的数值门限，不能代替FP64基准对照；不裁剪特征值、不改变阻尼。
- FP64原始PSD门限1e-10、开方重构1e-10、SVD检查1e-8、部署漂移1e-4保持原值。

每种尺寸用相同GPU驻留的原生FP32输入，交替执行FP64/FP32各3次Full-fit单轮，报告中位耗时。
包括精度转换和FP64归约，不含磁盘IO、模型前向或SVD；不可当作整轮提速比。
两卡依次测量：q/v在cuda0，down在cuda1，避免并发异尺寸负载干扰精度对照计时。
启动时先串行预热线性代数。整个短测累计活跃上限45分钟，输出上限64GiB。

```bash
PY=../conda_envs/qera-original-a/bin/python
$PY -B tools/kronecker_precision_pilot/test_precision.py
CUDA_VISIBLE_DEVICES=0,1 $PY -u -B tools/kronecker_precision_pilot/run.py \
  --from-config ../qer_multimodule_v1.json \
  --output ../qera_runs/precision_pilot_v1/run_01
```

`summary.json`明确记录passed及适用范围；失败候选保留failed.json，不用另一种精度冒充结果。
短测通过只支持这三模块、两样本、八轮及所列KL窗口；正式128/256统计与跨层仍需后续验收。
