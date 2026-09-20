# FP64末轮对照

纯FP32乘法短测出现原始因子负特征值相对阻尼过大的失败，保留原结果和全部门限。
此独立对照尝试FP64边际统计/初始化，Token-joint前2轮FP32末轮FP64、Full-fit前7轮FP32末轮FP64。
总轮数仍分别为3/8，最终使用原来的严格FP64 PSD门限1e-10；不裁剪、不改变阻尼。
Marginal和Sequence-one-step继续采用原FP64参考版本。

复用源短测已冻结的两窗口缓存、FP64参考候选和原train池的4窗口KL结果。
不修改源短测，不触碰官方validation/test，不自动切换正式实验。
FP64参考目标、因子相对差异、实际KL门限全部沿用源短测。
范围仍仅为L10三种尺寸与两样本，不能据此宣称128/256窗口及跨层已通过。

```bash
PY=../conda_envs/qera-original-a/bin/python
$PY -B tools/kronecker_precision_polish/test_polish.py
CUDA_VISIBLE_DEVICES=0,1 $PY -u -B tools/kronecker_precision_polish/polish.py \
  --source ../qera_runs/precision_pilot_v1/run_01 \
  --output ../qera_runs/precision_polish_v1/run_01
```

独立累计时间上限30分钟、磁盘上限32GiB；结果为summary.json。
