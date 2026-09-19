# Functional-gradient step 1：双卡 RTX 4090 独立版本

这是在原 613 冻结实现之外新增的运行版本，继续原实验一/三的既有结果。代码和 CPU 测试已准备好；**实际两张 4090 的完整形状 pilot 尚未通过，不能称为 GPU 验收通过。** 不会自动提交作业、连接 SSH、申请 GPU 或启动正式实验。

## 执行方式与科学设置

一个 worker、一个 FP32/eager teacher。embedding 和层 0–15 在逻辑 GPU 0；层 16–31、norm、lm_head 在逻辑 GPU 1。不是 DDP，也不是每张卡各加载一份 8B 模型。固定使用两张可见 4090，目标模块的统计与 FP64 计算在该模块所在设备执行；模型卸载后释放两卡缓存，再做 full-channel dense eig/SVD。

共享 `model.rotary_emb` 显式放在 GPU 0；各层 RoPE 随层放置。首次前向前检查全部参数及包含非持久缓存的全部 buffer，并单独移动/检查 `original_inv_freq` Tensor 引用，保持频率精度和值不变。检查结果写入 `device_placement.json`。`test_device_layout.py --cuda` 使用随机四层小 Llama 复现旧设备错误并检验修复后的两卡前向、反传与 checkpoint；它不代替完整 8B pilot。

四层 `{0,10,20,31}`、28 模块、rank 16/32/64/128、原 8×4 拟合标签、8×16 新标签、四种候选方向/校准方式及 None、rank64 全词表 KL 均继承原方案。17 个 Rx 在 CPU 保存，评分时逐个移到目标设备，减少显存驻留；不改变收缩顺序、精度、样本数或候选。

独立运行身份同时包含本目录源码、借用的冻结父源码、固定 plan 和目标机本地配置。原目录保持只读。旧源码的字节哈希、父资产文件/张量哈希、实际部署残差身份继续核验。

目标 checkpoint 可以没有原服务器的 `DOWNLOAD_MANIFEST.json`，但必须通过 **全部 291 个 FP32 参数的逐项精确哈希校验** 和模型行为配置比较；缺少 manifest 会明确记录，不伪造相同来源。主要 Python 库版本必须与父实验一致；PyTorch 要求基础版本相同，记录 CUDA build、Python 和设备映射的变化。

跨显卡不假设前向输出逐位相同。原两模块每个窗口的第一个旧标签重放 x 和 S，分别要求相对误差 ≤1e-5 和 ≤1e-4，并保存与原 hidden hash 是否相等。原 S/标签本身不改写，拟合仍直接使用原 S。原始共享权重 autograd 验收（≤1e-5）、SVD 重构、rank64 父补偿复核、FP32 部署误差和 KL 重复验收保留。若精确参数身份或数值重放失败，停止排查，不自动放宽阈值。

2026-09-19 经用户明确同意，plan 版本由 `functional_gradient_step1_4090_model_parallel_v1` 修订为 `functional_gradient_step1_4090_model_parallel_v2`：仅将跨硬件 `parent_S_relative_tolerance` 从 1e-5 改为 1e-4。依据是迁移诊断发现 A6000 与双卡 4090 的旧 S 重放存在小幅数值差异，而参数身份、输入门限及同机 autograd 检查通过。这是观察诊断结果后作出的、有记录的迁移验收修订，不是原门限下验收成功，也不能据此保证最终实验结论不受影响。其余 plan 设置逐项保持不变。源码目录和本地配置路径保留兼容名称；新版 plan/README 自动进入运行身份。必须使用新输出目录（本次 `run_04`）重新运行完整 pilot；旧失败记录和诊断均不能替代新版 pilot 验收。

重放失败时保存 `modules/*/portability/wXX.failed.json`，包含实际误差、阈值和已完成的梯度审计。控制器在主日志中附上失败 worker 的日志路径及末尾异常。独立诊断工具 `tools/teacher_kl_migration/diagnose_parent_replay.py --config CONFIG --output NEW_DIAGNOSIS_DIR` 对两个父模块的全部八个窗口各重放第一个旧标签，输出诊断报告；它不构造候选、不修改阈值，也不会生成 pilot 验收。诊断必须使用空闲双卡，输出为配置中 output_parent 下的新目录；建议外部 `timeout 510s` 限制总耗时。

## 1. 迁移并核验父资产

迁移包通过私有 SSH 传输，**不上传 GitHub**。需要 `parents.tar`、`manifest.json` 和从可信渠道取得的 manifest SHA256。包内包含 `exp01/`、`exp03/`、`vendor/`；模型仍使用目标机现有 checkpoint。

在仓库根目录执行（替换 `<MANIFEST_SHA256>`）：

```bash
python -B tools/teacher_kl_migration/parent_bundle.py extract \
  --archive ../teacher_kl_transfer/parents.tar \
  --manifest ../teacher_kl_transfer/manifest.json \
  --manifest-sha256 <MANIFEST_SHA256> \
  --destination ../teacher_kl_assets/parents_v1
```

逐文件流式核验后才写 `migration_verified.json`，拒绝已有目标目录、路径越界和链接。失败会保留尚未通过验收的目录，不能把它当成已迁移；检查原因后选择新目标目录，不自动删除。解包期间保留 tar 与展开文件需约两份父资产的空间；正式输出另预留约 192 GiB，文件系统总剩余空间不代表用户配额。

## 2. 创建目标机本地配置

```bash
python -B experiments/qer_functional_gradient_4090_v1/configure.py \
  --assets ../teacher_kl_assets/parents_v1 \
  --model ../modelzoo/Meta-Llama-3.1-8B \
  --output-parent ../qera_runs/functional_gradient_4090_v1 \
  --output ../qer_4090_v1.json
```

配置只在目标主机保存，包含绝对路径及迁移清单身份。配置存在时拒绝覆盖。修改配置/代码需要新的运行目录，不能绕过旧 identity。

## 3. 明确启动 pilot

激活已检查的环境，确认两张卡可用后：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash experiments/qer_functional_gradient_4090_v1/run.sh \
  ../qer_4090_v1.json \
  ../qera_runs/functional_gradient_4090_v1/run_04 pilot
```

先验证原两个模块，再跑 L0.q_proj、L0.gate_proj、L0.down_proj 的完整形状。它包括原 32 样本构造、两次 dense SVD、首条新样本的 17 个候选投影、五个 KL 及重放。CPU 测试不能替代这一阶段。日志在运行目录 `logs/`，出错同时保存模块状态。

pilot 累计预算仍为 2 小时，正式阶段仍为单 worker 14 小时；4090 的 FP64 速度必须实测，不能预先保证在预算内完成。GPU allocated 峰值门限改为每卡 21.5 GiB，主机峰值须小于可见 cgroup 限额的 90%。内核没有 `memory.peak` 时，每 50 ms 采样 `memory.current`，在验收记录中明确标记 `sampled_memory_current_50ms`；这不是内核精确峰值，可能漏掉更短的尖峰。

## 4. 查看验收后才启动 formal

检查 `pilot_acceptance.json` 的数值结果、每卡显存、主机内存测量方式、耗时估计，以及 `modules/*/portability/` 的跨显卡重放。成功后使用同一配置/输出目录，将上面命令最后的 `pilot` 改为 `formal`。控制器会再次校验匹配的 pilot 身份和资源条件；正式阶段先构造并冻结全部模块，再完成全量评价。中断后保留检查点；未关闭的控制器账目需要先核对，不重置累计预算。

完整结果要求 28 模块、60,928 个新标签评分、1,120 个 KL 主评分、140 个重放和独立复核通过。配置、资产迁移、pilot、formal 四步均不会因为拉取代码而自动运行。
