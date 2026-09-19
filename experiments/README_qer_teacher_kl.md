# Teacher-KL 实验代码与双卡 4090 迁移说明

本次交付包含六个阶段的原始实验代码、协议、固定配置与测试。最新 functional-gradient 阶段使用从 0 编号的 `{0,10,20,31}`，共 28 个模块。本次只推送代码，不启动实验。

这六个目录在 `.gitattributes` 中关闭自动换行转换，以保留参与 SHA256 身份计算的原始字节。请勿批量格式化这些冻结目录。交付前九个 CPU 测试入口全部通过，并核对五个历史阶段的本地源码与已保存 identity 一致；尚未运行双卡 4090 验收。

## 代码索引

| 阶段 | 目录 | 主要内容 |
| --- | --- | --- |
| 实验一 | `qer_teacher_kl_exp01/` | 冻结 FP32 teacher、量化残差、rank-64 方向、MC 曲率与 teacher KL；入口说明见 `REVIEW.md` |
| 幅度扩展 | `qer_teacher_kl_alpha_extension/` | 复用实验一，扩展 alpha 并检查 KL 与局部二次预测 |
| 实验二 | `qer_teacher_kl_exp02/` | 结构消融与同一曲率下的配对比较 |
| 实验三 | `qer_teacher_kl_exp03/` | 完整 A/G、single-Kronecker 拟合与 weighted SVD |
| 实验三诊断 | `qer_teacher_kl_exp03_diagnosis/` | 冻结候选在原样本和新标签下的诊断 |
| Functional-gradient step 1 | `qer_functional_gradient_step1/` | 固定 marginal 几何下 Residual/Gradient SVD、幅度校准、四个 rank、28 模块评价 |

在仓库根目录执行 `git pull --ff-only origin main` 获取代码。各子目录的 `protocol.md` 是原方案，`README.md`/`REVIEW.md` 是相应实现说明。不要把原方案中的历史启动要求当成此次启动授权。

## 双卡 4090 的当前适配状态

新增独立候选运行版本 [`qer_functional_gradient_4090_v1/`](qer_functional_gradient_4090_v1/README.md)，用于迁移父资产后继续最新阶段。它已通过本地测试，实际双卡 pilot 尚待执行。下述限制描述的是保留的六个原始冻结目录。

**当前提交保留 613/A6000 的冻结实现，不是已经验收的 4090 运行版本。** 原始源码参与父实验 identity 校验，因此本次没有原地修改这些源码、配置和父身份。

- 当前配置中的模型、数据、QERA vendor、父源码和父产物路径指向 `/home/cck`、`/data1/cck`、`/data2/cck`。启动入口还校验 `cck`、UID 1001 和 `labgpu` cgroup。拉取代码本身不会满足这些条件。
- teacher 要求完整 FP32。8B 参数的权重本身约需 30 GiB，未计激活和工作区；不能在一张 24 GiB 4090 上直接加载当前单卡配置。
- 实验一 `run.py:load_model` 已有按 `gpus` 数量划分层的实现，但当前冻结配置为 1。新阶段的 controller/worker 则按每个 worker 一张 GPU 设计，并将模型配置限制为单卡。不能只把 `workers` 改为 2 就认为完成了双卡 4090 适配。
- 迁移目标应先评估一个 worker 跨两卡承载同一个 FP32 teacher，逐阶段核对 hook、标签采样、反传、候选张量与线性代数所在设备。大 MLP 的完整 FP64 factor、特征分解及 dense SVD 也要单独测峰值。不能通过静默改成 FP16/BF16、对角统计或截断几何来规避原方案。
- 新设备映射会改变当前严格检查的 teacher 元数据，配置/源码修改也会改变实验 identity。需要新版本目录、新输出目录及显式的迁移来源记录；保留原参数、样本、标签、候选和父资产哈希校验。不能直接复用旧运行目录或删掉身份校验。
- 正式运行前必须在实际双卡 4090 上完成完整形状的数值与资源 pilot。已有 CPU 测试不能证明显存够用或 GPU 路径可用。

## GitHub 不包含的运行资产

本提交只包含代码和小型配置/协议，不含模型权重、数据集、缓存 S、factor、补偿张量和历史运行结果。后续阶段不仅依赖前序代码，还依赖冻结的前序产物：

1. 相同 Llama-3.1-8B checkpoint 与 `DOWNLOAD_MANIFEST.json`，原始/预处理数据及匹配版本的 QERA vendor 源码。
2. 实验一完整运行目录：identity、teacher identity、环境、窗口、标签、量化权重、冻结方向和相应清单。
3. 各阶段配置引用的前序运行目录；例如实验三还引用 alpha 扩展输出。具体清单以 `config.json`/`plan.json` 和 `parent_expected.json` 为准。
4. 从 functional-gradient 阶段继续时，需要实验三的原 8×4 拟合窗口/标签、两模块的 S 缓存、factor、correction、冻结清单和环境身份等资产。现有实现不会用新随机样本替代缺失文件。

服务器暂不可访问时，可以先拉取和阅读代码；后续须迁移上述资产，或在新环境中按依赖顺序重建并形成新的完整来源链。

## 本地检查入口

每个测试在自己的子目录运行，避免不同阶段同名模块互相遮蔽：

```bash
cd experiments/qer_functional_gradient_step1
python -B test_suite.py
python -B test_construction_fixture.py
```

其他阶段的 `test_*.py` 同样在各自目录运行。CPU 构造 fixture 模拟了 CUDA 传输，仅验证公式、文件冻结与恢复，不验证 GPU 执行。服务器环境版本可从原运行的 `environment.json` 核对；迁移后的环境差异和重放误差应单独记录。
