# Functional geometry search pilot v1

对应 [2026-09-20 协议](protocol.md)。这是独立的新实验目录；不改写上一轮 functional-gradient-SVD 的代码、冻结资产或结果。本目录的代码不会登录服务器、申请 GPU、改变代理、安装环境或自动启动下一轮实验。

## 本轮做什么

- 固定 teacher、父 MXINT3 Wq、理想 rank64。SVD 的输入始终是左右变换后的 **E**，不是 H(E)。
- 使用父 teacher sampled-label Marginal **solve** factors；已含两侧 0.001 阻尼，不再加一次。缺失时从原 8×4 文本/标签重建，同归一化、同阻尼，成本单列。
- `(a,b) ∈ {0,0.5,1}²`，完整 FP64 dense SVD 九次；复用一次 A/G 特征分解。
- 32 篇新 train 文章 ×4 标签选参数，冻结后用另外 16 篇 ×8 标签检验，并测同一 FP32 部署权重的真实全词表 teacher KL。
- 模块预先固定 L10.v_proj → L31.q_proj。默认两模块，可在配置阶段选一个；不能看到 L10 效果后临时调整。
- 每模块 256 个新梯度，两模块 512 个。第一个搜索窗口前两份标签包含 autograd/投影审计，不额外扩大新样本预算。另每模块做一份原历史标签的数值重放；因子重建最多额外 32 份历史梯度。
- 一次反传给所有候选评分；Selected 等于 A-only 或 Marginal-AG 时复用同一文件、物理评分和 KL，逻辑角色仍保留。

## 明天在现有双卡 4090 上执行

以下命令在你的 `KFAC-QERA-teacher-kl` 中执行。`prepare` 是 CPU 数据准备；**只有最后的 `run` 会启动 GPU 实验**。这份 pilot 本身就是新实验，不接上一轮的 `pilot/formal` 命令。

```bash
cd /share/home/tm902089733300000/a913520780/chengkang/KFAC-QERA-teacher-kl
git pull --ff-only origin main
conda activate /share/home/tm902089733300000/a913520780/chengkang/conda_envs/qera-original-a

python -B experiments/qer_functional_geometry_search_v1/test_geometry_search.py
```

先检查新增文本所需资产：

```bash
ls -ld ../teacher_kl_assets/geometry_inputs/wikitext2/train
ls -ld ../teacher_kl_assets/geometry_inputs/wikitext2/validation
ls -l ../teacher_kl_assets/geometry_inputs/calibration.safetensors
```

上轮的 `parents_v1` 包含父结果、标签和因子，**不能假定它包含原始 WikiText2 train 文本和最初的 256 个 SlimPajama 校准窗口**。这两项缺失时，按下节迁移小包；已有同哈希副本可直接通过 `--wikitext` / `--calibration` 指向原路径，不必复制。禁止重新下载一个不同版本数据集来绕过哈希检查。

```bash
python -B experiments/qer_functional_geometry_search_v1/configure.py \
  --from-portable-config ../qer_4090_v1.json \
  --profile dual4090 --modules 2 --hours 12 \
  --output-parent ../qera_runs/geometry_search_v1 \
  --output ../qer_geometry_v1.json

bash experiments/qer_functional_geometry_search_v1/run.sh \
  ../qer_geometry_v1.json ../qera_runs/geometry_search_v1/run_01 prepare
```

`configure` 只创建配置（拒绝覆盖）；`prepare` 核对父资产哈希、模型 tokenizer 和原始数据，排除历史文本并冻结 48 篇文章。无 GPU 工作。若不足 48 篇，输出 `data/data_shortfall.json` 并停止，不能补拼文章或偷偷缩减样本。

准备通过、GPU 空闲后手动启动。建议在你已有的 tmux 会话中运行：

```bash
mkdir -p ../qera_runs/geometry_search_v1
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 bash experiments/qer_functional_geometry_search_v1/run.sh \
  ../qer_geometry_v1.json ../qera_runs/geometry_search_v1/run_01 run \
  2>&1 | tee -a ../qera_runs/geometry_search_v1/run_01.console.log
```

控制台打印 START/COST 和逐份梯度进度；日志放在运行目录**旁边**。中断后，在相同代码和配置下重跑相同命令，会校验已有原子记录并跳过已提交样本。不要删除冻结文件来强行续跑。SIGINT/SIGTERM 在当前原子操作结束后停下，GPU 内核不会强行截断。没有后台提交、自动 SSH 或自动下一组实验。

默认 `--hours 12` 是本配置的**累计活动墙钟上限**，包含多次尝试和 CPU prepare，不是预计耗时。达到预算后保存不完整状态；重启不会清零预算。不要改配置绕过冻结。资源估计来自本次实际样本和分解耗时，写入 `resource_usage.json`；未测模块不凭空给 ETA。单次不可中断内核可能略超过时间上限。

CPU 单独重做报告：

```bash
bash experiments/qer_functional_geometry_search_v1/run.sh \
  ../qer_geometry_v1.json ../qera_runs/geometry_search_v1/run_01 report
```

## 若需要迁移原始文本和历史校准窗口

只需小包，不需要重传 17 GiB 父结果或 teacher。工具只读取指定路径并创建新的包/解包目录；没有任何 SSH、删除和覆盖逻辑。

在 613 的 **cck 用户**下、含本版代码的仓库里运行（下面是历史父结果位置；工具会核对实际存在和哈希，不能把历史路径当作当前可用性的保证）：

```bash
python -B experiments/qer_functional_geometry_search_v1/inputs_bundle.py pack \
  --exp01 /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp01_v2/20260918_r1 \
  --exp03 /data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp03/20260918_r1 \
  --output "$HOME/geometry_inputs_transfer_v1"
```

`pack` 从父 manifest 找原始语料和校准文件；若路径已迁移，显式追加 `--wikitext 当前目录 --calibration 当前文件`。记录终端返回的 `manifest_sha256`。通过你现有的 cck 连接传输 `inputs.tar` 和 `manifest.json` 到 4090，例如放在 `../teacher_kl_transfer/geometry_inputs/`。**不使用 613 管理员连接**。

在 4090 执行（用实际打印的 SHA256 替换占位符）：

```bash
python -B experiments/qer_functional_geometry_search_v1/inputs_bundle.py extract \
  --archive ../teacher_kl_transfer/geometry_inputs/inputs.tar \
  --manifest ../teacher_kl_transfer/geometry_inputs/manifest.json \
  --manifest-sha256 SOURCE_MANIFEST_SHA256 \
  --destination ../teacher_kl_assets/geometry_inputs
```

解包拒绝覆盖已有目录、路径穿越、符号链接、重复成员、额外成员和错误哈希。`prepare` 再按父 manifest 独立验证原语料/校准身份。

## 单 A6000

使用同样的父资产布局：`ASSETS/exp01` 与 `ASSETS/exp03`，可用指向对应只读父结果的目录链接；配置只读父数据，产物必须写到独立输出目录。

```bash
python -B experiments/qer_functional_geometry_search_v1/configure.py \
  --assets /YOUR/parent_assets --model /YOUR/Llama-3.1-8B \
  --wikitext /YOUR/wikitext2/dataset --calibration /YOUR/calibration.safetensors \
  --profile a6000 --modules 2 --hours 12 \
  --output-parent /YOUR/runs/geometry_search_v1 --output /YOUR/qer_geometry_v1.json
```

先执行 `prepare`。取得你有权使用的单张 A6000 后，用 `CUDA_VISIBLE_DEVICES=获配设备` 执行 `run`；进程内部只看到一张 GPU。不要绕过服务器租约/调度规则。程序不登录管理员账户，也不修改调度器。

## 环境与历史覆盖

沿用已运行成功的父环境：torch 2.3.0、transformers 4.44.2、datasets 2.21.0、numpy 1.26.4、safetensors 0.4.5、accelerate 0.33.0。GPU 执行会对照父 environment.json 验证这些版本；Python 小版本不要求相同。无需 matplotlib；热图为独立 SVG。

已知历史范围：Exp1/alpha/Exp2 共用旧 validation 窗口；Exp3/diagnosis/functional-gradient 共用旧 8 个 fit 窗口；加上原 256 个 SlimPajama 校准窗口。排除历史文章标题/正文身份、所有旧 validation 文章，以及历史窗口和新文章间的精确连续 64-token 重叠；新文章之间也检查。两个模块共享新文章和标签。

若在这些实验之外另用过文本，**配置前**通过可重复参数补充 `--extra-history-windows 文件.safetensors`（input_ids、可选全 1 attention_mask，2048 长度）和 `--extra-article-manifests 文件.json`（windows 中含 article_title/article_text_sha256）。无法发现未登记的外部历史文本，不能宣称排除了它们。

## 验收与产物

- 参数逐张量精确哈希，tokenizer/父 Wq/文本/标签/掩码哈希；原 S 若存在，历史重放 ≤1e-4；输入重放 ≤1e-5，共享权重 autograd ≤1e-5。
- SPD 不裁剪；SVD 谱尾恒等式 ≤1e-8。旧 Marginal 端点目标核对；非简并时还核对理想 C（1e-7）；简并记录矩阵不唯一。
- 使用父 FP32 部署次序：`W_deploy = Wq + FP32(P64 @ Q64)`。主 q 和 KL 都用同一部署权重。理想/部署 q 漂移按整个阶段、每个候选记录，超过 1e-4 停止并保留原始分数；接近零以固定 1e-12 尺度保护。
- 检验标签仅在该模块 selection_freeze 后访问。所有候选共享一次 sum-NLL 反传，`(sum_t g_tᵀ Rx_t)²/(2T)` 保留跨位置项。无全量新增 S 文件。
- q 为 2000 次文章配对 bootstrap，另报固定文章内标签配对 bootstrap；KL 只做文章 bootstrap。比值由均值计算，不裁剪恢复率。符号判断前施加数值保护：q 使用 2e-10 相对损伤尺度，KL 使用两倍预定重放容差/观测差的较大值。无旧 2pp 阈值。
- None ≤1e-12 时明确 DENOMINATOR_UNRESOLVED；不强选参数，也不把未做检验当成方法失败。Baseline-selected 的角色分数和区间必须精确一致。
- 输出默认限制 10 GiB，主机 staging 多候选 Rx，单 worker，模型与大矩阵分解分阶段。不会删除父资产。

重点查看 `RESULTS.md`、`summary.json`、`comparisons.csv`、`resource_usage.json` 和每模块 `search_heatmap.svg`。底层 `atomic/`、真实标签文件、selection/candidate freezes 支持独立复算；报告再次检查冻结绑定。资源记录中的原子输出 I/O 时间是阶段耗时的子集，不能重复相加。

本地验收覆盖完整两模块模拟（仍按 32×4+16×8 的标签预算）、真实小型 Llama 的反传/KL、SPD/谱端点/重排去重、预算和断点恢复、缺因子重建、统计与迁移校验。**这些是 CPU 验证，不是本次 8B/A6000/4090 的真实 GPU 验收或实验结果**。首次真实运行会在正式预算内执行数值检查，失败即保存诊断并停止。
