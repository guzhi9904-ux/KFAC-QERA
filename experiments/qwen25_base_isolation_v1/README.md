# Qwen2.5-7B Base：独立 MXINT3 协议迁移

> 归档状态（2026-09-11）：A 收集已完成；Full-A 在 layer1 `down_proj` 求解出现目标暴涨，尚未验证解决。此目录归档此前交付的隔离入口和离线/续跑改动，不代表全流程已通过。只读诊断工具位于 `tools/precision_audit/qwen_full_a_audit_v1.py`，处理记录见 [ISSUE_LOG](../../docs/research/ISSUE_LOG.md)。

本入口只新增 Qwen 实验。不修改、运行或获取 Llama A/DG/Full-G 的写锁。
不得在正在运行 Full-G 的 `KFAC-QERA-98ad0c5` 目录中部署；入口会检查路径并拒绝执行。
使用独立发布目录、独立结果目录和独立数据缓存。共享模型、旧结果和官方 QERA 源码只读。
不安装依赖，不升级现有 conda 环境。Python 禁止向共享官方源码写入 pyc。

## 对齐范围与证据

参数来自服务器上实际保存的 `config_resolved.json` 和 Llama DG/MXINT3/Full-G manifest，
不是仓库中的旧 Qwen 模板（该模板是 Instruct、BF16 profiling、不同 batch）。
程序核对 manifest 链、实际旧 A 累计张量的 dtype/count、冻结的源码哈希；
独立目录中的旧计算辅助代码也必须与 Full-G 发布版本逐字节相同。

| 项目 | 本次 Qwen 设置 |
| --- | --- |
| 模型 | Qwen2.5-7B **Base**，本地 BF16 权重；28 层、196 个投影、112 组输入统计 |
| 量化 | MXINT3，block size 32，axis -1；只量化 7 类投影 weight |
| 保护参数 | QKV bias、embedding、lm_head、norm 均保持 teacher 值；补偿加在投影输出 |
| rank | 8 / 16 / 32 / 64；求 rank64，再取前缀 |
| 校准数据 | 冻结 SlimPajama revision 的前 5120 原始行，官方预处理，前 256 个完整 2048-token 窗口 |
| 数据校验 | 先用 Llama tokenizer 重放，必须逐 token 重现旧校准和 WikiText2 窗口；再用 Qwen tokenizer 重编码 |
| A 收集 | FP32 teacher、FP32 GEMM；Full-A 在 CPU FP64 累计，Diag-A 在 CPU FP32 累计；batch 继承实际旧 A 配置 |
| A 求根 | 冻结官方 SciPy sqrtm；沿用对称化、去虚部和归一化规则，不新增 damping |
| DG 收集 | FP32 frozen teacher，bs1；sequence CE SUM 梯度；CE chunk128；FP64 square/reduce/累加；CPU 保存激活 |
| 计数 | A：524288 输入 token；DG：524032 有效预测位置；不对梯度做中心化 |
| G 缩放 | 通道均值归一为 1，再 floor 1e-6，平方根转 FP32 |
| 求解 | FP32，full_matrices=True；冻结官方 A 逆处理；每模块、每 A、每 rank 的 Identity-G product drift ≤ 0.001 |
| 评估 | BF16，bs8，CE chunk256；WikiText2 token-PPL；不使用 chat template，不切换 word-PPL |
| 配置数 | BF16 + W3 + DA/FA × GI/GD × 4 ranks = 18 |

`full_gi` / `full_gd` 的 `full` 指 **Full-A**，不是 Full-G。本次不收集 Qwen Full-G。
DG 定义继承前轮，不等同于逐 loss Fisher。
所有 Qwen 方法使用同一份 Qwen 数据、A 和 Wq；GD 额外使用同一份 Qwen DG。
不能复用 Llama 的 token IDs、A/G 数值或低秩因子，也不直接比较两个模型的绝对 PPL。

模型/分词器本来就不同，因此 WikiText2 窗口数与 token 总数按 Qwen 实测，不硬编码 138/282486。
相同原始前缀和相同 token 预算也不保证两个 tokenizer 的前 256 窗覆盖完全相同的文本长度。
Qwen 的模型放置改为双 4090 balanced，A 改为每 shard 8 输入组以控制 RAM；这些是明确记录的调度差异，
不是声称跨模型、跨设备逐 bit 相同。TF32 关闭，CPU threads=14，沿用 DG 阶段设置。

Base 身份检查包括显式 model_id、目录名和原始模型卡，不仅看结构。
模型卡只是一项来源证据，不是权重签名；下载来源应为官方
[Qwen2.5-7B Base](https://huggingface.co/Qwen/Qwen2.5-7B)。准备阶段会冻结实际下载的全部相关文件哈希。

## 安全部署

发布 ZIP 内有独立顶层目录 `KFAC-QERA-qwen25-base-v1/`。上传到 `chengkang`，只解压到该层，
**不要在旧 Full-G checkout 执行 git pull，不要把补丁覆盖到旧目录。**

```bash
cd /share/home/tm902089733300000/a913520780/chengkang
unzip -n qera_qwen25_base_v1.zip
cd KFAC-QERA-qwen25-base-v1
conda activate /share/home/tm902089733300000/a913520780/chengkang/conda_envs/qera-original-a
hostname
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv
```

必须在新租的机器上运行。对照旧服务器 GPU UUID，确认不是相同 GPU。
目录隔离不能阻止用户误在旧机器上启动，也不能消除共享存储带宽/容量竞争。
新结果固定在 `qera_runs/qwen2.5-7b-base-mxint3-v1`，日志放在其外，避免污染初始化检查。
等模型下载完整后，先跑只读参考审计与 Qwen 数据准备：

```bash
ENTRY=experiments/qwen25_base_isolation_v1/run_server.sh
LOG=/share/home/tm902089733300000/a913520780/chengkang/qera_runs/qwen25-base-v1.log
nohup bash "$ENTRY" prepare --allow-download >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

`prepare` 可能需从 Hugging Face 读取原始数据；ModelScope 模型下载成功不代表数据源也可访问。
数据使用新私有缓存，不改旧缓存。网络不通或 Llama token 重放不一致时停止，不切换数据源/分片/随机采样。
prepare 日志 `audit-qwen PASS` 仅表示输入审计通过，**不代表 GPU 数值或显存试跑已通过**。

先确认 prepare 进程正常退出，再测试 A（每次新收集至少 4 个窗口，保存后有意退出 75）：

```bash
nohup bash "$ENTRY" pilot-a >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

看完显存/RAM/耗时并确认没有 OOM、错位或 gate 错误，再选择逐阶段运行：

```bash
bash "$ENTRY" collect-a --max-hours 10
bash "$ENTRY" roots --max-hours 10
bash "$ENTRY" quantize
bash "$ENTRY" solve-gi --max-hours 10
bash "$ENTRY" evaluate-gi --max-hours 10
bash "$ENTRY" pilot-dg
bash "$ENTRY" collect-dg --max-hours 10
bash "$ENTRY" solve-gd --max-hours 10
bash "$ENTRY" evaluate-gd --max-hours 10
bash "$ENTRY" summary
```

这些逐阶段命令不自动后台化；如需后台运行，使用上面的 `nohup ... >> "$LOG" 2>&1 &` 形式。
不要同时启动多条；同一个 Qwen 输出有独占锁。退出 75 后继续同一阶段，完成后再进入下一阶段。
试跑确认后也可自动顺序执行全部剩余阶段：

```bash
nohup bash "$ENTRY" run --max-hours 10 >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

`run` 自动经过 A→roots/W3→GI求解/PPL→DG→GD求解/PPL；不会在 GI 完成后停下来等待 DG 试跑确认。
保守做法是先逐阶段完成两种 pilot。中断后必须保持代码目录、环境和配置不变，不重新解压覆盖旧发布。

## 断点和资源

- A 共 14 shard，每个 shard 都跑完整 256 窗；进度会打印 shard、window、ETA、RAM/GPU 峰值。
- A 保存间隔继承实际旧 A 配置（模板为 32 窗）；DG 每 8 窗保存。pilot、SIGTERM/SIGINT、时间预算会在当前完整 batch 后补存。
- `CURRENT.json` 只指向全部张量/计数已经写完的 generation；断电/kill -9 最多重算最后一次已提交检查点之后的工作，不重复累计。
- 每个 shard 只保留最新已提交 generation 和写入中的新 generation；仅清理自己拥有的旧检查点。
- roots、量化、求解按模块续跑；评估每 batch 保存逐窗口 NLL。未完成配置不写入完整结果汇总。
- `--max-hours` 是协作式预算，不是硬截止；SciPy sqrtm/SVD/大文件写入期间不能立即退出，必须给租期留余量。
- Full-A 原始矩阵约 83 GiB，根约 42 GiB，另有 W3、因子、数据缓存和滚动检查点；入口要求至少 180 GiB 空闲。
- 还需另外预留 Llama Full-G 后续的磁盘需求；共享挂载的 `df` 空闲量不一定等于账号配额。
- Qwen 大词表和 18944 维 MLP 会改变峰值和耗时。双 4090 未做整模型本地验证；任何 OOM 都停止审查，不能偷偷改 bs、dtype、attention 或 CE 定义。

结果：`evaluation/ppl_summary_wikitext2.csv`、`wikitext2_per_window.csv`、`status.json`。
逐阶段重新验证实际输入哈希、源码哈希和环境版本；换路径、改配置、损坏文件时拒绝续跑。

## 本地验证边界

测试包含原始 A 精度细节、Qwen 模块/bias 清单、独立目录保护、旧辅助源码一致性、
checkpoint 写入/指针/清理故障注入、真实收集循环中断重启与连续累计对照、
小模型全流程、动态 WikiText2 窗口、Identity-G 失败阻断、评估逐 batch 续跑。
小模型测试使用独立量化/SVD参考函数，另有旧测试覆盖冻结官方实现；不能替代服务器上的 Qwen 数值门禁。

```powershell
$env:PYTHONPATH = "$PWD/src"
# 仅本机 Anaconda/PyTorch 存在双 OpenMP 冲突时使用；不是服务器协议变更。
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
D:/anaconda/python.exe -m pytest tests experiments -o addopts='' -q
```
