# Fixed-geometry functional-gradient SVD — code delivery v1

**本次只部署代码，不申请GPU、不提交实验、不安排自动执行。** GPU真实形状试跑尚未运行；不得将本地公式测试写成服务器数值验收通过。

用户已确认使用从0编号的 **{0,10,20,31}**，每层 `q/k/v/o_proj`、`gate/up/down_proj`，共28个独立干预模块。这覆盖协议第13节的默认首层1；其他科学设置不变。原L10.v_proj、L31.q_proj优先。

## 已实现

- 只读校验父Exp-3 identity、源码、冻结数据、factor、correction及64份原S缓存；原两模块复用父资产。
- 新增26模块使用同一MXINT3参数、原8×4标签、完整FP64 A/G、canonical L/T及一次eta=0.001 trace-relative damping。每个模块单独量化/统计，几何不跨模块复用。
- 两次完整dense SVD，各取16/32/64/128前缀；Residual/Gradient方向均保留原幅度和同规则有符号beta，不裁剪beta。保存M、低秩因子、beta及实际部署身份。
- 科学评分使用真实FP32部署残差。为保证可逐位重放，同时在服务器保存16份FP32部署权重；这些大文件不应默认下载到本地。GPU上逐候选构造，评价时仅保留17份Rx及必要工作区。
- 128份新标签使用base seed 2026091901，命名空间`qer-functional-gradient-step1-v1`，与fit及旧diagnosis种子分离。两个worker共享文件锁保护的同一标签集；每份反传同时投影17候选。
- rank64五候选全词表KL，预定窗口0重放；每次干预后恢复并验证teacher权重。第一条数值pilot样本计入正式集合一次。
- 独立频数加权bootstrap复核、实际幅度阈值与方向分开、28模块完整/缺失清单、逐rank曲线和Markdown报告。
- 按模块与样本保存检查点。父资产只读；新临时S在完整构造/原经验数值验收通过、保存重建身份并冻结候选后立即清理，不累积到28模块全量评价之后，不保存逐样本G Gram。

## 入口与依赖

服务器代码目录：`/home/cck/projects/qer_functional_gradient_step1_v1`

输出父目录：`/data2/cck/KFAC-QERA/runs/qer_functional_gradient_step1/`

Python：`/home/cck/miniconda3/envs/kfac-qera/bin/python`。复用已冻结的 `/home/cck/projects/qer_teacher_kl_exp01_v2_r1` 和 `/home/cck/projects/qer_teacher_kl_exp03_r1`；不安装、升级或改动这些依赖。

`test_suite.py` 只进行CPU小矩阵及统计测试；`test_construction_fixture.py` 在CPU上以模拟CUDA传输验证32份样本、17个候选的构造、冻结文件和只读恢复。后者不验证CUDA执行。`controller.py`、`worker.py`和`run.sh`必须在cck的labgpu租约内运行，不会自行申请/释放GPU，也没有管理员入口。

## 稍后有空闲资源时的执行顺序

1. 在 `613-cck` 下确认 `id -un` 为cck，执行 `gpu status` 检查实际可用资源。不要绕过仍隔离的旧租约。
2. 申请匹配的资源：单卡可从64GiB主机内存起；若只有32GiB，可先在32GiB上做单卡pilot。双worker须申请同一租约2卡、共96GiB，不是每个worker96GiB。
3. 先运行pilot。它实际覆盖原两模块，以及L0.q_proj、L0.gate_proj、L0.down_proj：完整形状、完整32份构造统计、两种SVD、17候选投影、5个KL及重放。只按数值和资源选择执行方式。
4. 检查 `pilot_acceptance.json`：所有数值通过，GPU allocated峰值<45GiB，作业总主机内存峰值<申请量的90%。双卡必须实际运行重叠worker；仅测单卡不能直接开始双卡formal。
5. 同一输出目录运行formal。先补齐全部28模块构造、冻结总清单，再完成全量新标签与KL。pilot中已保存的首条评价复用一次，其他模块按固定清单执行。
6. 完成后检查 `status.json=COMPLETE_VERIFIED`、`verification.json`以及所有逻辑计数，再核对作业确已停止并正常释放本次租约。若调度器拒绝释放，记录原因交给用户，不使用管理员连接或强制清卡。

以下是**待后续显式执行的命令模板**，本次部署不执行。将 `<LEASE_ID>` 替换为实际租约，request-id每次使用新的16位十六进制值。单卡的workers为1，双卡为2。

```bash
# 只在613-cck / cck下操作
gpu status
# 资源充足时择一申请，申请操作不会运行模型：
gpu acquire --gpus 1 --mem 64G --idle 1h --request-id <NEW_REQUEST_ID> --no-wait
# 或：gpu acquire --gpus 2 --mem 96G --idle 1h --request-id <NEW_REQUEST_ID> --no-wait

# 租约必须READY；同一份run目录同时只允许一个controller。
gpu run --lease <LEASE_ID> --request-id <NEW_REQUEST_ID> -- /bin/bash \
  /home/cck/projects/qer_functional_gradient_step1_v1/run.sh \
  /data2/cck/KFAC-QERA/runs/qer_functional_gradient_step1/20260919_r1 pilot 1

# pilot通过后，确认租约READY（若已过期则重新申请匹配资源），再提交：
gpu run --lease <LEASE_ID> --request-id <NEW_REQUEST_ID> -- /bin/bash \
  /home/cck/projects/qer_functional_gradient_step1_v1/run.sh \
  /data2/cck/KFAC-QERA/runs/qer_functional_gradient_step1/20260919_r1 formal 1
```

## 恢复、预算和产物

同一源码、plan、父身份、worker数量和不小于pilot的主机内存下，可以重新提交同一stage与输出目录，已验证的模块/样本不重复计数。源码改变必须新版本/新目录。异常退出留下未关闭controller账目时，先核对进程与已消耗时间，不自动重置预算。

pilot累计上限2小时；正式单卡累计上限14小时，双卡9小时。模块检查点不绕过累计预算。短试跑会记录成本估计，但不承诺未经实测的结束时间，也不会根据效果增大样本、挑选rank或修改参数。

磁盘规划上限192GiB（持久factor/M/部署权重、临时S及日志的总预算；每30秒在原子边界检查）；父缓存不复制。单个大MLP临时S约14GiB，双worker约28GiB，统计检查点及原子写入另有短时开销；保存部署权重是为了固定舍入后的真实干预。`candidate_freeze.json`后的张量不可修改。输出包括协议要求的构造、方向统计、60,928行新标签评分、1,120个KL主评分、140个固定重放、rank曲线、逐模块数值验收、独立复核与资源记录。

CPU小测试覆盖显式H/column-vec、H=K、收益和代理配方、gauge/整体缩放、正负及零射线、真实dense SVD前缀、SE/配对bootstrap与KL统计。它们不替代未来A6000、浅层反传和14336² factor的真实资源pilot。
