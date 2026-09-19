# 613 → 双卡 4090：冻结资产迁移检查

用户已选择迁移 613 已有产物，继续 functional-gradient step 1；不从实验一重跑。本目录独立于六个冻结源码目录，不改变原有源码身份。

## 先在 4090 主机检查环境

在 `KFAC-QERA-teacher-kl` 根目录、已激活的 `qera-original-a` 环境执行：

```bash
git pull --ff-only origin main
python -B tools/teacher_kl_migration/preflight_4090.py \
  --base .. --output "../qer_4090_preflight_$(date +%Y%m%d_%H%M%S).json"
```

把终端输出或生成的 JSON 发回。脚本只读显卡状态、软件包版本、可见 cgroup 内存限制、文件系统剩余空间、模型 safetensors 文件头及有深度/数量上限的 identity 清单。只创建报告文件，不 import torch、不加载模型、不下载数据、不申请显存、不执行旧启动脚本，不改变环境。报告中的 GPU 数量是 nvidia-smi 看到的设备；同时报告 CUDA_VISIBLE_DEVICES，不能据此断言进程可使用所有物理设备。

模型在其他位置时增加 `--model /path/to/checkpoint`；资产已迁移到其他位置时增加 `--assets /path/to/parents`。模型文件头和形状相同只证明布局相符，不能证明参数内容相同。`ready_for_experiment` 始终为 false，直到另行完成迁移、两卡实现和数值 pilot。

本工具不附带原服务器的参数哈希、环境或 checkpoint 身份参考。这些资料保留在原实验产物及本地副本中，后续迁移时单独核验。原配置中的路径、设备映射和环境约束仍然保留；不得直接删除校验以绕过迁移差异。

## 迁移范围

迁移至少要保留实验一和实验三的原 token/标签、原两模块的 S 缓存、每窗口 x 缓存、原 quantized/directions/factors/corrections、相关清单与环境身份。还需要哈希匹配的官方 MXINT3 量化源码，以及与父 teacher 一致的模型参数。

这些父产物不上传 GitHub，不用 4090 上更早的同名/相近实验目录替换。传输方式和目标路径待主机检查后确定；当前未启动数据传输，也未创建实验作业。

## 后续验收顺序

1. 校验目标模型参数和父资产文件，记录环境差异，保留原实验来源链。
2. 在独立版本中适配一个进程跨两卡加载 FP32 teacher；根据目标层定位张量设备，单独安排全尺寸 FP64 线性代数。不能使用当前每卡一个模型的双 worker 入口。
3. 重放原两模块的参考前向/梯度/收缩，并测浅层、大 gate/down 模块的完整形状资源。跨 GPU 型号的浮点差异不能用删掉哈希校验或改标签来掩盖，应分别记录参数身份和数值重放误差。
4. 通过数值、显存、主机内存和时间预算验收后，再执行固定 `{0,10,20,31}` 的 28 模块正式实验。

当前交付为环境/身份检查工具；双卡运行适配及 GPU pilot 尚未完成。
