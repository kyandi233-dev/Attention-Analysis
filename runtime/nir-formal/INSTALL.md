# INSTALL｜NIR Formal Runtime

本文件是 `runtime/nir-formal/` 在 NVIDIA/CUDA Windows 机器上的安装入口。当前正式分支为 `nvidia-cuda`，package version 为 `1.0.1`。

当前 RTX 5070 工作站的 **RITnet 四分类 post-hoc 正式补跑**请优先看仓库根目录：

[`../../NVIDIA-RITnet全分类补跑使用说明.md`](../../NVIDIA-RITnet全分类补跑使用说明.md)

该根目录说明已经把“GitHub 拉取 → 激活环境 → CUDA/ORT 检查 → `J:/Data` 检查 → pytest → dry-run → 单被试验收 → 72 人正式补跑 → 恢复与输出检查”整理成可直接执行的 PowerShell 顺序。

运行接续、故障恢复、`.run.lock`、`completion.json` 状态判断和“看到什么状态下一步做什么”的规则见 [`RUNBOOK.md`](RUNBOOK.md)。后续执行者不应仅依赖聊天上下文判断正式任务状态。

原始行为时间线缺失时的受限任务窗恢复规则见 [`RECOVERY.md`](RECOVERY.md)。恢复结果单独输出并标记为 `recovery_complete`，不得混入正式完整 NIR 结果。

## 1. 获取当前分支

```powershell
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git switch nvidia-cuda
```

如果仓库已经存在：

```powershell
git switch nvidia-cuda
git pull --ff-only
```

## 2. 创建独立环境

当前 NVIDIA 工作站实际使用的 Conda 路径环境为：

```text
D:\conda_envs\eye-ai
```

已有成功环境时优先直接激活，不要无理由重建：

```powershell
conda activate D:\conda_envs\eye-ai
python --version
where.exe python
```

若新电脑完全没有该环境，推荐 Python 3.10/3.11 的独立 Conda 环境。示例：

```powershell
conda create -p D:\conda_envs\eye-ai python=3.11 -y
conda activate D:\conda_envs\eye-ai
```

先按目标 NVIDIA 驱动/CUDA 环境安装可用的 PyTorch CUDA 版本，再进入 runtime 安装其余依赖：

```powershell
cd runtime\nir-formal
pip install -r requirements.txt
```

当前 RITnet full-class extension 使用 ONNX Runtime CUDA；若当前环境尚未安装对应 GPU 包：

```powershell
pip install onnxruntime-gpu==1.24.4
```

安装后必须确认：

```powershell
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

输出必须包含：

```text
CUDAExecutionProvider
```

不要把根 `pyproject.toml` 的通用 Python 依赖误认为正式 CUDA runtime 的完整环境声明；GPU/PyTorch/ORT CUDA 后端由本 runtime 和目标机器共同决定。

## 3. 检查冻结资产

确认以下文件存在：

```text
models/nir-eye-yolo26n-best.pt
models/ritnet-best_model.pkl
models/nir-eye-yolo26n-best.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
ritnet/
config.yaml
run_pipeline.py
run_formal_batch.py
run_ritnet_fullclass_batch.py
```

然后执行 runtime 自包含测试与环境检查：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

`check-env` 必须能够识别目标 NVIDIA GPU/CUDA、PyTorch、Ultralytics、OpenCV 以及冻结模型；RITnet full-class 还必须单独确认 `CUDAExecutionProvider`。

## 4. 当前 RTX 5070 工作站数据根

当前 `nvidia-cuda` 分支对应的 RTX 5070 工作站正式原始数据根固定为：

```text
J:/Data
```

当前 `runtime/nir-formal/config.yaml` 与 Behavior 正式配置都应指向 `J:/Data`。这台 NVIDIA 机器不要套用 AMD 工作站的数据根。

检查：

```powershell
Test-Path "J:\Data"
Select-String -Path ".\config.yaml" -Pattern "J:/Data"
```

正式完整 NIR 主链的数据发现仍可运行：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

但当前 RITnet 四分类补跑应使用自己的 batch dry-run：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

## 5. 历史完整 NIR 正式运行

以下是完整 YOLO + ROI + RITnet 正式主链入口。**当前 full-class 补跑不需要重新执行这一链。**

确认完整主链 dry-run 正确后：

```powershell
python run_formal_batch.py
```

仅在目标 NVIDIA 机器完成 CUDA EP 环境验收、短测和 parity 后才选择完整主链的可选高速 profile：

```powershell
python run_formal_batch.py --backend ort-cuda
```

需要只跑少量被试时：

```powershell
python run_formal_batch.py --subjects sub-031,sub-033
```

需要显式重跑已完成目录时：

```powershell
python run_formal_batch.py --subjects sub-031 --force
```

再次强调：这些 `run_formal_batch.py` 命令用于完整正式主链，不是当前 RITnet full-class post-hoc 补全入口。

## 6. 当前 RITnet full-class 正式补跑

技术说明见 [`RITNET_FULLCLASS_EXTENSION.md`](RITNET_FULLCLASS_EXTENSION.md)，完整操作顺序见仓库根目录 [`../../NVIDIA-RITnet全分类补跑使用说明.md`](../../NVIDIA-RITnet全分类补跑使用说明.md)。

首名 RTX 5070 完整被试验收：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects "sub-XXX" `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

其中 `sub-XXX` 必须换成 full-class dry-run 中实际存在于 NVIDIA 队列的一名完整被试。

验收通过后正式全量：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

production 不加 `--validate-pupil`，使用 labels-only CUDA inference；不重新跑 YOLO。

## 7. 单被试完整主链诊断

如果需要绕过批处理直接跑一个完整 NIR 视频，使用实际视频路径：

```powershell
python run_pipeline.py formal --video "<实际视频路径>" --device 0
```

这仍然属于完整主链诊断，不等同于当前 full-class post-hoc 补跑。

出现 `initializing`、`running`、`failed`、stale lock、重复实例或批处理恢复问题时，按 `RUNBOOK.md` 的决策规则处理，不要仅依据 `processed_frames=0` 或旧 marker 判断任务是否仍在运行。

## 8. NVIDIA 基线与 AMD 分支边界

NVIDIA/CUDA 历史全量基线已由 `nvidia-v1.0.0` 冻结；当前完整性修复版为 `1.0.1`，RITnet full-class 补全阶段计划冻结为 `nvidia-v1.2-ritnet-fullclass`。

AMD/DirectML 使用独立分支 `amd-DirectML`。AMD 分支可以修改设备后端、依赖和必要的 inference 适配，但不得反向改写已冻结 NVIDIA 正式分析参数和历史结果；同样，NVIDIA 当前 `J:/Data` 也不应反向覆盖 AMD 工作站的数据根。
