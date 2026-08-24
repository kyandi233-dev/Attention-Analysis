# INSTALL｜NIR Formal Runtime

本文件是 `runtime/nir-formal/` 在新的 NVIDIA/CUDA Windows 机器上的安装入口。当前正式分支为 `nvidia-cuda`，package version 为 `1.0.1`。

运行接续、故障恢复、`.run.lock`、`completion.json` 状态判断和“看到什么状态下一步做什么”的规则见 [`RUNBOOK.md`](RUNBOOK.md)。后续执行者不应仅依赖聊天上下文判断正式任务状态。

## 1. 获取当前分支

```powershell
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git switch nvidia-cuda
```

如果仓库已经存在：

```powershell
git switch nvidia-cuda
git pull
```

## 2. 创建独立环境

推荐 Python 3.10/3.11 的独立 Conda 环境。示例：

```powershell
conda create -p D:\conda_envs\eye-ai python=3.11 -y
conda activate D:\conda_envs\eye-ai
```

先按目标 NVIDIA 驱动/CUDA 环境安装可用的 PyTorch CUDA 版本，再进入 runtime 安装其余依赖：

```powershell
cd runtime\nir-formal
pip install -r requirements.txt
```

若需短测可选 ORT CUDA FP32 profile，另安装与当前 CUDA/PyTorch 兼容的 GPU 包：

```powershell
pip install onnxruntime-gpu==1.24.4
```

不要把根 `pyproject.toml` 的通用 Python 依赖误认为正式 CUDA runtime 的完整环境声明；GPU/PyTorch 后端由本 runtime 和目标机器共同决定。

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
```

然后执行 runtime 自包含测试与环境检查：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

`check-env` 必须能够识别目标 NVIDIA GPU/CUDA、PyTorch、Ultralytics、OpenCV 以及两份冻结模型。

## 4. 挂载正式数据

正式数据在逻辑上位于 `正式实验` 与 `Data` 两个目录，但两块外接存储设备的 Windows 盘符可能在 `E:` / `F:` 之间交换。`config.yaml` 已经声明四个候选根：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

因此正常情况下不需要因为硬盘连接顺序改变而编辑配置。程序会忽略不存在的候选根。

挂载数据盘后先检查：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

确认输出中的被试编号和实际视频路径正确。如果同一被试被复制到了多个有效候选根，批处理应报告 duplicate；不要通过调整 roots 顺序来偷偷选择其中一份。

## 5. 正式运行

确认 dry-run 正确后：

```powershell
python run_formal_batch.py
```

仅在目标 NVIDIA 机器完成 CUDA EP 环境验收、短测和 parity 后才选择高速 profile：

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

## 6. 单被试诊断

如果需要绕过批处理直接跑一个视频，使用实际视频路径：

```powershell
python run_pipeline.py formal --video "<实际盘符>:\<数据根>\sub-033_\nir\sub-033_nir.avi" --device 0
```

不要在安装文档里把 `E:` 或 `F:` 固定解释成某一台机器或某一块硬盘。

出现 `initializing`、`running`、`failed`、stale lock、重复实例或批处理恢复问题时，按 `RUNBOOK.md` 的决策规则处理，不要仅依据 `processed_frames=0` 或旧 marker 判断任务是否仍在运行。

## 7. NVIDIA 基线与 AMD 分支边界

NVIDIA/CUDA 历史全量基线已由 `nvidia-v1.0.0` 冻结；当前完整性修复版为 `1.0.1`。仓库级 current baseline tests 与 runtime tests 的具体集合见根 `.github/workflows/ci.yml` 和 `tests/README.md`；新 NVIDIA 机器还应额外运行本页的 `check-env`、数据发现和 dry-run。

AMD/DirectML 适配应从该 `1.0.0` NVIDIA 基线节点创建 `amd-DirectML`。AMD 分支可以修改设备后端、依赖和必要的 inference 适配，但不得反向改写已冻结 NVIDIA 正式分析参数和历史结果。
