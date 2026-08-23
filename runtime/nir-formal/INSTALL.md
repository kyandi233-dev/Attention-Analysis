# INSTALL｜NIR Formal Runtime

本文件是 `runtime/nir-formal/` 在 AMD/DirectML Windows 机器上的安装入口。当前分支为 `amd-DirectML`，package version 为 `0.1.0`。

## 1. 获取当前分支

```powershell
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git switch amd-DirectML
```

如果仓库已经存在：

```powershell
git switch amd-DirectML
git pull
```

## 2. 创建独立环境

推荐 Python 3.11 的独立 Conda 环境。当前已验证环境为：

```powershell
conda create -p D:\CondaEnvs\nir-amd python=3.11 -y
conda activate D:\CondaEnvs\nir-amd
```

进入 runtime 安装依赖：

```powershell
cd runtime\nir-formal
pip install -r requirements.txt
```

`requirements.txt` 已固定 `onnxruntime-directml==1.24.4`，不再需要 Ultralytics、PyTorch 或 CUDA。DirectML 需要 Windows 10 1903+ 与 DirectX 12 可用 GPU/驱动。

## 3. 检查冻结资产

确认以下文件存在：

```text
models/nir-eye-yolo26n-best.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
directml_runtime.py
config.yaml
run_pipeline.py
run_formal_batch.py
```

然后执行 runtime 自包含测试与环境检查：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

`check-env` 必须列出 `DmlExecutionProvider` 并显示 YOLO/RITnet 两个 session 都以它为首选 provider；否则立即失败，不退回纯 CPU。

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

## 7. AMD 版本边界

AMD 版本从 NVIDIA 基线 `e63675a` 分出，只替换推理后端与运行依赖。FocusWave phase、YOLO confidence 0.40、ROI、FP32、CSV schema 保持不变；RITnet 固定 batch=16，尾批补位并丢弃补位输出。默认输出包含 `amd-directml`隔离层。
