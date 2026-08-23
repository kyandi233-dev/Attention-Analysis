# INSTALL｜NIR Formal Runtime

本文件是 `runtime/nir-formal/` 在新的 NVIDIA/CUDA Windows 机器上的安装入口。当前正式分支为 `nvidia-cuda`；正式全量分析已经完成，本安装流程用于复现与迁移。

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

不要把根 `pyproject.toml` 的通用 Python 依赖误认为正式 CUDA runtime 的完整环境声明；GPU/PyTorch 后端由本 runtime 和目标机器共同决定。

## 3. 检查冻结资产

确认以下文件存在：

```text
models/nir-eye-yolo26n-best.pt
models/ritnet-best_model.pkl
ritnet/
config.yaml
run_pipeline.py
run_formal_batch.py
```

然后执行：

```powershell
pytest -q
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

## 7. NVIDIA 基线与 AMD 分支边界

本目录当前记录的是已经运行过的 NVIDIA/CUDA 正式口径。准备 AMD/DirectML 适配时，应先确认 NVIDIA 基线测试和环境检查通过并冻结版本，再从该节点创建 `amd-DirectML`。

AMD 分支可以修改设备后端、依赖和必要的 inference 适配，但不得反向改写已冻结 NVIDIA 正式分析参数和历史结果。
