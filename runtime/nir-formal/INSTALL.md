# 安装与迁移

本目录是正式 NIR 分析的自包含运行包。模型权重、RITnet 运行源码、配置和执行脚本都位于本目录内；实验原始视频和正式输出不存入 runtime。

## 新电脑推荐配置

1. 安装 NVIDIA 驱动、Anaconda/Miniconda，并创建 Python 3.11 环境：

```powershell
conda create -n eye-ai python=3.11 -y
conda activate eye-ai
```

2. 根据该电脑的 NVIDIA 驱动/CUDA 条件安装 GPU 版 PyTorch 与 torchvision。PyTorch 单独安装，不写入 `requirements.txt`，避免普通 pip 解析时替换为不合适的构建。

3. 进入本目录并安装其余依赖：

```powershell
pip install -r requirements.txt
```

4. 检查环境：

```powershell
python run_pipeline.py check-env
```

只有 `check-env` 确认 CUDA、YOLO 权重和 RITnet 权重均可用后，再开始正式运行。

## 数据位置

数据不需要复制进 runtime。可以直接通过命令行指定根目录：

```powershell
python run_pipeline.py formal --subject sub-033 --root "F:\正式实验"
```

批处理的数据根目录、被试 include/exclude 和输出目录可在 `config.yaml` 中配置。

## 正式运行

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
python run_formal_batch.py
```

正式默认口径为逐帧 YOLO26n + RITnet batch 16 + FP32。除非有新的验证记录，不要静默改变这些科研参数。

## 迁移检查

复制或 clone 后至少确认：

- `models/nir-eye-yolo26n-best.pt` 存在；
- `models/ritnet-best_model.pkl` 存在；
- `ritnet/` 存在；
- `config.yaml` 可读取；
- `python run_pipeline.py check-env` 通过。

历史 CUDA/PyTorch/OpenCV 排错过程仍可参考仓库 `docs/010-nir/08-22-04-NIR新电脑GPU环境配置与正式批处理运行指南.md`，但当前运行入口以本目录文件为准。
