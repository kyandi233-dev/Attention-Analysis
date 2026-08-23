# NIR Formal Runtime

这是 Attention-Analysis 当前正式 NIR 分析的自包含 NVIDIA/CUDA 运行包。正式全量分析已经执行；本目录用于复现、迁移、结果复核和后续平台分支开发的基线，不再代表候选路线。

## 当前正式流程

```text
FocusWave v3.1.3 phase windows
        ↓
逐帧 YOLO26n 眼框
        ↓
标准化 ROI
        ↓
RITnet batch inference
        ↓
frames.csv / eyes.csv / summary.json / run_manifest.json / phase_windows.json / overlays
```

正式配置：`config.yaml`。正式模型：

```text
models/nir-eye-yolo26n-best.pt
models/ritnet-best_model.pkl
```

当前 NVIDIA/CUDA 维护分支为 `nvidia-cuda`。AMD/DirectML 版本应从经过最终检查并冻结的 NVIDIA 基线节点创建独立 `amd-DirectML` 分支，不直接修改本 runtime 的既有 CUDA 复现口径。

## 正式原始数据发现

正式原始数据在逻辑上位于两个目录：`正式实验` 与 `Data`。两块外接存储设备在 Windows 下的盘符可能随连接顺序在 `E:` / `F:` 之间交换，因此 `config.yaml` 同时声明四个候选根：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

`run_formal_batch.py` 会忽略不存在的候选根，并在所有有效根中按 `sub-*_/nir/*_nir.avi` 发现被试。若同一被试的视频同时出现在多个有效根，会直接报告 duplicate，不静默选择其中一份。

这四个候选路径表示“两种逻辑目录 × 两种可能盘符”，并不表示存在四份正式数据。

## 环境安装

新 NVIDIA/CUDA 机器从 [`INSTALL.md`](INSTALL.md) 开始。安装完成后，在本目录执行：

```powershell
pytest -q
python run_pipeline.py check-env
```

`check-env` 应确认 Python、PyTorch/CUDA、Ultralytics、OpenCV、YOLO 权重和 RITnet 权重可用。

## 数据发现与批处理

先检查当前实际挂载的数据：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

`--dry-run` 会打印 selected subjects、实际视频路径、device、precision、batch size 和将要执行的命令，不运行正式推理。确认被试列表和视频路径正确后运行：

```powershell
python run_formal_batch.py
```

`batch.subjects.include: []` 表示发现所有编号不低于 `formal.min_subject_number` 的完整正式被试；`exclude` 用于显式排除。命令行 `--subjects sub-031,sub-033` 可临时覆盖 include。

## 单被试运行

优先直接使用实际视频路径，不把可变盘符写死：

```powershell
python run_pipeline.py formal `
  --video "<实际盘符>:\<数据根>\sub-033_\nir\sub-033_nir.avi" `
  --device 0
```

正式批处理仍推荐使用 `run_formal_batch.py`，因为它会统一执行候选根发现、重复被试检查、skip-completed 和输出命名。

## 关键正式参数

当前 `config.yaml` 冻结的主要参数包括：

- YOLO confidence：0.40
- YOLO imgsz：640
- YOLO NMS IoU：0.70
- tracking：`none`（正式主链逐帧 YOLO）
- 标准 ROI：320 × 160
- RITnet 输入：640 × 400
- RITnet batch size：16
- RITnet precision：fp32
- FocusWave release：v3.1.3
- 正式被试编号下限：31
- phases：baseline / instructions / practice / block1 / block2
- baseline：180 s
- 正式 block 数：2

这些参数属于已经运行过的 NVIDIA/CUDA 正式分析口径。后续 AMD/DirectML 适配如果需要改变后端、precision 或 batch 行为，应在 AMD 分支中明确记录，而不是静默改写本基线。

## 输出与恢复

默认正式输出根为：

```text
outputs/formal
```

实际全量结果应保存在仓库外独立分析目录；仓库中的相对输出设置主要用于 runtime 自检和可复现说明。

单次运行目录名包含 subject、FocusWave release、RITnet batch size 和 precision。`skip_completed: true` 时，如果预期 run directory 已存在 `summary.json`，批处理会跳过该被试；使用 `--force` 可显式重跑。

## 代码边界

本目录包含正式运行所需的 frozen runtime 代码和依赖说明。根仓库 `src/attention_pipeline/nir/` 中仍保留部分项目级几何、benchmark、review、sequence 等历史兼容逻辑，但它们不是本正式 runtime 的执行入口。

历史 tracking / PuRe / PuReST / 多算法候选路线通过 `docs/020-nir/`、`docs/工作记录/`、tags 和 Git 历史追溯。正式主链不依赖 `runtime/legacy/`。

## 最小验收

在准备冻结 NVIDIA 基线或从其创建平台分支前，至少检查：

```powershell
# repo root
pytest -q

# runtime/nir-formal
pytest -q
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

其中最后两项需要正式数据盘实际挂载；`check-env` 需要 NVIDIA/CUDA 环境可用。任何重复被试、模型缺失或环境后端错误都应显式失败，不用插值或默认路径掩盖。
