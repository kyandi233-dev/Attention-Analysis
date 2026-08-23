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
models/nir-eye-yolo26n-best.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
```

当前 NVIDIA/CUDA 维护分支为 `nvidia-cuda`，package 版本为 `1.0.1`。历史全量 `1.0.0` 基线由 tag `nvidia-v1.0.0` 冻结；默认 `pytorch-cuda` 继续使用原 `.pt/.pkl`，新增 `ort-cuda` 只是显式可选的 FP32 高速 profile。

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
python -m pytest tests -q
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

默认仍是冻结的 PyTorch CUDA 路径。在目标 NVIDIA 机器安装 `onnxruntime-gpu`、通过 parity 和短测后，可显式选择：

```powershell
python run_formal_batch.py --backend ort-cuda
```

`ort-cuda` 固定 FP32、RITnet batch=16、尾批用最后一个真实 ROI 补位并丢弃补位输出。它只注册 `CUDAExecutionProvider`，禁用 CPU EP fallback 与运行期 fallback，并关闭 TF32；CUDA EP 不可用或任一节点不能在 CUDA 执行时直接失败。

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

单次运行目录名包含 subject、FocusWave release、RITnet batch size、precision，ORT profile 另带 `ort-cuda`。`skip_completed: true` 只会跳过通过身份、phase、帧集合、计数和产物校验的 `completion.json: complete`；只有 `summary.json`、smoke、partial、读帧失败或损坏 marker 都会重跑。`--max-frames` 或非完整 phase 只会发布 `smoke_complete`，读帧失败发布 `failed` 并返回非零码。

## 代码边界

本目录包含正式运行所需的 frozen runtime 代码和依赖说明。根仓库 `src/attention_pipeline/nir/` 中仍保留部分项目级几何、benchmark、review、sequence 等历史兼容逻辑，但它们不是本正式 runtime 的执行入口。

历史 tracking / PuRe / PuReST / 多算法候选路线通过 `docs/020-nir/`、`docs/工作记录/`、tags 和 Git 历史追溯。正式主链不依赖 `runtime/legacy/`。

## 最小验收

准备复现 NVIDIA `1.0.1` 或在新机器上短测 profile 时，仓库级 current baseline 与 runtime 自检应分开执行。根仓库 current baseline 使用与 `.github/workflows/ci.yml` 相同的可移植测试集合：

```powershell
# repo root
python -m pytest -q tests/test_behavior_formal_bb.py tests/test_current_data_roots.py tests/test_formal_nir.py tests/test_io.py tests/test_nir.py tests/test_portable_nir_gpu_package.py
python -m pytest runtime/nir-formal/tests -q
```

在实际 NVIDIA/CUDA 机器并挂载正式数据盘后，再执行环境/数据验收：

```powershell
cd runtime\nir-formal
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

最后三项依赖目标机器的 GPU/CUDA 与正式数据盘，不属于干净 CI 能模拟的项目级测试。任何重复被试、模型缺失或环境后端错误都应显式失败，不用插值或默认路径掩盖。
