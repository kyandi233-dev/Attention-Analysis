# NIR Formal Runtime

这是 Attention-Analysis `amd-DirectML` 分支的正式 NIR 运行包。它保持已完成全量分析的 NVIDIA 科研口径，但将 YOLO 和 RITnet 推理后端替换为 ONNX Runtime DirectML。

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
models/nir-eye-yolo26n-best.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
```

当前 AMD package version 为 `0.1.0`。此分支从 NVIDIA 冻结节点 `e63675ad15c17db6ea2ac7a3bb1c1ac6fc106e06` 创建；NVIDIA/CUDA 复现仍在 `nvidia-cuda` 分支维护。

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

新 AMD/DirectML Windows 机器从 [`INSTALL.md`](INSTALL.md) 开始。已配置机器使用 `D:\CondaEnvs\nir-amd`。安装完成后，在本目录执行：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

`check-env` 必须确认 `DmlExecutionProvider` 存在且是两个 session 的首选 provider，并实际加载两个 ONNX 图。DirectML 不可用或 session 创建后变成 CPU 首选时立即失败。ORT 仍会保留 CPU EP 处理少量图节点，但运行时 provider failover 已禁用，不允许整个 session 静默退回纯 CPU。

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

这些科研参数与 NVIDIA 正式口径一致。AMD 版本额外强制 RITnet 固定 FP32 batch=16；尾批复制最后一个真实 ROI 补齐 16，推理后只保留真实 ROI 数量的输出。RITnet ONNX 在图内把四通道 logits 压缩成 UINT8 四分类 label map 和 FP32 pupil-probability map，减少 DirectML→CPU 数据搬运；四类 argmax 语义仍完整保留。

## 眨眼解释边界

RITnet 输出 background、sclera、iris、pupil 四类分割；当前正式后处理只使用 pupil 类拟合椭圆。把 sclera、iris、pupil 合成 ocular mask 后，可以派生候选眼裂高度/宽度和被试内 normalized openness，而且不需要第二次 RITnet forward。

但本 runtime 尚未把该候选量验证为正式眨眼指标。`ritnet_missing`、`yolo_missing`、瞳孔面积下降或低置信度都不能单独解释为 blink。AMD `0.1.0` 为保持既有 CSV schema，没有新增正式 openness/blink/PERCLOS 列。完整派生逻辑、unknown 门控、时间戳、基线与验证要求见 [`docs/020-nir/021-眨眼检测边界与RITnet派生开合度.md`](../../docs/020-nir/021-眨眼检测边界与RITnet派生开合度.md)。

## 输出与恢复

默认正式输出根为：

```text
outputs/amd-directml/formal
```

实际全量结果应保存在仓库外独立分析目录；仓库中的相对输出设置主要用于 runtime 自检和可复现说明。

单次运行目录名包含 subject、FocusWave release、RITnet batch size 和 precision。`skip_completed: true` 时，如果预期 run directory 已存在 `summary.json`，批处理会跳过该被试；使用 `--force` 可显式重跑。

## 代码边界

本目录包含正式运行所需的 frozen runtime 代码和依赖说明。根仓库 `src/attention_pipeline/nir/` 中仍保留部分项目级几何、benchmark、review、sequence 等历史兼容逻辑，但它们不是本正式 runtime 的执行入口。

历史 tracking / PuRe / PuReST / 多算法候选路线通过 `docs/020-nir/`、`docs/工作记录/`、tags 和 Git 历史追溯。正式主链不依赖 `runtime/legacy/`。

## 最小验收

在新机器复现时，仓库级回归与 DirectML 硬件自检分开执行：

```powershell
# repo root
python -m pytest -q tests/test_behavior_formal_bb.py tests/test_current_data_roots.py tests/test_formal_nir.py tests/test_io.py tests/test_nir.py tests/test_portable_nir_gpu_package.py
python -m pytest runtime/nir-formal/tests -q
```

在实际 AMD/DirectML 机器并挂载正式数据盘后，再执行环境/数据验收：

```powershell
cd runtime\nir-formal
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

轻量端到端验证可使用 `formal --phases block1 --max-frames 600`；运行目录会附加 `_smoke600`，summary 中会写入 `truncated_for_smoke_test: true`，不得当作完整被试结果。
