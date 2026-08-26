# RGB

> 2026-08-26｜当前 NVIDIA RGB 工作线为 `rgb-nvidia`。正式目标与 AMD 一致：**从 baseline 开始连续到 Block2 结束，完整保存 Face、Pose、Motion 三类 raw；tracking、眼睑、blink/PERCLOS、Pose features、QC 与统计聚合后移。**

## 最重要的两个入口

- **正式方法、算法与参数依据**：[`049-RGB正式分析方法与参数依据.md`](049-RGB正式分析方法与参数依据.md)
- **当前实现与全量状态快照**：[`410-RGB当前状态与全量执行总结_20260826.md`](410-RGB当前状态与全量执行总结_20260826.md)

049 记录长期稳定的方法学；410 记录截至当前的 AMD 实机状态、NVIDIA Gate、跨后端关系与下一步。

## 当前两端后端

```text
AMD:
Py-Feat RetinaFace ONNX
→ custom decode / NMS / crop
→ Py-Feat multitask scientific core ONNX
→ ONNX Runtime DirectML

NVIDIA:
Py-Feat 2.1.1 Detectorv2
→ native PyTorch CUDA
```

AMD 并没有更换另一套 face detector；两端共同的科学核心仍为 Py-Feat 2.1.1。区别是执行后端与 glue code，因此跨平台正式合并前需要 representative parity。

## 当前正式配置

| 模块 | 正式规则 |
|---|---|
| analysis span | baseline start → Block2 end，连续保留 |
| Motion | source-video full FPS，OpenCV frame difference |
| Pose | MediaPipe Pose Landmarker Lite，10 Hz |
| Face | Py-Feat 2.1.1 scientific core，15 Hz |
| Face threshold | 0.5 |
| Identity | disabled |
| Raw retention | all detected faces / poses + no-detection placeholders + complete timestamps |

NVIDIA Face 必须证明：

```text
execution_backend = pytorch_cuda
device = cuda / cuda:<index>
```

不允许 silent CPU fallback。

## NVIDIA 当前实现

Face 使用 native API：

```python
from feat import Detectorv2

detector = Detectorv2(device="cuda", identity_model=None)

fex = detector.detect(
    batch,
    data_type="tensor",
    batch_size=len(batch),
    num_workers=0,
    pin_memory=False,
    face_detection_threshold=0.5,
    progress_bar=False,
)
```

native Detectorv2 对外只有一个端到端 CUDA batch，因此 AMD 的 `RetinaFace B32 + multitask B64` 不是 NVIDIA 的两个 batch 参数。

当前候选：

```text
native_cuda_batch = 16  # 当前默认
benchmark candidates = 16 / 32 / 64
```

batch 是性能参数，可根据 RTX 5070 throughput 与 peak VRAM 调整，不改变科学定义。

## NVIDIA 当前 Gate

代表被试：`sub-130`。

在正式 NVIDIA cohort 前需要完成：

```text
sub-130 full-span Motion / Pose / Face raw
→ final validator
→ stable nullable Parquet schema test
→ native CUDA batch 16 / 32 / 64 benchmark
→ AMD DirectML ↔ native Py-Feat representative parity
```

在以上 Gate 完成前，不启动 NVIDIA 全 cohort。

## 正式数据流

```text
face_formal_prepare.py
        ↓
生成 15 Hz Face frame schedule
        ↓
┌────────────────┬─────────────────┬────────────────────────┐
│ Motion         │ Pose            │ Face                   │
│ full FPS       │ 10 Hz           │ 15 Hz                  │
│ OpenCV CPU     │ MediaPipe       │ Py-Feat / PyTorch CUDA │
└────────────────┴─────────────────┴────────────────────────┘
        三条独立 reader 并行
                     ↓
rgb_formal_validate.py
                     ↓
sub-XXX_manifest.json
```

当前默认不采用 shared single-decode。AMD 实测没有显示该模式比三 reader 并行更快，因此 NVIDIA 在 RTX 5070 有反向证据前保持独立 reader。

## Raw 可回溯范围

Face raw 保存：所有 detected faces、no-face planned sample、bbox / FaceScore、20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、478 mesh、68 compatibility landmarks、blendshapes（含 `eyeBlinkLeft/Right`）以及完整 frame/capture/unix_ms/behavior identity。

Pose raw 保存 33 landmarks、normalized/world coordinates、visibility/presence、多 pose/no-pose 与时间身份。Motion raw 保存 source-video full-FPS 帧差、motion energy 与 timestamp/capture gap。

因此 tracking、primary face、EAR、眼睑开度、blink、PERCLOS、Pose features、trial/block/probe/sliding-window 统计都可以后算，不需要重新跑 CUDA Face 或 MediaPipe Pose。

## 当前运行入口

详细环境与命令优先看根 README 和：

[`046-NVIDIA-CUDA-RGB运行路线.md`](046-NVIDIA-CUDA-RGB运行路线.md)

单被试 Gate：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda
```

指定 batch：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda `
  -FaceBatch 32
```

## 文档索引

| 编号 | 内容 |
|---|---|
| 041 | RGB 分析目标与数据流 |
| 042 | 面部分析工具与 Benchmark |
| 043 | 姿态与运动量分析方法 |
| 044 | RGB 输出 Schema 与信息保留原则 |
| 046 | NVIDIA CUDA RGB 运行路线 |
| 049 | **RGB 正式分析方法与参数依据** |
| 410 | **RGB 当前状态与全量执行总结** |

后续新增文档继续按 `410 → 411 → 412 ...` 编号，不使用 `050`。

## 当前优先级

```text
AMD cohort raw 全量继续运行
→ NVIDIA sub-130 full-span Gate
→ RTX 5070 batch benchmark
→ AMD DirectML ↔ NVIDIA native Py-Feat parity
→ NVIDIA cohort
→ tracking / Pose features / EAR / blink / PERCLOS
→ cohort QC 与统计分析
```
