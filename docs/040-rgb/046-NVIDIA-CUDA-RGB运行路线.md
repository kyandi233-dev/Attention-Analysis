# 046｜NVIDIA CUDA RGB 运行路线

**当前开发 Branch：** `rgb-nvidia`

> `nvidia-cuda` 是 NVIDIA 综合线，目前仍有人在使用；本轮 RGB 正式化只修改 `rgb-nvidia`。待 NVIDIA RGB 在 RTX 5070 上完成代表被试验收后，再选择性同步回 `nvidia-cuda`。

## 1. 当前目标

NVIDIA RGB 与 AMD RGB 保持相同 scientific contract，但执行后端不同：

```text
AMD:    Py-Feat scientific core → ONNX Runtime DirectML
NVIDIA: Py-Feat 2.1.1 Detectorv2 → native PyTorch CUDA
```

正式范围统一为：

```text
baseline start
→ instructions / practice / transition
→ Block1
→ inter-block transition
→ Block2 end
```

当前正式策略已经同步为 **raw-first**：先一次性保存昂贵/不可直接恢复的 Face、Pose、Motion raw；tracking、眼睑、Pose features、blink/PERCLOS、QC 和统计聚合全部后移。

## 2. NVIDIA Face 后端不能照抄 AMD DirectML

NVIDIA 正式 Face runner：

```text
scripts/face_formal_cuda.py
```

它使用 Py-Feat 2.1.1 原生 PyTorch/CUDA：

```python
from feat import Detectorv2

detector = Detectorv2(
    device="cuda",
    identity_model=None,
)

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

正式 runner 还有两层硬保护：

- `torch.cuda.is_available()` 必须为 `True`；
- `--device` 必须是 `cuda` 或 `cuda:<index>`。

因此 **不允许静默退回 CPU**。

### Batch 语义

AMD DirectML 是两个独立 ONNX 模型，因此有：

```text
RetinaFace batch
multitask batch
```

NVIDIA native Detectorv2 是一个端到端的 Py-Feat 调用，因此正式配置只使用：

```yaml
face:
  native_cuda_batch: 16
  native_cuda_prefetch_batches: 2
```

RTX 5070 的最优 `native_cuda_batch` 必须实机 benchmark，不能把 AMD 当前 B32/B64 机械复制过来。

## 3. 当前正式单被试流程

入口：

```text
scripts/run_rgb_formal_subject.ps1
```

默认流程：

```text
0. 检查 attention-face-cuda：
   py-feat == 2.1.1
   torch.cuda.is_available() == True

1. 生成 Face 15 Hz timestamp frame manifest

2. 三条 raw 并行：
   Motion → full-fps OpenCV
   Pose   → 10 Hz MediaPipe landmarks
   Face   → 15 Hz native PyTorch/CUDA Detectorv2

3. raw-only validator
   → sub-XXX_manifest.json
```

默认不使用 single shared decode。此前临时分支中的 `rgb_formal_full_runner_v1.py` 仅作为设计参考；当前正式架构优先保留三条独立 reader 的并行度。

## 4. 正式完成标准

只要求：

```text
sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json

sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json

sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json

sub-XXX_manifest.json
```

最终 validator 必须得到：

```text
completion_status = complete
extraction_complete = true
```

同时 NVIDIA Face manifest 还必须明确记录：

```text
execution_backend = pytorch_cuda
device = cuda / cuda:<index>
```

否则不能判定为 NVIDIA 正式完成。

## 5. 当前 raw 信息保留

### Face 15 Hz

Py-Feat native scientific core 保留：

- 所有检测到的人脸；
- no-face planned sample placeholder；
- FaceRect / FaceScore / canonical bbox；
- 20 AU；
- 7 emotion；
- valence / arousal；
- gaze；
- 6DoF head pose；
- 478 mesh；
- 68 compatibility landmarks；
- native blendshapes，包括 `eyeBlinkLeft/Right`；
- frame identity / timestamp / phase / behavior context；
- identity branch 固定关闭：`identity_model=None`。

### Pose 10 Hz

保留所有返回 pose 的 33 landmarks、normalized/world coordinates、visibility/presence、多 pose 以及 no-pose placeholder。

### Motion full FPS

保留亮度、相邻帧差、global motion energy、capture/timestamp gap 和 behavior context。

## 6. Derived 全部后移

以下都不再阻挡正式 raw 抽取：

```text
Pose features
Face tracking
primary-face selection
EAR / eyel睑开度 / aperture-iris
blink events
PERCLOS
QC
统计聚合
```

它们都可以从已经落盘的 Face/Pose/Motion raw 重建。

## 7. Stable Parquet schema

临时分支 `codex/rgb-nvidia-formal-pipeline-v1` 曾出现真实错误：

```text
ArrowNotImplementedError:
Unsupported cast from double to null using cast_null
```

正式 CUDA Face writer 已吸收等价修复：首个 streaming chunk 即使某 numeric 字段全为空，也使用稳定 nullable dtype，而不是让 PyArrow 把该列冻结成 `null`。

对应回归测试：

```text
tests/test_rgb_formal_schema.py
```

provenance：

```text
ported-from: 51d17c9a6b7db7a1114380910bb111db38293512
```

## 8. NVIDIA 环境与固定路径

仓库：

```text
D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda
```

原始数据：

```text
J:\Data
```

RGB 输出：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

RGB core：

```text
D:\conda_envs\attention-rgb
```

Face CUDA：

```text
D:\conda_envs\attention-face-cuda
```

## 9. 单被试实机验收

先同步 `rgb-nvidia`：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

git fetch origin --prune
git switch rgb-nvidia
git pull --ff-only
git status --short --branch
```

先检查 CUDA：

```powershell
D:\conda_envs\attention-face-cuda\python.exe -c "import torch,importlib.metadata as m; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, m.version('py-feat'))"
```

代表被试仍使用 NVIDIA 数据盘实际存在的：

```text
sub-130
```

正式 raw-first pilot：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda
```

如果只测试 CUDA batch：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda `
  -FaceBatch 16
```

**不要在 sub-130 首次 full-span 验收通过前启动整 cohort。**

## 10. Cohort runner

代码入口已经存在：

```text
scripts/run_rgb_formal_cohort.ps1
```

它具备：

- audit / inventory refresh；
- `analysis_eligible=True` 筛选；
- 已完成被试 skip；
- 单 raw branch resume；
- 单个被试失败记录并继续；
- `cohort_status.csv` / `cohort_manifest.json`。

但它目前属于 **implemented, pending RTX 5070 pilot validation**，不是“已经批准正式全量”。

## 11. 当前 Gate

进入 NVIDIA RGB cohort 前至少确认：

1. sub-130 15 Hz frame grid 正确；
2. native CPU reference ↔ native PyTorch CUDA representative parity 通过；
3. full-span `face_raw.parquet` 字段完整；
4. Motion / Pose / Face 三 raw 全部通过 validator；
5. `execution_backend=pytorch_cuda` 与 CUDA device evidence 存在；
6. stable nullable schema test 通过；
7. CUDA batch / peak memory / throughput 已记录；
8. no-face / multi-face / capture-gap 情况不丢帧身份。

当前结论是：**NVIDIA 正式代码已经同步到 raw-first + 三线并行架构，但仍需要 RTX 5070 的 sub-130 实机验收后才能冻结为 cohort 正式入口。**
