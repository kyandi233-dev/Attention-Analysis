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

当前正式策略为 **raw-first**：先一次性保存昂贵/不可直接恢复的 Face、Pose、Motion raw；tracking、主脸选择、眼睑、Pose features、blink/PERCLOS、QC 和统计聚合全部后移。

## 2. NVIDIA Face 后端不能照抄 AMD DirectML

NVIDIA 正式 Face runner：

```text
scripts/face_formal_cuda.py
```

必须保持 Py-Feat 2.1.1 原生 PyTorch/CUDA 调用：

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

正式 runner 有硬保护：

- `py-feat == 2.1.1`；
- `torch.cuda.is_available()` 必须为 `True`；
- `--device` 必须为 `cuda` 或 `cuda:<index>`；
- validator 要求 manifest 明确记录 `execution_backend=pytorch_cuda`；
- 不允许 silent CPU fallback。

### Batch 语义

AMD DirectML 当前可以分别调 RetinaFace batch 和 multitask batch，因为它实际运行两个独立 ONNX 模型；AMD 最近测试到 B32/B64 只是 AMD executor 的性能参数。

NVIDIA native `Detectorv2` 是一个端到端调用，只有一个正式 CUDA batch：

```yaml
face:
  native_cuda_batch: 16
  native_cuda_prefetch_batches: 2
  native_cuda_num_workers: 0
  native_cuda_pin_memory: false
  native_cuda_batch_candidates: [16, 32, 64]
```

因此：

```text
AMD B32/B64 ≠ NVIDIA B32/B64
```

不能把两套数字机械对应。RTX 5070 最优 `native_cuda_batch` 必须根据同一代表被试的吞吐、峰值显存和稳定性实测后冻结。

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

### 为什么 NVIDIA 默认仍使用三条独立 reader

AMD 侧已经做过 single shared decode 实验：一个 AVI reader 同时供给 Motion/Pose/Face 时，会由最慢消费者对共享生产者产生反压，实测并不一定比三条独立 reader 更快。因此 NVIDIA 当前正式默认继续采用：

```text
Motion reader ─┐
Pose reader   ├─ parallel
Face reader   ┘
```

而不是把临时分支 `rgb_formal_full_runner_v1.py` 的 single-pass draft 直接升级为 NVIDIA 正式入口。

这不是否定 single-pass 的理论价值，而是当前没有 RTX 5070 实测证据支持替换已实现的并行架构。

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

同时 NVIDIA Face manifest 必须明确记录：

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
EAR / 眼睑开度 / aperture-iris
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

同步 `rgb-nvidia`：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

git fetch origin --prune
git switch rgb-nvidia
git pull --ff-only
git status --short --branch
```

检查 CUDA：

```powershell
D:\conda_envs\attention-face-cuda\python.exe -c "import torch,importlib.metadata as m; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, m.version('py-feat'))"
```

代表被试使用 NVIDIA 数据盘实际存在的：

```text
sub-130
```

正式 raw-first pilot：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda
```

若要测试 native CUDA batch，可以通过 runner 的 `-FaceBatch` 显式覆盖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda `
  -FaceBatch 32
```

候选值当前记录为：

```text
16 → 32 → 64
```

不要直接把 64 设为正式默认；先记录：

```text
input_frames_per_sec_total
cuda_peak_memory_allocated_bytes
total_wall_with_parquet_write
```

确认吞吐确实继续增加且显存有安全余量，再冻结正式 batch。

## 10. Cohort runner

代码入口：

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

当前结论：**NVIDIA 正式代码与 AMD 保持相同 raw-first scientific contract，但 Face executor 始终使用 native PyTorch/CUDA 自己的调用合同；在 RTX 5070 sub-130 实机 Gate 完成前，不批准 cohort 全量。**
