# RGB

> 2026-08-26｜当前 NVIDIA RGB 开发工作线为 `rgb-nvidia`。`nvidia-cuda` 是正在使用的 NVIDIA 综合线，本轮不直接修改；待 `rgb-nvidia` 在 RTX 5070 上完成代表被试验收后，再选择性同步回综合线。

RGB 当前正式主线为 **Face + Pose + Motion**。目标是从 baseline 开始连续到 Block2 结束，把以后不能轻易恢复、或重跑代价较高的 raw 信息完整落盘；tracking、眼睑、blink/PERCLOS、Pose features、QC 和统计聚合全部后移。

> **跨平台方法学总说明：** [`049-RGB正式分析方法与参数依据.md`](049-RGB正式分析方法与参数依据.md)  
> 该文档统一记录 Face/Pose/Motion 算法、30/full-FPS—15 Hz—10 Hz 的参数依据、AMD DirectML 与 NVIDIA native CUDA 的实现差异、raw-first 原则以及 EAR/blink/PERCLOS 的下游重建逻辑。

## 当前后端

```text
AMD:    Py-Feat scientific core → ONNX Runtime DirectML
NVIDIA: Py-Feat 2.1.1 Detectorv2 → native PyTorch CUDA
```

NVIDIA Face **不使用 AMD ONNX/DirectML executor**。正式入口 `scripts/face_formal_cuda.py` 明确要求 CUDA，并禁止静默 CPU fallback。

NVIDIA 原始数据根：

```text
J:\Data
```

正式 RGB 输出：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

代表被试：

```text
sub-130
```

## 当前状态

| 模块 | NVIDIA 路线 | 状态 |
|---|---|---|
| Face scientific core | Py-Feat 2.1.1 Detectorv2 | 冻结 |
| Face backend | native PyTorch CUDA | formal runner 已实现，待 RTX 5070 full-span 验收 |
| Face cadence | timestamp-driven 15 Hz | Accepted |
| Motion | OpenCV frame difference full FPS | raw formal wrapper 已实现 |
| Pose | MediaPipe Pose 10 Hz | raw formal wrapper 已实现 |
| 单被试总控 | Motion / Pose / Face 三线并行 | 已实现，待 sub-130 实机验收 |
| raw validator | Face + Pose + Motion 完整性 + CUDA backend evidence | 已实现 |
| cohort resume | eligible discovery / skip / resume / failure continue | 已实现代码，尚未批准全量 |
| tracking / eyelid | 从 Face raw 后算 | 不阻挡抽取 |
| Pose features | 从 landmarks 后算 | 不阻挡抽取 |
| blink / PERCLOS | downstream | 不阻挡抽取 |

## 正式单被试流程

```text
face_formal_prepare.py
        ↓
生成 15 Hz timestamp frame grid
        ↓
┌────────────────┬─────────────────┬────────────────────────┐
│ Motion         │ Pose            │ Face                   │
│ full FPS       │ 10 Hz           │ 15 Hz                  │
│ OpenCV CPU     │ MediaPipe CPU   │ Py-Feat / PyTorch CUDA │
└────────────────┴─────────────────┴────────────────────────┘
        三条并行
             ↓
rgb_formal_validate.py
             ↓
sub-XXX_manifest.json
```

当前默认**不采用 shared single-decode**。AMD 侧已经实测过把 Motion/Pose/Face 绑定到单一视频生产者可能降低整体墙钟吞吐，因此 NVIDIA 在没有 RTX 5070 反向证据前继续保留三条独立 reader 的并行度。

正式入口：

```text
scripts/run_rgb_formal_subject.ps1
```

只要以下三类 raw 完整即可进入下一被试：

```text
sub-XXX_motion_raw.parquet
sub-XXX_pose_landmarks.parquet
sub-XXX_face_raw.parquet
```

对应 manifest 也必须完整，最终：

```text
completion_status = complete
extraction_complete = true
qc_pass = null
```

NVIDIA validator 还额外要求 Face manifest 证明：

```text
execution_backend = pytorch_cuda
device = cuda / cuda:<index>
```

## NVIDIA Face 调用方式

核心调用保持 Py-Feat 2.1.1 native API：

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

这里不能照抄 AMD 的执行器设计。AMD DirectML 当前会分别调 RetinaFace batch 和 multitask batch；NVIDIA native Detectorv2 只有**一个端到端 CUDA batch**。

当前 NVIDIA 配置：

```yaml
face:
  native_cuda_batch: 16
  native_cuda_prefetch_batches: 2
  native_cuda_num_workers: 0
  native_cuda_pin_memory: false
  native_cuda_batch_candidates: [16, 32, 64]
```

AMD 当前 B32/B64 只是两个 ONNX 模型各自的 batch，不代表 NVIDIA 也应该直接设成 32/64。RTX 5070 要通过 `-FaceBatch 16/32/64` 实测吞吐和峰值显存后再冻结正式值。

## Face raw 保留

Face raw 保留 Py-Feat native non-identity scientific outputs，包括：

- 所有 detected faces；
- no-face planned sample placeholder；
- FaceRect / FaceScore / canonical bbox；
- 20 AU；
- 7 emotion；
- valence / arousal；
- gaze；
- 6DoF head pose；
- 478 mesh；
- 68 compatibility landmarks；
- blendshapes（含 `eyeBlinkLeft/Right`）；
- subject / AVI frame / capture frame / unix_ms / phase / behavior context。

Identity 不属于 accepted scientific core，固定 `identity_model=None`。

## Derived 后移

以下项目不再进入正式 raw 抽取关键路径：

```text
Face tracking
primary-face selection
EAR
眼睑开度 / aperture-iris
blink
PERCLOS
Pose features
QC
统计聚合
```

只要 Face/Pose/Motion raw 已保存，这些都可以之后重算，不需要重新跑 CUDA Face 或 MediaPipe Pose。

## Stable Parquet schema

临时分支 `codex/rgb-nvidia-formal-pipeline-v1` 记录过 streaming Parquet 首 chunk 全空导致 `double → null` cast 失败的真实故障。当前 CUDA Face formal writer 已吸收稳定 nullable dtype 规则，并新增：

```text
tests/test_rgb_formal_schema.py
```

provenance：

```text
51d17c9a6b7db7a1114380910bb111db38293512
```

## 当前运行方式

同步分支：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

git fetch origin --prune
git switch rgb-nvidia
git pull --ff-only
git status --short --branch
```

检查 Face CUDA 环境：

```powershell
D:\conda_envs\attention-face-cuda\python.exe -c "import torch,importlib.metadata as m; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,m.version('py-feat'))"
```

单被试 full-span pilot：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda
```

指定 native CUDA batch：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda `
  -FaceBatch 32
```

完整 NVIDIA 运行路线见 [`046-NVIDIA-CUDA-RGB运行路线.md`](046-NVIDIA-CUDA-RGB运行路线.md)。正式方法学与参数依据见 [`049-RGB正式分析方法与参数依据.md`](049-RGB正式分析方法与参数依据.md)。

## 当前执行边界

`run_rgb_formal_cohort.ps1` 已实现 resume/skip/status，但 **在 sub-130 full-span raw、CPU↔CUDA representative parity、schema test 和 CUDA throughput/memory Gate 完成之前，不启动正式全 cohort**。

临时分支资产的吸收结果见 `RGB_TEMP_BRANCH_ABSORPTION_RESULT_20260826.md`。
