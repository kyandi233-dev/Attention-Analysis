# Attention-Analysis｜AMD RGB 正式工作线

当前 branch：`rgb-amd`。本分支用于 Windows AMD 工作站上的正式 RGB `Face + Pose + Motion` raw 全量提取。

> 当前状态：**正式三线并行 raw 主链已通过 full-span validator，AMD cohort 已进入正式全量执行。** `sub-036` 已验证 `completion_status=complete`、`extraction_complete=true`、`issues=[]`、`warnings=[]`。

## 1. 当前正式定义

正式分析范围：

```text
baseline start
→ instructions / practice / transition
→ Block1
→ inter-block transition
→ Block2
→ Block2 end
```

正式 RGB 三支：

| 模块 | 时间分辨率 | 算法 / 后端 |
|---|---:|---|
| Motion | 原视频 full FPS（约 30 fps） | OpenCV 相邻灰度帧差 |
| Pose | 10 Hz | MediaPipe Pose Landmarker Lite |
| Face | 15 Hz | Py-Feat 2.1.1 scientific core → ONNX Runtime DirectML |

Face 当前性能配置为：

```text
RetinaFace batch = 32
multitask batch = 64
face threshold = 0.5
NMS = 0.4
prefetch = 3
CPU postprocess inflight = 2
```

其中 batch / prefetch / inflight 属于性能参数；15 Hz、Py-Feat scientific core、0.5 detection threshold 等属于科学/测量定义，不在全量过程中按被试改变。

## 2. 正式主链

```text
face_formal_prepare.py
        ↓
生成 timestamp-driven 15 Hz Face frame schedule
        ↓
┌────────────────┬───────────────────┬─────────────────────────────┐
│ Motion         │ Pose              │ Face                        │
│ full FPS       │ 10 Hz             │ 15 Hz                       │
│ OpenCV         │ MediaPipe         │ Py-Feat ONNX / DirectML     │
└────────────────┴───────────────────┴─────────────────────────────┘
        三条独立 reader 并行
                     ↓
rgb_formal_validate.py
                     ↓
sub-XXX_manifest.json
```

正式 raw extraction **不再等待**：

```text
Face tracking
primary-face selection
EAR / eyelid
blink / PERCLOS
Pose features
QC
统计聚合
```

这些内容都可以从已落盘 raw 后续重建。

实验性的 `SharedDecode` 已测试，目前没有显示出比三条独立 reader 更好的墙钟吞吐，因此不作为正式 cohort 默认路径。

## 3. 每次打开新终端

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"

git status --short --branch
git fetch origin --prune
git switch rgb-amd
git pull --ff-only
git status --short --branch
```

设置 Face 模型目录：

```powershell
$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"
```

环境：

| 任务 | Conda 环境 |
|---|---|
| audit / Motion / Pose / validation | `D:\CondaEnvs\attention-rgb` |
| Face DirectML | `D:\CondaEnvs\attention-face-directml` |
| Py-Feat reference / ONNX export | `D:\CondaEnvs\attention-face-pyfeat` |

## 4. 正式全量运行

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1
```

正常续跑不要使用 `-Force`。

cohort runner 会自动：

```text
刷新 rgb_inventory.csv
→ 选择 analysis_eligible=True
→ 完整被试自动 skip
→ 已完成 raw branch 自动 resume/skip
→ 单被试失败记录后继续
→ 持续更新 cohort_status.csv
→ 最终 cohort_manifest.json
```

只跑指定被试：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1 `
  -Subjects sub-036,sub-037
```

## 5. 正式输出

根目录：

```text
D:\_AttentionData\Beijing-RGB
```

单被试核心 raw：

```text
sub-XXX_face_frames.csv
sub-XXX_face_prepare_manifest.json
sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json

sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json

sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json

sub-XXX_manifest.json
```

`extraction_complete=true` 表示三类 raw 完整；`qc_pass=null` 是当前预期状态，表示后续科学 QC 尚未执行，并非 QC 失败。

`face_tracks.parquet`、`eye_features.parquet`、`pose_features.parquet` 等属于 downstream derived，可以以后批量重建，不是 extraction complete 的必要文件。

## 6. 当前 raw 能支持的后续分析

当前 Face raw 已保存所有检测人脸、no-face 时点、bbox / score / 5-point landmarks、20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、478 mesh、68-point compatibility landmarks、native blendshapes（含 `eyeBlinkLeft/Right`）以及完整时间/行为身份。

因此后续无需重新运行 Py-Feat即可重建：

```text
Face tracking / primary face
EAR
眼睑开度 / 虹膜直径 / aperture-iris
blink rate / approximate duration
PERCLOS
AU / gaze / head-pose 时间窗特征
trial / block / probe / sliding-window 分析
baseline normalization
```

Pose raw 保存全部 33 landmarks、world coordinates、visibility/presence；Motion raw 保存 full-FPS 帧差和 gap 信息。具体边界与方法见 049 / 410。

## 7. 文档入口

| 内容 | 文档 |
|---|---|
| 当前 RGB docs 首页 | [`docs/040-rgb/README.md`](docs/040-rgb/README.md) |
| 输出 Schema / 信息保留 | [`044-RGB输出Schema与信息保留原则.md`](docs/040-rgb/044-RGB输出Schema与信息保留原则.md) |
| AMD 环境与运行指令 | [`045-RGB开发环境与运行指令.md`](docs/040-rgb/045-RGB开发环境与运行指令.md) |
| **正式方法、算法与参数依据** | [`049-RGB正式分析方法与参数依据.md`](docs/040-rgb/049-RGB正式分析方法与参数依据.md) |
| **当前实现/全量状态快照** | [`410-RGB当前状态与全量执行总结_20260826.md`](docs/040-rgb/410-RGB当前状态与全量执行总结_20260826.md) |

## 8. 当前执行顺序

```text
AMD cohort raw 全量继续运行
→ 按 cohort_status 补失败/缺失
→ NVIDIA sub-130 Gate + batch benchmark
→ AMD DirectML ↔ NVIDIA native Py-Feat representative parity
→ tracking / Pose features / EAR / blink / PERCLOS
→ cohort QC
→ trial / block / probe / sliding-window 与统计分析
```
