# RGB

> 2026-08-26｜当前 AMD RGB 正式工作线为 `rgb-amd`。RGB 当前目标：**从 baseline 开始连续到 Block2 结束，把 Face、Pose、Motion 三类原始信息完整落盘；能从 raw 重算的 tracking、眼睑派生、Pose features、QC、blink/PERCLOS 和统计聚合全部后移。**

RGB 主线为 **Face + Pose + Motion**，所有结果通过 `unix_ms` 与 Behavior / NIR 对齐。rPPG / HR / HRV 不属于当前正式 RGB 主线。

> **方法学总说明：** [`049-RGB正式分析方法与参数依据.md`](049-RGB正式分析方法与参数依据.md)  
> 该文档集中记录 30/full-FPS、15 Hz、10 Hz 的参数逻辑，Face/Pose/Motion 算法，AMD/NVIDIA 双后端实现、raw-first 原则以及 EAR/blink/PERCLOS 的可回溯性，适合作为报告“方法”部分技术底稿。

## 当前状态

| 模块 | 当前路线 | 状态 |
|---|---|---|
| Face | Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML | 已用于正式 full-span raw |
| Face cadence | timestamp-driven 15 Hz | 已冻结 |
| Face runtime batch | RetinaFace B32 + multitask B64 | 当前默认；属于性能参数，可按硬件吞吐/显存调整 |
| Pose | MediaPipe Pose Landmarker 10 Hz | 已验证 |
| Motion | OpenCV frame difference full-fps | 已验证 |
| 单被试正式总控 | **Motion + Pose + Py-Feat 三线并行 raw extraction** | 已通过 full-span validator |
| 被试最终完整性检查 | `rgb_formal_validate.py` → `sub-XXX_manifest.json` | raw-only 完成判定已生效 |
| cohort batch / resume | eligible 自动队列、已完成跳过、失败继续 | 已实现并用于正式全量 |
| tracking / primary face | 从 Face raw 后续重建 | 不阻挡全量 |
| EAR / eyelid / blink / PERCLOS | 从 478 mesh + blendshape 后续派生 | 不阻挡全量 |
| Pose features | 从 Pose landmarks 后续派生 | 不阻挡全量 |
| body/ROI motion derived | 后续派生 | 不阻挡全量 |

## 正式分析范围

```text
baseline start
→ instructions / practice / transition 等中间阶段连续保留
→ Block1
→ inter-block transition
→ Block2
→ Block2 stop
```

代码目前通过 `baseline_stop - 180 s` 得到 baseline start；在 FocusWave 当前固定 180 s baseline 下，语义就是“baseline 开始”。

## 单被试正式流程

```text
1. face_formal_prepare.py
   快速生成 15 Hz Face 帧清单

2. 三条 raw 同时开始
   ├─ Motion full-fps
   ├─ Pose 10 Hz landmarks
   └─ Py-Feat Face 15 Hz DirectML

3. rgb_formal_validate.py
   只检查三条 raw 是否完整

4. sub-XXX_manifest.json
   extraction_complete = true
```

三条 raw 相互没有科学依赖，因此默认使用独立 reader 并行。实验性 SharedDecode 已进行速度测试，但当前不作为正式 cohort 默认路径。

运行日志写入：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\_runlogs\
```

## 信息保留原则

昂贵模型尽量只运行一次。正式 raw 优先保存以后无法在不重跑模型的情况下恢复的信息：

- **Face raw**：所有检测到的人脸、bbox/confidence、5-point landmarks、20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、478 mesh、dlib68 compatibility landmarks、全部 native blendshapes（包括 `eyeBlinkLeft/Right`），以及 no-detection rows 和完整帧身份；
- **Pose raw**：所有返回 pose 的 33 landmarks、normalized/world coordinates、visibility/presence，多 pose 不提前删除；
- **Motion raw**：正式时间段 full-fps 的时间、亮度、帧差、motion energy、capture/timestamp gap 信息；
- QC 先保留 flag，不在 raw 层提前筛掉。

详细原则见 [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)。正式参数与算法依据见 [`049-RGB正式分析方法与参数依据.md`](049-RGB正式分析方法与参数依据.md)。

## 正式核心输出

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\
```

核心 raw：

```text
sub-XXX_face_frames.csv
sub-XXX_face_prepare_manifest.json
sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json

sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json

sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json

sub-XXX_manifest.json
```

以前或后续生成的：

```text
sub-XXX_face_tracks.parquet
sub-XXX_eye_features.parquet
sub-XXX_face_derived_manifest.json
sub-XXX_pose_features.parquet
sub-XXX_pose_features_manifest.json
```

都属于**可重建的 downstream derived（后续派生）**，保留但不再作为 extraction complete 的必要条件。

cohort 级：

```text
rgb_inventory.csv
cohort_status.csv
cohort_manifest.json
```

## 为什么 tracking 不再阻挡全量

`face_raw.parquet` 已保存每帧的人脸框、置信度、时间戳、478 点和完整面部输出。跨帧 tracking、主脸判断、EAR、眼睑开度、虹膜比都可以直接从这些 raw 重新计算，因此没有必要在每个被试完成 Py-Feat 后继续等待 tracking 才进入下一个被试。

## 当前运行方式

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"
git pull --ff-only

$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1
```

正式 cohort 使用 subject manifest 自动跳过已完成被试，并由各 raw stage 的 complete manifest 实现分支级 resume。不要为了普通续跑使用 `-Force`。

当前优先级：

```text
AMD cohort raw 全量
→ 检查失败/缺失与 QC
→ NVIDIA representative parity / full-span Gate
→ 统一做 tracking、Pose features、眼睑/blink/PERCLOS、QC 和统计派生
```
