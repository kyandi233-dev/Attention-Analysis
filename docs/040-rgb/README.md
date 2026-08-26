# RGB

> 2026-08-26｜当前 AMD RGB 正式化工作线为 `rgb-amd`。RGB 的当前目标很明确：**从 baseline 开始连续到 Block2 结束，把 Face、Pose、Motion 能稳定获得的数据完整落盘；QC、blink/PERCLOS 和统计聚合后移。**

RGB 当前主线为 **Face + Pose + Motion**，所有结果通过 `unix_ms` 与 Behavior / NIR 对齐。rPPG / HR / HRV 不属于当前正式 RGB 主线。

## 当前状态

| 模块 | 当前路线 | 状态 |
|---|---|---|
| Face | Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML | 已冻结 |
| Face cadence | timestamp-driven 15 Hz | 已冻结 |
| Pose | MediaPipe Pose Landmarker 10 Hz | 已验证 |
| Motion | OpenCV frame difference full-fps | 已验证 |
| 单被试正式总控 | Face + Pose + Motion + derived + final validation | **已实现** |
| 被试最终完整性检查 | `rgb_formal_validate.py` → `sub-XXX_manifest.json` | **已实现** |
| cohort batch / resume | 自动读取 eligible 被试、已完成跳过、失败继续 | **已实现，待实机验收** |
| blink / PERCLOS | 从已保存 eye signals 后续派生 | 后续完成，不阻挡抽取 |
| body_motion_energy | body ROI motion | 后续完成，不阻挡抽取 |

## 正式分析范围

统一口径：

```text
baseline start
→ instructions / practice / transition 等中间阶段连续保留
→ Block1
→ inter-block transition
→ Block2
→ Block2 stop
```

代码目前通过 `baseline_stop - 180 s` 得到 baseline start；这是 FocusWave 当前固定 180 s baseline 的实现方式，语义上就是“baseline 开始”。

## 单被试正式流程

```text
face_formal_prepare.py
→ rgb_formal_motion_pose.py
→ face_formal_directml.py
→ face_formal_derive.py
→ rgb_formal_validate.py
→ sub-XXX_manifest.json
```

最终 validator 只判断**数据抽取是否完整**，不判断 QC 是否通过。只有 Face/Pose/Motion 必需文件存在且关键帧数关系正确时，才写：

```text
completion_status = complete
extraction_complete = true
qc_pass = null
```

因此“跑完整”和“数据质量好”被明确分开。

## 信息保留原则

昂贵模型尽量只运行一次。正式 raw 优先保存以后无法在不重跑模型的情况下恢复的信息：

- Face：所有检测到的人脸、bbox/confidence、5-point landmarks、20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、478 mesh、dlib68 compatibility landmarks、全部 native blendshapes（包括 `eyeBlinkLeft/Right`），以及 no-detection rows 和完整帧身份；
- Pose：所有返回 pose 的 33 landmarks、normalized/world coordinates、visibility/presence，多 pose 不提前删除；
- Motion：正式时间段 full-fps 的时间、亮度、帧差、motion energy、capture/timestamp gap 信息；
- QC 先保留 flag，不在 raw 层提前筛掉。

Face detection threshold 继续使用当前 Py-Feat formal runner 已冻结的默认口径，不再为全量运行增加额外阈值研究。

详细原则见 [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)。

## 输出

```text
D:\_AttentionData\Beijing-RGB\
```

单被试目录：

```text
sub-XXX_face_frames.csv
sub-XXX_face_prepare_manifest.json
sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json
sub-XXX_face_tracks.parquet
sub-XXX_eye_features.parquet
sub-XXX_face_derived_manifest.json
sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json
sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json
sub-XXX_pose_features.parquet
sub-XXX_pose_features_manifest.json
sub-XXX_manifest.json
```

cohort 级：

```text
rgb_inventory.csv
cohort_status.csv
cohort_manifest.json
```

`cohort_status.csv` 在每个被试结束后都会重写一次，因此批处理被中断时仍能看到已经完成和失败到哪里；真正的 resume 判定以每个 `sub-XXX_manifest.json` 为准。

## 当前运行顺序

先做一次 `sub-031` 全程实机验收：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"

git pull --ff-only

$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-031
```

`sub-031` 通过后，全 cohort 使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1
```

cohort runner 会先刷新 `rgb_inventory.csv`，只处理 `analysis_eligible=True` 的正式被试；已有完整 manifest 的被试自动跳过，单个被试失败时记录错误并继续下一人。

## 当前优先级

```text
sub-031 单被试实机验收
→ 修复实际运行错误（如果有）
→ 正式 cohort 全量抽取
→ 检查 cohort_status / 缺失被试
→ 再做 QC、blink/PERCLOS、body motion 和统计派生
```

不要在全量抽取前继续扩展非必要科学规则。只要 raw 信息已经完整保存，后续派生规则都可以重新计算而无需重跑昂贵 Face inference。
