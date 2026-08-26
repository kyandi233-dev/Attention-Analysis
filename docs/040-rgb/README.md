# RGB

> 2026-08-26（Asia/Shanghai）｜当前 AMD RGB 正式化工作线为 `rgb-amd`。它与 `amd-DirectML` 属于同一个 Attention-Analysis 项目；拆分 branch 只是为了并行开发、减少冲突，不代表 RGB 与 NIR/Behavior 被拆成独立项目。

> 分支关系与同步规则见 [`../010-overview/015-并行分支与同步约定.md`](../010-overview/015-并行分支与同步约定.md)。日期型 `docs/工作记录/` 保留历史原文，不追溯改写。

RGB 当前主线为 **Face + Pose + Motion**。目标是形成可与 NIR、SART Behavior 按 `unix_ms` / trial / probe window 对齐的连续行为测量，而不是输出单一 Attention Score。rPPG / HR / HRV 不属于当前正式 RGB 主线。

## 当前状态

| 模块 | 当前路线 | AMD 状态 |
|---|---|---|
| Face backend | Py-Feat 2.1.1 Detectorv2 scientific core | **已冻结；ONNX Runtime DirectML 已通过 real-300 parity** |
| Face cadence | timestamp-driven 15 Hz | **Accepted** |
| Face runtime | original AVI + prefetch + RetinaFace B8 + pooled multitask B16 | **第一档工程优化已 Accepted；sub-031 3600 帧约 29.15 fps** |
| Face formal frame preparation | 完整正式时间段 timestamp-driven 15 Hz | **已实现：`face_formal_prepare.py`** |
| Face full-span DirectML | 原始 AVI 直接解码正式 15 Hz 帧 | **已实现：`face_formal_directml.py`；待首个完整被试实机验收** |
| Primary / eyelid | continuous temporal tracking + 478 mesh + EAR / aperture-iris / eyeBlink | **正式 derived 入口已实现；事件阈值仍可后续冻结** |
| Pose | MediaPipe Tasks Pose Landmarker | 10 Hz 科学层已验证；正式包装入口已实现 |
| Motion | OpenCV frame-difference motion energy | full-fps global Motion 科学层已验证；正式包装入口已实现 |
| 单被试总控 | Motion + Pose + Face + derived | **已实现：`run_rgb_formal_subject.ps1`；下一步就是 sub-031 全程实机验收** |
| 44 人批处理 / resume | 被试级自动队列与完成性恢复 | **尚未实现** |
| body_motion_energy | body ROI 内像素运动 | 尚未实现 |
| blink / perclos80_proxy | 基于已保存 eye signals 的事件/时间窗派生 | 原始信息已保留；最终阈值/窗口 QC 尚未冻结 |

## 当前最重要的工作目标

当前优先级不是继续扩大 gap/QC 讨论，而是先证明整条正式 pipeline 能自动跑完一个被试：

```text
sub-031
→ 完整正式时间段 15 Hz Face 帧位置清单
→ Motion full-fps
→ Pose 10 Hz + Pose derived
→ original AVI Face DirectML 15 Hz
→ continuous primary-face tracking
→ eyelid / openness derived
→ 正式 subject outputs + manifests
```

单被试全程通过后，再补：

```text
44 人批处理
→ completion / resume
→ 失败被试重试与汇总 manifest
```

blink-event threshold、`perclos80_proxy` 最终窗口以及 `body_motion_energy` 仍然需要完成，但它们不再阻挡“先证明完整抽取流程能运行”的工程验收。由于 expensive Face raw 已完整保存，这些 derived 规则可在不重跑 Face 模型的情况下继续冻结。

## 正式 Face 科学基线

当前 AMD Face 已冻结为：

```text
original AVI
→ timestamp-driven 15 Hz
→ reader / preprocess prefetch
→ RetinaFace DirectML B8
→ decode / NMS
→ 1.2 square-reflect face crop
→ cross-RetinaFace pending face chips
→ multitask DirectML B16
→ full scientific raw outputs
→ parquet
```

正式 raw 继续保留：

- 所有检测到的 faces，不在 raw 层只保存 primary；
- RetinaFace bbox/confidence/5-point landmarks；
- 20 AU；
- 7 emotion；
- valence / arousal；
- gaze；
- 6DoF head pose；
- 478 mesh normalized + original-frame coordinates；
- dlib68 compatibility landmarks；
- 全部 native blendshapes，包括 `eyeBlinkLeft/Right`；
- no-detection rows、frame identity、timestamps、phase / block / behavior context 与 QC flags。

信息保留总原则见 [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)。

## 正式输出位置

AMD RGB 正式与测试结果均位于 Git 仓库外：

```text
D:\_AttentionData\Beijing-RGB
```

测试/benchmark/review：

```text
D:\_AttentionData\Beijing-RGB\_test
```

正式被试：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\
```

正式文件内部也重复带被试编号，例如：

```text
sub-XXX_face_frames.csv
sub-XXX_face_raw.parquet
sub-XXX_face_tracks.parquet
sub-XXX_eye_features.parquet
sub-XXX_pose_landmarks.parquet
sub-XXX_pose_features.parquet
sub-XXX_motion_raw.parquet
...
```

Git pull、branch switch、merge 不管理这些正式结果。

## 当前 AMD 工作目录

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd
```

当前 branch：

```text
rgb-amd
```

`amd-DirectML` 仍保留为 AMD 综合线；其中成熟 NIR / Behavior / RGB 资产可以同步到 `rgb-amd`，而 `rgb-amd` 的成熟 RGB 改动之后也会回并综合线。

## 开发前优先阅读

1. [`../010-overview/015-并行分支与同步约定.md`](../010-overview/015-并行分支与同步约定.md)；
2. [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)；
3. [`045-RGB开发环境与运行指令.md`](045-RGB开发环境与运行指令.md)；
4. [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)；
5. [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)；
6. [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)；
7. [`../050-decisions/054-RGB-Face-Backend冻结.md`](../050-decisions/054-RGB-Face-Backend冻结.md)；
8. [`../050-decisions/055-RGB-Face-15Hz采样频率冻结.md`](../050-decisions/055-RGB-Face-15Hz采样频率冻结.md)；
9. [`../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md`](../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md)；
10. [`../050-decisions/057-RGB-Face第一档工程优化冻结.md`](../050-decisions/057-RGB-Face第一档工程优化冻结.md)。

## 当前续接

```text
sub-031 单被试正式全程实机验收
→ 修复实际出现的 orchestration / output / environment 问题
→ 44 人 batch + resume
→ body_motion_energy
→ blink event / perclos80_proxy 最终科学规则
→ 与 NIR / Behavior 进入后续多模态整合
```
