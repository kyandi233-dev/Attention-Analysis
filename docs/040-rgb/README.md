# RGB

> 2026-08-26（Asia/Shanghai）｜`amd-DirectML` 是 AMD 综合线，统一保存 NIR + Behavior + NIR-Behavior + RGB 的成熟状态；当前 AMD RGB 的并行开发工作线已经切到 `rgb-amd`，目的是减少同时改动不同模态造成的冲突，而不是把项目拆开。

> 分支关系见 [`../010-overview/015-并行分支与同步约定.md`](../010-overview/015-并行分支与同步约定.md)。日期型工作记录保持历史原文，不追溯改写。

RGB 当前主线为 **Face + Pose + Motion**，目标是形成可与 NIR、SART Behavior 按 `unix_ms` / trial / probe window 对齐的连续行为测量。

## 当前 AMD RGB 状态

| 模块 | 当前路线 | 状态 |
|---|---|---|
| Face backend | Py-Feat 2.1.1 Detectorv2 scientific core | **已冻结；ONNX Runtime DirectML real-300 parity 已通过** |
| Face cadence | timestamp-driven 15 Hz | **Accepted** |
| Face runtime | original AVI + prefetch + RetinaFace B8 + pooled multitask B16 | **第一档工程优化已 Accepted；sub-031 3600 帧约 29.15 fps** |
| Full-span Face preparation | 完整正式时间段 15 Hz frame manifest | **已实现** |
| Full-span Face DirectML | original AVI direct decode | **已实现；待首个完整被试实机验收** |
| Primary / eyelid | continuous tracking + 478 mesh + EAR / aperture-iris / eyeBlink | **正式 derived 入口已实现** |
| Pose | MediaPipe Pose 10 Hz | 科学层已验证；正式包装入口已实现 |
| Motion | OpenCV full-fps global motion | 科学层已验证；正式包装入口已实现 |
| 单被试总控 | Motion + Pose + Face + derived | **`run_rgb_formal_subject.ps1` 已实现** |
| 44 人 batch / resume | cohort automation | 尚未实现 |
| body_motion_energy | body ROI 内像素运动 | 尚未实现 |
| blink / perclos80_proxy | eye signal downstream derived | raw 信息已保留；最终科学规则尚未冻结 |

## 当前工程优先级

当前优先证明正式 pipeline 能完整跑完一个被试，而不是继续把 timestamp gap/QC 当成独立前置 Gate：

```text
sub-031
→ 完整正式时间段 Face 15 Hz 帧清单
→ Motion full-fps
→ Pose 10 Hz + derived
→ original AVI Face DirectML 15 Hz
→ continuous primary tracking
→ eyelid derived
→ 正式输出与 manifests
```

单被试通过后再实现 44 人 batch + completion/resume。blink threshold、`perclos80_proxy` 与 `body_motion_energy` 仍需完成，但 expensive Face raw 已完整保留，因此后续规则调整不应要求重跑 Face inference。

## 当前正式入口

```text
scripts/face_formal_prepare.py
scripts/rgb_formal_motion_pose.py
scripts/face_formal_directml.py
scripts/face_formal_derive.py
scripts/run_rgb_formal_subject.ps1
```

AMD RGB 当前 active development branch 为：

```text
rgb-amd
```

当前本地工作目录约定：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd
```

`amd-DirectML` 中这些正式入口继续保留，作为综合线已同步状态；后续新的 RGB 修改先在 `rgb-amd` 完成并验收，再同步/回并。

## 输出位置

AMD RGB 正式/测试结果均位于仓库外：

```text
D:\_AttentionData\Beijing-RGB
```

测试/benchmark：

```text
D:\_AttentionData\Beijing-RGB\_test
```

正式被试：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX
```

正式文件内部重复带 `sub-XXX_` 前缀，避免离开 subject 目录后失去身份。

## 科学信息保留

正式 Face raw 继续保留所有检测到的人脸和完整 scientific core 输出，包括 bbox/confidence、5-point Retina landmarks、20 AU、7 emotion、V/A、gaze、6DoF pose、478 mesh、68 compatibility landmarks、native blendshapes 与 no-detection rows。primary-face、EAR、aperture/iris、normalized openness、closure 等都属于可重算 derived。

详细 schema 见 [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)。

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
rgb-amd: sub-031 单被试正式全程实机验收
→ 44 人 batch + resume
→ body_motion_energy
→ blink / perclos80_proxy 最终科学规则
→ 成熟结果同步回 amd-DirectML
```
