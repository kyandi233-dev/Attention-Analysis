# RGB

> 2026-08-26（Asia/Shanghai）｜RGB 已从独立 `rgb-dev` 开发线选择性同步进入长期 AMD 硬件主线 `amd-DirectML`。本页描述当前有效状态；日期型工作记录保持历史原文，不追溯改写。

RGB 当前主线为 **Face + Pose + Motion**，目标是形成可与 NIR、SART Behavior 按 `unix_ms` / trial / probe window 对齐的连续行为测量，而不是输出单一 Attention Score。rPPG / HR / HRV 不属于当前正式 RGB 主线。

## 当前状态

| 模块 | 当前路线 | AMD 状态 |
|---|---|---|
| Face | Py-Feat 2.1.1 Detectorv2 scientific core | **backend 已冻结；ONNX Runtime DirectML 已验证** |
| Face cadence | timestamp-driven 15 Hz | **Accepted** |
| Face runtime | direct AVI + prefetch + RetinaFace B8 → pending multitask B16 | **Accepted；sub-031 3600 帧约 29.15 fps** |
| Primary / eyelid | temporal tracking + 478 mesh + EAR / aperture-iris / eyeBlink | **sub-031 representative dry-run 3600/3600 coverage；最终事件阈值待冻结** |
| Face QC | full-face 478 mesh + eyes/iris highlight + bbox/track/metrics | **v03 单层黑字版本可用** |
| Pose | MediaPipe Tasks Pose Landmarker | sub-031 10 Hz pilot/QC/features 已完成 |
| Motion | OpenCV frame-difference motion energy | sub-031 global Motion pilot/QC/review 已完成 |
| body_motion_energy | body ROI 内像素运动 | 待统一正式视频读取阶段实现 |

## 仍未完成的正式化 Gate

1. `sub-033` timestamp / capture-gap stress test；
2. continuous primary-face gates 最终 Accepted；
3. blink event threshold；
4. `perclos80_proxy` 最终窗口、分母与 QC；
5. direct full-video formal runner + completion/resume；
6. Face/Pose/Motion/body_motion_energy 统一视频读取与正式 schema 收口。

在这些 Gate 完成前，不进行 44 被试 Face 正式全量。

## 当前输出位置

AMD RGB 数据与所有正式/测试结果均位于 Git 仓库外：

```text
D:\_AttentionData\Beijing-RGB
```

测试、benchmark 与 review 使用：

```text
D:\_AttentionData\Beijing-RGB\_test
```

Git pull、切换分支或代码合并不会覆盖上述仓库外结果；重新运行具体 runner 时仍需遵循 completion/resume/versioned-output 规则。

## 开发前优先阅读

1. [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)
2. [`045-RGB开发环境与运行指令.md`](045-RGB开发环境与运行指令.md)
3. [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
4. [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
5. [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)
6. [`../050-decisions/054-RGB-Face-Backend冻结.md`](../050-decisions/054-RGB-Face-Backend冻结.md)
7. [`../050-decisions/055-RGB-Face-15Hz采样频率冻结.md`](../050-decisions/055-RGB-Face-15Hz采样频率冻结.md)
8. [`../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md`](../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md)
9. [`../050-decisions/057-RGB-Face第一档工程优化冻结.md`](../050-decisions/057-RGB-Face第一档工程优化冻结.md)

## 当前续接

当前方法学续接顺序：

```text
sub-031 full-face QC 确认
→ sub-033 gap stress
→ blink / perclos80_proxy 冻结
→ full-video formal runner
→ AMD/NVIDIA parity 与双机正式运行
```

最新硬件主线收口记录见：

- `docs/工作记录/08-26-16-硬件主线收口与共享科学层同步.md`
- `docs/工作记录/08-26-17-硬件主线Behavior-RGB同步阶段完成.md`
